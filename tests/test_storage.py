"""Storage integration tests using temporary, generated images only."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.readers import UnsupportedImage, _mapped_metadata, _reject_mrc_volume, _render_scientific_frame, _signal_layout, iter_frames
from backend.storage import ImportCancelled, Store, list_image_files


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.store = Store(self.root / "project")

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name="image.png", color=100, directory=None):
        path = (directory or self.sources) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (color, 20, 50)).save(path)
        return path

    def test_import_deduplicates_and_preserves_original(self):
        original = self.image()
        before = original.read_bytes()
        shutil.copyfile(original, self.sources / "duplicate.png")
        (self.sources / "broken.jpg").write_bytes(b"not a jpeg")
        (self.sources / "notes.txt").write_text("not an image")
        result = self.store.import_folder(self.sources)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["corrupt"], 1)
        self.assertEqual(original.read_bytes(), before)
        record = self.store.images()["items"][0]
        self.assertEqual(record["width"], 40)
        self.assertEqual(record["group"], str(self.sources))
        self.assertTrue(record["thumbnail"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(Path(record["render_path"]).is_file())
        self.assertEqual(self.store.stats()["counts"]["UNLABELED"], 1)
        self.assertEqual(self.store.import_folder(self.sources)["imported"], 0)

    def test_symlinks_are_not_followed(self):
        source = self.image()
        try:
            (self.sources / "link.png").symlink_to(source)
            (self.sources / "linked-folder").symlink_to(self.sources, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation unavailable")
        self.assertEqual(list_image_files(self.sources), [str(source)])
        with self.assertRaises(ValueError):
            list_image_files(self.sources / "linked-folder")

    def test_sixteen_bit_tiff_is_scaled_not_clipped(self):
        pixels = np.array([[1000, 2000], [3000, 5000]], dtype=np.uint16)
        path = self.sources / "high-depth.tiff"
        Image.fromarray(pixels).save(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(self.store.import_folder(self.sources)["imported"], 1)
        record = self.store.all_records()[0]
        with Image.open(record["render_path"]) as rendered:
            values = np.array(rendered)[:, :, 0]
        self.assertEqual(int(values.min()), 0)
        self.assertEqual(int(values.max()), 255)
        self.assertGreater(int(values[0, 1]), 0)
        self.assertLess(int(values[0, 1]), 255)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(record["metadata"]["provenance"]["normalization"]["min"], 1000)

    def test_stack_range_is_inclusive_and_frame_dedupe(self):
        frames = [Image.new("L", (10, 10), value) for value in [20, 80, 160]]
        frames[0].save(self.sources / "stack.tiff", save_all=True, append_images=frames[1:])
        self.assertEqual(self.store.import_folder(self.sources)["imported"], 1)
        result = self.store.import_folder(self.sources, frame_start=1, frame_end=2)
        self.assertEqual(result["imported"], 2)
        records = self.store.all_records()
        self.assertEqual([r["frame"] for r in records], [0, 1, 2])
        self.assertTrue(all(r["frames"] == 3 for r in records))
        self.assertEqual(self.store.import_folder(self.sources, 0, 2)["imported"], 0)
        self.assertEqual(self.store.import_folder(self.sources, 5, 6)["unsupported"], 1)
        with self.assertRaises(ValueError):
            self.store.import_folder(self.sources, 5, 1)

    def test_labels_metadata_validation_and_reopen(self):
        self.image()
        self.store.import_folder(self.sources)
        id_ = self.store.all_records()[0]["id"]
        self.assertEqual(self.store.label([id_, id_], "SEM", "sample-A")["updated"], 1)
        self.store.update_metadata([id_], {"sample": "gold", "acquired_at": "2026-10-15", "voltage_kv": 200})
        reopened = Store(self.store.root)
        record = reopened.training_records()[0]
        self.assertEqual(record["label"], "SEM")
        self.assertEqual(record["group"], "sample-A")
        self.assertEqual(record["metadata"]["sample"], "gold")
        self.assertIn("provenance", record["metadata"])
        with self.assertRaises(ValueError):
            self.store.label([id_, 99999], "TEM")
        self.assertEqual(self.store.training_records()[0]["label"], "SEM")
        for label in ["UNKNOWN", "sem"]:
            with self.assertRaises(ValueError):
                self.store.label([id_], label)
        with self.assertRaises(ValueError):
            self.store.label([True], "TEM")
        with self.assertRaises(ValueError):
            self.store.update_metadata([id_], {"provenance": {}})
        with self.assertRaises(ValueError):
            self.store.update_metadata([id_], {"acquired_at": "not a date"})
        self.store.label([id_], None)
        self.assertEqual(self.store.training_records(), [])

    def test_literal_search_and_pagination(self):
        for i, name in enumerate(["sample_1.png", "sample%2.png", "other.png"]):
            self.image(name, 100 + i)
        self.store.import_folder(self.sources)
        self.assertEqual(self.store.images(query="%")["total"], 1)
        self.assertEqual(self.store.images(query="sample_")["total"], 1)
        first = self.store.images(limit=1, offset=0)
        second = self.store.images(limit=1, offset=1)
        self.assertEqual(first["total"], 3)
        self.assertNotEqual(first["items"][0]["id"], second["items"][0]["id"])
        self.store.label([first["items"][0]["id"]], "OTHER")
        self.assertEqual(self.store.images(filter="OTHER")["total"], 1)
        self.assertEqual(self.store.images(filter="UNLABELED")["total"], 2)

    def test_missing_source_and_relink_validate_content(self):
        source = self.image()
        self.store.import_folder(self.sources)
        new = self.root / "new-sources"
        shutil.move(str(self.sources), new)
        missing = self.store.images()["items"][0]
        self.assertTrue(missing["missing"])
        self.assertIsNone(missing["thumbnail"])
        self.assertEqual(self.store.relink(self.sources, new)["updated"], 1)
        record = self.store.images()["items"][0]
        self.assertEqual(record["path"], str(new / source.name))
        self.assertFalse(record["missing"])
        self.assertEqual(record["group"], str(new))
        self.assertEqual(self.store.stats()["roots"], [str(new)])
        wrong = self.root / "wrong"
        wrong.mkdir()
        self.image(source.name, 222, wrong)
        self.assertEqual(self.store.relink(new, wrong)["updated"], 0)

    def test_relink_retains_manual_group(self):
        self.image()
        self.store.import_folder(self.sources)
        self.store.label([self.store.all_records()[0]["id"]], "TEM", "manual specimen")
        new = self.root / "moved"
        shutil.move(str(self.sources), new)
        self.store.relink(self.sources, new)
        self.assertEqual(self.store.all_records()[0]["group"], "manual specimen")

    def test_runs_atomic_update_and_sort(self):
        self.store.save_run({"id": "older", "created_at": "2026-01-01", "status": "done"})
        self.store.save_run({"id": "newer", "created_at": "2026-02-01", "status": "running"})
        self.store.save_run({"id": "newer", "created_at": "2026-02-01", "status": "done"})
        self.assertEqual([r["id"] for r in self.store.runs()], ["newer", "older"])
        self.assertEqual(self.store.runs()[0]["status"], "done")
        self.assertEqual(len(list(self.store.run_dir.glob("*.json"))), 2)
        with self.assertRaises(ValueError):
            self.store.save_run({"id": "../escape"})
        with self.assertRaises(ValueError):
            self.store.save_run({"id": "nan", "score": float("nan")})

    def test_background_connections_and_cancel(self):
        self.image()
        events = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            result = pool.submit(self.store.import_folder, self.sources, 0, 0, lambda *a: events.append(a)).result()
            counts = list(pool.map(lambda _: self.store.stats()["total"], range(12)))
        self.assertEqual(result["imported"], 1)
        self.assertEqual(counts, [1] * 12)
        self.assertEqual(events[-1][1], 1.0)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(ImportCancelled):
            self.store.import_folder(self.sources, cancel=cancel)

    def test_own_render_cache_is_excluded(self):
        self.image()
        self.store.import_folder(self.sources)
        self.assertEqual(self.store.import_folder(self.root)["imported"], 0)

    def test_invalid_directory_and_parameters(self):
        with self.assertRaises(ValueError):
            self.store.import_folder(self.root / "absent")
        with self.assertRaises(ValueError):
            self.store.images(offset=-1)
        with self.assertRaises(ValueError):
            self.store.images(filter="INVALID")

    def test_source_revision_replaces_old_labels_and_ids(self):
        self.image()
        self.store.import_folder(self.sources)
        original = self.store.all_records()[0]
        self.store.label([original["id"]], "SEM", "custom group")
        self.image(color=225)
        result = self.store.import_folder(self.sources)
        current = self.store.all_records()[0]
        self.assertEqual(result["total"], 1)
        self.assertNotEqual(original["id"], current["id"])
        self.assertNotEqual(original["sha256"], current["sha256"])
        self.assertIsNone(current["label"])
        provenance = current["metadata"]["provenance"]
        source_stat = Path(current["path"]).stat()
        self.assertEqual(provenance["source_size"], source_stat.st_size)
        self.assertEqual(provenance["source_mtime_ns"], source_stat.st_mtime_ns)

    def test_portable_cache_and_page_thumbnail_fetch(self):
        self.image()
        self.store.import_folder(self.sources)
        old_root = self.store.root
        new_root = self.root / "copied-project"
        shutil.copytree(old_root, new_root)
        reopened = Store(new_root)
        record = reopened.all_records()[0]
        self.assertNotIn("thumbnail", record)
        self.assertTrue(Path(record["render_path"]).is_relative_to(new_root))
        self.assertTrue(Path(record["render_path"]).exists())
        page = reopened.get_images([record["id"]])
        self.assertTrue(page[0]["thumbnail"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(reopened.thumbnail(record["id"]), page[0]["thumbnail"])

    def test_reimport_same_bytes_refreshes_source_stamp_preserving_label(self):
        source = self.image()
        self.store.import_folder(self.sources)
        record = self.store.all_records()[0]
        self.store.label([record["id"]], "TEM")
        before = source.stat()
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        self.store.import_folder(self.sources)
        current = self.store.all_records()[0]
        self.assertEqual(current["id"], record["id"])
        self.assertEqual(current["label"], "TEM")
        self.assertEqual(current["metadata"]["provenance"]["source_mtime_ns"], source.stat().st_mtime_ns)

    def test_portable_run_model_paths(self):
        model = self.store.root / "runs" / "portable-run"
        model.mkdir(parents=True)
        (model / "classifier.json").write_text('{"classes":["SEM","TEM","OTHER"]}')
        self.store.save_run({"id": "portable-run", "model_path": str(model)})
        copied = self.root / "copied-project"
        shutil.copytree(self.store.root, copied)
        reopened = Store(copied)
        run = reopened.runs()[0]
        self.assertEqual(run["model_path"], str(copied / "runs" / "portable-run"))
        self.assertTrue((Path(run["model_path"]) / "classifier.json").is_file())
        # Legacy Windows absolute paths rebase on macOS, too.
        (reopened.run_dir / "legacy.json").write_text(json.dumps({"id": "legacy", "model_path": "C:\\old\\project\\runs\\legacy"}))
        legacy = next(r for r in reopened.runs() if r["id"] == "legacy")
        self.assertEqual(legacy["model_path"], str(copied / "runs" / "legacy"))

    def test_windows_sources_relink_to_current_platform(self):
        self.image()
        self.store.import_folder(self.sources)
        with self.store._connection() as db:
            db.execute("UPDATE images SET path=?,specimen_group=?", ("C:\\Users\\Scientist\\Images\\image.png", "C:\\Users\\Scientist\\Images"))
            db.execute("DELETE FROM source_roots")
            db.execute("INSERT INTO source_roots VALUES(?)", ("C:\\Users\\Scientist\\Images",))
        result = self.store.relink("C:/Users/Scientist/Images", self.sources)
        self.assertEqual(result["updated"], 1)
        record = self.store.all_records()[0]
        self.assertEqual(record["path"], str(self.sources / "image.png"))
        self.assertEqual(record["group"], str(self.sources))
        self.assertEqual(self.store.stats()["roots"], [str(self.sources)])

    def test_windows_unc_sources_relink(self):
        self.image()
        self.store.import_folder(self.sources)
        with self.store._connection() as db:
            db.execute("UPDATE images SET path=?,specimen_group=?", ("\\\\lab-server\\microscope\\image.png", "\\\\lab-server\\microscope"))
            db.execute("DELETE FROM source_roots")
            db.execute("INSERT INTO source_roots VALUES(?)", ("\\\\lab-server\\microscope",))
        self.assertEqual(self.store.relink("\\\\lab-server\\microscope", self.sources)["updated"], 1)
        self.assertEqual(self.store.stats()["roots"], [str(self.sources)])

    @unittest.skipUnless(importlib.util.find_spec("rsciio"), "RosettaSciIO optional reader dependencies not installed")
    def test_real_generated_mrc_reader_selected_frames(self):
        pixels = np.arange(3 * 8 * 10, dtype="<f4").reshape((3, 8, 10))
        header = bytearray(1024)
        struct.pack_into("<4i", header, 0, 10, 8, 3, 2)
        struct.pack_into("<3i", header, 28, 10, 8, 3)
        struct.pack_into("<3f", header, 40, 10, 8, 3)
        struct.pack_into("<3f", header, 52, 90, 90, 90)
        struct.pack_into("<3i", header, 64, 1, 2, 3)
        struct.pack_into("<3f", header, 76, float(pixels.min()), float(pixels.max()), float(pixels.mean()))
        header[208:212] = b"MAP "
        header[212:216] = bytes([68, 68, 0, 0])
        source = self.sources / "stack.mrc"
        source.write_bytes(header + pixels.tobytes())
        before = source.read_bytes()
        result = self.store.import_folder(self.sources, 1, 2)
        self.assertEqual(result["imported"], 2, result)
        self.assertEqual([(r["frame"], r["frames"], r["width"], r["height"]) for r in self.store.all_records()], [(1, 3, 10, 8), (2, 3, 10, 8)])
        self.assertEqual(source.read_bytes(), before)
        # On Windows an outstanding mmap prevents this immediate move. Keep
        # rendered results alive to ensure their pixels own independent memory.
        moved = source.with_name("moved-stack.mrc")
        source.rename(moved)
        frames = iter_frames(moved, 0, 2)
        first = next(frames)
        try:
            renamed = moved.with_name("cancelled-stack.mrc")
            moved.rename(renamed)  # Also works while the generator is suspended.
        finally:
            frames.close()
        renamed.unlink()
        self.assertEqual(first["image"].size, (10, 8))
        self.assertEqual(len(first["image"].tobytes()), 10 * 8 * 3)


class ScientificLayoutTests(unittest.TestCase):
    def test_mapped_frame_owns_pixels_and_releases_source_without_gc(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "mapped-frame.bin"
            np.arange(16, dtype="float32").tofile(source)
            retained_mapping = np.memmap(source, dtype="float32", mode="r", shape=(4, 4))
            view = np.asarray(retained_mapping)[1:, 1:]
            image, width, height, _, _ = _render_scientific_frame(view)
            self.assertTrue(retained_mapping._mmap.closed)
            source.unlink()
            self.assertEqual((width, height), (3, 3))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(image.getpixel((2, 2)), (255, 255, 255))

    def test_mapped_source_released_when_normalization_rejects_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "invalid-frame.bin"
            np.full((4, 4), np.nan, dtype="float32").tofile(source)
            retained_mapping = np.memmap(source, dtype="float32", mode="r", shape=(4, 4))
            with self.assertRaises(UnsupportedImage):
                _render_scientific_frame(np.asarray(retained_mapping))
            self.assertTrue(retained_mapping._mmap.closed)
            source.unlink()

    def test_only_two_signal_axes_and_one_navigation_axis(self):
        array = np.zeros((3, 4, 5), dtype=np.uint16)
        axes = [{"index_in_array": i, "navigate": i == 0} for i in range(3)]
        self.assertEqual(_signal_layout({"data": array, "axes": axes}), ((3, 4, 5), 0))
        with self.assertRaises(UnsupportedImage):
            _signal_layout({"data": array, "axes": []})
        with self.assertRaises(UnsupportedImage):
            _signal_layout({"data": np.zeros((2, 3, 4, 5)), "axes": []})
        with self.assertRaises(UnsupportedImage):
            _signal_layout({"data": array, "axes": axes, "metadata": {"Signal": {"signal_type": "EELS"}}})

    def test_mrc_volume_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "volume.mrc"
            header = bytearray(1024)
            struct.pack_into("<3i", header, 0, 20, 20, 5)
            struct.pack_into("<i", header, 88, 1)
            path.write_bytes(header)
            with self.assertRaises(UnsupportedImage):
                _reject_mrc_volume(path)
            struct.pack_into("<i", header, 88, 0)
            path.write_bytes(header)
            _reject_mrc_volume(path)

    def test_official_metadata_mapping(self):
        signal = {"metadata": {"General": {"title": "sample"}},
                  "original_metadata": {"ImageTags": {"Microscope Info": {"Voltage": 200000}}},
                  "mapping": {"ImageTags.Microscope Info.Voltage": ("Acquisition_instrument.TEM.beam_energy", lambda value: value / 1000)}}
        self.assertEqual(_mapped_metadata(signal)["Acquisition_instrument"]["TEM"]["beam_energy"], 200)
        signal = {"original_metadata": {"ObjectInfo": {"ExperimentalDescription": {"High tension_kV": 200.0}}},
                  "mapping": {"ObjectInfo.ExperimentalDescription.High_tension_kV": ("Acquisition_instrument.TEM.beam_energy", None)}}
        self.assertEqual(_mapped_metadata(signal)["Acquisition_instrument"]["TEM"]["beam_energy"], 200)


if __name__ == "__main__":
    unittest.main()
