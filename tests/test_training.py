"""Algorithm/integration tests use temporary generated data; never download models."""
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from backend import training as t

try:
    import numpy as np
except ImportError:
    np = None


def training_records(groups=5):
    return [{"id": f"{label}-{g}-{i}", "path": f"/{label}/{g}/{i}.png", "label": label,
             "group": f"{label}-{g}", "sha256": f"{label}-{g}-{i}", "frame": 0}
            for label in t.CLASSES for g in range(groups) for i in range(2)]


def head():
    coef = [[0.] * t.DIM for _ in t.CLASSES]
    for i in range(3):
        coef[i][i] = 12.
    return {"schema": 1, "classes": list(t.CLASSES), "dimension": t.DIM,
            "coef": coef, "intercept": [0., 0., 0.]}


class SplitAndGateTests(unittest.TestCase):
    def test_specimen_split_reproducible_and_disjoint(self):
        records = training_records()
        splits = t.split_records(records, "finetune")
        self.assertEqual(splits, t.split_records(records, "finetune"))
        sets = [{r["group"] for r in splits[s]} for s in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for rows in splits.values():
            self.assertEqual({r["label"] for r in rows}, set(t.CLASSES))
        for label in ("SEM", "TEM"):
            self.assertGreaterEqual(len({r["group"] for r in splits["train"] if r["label"] == label}), 2)
        self.assertEqual(sum(map(len, splits.values())), len(records))

    def test_small_data_rejected_and_duplicates_cannot_leak(self):
        with self.assertRaisesRegex(ValueError, "최소 3"):
            t.split_records(training_records(2))
        with self.assertRaisesRegex(ValueError, "최소 4"):
            t.split_records(training_records(3), "finetune")
        records = training_records()
        extra = dict(records[0], id="duplicate", group="different-folder")
        splits = t.split_records(records + [extra])
        self.assertEqual(sum(map(len, splits.values())), len(records))
        with self.assertRaisesRegex(ValueError, "서로 다른 라벨"):
            t.split_records(records + [dict(extra, label="TEM")])

    def test_triplets_no_other_anchor_and_cross_specimen(self):
        rows = t.split_records(training_records(), "finetune")["train"]
        for a, p, n in t.make_triplets(rows):
            self.assertIn(a["label"], ("SEM", "TEM"))
            self.assertEqual(a["label"], p["label"])
            self.assertNotEqual(a["group"], p["group"])
            self.assertNotEqual(a["label"], n["label"])

    def test_rejection_including_other_tie_manual_other_and_nan(self):
        self.assertEqual(t.accepted_label([.98, .01, .01], .95), "SEM")
        self.assertIsNone(t.accepted_label([.98, .01, .01], .95, "OTHER"))
        self.assertIsNone(t.accepted_label([.49, .02, .49], .4))
        self.assertIsNone(t.accepted_label([.4, .3, .3], .95))
        self.assertIsNone(t.accepted_label([float("nan"), .2, .1], .1))
        self.assertIsNone(t.accepted_label([.01, .01, .98], .95, "SEM"))

    def test_validation_threshold_and_empty_precision(self):
        selected = t.select_threshold(["OTHER", "SEM"], [[.97, .02, .01], [.98, .01, .01]], .95)
        self.assertGreater(selected, .97)
        self.assertLessEqual(selected, .98)
        # Test OTHER at .99 stays a false positive; no test-set threshold tuning.
        measured = t.metrics(["OTHER", "SEM", "TEM"], [[.99, .005, .005], [.98, .01, .01], [.1, .2, .7]], selected)
        self.assertEqual(measured["false_positives"], 1)
        self.assertEqual(measured["test_count"], 3)
        self.assertEqual(measured["precision"], .5)
        self.assertEqual(measured["recall"], .5)
        self.assertIsNone(t.metrics(["SEM"], [[.8, .1, .1]], .95)["precision"])

    def test_json_classifier_validation(self):
        with tempfile.TemporaryDirectory() as d:
            t._atomic_json(Path(d) / "classifier.json", head())
            run = {"model_path": d}
            self.assertEqual(t._read_head(run), head())
            malformed = head()
            malformed["classes"] = ["OTHER", "SEM", "TEM"]
            t._atomic_json(Path(d) / "classifier.json", malformed)
            with self.assertRaises(ValueError):
                t._read_head(run)

    def test_only_native_last_two_blocks_trainable(self):
        class Parameter:
            def __init__(self): self.requires_grad = True
            def requires_grad_(self, flag): self.requires_grad = flag
        class Module:
            def __init__(self): self.p = Parameter()
            def parameters(self): return [self.p]
        class Language:
            def __init__(self):
                self.layers = [Module() for _ in range(4)]
                self.norm, self.embedding_projection = Module(), Module()
        class Backbone:
            def __init__(self): self.language_model, self.vision_tower = Language(), Module()
        class Model:
            def __init__(self):
                self.transformers_model = Backbone()
                lang = self.transformers_model.language_model
                self.named = [(f"layer.{i}", m.p) for i, m in enumerate(lang.layers)] + [
                    ("norm", lang.norm.p), ("projection", lang.embedding_projection.p),
                    ("vision", self.transformers_model.vision_tower.p)]
            def parameters(self): return [p for _, p in self.named]
            def named_parameters(self): return self.named
        model = Model()
        self.assertEqual(t.choose_partial_parameters(model), ["layer.2", "layer.3", "norm", "projection"])
        self.assertFalse(model.transformers_model.vision_tower.p.requires_grad)

    def test_partial_snapshot_not_reported_cached(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "modules.json").write_text("[]")
            (p / "config.json").write_text("{}")
            self.assertFalse(t._snapshot_complete(p))
            (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
            (p / "a.safetensors").write_bytes(b"test")
            self.assertFalse(t._snapshot_complete(p))
            (p / "b.safetensors").write_bytes(b"test")
            self.assertTrue(t._snapshot_complete(p))


@unittest.skipIf(np is None, "numpy is required for vector/index integration tests")
class IndexSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        t._atomic_json(self.run_dir / "classifier.json", head())
        self.run = {"id": "run", "model_path": str(self.run_dir), "mode": "classifier", "threshold": .95}
        self.index_dir, self.cache = self.root / "index", self.root / "cache"
        self.cancel = threading.Event()
        self.seen_queries = []
        outer = self
        class FakeModel:
            device = "cpu"
            def encode(self, inputs, **kwargs):
                outer.seen_queries.append((inputs, kwargs))
                result = np.zeros((len(inputs), t.DIM), dtype=np.float32)
                result[:, 0] = 1
                return result
        self.model = FakeModel()

    def tearDown(self):
        self.tmp.cleanup()

    def records(self, n=6):
        records = []
        for i in range(n):
            path = self.root / f"image-{i}.png"
            path.write_bytes(b"temporary-test-input")
            stat = path.stat()
            records.append({"id": str(i), "path": str(path), "render_path": str(path), "sha256": str(i),
                            "label": None, "group": str(i), "frame": 0, "name": f"image-{i}.png",
                            "metadata": {"sample": "sample-A" if i < 3 else "sample-B", "acquired_at": "2026-10-01",
                                         "description": "pore" if i % 3 == 2 else "microscope",
                                         "provenance": {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}}})
        return records

    @staticmethod
    def encode(model, rows, batch_size=1, cancel=None):
        t._check(cancel)
        result = np.zeros((len(rows), t.DIM), dtype=np.float32)
        for i, row in enumerate(rows):
            result[i, int(row["id"]) % 3] = 1
        return result

    def build(self, records):
        with patch.object(t, "_load_model", return_value=self.model), patch.object(t, "_encode", side_effect=self.encode):
            return t.index(records, self.run, {}, self.index_dir, self.cache, None, self.cancel)

    def test_persistent_index_hard_gate_and_hybrid_query(self):
        records = self.records()
        self.assertEqual(self.build(records)["indexed"], 6)
        records[0]["label"] = "OTHER"
        records[2]["label"] = "SEM"  # Manual SEM must not override OTHER detector.
        with patch.object(t, "_load_model", return_value=self.model):
            result = t.search(records, self.run, {"query": "pore"}, self.index_dir, self.cache)
        self.assertEqual({item["id"] for item in result["items"]}, {"1", "3", "4"})
        self.assertEqual(result["excluded"], 3)
        self.assertEqual(self.seen_queries[0][0], [t.QUERY_PREFIX + "pore"])
        self.assertEqual(self.seen_queries[0][1]["prompt"], "")
        self.assertTrue(all(item["similarity"] is not None for item in result["items"]))
        self.assertEqual(self.build(records)["reused"], 6)

    def test_filters_pagination_and_latest_metadata(self):
        records = self.records()
        self.build(records)
        records[3]["metadata"]["sample"] = "renamed"
        result = t.search(records, self.run, {"query": "", "sample": "renamed", "label": "SEM",
                                             "date_from": "2026-10-01", "date_to": "2026-10-02"}, self.index_dir, self.cache)
        self.assertEqual([item["id"] for item in result["items"]], ["3"])
        first = t.search(records, self.run, {"limit": 1}, self.index_dir, self.cache)
        second = t.search(records, self.run, {"limit": 1, "offset": 1}, self.index_dir, self.cache)
        self.assertNotEqual(first["items"][0]["id"], second["items"][0]["id"])
        self.assertEqual(first["total"], 4)
        self.assertEqual(t.search(records, self.run, {"threshold": 1}, self.index_dir, self.cache)["total"], 0)

    def test_changed_source_and_head_invalidate(self):
        records = self.records()
        self.build(records)
        Path(records[0]["path"]).write_bytes(b"changed-source")
        Path(records[1]["path"]).unlink()
        result = t.search(records, self.run, {}, self.index_dir, self.cache)
        self.assertEqual(result["stale"], 2)
        self.assertNotIn("0", [item["id"] for item in result["items"]])
        modified = head()
        modified["intercept"][0] = .1
        t._atomic_json(self.run_dir / "classifier.json", modified)
        with self.assertRaisesRegex(ValueError, "다시"):
            t.search(records, self.run, {}, self.index_dir, self.cache)

    def test_cancelled_index_resumes_committed_chunks(self):
        records = self.records(70)
        calls = 0
        def cancelling(model, rows, batch_size=1, cancel=None):
            nonlocal calls
            calls += 1
            if calls == 65:
                raise t.Cancelled("test cancellation")
            return self.encode(model, rows, batch_size, cancel)
        with patch.object(t, "_load_model", return_value=self.model), patch.object(t, "_encode", side_effect=cancelling):
            with self.assertRaises(t.Cancelled):
                t.index(records, self.run, {}, self.index_dir, self.cache, None, self.cancel)
        self.assertFalse((self.index_dir / "manifest.json").exists())
        resumed = self.build(records)
        self.assertEqual(resumed["reused"], 64)
        self.assertEqual(resumed["indexed"], 70)
        self.assertFalse((self.index_dir / "checkpoint.json").exists())

    def test_missing_index_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "먼저 색인"):
            t.search(self.records(), self.run, {}, self.index_dir, self.cache)

    def test_classifier_training_persists_heldout_run(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest("scikit-learn is required for full frozen-head integration")
        records = training_records()
        for row in records:
            path = self.root / (row["id"] + ".png")
            path.write_bytes(b"temporary-training-input")
            row["path"] = row["render_path"] = str(path)
        def encode(model, rows, batch_size=1, cancel=None):
            vectors = np.zeros((len(rows), t.DIM), dtype=np.float32)
            for i, row in enumerate(rows):
                vectors[i, t.CLASSES.index(row["label"])] = 1
            return vectors
        output = self.root / "trained"
        with patch.object(t, "_load_model", return_value=self.model), patch.object(t, "_encode", side_effect=encode):
            run = t.train(records, {"threshold": .5}, output, self.cache, None, self.cancel)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["metrics"]["false_positives"], 0)
        self.assertEqual(run["metrics"]["precision"], 1)
        self.assertEqual(run["metrics"]["test_count"], run["splits"]["test"]["images"])
        self.assertEqual(json.loads((output / "run.json").read_text())["threshold"], run["threshold"])
        self.assertTrue((output / "split.json").is_file())
        self.assertFalse((output / "training-features.npy").exists())
        self.assertEqual(t._read_head(run)["classes"], list(t.CLASSES))


if __name__ == "__main__":
    unittest.main()
