"""Read-only microscopy adapters. Heavy dependencies are imported only on demand.

Frame ranges are inclusive and zero based. A frame is a two-dimensional image;
spectra, navigation maps with >1 navigation axis, and MRC volumes are rejected.
Scientific images and >8-bit rasters are min/max normalized for model/display
copies only. Source files and their numeric pixels are never written.
"""
from __future__ import annotations

import importlib
import copy
import math
import mmap
import struct
import threading
import warnings
from pathlib import Path

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
SCIENTIFIC_EXTENSIONS = {".dm3", ".dm4", ".ser", ".emi", ".mrc"}
SUPPORTED_EXTENSIONS = RASTER_EXTENSIONS | SCIENTIFIC_EXTENSIONS
PREPROCESS_VERSION = "rgb-minmax-2048-v1"
_READER_GRAPH_LOCK = threading.RLock()


class UnsupportedImage(ValueError):
    pass


class MissingReader(ImportError):
    pass


def _pillow():
    try:
        from PIL import Image, ImageOps
        return Image, ImageOps
    except ImportError as exc:
        raise MissingReader("이미지를 가져오려면 먼저 환경 설정에서 Pillow를 설치해 주세요.") from exc


def _json_safe(value, depth=0):
    """Bound metadata, avoiding embedded data arrays or binary payloads."""
    if depth > 7:
        return "[생략]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:4000]
    if isinstance(value, dict):
        return {str(k)[:200]: _json_safe(v, depth + 1) for k, v in list(value.items())[:80]}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value[:40]]
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return _json_safe(value.item(), depth + 1)
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)[:1000]


def _normalize(image):
    Image, ImageOps = _pillow()
    image = ImageOps.exif_transpose(image)
    numeric = image.mode in {"I", "F", "I;16", "I;16B", "I;16L", "I;16N"}
    if numeric:
        # NumPy handles signed/float/16-bit values, NaNs and endian variants.
        try:
            import numpy as np
        except ImportError as exc:
            raise MissingReader("16비트 이미지 처리를 위해 환경 설정에서 NumPy를 설치해 주세요.") from exc
        values = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(values)
        if not finite.any():
            raise UnsupportedImage("유효한 픽셀 값이 없는 이미지입니다.")
        lo, hi = float(values[finite].min()), float(values[finite].max())
        values = np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo)
        values = np.clip((values - lo) * (255.0 / (hi - lo)), 0, 255) if hi > lo else np.zeros_like(values)
        image = Image.fromarray(values.astype("uint8"), "L").convert("RGB")
        normalization = {"method": "minmax", "min": lo, "max": hi}
    else:
        if image.mode == "RGBA" or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            base = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(base, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        normalization = {"method": "rgb"}
    width, height = image.size
    if width < 2 or height < 2:
        raise UnsupportedImage("2차원 이미지의 너비와 높이는 각각 2픽셀 이상이어야 합니다.")
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    normalization.update({"version": PREPROCESS_VERSION, "render_size": list(image.size)})
    return image, width, height, normalization


def _raster_frames(path, start, end):
    Image, _ = _pillow()
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            frames = getattr(source, "n_frames", 1)
            if start >= frames:
                raise UnsupportedImage(f"선택한 시작 프레임이 이미지 프레임 수({frames})를 초과합니다.")
            for frame in range(start, min(end + 1, frames)):
                source.seek(frame)
                source.load()
                original_mode = source.mode
                image, width, height, normalization = _normalize(source.copy())
                exif = source.getexif()
                captured = str(exif.get(36867, exif.get(306, "")))
                if len(captured) >= 10:
                    captured = captured[:10].replace(":", "-") + captured[10:]
                yield {"image": image, "frame": frame, "frames": frames, "width": width, "height": height,
                       "metadata": {"acquired_at": captured, "provenance": {"reader": "Pillow", "format": source.format,
                         "original_mode": original_mode, "frame": frame, "normalization": normalization}}}


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _signal_layout(signal):
    signal_type = str(_get(signal.get("metadata", {}), "Signal", "signal_type") or "").upper()
    if signal_type.startswith(("EELS", "EDS", "EDX", "CL", "RAMAN", "XPS")):
        raise UnsupportedImage("분광 데이터는 지원하지 않습니다. 2D 현미경 이미지만 선택해 주세요.")
    shape = tuple(signal["data"].shape)
    if len(shape) not in (2, 3):
        raise UnsupportedImage("스펙트럼·4D 데이터·볼륨은 지원하지 않습니다. 2D 이미지 또는 이미지 스택만 선택해 주세요.")
    axes = signal.get("axes", [])
    annotated = [a for a in axes if "navigate" in a]
    if annotated:
        image_axes = [int(a.get("index_in_array", i)) for i, a in enumerate(axes) if not a.get("navigate", False)]
        nav_axes = [int(a.get("index_in_array", i)) for i, a in enumerate(axes) if a.get("navigate", False)]
        if len(image_axes) != 2 or len(nav_axes) > 1:
            raise UnsupportedImage("스펙트럼·4D 데이터·볼륨은 지원하지 않습니다.")
        return shape, nav_axes[0] if nav_axes else None
    if len(shape) == 3:
        # No guessing whether an unannotated third dimension is a volume.
        raise UnsupportedImage("세 번째 축의 의미가 없는 데이터는 볼륨과 구분할 수 없어 건너뜁니다.")
    return shape, None


def _mapped_metadata(signal):
    """Apply the reader's public metadata mapping for the handful of UI fields."""
    metadata = copy.deepcopy(signal.get("metadata", {}))
    original = signal.get("original_metadata", {})
    wanted = {"General.date", "General.time", "Sample.description", "Acquisition_instrument.TEM.magnification",
              "Acquisition_instrument.SEM.magnification", "Acquisition_instrument.TEM.beam_energy",
              "Acquisition_instrument.SEM.beam_energy"}
    for source, mapping in signal.get("mapping", {}).items():
        if not isinstance(mapping, (list, tuple)) or len(mapping) != 2 or mapping[0] not in wanted:
            continue
        target, convert = mapping
        value = original
        try:
            for key in source.split("."):
                if key not in value:
                    # Reader mapping uses Box-style names: both literal
                    # underscores and spaces can occur within one real key.
                    matches = [actual for actual in value if isinstance(actual, str) and actual.replace(" ", "_") == key]
                    if len(matches) == 1:
                        key = matches[0]
                value = value[key]
            value = convert(value) if convert else value
            node = metadata
            parts = target.split(".")
            for key in parts[:-1]:
                node = node.setdefault(key, {})
            node[parts[-1]] = value
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return metadata


def _reject_mrc_volume(path):
    with open(path, "rb") as source:
        header = source.read(1024)
    if len(header) < 1024:
        raise ValueError("MRC 헤더가 손상되었습니다.")
    for endian in ("<", ">"):
        nx, ny, nz = struct.unpack_from(endian + "3i", header, 0)
        if 0 < nx < 1000000 and 0 < ny < 1000000 and 0 < nz < 1000000:
            ispg = struct.unpack_from(endian + "i", header, 88)[0]
            if nz > 1 and ispg != 0:
                raise UnsupportedImage("3D MRC 볼륨은 지원하지 않습니다. ISPG=0인 2D 이미지 스택만 지원합니다.")
            return
    raise ValueError("MRC 크기 정보가 올바르지 않습니다.")


def _close_mapped_arrays(*arrays):
    """Release mappings owned by this decode, including ndarray.base views.

    RosettaSciIO's lazy DM/MRC readers can return numpy.memmap slices. Merely
    dropping the result is insufficient when a Dask task retains a reference,
    and Windows then prevents the source from being moved or removed.
    Call only after copying pixels out of the mapping; never return those views.
    """
    seen = set()
    for array in arrays:
        while array is not None and id(array) not in seen:
            seen.add(id(array))
            mapping = array if isinstance(array, mmap.mmap) else getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
            array = getattr(array, "base", None)


def _render_scientific_frame(data):
    import numpy as np
    Image, _ = _pillow()
    pixels = None
    try:
        if hasattr(data, "compute"):
            # Avoid worker-thread task state retaining a mapped chunk after
            # compute() returns. This computes only the selected 2D slice.
            data = data.compute(scheduler="synchronous")
        pixels = np.asarray(data)
        if pixels.ndim != 2 or np.iscomplexobj(pixels) or pixels.dtype.kind not in "biuf":
            raise UnsupportedImage("실수형 2D 이미지 데이터만 지원합니다.")
        dtype = str(pixels.dtype)
        # A private copy must outlive explicit mmap closure, including errors
        # during normalization and consumers that stop after the first yield.
        copied = np.array(pixels, dtype=np.float32, copy=True, order="C")
        rendered = _normalize(Image.fromarray(copied, "F"))
        return (*rendered, dtype)
    finally:
        _close_mapped_arrays(pixels, data)


def _owned_memmap_chunk(slices, file, dtypes, shape, key=None, positions=False, **kwargs):
    """A RosettaSciIO Dask chunk whose source mapping has an explicit owner.

    RosettaSciIO 0.15's slice_memmap returns a live mapped view. Dask also calls
    it while inferring _meta, so cleaning only compute() results misses maps.
    Materialize just this requested chunk and close its mapping before return.
    """
    import numpy as np
    selection = np.squeeze(slices)[()]
    mapped = np.memmap(file, dtypes, shape=shape, **kwargs)
    try:
        values = mapped[key] if key is not None else mapped
        if positions:
            output = np.array(values[selection], copy=True, subok=False)
            if -1 in selection:
                output[selection == -1] = 0
            return output
        if selection.ndim == 1:
            index = slice(*tuple(selection))
        else:
            index = tuple(slice(int(pair[0]), int(pair[1])) for pair in selection)
        return np.array(values[index], copy=True, subok=False)
    finally:
        _close_mapped_arrays(mapped)


def _read_scientific_signals(adapter, path):
    # Scope the compatibility adapter to graph creation. Dask stores the owned
    # callable in its graph; decoding later does not depend on this module hook.
    # Serializing installation/restoration also protects concurrent importers.
    distributed = importlib.import_module("rsciio.utils._distributed")
    with _READER_GRAPH_LOCK:
        original = distributed.slice_memmap
        distributed.slice_memmap = _owned_memmap_chunk
        try:
            return adapter.file_reader(str(path), lazy=True)
        finally:
            distributed.slice_memmap = original


def _scientific_frames(path, start, end):
    suffix = path.suffix.lower()
    if suffix == ".mrc":
        _reject_mrc_volume(path)
    plugin = "digitalmicrograph" if suffix in {".dm3", ".dm4"} else "tia" if suffix in {".ser", ".emi"} else "mrc"
    try:
        try:
            adapter = importlib.import_module("rsciio." + plugin)
        except ModuleNotFoundError:
            if plugin != "tia":
                raise
            adapter = importlib.import_module("rsciio.fei")
        import numpy as np
    except ImportError as exc:
        raise MissingReader("과학 이미지 형식 처리를 위해 환경 설정에서 RosettaSciIO를 설치해 주세요.") from exc
    Image, _ = _pillow()
    signals = _read_scientific_signals(adapter, path)
    eligible = []
    rejected = []
    total = 0
    for number, signal in enumerate(signals):
        try:
            shape, nav = _signal_layout(signal)
            count = shape[nav] if nav is not None else 1
            eligible.append((number, signal, shape, nav, count, total))
            total += count
        except UnsupportedImage as exc:
            rejected.append(str(exc))
    if not eligible:
        raise UnsupportedImage(rejected[0] if rejected else "2D 이미지가 들어 있지 않습니다.")
    if start >= total:
        raise UnsupportedImage(f"선택한 시작 프레임이 이미지 프레임 수({total})를 초과합니다.")
    for number, signal, shape, nav, count, offset in eligible:
        for frame in range(max(start, offset), min(end + 1, offset + count)):
            data = signal["data"]
            if nav is not None:
                selection = [slice(None)] * len(shape)
                selection[nav] = frame - offset
                data = data[tuple(selection)]
            image, width, height, normalization, dtype = _render_scientific_frame(data)
            md = _mapped_metadata(signal)
            tem = _get(md, "Acquisition_instrument", "TEM") or _get(md, "Acquisition_instrument", "SEM") or {}
            metadata = {"sample": _get(md, "Sample", "description") or "", "acquired_at": _get(md, "General", "date") or "",
                        "magnification": tem.get("magnification", ""), "voltage_kv": tem.get("beam_energy", ""),
                        "description": _get(md, "General", "title") or "",
                        "provenance": {"reader": "RosettaSciIO/" + plugin, "format": suffix[1:], "signal_index": number,
                                       "frame": frame, "signal_frame": frame - offset, "shape": list(shape),
                                       "dtype": dtype, "axes": _json_safe(signal.get("axes", [])),
                                       "source_metadata": _json_safe(md), "normalization": normalization,
                                       "original_metadata_keys": list(signal.get("original_metadata", {}))[:40],
                                       "unsupported_signals": len(rejected)}}
            yield {"image": image, "frame": frame, "frames": total, "width": width, "height": height, "metadata": _json_safe(metadata)}


def iter_frames(path, frame_start=0, frame_end=0):
    path = Path(path)
    if path.suffix.lower() in RASTER_EXTENSIONS:
        yield from _raster_frames(path, frame_start, frame_end)
    elif path.suffix.lower() in SCIENTIFIC_EXTENSIONS:
        yield from _scientific_frames(path, frame_start, frame_end)
    else:
        raise UnsupportedImage("지원하지 않는 파일 형식입니다.")
