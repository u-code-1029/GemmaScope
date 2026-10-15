#!/usr/bin/env python3
"""Opt-in real model smoke test. Synthetic files stay in a temporary directory.

This verifies execution and serialization, not SEM/TEM accuracy. Default is
strictly offline; use --download to explicitly permit fetching model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
try:
    import resource
except ImportError:  # Windows has no stdlib resource module.
    resource = None
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import training as t


def parameter_digest(model, trainable):
    import torch
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != trainable:
            continue
        count += parameter.numel()
        digest.update(name.encode())
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
        del raw
    return digest.hexdigest(), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".gemmascope" / "models")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "test-results" / "model-smoke.json")
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    cancel = threading.Event()
    result = {"model_id": t.MODEL_ID, "revision": t.MODEL_REVISION, "synthetic_execution_test": True,
              "accuracy_evaluated": False, "status": "running", "finetune_requested": args.finetune}
    def progress(phase, fraction, message):
        print(json.dumps({"phase": phase, "progress": fraction, "message": message}, ensure_ascii=False), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="gemmascope-model-smoke-") as temporary:
            directory = Path(temporary)
            rng = np.random.default_rng(42)
            records = []
            for c, label in enumerate(t.CLASSES):
                for group in range(2):
                    pixels = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
                    pixels[:, :, c] = np.clip(pixels[:, :, c].astype(np.int16) + 30, 0, 255).astype(np.uint8)
                    path = directory / f"{label}-{group}.png"
                    Image.fromarray(pixels).save(path)
                    records.append({"id": path.stem, "path": str(path), "render_path": str(path),
                                    "label": label, "group": path.stem, "sha256": path.stem, "frame": 0})
            progress("model", 0, "실제 모델을 불러옵니다. 이 검사는 정확도를 측정하지 않습니다.")
            model = t._load_model(args.cache, args.device, not args.download)
            base = t._encode(model, records[:2], 1, cancel)
            query = np.asarray(model.encode([t.QUERY_PREFIX + "microscopy texture"], prompt="",
                               normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
            assert base.shape == (2, t.DIM) and query.shape == (1, t.DIM)
            assert np.isfinite(base).all() and np.isfinite(query).all()
            assert np.allclose(np.linalg.norm(base, axis=1), 1, atol=.01)
            result.update(device=str(model.device), dtype=str(model.dtype), image_encode=True, text_encode=True,
                          embedding_shape=list(base.shape), base_cosine=float(base[0] @ query[0]))
            if args.finetune:
                selected = t.choose_partial_parameters(model)
                before_trainable, trainable_count = parameter_digest(model, True)
                before_frozen, frozen_count = parameter_digest(model, False)
                output = directory / "run"
                output.mkdir()
                config = t._config({"mode": "finetune", "epochs": 1, "batch_size": 1,
                                    "gradient_accumulation": 1, "learning_rate": 2e-5, "device": args.device})
                # Production train() also invalidates this cache before mutation.
                t._MODEL_KEY = None
                adaptation = t._finetune(model, records, config, output, progress, cancel)
                after_trainable, _ = parameter_digest(model, True)
                after_frozen, _ = parameter_digest(model, False)
                assert before_trainable != after_trainable, "Trainable model weights did not change"
                assert before_frozen == after_frozen, "Frozen model weights changed"
                adapted = t._encode(model, records[:2], 1, cancel)
                del model
                t._clear_model()
                reloaded = t._load_model(args.cache, args.device, True, output / "encoder")
                restored = t._encode(reloaded, records[:2], 1, cancel)
                max_error = float(np.max(np.abs(restored - adapted)))
                tolerance = .005 if "bfloat16" in str(reloaded.dtype) else .0001
                assert np.allclose(restored, adapted, atol=tolerance, rtol=tolerance), "Saved/reloaded embeddings differ"
                result.update(finetune=True, trainable_parameters=trainable_count, frozen_parameters=frozen_count,
                              trainable_weights_changed=True, frozen_weights_unchanged=True,
                              save_reload_max_abs_error=max_error, selected_parameter_tensors=len(selected),
                              optimizer_steps=adaptation["optimizer_steps"], training_loss=adaptation["training_loss"],
                              train_runtime_seconds=adaptation["train_runtime_seconds"])
                del reloaded
                t._clear_model()
            result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if resource is not None:
            maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_process_rss_mib"] = round(maximum_rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)
        t._atomic_json(args.report, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
