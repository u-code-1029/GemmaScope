"""Local microscopy detector training and gated multimodal retrieval.

Importing this module does not import ML dependencies or contact a network.
Model weights are downloaded only by an explicit train/index operation with
offline=False. Image contents and training data are never uploaded.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from collections import Counter, defaultdict
import uuid

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
CLASSES = ("SEM", "TEM", "OTHER")
DIM = 768
PREPROCESS_VERSION = "rgb-first-frame-normalized-v1-st6.1-768"
QUERY_PREFIX = "task: search result | query: "
_MODEL_HANDLE = None
_MODEL_KEY = None
_SMOKE_KEY = None


class Cancelled(Exception):
    pass


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("작업을 취소했습니다. 완료한 색인 배치는 다음 실행에 재사용됩니다.")


def _emit(progress, phase, fraction, message):
    if progress:
        progress(phase, max(0.0, min(1.0, float(fraction))), message)


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _threshold(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("통과 기준은 0~1 사이여야 합니다.")
    return value


def _config(config):
    result = dict(config)
    result.setdefault("mode", "classifier")
    if result["mode"] not in ("classifier", "finetune"):
        raise ValueError("지원하지 않는 학습 모드입니다.")
    for key, default, low, high in (("epochs", 3, 1, 100), ("batch_size", 1, 1, 64),
                                   ("gradient_accumulation", 8, 1, 128)):
        value = int(result.get(key, default))
        if not low <= value <= high:
            raise ValueError(f"{key}: {low}~{high} 사이 값을 사용하세요.")
        result[key] = value
    result["learning_rate"] = float(result.get("learning_rate", 2e-5))
    if not 0 < result["learning_rate"] <= .01:
        raise ValueError("학습률은 0보다 크고 0.01 이하여야 합니다.")
    result["threshold"] = _threshold(result.get("threshold", .95))
    result.setdefault("device", "auto")
    result.setdefault("offline", False)
    return result


def environment(cache_dir):
    packages = {"numpy": "numpy", "PIL": "Pillow", "torch": "torch",
                "torchvision": "torchvision",
                "sentence_transformers": "sentence-transformers", "transformers": "transformers",
                "datasets": "datasets", "accelerate": "accelerate", "sklearn": "scikit-learn",
                "safetensors": "safetensors"}
    missing = []
    for module, package in packages.items():
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(package)
        except (ValueError, ModuleNotFoundError):
            missing.append(package)
    device, error, compatible = "cpu", None, False
    if not missing:
        try:
            import torch
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            from importlib.metadata import version
            st_version = tuple(int(x) for x in version("sentence-transformers").split(".")[:2])
            compatible = "embedding_gemma2" in CONFIG_MAPPING and st_version >= (6, 1)
            if not compatible:
                error = "EmbeddingGemma 2 지원 버전이 필요합니다. 학습 환경을 다시 설치하세요."
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except Exception as exc:
            error = "학습 라이브러리 점검 실패: " + str(exc)
    snapshot = Path(cache_dir) / "models--google--embeddinggemma-2" / "snapshots" / MODEL_REVISION
    cached = _snapshot_complete(snapshot)
    smoke_path = Path(cache_dir) / "gemmascope-smoke.json"
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke_ok = smoke.get("revision") == MODEL_REVISION and smoke.get("preprocess") == PREPROCESS_VERSION
    except (OSError, ValueError):
        smoke_ok = False
    return {"python": sys.executable, "python_version": sys.version.split()[0], "device": device,
            "ml_ready": not missing and compatible, "missing": missing,
            "model_cached": cached, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "compatibility_error": error, "model_smoke_tested": smoke_ok}


def _snapshot_complete(snapshot):
    if not (snapshot / "modules.json").is_file() or not (snapshot / "config.json").is_file():
        return False
    shard_index = snapshot / "model.safetensors.index.json"
    if shard_index.is_file():
        try:
            shards = set(json.loads(shard_index.read_text())["weight_map"].values())
            return bool(shards) and all((snapshot / name).is_file() for name in shards)
        except (OSError, ValueError, KeyError):
            return False
    return (snapshot / "model.safetensors").is_file()


def _device(requested):
    import torch
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("장치는 auto, cpu, mps 또는 cuda여야 합니다.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다. PyTorch CUDA 설치 또는 CPU를 선택하세요.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("이 환경에서 Apple MPS를 사용할 수 없습니다. CPU를 선택하세요.")
    return requested


def _clear_model():
    global _MODEL_HANDLE, _MODEL_KEY
    _MODEL_HANDLE = _MODEL_KEY = None
    gc.collect()
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _load_model(cache_dir, device="auto", offline=True, source=None):
    global _MODEL_HANDLE, _MODEL_KEY
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError as exc:
        raise RuntimeError("학습 환경이 없습니다. 설정에서 학습 환경 설치를 실행하세요.") from exc
    if "embedding_gemma2" not in CONFIG_MAPPING:
        raise RuntimeError("설치된 Transformers가 EmbeddingGemma 2를 지원하지 않습니다. 환경을 다시 설치하세요.")
    device = _device(device)
    source = str(source or MODEL_ID)
    key = (source, device, str(Path(cache_dir).resolve()), MODEL_REVISION)
    if _MODEL_KEY == key and _MODEL_HANDLE is not None:
        return _MODEL_HANDLE
    _clear_model()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    kwargs = dict(device=device, cache_folder=str(cache_dir), local_files_only=bool(offline),
                  trust_remote_code=False, config_kwargs={"audio_config": None},
                  model_kwargs={"torch_dtype": dtype, "use_safetensors": True})
    if source == MODEL_ID:
        kwargs["revision"] = MODEL_REVISION
    try:
        model = SentenceTransformer(source, **kwargs)
        model.eval()
    except Exception as exc:
        hint = "오프라인 캐시에 모델이 없습니다. 온라인 모드에서 한 번 내려받으세요." if offline else (
            "모델 다운로드/로드에 실패했습니다. 인터넷 연결과 학습 환경 버전을 확인하세요.")
        raise RuntimeError(hint + " " + str(exc)) from exc
    _MODEL_HANDLE, _MODEL_KEY = model, key
    return model


def _oom(exc):
    return any(word in str(exc).lower() for word in ("out of memory", "mps backend out", "allocation"))


def _encode(model, records, batch_size=1, cancel=None):
    global _SMOKE_KEY
    import numpy as np
    from PIL import Image, ImageOps
    images = []
    try:
        for record in records:
            _check(cancel)
            source = record.get("render_path") or record.get("path")
            with Image.open(source) as original:
                original.seek(0)
                images.append(ImageOps.exif_transpose(original).convert("RGB"))
        vectors = np.asarray(model.encode(images, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if vectors.shape != (len(records), DIM) or not np.isfinite(vectors).all():
            raise ValueError("모델이 유효한 768차원 이미지 임베딩을 반환하지 않았습니다.")
        if _MODEL_KEY and _MODEL_KEY[0] == MODEL_ID and _SMOKE_KEY != _MODEL_KEY:
            _atomic_json(Path(_MODEL_KEY[2]) / "gemmascope-smoke.json",
                         {"revision": MODEL_REVISION, "preprocess": PREPROCESS_VERSION,
                          "device": str(model.device), "image_encode": True,
                          "checked_at": datetime.now(timezone.utc).isoformat()})
            _SMOKE_KEY = _MODEL_KEY
        return vectors
    finally:
        for im in images:
            im.close()


def split_records(records, mode="classifier", seed=42):
    """Split whole specimen groups; duplicates cannot connect independent splits."""
    unique, hashes = [], {}
    group_parent = {}
    def root(group):
        group_parent.setdefault(group, group)
        while group_parent[group] != group:
            group_parent[group] = group_parent[group_parent[group]]
            group = group_parent[group]
        return group
    for original in records:
        record = dict(original)
        if record.get("label") not in CLASSES:
            continue
        group = str(record.get("group") or "").strip()
        if not group:
            raise ValueError("모든 학습 이미지에 시료 그룹을 지정하세요.")
        record["group"] = group
        root(group)
        # Different frames of a stack are distinct images, still in one group.
        key = (record.get("sha256") or str(record.get("render_path") or record["path"]), record.get("frame", 0))
        if key in hashes:
            previous = hashes[key]
            if previous["label"] != record["label"]:
                raise ValueError("같은 이미지에 서로 다른 라벨이 지정되어 있습니다.")
            group_parent[root(group)] = root(previous["group"])
        else:
            hashes[key] = record
            unique.append(record)
    grouped = defaultdict(list)
    for record in unique:
        record["group"] = root(record["group"])
        grouped[record["group"]].append(record)
    counts = {g: Counter(r["label"] for r in rows) for g, rows in grouped.items()}
    groups_per_class = {c: [g for g in grouped if counts[g][c]] for c in CLASSES}
    minimum = {c: 4 if mode == "finetune" and c != "OTHER" else 3 for c in CLASSES}
    for c in CLASSES:
        if len(groups_per_class[c]) < minimum[c]:
            raise ValueError(f"{c}: 독립된 시료 그룹이 최소 {minimum[c]}개 필요합니다. "
                             "같은 시료의 이미지는 같은 그룹으로 묶으세요.")
    totals = Counter(r["label"] for r in unique)
    targets = {s: {c: totals[c] * f for c in CLASSES} for s, f in
               (("train", .7), ("validation", .15), ("test", .15))}
    best, best_cost = None, float("inf")
    rng = random.Random(seed)
    for attempt in range(384):
        bins = {s: [] for s in targets}
        remaining = set(grouped)
        order = list(bins)
        rng.shuffle(order)
        valid = True
        for s in order:
            required = {c: 2 if s == "train" and mode == "finetune" and c != "OTHER" else 1 for c in CLASSES}
            while any(required.values()):
                unmet = [c for c in CLASSES if required[c] > 0]
                c = min(unmet, key=lambda x: sum(counts[g][x] > 0 for g in remaining))
                available = sorted(g for g in remaining if counts[g][c])
                if not available:
                    valid = False
                    break
                rng.shuffle(available)
                g = max(available, key=lambda g: sum(required[x] > 0 and counts[g][x] > 0 for x in CLASSES))
                bins[s].append(g)
                remaining.remove(g)
                for x in CLASSES:
                    if counts[g][x] > 0:
                        required[x] = max(0, required[x] - 1)
            if not valid:
                break
        if not valid:
            continue
        current = {s: Counter({c: sum(counts[g][c] for g in bins[s]) for c in CLASSES}) for s in bins}
        rest = sorted(remaining)
        rng.shuffle(rest)
        rest.sort(key=lambda g: -len(grouped[g]))
        for g in rest:
            s = min(bins, key=lambda s: sum(((current[s][c] + counts[g][c] - targets[s][c]) ** 2
                     - (current[s][c] - targets[s][c]) ** 2) / max(1, targets[s][c]) for c in CLASSES))
            bins[s].append(g)
            current[s].update(counts[g])
        cost = sum((current[s][c] - targets[s][c]) ** 2 / max(1, targets[s][c]) for s in bins for c in CLASSES)
        if cost < best_cost:
            best, best_cost = bins, cost
    if best is None:
        raise ValueError("시료 그룹을 중복 없이 학습/검증/시험으로 나눌 수 없습니다. 독립 시료를 더 추가하세요.")
    return {s: [r for g in groups for r in grouped[g]] for s, groups in best.items()}


def choose_partial_parameters(model):
    """Freeze all parameters, then select native final-two-block parameters."""
    backbone = getattr(model, "transformers_model", None)
    if backbone is None:
        first = model[0]
        backbone = getattr(first, "auto_model", None)
    language = getattr(backbone, "language_model", None)
    if language is None or len(getattr(language, "layers", [])) < 2:
        raise RuntimeError("모델의 마지막 두 블록을 확인할 수 없습니다. 지원되는 모델 버전인지 확인하세요.")
    for p in model.parameters():
        p.requires_grad_(False)
    modules = [*language.layers[-2:], language.norm, language.embedding_projection]
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise RuntimeError("미세조정할 모델 가중치를 찾지 못했습니다.")
    return names


def make_triplets(records, seed=42):
    rng = random.Random(seed)
    triplets = []
    for record in records:
        if record["label"] == "OTHER":
            continue
        positives = [r for r in records if r["label"] == record["label"] and r["group"] != record["group"]]
        negatives = [r for r in records if r["label"] != record["label"]]
        if not positives or not negatives:
            raise ValueError("SEM/TEM 각각 다른 시료의 양성 이미지와 음성 이미지가 필요합니다.")
        triplets.append((record, rng.choice(positives), rng.choice(negatives)))
    return triplets


def _finetune(model, records, config, output_dir, progress, cancel):
    import torch
    from datasets import Dataset, Features, Image as DatasetImage
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import TripletLoss, TripletDistanceMetric
    from transformers import TrainerCallback
    selected = choose_partial_parameters(model)
    triplets = make_triplets(records)
    columns = {name: [str(rows[i].get("render_path") or rows[i]["path"]) for rows in triplets]
               for i, name in enumerate(("anchor", "positive", "negative"))}
    dataset = Dataset.from_dict(columns, features=Features({k: DatasetImage() for k in columns}))
    device = str(model.device).split(":")[0]
    class Monitor(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            _check(cancel)
        def on_substep_end(self, args, state, control, **kwargs):
            _check(cancel)
        def on_step_end(self, args, state, control, **kwargs):
            _check(cancel)
            _emit(progress, "finetune", .1 + .4 * state.global_step / max(1, state.max_steps),
                  f"마지막 2개 블록 미세조정 · {state.global_step}/{state.max_steps} 스텝")
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "trainer"), num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"], learning_rate=config["learning_rate"],
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(), fp16=False,
        use_cpu=device == "cpu", dataloader_num_workers=0, dataloader_pin_memory=False,
        report_to="none", push_to_hub=False, save_strategy="no", logging_strategy="steps",
        logging_steps=1, disable_tqdm=True, seed=42, remove_unused_columns=False)
    loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=.2)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset,
                                         loss=loss, callbacks=[Monitor()])
    _emit(progress, "preflight", .1, "첫 미세조정 스텝으로 메모리와 역전파를 점검합니다.")
    # The first real optimizer step is the preflight; it remains part of training.
    output = trainer.train()
    _check(cancel)
    model.eval()
    model.save_pretrained(str(output_dir / "encoder"), safe_serialization=True)
    return {"trainable_names": selected, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "triplets": len(triplets), "optimizer_steps": trainer.state.global_step,
            "training_loss": float(output.training_loss),
            "train_runtime_seconds": float(output.metrics["train_runtime"]) if "train_runtime" in output.metrics else None}


def validate_head(head):
    if head.get("schema") != 1 or head.get("classes") != list(CLASSES) or head.get("dimension") != DIM:
        raise ValueError("지원하지 않는 분류기 파일입니다.")
    coef, bias = head.get("coef"), head.get("intercept")
    if not isinstance(coef, list) or len(coef) != 3 or not isinstance(bias, list) or len(bias) != 3:
        raise ValueError("분류기 배열 크기가 잘못되었습니다.")
    for row in coef:
        if not isinstance(row, list) or len(row) != DIM:
            raise ValueError("분류기 차원이 잘못되었습니다.")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError("분류기에 잘못된 숫자가 있습니다.")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in bias):
        raise ValueError("분류기에 잘못된 숫자가 있습니다.")
    return head


def _head_path(run):
    return Path(run["model_path"]) / "classifier.json"


def _read_head(run):
    path = _head_path(run)
    if not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("저장된 분류기가 없습니다. 학습을 완료하거나 모델 폴더를 확인하세요.")
    return validate_head(json.loads(path.read_text(encoding="utf-8")))


def probabilities(head, vectors):
    import numpy as np
    logits = np.asarray(vectors, dtype=np.float64) @ np.asarray(head["coef"], dtype=np.float64).T
    logits += np.asarray(head["intercept"], dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    result = exp / exp.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all():
        raise ValueError("분류 점수 계산에 실패했습니다.")
    return result


def accepted_label(scores, threshold, manual_label=None):
    if (manual_label == "OTHER" or threshold >= 1 or len(scores) != 3 or
            any(not math.isfinite(float(x)) or not 0 <= x <= 1 for x in scores)):
        return None
    winner = max(range(3), key=lambda i: scores[i])
    # Ties with OTHER are deliberately rejected.
    if winner == 2 or scores[winner] <= scores[2] or scores[winner] < threshold:
        return None
    return CLASSES[winner]


def metrics(labels, scores, threshold):
    confusion = {c: {p: 0 for p in ("SEM", "TEM", "REJECT")} for c in CLASSES}
    accepted = tp = fp = subtype_correct = positives = 0
    for actual, row in zip(labels, scores):
        predicted = accepted_label(row, threshold)
        confusion[actual][predicted or "REJECT"] += 1
        positives += actual != "OTHER"
        if predicted:
            accepted += 1
            fp += actual == "OTHER"
            tp += actual != "OTHER"
            subtype_correct += actual == predicted
    return {"precision": tp / accepted if accepted else None,
            "recall": tp / positives if positives else None, "false_positives": fp,
            "test_count": len(labels), "accepted": accepted, "true_positives": tp,
            "other_count": sum(x == "OTHER" for x in labels), "positive_count": positives,
            "subtype_accuracy": subtype_correct / tp if tp else None,
            "class_counts": dict(Counter(labels)), "confusion": confusion}


def select_threshold(labels, scores, minimum=.95):
    threshold = _threshold(minimum)
    for actual, row in zip(labels, scores):
        if actual == "OTHER" and accepted_label(row, threshold):
            threshold = min(1.0, math.nextafter(float(max(row[:2])), math.inf))
    return threshold


def train(records, config, output_dir, cache_dir, progress, cancel):
    global _MODEL_KEY
    config = _config(config)
    output_dir = Path(output_dir)
    if any(not _source_valid(r) for r in records if r.get("label") in CLASSES):
        raise ValueError("학습 원본이 변경되었거나 없습니다. 다시 가져오거나 경로를 연결한 뒤 학습하세요.")
    splits = split_records(records, config["mode"])
    _check(cancel)
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "model", .02, "EmbeddingGemma 2를 로컬에 불러옵니다. 이미지 데이터는 업로드하지 않습니다.")
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        model = _load_model(cache_dir, config["device"], config["offline"])
        adaptation = None
        if config["mode"] == "finetune":
            # The cached base instance is about to mutate: never reuse it as
            # the unchanged text/image retrieval encoder after this point.
            _MODEL_KEY = None
            adaptation = _finetune(model, splits["train"], config, output_dir, progress, cancel)
        all_records = [r for s in ("train", "validation", "test") for r in splits[s]]
        features = np.lib.format.open_memmap(output_dir / "training-features.npy", mode="w+",
                                            dtype="float32", shape=(len(all_records), DIM))
        start = .52 if adaptation else .12
        for offset in range(0, len(all_records), config["batch_size"]):
            _check(cancel)
            batch = all_records[offset:offset + config["batch_size"]]
            features[offset:offset + len(batch)] = _encode(model, batch, config["batch_size"], cancel)
            _emit(progress, "embedding", start + (.83 - start) * (offset + len(batch)) / len(all_records),
                  f"이미지 특징 추출 · {offset + len(batch)}/{len(all_records)}")
        features.flush()
        ntrain, nval = len(splits["train"]), len(splits["validation"])
        _check(cancel)
        _emit(progress, "classifier", .85, "SEM / TEM / 기타 분류기를 학습합니다.")
        classifier = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=42)
        classifier.fit(features[:ntrain], [r["label"] for r in splits["train"]])
        order = [list(classifier.classes_).index(c) for c in CLASSES]
        head = validate_head({"schema": 1, "classes": list(CLASSES), "dimension": DIM,
                              "coef": classifier.coef_[order].tolist(), "intercept": classifier.intercept_[order].tolist()})
        _atomic_json(output_dir / "classifier.json", head)
        validation_scores = probabilities(head, features[ntrain:ntrain + nval])
        threshold = select_threshold([r["label"] for r in splits["validation"]], validation_scores, config["threshold"])
        validation_metrics = metrics([r["label"] for r in splits["validation"]], validation_scores, threshold)
        test_metrics = metrics([r["label"] for r in splits["test"]],
                               probabilities(head, features[ntrain + nval:]), threshold)
        _check(cancel)
        _atomic_json(output_dir / "split.json", {s: [{k: r.get(k) for k in ("id", "sha256", "frame", "group", "label")}
                                                       for r in rows] for s, rows in splits.items()})
        del features
        (output_dir / "training-features.npy").unlink(missing_ok=True)
        run = {"id": output_dir.name, "created_at": datetime.now(timezone.utc).isoformat(),
               "mode": config["mode"], "status": "completed", "metrics": test_metrics,
               "validation_metrics": validation_metrics, "threshold": threshold,
               "model_path": str(output_dir.resolve()), "model_id": MODEL_ID,
               "base_revision": MODEL_REVISION, "preprocess": PREPROCESS_VERSION,
               "device": str(model.device), "config": config, "adaptation": adaptation,
               "splits": {s: {"images": len(rows), "groups": len(set(r["group"] for r in rows)),
                               "classes": dict(Counter(r["label"] for r in rows))} for s, rows in splits.items()},
               "notice": "검증 자료에서 기준을 정하고 독립 시험 자료에서 평가했습니다. 새 이미지에서 오탐 0을 보장하지 않습니다."}
        _atomic_json(output_dir / "run.json", run)
        _emit(progress, "done", 1, "학습과 독립 시험 평가를 완료했습니다.")
        return run
    except ImportError as exc:
        raise RuntimeError("학습 라이브러리가 없습니다. 설정에서 학습 환경 설치를 실행하세요.") from exc
    except Exception as exc:
        if _oom(exc):
            _clear_model()
            raise RuntimeError("메모리가 부족합니다. 다른 앱을 닫고 배치 1 또는 분류기 모드를 사용하세요. 현재 모델은 완료 상태로 저장되지 않았습니다.") from exc
        raise


def _signature(run):
    if run.get("base_revision", MODEL_REVISION) != MODEL_REVISION:
        raise ValueError("저장된 모델의 원본 리비전이 현재 앱과 다릅니다. 같은 리비전의 앱 또는 새 학습 모델을 사용하세요.")
    head = _read_head(run)
    encoder_files = []
    if run["mode"] == "finetune":
        encoder = Path(run["model_path"]) / "encoder"
        if not encoder.is_dir():
            raise ValueError("미세조정 인코더가 없습니다. 저장된 모델 폴더를 확인하세요.")
        for file in sorted(encoder.rglob("*")):
            if file.is_file() and file.suffix in (".json", ".safetensors"):
                stat = file.stat()
                encoder_files.append([str(file.relative_to(encoder)), stat.st_size, stat.st_mtime_ns])
    return _digest({"run": run["id"], "head": head, "revision": run.get("base_revision", MODEL_REVISION),
                    "preprocess": PREPROCESS_VERSION, "mode": run["mode"], "created_at": run.get("created_at"),
                    "encoder_files": encoder_files})


def _fingerprint(record):
    source = Path(record.get("render_path") or record["path"])
    try:
        stat = source.stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
    except OSError:
        stamp = None
    return _digest([record.get("sha256"), record.get("frame", 0), str(source), stamp])


def _source_valid(record):
    """Never show new original-file contents under an old image prediction."""
    try:
        source = Path(record["path"])
        if source.is_symlink() or not source.is_file():
            return False
        stat = source.stat()
    except OSError:
        return False
    provenance = (record.get("metadata") or {}).get("provenance") or {}
    size, mtime = provenance.get("source_size"), provenance.get("source_mtime_ns")
    if not ((size is None or stat.st_size == int(size)) and (mtime is None or stat.st_mtime_ns == int(mtime))):
        return False
    for dependency in provenance.get("source_dependencies") or []:
        try:
            dependent = Path(dependency["path"])
            if dependent.is_symlink() or not dependent.is_file():
                return False
            dstat = dependent.stat()
            if dstat.st_size != int(dependency["size"]) or dstat.st_mtime_ns != int(dependency["mtime_ns"]):
                return False
        except (OSError, KeyError, TypeError, ValueError):
            return False
    return True


def _index_db(folder):
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(folder / "entries.sqlite"))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
               "chunk TEXT, position INTEGER, scores TEXT, status TEXT NOT NULL, error TEXT, source TEXT)")
    return db


def index(records, run, config, index_dir, cache_dir, progress, cancel):
    import numpy as np
    if len(records) > 100000:
        raise ValueError("한 프로젝트에 최대 100,000개 이미지를 색인할 수 있습니다.")
    if not records:
        raise ValueError("색인할 이미지가 없습니다. 먼저 폴더를 가져오세요.")
    head = _read_head(run)
    signature = _signature(run)
    index_dir = Path(index_dir)
    folder = index_dir / signature
    db = _index_db(folder)
    device, offline = config.get("device", "auto"), config.get("offline", False)
    batch_size = max(1, min(16, int(config.get("batch_size", 1))))
    reused = errors = 0
    try:
        pending, ready_for_detector = [], []
        for record in records:
            _check(cancel)
            if not _source_valid(record):
                db.execute("INSERT OR REPLACE INTO entries VALUES(?,?,?,?,?,?,?,?)",
                           (str(record["id"]), _fingerprint(record), None, None, None, "error",
                            "원본 파일이 변경되었거나 없습니다. 다시 가져오거나 경로를 연결하세요.", json.dumps(record, ensure_ascii=False)))
                errors += 1
                continue
            saved = db.execute("SELECT * FROM entries WHERE id=?", (str(record["id"]),)).fetchone()
            files_ok = saved and saved["chunk"] and (folder / saved["chunk"]).is_file()
            if files_ok and saved["fingerprint"] == _fingerprint(record) and saved["status"] == "complete":
                reused += 1
            elif files_ok and saved["fingerprint"] == _fingerprint(record) and saved["status"] == "base":
                ready_for_detector.append(record)
            else:
                pending.append(record)
        if pending:
            _emit(progress, "index_model", .01, "검색용 원본 EmbeddingGemma 2를 불러옵니다.")
            model = _load_model(cache_dir, device, offline)
            chunk_size = 64
            for offset in range(0, len(pending), chunk_size):
                _check(cancel)
                # A bad source must not discard healthy images in its batch.
                batch, vectors = [], []
                for j, record in enumerate(pending[offset:offset + chunk_size]):
                    try:
                        vector = _encode(model, [record], 1, cancel)[0]
                        batch.append(record)
                        vectors.append(vector)
                        if j % 8 == 0:
                            _emit(progress, "index_base", .05 + .55 * (reused + offset + j + 1) / len(records),
                                  f"원본 검색 특징 추출 · {reused + offset + j + 1}/{len(records)}")
                    except Cancelled:
                        raise
                    except Exception as exc:
                        if _oom(exc):
                            raise
                        errors += 1
                        db.execute("INSERT OR REPLACE INTO entries VALUES(?,?,?,?,?,?,?,?)",
                                   (str(record["id"]), _fingerprint(record), None, None, None, "error", str(exc), json.dumps(record, ensure_ascii=False)))
                if batch:
                    chunk = uuid.uuid4().hex + ".npy"
                    temp = folder / (chunk + ".tmp")
                    with temp.open("wb") as stream:
                        np.save(stream, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
                    os.replace(temp, folder / chunk)
                    scores = probabilities(head, vectors) if run["mode"] == "classifier" else None
                    for pos, record in enumerate(batch):
                        score = json.dumps(scores[pos].tolist()) if scores is not None else None
                        db.execute("INSERT OR REPLACE INTO entries VALUES(?,?,?,?,?,?,?,?)",
                                   (str(record["id"]), _fingerprint(record), chunk, pos, score,
                                    "complete" if score else "base", None, json.dumps(record, ensure_ascii=False)))
                    if scores is None:
                        ready_for_detector.extend(batch)
                db.commit()
                done = reused + min(offset + chunk_size, len(pending))
                _atomic_json(index_dir / "checkpoint.json", {"run_id": run["id"], "signature": signature,
                              "completed_base": done, "total": len(records), "updated_at": datetime.now(timezone.utc).isoformat()})
                _emit(progress, "index_base", .05 + .55 * done / len(records), f"원본 검색 특징 추출 · {done}/{len(records)}")
        if ready_for_detector:
            # Discard the base model before loading the adapted detector.
            if "model" in locals():
                del model
            _clear_model()
            model = _load_model(cache_dir, device, True, Path(run["model_path"]) / "encoder")
            for offset, record in enumerate(ready_for_detector):
                _check(cancel)
                try:
                    score = probabilities(head, _encode(model, [record], 1, cancel))[0].tolist()
                    db.execute("UPDATE entries SET scores=?,status='complete',error=NULL WHERE id=?",
                               (json.dumps(score), str(record["id"])))
                except Cancelled:
                    raise
                except Exception as exc:
                    if _oom(exc):
                        raise
                    errors += 1
                    db.execute("UPDATE entries SET status='error',error=? WHERE id=?", (str(exc), str(record["id"])))
                db.commit()
                _emit(progress, "index_detector", .6 + .38 * (offset + 1) / len(ready_for_detector),
                      f"미세조정 모델 판별 · {offset + 1}/{len(ready_for_detector)}")
        db.commit()
        _check(cancel)
        current_ids = {str(r["id"]) for r in records}
        indexed = sum(row["id"] in current_ids and row["status"] == "complete"
                      for row in db.execute("SELECT id,status FROM entries"))
        manifest = {"schema": 1, "run_id": run["id"], "signature": signature,
                    "model_id": MODEL_ID, "base_revision": MODEL_REVISION, "preprocess": PREPROCESS_VERSION,
                    "dimension": DIM, "total": len(records), "indexed": indexed,
                    "errors": len(records) - indexed, "reused": reused,
                    "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed"}
        _atomic_json(index_dir / "manifest.json", manifest)
        (index_dir / "checkpoint.json").unlink(missing_ok=True)
        _emit(progress, "done", 1, f"색인 완료 · {indexed}개, 처리 불가 {len(records) - indexed}개")
        return manifest
    except Exception as exc:
        if _oom(exc):
            _clear_model()
            raise RuntimeError("색인 중 메모리가 부족합니다. 다른 앱을 닫고 다시 실행하면 완료한 배치를 재사용합니다.") from exc
        raise
    finally:
        db.close()


def _metadata_match(record, config):
    metadata = record.get("metadata") or {}
    sample = str(config.get("sample", "")).strip()
    if sample and str(metadata.get("sample", "")).casefold() != sample.casefold():
        return False
    date = str(metadata.get("acquired_at") or "")[:10]
    for key, compare in (("date_from", lambda d, bound: d >= bound), ("date_to", lambda d, bound: d <= bound)):
        bound = str(config.get(key) or "")
        if bound:
            try:
                datetime.strptime(bound, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError("날짜 필터는 YYYY-MM-DD 형식이어야 합니다.") from exc
            if not date or not compare(date, bound):
                return False
    return True


def search(records, run, config, index_dir, cache_dir):
    import numpy as np
    index_dir = Path(index_dir)
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("이 모델의 색인이 없습니다. 먼저 색인 만들기를 실행하세요.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = _signature(run)
    if manifest.get("signature") != signature or manifest.get("preprocess") != PREPROCESS_VERSION:
        raise ValueError("모델 또는 전처리 버전이 변경되었습니다. 색인을 다시 만드세요.")
    threshold = _threshold(config.get("threshold", run["threshold"]))
    label_filter = config.get("label", "ALL")
    if label_filter not in ("ALL", "SEM", "TEM"):
        raise ValueError("검색 라벨은 ALL, SEM 또는 TEM이어야 합니다.")
    query = str(config.get("query", "")).strip()
    if len(query) > 4000:
        raise ValueError("검색어는 4,000자 이하로 입력하세요.")
    latest = {str(r["id"]): r for r in records}
    folder = index_dir / signature
    if not (folder / "entries.sqlite").is_file():
        raise ValueError("색인 파일이 없습니다. 색인을 다시 만드세요.")
    db = sqlite3.connect((folder / "entries.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    candidates, excluded, stale, indexed, incomplete, seen = [], 0, 0, 0, 0, set()
    try:
        for row in db.execute("SELECT * FROM entries"):
            record = latest.get(row["id"])
            if record is None:
                continue
            seen.add(row["id"])
            if not _source_valid(record) or row["fingerprint"] != _fingerprint(record):
                stale += 1
                continue
            if row["status"] != "complete":
                # A saved base embedding or failed detector is not a model
                # rejection. It still needs indexing, even though its ID is
                # present in the checkpoint database.
                incomplete += 1
                continue
            if not row["chunk"] or not (folder / row["chunk"]).is_file():
                stale += 1
                continue
            indexed += 1
            scores = json.loads(row["scores"])
            label = accepted_label(scores, threshold, record.get("label"))
            # This gate precedes metadata matching and BOTH ranking channels.
            if not label:
                excluded += 1
                continue
            if label_filter != "ALL" and label != label_filter:
                continue
            if not _metadata_match(record, config):
                continue
            candidates.append({"record": record, "label": label, "score": float(max(scores[:2])),
                               "chunk": row["chunk"], "position": row["position"], "similarity": None})
    finally:
        db.close()
    if query and candidates:
        # Retrieval always uses the unchanged base model, never the detector.
        model = _load_model(cache_dir, config.get("device", "auto"), True)
        vector = np.asarray(model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False, prompt=""), dtype=np.float32)[0]
        if vector.shape != (DIM,) or not np.isfinite(vector).all():
            raise ValueError("검색어 임베딩 계산에 실패했습니다.")
        by_chunk = defaultdict(list)
        for i, candidate in enumerate(candidates):
            by_chunk[candidate["chunk"]].append(i)
        for chunk, positions in by_chunk.items():
            matrix = np.load(folder / chunk, mmap_mode="r", allow_pickle=False)
            for i in positions:
                candidates[i]["similarity"] = float(matrix[candidates[i]["position"]] @ vector)
            del matrix
        tokens = re.findall(r"[\w]+", query.casefold())
        lexical = {}
        for i, item in enumerate(candidates):
            record = item["record"]
            metadata = record.get("metadata") or {}
            searchable = " ".join(str(v) for v in (record.get("name", ""),
                         *(metadata.get(k, "") for k in ("sample", "acquired_at", "magnification", "voltage_kv", "description")))).casefold()
            count = sum(searchable.count(token) for token in tokens)
            if count:
                lexical[i] = count
        semantic_order = sorted(range(len(candidates)), key=lambda i: (-candidates[i]["similarity"], str(candidates[i]["record"]["id"])))
        lexical_order = sorted(lexical, key=lambda i: (-lexical[i], str(candidates[i]["record"]["id"])))
        rrf = defaultdict(float)
        for rank_list in (semantic_order, lexical_order):
            for rank, i in enumerate(rank_list, 1):
                rrf[i] += 1 / (60 + rank)
        candidates = [candidates[i] for i in sorted(rrf, key=lambda i: (-rrf[i], -candidates[i]["similarity"]))]
    else:
        candidates.sort(key=lambda item: (-item["score"], str(item["record"]["id"])))
    offset = max(0, int(config.get("offset", 0)))
    limit = max(1, min(100000, int(config.get("limit", 60))))
    items = []
    for item in candidates[offset:offset + limit]:
        record = item["record"]
        items.append({"id": record["id"], "path": record["path"],
                      "name": record.get("name") or Path(record["path"]).name,
                      "frame": record.get("frame", 0), "label": item["label"],
                      "score": item["score"], "similarity": item["similarity"],
                      "metadata": record.get("metadata") or {}, "thumbnail": record.get("thumbnail")})
    unindexed = len(latest) - len(seen) + incomplete
    return {"items": items, "total": len(candidates), "excluded": excluded, "indexed": indexed,
            "stale": stale, "unindexed": unindexed, "incomplete": incomplete,
            "threshold": threshold, "run_id": run["id"],
            "warning": "변경된 이미지나 색인이 완료되지 않은 이미지가 있습니다. 검색 색인을 다시 생성하세요." if stale or unindexed else None}
