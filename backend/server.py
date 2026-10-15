"""JSONL desktop worker. No network server; stdout is protocol only."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

from storage import Store
import training

OUTPUT = sys.stdout
WRITE_LOCK = threading.Lock()


def emit(value):
    # ML libraries occasionally print to stdout. The main function redirects it.
    with WRITE_LOCK:
        OUTPUT.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        OUTPUT.flush()


def bounded_float(value, minimum, maximum, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 값을 확인해주세요.")
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name}: {minimum}~{maximum} 범위로 입력해주세요.")
    return number


class Worker:
    def __init__(self, data_dir, cache_dir):
        self.root = Path(data_dir).resolve()
        self.cache = Path(cache_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root)
        self.job = None
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None

    def active(self):
        return bool(self.job and self.job["status"] == "running")

    def require_idle(self):
        if self.active():
            raise ValueError("진행 중인 작업을 완료하거나 취소한 뒤 다시 시도하세요.")

    def progress(self, phase, value, message):
        with self.lock:
            if not self.job:
                return
            self.job.update(phase=str(phase), progress=max(0, min(1, float(value))), message=str(message))
            if not self.job["logs"] or self.job["logs"][-1] != str(message):
                self.job["logs"] = (self.job["logs"] + [str(message)])[-150:]
            event = copy.deepcopy(self.job)
        emit({"event": {"type": "job", "job": event}})

    def start(self, phase, function):
        self.require_idle()
        self.cancel = threading.Event()
        with self.lock:
            self.job = {"id": uuid.uuid4().hex, "status": "running", "kind": phase, "phase": phase, "progress": 0, "message": "작업을 준비합니다.", "logs": []}
            response = copy.deepcopy(self.job)

        def target():
            try:
                result = function()
                if self.cancel.is_set():
                    raise training.Cancelled("작업을 취소했습니다.")
                with self.lock:
                    self.job.update(status="completed", progress=1, result=result, message="작업이 완료되었습니다.")
            except Exception as error:
                cancelled = self.cancel.is_set() or isinstance(error, training.Cancelled)
                with self.lock:
                    self.job.update(status="cancelled" if cancelled else "failed", error=str(error), message="작업을 취소했습니다." if cancelled else str(error))
                if not cancelled:
                    traceback.print_exc(file=sys.stderr)
            with self.lock:
                event = copy.deepcopy(self.job)
            emit({"event": {"type": "job", "job": event}})

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        return {"job": response}

    def run(self, run_id):
        for run in self.store.runs():
            if run["id"] == run_id:
                if run.get("status") != "completed":
                    raise ValueError("완료된 모델을 선택해주세요.")
                return run
        raise ValueError("학습 모델을 찾을 수 없습니다. 학습을 완료한 뒤 모델을 선택해주세요.")

    def index_dir(self, run):
        # IDs come from persisted model records, never arbitrary user paths.
        return self.root / "indexes" / str(run["id"])

    def handle(self, action, payload):
        if not isinstance(payload, dict):
            raise ValueError("요청 형식이 올바르지 않습니다.")
        if action == "status":
            environment = training.environment(self.cache)
            with self.lock:
                job = copy.deepcopy(self.job)
            return {"environment": environment, "project": self.store.stats(), "runs": self.store.runs(), "job": job, "indexes": self.indexes()}
        if action == "images":
            return self.store.images(filter=payload.get("filter", "ALL"), query=payload.get("query", ""), offset=int(payload.get("offset", 0)), limit=min(200, max(1, int(payload.get("limit", 60)))))
        if action == "runs":
            return {"runs": self.store.runs()}
        if action == "cancel":
            self.cancel.set()
            return {"cancelled": True}
        if action == "known_path":
            path = str(Path(payload["path"]).resolve())
            return {"known": any(str(Path(item["path"]).resolve()) == path for item in self.store.all_records())}
        if action == "project":
            self.require_idle()
            root = Path(payload["path"]).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self.store = Store(root)
            self.root = root
            self.job = None
            return {"project": self.store.stats()}
        if action == "import":
            source = Path(payload["path"]).expanduser().absolute()
            if source.is_symlink() or not source.is_dir():
                raise ValueError("읽을 수 있는 이미지 폴더를 선택해주세요.")
            first = int(payload.get("frame_start", 0))
            last = int(payload.get("frame_end", first))
            if first < 0 or last < first or last - first > 100000:
                raise ValueError("프레임 범위를 확인해주세요. 0부터 시작하며 끝은 시작 이상이어야 합니다.")
            return self.start("import", lambda: self.store.import_folder(source, first, last, self.progress, self.cancel))
        if action == "label":
            self.require_idle()
            return self.store.label(payload.get("ids", []), payload.get("label"), payload.get("group"))
        if action == "remove_images":
            self.require_idle()
            return self.store.remove_images(payload.get("ids", []))
        if action == "restore_images":
            self.require_idle()
            return self.store.restore_images(payload.get("ids", []))
        if action == "metadata":
            self.require_idle()
            return self.store.update_metadata(payload.get("ids", []), payload.get("metadata", {}))
        if action == "import_metadata":
            self.require_idle()
            source = Path(payload["path"]).resolve()
            if source.suffix.lower() != ".csv" or source.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("32MB 이하의 CSV 파일을 선택해주세요.")
            records = self.store.all_records()
            lookup = {(os.path.normcase(os.path.abspath(item["path"])), int(item.get("frame", 0))): item["id"] for item in records}
            updated = skipped = 0
            errors = []
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "path" not in reader.fieldnames:
                    raise ValueError("CSV에 path 열이 필요합니다. frame은 생략하면 0입니다.")
                for line, row in enumerate(reader, 2):
                    try:
                        target = Path(row["path"])
                        if not target.is_absolute():
                            target = source.parent / target
                        key = (os.path.normcase(os.path.abspath(target)), int(row.get("frame") or 0))
                        record_id = lookup.get(key)
                        if record_id is None:
                            raise ValueError("등록된 이미지 경로/프레임과 일치하지 않습니다.")
                        fields = {k: row[k] for k in ("sample", "acquired_at", "magnification", "voltage_kv", "description") if row.get(k) is not None}
                        self.store.update_metadata([record_id], fields)
                        updated += 1
                    except (ValueError, KeyError, TypeError) as error:
                        skipped += 1
                        if len(errors) < 30:
                            errors.append({"line": line, "reason": str(error)})
            return {"updated": updated, "skipped": skipped, "errors": errors}
        if action == "relink":
            self.require_idle()
            return self.store.relink(payload["old_root"], payload["new_root"])
        if action == "train":
            config = self.training_config(payload)
            records = self.store.training_records()
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            output = self.root / "runs" / run_id

            def train():
                try:
                    run = training.train(records, config, output, self.cache, self.progress, self.cancel)
                    self.store.save_run(run)
                    return run
                except Exception as error:
                    self.store.save_run({"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                        "mode": config["mode"], "status": "cancelled" if self.cancel.is_set() else "failed",
                        "metrics": {}, "threshold": config["threshold"], "config": config,
                        "model_path": str(output), "error": str(error), "logs": self.job.get("logs", [])[-100:]})
                    raise

            return self.start("train", train)
        if action in ("index", "scan"):
            run = self.run(payload.get("run_id"))
            config = dict(payload)
            config["threshold"] = bounded_float(payload.get("threshold", run.get("threshold", .95)), 0, 1, "통과 기준")
            if config.get("device", "auto") not in ("auto", "cpu", "mps", "cuda"):
                raise ValueError("학습 장치가 올바르지 않습니다.")
            records = self.store.all_records()
            return self.start("index", lambda: training.index(records, run, config, self.index_dir(run), self.cache, self.progress, self.cancel))
        if action == "search":
            self.require_idle()
            run = self.run(payload.get("run_id"))
            config = dict(payload)
            config["threshold"] = bounded_float(payload.get("threshold", run.get("threshold", .95)), 0, 1, "통과 기준")
            config["limit"] = min(100000 if payload.get("export") else 200, max(1, int(payload.get("limit", 60))))
            config["offset"] = max(0, int(payload.get("offset", 0)))
            records = self.store.all_records()
            result = training.search(records, run, config, self.index_dir(run), self.cache)
            lookup = {str(record["id"]): record for record in records}
            for item in result["items"]:
                record = lookup.get(str(item["id"]))
                # Search module uses current annotations too. This assertion is a second barrier.
                if record is None or record.get("label") == "OTHER":
                    raise RuntimeError("검색 제외 필터가 일치하지 않습니다. 인덱스를 다시 생성하세요.")
                if not payload.get("export"):
                    item["thumbnail"] = self.store.thumbnail(item["id"])
                item.setdefault("frame", record.get("frame", 0))
            return result
        if action == "export_model":
            self.require_idle()
            run = self.run(payload["run_id"])
            source = Path(run["model_path"]).resolve()
            # Export whole run including classifier, metadata, and any partial fine-tuned model.
            if source.parent.name == "runs":
                bundle = source
            else:
                bundle = source
                while bundle.parent != bundle and bundle.parent.name != "runs":
                    bundle = bundle.parent
                if bundle.parent.name != "runs":
                    raise ValueError("모델 저장 경로를 확인할 수 없습니다.")
            target = Path(payload["path"]).resolve() / ("gemmascope-" + str(run["id"]))
            if target.exists():
                raise ValueError("같은 이름의 모델 폴더가 이미 있습니다. 다른 내보내기 위치를 선택해주세요.")
            if target == bundle or bundle in target.parents:
                raise ValueError("모델 폴더 밖의 위치를 선택해주세요.")
            shutil.copytree(bundle, target)
            return {"path": str(target)}
        raise ValueError("지원하지 않는 작업입니다: " + str(action))

    def indexes(self):
        directory = self.root / "indexes"
        if not directory.exists():
            return []
        return [{"run_id": entry.name, "path": str(entry)} for entry in directory.iterdir() if entry.is_dir()]

    @staticmethod
    def training_config(payload):
        config = dict(payload)
        if config.get("mode", "classifier") not in ("classifier", "finetune"):
            raise ValueError("학습 모드를 확인해주세요.")
        config["mode"] = config.get("mode", "classifier")
        config["epochs"] = int(bounded_float(payload.get("epochs", 3), 1, 100, "에포크"))
        config["batch_size"] = int(bounded_float(payload.get("batch_size", 1), 1, 64, "배치 크기"))
        config["gradient_accumulation"] = int(bounded_float(payload.get("gradient_accumulation", 8), 1, 128, "누적 배치"))
        config["learning_rate"] = bounded_float(payload.get("learning_rate", 2e-5), 1e-8, .01, "학습률")
        config["threshold"] = bounded_float(payload.get("threshold", .95), 0, 1, "통과 기준")
        if config.get("device", "auto") not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("학습 장치를 확인해주세요.")
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    options = parser.parse_args()
    worker = Worker(options.data_dir, options.cache_dir)
    # Protect JSONL stdout against library progress/debug prints, including job threads.
    sys.stdout = sys.stderr
    for line in sys.stdin:
        request_id = None
        try:
            if len(line) > 4 * 1024 * 1024:
                raise ValueError("요청이 너무 큽니다.")
            request = json.loads(line)
            request_id = request.get("id")
            result = worker.handle(request.get("action"), request.get("payload", {}))
            emit({"id": request_id, "result": result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({"id": request_id, "error": str(error)})
    worker.cancel.set()


if __name__ == "__main__":
    main()
