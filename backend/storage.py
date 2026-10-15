"""Thread-safe local project catalog; originals are never modified."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

try:
    from .readers import SUPPORTED_EXTENSIONS, MissingReader, UnsupportedImage, iter_frames
except ImportError:
    from readers import SUPPORTED_EXTENSIONS, MissingReader, UnsupportedImage, iter_frames

LABELS = {"SEM", "TEM", "OTHER"}
METADATA_FIELDS = {"sample", "acquired_at", "magnification", "voltage_kv", "description"}


class ImportCancelled(RuntimeError):
    pass


def _directory(path):
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("실제로 존재하는 폴더를 선택해 주세요. 바로가기 폴더는 지원하지 않습니다.")
    return candidate.resolve()


def list_image_files(path, cancel=None):
    root = _directory(path)
    found = []
    def error(exc):
        if Path(exc.filename) == root:
            raise ValueError("선택한 폴더를 읽을 권한이 없습니다.") from exc
    for current, dirs, files in os.walk(root, followlinks=False, onerror=error):
        if cancel and cancel.is_set():
            raise ImportCancelled("가져오기를 취소했습니다. 이미 가져온 항목은 유지됩니다.")
        dirs[:] = sorted(d for d in dirs if d not in {".git", ".venv", "node_modules", "__pycache__"} and not (Path(current) / d).is_symlink())
        for filename in sorted(files):
            candidate = Path(current) / filename
            if candidate.suffix.lower() in SUPPORTED_EXTENSIONS and not candidate.is_symlink() and candidate.is_file():
                found.append(str(candidate))
    return found


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_image(image, path, format, **kwargs):
    import io
    out = io.BytesIO()
    image.save(out, format=format, **kwargs)
    _atomic_bytes(path, out.getvalue())


def _sha256(path, cancel=None):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            if cancel and cancel.is_set():
                raise ImportCancelled("가져오기를 취소했습니다. 이미 가져온 항목은 유지됩니다.")
            digest.update(chunk)
    # EMI references SER sidecars. Include their content in cache identity.
    if Path(path).suffix.lower() == ".emi":
        for sidecar in sorted(Path(path).parent.glob(Path(path).stem + "_*.ser")):
            if not sidecar.is_symlink():
                digest.update(_sha256(sidecar, cancel).encode("ascii"))
    return digest.hexdigest()


def _source_stamp(filename):
    source = Path(filename)
    stat = source.stat()
    stamp = {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}
    if source.suffix.lower() == ".emi":
        stamp["source_dependencies"] = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                         for p in sorted(source.parent.glob(source.stem + "_*.ser")) if not p.is_symlink()]
    return stamp


def _source_path(value):
    """Interpret saved source paths independently of the current host OS."""
    text = str(value)
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("\\\\", "//")):
        return PureWindowsPath(text)
    if os.name == "posix":
        # Normalizes macOS /var -> /private/var even if the leaf was moved.
        return PurePosixPath(str(Path(text).expanduser().resolve()))
    return PurePosixPath(text)


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.sqlite3"
        self.render_dir = self.root / "renders"
        self.thumb_dir = self.root / "thumbnails"
        self.run_dir = self.root / "run-metadata"
        for folder in (self.render_dir, self.thumb_dir, self.run_dir):
            folder.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, name TEXT NOT NULL,
                  sha256 TEXT NOT NULL, frame INTEGER NOT NULL DEFAULT 0,
                  frames INTEGER NOT NULL DEFAULT 1, label TEXT,
                  specimen_group TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                  render_path TEXT NOT NULL, thumbnail_path TEXT NOT NULL,
                  metadata TEXT NOT NULL DEFAULT '{}', imported_at TEXT NOT NULL,
                  UNIQUE(sha256, frame), CHECK(label IS NULL OR label IN ('SEM','TEM','OTHER')));
                CREATE INDEX IF NOT EXISTS images_label ON images(label);
                CREATE INDEX IF NOT EXISTS images_path ON images(path);
                CREATE TABLE IF NOT EXISTS source_roots (path TEXT PRIMARY KEY);
            """)

    @contextmanager
    def _connection(self):
        # Every method owns its connection: no thread-affine connection is shared.
        db = sqlite3.connect(str(self.db_path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def stats(self):
        with self._connection() as db:
            rows = db.execute("SELECT label, COUNT(*) n FROM images GROUP BY label").fetchall()
            counts = {"SEM": 0, "TEM": 0, "OTHER": 0, "UNLABELED": 0}
            for row in rows:
                counts[row["label"] or "UNLABELED"] = row["n"]
            groups = db.execute("SELECT COUNT(DISTINCT specimen_group) FROM images").fetchone()[0]
            roots = [r[0] for r in db.execute("SELECT path FROM source_roots ORDER BY path")]
        total = sum(counts.values())
        return {"root": str(self.root), "total": total, "counts": counts, "labeled": total - counts["UNLABELED"], "groups": groups, "roots": roots}

    def import_folder(self, path, frame_start=0, frame_end=0, progress=None, cancel=None):
        if type(frame_start) is not int or type(frame_end) is not int or not 0 <= frame_start <= frame_end:
            raise ValueError("프레임 범위는 0 이상의 정수이며 끝 값은 시작 값 이상이어야 합니다.")
        source_root = _directory(path)
        if progress:
            progress("discover", 0.0, "가져올 이미지 파일을 찾는 중입니다.")
        files = [p for p in list_image_files(source_root, cancel=cancel) if not any(Path(p).is_relative_to(x) for x in (self.render_dir, self.thumb_dir))]
        with self._connection() as db:
            db.execute("INSERT OR IGNORE INTO source_roots(path) VALUES(?)", (str(source_root),))
        imported = skipped = duplicates = corrupt = unsupported = replaced = 0
        errors = []
        for position, filename in enumerate(files):
            if cancel and cancel.is_set():
                raise ImportCancelled("가져오기를 취소했습니다. 이미 가져온 항목은 유지됩니다.")
            if progress:
                progress("import", position / max(1, len(files)), f"이미지 가져오는 중 {position + 1}/{len(files)} · {Path(filename).name}")
            try:
                source_stamp = _source_stamp(filename)
                digest = _sha256(filename, cancel)
                if source_stamp != _source_stamp(filename):
                    raise ValueError("가져오는 동안 원본 파일이 변경되었습니다. 다시 가져와 주세요.")
                with self._connection() as db:
                    # A revised source must never retain a previous manual label
                    # or be mistaken for the old indexed frame.
                    replaced += db.execute("DELETE FROM images WHERE path=? AND sha256<>?", (filename, digest)).rowcount
                    existing = {r[0] for r in db.execute("SELECT frame FROM images WHERE sha256=?", (digest,))}
                    # A touch/copy can alter mtime without changing content.
                    # A verified reimport refreshes integrity stamps so those
                    # records become indexable again, retaining valid labels.
                    for row in db.execute("SELECT id,metadata FROM images WHERE path=? AND sha256=?", (filename, digest)).fetchall():
                        metadata = json.loads(row["metadata"])
                        metadata.setdefault("provenance", {}).update(source_stamp)
                        db.execute("UPDATE images SET metadata=? WHERE id=?", (json.dumps(metadata, ensure_ascii=False), row["id"]))
                if all(f in existing for f in range(frame_start, frame_end + 1)):
                    skipped += 1
                    duplicates += 1
                    continue
                for item in iter_frames(filename, frame_start, frame_end):
                    if cancel and cancel.is_set():
                        raise ImportCancelled("가져오기를 취소했습니다. 이미 가져온 항목은 유지됩니다.")
                    if item["frame"] in existing:
                        duplicates += 1
                        skipped += 1
                        continue
                    key = f"{digest}-f{item['frame']}"
                    render = self.render_dir / (key + ".png")
                    thumbnail = self.thumb_dir / (key + ".jpg")
                    _save_image(item["image"], render, "PNG")
                    thumb = item["image"].copy()
                    thumb.thumbnail((256, 256))
                    _save_image(thumb, thumbnail, "JPEG", quality=82)
                    metadata = item["metadata"]
                    if source_stamp != _source_stamp(filename):
                        raise ValueError("가져오는 동안 원본 파일이 변경되었습니다. 다시 가져와 주세요.")
                    provenance = metadata.setdefault("provenance", {})
                    provenance.update({"source_sha256": digest, "source_path": filename, **source_stamp})
                    for field in METADATA_FIELDS:
                        metadata.setdefault(field, "")
                    with self._connection() as db:
                        inserted = db.execute("""INSERT OR IGNORE INTO images(path,name,sha256,frame,frames,specimen_group,width,height,render_path,thumbnail_path,metadata,imported_at)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (filename, Path(filename).name, digest, item["frame"], item["frames"], str(Path(filename).parent),
                            item["width"], item["height"], str(render), str(thumbnail), json.dumps(metadata, ensure_ascii=False, allow_nan=False), datetime.now(timezone.utc).isoformat())).rowcount
                    imported += inserted
                    if not inserted:
                        duplicates += 1
                        skipped += 1
            except ImportCancelled:
                raise
            except MissingReader as exc:
                unsupported += 1
                skipped += 1
                if len(errors) < 30:
                    errors.append({"path": filename, "reason": str(exc), "type": "dependency"})
            except UnsupportedImage as exc:
                unsupported += 1
                skipped += 1
                if len(errors) < 30:
                    errors.append({"path": filename, "reason": str(exc), "type": "unsupported"})
            except Exception as exc:
                corrupt += 1
                skipped += 1
                if len(errors) < 30:
                    errors.append({"path": filename, "reason": "파일을 읽을 수 없습니다: " + str(exc)[:250], "type": "corrupt"})
        if progress:
            progress("import", 1.0, f"가져오기 완료 · {imported}개 추가, {skipped}개 건너뜀")
        return {"imported": imported, "skipped": skipped, "total": self.stats()["total"], "files": len(files),
                "duplicates": duplicates, "corrupt": corrupt, "unsupported": unsupported, "replaced": replaced, "errors": errors}

    def _record(self, row, thumbnail=False):
        value = dict(row)
        value["group"] = value.pop("specimen_group")
        value["metadata"] = json.loads(value["metadata"])
        value["missing"] = not Path(value["path"]).is_file()
        # Cache copies belong to the project and remain portable after copying
        # or moving the project directory between computers/platforms.
        value["render_path"] = str(self.render_dir / re.split(r"[\\/]", value["render_path"])[-1])
        thumb_path = self.thumb_dir / re.split(r"[\\/]", value.pop("thumbnail_path"))[-1]
        if thumbnail:
            try:
                value["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(Path(thumb_path).read_bytes()).decode("ascii") if not value["missing"] else None
            except OSError:
                value["thumbnail"] = None
        return value

    def images(self, filter="ALL", query="", offset=0, limit=60):
        if filter not in LABELS | {"ALL", "UNLABELED"}:
            raise ValueError("이미지 필터가 올바르지 않습니다.")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("페이지 범위가 올바르지 않습니다. 한 번에 1~500개를 조회할 수 있습니다.")
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("검색어는 1000자 이하의 문자열이어야 합니다.")
        clauses, values = [], []
        if filter == "UNLABELED":
            clauses.append("label IS NULL")
        elif filter in LABELS:
            clauses.append("label=?")
            values.append(filter)
        if query:
            literal = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            clauses.append("(name LIKE ? ESCAPE '\\' OR path LIKE ? ESCAPE '\\' OR specimen_group LIKE ? ESCAPE '\\' OR json_extract(metadata,'$.sample') LIKE ? ESCAPE '\\' OR json_extract(metadata,'$.description') LIKE ? ESCAPE '\\')")
            values.extend([literal] * 5)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connection() as db:
            total = db.execute("SELECT COUNT(*) FROM images" + where, values).fetchone()[0]
            rows = db.execute("SELECT * FROM images" + where + " ORDER BY id DESC LIMIT ? OFFSET ?", values + [limit, offset]).fetchall()
        return {"items": [self._record(r, True) for r in rows], "total": total}

    def get_images(self, ids):
        """Fetch thumbnails only for a result page, never for the entire index."""
        if not isinstance(ids, (list, tuple)) or len(ids) > 200 or any(type(x) is not int or x <= 0 for x in ids):
            raise ValueError("썸네일은 올바른 이미지 ID로 최대 200개씩 조회해 주세요.")
        if not ids:
            return []
        with self._connection() as db:
            rows = db.execute("SELECT * FROM images WHERE id IN (" + ",".join("?" * len(ids)) + ")", ids).fetchall()
        by_id = {row["id"]: self._record(row, True) for row in rows}
        return [by_id[id_] for id_ in dict.fromkeys(ids) if id_ in by_id]

    def thumbnail(self, id_):
        rows = self.get_images([id_])
        return rows[0].get("thumbnail") if rows else None

    def _ids(self, ids, db):
        if not isinstance(ids, (list, tuple)) or any(type(x) is not int or x <= 0 for x in ids):
            raise ValueError("이미지 ID는 양의 정수 목록이어야 합니다.")
        ids = list(dict.fromkeys(ids))
        found = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            found.update(r[0] for r in db.execute("SELECT id FROM images WHERE id IN (" + ",".join("?" * len(chunk)) + ")", chunk))
        if len(found) != len(ids):
            raise ValueError("선택한 이미지 중 프로젝트에 없는 항목이 있습니다. 목록을 새로 고쳐 주세요.")
        return ids

    def label(self, ids, label, group=None):
        if label is not None and (not isinstance(label, str) or label not in LABELS):
            raise ValueError("레이블은 SEM, TEM, OTHER 또는 미분류여야 합니다.")
        if group is not None and (not isinstance(group, str) or not group.strip() or len(group) > 2000):
            raise ValueError("시료 그룹은 1~2000자의 이름이어야 합니다.")
        with self._lock, self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ids = self._ids(ids, db)
            if group is None:
                db.executemany("UPDATE images SET label=? WHERE id=?", [(label, id_) for id_ in ids])
            else:
                db.executemany("UPDATE images SET label=?,specimen_group=? WHERE id=?", [(label, group.strip(), id_) for id_ in ids])
        return {"updated": len(ids)}

    def update_metadata(self, ids, metadata):
        if not isinstance(metadata, dict) or set(metadata) - METADATA_FIELDS:
            raise ValueError("수정 가능한 메타데이터 항목이 아닙니다.")
        for key, value in metadata.items():
            if value is not None and (not isinstance(value, (str, int, float)) or isinstance(value, bool) or len(str(value)) > 8000):
                raise ValueError("메타데이터 값은 8000자 이하의 문자 또는 숫자여야 합니다.")
            if key == "acquired_at" and value:
                try:
                    datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("촬영일은 YYYY-MM-DD 형식으로 입력해 주세요.") from exc
        json.dumps(metadata, allow_nan=False)
        with self._lock, self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ids = self._ids(ids, db)
            for id_ in ids:
                current = json.loads(db.execute("SELECT metadata FROM images WHERE id=?", (id_,)).fetchone()[0])
                current.update({k: "" if v is None else v for k, v in metadata.items()})
                current["edited_at"] = datetime.now(timezone.utc).isoformat()
                db.execute("UPDATE images SET metadata=? WHERE id=?", (json.dumps(current, ensure_ascii=False, allow_nan=False), id_))
        return {"updated": len(ids)}

    def all_records(self):
        with self._connection() as db:
            return [self._record(row) for row in db.execute("SELECT * FROM images ORDER BY id")]

    def training_records(self):
        with self._connection() as db:
            return [self._record(row) for row in db.execute("SELECT * FROM images WHERE label IS NOT NULL ORDER BY id")]

    def relink(self, old_root, new_root):
        old = _source_path(old_root)
        new = _directory(new_root)
        replacements = []
        verified_sources = {}
        with self._lock, self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT id,path,specimen_group,sha256,metadata FROM images"):
                path = _source_path(row["path"])
                if not path.is_relative_to(old):
                    continue
                relative = path.relative_to(old)
                if ".." in relative.parts:
                    continue
                candidate = new.joinpath(*relative.parts)
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                candidate_key = str(candidate)
                if candidate_key not in verified_sources:
                    verified_sources[candidate_key] = _sha256(candidate)
                if verified_sources[candidate_key] != row["sha256"]:
                    continue
                group = row["specimen_group"]
                if _source_path(group) == path.parent:
                    group = str(candidate.parent)
                metadata = json.loads(row["metadata"])
                new_stat = candidate.stat()
                provenance = metadata.setdefault("provenance", {})
                provenance.update({"current_source_path": str(candidate), "source_size": new_stat.st_size, "source_mtime_ns": new_stat.st_mtime_ns})
                for dependency in provenance.get("source_dependencies", []):
                    dependency_path = _source_path(dependency["path"])
                    if dependency_path.is_relative_to(old):
                        dependency_path = new.joinpath(*dependency_path.relative_to(old).parts)
                        dependency_stat = dependency_path.stat()
                        dependency.update({"path": str(dependency_path), "size": dependency_stat.st_size, "mtime_ns": dependency_stat.st_mtime_ns})
                replacements.append((str(candidate), group, json.dumps(metadata, ensure_ascii=False), row["id"]))
            db.executemany("UPDATE images SET path=?,specimen_group=?,metadata=? WHERE id=?", replacements)
            if replacements:
                # Keep old roots while some missing/unmatched records still refer to them.
                remaining = any(_source_path(r[0]).is_relative_to(old) for r in db.execute("SELECT path FROM images"))
                if not remaining:
                    for row in db.execute("SELECT path FROM source_roots").fetchall():
                        if _source_path(row[0]) == old:
                            db.execute("DELETE FROM source_roots WHERE path=?", (row[0],))
                db.execute("INSERT OR IGNORE INTO source_roots(path) VALUES(?)", (str(new),))
        return {"updated": len(replacements)}

    def save_run(self, run):
        if not isinstance(run, dict) or not isinstance(run.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run["id"]):
            raise ValueError("학습 실행 ID가 올바르지 않습니다.")
        run = dict(run)
        if isinstance(run.get("model_path"), str):
            model_path = Path(run["model_path"]).resolve()
            if model_path.is_relative_to(self.root):
                run["model_relative_path"] = model_path.relative_to(self.root).as_posix()
        encoded = json.dumps(run, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        with self._lock:
            _atomic_bytes(self.run_dir / (run["id"] + ".json"), encoded)

    def runs(self):
        result = []
        with self._lock:
            for path in self.run_dir.glob("*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(item, dict) and isinstance(item.get("id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", item["id"]):
                        relative = item.get("model_relative_path")
                        if isinstance(relative, str):
                            candidate = (self.root / relative).resolve()
                            if candidate.is_relative_to(self.root):
                                item["model_path"] = str(candidate)
                        else:
                            # Compatibility with runs saved before relative paths.
                            parts = re.split(r"[\\/]", str(item.get("model_path", "")))
                            if len(parts) >= 2 and parts[-2:] == ["runs", item["id"]]:
                                item["model_path"] = str(self.root / "runs" / item["id"])
                        result.append(item)
                except (OSError, ValueError):
                    continue
        return sorted(result, key=lambda r: (str(r.get("created_at", "")), r["id"]), reverse=True)
