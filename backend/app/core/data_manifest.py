from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TIMESTAMP_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}$"),
)


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    size: int
    mtime_ns: int
    hash: str
    timestamp: str
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "hash": self.hash,
            "timestamp": self.timestamp,
            "source": self.source,
        }


def _iter_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file()], key=lambda p: p.as_posix())


def _should_skip(relative_path: str) -> bool:
    rel = relative_path.replace("\\", "/")
    # Avoid self-referential churn from generated manifest artifacts.
    if rel.startswith("processed/manifest"):
        return True
    return False


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_timestamp(relative_path: str) -> str:
    parts = relative_path.replace("\\", "/").split("/")
    stem = Path(relative_path).stem
    candidates = list(parts) + [stem]
    for token in candidates:
        for pattern in TIMESTAMP_PATTERNS:
            if pattern.match(token):
                return token
    return ""


def _infer_source(relative_path: str) -> str:
    parts = relative_path.replace("\\", "/").split("/")
    if not parts:
        return "unknown"
    if parts[0] == "ais_heatmap":
        return "ais_heatmap"
    if parts[0] == "raw":
        return "raw"
    if parts[0] == "processed":
        if len(parts) > 1:
            return f"processed/{parts[1]}"
        return "processed"
    return parts[0]


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("path", "")).replace("\\", "/")
            if key:
                rows[key] = row
    return rows


def build_data_manifest(
    *,
    data_root: Path,
    manifest_path: Path,
    state_path: Path,
    full_scan: bool = False,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    manifest_path = manifest_path.resolve()
    state_path = state_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    prev_entries: dict[str, dict[str, Any]] = {}
    if (not full_scan) and state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            prev_entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            prev_entries = {}

    file_paths = _iter_files(data_root)
    records: list[ManifestRecord] = []
    hashed_files = 0
    reused_hash_files = 0

    for file_path in file_paths:
        if file_path == manifest_path or file_path == state_path:
            continue
        rel = file_path.relative_to(data_root).as_posix()
        if _should_skip(rel):
            continue
        stat = file_path.stat()
        prev = prev_entries.get(rel)

        hash_value = ""
        if (
            (not full_scan)
            and isinstance(prev, dict)
            and int(prev.get("size", -1)) == int(stat.st_size)
            and int(prev.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            and isinstance(prev.get("hash"), str)
        ):
            hash_value = str(prev["hash"])
            reused_hash_files += 1
        else:
            hash_value = _sha256_file(file_path)
            hashed_files += 1

        record = ManifestRecord(
            path=rel,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            hash=hash_value,
            timestamp=_extract_timestamp(rel),
            source=_infer_source(rel),
        )
        records.append(record)

    records.sort(key=lambda r: r.path)
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False))
            f.write("\n")

    next_entries = {
        r.path: {
            "size": r.size,
            "mtime_ns": r.mtime_ns,
            "hash": r.hash,
            "timestamp": r.timestamp,
            "source": r.source,
        }
        for r in records
    }
    state_payload = {
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": next_entries,
    }
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    prev_keys = set(prev_entries.keys())
    next_keys = set(next_entries.keys())
    removed_paths = sorted(prev_keys - next_keys)

    return {
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "state_path": str(state_path),
        "total_files": len(records),
        "hashed_files": hashed_files,
        "reused_hash_files": reused_hash_files,
        "removed_files": len(removed_paths),
        "removed_examples": removed_paths[:10],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def diff_manifests(*, old_manifest: Path, new_manifest: Path) -> dict[str, Any]:
    old_rows = load_manifest(old_manifest)
    new_rows = load_manifest(new_manifest)

    old_keys = set(old_rows.keys())
    new_keys = set(new_rows.keys())
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)

    changed: list[dict[str, Any]] = []
    for key in sorted(old_keys & new_keys):
        old_item = old_rows[key]
        new_item = new_rows[key]
        if str(old_item.get("hash", "")) != str(new_item.get("hash", "")):
            changed.append(
                {
                    "path": key,
                    "old_hash": old_item.get("hash", ""),
                    "new_hash": new_item.get("hash", ""),
                    "old_size": old_item.get("size"),
                    "new_size": new_item.get("size"),
                }
            )

    return {
        "old_manifest": str(old_manifest),
        "new_manifest": str(new_manifest),
        "old_count": len(old_rows),
        "new_count": len(new_rows),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added": added,
        "removed": removed,
        "changed": changed,
    }
