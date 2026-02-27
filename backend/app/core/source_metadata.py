from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings


REQUIRED_FIELDS = ("source", "product_id", "valid_time", "ingested_at")
TS_FMT = "%Y-%m-%d_%H"


@dataclass(frozen=True)
class SourceMetaCheck:
    timestamp: str
    ok: bool
    missing_fields: list[str]
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "ok": self.ok,
            "missing_fields": self.missing_fields,
            "metadata": self.metadata,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_to_valid_time(timestamp: str) -> str:
    dt = datetime.strptime(timestamp, TS_FMT).replace(tzinfo=timezone.utc)
    return dt.isoformat()


def build_source_metadata(
    *,
    source: str,
    product_id: str,
    valid_time: str,
    ingested_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": str(source).strip(),
        "product_id": str(product_id).strip(),
        "valid_time": str(valid_time).strip(),
        "ingested_at": str(ingested_at).strip() if ingested_at is not None else utc_now_iso(),
    }
    if extra:
        out.update(extra)
    return out


def ensure_required_source_metadata(
    *,
    metadata: dict[str, Any] | None,
    timestamp: str,
    fallback_source: str = "unknown",
    fallback_product_id: str = "unknown",
) -> dict[str, Any]:
    raw = dict(metadata or {})
    source = str(raw.get("source", "")).strip() or fallback_source
    product_id = str(raw.get("product_id", "")).strip() or fallback_product_id
    valid_time = str(raw.get("valid_time", "")).strip() or timestamp_to_valid_time(timestamp)
    ingested_at = str(raw.get("ingested_at", "")).strip() or utc_now_iso()
    raw.update(
        {
            "source": source,
            "product_id": product_id,
            "valid_time": valid_time,
            "ingested_at": ingested_at,
        }
    )
    return raw


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_source_meta(annotation_dir: Path) -> dict[str, Any]:
    meta = _read_json(annotation_dir / "meta.json")
    source = meta.get("source")
    if isinstance(source, dict):
        return source
    latest_meta = _read_json(annotation_dir / "latest_meta.json")
    if latest_meta:
        return latest_meta
    return {}


def read_source_metadata(annotation_dir: Path) -> dict[str, Any]:
    return _extract_source_meta(annotation_dir)


def validate_source_metadata(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [k for k in REQUIRED_FIELDS if not str(metadata.get(k, "")).strip()]
    return len(missing) == 0, missing


def build_source_metadata_report(
    *,
    settings: Settings,
    sample_limit: int = 200,
) -> dict[str, Any]:
    root = settings.annotation_pack_root
    timestamps: list[str] = []
    if root.exists():
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and len(folder.name) == 13 and folder.name[4] == "-" and folder.name[7] == "-" and folder.name[10] == "_":
                timestamps.append(folder.name)

    if len(timestamps) > sample_limit:
        # Keep deterministic spread while avoiding full heavy scan.
        step = max(1, len(timestamps) // sample_limit)
        timestamps = timestamps[::step][:sample_limit]

    checks: list[SourceMetaCheck] = []
    for ts in timestamps:
        ann = root / ts
        meta = _extract_source_meta(ann)
        ok, missing = validate_source_metadata(meta)
        checks.append(SourceMetaCheck(timestamp=ts, ok=ok, missing_fields=missing, metadata=meta))

    ok_count = sum(1 for c in checks if c.ok)
    total = len(checks)
    coverage = float(ok_count / total) if total > 0 else 0.0
    status = "pass"
    if total == 0:
        status = "fail"
    elif coverage < 1.0:
        # Missing source metadata is warning-only by design (non-blocking).
        status = "warn"

    by_source: dict[str, int] = {}
    by_product: dict[str, int] = {}
    for c in checks:
        src = str(c.metadata.get("source", "")).strip() or "unknown"
        pid = str(c.metadata.get("product_id", "")).strip() or "unknown"
        by_source[src] = by_source.get(src, 0) + 1
        by_product[pid] = by_product.get(pid, 0) + 1

    return {
        "summary": {
            "status": status,
            "sampled_count": total,
            "ok_count": ok_count,
            "missing_count": total - ok_count,
            "coverage": round(coverage, 6),
        },
        "distribution": {
            "by_source": dict(sorted(by_source.items(), key=lambda kv: kv[0])),
            "by_product_id": dict(sorted(by_product.items(), key=lambda kv: kv[0])),
        },
        "checks": [c.to_json() for c in checks[:300]],
        "generated_at": utc_now_iso(),
    }
