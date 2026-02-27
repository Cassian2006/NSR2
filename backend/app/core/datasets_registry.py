from __future__ import annotations

import hashlib
from math import ceil
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.source_metadata import read_source_metadata, validate_source_metadata


def _scan_timestamps(annotation_pack_root: Path) -> list[str]:
    out: list[str] = []
    if not annotation_pack_root.exists():
        return out
    for folder in sorted(annotation_pack_root.iterdir()):
        if not folder.is_dir():
            continue
        ts = folder.name
        if len(ts) == 13 and ts[4] == "-" and ts[7] == "-" and ts[10] == "_":
            out.append(ts)
    return out


def _has_ais_for_timestamp(ais_root: Path, timestamp: str) -> bool:
    for _ in ais_root.rglob(f"{timestamp}.npy"):
        return True
    return False


def _build_registry_row(settings: Settings, timestamp: str) -> dict[str, Any]:
    ann = settings.annotation_pack_root / timestamp
    source_meta = read_source_metadata(ann)
    source_ok, source_missing = validate_source_metadata(source_meta)

    has_x_stack = (ann / "x_stack.npy").exists()
    has_blocked_mask = (ann / "blocked_mask.npy").exists()
    has_meta = (ann / "meta.json").exists()
    has_ais_heatmap = _has_ais_for_timestamp(settings.ais_heatmap_root, timestamp)
    has_unet_pred = (settings.pred_root / "unet_v1" / f"{timestamp}.npy").exists()
    has_unet_uncertainty = (settings.pred_root / "unet_v1" / f"{timestamp}_uncertainty.npy").exists()

    required_checks = {
        "x_stack": has_x_stack,
        "blocked_mask": has_blocked_mask,
        "source_metadata": source_ok,
        "ais_heatmap": has_ais_heatmap,
    }
    score = sum(1 for v in required_checks.values() if v) / float(len(required_checks))
    is_complete = all(required_checks.values())

    return {
        "timestamp": timestamp,
        "month": timestamp[:7],
        "source": str(source_meta.get("source", "")).strip() or "unknown",
        "product_id": str(source_meta.get("product_id", "")).strip() or "unknown",
        "valid_time": str(source_meta.get("valid_time", "")).strip(),
        "ingested_at": str(source_meta.get("ingested_at", "")).strip(),
        "source_metadata_ok": source_ok,
        "source_metadata_missing_fields": source_missing,
        "has_x_stack": has_x_stack,
        "has_blocked_mask": has_blocked_mask,
        "has_meta": has_meta,
        "has_ais_heatmap": has_ais_heatmap,
        "has_unet_pred": has_unet_pred,
        "has_unet_uncertainty": has_unet_uncertainty,
        "completeness_score": round(score, 6),
        "is_complete": is_complete,
    }


def _rates(counter: dict[str, int], total: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key, count in sorted(counter.items(), key=lambda kv: kv[0]):
        out[key] = {"count": int(count), "rate": round(float(count / total), 6) if total > 0 else 0.0}
    return out


def _dataset_version(items: list[dict[str, Any]]) -> str:
    if not items:
        return "ds-empty"
    parts: list[str] = []
    for row in items:
        parts.append(
            "|".join(
                [
                    str(row.get("timestamp", "")),
                    str(row.get("source", "")),
                    str(row.get("product_id", "")),
                    "1" if bool(row.get("is_complete", False)) else "0",
                ]
            )
        )
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"ds-{len(items)}-{digest}"


def build_datasets_registry(
    *,
    settings: Settings,
    month: str | None = None,
    source: str | None = None,
    is_complete: bool | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    page = max(1, int(page))
    page_size = min(200, max(1, int(page_size)))

    rows = [_build_registry_row(settings, ts) for ts in _scan_timestamps(settings.annotation_pack_root)]
    rows.sort(key=lambda r: str(r["timestamp"]))

    if month and month.lower() not in {"all", "*"}:
        rows = [r for r in rows if str(r["month"]) == month]
    if source and source.strip():
        source_norm = source.strip().lower()
        rows = [r for r in rows if str(r["source"]).lower() == source_norm]
    if is_complete is not None:
        rows = [r for r in rows if bool(r["is_complete"]) is bool(is_complete)]

    total_filtered = len(rows)
    total_pages = max(1, int(ceil(total_filtered / page_size))) if total_filtered > 0 else 1
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = start + page_size
    paged = rows[start:end]

    month_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    product_counts: dict[str, int] = {}
    for row in rows:
        mk = str(row["month"])
        sk = str(row["source"])
        pk = str(row["product_id"])
        month_counts[mk] = month_counts.get(mk, 0) + 1
        source_counts[sk] = source_counts.get(sk, 0) + 1
        product_counts[pk] = product_counts.get(pk, 0) + 1

    complete_count = sum(1 for r in rows if bool(r["is_complete"]))
    return {
        "summary": {
            "total_samples": total_filtered,
            "complete_samples": complete_count,
            "complete_rate": round(float(complete_count / total_filtered), 6) if total_filtered > 0 else 0.0,
            "month_coverage": dict(sorted(month_counts.items(), key=lambda kv: kv[0])),
            "data_version": _dataset_version(rows),
            "source_coverage_rate": _rates(source_counts, total_filtered),
            "product_coverage_rate": _rates(product_counts, total_filtered),
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        },
        "filters": {
            "month": month or "all",
            "source": source or "",
            "is_complete": is_complete,
        },
        "items": paged,
    }
