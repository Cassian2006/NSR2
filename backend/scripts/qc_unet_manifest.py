from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.train_quality import evaluate_sample_quality


@dataclass(frozen=True)
class AnnotationQcResult:
    reasons: list[str]
    stats: dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quality checks for U-Net manifest samples.")
    p.add_argument(
        "--manifest",
        default="",
        help="Input manifest CSV. Defaults to data/processed/unet_manifest_labeled.csv",
    )
    p.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=2e-4,
        help="Minimum positive ratio (caution+blocked).",
    )
    p.add_argument(
        "--max-nan-ratio",
        type=float,
        default=0.95,
        help="Maximum allowed non-finite ratio in x_stack.",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="Output JSON report path. Defaults to data/processed/unet_manifest_quality_report.json",
    )
    p.add_argument(
        "--out-md",
        default="",
        help="Output Markdown report path. Defaults to the same stem as --out-json with .md suffix.",
    )
    p.add_argument(
        "--min-component-pixels",
        type=int,
        default=24,
        help="Pixels threshold below which a caution connected component is considered tiny.",
    )
    p.add_argument(
        "--max-tiny-components",
        type=int,
        default=16,
        help="Maximum allowed tiny caution connected components per sample.",
    )
    p.add_argument(
        "--max-tiny-components-ratio",
        type=float,
        default=0.25,
        help="Maximum allowed tiny caution components ratio among all caution components.",
    )
    p.add_argument(
        "--max-conflict-ratio",
        type=float,
        default=0.0,
        help="Maximum allowed overlap ratio between caution and blocked masks (over caution area).",
    )
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_binary_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2D: {path}")
        return (arr > 0).astype(np.uint8)
    try:
        from PIL import Image

        img = Image.open(path).convert("L")
        arr = np.asarray(img)
        return (arr >= 127).astype(np.uint8)
    except Exception as exc:
        raise RuntimeError(f"failed to read mask image {path}: {exc}") from exc


def _connected_components_areas(mask: np.ndarray) -> list[int]:
    """8-neighborhood connected components for sparse caution masks."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    areas: list[int] = []
    ys, xs = np.where(mask > 0)
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            for dy, dx in neighbors:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or mask[ny, nx] == 0:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        areas.append(area)
    return areas


def evaluate_annotation_quality(
    *,
    y_class: np.ndarray,
    blocked_mask: np.ndarray,
    caution_mask: np.ndarray,
    min_component_pixels: int,
    max_tiny_components: int,
    max_tiny_components_ratio: float,
    max_conflict_ratio: float,
) -> AnnotationQcResult:
    reasons: list[str] = []
    stats: dict[str, float] = {}

    if y_class.ndim != 2:
        reasons.append("bad_y_ndim")
        return AnnotationQcResult(reasons=reasons, stats=stats)

    if blocked_mask.shape != caution_mask.shape or blocked_mask.shape != y_class.shape:
        reasons.append("mask_shape_mismatch")
        stats["blocked_h"] = float(blocked_mask.shape[0]) if blocked_mask.ndim == 2 else -1.0
        stats["blocked_w"] = float(blocked_mask.shape[1]) if blocked_mask.ndim == 2 else -1.0
        stats["caution_h"] = float(caution_mask.shape[0]) if caution_mask.ndim == 2 else -1.0
        stats["caution_w"] = float(caution_mask.shape[1]) if caution_mask.ndim == 2 else -1.0
        stats["y_h"] = float(y_class.shape[0])
        stats["y_w"] = float(y_class.shape[1])
        return AnnotationQcResult(reasons=reasons, stats=stats)

    caution_pixels = int((caution_mask > 0).sum())
    blocked_pixels = int((blocked_mask > 0).sum())
    overlap_pixels = int(np.logical_and(caution_mask > 0, blocked_mask > 0).sum())
    caution_total = max(1, caution_pixels)
    overlap_ratio = float(overlap_pixels / caution_total)

    stats["caution_pixels"] = float(caution_pixels)
    stats["blocked_pixels"] = float(blocked_pixels)
    stats["overlap_pixels"] = float(overlap_pixels)
    stats["overlap_ratio"] = overlap_ratio

    if caution_pixels <= 0:
        reasons.append("empty_caution_annotation")

    if overlap_ratio > float(max_conflict_ratio):
        reasons.append(f"caution_blocked_conflict_ratio>{max_conflict_ratio}")

    areas = _connected_components_areas(caution_mask > 0)
    tiny = [a for a in areas if a < int(min_component_pixels)]
    total_comp = max(1, len(areas))
    tiny_comp_count = len(tiny)
    tiny_comp_ratio = float(tiny_comp_count / total_comp)

    stats["caution_components"] = float(len(areas))
    stats["caution_tiny_components"] = float(tiny_comp_count)
    stats["caution_tiny_components_ratio"] = tiny_comp_ratio
    stats["caution_tiny_pixel_ratio"] = float(sum(tiny) / caution_total) if caution_pixels > 0 else 0.0

    if tiny_comp_count > int(max_tiny_components) and tiny_comp_ratio > float(max_tiny_components_ratio):
        reasons.append(
            f"caution_tiny_regions>{max_tiny_components}_ratio>{max_tiny_components_ratio}"
        )

    return AnnotationQcResult(reasons=reasons, stats=stats)


def _reason_stats(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in items:
        for reason in row.get("reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines: list[str] = [
        "# U-Net Manifest Annotation QC Report",
        "",
        f"- manifest: `{report.get('manifest', '')}`",
        f"- rows: `{summary.get('rows', 0)}`",
        f"- ok_rows: `{summary.get('ok_rows', 0)}`",
        f"- bad_rows: `{summary.get('bad_rows', 0)}`",
        "",
        "## Reason Counts",
    ]
    reason_counts = report.get("reason_counts", {})
    if not reason_counts:
        lines.append("- none")
    else:
        for reason, count in reason_counts.items():
            lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Bad Samples (Top 50)"])
    bad = report.get("bad_samples", [])[:50]
    if not bad:
        lines.append("- none")
    else:
        for item in bad:
            ts = item.get("timestamp", "")
            split = item.get("split", "")
            reasons = ",".join(item.get("reasons", []))
            lines.append(f"- `{ts}` [{split}] -> `{reasons}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    manifest = Path(args.manifest) if args.manifest else (data_root / "processed" / "unet_manifest_labeled.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows = _load_rows(manifest)
    bad: list[dict] = []
    ok = 0

    for row in rows:
        x_path = Path(str(row.get("x_path", "")).strip())
        y_path = Path(str(row.get("y_path", "")).strip())
        if not x_path.exists() or not y_path.exists():
            bad.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "split": str(row.get("split", "")),
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "reasons": ["missing_file"],
                    "stats": {},
                }
            )
            continue
        try:
            x = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            qc = evaluate_sample_quality(
                np.asarray(x),
                np.asarray(y),
                min_foreground_ratio=args.min_foreground_ratio,
                max_nan_ratio=args.max_nan_ratio,
            )
        except Exception as exc:
            bad.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "split": str(row.get("split", "")),
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "reasons": [f"load_error:{exc}"],
                    "stats": {},
                }
            )
            continue

        reasons = list(qc.reasons)
        stats = dict(qc.stats)

        blocked_mask: np.ndarray | None = None
        caution_mask: np.ndarray | None = None

        blocked_path = Path(str(row.get("blocked_path", "")).strip()) if str(row.get("blocked_path", "")).strip() else None
        caution_path = Path(str(row.get("caution_path", "")).strip()) if str(row.get("caution_path", "")).strip() else None

        try:
            blocked_mask = _load_binary_mask(blocked_path) if blocked_path else (np.asarray(y) == 2).astype(np.uint8)
        except Exception as exc:
            reasons.append("blocked_mask_load_error")
            stats["blocked_mask_load_error"] = 1.0
            stats["blocked_mask_error_msg_len"] = float(len(str(exc)))
            blocked_mask = (np.asarray(y) == 2).astype(np.uint8)

        try:
            caution_mask = _load_binary_mask(caution_path) if caution_path else (np.asarray(y) == 1).astype(np.uint8)
        except Exception as exc:
            reasons.append("caution_mask_load_error")
            stats["caution_mask_load_error"] = 1.0
            stats["caution_mask_error_msg_len"] = float(len(str(exc)))
            caution_mask = (np.asarray(y) == 1).astype(np.uint8)

        anno_qc = evaluate_annotation_quality(
            y_class=np.asarray(y),
            blocked_mask=np.asarray(blocked_mask),
            caution_mask=np.asarray(caution_mask),
            min_component_pixels=int(args.min_component_pixels),
            max_tiny_components=int(args.max_tiny_components),
            max_tiny_components_ratio=float(args.max_tiny_components_ratio),
            max_conflict_ratio=float(args.max_conflict_ratio),
        )
        reasons.extend(anno_qc.reasons)
        stats.update(anno_qc.stats)

        if not reasons:
            ok += 1
        else:
            bad.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "split": str(row.get("split", "")),
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "reasons": sorted(set(reasons)),
                    "stats": stats,
                }
            )

    reason_counts = _reason_stats(bad)
    report = {
        "manifest": str(manifest),
        "summary": {
            "rows": len(rows),
            "ok_rows": ok,
            "bad_rows": len(bad),
        },
        "thresholds": {
            "min_foreground_ratio": float(args.min_foreground_ratio),
            "max_nan_ratio": float(args.max_nan_ratio),
            "min_component_pixels": int(args.min_component_pixels),
            "max_tiny_components": int(args.max_tiny_components),
            "max_tiny_components_ratio": float(args.max_tiny_components_ratio),
            "max_conflict_ratio": float(args.max_conflict_ratio),
        },
        "reason_counts": reason_counts,
        "bad_samples": bad,
    }
    out = (
        Path(args.out_json)
        if args.out_json
        else (data_root / "processed" / "unet_manifest_quality_report.json")
    )
    out_md = (
        Path(args.out_md)
        if args.out_md
        else out.with_suffix(".md")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"manifest={manifest}")
    print(f"rows={len(rows)} ok={ok} bad={len(bad)}")
    print(f"report_json={out}")
    print(f"report_md={out_md}")


if __name__ == "__main__":
    main()
