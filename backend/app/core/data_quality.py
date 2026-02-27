from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings


TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}$")
TS_FMT = "%Y-%m-%d_%H"
STATUS_RANK = {"pass": 2, "warn": 1, "fail": 0}
CRITICAL_CHECKS = {"required_files_coverage", "shape_consistency", "timeline_continuity"}
CHECK_POLICY: dict[str, dict[str, str]] = {
    "required_files_coverage": {
        "severity": "critical",
        "fail_action": "block_release",
        "warn_action": "manual_review",
        "fix_hint": "补齐缺失的 x_stack.npy / blocked_mask.npy，并重跑数据构建。",
    },
    "shape_consistency": {
        "severity": "critical",
        "fail_action": "block_release",
        "warn_action": "manual_review",
        "fix_hint": "统一网格 shape 后再进入渲染/训练链路。",
    },
    "timeline_continuity": {
        "severity": "critical",
        "fail_action": "block_release",
        "warn_action": "manual_review",
        "fix_hint": "修复时间轴错位/大间断，重建 timestamps index。",
    },
    "auxiliary_layers_coverage": {
        "severity": "degradable",
        "fail_action": "degrade_to_warn",
        "warn_action": "allow_with_warning",
        "fix_hint": "补齐 AIS / UNet 缓存，或开启按需推理缓存策略。",
    },
    "numeric_quality_sampled": {
        "severity": "degradable",
        "fail_action": "degrade_to_warn",
        "warn_action": "allow_with_warning",
        "fix_hint": "排查异常 NaN/Inf 比例，清洗异常样本。",
    },
}


def _status(score: float) -> str:
    if score >= 0.95:
        return "pass"
    if score >= 0.75:
        return "warn"
    return "fail"


def _pick_sample_timestamps(timestamps: list[str], limit: int) -> list[str]:
    if len(timestamps) <= limit:
        return timestamps
    idx = np.linspace(0, len(timestamps) - 1, num=limit, dtype=np.int64)
    return [timestamps[int(i)] for i in idx.tolist()]


def _safe_load_shape(path: Path) -> tuple[int, ...] | None:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None
    return tuple(int(v) for v in arr.shape)


def _find_ais(settings: Settings, ts: str) -> Path | None:
    for p in settings.ais_heatmap_root.rglob(f"{ts}.npy"):
        return p
    return None


def _check_with_policy(*, name: str, status: str, detail: dict[str, Any]) -> dict[str, Any]:
    policy = CHECK_POLICY.get(
        name,
        {
            "severity": "degradable",
            "fail_action": "degrade_to_warn",
            "warn_action": "allow_with_warning",
            "fix_hint": "查看该检查明细并按数据契约修复。",
        },
    )
    effective_status = status
    if status == "fail" and policy["severity"] != "critical":
        # Non-critical fails are downgraded to warn to avoid hard blocking.
        effective_status = "warn"
    return {
        "name": name,
        "status": effective_status,
        "raw_status": status,
        "severity": policy["severity"],
        "action": policy["fail_action"] if status == "fail" else policy["warn_action"] if status == "warn" else "none",
        "fix_hint": policy["fix_hint"],
        "detail": detail,
    }


def build_data_quality_gate(report: dict[str, Any]) -> dict[str, Any]:
    checks = [c for c in report.get("checks", []) if isinstance(c, dict)]
    blockers = [
        c["name"]
        for c in checks
        if str(c.get("severity", "degradable")) == "critical" and str(c.get("status", "pass")) != "pass"
    ]
    warnings = [c["name"] for c in checks if str(c.get("status", "pass")) == "warn"]
    fails = [c["name"] for c in checks if str(c.get("status", "pass")) == "fail"]

    if blockers:
        status = "FAIL"
        block_release = True
    elif warnings or fails:
        status = "WARN"
        block_release = False
    else:
        status = "PASS"
        block_release = False

    actions: list[str] = []
    if status == "FAIL":
        actions.append("阻断发布：先修复 critical 检查项后再继续。")
    elif status == "WARN":
        actions.append("允许降级运行：记录告警并安排补数/重算。")
    else:
        actions.append("质量通过：可进入下一阶段。")

    return {
        "status": status,
        "block_release": block_release,
        "blockers": blockers,
        "warnings": warnings,
        "fails": fails,
        "actions": actions,
    }


def build_data_quality_report(
    *,
    settings: Settings,
    sample_limit: int = 80,
) -> dict[str, Any]:
    ann_root = settings.annotation_pack_root
    timestamps = []
    if ann_root.exists():
        for folder in sorted(ann_root.iterdir()):
            if folder.is_dir() and TS_RE.match(folder.name):
                timestamps.append(folder.name)

    checks: list[dict[str, Any]] = []
    issues: list[str] = []

    if not timestamps:
        empty_gate = {
            "status": "FAIL",
            "block_release": True,
            "blockers": ["required_files_coverage"],
            "warnings": [],
            "fails": ["required_files_coverage"],
            "actions": ["阻断发布：annotation_pack 为空，先补齐样本。"],
        }
        return {
            "summary": {
                "status": "fail",
                "gate_status": empty_gate["status"],
                "block_release": empty_gate["block_release"],
                "message": "No annotation_pack timestamps found",
                "timestamp_count": 0,
            },
            "checks": [],
            "issues": ["annotation_pack_empty"],
            "gate": empty_gate,
        }

    x_present = 0
    blocked_present = 0
    x_shapes: dict[tuple[int, ...], int] = {}
    blocked_shapes: dict[tuple[int, ...], int] = {}
    pred_present = 0
    unc_present = 0
    ais_present = 0
    month_counts: dict[str, int] = {}

    for ts in timestamps:
        month_counts[ts[:7]] = month_counts.get(ts[:7], 0) + 1
        folder = ann_root / ts
        x_path = folder / "x_stack.npy"
        b_path = folder / "blocked_mask.npy"
        p_path = settings.pred_root / "unet_v1" / f"{ts}.npy"
        u_path = settings.pred_root / "unet_v1" / f"{ts}_uncertainty.npy"
        if x_path.exists():
            x_present += 1
            shp = _safe_load_shape(x_path)
            if shp is not None:
                x_shapes[shp] = x_shapes.get(shp, 0) + 1
        if b_path.exists():
            blocked_present += 1
            shp = _safe_load_shape(b_path)
            if shp is not None:
                blocked_shapes[shp] = blocked_shapes.get(shp, 0) + 1
        if p_path.exists():
            pred_present += 1
        if u_path.exists():
            unc_present += 1
        if _find_ais(settings, ts) is not None:
            ais_present += 1

    ts_count = len(timestamps)
    checks.append(
        _check_with_policy(
            name="required_files_coverage",
            status=_status(min(x_present, blocked_present) / ts_count),
            detail={
                "timestamps": ts_count,
                "x_stack_present": x_present,
                "blocked_mask_present": blocked_present,
            },
        )
    )

    dominant_x_shape = max(x_shapes, key=x_shapes.get) if x_shapes else None
    dominant_b_shape = max(blocked_shapes, key=blocked_shapes.get) if blocked_shapes else None
    x_consistency = (x_shapes.get(dominant_x_shape, 0) / x_present) if (dominant_x_shape and x_present) else 0.0
    b_consistency = (blocked_shapes.get(dominant_b_shape, 0) / blocked_present) if (dominant_b_shape and blocked_present) else 0.0
    checks.append(
        _check_with_policy(
            name="shape_consistency",
            status=_status(min(x_consistency, b_consistency)),
            detail={
                "dominant_x_shape": list(dominant_x_shape) if dominant_x_shape else None,
                "dominant_blocked_shape": list(dominant_b_shape) if dominant_b_shape else None,
                "x_shape_variants": len(x_shapes),
                "blocked_shape_variants": len(blocked_shapes),
                "x_consistency": round(float(x_consistency), 4),
                "blocked_consistency": round(float(b_consistency), 4),
            },
        )
    )

    pred_cov = float(pred_present / ts_count)
    unc_cov = float(unc_present / ts_count)
    ais_cov = float(ais_present / ts_count)
    aux_status = _status(ais_cov)
    # U-Net pred/uncertainty can be generated lazily on demand in current pipeline.
    # Treat low precomputed coverage as warn instead of fail when AIS layer is healthy.
    if aux_status == "pass" and (pred_cov < 0.3 or unc_cov < 0.2):
        aux_status = "warn"
    checks.append(
        _check_with_policy(
            name="auxiliary_layers_coverage",
            status=aux_status,
            detail={
                "pred_present": pred_present,
                "uncertainty_present": unc_present,
                "ais_present": ais_present,
                "pred_coverage": round(pred_cov, 4),
                "uncertainty_coverage": round(unc_cov, 4),
                "ais_coverage": round(ais_cov, 4),
                "note": "auxiliary coverage is degradable; missing cache should warn, not hard fail.",
            },
        )
    )

    # Temporal continuity check.
    dt = [datetime.strptime(ts, TS_FMT) for ts in timestamps]
    deltas = np.array(
        [(dt[i] - dt[i - 1]).total_seconds() / 3600.0 for i in range(1, len(dt))],
        dtype=np.float64,
    )
    median_delta = float(np.median(deltas)) if deltas.size else 0.0
    if median_delta <= 0:
        continuity = 0.0
        large_gaps = []
    else:
        gap_mask = deltas > (median_delta * 1.5 + 1e-6)
        large_gaps = [
            {
                "from": timestamps[i],
                "to": timestamps[i + 1],
                "hours": float(deltas[i]),
            }
            for i, bad in enumerate(gap_mask.tolist())
            if bad
        ]
        continuity = 1.0 - float(len(large_gaps) / max(1, len(deltas)))
    checks.append(
        _check_with_policy(
            name="timeline_continuity",
            status=_status(continuity),
            detail={
                "median_step_hours": round(median_delta, 3),
                "gap_count": len(large_gaps),
                "largest_gap_hours": round(float(np.max(deltas)) if deltas.size else 0.0, 3),
                "sample_gaps": large_gaps[:20],
            },
        )
    )

    # Numeric quality sampling.
    sample_ts = _pick_sample_timestamps(timestamps, max(8, int(sample_limit)))
    nan_ratios = []
    sea_nan_ratios = []
    land_nan_ratios = []
    blocked_ratios = []
    channel_counts = []
    for ts in sample_ts:
        folder = ann_root / ts
        x_path = folder / "x_stack.npy"
        b_path = folder / "blocked_mask.npy"
        if not x_path.exists() or not b_path.exists():
            continue
        try:
            x = np.load(x_path, mmap_mode="r")
            b = np.load(b_path, mmap_mode="r")
        except Exception:
            continue
        if x.ndim == 3:
            channel_counts.append(int(x.shape[0]))
            nan_ratios.append(float(1.0 - np.mean(np.isfinite(x))))
            if b.ndim == 2 and b.shape == x.shape[1:]:
                sea_mask = b <= 0
                land_mask = b > 0
                if np.any(sea_mask):
                    sea_nan_ratios.append(float(1.0 - np.mean(np.isfinite(x[:, sea_mask]))))
                if np.any(land_mask):
                    land_nan_ratios.append(float(1.0 - np.mean(np.isfinite(x[:, land_mask]))))
        if b.ndim == 2:
            blocked_ratios.append(float(np.mean(b > 0)))
    nan_mean = float(np.mean(nan_ratios)) if nan_ratios else math.inf
    nan_p95 = float(np.percentile(nan_ratios, 95)) if nan_ratios else math.inf
    sea_nan_mean = float(np.mean(sea_nan_ratios)) if sea_nan_ratios else math.inf
    sea_nan_p95 = float(np.percentile(sea_nan_ratios, 95)) if sea_nan_ratios else math.inf
    land_nan_mean = float(np.mean(land_nan_ratios)) if land_nan_ratios else math.inf
    land_nan_p95 = float(np.percentile(land_nan_ratios, 95)) if land_nan_ratios else math.inf
    blocked_mean = float(np.mean(blocked_ratios)) if blocked_ratios else 0.0
    channel_var = len(set(channel_counts))
    numeric_score = 1.0
    if not np.isfinite(nan_mean) or not np.isfinite(nan_p95):
        numeric_score = 0.0
    elif nan_p95 > 0.6:
        numeric_score *= 0.5
    elif nan_p95 > 0.4:
        numeric_score *= 0.75
    elif nan_p95 > 0.3:
        numeric_score *= 0.9
    if np.isfinite(sea_nan_p95) and sea_nan_p95 > 0.65:
        numeric_score *= 0.8
    if channel_var > 1:
        numeric_score *= 0.7
    checks.append(
        _check_with_policy(
            name="numeric_quality_sampled",
            status=_status(numeric_score),
            detail={
                "sampled_timestamps": len(sample_ts),
                "nan_ratio_mean": round(nan_mean, 6) if np.isfinite(nan_mean) else None,
                "nan_ratio_p95": round(nan_p95, 6) if np.isfinite(nan_p95) else None,
                "sea_nan_ratio_mean": round(sea_nan_mean, 6) if np.isfinite(sea_nan_mean) else None,
                "sea_nan_ratio_p95": round(sea_nan_p95, 6) if np.isfinite(sea_nan_p95) else None,
                "land_nan_ratio_mean": round(land_nan_mean, 6) if np.isfinite(land_nan_mean) else None,
                "land_nan_ratio_p95": round(land_nan_p95, 6) if np.isfinite(land_nan_p95) else None,
                "blocked_ratio_mean": round(blocked_mean, 6),
                "channel_count_variants": channel_var,
                "channel_count_values": sorted(list(set(channel_counts)))[:10],
            },
        )
    )

    for c in checks:
        if c["status"] != "pass":
            issues.append(c["name"])

    overall = min(checks, key=lambda c: STATUS_RANK.get(str(c.get("status")), 0))["status"] if checks else "fail"
    gate = build_data_quality_gate({"checks": checks})
    summary = {
        "status": overall,
        "gate_status": gate["status"],
        "block_release": gate["block_release"],
        "timestamp_count": ts_count,
        "month_counts": month_counts,
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "issues_count": len(issues),
    }
    return {"summary": summary, "checks": checks, "issues": issues, "gate": gate}
