from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.model.uncertainty_calibration import (
    calibrate_confidence,
    reliability_bins,
    suggest_uncertainty_thresholds,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate U-Net uncertainty/confidence using labeled samples.")
    p.add_argument(
        "--manifest",
        default="",
        help="Path to labeled manifest CSV. Defaults to data/processed/unet_manifest_labeled.csv",
    )
    p.add_argument("--model-version", default="unet_v1")
    p.add_argument("--stride", type=int, default=4, help="Pixel stride for sampling each mask.")
    p.add_argument("--max-samples", type=int, default=0, help="Optional max sample count.")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to outputs/calibration",
    )
    return p.parse_args()


def _normalize_ts(ts: str) -> list[str]:
    t = ts.strip()
    if not t:
        return []
    cands = [t]
    cands.append(t.replace(":", "-"))
    cands.append(t.replace("-", "_", 2).replace(":", "_"))
    # Keep ordering but unique.
    out: list[str] = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: str(v) for k, v in row.items()})
    return rows


def _resolve_pred_paths(pred_root: Path, model_version: str, timestamp: str) -> tuple[Path | None, Path | None]:
    model_root = pred_root / model_version
    for cand in _normalize_ts(timestamp):
        pred = model_root / f"{cand}.npy"
        unc = model_root / f"{cand}_uncertainty.npy"
        if pred.exists() and unc.exists():
            return pred, unc
    return None, None


def _to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Uncertainty Calibration Report",
        "",
        f"- model_version: `{s['model_version']}`",
        f"- used_samples: `{s['used_samples']}`",
        f"- skipped_samples: `{s['skipped_samples']}`",
        f"- sampled_pixels: `{s['sampled_pixels']}`",
        f"- temperature: `{s['temperature']:.4f}`",
        f"- improved_metric: `{s['improved_metric']}`",
        "",
        "## Metrics",
        f"- ECE(before -> after): `{s['ece_before']:.6f} -> {s['ece_after']:.6f}`",
        f"- Brier(before -> after): `{s['brier_before']:.6f} -> {s['brier_after']:.6f}`",
        f"- NLL(before -> after): `{s['nll_before']:.6f} -> {s['nll_after']:.6f}`",
        "",
        "## Threshold Suggestions",
    ]
    for item in report.get("threshold_suggestions", []):
        lines.append(
            "- target_error={target:.2f}, uncertainty_tau={tau:.2f}, coverage={cov:.3f}, expected_error={err:.3f}".format(
                target=float(item["target_error_rate"]),
                tau=float(item["uncertainty_threshold"]),
                cov=float(item["coverage"]),
                err=float(item["expected_error"]),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- confidence is computed as `1 - uncertainty`.")
    lines.append("- calibrated confidence uses temperature scaling in logit space.")
    lines.append("- reliability bins are included in JSON for plotting/analysis.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = Path(args.manifest) if args.manifest else (repo_root / "data" / "processed" / "unet_manifest_labeled.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows = _read_manifest_rows(manifest)
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError("manifest has no rows")

    stride = max(1, int(args.stride))
    all_conf: list[np.ndarray] = []
    all_outcome: list[np.ndarray] = []
    used = 0
    skipped = 0

    for row in rows:
        ts = str(row.get("timestamp", "")).strip()
        y_path = Path(str(row.get("y_path", "")).strip())
        if not ts or (not y_path.exists()):
            skipped += 1
            continue
        pred_path, unc_path = _resolve_pred_paths(settings.pred_root, args.model_version, ts)
        if pred_path is None or unc_path is None:
            skipped += 1
            continue
        try:
            y = np.load(y_path).astype(np.int16)[::stride, ::stride]
            pred = np.load(pred_path).astype(np.int16)[::stride, ::stride]
            unc = np.load(unc_path).astype(np.float32)[::stride, ::stride]
        except Exception:
            skipped += 1
            continue
        if y.shape != pred.shape or y.shape != unc.shape:
            skipped += 1
            continue
        valid = np.isfinite(unc) & np.isin(y, [0, 1, 2]) & np.isin(pred, [0, 1, 2])
        if not np.any(valid):
            skipped += 1
            continue
        conf = 1.0 - np.clip(unc[valid], 0.0, 1.0)
        outcome = (pred[valid] == y[valid]).astype(np.float64)
        all_conf.append(conf.astype(np.float64))
        all_outcome.append(outcome.astype(np.float64))
        used += 1

    if not all_conf:
        raise RuntimeError("no valid sample found for calibration")

    confidence = np.concatenate(all_conf, axis=0)
    outcome = np.concatenate(all_outcome, axis=0)
    cal = calibrate_confidence(confidence, outcome, n_bins=max(6, int(args.bins)))

    unc = 1.0 - confidence
    suggestions = suggest_uncertainty_thresholds(unc, outcome)
    report = {
        "summary": {
            "manifest": str(manifest),
            "model_version": args.model_version,
            "used_samples": used,
            "skipped_samples": skipped,
            "sampled_pixels": int(confidence.size),
            "temperature": float(cal.temperature),
            "ece_before": float(cal.ece_before),
            "ece_after": float(cal.ece_after),
            "brier_before": float(cal.brier_before),
            "brier_after": float(cal.brier_after),
            "nll_before": float(cal.nll_before),
            "nll_after": float(cal.nll_after),
            "improved_metric": cal.improved_metric,
            "stride": stride,
            "bins": max(6, int(args.bins)),
        },
        "reliability_before": reliability_bins(confidence, outcome, n_bins=max(6, int(args.bins))),
        "reliability_after": reliability_bins(cal.confidence_after, outcome, n_bins=max(6, int(args.bins))),
        "threshold_suggestions": suggestions,
    }

    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"uncertainty_calibration_{stamp}.json"
    md_path = out_dir / f"uncertainty_calibration_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={json_path}")
    print(f"md={md_path}")
    print(
        "metrics=ece:{:.6f}->{:.6f},brier:{:.6f}->{:.6f},nll:{:.6f}->{:.6f},improved={}".format(
            report["summary"]["ece_before"],
            report["summary"]["ece_after"],
            report["summary"]["brier_before"],
            report["summary"]["brier_after"],
            report["summary"]["nll_before"],
            report["summary"]["nll_after"],
            report["summary"]["improved_metric"],
        )
    )


if __name__ == "__main__":
    main()
