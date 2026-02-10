from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.tiny_unet import TinyUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter potentially hard/noisy training samples by model disagreement "
            "(cross-entropy + caution IoU) and export a new manifest."
        )
    )
    p.add_argument(
        "--manifest",
        default="",
        help="Input manifest CSV. Defaults to data/processed/unet_manifest_labeled.csv",
    )
    p.add_argument(
        "--summary-json",
        default="",
        help="Training summary.json providing best checkpoint + normalization stats.",
    )
    p.add_argument(
        "--drop-train-topk",
        type=int,
        default=8,
        help="Drop top-K hardest rows in train split only.",
    )
    p.add_argument(
        "--out-manifest",
        default="",
        help="Output filtered manifest path. Defaults to data/processed/unet_manifest_filtered.csv",
    )
    p.add_argument(
        "--report-json",
        default="",
        help="Output report json path. Defaults to data/processed/unet_manifest_filtered.report.json",
    )
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _sample_scores(
    rows: list[dict[str, str]],
    model: TinyUNet,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for r in rows:
        x_path = Path(str(r.get("x_path", "")).strip())
        y_path = Path(str(r.get("y_path", "")).strip())
        if not x_path.exists() or not y_path.exists():
            continue

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xn = (x - mean[:, None, None]) / std[:, None, None]

        xt = torch.from_numpy(xn[None, ...])
        yt = torch.from_numpy(y[None, ...])
        with torch.no_grad():
            logits = model(xt)
            ce = float(F.cross_entropy(logits, yt).item())
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

        p = pred == 1
        t = y == 1
        caution_iou = float((p & t).sum() / max(1, (p | t).sum()))

        out.append(
            {
                "timestamp": str(r.get("timestamp", "")),
                "split": str(r.get("split", "")),
                "x_path": str(x_path),
                "y_path": str(y_path),
                "ce_loss": ce,
                "caution_iou": caution_iou,
            }
        )
    return out


def _pick_hard_train(scores: list[dict[str, float | str]], topk: int) -> set[tuple[str, str]]:
    train = [s for s in scores if str(s["split"]) == "train"]
    if not train or topk <= 0:
        return set()

    ce = np.asarray([float(s["ce_loss"]) for s in train], dtype=np.float64)
    ci = np.asarray([float(s["caution_iou"]) for s in train], dtype=np.float64)

    ce_z = (ce - ce.mean()) / max(1e-8, ce.std())
    ci_inv_z = ((1.0 - ci) - (1.0 - ci).mean()) / max(1e-8, (1.0 - ci).std())
    hard_score = 0.65 * ce_z + 0.35 * ci_inv_z

    ranked_idx = np.argsort(-hard_score)
    take = min(topk, len(train))
    drop: set[tuple[str, str]] = set()
    for i in ranked_idx[:take]:
        row = train[int(i)]
        key = (str(row["timestamp"]), str(row["x_path"]))
        drop.add(key)
    return drop


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    outputs_root = repo_root / "outputs"

    manifest = Path(args.manifest) if args.manifest else (data_root / "processed" / "unet_manifest_labeled.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        train_runs = sorted((outputs_root / "train_runs").glob("unet_quick_*/summary.json"))
        if not train_runs:
            raise FileNotFoundError("No train summary found under outputs/train_runs.")
        summary_path = train_runs[-1]
    if not summary_path.exists():
        raise FileNotFoundError(f"summary json not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    ckpt_path = Path(str(summary["best_ckpt"]))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    in_channels = int(summary["in_channels"])
    mean = np.asarray(summary["norm_mean"], dtype=np.float32)
    std = np.asarray(summary["norm_std"], dtype=np.float32)

    model = TinyUNet(in_channels=in_channels, n_classes=3, base=24)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = _load_rows(manifest)
    if not rows:
        raise RuntimeError(f"manifest empty: {manifest}")

    scores = _sample_scores(rows, model=model, mean=mean, std=std)
    drop_keys = _pick_hard_train(scores=scores, topk=args.drop_train_topk)

    kept_rows: list[dict[str, str]] = []
    dropped_rows: list[dict[str, str]] = []
    for r in rows:
        key = (str(r.get("timestamp", "")), str(r.get("x_path", "")))
        if key in drop_keys:
            dropped_rows.append(r)
        else:
            kept_rows.append(r)

    out_manifest = (
        Path(args.out_manifest)
        if args.out_manifest
        else (data_root / "processed" / "unet_manifest_filtered.csv")
    )
    out_report = (
        Path(args.report_json)
        if args.report_json
        else (data_root / "processed" / "unet_manifest_filtered.report.json")
    )
    fieldnames = list(rows[0].keys())
    _write_rows(out_manifest, kept_rows, fieldnames=fieldnames)

    dropped_ts = {str(r.get("timestamp", "")) for r in dropped_rows}
    dropped_scores = [
        s
        for s in scores
        if str(s.get("timestamp", "")) in dropped_ts and str(s.get("split", "")) == "train"
    ]
    dropped_scores = sorted(
        dropped_scores,
        key=lambda x: (float(x["ce_loss"]), -float(x["caution_iou"])),
        reverse=True,
    )

    report = {
        "input_manifest": str(manifest),
        "summary_json": str(summary_path),
        "checkpoint": str(ckpt_path),
        "drop_train_topk": int(args.drop_train_topk),
        "rows_in": len(rows),
        "rows_out": len(kept_rows),
        "rows_dropped": len(dropped_rows),
        "train_dropped": int(sum(1 for r in dropped_rows if str(r.get("split", "")) == "train")),
        "val_dropped": int(sum(1 for r in dropped_rows if str(r.get("split", "")) == "val")),
        "dropped_samples": dropped_scores,
        "out_manifest": str(out_manifest),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"manifest_in={manifest}")
    print(f"manifest_out={out_manifest}")
    print(f"rows_in={len(rows)} rows_out={len(kept_rows)} dropped={len(dropped_rows)}")
    print(f"report={out_report}")


if __name__ == "__main__":
    main()
