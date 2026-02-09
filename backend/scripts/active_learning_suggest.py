from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.tiny_unet import TinyUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Active-learning ranking + AI-assisted caution suggestions for unlabeled timestamps"
    )
    p.add_argument(
        "--annotation-root",
        default="",
        help="Root of timestamp folders with x_stack/blocked_mask/caution_mask. Defaults to data/processed/annotation_pack",
    )
    p.add_argument(
        "--summary-json",
        default="",
        help="Training summary.json path (from train_unet_quick). If empty, auto-pick latest.",
    )
    p.add_argument(
        "--checkpoint",
        default="",
        help="Model checkpoint .pt path. If empty, use best_ckpt from summary.",
    )
    p.add_argument("--top-k", type=int, default=20, help="How many samples to export for next labeling round.")
    p.add_argument(
        "--pred-threshold",
        type=float,
        default=0.45,
        help="Threshold on caution probability for suggested caution mask.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output root directory. Defaults to outputs/active_learning/<runid>",
    )
    return p.parse_args()


def _list_train_summaries(outputs_root: Path) -> list[Path]:
    root = outputs_root / "train_runs"
    if not root.exists():
        return []
    return sorted(root.glob("unet_quick_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_image(path: Path, fallback_shape: tuple[int, int]) -> np.ndarray:
    if path.exists():
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = fallback_shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def _overlay_suggestion(base_rgb: np.ndarray, suggested: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32)
    # blocked tint red for orientation
    blk = blocked > 0
    out[blk, 0] = out[blk, 0] * 0.55 + 255 * 0.45
    out[blk, 1] = out[blk, 1] * 0.55
    out[blk, 2] = out[blk, 2] * 0.55

    # suggestion tint cyan
    s = suggested > 0
    out[s, 0] = out[s, 0] * 0.35
    out[s, 1] = out[s, 1] * 0.35 + 255 * 0.65
    out[s, 2] = out[s, 2] * 0.35 + 255 * 0.65

    # draw suggestion boundary white
    edge = np.zeros_like(s, dtype=bool)
    edge[1:, :] |= s[1:, :] != s[:-1, :]
    edge[:-1, :] |= s[:-1, :] != s[1:, :]
    edge[:, 1:] |= s[:, 1:] != s[:, :-1]
    edge[:, :-1] |= s[:, :-1] != s[:, 1:]
    out[edge] = [255, 255, 255]
    return np.clip(out, 0, 255).astype(np.uint8)


def _entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=0)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    outputs_root = repo_root / "outputs"

    annotation_root = (
        Path(args.annotation_root) if args.annotation_root else (data_root / "processed" / "annotation_pack")
    )
    if not annotation_root.exists():
        raise FileNotFoundError(f"annotation-root not found: {annotation_root}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        candidates = _list_train_summaries(outputs_root)
        if not candidates:
            raise FileNotFoundError("No train summary found under outputs/train_runs.")
        summary_path = candidates[0]
    if not summary_path.exists():
        raise FileNotFoundError(f"summary json not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(str(summary["best_ckpt"]))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    norm_mean = np.asarray(summary["norm_mean"], dtype=np.float32)
    norm_std = np.asarray(summary["norm_std"], dtype=np.float32)
    in_channels = int(summary["in_channels"])

    # candidate pool: timestamps with x_stack + blocked, but caution mask empty.
    candidates: list[Path] = []
    for folder in sorted(annotation_root.iterdir()):
        if not folder.is_dir() or len(folder.name) != 13:
            continue
        x_path = folder / "x_stack.npy"
        b_path = folder / "blocked_mask.npy"
        if not x_path.exists() or not b_path.exists():
            continue
        c_path = folder / "caution_mask.npy"
        if c_path.exists():
            c = np.load(c_path, mmap_mode="r")
            if int((c > 0).sum()) > 0:
                continue
        candidates.append(folder)

    if not candidates:
        raise RuntimeError("No unlabeled candidates found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_channels=in_channels, n_classes=3, base=24).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_root / "active_learning" / f"active_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for i, folder in enumerate(candidates, start=1):
        ts = folder.name
        x = np.load(folder / "x_stack.npy").astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.shape[0] != in_channels:
            continue
        blocked = np.load(folder / "blocked_mask.npy").astype(np.uint8)
        sea = blocked == 0

        xn = (x - norm_mean[:, None, None]) / norm_std[:, None, None]
        xt = torch.from_numpy(xn[None, ...]).to(device)
        with torch.no_grad():
            logits = model(xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        entropy = _entropy_from_probs(probs)
        caution_prob = probs[1]
        margin = 1.0 - np.abs(caution_prob - 0.5) * 2.0

        # Uncertainty near blocked boundaries is particularly useful to annotate.
        edge = np.zeros_like(blocked, dtype=bool)
        edge[1:, :] |= blocked[1:, :] != blocked[:-1, :]
        edge[:-1, :] |= blocked[:-1, :] != blocked[1:, :]
        edge[:, 1:] |= blocked[:, 1:] != blocked[:, :-1]
        edge[:, :-1] |= blocked[:, :-1] != blocked[:, 1:]
        near = edge.copy()
        for _ in range(2):
            n = near.copy()
            n[1:, :] |= near[:-1, :]
            n[:-1, :] |= near[1:, :]
            n[:, 1:] |= near[:, :-1]
            n[:, :-1] |= near[:, 1:]
            near = n
        near_sea = near & sea

        if int(sea.sum()) == 0:
            continue
        ent_mean = float(entropy[sea].mean())
        ent_p95 = float(np.percentile(entropy[sea], 95))
        margin_mean = float(margin[sea].mean())
        near_unc = float(entropy[near_sea].mean()) if int(near_sea.sum()) > 0 else ent_mean
        pred_ratio = float((caution_prob[sea] > args.pred_threshold).mean())

        # Composite active-learning score.
        score = 0.45 * ent_p95 + 0.30 * ent_mean + 0.15 * near_unc + 0.10 * margin_mean

        rows.append(
            {
                "timestamp": ts,
                "score": score,
                "entropy_mean": ent_mean,
                "entropy_p95": ent_p95,
                "near_boundary_unc": near_unc,
                "margin_mean": margin_mean,
                "pred_caution_ratio": pred_ratio,
            }
        )
        if i % 50 == 0:
            print(f"scored={i}/{len(candidates)}")

    rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    top = rows[: max(1, args.top_k)]

    # Export top-k previews + suggested masks.
    labelme_dir = out_dir / "labelme_active_topk"
    labelme_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, object]] = []
    for rank, row in enumerate(top, start=1):
        ts = str(row["timestamp"])
        folder = annotation_root / ts
        x = np.load(folder / "x_stack.npy").astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        blocked = np.load(folder / "blocked_mask.npy").astype(np.uint8)
        sea = blocked == 0

        xn = (x - norm_mean[:, None, None]) / norm_std[:, None, None]
        xt = torch.from_numpy(xn[None, ...]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(xt), dim=1).cpu().numpy()[0]
        caution_prob = probs[1]
        suggested = ((caution_prob >= args.pred_threshold) & sea).astype(np.uint8)

        # save per-timestamp suggestion for manual refinement reference
        np.save(folder / "caution_suggested.npy", suggested.astype(np.uint8))
        Image.fromarray((suggested * 255).astype(np.uint8)).save(folder / "caution_suggested.png")

        base = _load_image(folder / "quicklook_blocked_overlay.png", fallback_shape=suggested.shape)
        overlay = _overlay_suggestion(base, suggested, blocked)
        heat = (np.clip(caution_prob, 0.0, 1.0) * 255.0).astype(np.uint8)

        file_base = f"active_{rank:03d}"
        img_path = labelme_dir / f"{file_base}.png"
        sug_path = labelme_dir / f"{file_base}_suggest.png"
        heat_path = labelme_dir / f"{file_base}_cprob.png"
        Image.fromarray(base).save(img_path)
        Image.fromarray(overlay).save(sug_path)
        Image.fromarray(heat).save(heat_path)
        mapping_rows.append(
            {
                "rank": rank,
                "filename": f"{file_base}.png",
                "timestamp": ts,
                "score": float(row["score"]),
                "pred_caution_ratio": float(row["pred_caution_ratio"]),
            }
        )

    ranking_csv = out_dir / "ranking.csv"
    with ranking_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "score",
                "entropy_mean",
                "entropy_p95",
                "near_boundary_unc",
                "margin_mean",
                "pred_caution_ratio",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    mapping_csv = labelme_dir / "mapping.csv"
    with mapping_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "filename", "timestamp", "score", "pred_caution_ratio"])
        w.writeheader()
        w.writerows(mapping_rows)

    readme = labelme_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Active-learning top-k label set",
                "- Label `active_XXX.png` in Labelme with label: caution",
                "- `active_XXX_suggest.png` is AI suggestion overlay reference",
                "- `active_XXX_cprob.png` is caution probability heat map (grayscale)",
                "- Use mapping.csv to map filename -> timestamp",
            ]
        ),
        encoding="utf-8",
    )

    summary_out = out_dir / "summary.json"
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_json": str(summary_path),
                "checkpoint": str(ckpt_path),
                "annotation_root": str(annotation_root),
                "candidate_count": len(candidates),
                "ranked_count": len(rows),
                "top_k": len(top),
                "threshold": args.pred_threshold,
                "ranking_csv": str(ranking_csv),
                "labelme_dir": str(labelme_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"ranked={len(rows)} candidates={len(candidates)}")
    print(f"topk={len(top)}")
    print(f"ranking_csv={ranking_csv}")
    print(f"labelme_dir={labelme_dir}")


if __name__ == "__main__":
    main()
