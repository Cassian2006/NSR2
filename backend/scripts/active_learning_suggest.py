from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.active_explain import explain_sample, load_channel_names, render_explanation_card
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
        default=0.60,
        help="Threshold on caution probability for suggested caution mask.",
    )
    p.add_argument(
        "--max-suggest-ratio",
        type=float,
        default=0.06,
        help=(
            "Upper bound of suggested caution pixels over sea pixels. "
            "When exceeded, raise effective threshold by quantile to cap suggestion size."
        ),
    )
    p.add_argument(
        "--smooth-min-neighbors",
        type=int,
        default=2,
        help="Keep a suggested pixel only if >=N pixels are active in its 3x3 neighborhood. 0 disables.",
    )
    p.add_argument(
        "--smooth-iters",
        type=int,
        default=1,
        help="How many times to apply neighborhood smoothing on suggestion mask.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output root directory. Defaults to outputs/active_learning/<runid>",
    )
    p.add_argument(
        "--w-uncertainty",
        type=float,
        default=0.55,
        help="Weight of uncertainty component in final score.",
    )
    p.add_argument(
        "--w-route-impact",
        type=float,
        default=0.30,
        help="Weight of route-impact component in final score.",
    )
    p.add_argument(
        "--w-class-balance",
        type=float,
        default=0.15,
        help="Weight of class-balance component in final score.",
    )
    p.add_argument(
        "--class-balance-target",
        type=float,
        default=-1.0,
        help="Desired caution ratio on sea. Negative means auto from labeled set.",
    )
    p.add_argument(
        "--class-balance-width",
        type=float,
        default=0.025,
        help="Width for class-balance preference curve (smaller = stricter around target).",
    )
    return p.parse_args()


def _list_train_summaries(outputs_root: Path) -> list[Path]:
    root = outputs_root / "train_runs"
    if not root.exists():
        return []
    return sorted(root.glob("unet_quick_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_image(paths: list[Path], fallback_shape: tuple[int, int]) -> np.ndarray:
    for path in paths:
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


def _normalize_weights(w_uncertainty: float, w_route_impact: float, w_class_balance: float) -> tuple[float, float, float]:
    w = np.asarray([w_uncertainty, w_route_impact, w_class_balance], dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        return (0.55, 0.30, 0.15)
    w /= s
    return (float(w[0]), float(w[1]), float(w[2]))


def _robust_minmax(values: np.ndarray, *, q_low: float = 5.0, q_high: float = 95.0) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.percentile(values, q_low))
    hi = float(np.percentile(values, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _score_class_balance(pred_ratio: float, target_ratio: float, width: float) -> float:
    width = max(1e-6, float(width))
    z = (float(pred_ratio) - float(target_ratio)) / width
    return float(math.exp(-0.5 * z * z))


def _estimate_labeled_caution_ratio(annotation_root: Path) -> float:
    ratios: list[float] = []
    for folder in sorted(annotation_root.iterdir()):
        if not folder.is_dir() or len(folder.name) != 13:
            continue
        b_path = folder / "blocked_mask.npy"
        c_path = folder / "caution_mask.npy"
        if not b_path.exists() or not c_path.exists():
            continue
        try:
            blocked = np.load(b_path, mmap_mode="r")
            caution = np.load(c_path, mmap_mode="r")
        except Exception:
            continue
        if blocked.ndim != 2 or caution.ndim != 2 or blocked.shape != caution.shape:
            continue
        sea = blocked == 0
        sea_count = int(sea.sum())
        if sea_count <= 0:
            continue
        ratio = float((caution[sea] > 0).mean())
        if np.isfinite(ratio):
            ratios.append(ratio)
    if not ratios:
        return 0.03
    return float(np.median(np.asarray(ratios, dtype=np.float32)))


def _cap_threshold_by_ratio(
    caution_prob: np.ndarray,
    sea: np.ndarray,
    base_threshold: float,
    max_ratio: float,
) -> float:
    if max_ratio <= 0.0:
        return float(base_threshold)
    sea_count = int(sea.sum())
    if sea_count == 0:
        return float(base_threshold)
    base_ratio = float((caution_prob[sea] >= base_threshold).mean())
    if base_ratio <= max_ratio:
        return float(base_threshold)
    sea_probs = caution_prob[sea]
    quantile = max(0.0, min(1.0, 1.0 - max_ratio))
    cap_threshold = float(np.quantile(sea_probs, quantile))
    return float(max(base_threshold, cap_threshold))


def _smooth_binary_mask(mask: np.ndarray, min_neighbors: int, iters: int) -> np.ndarray:
    if min_neighbors <= 0 or iters <= 0:
        return (mask > 0).astype(np.uint8)
    out = (mask > 0).astype(np.uint8)
    for _ in range(iters):
        padded = np.pad(out, 1, mode="constant", constant_values=0)
        neighbors = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        out = (neighbors >= min_neighbors).astype(np.uint8)
    return out


def _compute_scores(
    rows: list[dict[str, object]],
    *,
    class_balance_target: float,
    class_balance_width: float,
    w_uncertainty: float,
    w_route_impact: float,
    w_class_balance: float,
) -> list[dict[str, object]]:
    if not rows:
        return rows
    unc_raw = np.asarray([float(r.get("uncertainty_raw", 0.0)) for r in rows], dtype=np.float32)
    route_raw = np.asarray([float(r.get("route_impact_raw", 0.0)) for r in rows], dtype=np.float32)
    unc_norm = _robust_minmax(unc_raw)
    route_norm = _robust_minmax(route_raw)
    wu, wr, wc = _normalize_weights(w_uncertainty, w_route_impact, w_class_balance)

    out: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        pred_ratio = float(row.get("pred_caution_ratio", 0.0))
        class_balance_score = _score_class_balance(
            pred_ratio=pred_ratio,
            target_ratio=class_balance_target,
            width=class_balance_width,
        )
        uncertainty_score = float(unc_norm[idx])
        route_impact_score = float(route_norm[idx])
        score = float(wu * uncertainty_score + wr * route_impact_score + wc * class_balance_score)
        enriched = dict(row)
        enriched.update(
            {
                "uncertainty_score": uncertainty_score,
                "route_impact_score": route_impact_score,
                "class_balance_score": class_balance_score,
                "score": score,
                "score_w_uncertainty": wu,
                "score_w_route_impact": wr,
                "score_w_class_balance": wc,
            }
        )
        out.append(enriched)
    return out


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

    auto_target = _estimate_labeled_caution_ratio(annotation_root=annotation_root)
    class_balance_target = (
        float(args.class_balance_target)
        if float(args.class_balance_target) >= 0.0
        else float(np.clip(auto_target, 0.005, 0.20))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_channels=in_channels, n_classes=3, base=24).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
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
        ais_raw = x[-1]
        sea_vals = ais_raw[sea]
        if sea_vals.size > 0 and np.isfinite(sea_vals).any():
            lo = float(np.percentile(sea_vals, 5))
            hi = float(np.percentile(sea_vals, 95))
            if hi > lo:
                ais_norm = np.clip((ais_raw - lo) / (hi - lo), 0.0, 1.0)
            else:
                ais_norm = np.zeros_like(ais_raw, dtype=np.float32)
        else:
            ais_norm = np.zeros_like(ais_raw, dtype=np.float32)
        corridor_weighted_caution = float((caution_prob[sea] * ais_norm[sea]).mean()) if int(sea.sum()) > 0 else 0.0

        uncertainty_raw = 0.55 * ent_p95 + 0.30 * ent_mean + 0.15 * margin_mean
        route_impact_raw = 0.60 * near_unc + 0.40 * corridor_weighted_caution

        rows.append(
            {
                "timestamp": ts,
                "uncertainty_raw": uncertainty_raw,
                "route_impact_raw": route_impact_raw,
                "entropy_mean": ent_mean,
                "entropy_p95": ent_p95,
                "near_boundary_unc": near_unc,
                "margin_mean": margin_mean,
                "pred_caution_ratio": pred_ratio,
                "corridor_weighted_caution": corridor_weighted_caution,
            }
        )
        if i % 50 == 0:
            print(f"scored={i}/{len(candidates)}")

    rows = _compute_scores(
        rows,
        class_balance_target=class_balance_target,
        class_balance_width=float(args.class_balance_width),
        w_uncertainty=float(args.w_uncertainty),
        w_route_impact=float(args.w_route_impact),
        w_class_balance=float(args.w_class_balance),
    )
    rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    top = rows[: max(1, args.top_k)]

    # Export top-k previews + suggested masks.
    labelme_dir = out_dir / "labelme_active_topk"
    labelme_dir.mkdir(parents=True, exist_ok=True)
    explain_dir = out_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)
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
        entropy = _entropy_from_probs(probs)
        caution_prob = probs[1]
        effective_threshold = _cap_threshold_by_ratio(
            caution_prob=caution_prob,
            sea=sea,
            base_threshold=args.pred_threshold,
            max_ratio=args.max_suggest_ratio,
        )
        suggested = ((caution_prob >= effective_threshold) & sea).astype(np.uint8)
        suggested = _smooth_binary_mask(
            suggested,
            min_neighbors=args.smooth_min_neighbors,
            iters=args.smooth_iters,
        )
        suggested &= sea.astype(np.uint8)

        # save per-timestamp suggestion for manual refinement reference
        np.save(folder / "caution_suggested.npy", suggested.astype(np.uint8))
        Image.fromarray((suggested * 255).astype(np.uint8)).save(folder / "caution_suggested.png")

        base = _load_image(
            [
                folder / "quicklook_blocked_overlay.png",
                folder / "quicklook.png",
                folder / "quicklook_riskhint.png",
            ],
            fallback_shape=suggested.shape,
        )
        overlay = _overlay_suggestion(base, suggested, blocked)
        heat = (np.clip(caution_prob, 0.0, 1.0) * 255.0).astype(np.uint8)

        file_base = f"active_{rank:03d}"
        img_path = labelme_dir / f"{file_base}.png"
        sug_path = labelme_dir / f"{file_base}_suggest.png"
        heat_path = labelme_dir / f"{file_base}_cprob.png"
        Image.fromarray(base).save(img_path)
        Image.fromarray(overlay).save(sug_path)
        Image.fromarray(heat).save(heat_path)

        channel_names = load_channel_names(folder, n_channels=int(x.shape[0]))
        explain = explain_sample(
            x_stack=x,
            blocked_mask=blocked,
            caution_prob=caution_prob,
            entropy=entropy,
            channel_names=channel_names,
            suggested_mask=suggested,
        )
        explain["timestamp"] = ts
        explain["rank"] = rank
        explain["score"] = float(row["score"])
        explain_json_path = explain_dir / f"{file_base}_explain.json"
        explain_png_path = explain_dir / f"{file_base}_explain.png"
        explain_json_path.write_text(json.dumps(explain, ensure_ascii=False, indent=2), encoding="utf-8")
        render_explanation_card(explain, explain_png_path, title=f"{ts} (rank={rank})")

        mapping_rows.append(
            {
                "rank": rank,
                "filename": f"{file_base}.png",
                "timestamp": ts,
                "score": float(row["score"]),
                "uncertainty_score": float(row["uncertainty_score"]),
                "route_impact_score": float(row["route_impact_score"]),
                "class_balance_score": float(row["class_balance_score"]),
                "pred_caution_ratio": float(row["pred_caution_ratio"]),
                "dominant_factor": str(explain.get("dominant_factor", "")),
                "explain_json": str(explain_json_path),
                "explain_png": str(explain_png_path),
            }
        )

    ranking_csv = out_dir / "ranking.csv"
    with ranking_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "score",
                "uncertainty_score",
                "route_impact_score",
                "class_balance_score",
                "score_w_uncertainty",
                "score_w_route_impact",
                "score_w_class_balance",
                "uncertainty_raw",
                "route_impact_raw",
                "entropy_mean",
                "entropy_p95",
                "near_boundary_unc",
                "margin_mean",
                "corridor_weighted_caution",
                "pred_caution_ratio",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    mapping_csv = labelme_dir / "mapping.csv"
    with mapping_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "filename",
                "timestamp",
                "score",
                "uncertainty_score",
                "route_impact_score",
                "class_balance_score",
                "pred_caution_ratio",
                "dominant_factor",
                "explain_json",
                "explain_png",
            ],
        )
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
                "max_suggest_ratio": args.max_suggest_ratio,
                "smooth_min_neighbors": args.smooth_min_neighbors,
                "smooth_iters": args.smooth_iters,
                "class_balance_target": class_balance_target,
                "class_balance_target_auto_estimate": auto_target,
                "class_balance_width": float(args.class_balance_width),
                "w_uncertainty": float(rows[0]["score_w_uncertainty"]) if rows else 0.55,
                "w_route_impact": float(rows[0]["score_w_route_impact"]) if rows else 0.30,
                "w_class_balance": float(rows[0]["score_w_class_balance"]) if rows else 0.15,
                "ranking_csv": str(ranking_csv),
                "labelme_dir": str(labelme_dir),
                "explain_dir": str(explain_dir),
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
