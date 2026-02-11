from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.losses import FocalDiceLoss
from app.model.tiny_unet import TinyUNet
from app.model.train_quality import build_hard_sample_weights, evaluate_sample_quality


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick U-Net baseline training on labeled NSR samples")
    p.add_argument(
        "--manifest",
        default="",
        help="Path to manifest CSV. Defaults to data/processed/unet_manifest_labeled.csv",
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--steps-per-epoch", type=int, default=60)
    p.add_argument("--val-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--focus-caution-prob",
        type=float,
        default=0.6,
        help="Probability of sampling patches centered on caution pixels when available.",
    )
    p.add_argument(
        "--aug-noise-std",
        type=float,
        default=0.03,
        help="Gaussian noise std applied after normalization during training augment.",
    )
    p.add_argument(
        "--aug-gamma-max",
        type=float,
        default=0.15,
        help="Max absolute gamma jitter for training augment. 0 disables.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output run directory. Defaults to outputs/train_runs/unet_quick_<timestamp>",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--loss",
        choices=["ce", "focal_dice"],
        default="focal_dice",
        help="Training loss type.",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--dice-smooth", type=float, default=1.0)
    p.add_argument("--loss-lambda-ce", type=float, default=0.4)
    p.add_argument("--loss-lambda-focal", type=float, default=0.3)
    p.add_argument("--loss-lambda-dice", type=float, default=0.3)
    p.add_argument(
        "--qc-min-foreground-ratio",
        type=float,
        default=2e-4,
        help="Minimum positive ratio (caution+blocked) required by sample QC.",
    )
    p.add_argument(
        "--qc-max-nan-ratio",
        type=float,
        default=0.2,
        help="Maximum allowed non-finite ratio in x_stack for sample QC.",
    )
    p.add_argument(
        "--no-qc-drop",
        action="store_true",
        help="Keep QC-failed samples (still reported), instead of dropping them from training.",
    )
    p.add_argument(
        "--hard-sample-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold for hard-sample upweighting.",
    )
    p.add_argument(
        "--hard-sample-boost",
        type=float,
        default=2.0,
        help="Weight boost multiplier for hard samples.",
    )
    return p.parse_args()


@dataclass(frozen=True)
class SampleItem:
    timestamp: str
    x_path: Path
    y_path: Path
    split: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[SampleItem]:
    items: list[SampleItem] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_path = Path(str(row.get("x_path", "")).strip())
            y_path = Path(str(row.get("y_path", "")).strip())
            split = str(row.get("split", "train")).strip() or "train"
            ts = str(row.get("timestamp", "")).strip()
            if not x_path.exists() or not y_path.exists():
                continue
            items.append(SampleItem(timestamp=ts, x_path=x_path, y_path=y_path, split=split))
    return items


def apply_quality_gate(
    items: list[SampleItem],
    *,
    min_foreground_ratio: float,
    max_nan_ratio: float,
    drop_failed: bool,
) -> tuple[list[SampleItem], dict]:
    kept: list[SampleItem] = []
    dropped: list[dict[str, object]] = []
    checked = 0

    for it in items:
        try:
            x = np.load(it.x_path, mmap_mode="r")
            y = np.load(it.y_path, mmap_mode="r")
            qc = evaluate_sample_quality(
                np.asarray(x),
                np.asarray(y),
                min_foreground_ratio=min_foreground_ratio,
                max_nan_ratio=max_nan_ratio,
            )
        except Exception as exc:
            qc = None
            reasons = [f"load_error:{exc}"]
            stats = {}
        else:
            reasons = qc.reasons
            stats = qc.stats
        checked += 1

        if reasons and drop_failed:
            dropped.append(
                {
                    "timestamp": it.timestamp,
                    "split": it.split,
                    "x_path": str(it.x_path),
                    "y_path": str(it.y_path),
                    "reasons": reasons,
                    "stats": stats,
                }
            )
            continue
        kept.append(it)

    report = {
        "checked": checked,
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_failed": bool(drop_failed),
        "qc_min_foreground_ratio": float(min_foreground_ratio),
        "qc_max_nan_ratio": float(max_nan_ratio),
        "dropped_samples": dropped,
    }
    return kept, report


def split_items(items: list[SampleItem], seed: int) -> tuple[list[SampleItem], list[SampleItem]]:
    train = [x for x in items if x.split == "train"]
    val = [x for x in items if x.split == "val"]
    if train and val:
        return train, val

    # Fallback split when manifest has no val rows.
    rng = random.Random(seed)
    all_items = items[:]
    rng.shuffle(all_items)
    cut = max(1, int(len(all_items) * 0.2))
    val = all_items[:cut]
    train = all_items[cut:]
    if not train:
        train = val[:]
    return train, val


def estimate_channel_stats(items: list[SampleItem], stride: int = 8) -> tuple[np.ndarray, np.ndarray]:
    sample_x = np.load(items[0].x_path, mmap_mode="r")
    c = int(sample_x.shape[0])
    sum_c = np.zeros(c, dtype=np.float64)
    sum2_c = np.zeros(c, dtype=np.float64)
    n = 0
    for it in items:
        x = np.load(it.x_path, mmap_mode="r").astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xs = x[:, ::stride, ::stride]
        flat = xs.reshape(c, -1)
        sum_c += flat.sum(axis=1)
        sum2_c += (flat * flat).sum(axis=1)
        n += flat.shape[1]
    mean = sum_c / max(1, n)
    var = (sum2_c / max(1, n)) - mean * mean
    std = np.sqrt(np.clip(var, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)


def estimate_class_weights(items: list[SampleItem], stride: int = 4) -> np.ndarray:
    counts = np.zeros(3, dtype=np.float64)
    for it in items:
        y = np.load(it.y_path, mmap_mode="r")[::stride, ::stride]
        bc = np.bincount(y.reshape(-1).astype(np.int64), minlength=3).astype(np.float64)
        counts += bc
    counts = np.clip(counts, 1.0, None)
    inv = counts.sum() / counts
    weights = inv / inv.mean()
    return weights.astype(np.float32)


class RandomPatchDataset(IterableDataset):
    def __init__(
        self,
        items: list[SampleItem],
        mean: np.ndarray,
        std: np.ndarray,
        patch_size: int,
        steps: int,
        seed: int,
        augment: bool,
        focus_caution_prob: float = 0.6,
        aug_noise_std: float = 0.03,
        aug_gamma_max: float = 0.15,
        item_probs: np.ndarray | None = None,
    ) -> None:
        self.items = items
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.patch = patch_size
        self.steps = steps
        self.seed = seed
        self.augment = augment
        self.focus_caution_prob = float(max(0.0, min(1.0, focus_caution_prob)))
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_gamma_max = float(max(0.0, aug_gamma_max))
        if item_probs is not None and len(item_probs) == len(items):
            p = np.asarray(item_probs, dtype=np.float64)
            p = np.clip(p, 0.0, None)
            total = float(p.sum())
            self.item_probs = (p / total) if total > 0 else None
        else:
            self.item_probs = None

    def _sample_patch_origin(
        self,
        y: np.ndarray,
        h: int,
        w: int,
        ph: int,
        pw: int,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        if self.augment and rng.random() < self.focus_caution_prob:
            ys, xs = np.where(y == 1)
            if ys.size > 0:
                k = int(rng.integers(0, ys.size))
                cy = int(ys[k])
                cx = int(xs[k])
                top = int(np.clip(cy - ph // 2, 0, h - ph))
                left = int(np.clip(cx - pw // 2, 0, w - pw))
                return top, left
        top = int(rng.integers(0, h - ph + 1)) if h > ph else 0
        left = int(rng.integers(0, w - pw + 1)) if w > pw else 0
        return top, left

    def _apply_augment(
        self, xp: np.ndarray, yp: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            xp = xp[:, :, ::-1].copy()
            yp = yp[:, ::-1].copy()
        if rng.random() < 0.3:
            xp = xp[:, ::-1, :].copy()
            yp = yp[::-1, :].copy()
        # 90-degree rotations preserve masks exactly and improve orientation robustness.
        if rng.random() < 0.5:
            k = int(rng.integers(1, 4))
            xp = np.rot90(xp, k=k, axes=(1, 2)).copy()
            yp = np.rot90(yp, k=k, axes=(0, 1)).copy()
        if self.aug_gamma_max > 0:
            gamma = float(rng.uniform(-self.aug_gamma_max, self.aug_gamma_max))
            xp = xp * (1.0 + gamma)
        if self.aug_noise_std > 0:
            noise = rng.normal(0.0, self.aug_noise_std, size=xp.shape).astype(np.float32)
            xp = xp + noise
        return xp.astype(np.float32), yp.astype(np.int64)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers

        rng = np.random.default_rng(self.seed + worker_id)
        local_steps = self.steps // n_workers + int(worker_id < (self.steps % n_workers))
        for _ in range(local_steps):
            if self.item_probs is not None:
                idx = int(rng.choice(len(self.items), p=self.item_probs))
            else:
                idx = int(rng.integers(0, len(self.items)))
            it = self.items[idx]
            x = np.load(it.x_path, mmap_mode="r")
            y = np.load(it.y_path, mmap_mode="r")
            _, h, w = x.shape
            ph = min(self.patch, h)
            pw = min(self.patch, w)
            top, left = self._sample_patch_origin(y, h, w, ph, pw, rng)

            xp = x[:, top : top + ph, left : left + pw].astype(np.float32)
            yp = y[top : top + ph, left : left + pw].astype(np.int64)
            xp = np.nan_to_num(xp, nan=0.0, posinf=0.0, neginf=0.0)
            xp = (xp - self.mean) / self.std

            if self.augment:
                xp, yp = self._apply_augment(xp, yp, rng)

            yield torch.from_numpy(xp), torch.from_numpy(yp)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    n_batch = 0
    pix_correct = 0
    pix_total = 0
    i_inter = np.zeros(3, dtype=np.float64)
    i_union = np.zeros(3, dtype=np.float64)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            pix_correct += int((pred == yb).sum().item())
            pix_total += int(yb.numel())
            for cls in range(3):
                p = pred == cls
                t = yb == cls
                i_inter[cls] += float((p & t).sum().item())
                i_union[cls] += float((p | t).sum().item())

        loss_sum += float(loss.item())
        n_batch += 1

    iou = np.divide(i_inter, np.maximum(i_union, 1.0))
    return {
        "loss": loss_sum / max(1, n_batch),
        "pixel_acc": pix_correct / max(1, pix_total),
        "iou_safe": float(iou[0]),
        "iou_caution": float(iou[1]),
        "iou_blocked": float(iou[2]),
        "miou": float(np.nanmean(iou)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    outputs_root = repo_root / "outputs"
    manifest = Path(args.manifest) if args.manifest else (data_root / "processed" / "unet_manifest_labeled.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_root / "train_runs" / f"unet_quick_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_items = read_manifest(manifest)
    if len(raw_items) < 4:
        raise RuntimeError(f"Too few labeled samples in manifest: {len(raw_items)}")
    items, qc_report = apply_quality_gate(
        raw_items,
        min_foreground_ratio=args.qc_min_foreground_ratio,
        max_nan_ratio=args.qc_max_nan_ratio,
        drop_failed=not args.no_qc_drop,
    )
    if len(items) < 4:
        raise RuntimeError(
            f"Too few labeled samples after QC: {len(items)} (raw={len(raw_items)}, dropped={qc_report['dropped']})"
        )
    train_items, val_items = split_items(items, seed=args.seed)
    if not val_items:
        # fallback: keep one for val
        val_items = train_items[:1]
        train_items = train_items[1:] if len(train_items) > 1 else train_items

    mean, std = estimate_channel_stats(train_items, stride=8)
    class_weights = estimate_class_weights(train_items, stride=4)

    sample_x = np.load(train_items[0].x_path, mmap_mode="r")
    in_channels = int(sample_x.shape[0])

    hard_probs, hard_meta = build_hard_sample_weights(
        [it.y_path for it in train_items],
        hard_quantile=args.hard_sample_quantile,
        hard_boost=args.hard_sample_boost,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_channels=in_channels, n_classes=3, base=24).to(device)
    class_weights_t = torch.tensor(class_weights, device=device)
    if args.loss == "ce":
        criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        criterion = FocalDiceLoss(
            num_classes=3,
            class_weights=class_weights_t,
            focal_gamma=args.focal_gamma,
            focal_alpha=(class_weights_t / class_weights_t.sum().clamp_min(1e-8)),
            dice_smooth=args.dice_smooth,
            lambda_ce=args.loss_lambda_ce,
            lambda_focal=args.loss_lambda_focal,
            lambda_dice=args.loss_lambda_dice,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = RandomPatchDataset(
        items=train_items,
        mean=mean,
        std=std,
        patch_size=args.patch_size,
        steps=args.steps_per_epoch,
        seed=args.seed,
        augment=True,
        focus_caution_prob=args.focus_caution_prob,
        aug_noise_std=args.aug_noise_std,
        aug_gamma_max=args.aug_gamma_max,
        item_probs=hard_probs,
    )
    val_ds = RandomPatchDataset(
        items=val_items,
        mean=mean,
        std=std,
        patch_size=args.patch_size,
        steps=args.val_steps,
        seed=args.seed + 1000,
        augment=False,
        focus_caution_prob=0.0,
        aug_noise_std=0.0,
        aug_gamma_max=0.0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    metrics: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device=device, optimizer=optimizer)
        va = run_epoch(model, val_loader, criterion, device=device, optimizer=None)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_pixel_acc": tr["pixel_acc"],
            "train_miou": tr["miou"],
            "val_loss": va["loss"],
            "val_pixel_acc": va["pixel_acc"],
            "val_miou": va["miou"],
            "val_iou_caution": va["iou_caution"],
            "val_iou_blocked": va["iou_blocked"],
        }
        metrics.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_miou={row['val_miou']:.4f} val_iou_caution={row['val_iou_caution']:.4f}"
        )
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

    summary = {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "device": str(device),
        "raw_manifest_rows": len(raw_items),
        "post_qc_rows": len(items),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "val_steps": args.val_steps,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "lr": args.lr,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "dice_smooth": args.dice_smooth,
        "loss_lambda_ce": args.loss_lambda_ce,
        "loss_lambda_focal": args.loss_lambda_focal,
        "loss_lambda_dice": args.loss_lambda_dice,
        "focus_caution_prob": args.focus_caution_prob,
        "aug_noise_std": args.aug_noise_std,
        "aug_gamma_max": args.aug_gamma_max,
        "hard_sample_quantile": args.hard_sample_quantile,
        "hard_sample_boost": args.hard_sample_boost,
        "hard_sample_meta": hard_meta,
        "in_channels": in_channels,
        "class_weights": class_weights.tolist(),
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "best_val_loss": best_val,
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "quality_report": qc_report,
        "metrics": metrics,
    }
    quality_report_path = out_dir / "quality_report.json"
    with quality_report_path.open("w", encoding="utf-8") as f:
        json.dump(qc_report, f, ensure_ascii=False, indent=2)
    summary["quality_report_file"] = str(quality_report_path)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved_summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
