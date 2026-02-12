from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SampleQuality:
    ok: bool
    reasons: list[str]
    stats: dict[str, float]


def evaluate_sample_quality(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_foreground_ratio: float = 1e-4,
    max_nan_ratio: float = 0.2,
) -> SampleQuality:
    reasons: list[str] = []

    if x.ndim != 3:
        reasons.append(f"bad_x_ndim:{x.ndim}")
    if y.ndim != 2:
        reasons.append(f"bad_y_ndim:{y.ndim}")
    if x.ndim == 3 and y.ndim == 2 and x.shape[1:] != y.shape:
        reasons.append(f"shape_mismatch:{x.shape[1:]}!={y.shape}")

    if y.ndim == 2:
        valid_mask = np.isin(y, [0, 1, 2])
        if not bool(valid_mask.all()):
            reasons.append("invalid_label_values")
    else:
        valid_mask = np.zeros((), dtype=bool)

    x_finite = np.isfinite(x) if x.ndim == 3 else np.zeros((), dtype=bool)
    nan_ratio = float(1.0 - np.mean(x_finite)) if x.ndim == 3 else 1.0
    if nan_ratio > max_nan_ratio:
        reasons.append(f"nan_ratio>{max_nan_ratio}")

    if y.ndim == 2:
        total = max(1, int(y.size))
        caution_ratio = float((y == 1).sum() / total)
        blocked_ratio = float((y == 2).sum() / total)
    else:
        caution_ratio = 0.0
        blocked_ratio = 0.0
    foreground_ratio = caution_ratio + blocked_ratio
    if foreground_ratio < min_foreground_ratio:
        reasons.append(f"foreground_ratio<{min_foreground_ratio}")

    stats = {
        "nan_ratio": nan_ratio,
        "caution_ratio": caution_ratio,
        "blocked_ratio": blocked_ratio,
        "foreground_ratio": foreground_ratio,
    }
    return SampleQuality(ok=not reasons, reasons=reasons, stats=stats)


def _mask_entropy(y: np.ndarray) -> float:
    counts = np.bincount(y.reshape(-1).astype(np.int64), minlength=3).astype(np.float64)
    probs = counts / max(1.0, counts.sum())
    valid = probs > 0
    ent = float(-(probs[valid] * np.log(probs[valid])).sum())
    return ent / float(np.log(3.0))


def _mask_edge_density(y: np.ndarray) -> float:
    if y.shape[0] < 2 or y.shape[1] < 2:
        return 0.0
    edge_h = float((y[:, 1:] != y[:, :-1]).mean())
    edge_v = float((y[1:, :] != y[:-1, :]).mean())
    return 0.5 * (edge_h + edge_v)


def hard_sample_score(y: np.ndarray) -> float:
    total = max(1, int(y.size))
    caution = float((y == 1).sum() / total)
    blocked = float((y == 2).sum() / total)
    foreground = caution + blocked
    entropy = _mask_entropy(y)
    edge_density = _mask_edge_density(y)
    rare_focus = float(min(1.0, foreground * 3.0))
    score = 0.45 * entropy + 0.35 * edge_density + 0.20 * rare_focus
    return float(np.clip(score, 0.0, 1.0))


def build_hard_sample_weights(
    y_paths: Iterable[Path],
    *,
    hard_quantile: float = 0.8,
    hard_boost: float = 2.0,
    hard_target_ratio: float = 0.0,
    hard_max_ratio: float = 0.8,
) -> tuple[np.ndarray, dict[str, float | int], np.ndarray]:
    paths = list(y_paths)
    if not paths:
        return np.zeros((0,), dtype=np.float64), {"hard_count": 0, "hard_threshold": 0.0}, np.zeros((0,), dtype=bool)

    scores = np.zeros((len(paths),), dtype=np.float64)
    for i, path in enumerate(paths):
        try:
            y = np.load(path, mmap_mode="r").astype(np.int64)
            scores[i] = hard_sample_score(y)
        except Exception:
            scores[i] = 0.0

    q = float(np.clip(hard_quantile, 0.0, 1.0))
    threshold = float(np.quantile(scores, q))
    weights = np.ones((len(paths),), dtype=np.float64)
    hard_mask = scores >= threshold
    boost = max(1.0, float(hard_boost))
    if hard_mask.any():
        weights[hard_mask] *= boost
    hard_weight_ratio_pre = float(weights[hard_mask].sum() / max(1e-12, float(weights.sum())))

    # Optional ratio control to avoid over-replaying hard pool.
    target = float(np.clip(hard_target_ratio, 0.0, 1.0))
    cap = float(np.clip(hard_max_ratio, 0.0, 1.0))
    if hard_mask.any() and (~hard_mask).any():
        hard_sum = float(weights[hard_mask].sum())
        easy_sum = float(weights[~hard_mask].sum())
        if target > 0.0:
            weights[hard_mask] *= target / max(1e-12, hard_sum)
            weights[~hard_mask] *= (1.0 - target) / max(1e-12, easy_sum)
            hard_sum = float(weights[hard_mask].sum())
            easy_sum = float(weights[~hard_mask].sum())
        hard_ratio_now = hard_sum / max(1e-12, hard_sum + easy_sum)
        if hard_ratio_now > cap:
            weights[hard_mask] *= cap / max(1e-12, hard_sum)
            weights[~hard_mask] *= (1.0 - cap) / max(1e-12, easy_sum)

    weights = weights / max(1e-12, float(weights.sum()))
    hard_weight_ratio = float(weights[hard_mask].sum()) if hard_mask.any() else 0.0

    meta: dict[str, float | int] = {
        "hard_count": int(hard_mask.sum()),
        "hard_threshold": threshold,
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max(initial=0.0)),
        "hard_quantile": q,
        "hard_boost": boost,
        "hard_target_ratio": target,
        "hard_max_ratio": cap,
        "hard_weight_ratio_pre": hard_weight_ratio_pre,
        "hard_weight_ratio": hard_weight_ratio,
    }
    return weights, meta, hard_mask


def sample_indices_from_weights(
    probs: np.ndarray,
    n_draws: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        return np.zeros((0,), dtype=np.int64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0:
        p = np.ones_like(p, dtype=np.float64) / float(len(p))
    else:
        p = p / total
    rng = np.random.default_rng(seed)
    return rng.choice(len(p), size=max(0, int(n_draws)), replace=True, p=p).astype(np.int64)
