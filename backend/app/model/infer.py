from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from app.core.config import Settings

_disable_torch = os.getenv("NSR_DISABLE_TORCH", "").strip().lower() in {"1", "true", "yes", "on"}
if _disable_torch:
    torch = None  # type: ignore[assignment]
    _torch_import_error: Exception | None = RuntimeError("torch disabled by NSR_DISABLE_TORCH")
else:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in non-torch deploy envs
        torch = None  # type: ignore[assignment]
        _torch_import_error = exc
    else:
        _torch_import_error = None


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceBundle:
    model: Any
    norm_mean: np.ndarray
    norm_std: np.ndarray
    device: str


def _ensure_torch_available() -> None:
    if torch is not None:
        return
    err_msg = (
        "PyTorch is required for on-demand inference but is not installed in this runtime. "
        "Install torch or pre-generate prediction caches under outputs/pred/<model_version>/<timestamp>.npy."
    )
    if _torch_import_error is not None:
        err_msg = f"{err_msg} Original import error: {_torch_import_error}"
    raise InferenceError(err_msg)


def _load_checkpoint(ckpt_path: Path, device: str):
    """
    Prefer safer torch loading when available; fall back for older torch versions.
    """
    _ensure_torch_available()
    try:
        assert torch is not None
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        assert torch is not None
        return torch.load(ckpt_path, map_location=device)


def _class_stats(pred: np.ndarray) -> dict[str, int]:
    return {
        "safe": int((pred == 0).sum()),
        "caution": int((pred == 1).sum()),
        "blocked": int((pred == 2).sum()),
    }


def _class_ratios(class_hist: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {"safe": 0.0, "caution": 0.0, "blocked": 0.0}
    return {
        "safe": class_hist["safe"] / total,
        "caution": class_hist["caution"] / total,
        "blocked": class_hist["blocked"] / total,
    }


def _uncertainty_proxy_from_pred(pred: np.ndarray) -> np.ndarray:
    """
    Backfill uncertainty for legacy cached predictions that were saved before
    probability entropy export existed.
    """
    if pred.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    p = pred.astype(np.int16)
    edge = np.zeros_like(p, dtype=np.float32)
    edge[1:, :] = np.maximum(edge[1:, :], (p[1:, :] != p[:-1, :]).astype(np.float32))
    edge[:-1, :] = np.maximum(edge[:-1, :], (p[:-1, :] != p[1:, :]).astype(np.float32))
    edge[:, 1:] = np.maximum(edge[:, 1:], (p[:, 1:] != p[:, :-1]).astype(np.float32))
    edge[:, :-1] = np.maximum(edge[:, :-1], (p[:, :-1] != p[:, 1:]).astype(np.float32))
    # Small neighborhood average for a smoother proxy.
    pad = np.pad(edge, ((1, 1), (1, 1)), mode="edge")
    smooth = (
        pad[1:-1, 1:-1]
        + pad[:-2, 1:-1]
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    ) / 5.0
    return np.clip(0.12 + 0.78 * smooth, 0.0, 1.0).astype(np.float32)


def _resolve_summary_path(settings: Settings, model_version: str) -> Path:
    default_summary = settings.unet_default_summary
    if model_version == "unet_v1" and default_summary.exists():
        return default_summary

    candidate = settings.outputs_root / "train_runs" / model_version / "summary.json"
    if candidate.exists():
        return candidate
    if default_summary.exists():
        return default_summary
    raise InferenceError(
        f"No summary.json found for model_version={model_version}. "
        f"Tried {candidate} and default {default_summary}."
    )


def _resolve_ckpt_path(summary: dict, summary_path: Path) -> Path:
    ckpt_raw = str(summary.get("best_ckpt", "")).strip()
    if not ckpt_raw:
        raise InferenceError(f"best_ckpt missing in summary: {summary_path}")
    ckpt_path = Path(ckpt_raw)
    if not ckpt_path.is_absolute():
        candidates = [(summary_path.parent / ckpt_path).resolve()]
        if len(summary_path.parents) >= 4:
            candidates.append((summary_path.parents[3] / ckpt_path).resolve())
        for c in candidates:
            if c.exists():
                return c
        raise InferenceError(f"Checkpoint not found: {candidates[0]}")
    if not ckpt_path.exists():
        raise InferenceError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


@lru_cache(maxsize=8)
def _load_bundle(summary_path_str: str) -> InferenceBundle:
    _ensure_torch_available()
    summary_path = Path(summary_path_str)
    if not summary_path.exists():
        raise InferenceError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    in_channels = int(summary.get("in_channels", 0))
    if in_channels <= 0:
        raise InferenceError(f"Invalid in_channels in summary: {summary_path}")

    norm_mean = np.asarray(summary.get("norm_mean", []), dtype=np.float32)
    norm_std = np.asarray(summary.get("norm_std", []), dtype=np.float32)
    if norm_mean.shape[0] != in_channels or norm_std.shape[0] != in_channels:
        raise InferenceError(
            f"Normalization stats mismatch in summary: {summary_path} "
            f"(in_channels={in_channels}, mean={norm_mean.shape}, std={norm_std.shape})"
        )
    norm_std = np.where(norm_std < 1e-6, 1.0, norm_std)

    ckpt_path = _resolve_ckpt_path(summary, summary_path)

    assert torch is not None
    from app.model.tiny_unet import TinyUNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet(in_channels=in_channels, n_classes=3, base=24).to(device)
    ckpt = _load_checkpoint(ckpt_path, device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    return InferenceBundle(
        model=model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
    )


def _load_x_stack(settings: Settings, timestamp: str) -> np.ndarray:
    x_path = settings.annotation_pack_root / timestamp / "x_stack.npy"
    if not x_path.exists():
        raise InferenceError(f"x_stack not found for timestamp={timestamp}: {x_path}")
    x = np.load(x_path).astype(np.float32)
    if x.ndim != 3:
        raise InferenceError(f"x_stack shape must be (C,H,W), got {x.shape} for {x_path}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _load_channel_names(settings: Settings, timestamp: str, in_channels: int) -> list[str]:
    meta_path = settings.annotation_pack_root / timestamp / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = meta.get("channel_names")
            if isinstance(names, list) and len(names) == in_channels:
                return [str(v) for v in names]
        except Exception:
            pass
    return [f"ch_{i}" for i in range(in_channels)]


def _load_blocked_mask(settings: Settings, timestamp: str, shape: tuple[int, int]) -> np.ndarray:
    blocked_path = settings.annotation_pack_root / timestamp / "blocked_mask.npy"
    if not blocked_path.exists():
        return np.zeros(shape, dtype=bool)
    blocked = np.load(blocked_path).astype(np.uint8) > 0
    if blocked.shape != shape:
        return np.zeros(shape, dtype=bool)
    return blocked


def _normalize_within_sea(field: np.ndarray, sea_mask: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(field.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vals = v[sea_mask]
    if vals.size == 0:
        return np.zeros_like(v, dtype=np.float32)
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    out = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def _run_heuristic_fallback(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str,
    x: np.ndarray,
    output_path: Path,
    uncertainty_path: Path,
) -> dict:
    h, w = int(x.shape[1]), int(x.shape[2])
    channel_names = _load_channel_names(settings, timestamp, in_channels=int(x.shape[0]))
    blocked = _load_blocked_mask(settings, timestamp, shape=(h, w))
    sea = ~blocked

    def idx(name: str) -> int | None:
        try:
            return channel_names.index(name)
        except ValueError:
            return None

    ice_idx = idx("ice_conc")
    wave_idx = idx("wave_hs")
    u_idx = idx("wind_u10")
    v_idx = idx("wind_v10")

    ice = _normalize_within_sea(x[ice_idx], sea) if ice_idx is not None else np.zeros((h, w), dtype=np.float32)
    wave = _normalize_within_sea(x[wave_idx], sea) if wave_idx is not None else np.zeros((h, w), dtype=np.float32)
    if u_idx is not None and v_idx is not None:
        wind_speed = np.sqrt(np.square(x[u_idx]) + np.square(x[v_idx])).astype(np.float32)
        wind = _normalize_within_sea(wind_speed, sea)
    else:
        wind = np.zeros((h, w), dtype=np.float32)

    risk = 0.5 * ice + 0.3 * wave + 0.2 * wind
    sea_vals = risk[sea]
    threshold = float(np.percentile(sea_vals, 82)) if sea_vals.size else 1.0
    caution = sea & (risk >= threshold)

    pred = np.zeros((h, w), dtype=np.uint8)
    pred[caution] = 1
    pred[blocked] = 2

    uncertainty = np.clip(1.0 - np.abs(risk - threshold) / 0.35, 0.0, 1.0).astype(np.float32)
    uncertainty[blocked] = 0.05

    np.save(output_path, pred)
    np.save(uncertainty_path, uncertainty)

    class_hist = _class_stats(pred)
    total = int(pred.size)
    return {
        "shape": [h, w],
        "class_hist": class_hist,
        "class_ratio": _class_ratios(class_hist, total),
        "cache_hit": False,
        "model_version": model_version,
        "fallback_mode": "heuristic_no_torch",
        "uncertainty_file": str(uncertainty_path),
        "uncertainty_mean": float(np.nanmean(uncertainty)),
        "uncertainty_p90": float(np.nanpercentile(uncertainty, 90)),
    }


def run_unet_inference(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str,
    output_path: Path,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_path = output_path.with_name(f"{output_path.stem}_uncertainty.npy")
    x = _load_x_stack(settings, timestamp=timestamp)
    expected_hw = (int(x.shape[1]), int(x.shape[2]))

    if output_path.exists():
        pred = np.load(output_path)
        if pred.ndim == 2 and pred.shape == expected_hw:
            class_hist = _class_stats(pred)
            total = int(pred.size)
            unc: np.ndarray | None = None
            if uncertainty_path.exists():
                loaded = np.load(uncertainty_path).astype(np.float32)
                if loaded.shape == expected_hw:
                    unc = loaded
            if unc is None:
                unc = _uncertainty_proxy_from_pred(pred)
                np.save(uncertainty_path, unc.astype(np.float32))
            uncertainty_mean = float(np.nanmean(unc))
            uncertainty_p90 = float(np.nanpercentile(unc, 90))
            return {
                "shape": list(pred.shape),
                "class_hist": class_hist,
                "class_ratio": _class_ratios(class_hist, total),
                "cache_hit": True,
                "model_version": model_version,
                "uncertainty_file": str(uncertainty_path) if uncertainty_path.exists() else "",
                "uncertainty_mean": uncertainty_mean,
                "uncertainty_p90": uncertainty_p90,
            }

    if torch is None:
        return _run_heuristic_fallback(
            settings=settings,
            timestamp=timestamp,
            model_version=model_version,
            x=x,
            output_path=output_path,
            uncertainty_path=uncertainty_path,
        )
    summary_path = _resolve_summary_path(settings, model_version=model_version)
    bundle = _load_bundle(str(summary_path.resolve()))
    if x.shape[0] != bundle.norm_mean.shape[0]:
        raise InferenceError(
            f"Input channels mismatch for timestamp={timestamp}: "
            f"x_stack has {x.shape[0]}, model expects {bundle.norm_mean.shape[0]}"
        )

    xn = (x - bundle.norm_mean[:, None, None]) / bundle.norm_std[:, None, None]
    assert torch is not None
    xt = torch.from_numpy(xn[None, ...]).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(xt)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1) / float(np.log(probs.shape[1]))
        uncertainty = entropy.squeeze(0).cpu().numpy().astype(np.float32)

    np.save(output_path, pred)
    np.save(uncertainty_path, uncertainty)
    class_hist = _class_stats(pred)
    total = int(pred.size)
    return {
        "shape": list(pred.shape),
        "class_hist": class_hist,
        "class_ratio": _class_ratios(class_hist, total),
        "cache_hit": False,
        "model_version": model_version,
        "model_summary": str(summary_path),
        "device": bundle.device,
        "uncertainty_file": str(uncertainty_path),
        "uncertainty_mean": float(np.nanmean(uncertainty)),
        "uncertainty_p90": float(np.nanpercentile(uncertainty, 90)),
    }
