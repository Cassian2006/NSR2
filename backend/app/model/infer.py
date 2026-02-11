from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from app.core.config import Settings

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in non-torch deploy envs
    torch = None  # type: ignore[assignment]
    _torch_import_error: Exception | None = exc
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

    _ensure_torch_available()
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
