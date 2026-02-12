from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import get_dataset_service
from app.model.tiny_unet import TinyUNet

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_files(pattern: str, roots: list[Path], count: int = 2) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))
    unique: list[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[: max(1, int(count))]


def _resolve_ckpt(summary: dict[str, Any], summary_path: Path) -> Path:
    raw = str(summary.get("best_ckpt", "")).strip() or str(summary.get("last_ckpt", "")).strip()
    if not raw:
        raise FileNotFoundError(f"checkpoint path missing in {summary_path}")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    base = summary_path.resolve()
    cands = [(base.parent / p).resolve(), (Path.cwd().resolve() / p).resolve()]
    for i in range(2, min(7, len(base.parents))):
        cands.append((base.parents[i] / p).resolve())
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"checkpoint not found for {summary_path}: {cands[0]}")


def extract_train_metrics(summary_path: Path) -> dict[str, float | int | str]:
    s = _read_json(summary_path)
    metrics = s.get("metrics", []) or []
    val_mious = [float(x.get("val_miou", 0.0)) for x in metrics if isinstance(x, dict)]
    val_caution = [float(x.get("val_iou_caution", 0.0)) for x in metrics if isinstance(x, dict)]
    best_val_iou = float(max(val_mious)) if val_mious else 0.0
    final_val_iou = float(val_mious[-1]) if val_mious else 0.0
    best_caution_iou = float(max(val_caution)) if val_caution else 0.0
    return {
        "summary_path": str(summary_path),
        "epochs": int(s.get("epochs", 0)),
        "best_val_iou": best_val_iou,
        "final_val_iou": final_val_iou,
        "best_caution_iou": best_caution_iou,
        "best_val_loss": float(s.get("best_val_loss", 0.0)),
    }


def aggregate_route_metrics(benchmark_path: Path, *, mode: str = "static", planner: str = "astar") -> dict[str, float | int | str]:
    payload = _read_json(benchmark_path)
    rows = payload.get("rows", []) or []
    ok: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) != "ok":
            continue
        if str(row.get("mode", "")) != mode:
            continue
        if str(row.get("planner", "")) != planner:
            continue
        ok.append(row)

    if not ok:
        return {
            "benchmark_path": str(benchmark_path),
            "mode": mode,
            "planner": planner,
            "sample_count": 0,
            "route_safety": 0.0,
            "distance_km": 0.0,
            "caution_len_km": 0.0,
            "runtime_ms": 0.0,
        }

    dist = np.asarray([float(x.get("distance_km", 0.0)) for x in ok], dtype=np.float64)
    caut = np.asarray([float(x.get("caution_len_km", 0.0)) for x in ok], dtype=np.float64)
    rt = np.asarray([float(x.get("runtime_ms", 0.0)) for x in ok], dtype=np.float64)
    safety = np.clip(1.0 - caut / np.maximum(dist, 1e-6), 0.0, 1.0)

    return {
        "benchmark_path": str(benchmark_path),
        "mode": mode,
        "planner": planner,
        "sample_count": int(len(ok)),
        "route_safety": float(safety.mean()),
        "distance_km": float(dist.mean()),
        "caution_len_km": float(caut.mean()),
        "runtime_ms": float(rt.mean()),
    }


def _pick_infer_timestamps(max_count: int) -> list[str]:
    all_ts = get_dataset_service().list_timestamps(month="all")
    if not all_ts:
        return []
    max_count = max(1, int(max_count))
    if len(all_ts) <= max_count:
        return all_ts
    step = max(1, len(all_ts) // max_count)
    picks = all_ts[::step][:max_count]
    if all_ts[-1] not in picks and len(picks) < max_count:
        picks.append(all_ts[-1])
    return picks[:max_count]


def measure_inference_time(summary_path: Path, *, max_samples: int = 4) -> dict[str, float | int | str | bool]:
    if torch is None:
        return {"available": False, "reason": "torch_not_installed", "summary_path": str(summary_path)}

    settings = get_settings()
    summary = _read_json(summary_path)
    in_channels = int(summary.get("in_channels", 0))
    if in_channels <= 0:
        return {"available": False, "reason": "invalid_in_channels", "summary_path": str(summary_path)}
    mean = np.asarray(summary.get("norm_mean", []), dtype=np.float32)
    std = np.asarray(summary.get("norm_std", []), dtype=np.float32)
    if mean.shape[0] != in_channels or std.shape[0] != in_channels:
        return {"available": False, "reason": "norm_shape_mismatch", "summary_path": str(summary_path)}
    std = np.where(std < 1e-6, 1.0, std)

    ckpt_path = _resolve_ckpt(summary, summary_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet(in_channels=in_channels, n_classes=3, base=24).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    ts_list = _pick_infer_timestamps(max_count=max_samples)
    xs: list[np.ndarray] = []
    used_ts: list[str] = []
    for ts in ts_list:
        x_path = settings.annotation_pack_root / ts / "x_stack.npy"
        if not x_path.exists():
            continue
        x = np.load(x_path).astype(np.float32)
        if x.ndim != 3 or x.shape[0] != in_channels:
            continue
        xn = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0, posinf=0.0, neginf=0.0)
        xs.append(xn)
        used_ts.append(ts)
    if not xs:
        return {"available": False, "reason": "no_valid_x_stack", "summary_path": str(summary_path)}

    # Warmup once.
    with torch.no_grad():
        _ = model(torch.from_numpy(xs[0][None, ...]).to(device))

    times_ms: list[float] = []
    with torch.no_grad():
        for x in xs:
            t0 = time.perf_counter()
            _ = model(torch.from_numpy(x[None, ...]).to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            times_ms.append(float(dt))

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "available": True,
        "summary_path": str(summary_path),
        "device": device,
        "sample_count": int(arr.size),
        "timestamps": used_ts,
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def _delta(after: float, before: float) -> float:
    return float(after - before)


def build_conclusion(payload: dict[str, Any]) -> str:
    d = payload["deltas"]
    parts: list[str] = []
    if d["val_iou_delta"] > 0:
        parts.append(f"模型分割质量提升（val_iou +{d['val_iou_delta']:.4f}）")
    elif d["val_iou_delta"] < 0:
        parts.append(f"模型分割质量下降（val_iou {d['val_iou_delta']:.4f}）")
    else:
        parts.append("模型分割质量基本持平（val_iou 无显著变化）")

    if d["route_safety_delta"] > 0:
        parts.append(f"航线安全性提升（route_safety +{d['route_safety_delta']:.4f}）")
    elif d["route_safety_delta"] < 0:
        parts.append(f"航线安全性下降（route_safety {d['route_safety_delta']:.4f}）")
    else:
        parts.append("航线安全性基本持平")

    if d["route_length_delta_km"] < 0:
        parts.append(f"平均航程缩短（{d['route_length_delta_km']:.2f} km）")
    elif d["route_length_delta_km"] > 0:
        parts.append(f"平均航程增加（+{d['route_length_delta_km']:.2f} km）")
    else:
        parts.append("平均航程无变化")

    infer = payload.get("inference", {})
    if isinstance(infer, dict) and infer.get("after", {}).get("available") and infer.get("before", {}).get("available"):
        dt = float(d["inference_time_delta_ms"])
        if dt < 0:
            parts.append(f"推理耗时优化（{dt:.2f} ms）")
        elif dt > 0:
            parts.append(f"推理耗时上升（+{dt:.2f} ms）")
        else:
            parts.append("推理耗时无变化")
    else:
        parts.append("推理耗时未纳入有效对比（环境或模型条件不足）")

    return "；".join(parts) + "。建议以 route_safety 与 val_iou 的联合趋势作为下一轮迭代主目标。"


def _to_markdown(report: dict[str, Any]) -> str:
    b_train = report["before"]["train"]
    a_train = report["after"]["train"]
    b_route = report["before"]["route"]
    a_route = report["after"]["route"]
    d = report["deltas"]
    lines = [
        "# 闭环对比评估报告（标注 -> 训练 -> 规划）",
        "",
        f"- 生成时间: `{report['meta']['created_at']}`",
        f"- 规划模式: `{report['meta']['mode']}`",
        f"- 规划器: `{report['meta']['planner']}`",
        "",
        "## 训练对比",
        f"- before val_iou(best): `{b_train['best_val_iou']:.6f}`",
        f"- after  val_iou(best): `{a_train['best_val_iou']:.6f}`",
        f"- delta: `{d['val_iou_delta']:+.6f}`",
        "",
        "## 规划对比",
        f"- before route_safety: `{b_route['route_safety']:.6f}`",
        f"- after  route_safety: `{a_route['route_safety']:.6f}`",
        f"- delta route_safety: `{d['route_safety_delta']:+.6f}`",
        f"- before avg_distance_km: `{b_route['distance_km']:.3f}`",
        f"- after  avg_distance_km: `{a_route['distance_km']:.3f}`",
        f"- route_length_delta_km: `{d['route_length_delta_km']:+.3f}`",
        "",
        "## 推理效率对比",
    ]
    infer = report.get("inference", {})
    if isinstance(infer, dict) and infer.get("before", {}).get("available") and infer.get("after", {}).get("available"):
        b_inf = infer["before"]
        a_inf = infer["after"]
        lines.extend(
            [
                f"- before mean_ms: `{b_inf['mean_ms']:.3f}`",
                f"- after  mean_ms: `{a_inf['mean_ms']:.3f}`",
                f"- inference_time_delta_ms: `{d['inference_time_delta_ms']:+.3f}`",
            ]
        )
    else:
        lines.append("- 推理时间对比不可用（见 JSON inference 字段原因）。")

    lines.extend(
        [
            "",
            "## 结论",
            report["conclusion"],
            "",
            "## 论文/答辩引用建议",
            "- 主指标优先报告: `val_iou`、`route_safety`、`route_length_delta_km`、`inference_time_delta_ms`。",
            "- 若 route_safety 提升但 route_length 明显变长，建议并行报告成本-安全权衡曲线。",
        ]
    )
    return "\n".join(lines)


def generate_closed_loop_report(
    *,
    before_train_summary: Path,
    after_train_summary: Path,
    before_benchmark: Path,
    after_benchmark: Path,
    mode: str = "static",
    planner: str = "astar",
    include_inference: bool = True,
    infer_samples: int = 4,
) -> dict[str, Any]:
    before_train = extract_train_metrics(before_train_summary)
    after_train = extract_train_metrics(after_train_summary)
    before_route = aggregate_route_metrics(before_benchmark, mode=mode, planner=planner)
    after_route = aggregate_route_metrics(after_benchmark, mode=mode, planner=planner)

    deltas = {
        "val_iou_delta": _delta(float(after_train["best_val_iou"]), float(before_train["best_val_iou"])),
        "route_safety_delta": _delta(float(after_route["route_safety"]), float(before_route["route_safety"])),
        "route_length_delta_km": _delta(float(after_route["distance_km"]), float(before_route["distance_km"])),
        "inference_time_delta_ms": 0.0,
    }

    inference: dict[str, Any] = {
        "before": {"available": False, "reason": "skipped"},
        "after": {"available": False, "reason": "skipped"},
    }
    if include_inference:
        inference["before"] = measure_inference_time(before_train_summary, max_samples=infer_samples)
        inference["after"] = measure_inference_time(after_train_summary, max_samples=infer_samples)
        if inference["before"].get("available") and inference["after"].get("available"):
            deltas["inference_time_delta_ms"] = _delta(
                float(inference["after"]["mean_ms"]),
                float(inference["before"]["mean_ms"]),
            )

    report: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "planner": planner,
        },
        "before": {
            "train": before_train,
            "route": before_route,
        },
        "after": {
            "train": after_train,
            "route": after_route,
        },
        "inference": inference,
        "deltas": deltas,
        "conclusion": "",
    }
    report["conclusion"] = build_conclusion(report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate closed-loop comparison report for train/plan/inference.")
    p.add_argument("--before-train-summary", default="", help="Path to baseline training summary.json")
    p.add_argument("--after-train-summary", default="", help="Path to new training summary.json")
    p.add_argument("--before-benchmark", default="", help="Path to baseline planner benchmark json")
    p.add_argument("--after-benchmark", default="", help="Path to new planner benchmark json")
    p.add_argument("--mode", choices=["static", "dynamic"], default="static")
    p.add_argument("--planner", default="astar")
    p.add_argument("--infer-samples", type=int, default=4)
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--out-dir", default="", help="Defaults to outputs/closed_loop")
    return p.parse_args()


def _auto_pick(args: argparse.Namespace, repo_root: Path) -> tuple[Path, Path, Path, Path]:
    train_roots = [
        repo_root / "outputs" / "train_runs",
        repo_root / "backend" / "outputs" / "train_runs",
    ]
    bench_roots = [
        repo_root / "outputs" / "benchmarks",
        repo_root / "backend" / "outputs" / "benchmarks",
    ]
    train_files = _find_latest_files("*/summary.json", train_roots, count=2)
    bench_files = _find_latest_files("planner_benchmark_*.json", bench_roots, count=2)

    b_train = Path(args.before_train_summary) if args.before_train_summary else (train_files[1] if len(train_files) > 1 else train_files[0])
    a_train = Path(args.after_train_summary) if args.after_train_summary else train_files[0]
    b_bench = Path(args.before_benchmark) if args.before_benchmark else (bench_files[1] if len(bench_files) > 1 else bench_files[0])
    a_bench = Path(args.after_benchmark) if args.after_benchmark else bench_files[0]
    return b_train, a_train, b_bench, a_bench


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    b_train, a_train, b_bench, a_bench = _auto_pick(args, repo_root)
    for p in [b_train, a_train, b_bench, a_bench]:
        if not p.exists():
            raise FileNotFoundError(f"required input not found: {p}")

    report = generate_closed_loop_report(
        before_train_summary=b_train,
        after_train_summary=a_train,
        before_benchmark=b_bench,
        after_benchmark=a_bench,
        mode=str(args.mode),
        planner=str(args.planner),
        include_inference=not bool(args.skip_inference),
        infer_samples=max(1, int(args.infer_samples)),
    )

    settings = get_settings()
    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "closed_loop")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"closed_loop_report_{stamp}.json"
    md_path = out_dir / f"closed_loop_report_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"conclusion={report['conclusion']}")


if __name__ == "__main__":
    main()
