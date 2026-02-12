from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

Runner = Callable[..., Any]


STAGE_ORDER = ["qc", "suggest", "pack", "train", "eval", "report"]


@dataclass(frozen=True)
class StageConfig:
    name: str
    cmd: list[str]
    retries: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-click closed-loop pipeline: qc -> suggest -> pack -> train -> eval -> report"
    )
    p.add_argument("--run-id", default="", help="Pipeline run id. Defaults to UTC timestamp.")
    p.add_argument("--resume", action="store_true", help="Resume from existing state file.")
    p.add_argument("--stages", default="qc,suggest,pack,train,eval,report", help="Comma-separated subset of stages.")
    p.add_argument("--manifest", default="data/processed/unet_manifest_labeled.csv")
    p.add_argument("--annotation-root", default="data/processed/annotation_pack")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=1)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=2)
    p.add_argument("--train-val-steps", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--train-patch-size", type=int, default=128)
    p.add_argument("--train-loss-preset", default="caution_focus")
    p.add_argument("--planner-start", default="70.5,30.0")
    p.add_argument("--planner-goal", default="72.0,150.0")
    p.add_argument("--planner-list", default="astar,dstar_lite")
    p.add_argument("--planner-timestamps", type=int, default=3)
    p.add_argument("--planner-corridor-bias", type=float, default=0.2)
    p.add_argument("--report-mode", default="static", choices=["static", "dynamic"])
    p.add_argument("--report-planner", default="astar")
    p.add_argument("--report-infer-samples", type=int, default=2)
    p.add_argument("--retries-suggest", type=int, default=2)
    p.add_argument("--retries-train", type=int, default=2)
    p.add_argument("--retry-backoff-sec", type=float, default=1.2)
    p.add_argument("--out-root", default="outputs/pipelines")
    p.add_argument("--python-bin", default=sys.executable)
    return p.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_stages(raw: str) -> list[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        return STAGE_ORDER[:]
    out: list[str] = []
    for v in vals:
        if v not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {v}")
        if v not in out:
            out.append(v)
    return out


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _utc_now()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_state(run_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "args": vars(args),
        "status": "running",
        "stages": {},
        "artifacts": {},
        "context": {},
    }


def _latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _latest_two_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:2]


def execute_stage(
    *,
    state: dict[str, Any],
    stage: StageConfig,
    cwd: Path,
    state_path: Path,
    resume: bool,
    retry_backoff_sec: float,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    stages = state.setdefault("stages", {})
    current = stages.get(stage.name, {})
    if resume and current.get("status") == "success":
        return current

    attempts = 0
    max_attempts = max(1, int(stage.retries) + 1)
    while attempts < max_attempts:
        attempts += 1
        rec: dict[str, Any] = {
            "status": "running",
            "attempt": attempts,
            "max_attempts": max_attempts,
            "started_at": _utc_now(),
            "cmd": stage.cmd,
        }
        stages[stage.name] = rec
        _save_state(state_path, state)
        try:
            proc = runner(stage.cmd, cwd=str(cwd), text=True, capture_output=True)
            rc = int(getattr(proc, "returncode", 1))
            out = str(getattr(proc, "stdout", "") or "")
            err = str(getattr(proc, "stderr", "") or "")
        except Exception as exc:  # pragma: no cover - defensive
            rc = 1
            out = ""
            err = str(exc)

        rec.update(
            {
                "finished_at": _utc_now(),
                "returncode": rc,
                "stdout_tail": out[-4000:],
                "stderr_tail": err[-4000:],
                "attempt": attempts,
            }
        )
        if rc == 0:
            rec["status"] = "success"
            stages[stage.name] = rec
            _save_state(state_path, state)
            return rec

        rec["status"] = "failed"
        stages[stage.name] = rec
        _save_state(state_path, state)
        if attempts < max_attempts:
            time.sleep(max(0.0, float(retry_backoff_sec)) * attempts)

    return stages[stage.name]


def _resolve_input_path(raw: str, repo_root: Path, backend_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    cand_repo = (repo_root / p).resolve()
    if cand_repo.exists():
        return cand_repo
    return (backend_root / p).resolve()


def _build_stage_configs(
    args: argparse.Namespace,
    pipeline_dir: Path,
    context: dict[str, Any],
    *,
    repo_root: Path,
    backend_root: Path,
) -> list[StageConfig]:
    py = args.python_bin
    manifest_path = _resolve_input_path(str(args.manifest), repo_root, backend_root)
    annotation_root = _resolve_input_path(str(args.annotation_root), repo_root, backend_root)
    qc_json = pipeline_dir / "qc_report.json"
    qc_md = pipeline_dir / "qc_report.md"
    active_dir = pipeline_dir / "active_learning"
    batch_root = pipeline_dir / "label_batches"
    train_dir = pipeline_dir / "train_run"
    bench_dir = pipeline_dir / "benchmarks"
    report_dir = pipeline_dir / "reports"

    before_train = str(context.get("before_train_summary", ""))
    before_bench = str(context.get("before_benchmark", ""))
    after_train = str(train_dir / "summary.json")

    report_cmd = [
        py,
        "scripts/report_closed_loop_eval.py",
        "--before-train-summary",
        before_train or after_train,
        "--after-train-summary",
        after_train,
        "--before-benchmark",
        before_bench or str(bench_dir / "planner_benchmark_placeholder.json"),
        "--after-benchmark",
        str(bench_dir / "planner_benchmark_latest.json"),
        "--mode",
        str(args.report_mode),
        "--planner",
        str(args.report_planner),
        "--infer-samples",
        str(int(args.report_infer_samples)),
        "--out-dir",
        str(report_dir),
    ]

    return [
        StageConfig(
            name="qc",
            retries=0,
            cmd=[
                py,
                "scripts/qc_unet_manifest.py",
                "--manifest",
                str(manifest_path),
                "--out-json",
                str(qc_json),
                "--out-md",
                str(qc_md),
            ],
        ),
        StageConfig(
            name="suggest",
            retries=int(args.retries_suggest),
            cmd=[
                py,
                "scripts/active_learning_suggest.py",
                "--annotation-root",
                str(annotation_root),
                "--top-k",
                str(int(args.top_k)),
                "--out-dir",
                str(active_dir),
            ],
        ),
        StageConfig(
            name="pack",
            retries=0,
            cmd=[
                py,
                "scripts/prepare_unet_annotation_pack.py",
                "--out-root",
                str(annotation_root),
                "--export-batches",
                "--batch-size",
                str(int(args.batch_size)),
                "--batches-root",
                str(batch_root),
                "--resume-batches",
                "--only-unlabeled-batches",
                "--max-batches",
                str(int(args.max_batches)),
            ],
        ),
        StageConfig(
            name="train",
            retries=int(args.retries_train),
            cmd=[
                py,
                "scripts/train_unet_quick.py",
                "--manifest",
                str(manifest_path),
                "--epochs",
                str(int(args.train_epochs)),
                "--steps-per-epoch",
                str(int(args.train_steps)),
                "--val-steps",
                str(int(args.train_val_steps)),
                "--batch-size",
                str(int(args.train_batch_size)),
                "--patch-size",
                str(int(args.train_patch_size)),
                "--loss",
                "focal_dice",
                "--loss-preset",
                str(args.train_loss_preset),
                "--class-weight-mode",
                "uniform",
                "--hard-sample-quantile",
                "0.7",
                "--hard-sample-boost",
                "3.0",
                "--hard-sample-target-ratio",
                "0.55",
                "--hard-sample-max-ratio",
                "0.65",
                "--no-qc-drop",
                "--out-dir",
                str(train_dir),
            ],
        ),
        StageConfig(
            name="eval",
            retries=0,
            cmd=[
                py,
                "scripts/benchmark_planners.py",
                "--start",
                str(args.planner_start),
                "--goal",
                str(args.planner_goal),
                "--timestamps",
                str(int(args.planner_timestamps)),
                "--planners",
                str(args.planner_list),
                "--corridor-bias",
                str(float(args.planner_corridor_bias)),
                "--blocked-sources",
                "bathy,unet_blocked",
                "--out-dir",
                str(bench_dir),
            ],
        ),
        StageConfig(name="report", retries=0, cmd=report_cmd),
    ]


def _sync_runtime_artifacts(state: dict[str, Any], pipeline_dir: Path) -> None:
    artifacts = state.setdefault("artifacts", {})
    train_summary = pipeline_dir / "train_run" / "summary.json"
    if train_summary.exists():
        artifacts["after_train_summary"] = str(train_summary)
    bench_latest = _latest_file(pipeline_dir / "benchmarks", "planner_benchmark_*.json")
    if bench_latest is not None:
        artifacts["after_benchmark"] = str(bench_latest)
        # report stage expects a stable latest path.
        latest_alias = pipeline_dir / "benchmarks" / "planner_benchmark_latest.json"
        latest_alias.write_text(bench_latest.read_text(encoding="utf-8"), encoding="utf-8")
        artifacts["after_benchmark_alias"] = str(latest_alias)
    report_latest = _latest_file(pipeline_dir / "reports", "closed_loop_report_*.json")
    if report_latest is not None:
        artifacts["closed_loop_report_json"] = str(report_latest)
    report_md = _latest_file(pipeline_dir / "reports", "closed_loop_report_*.md")
    if report_md is not None:
        artifacts["closed_loop_report_md"] = str(report_md)


def _discover_baseline_context(repo_root: Path) -> dict[str, str]:
    train_roots = [repo_root / "outputs" / "train_runs", repo_root / "backend" / "outputs" / "train_runs"]
    bench_roots = [repo_root / "outputs" / "benchmarks", repo_root / "backend" / "outputs" / "benchmarks"]
    train_files: list[Path] = []
    for root in train_roots:
        if root.exists():
            train_files.extend(sorted(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    bench_files: list[Path] = []
    for root in bench_roots:
        if root.exists():
            bench_files.extend(sorted(root.glob("planner_benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True))

    out: dict[str, str] = {}
    if train_files:
        out["before_train_summary"] = str(train_files[0])
    if len(train_files) > 1:
        out["before_train_summary_prev"] = str(train_files[1])
    if bench_files:
        out["before_benchmark"] = str(bench_files[0])
    if len(bench_files) > 1:
        out["before_benchmark_prev"] = str(bench_files[1])
    return out


def run_pipeline(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    pipeline_root = backend_root / args.out_root
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pipeline_dir = pipeline_root / run_id
    state_path = pipeline_dir / "state.json"

    if args.resume and state_path.exists():
        state = _load_state(state_path)
        if not state:
            state = _new_state(run_id, args)
    else:
        state = _new_state(run_id, args)
        state["context"] = _discover_baseline_context(repo_root)
    _save_state(state_path, state)

    selected = set(_parse_stages(args.stages))
    configs = _build_stage_configs(
        args,
        pipeline_dir,
        state.get("context", {}),
        repo_root=repo_root,
        backend_root=backend_root,
    )

    for cfg in configs:
        if cfg.name not in selected:
            continue
        rec = execute_stage(
            state=state,
            stage=cfg,
            cwd=backend_root,
            state_path=state_path,
            resume=bool(args.resume),
            retry_backoff_sec=float(args.retry_backoff_sec),
            runner=runner,
        )
        _sync_runtime_artifacts(state, pipeline_dir)
        _save_state(state_path, state)
        if rec.get("status") != "success":
            state["status"] = "failed"
            _save_state(state_path, state)
            raise RuntimeError(f"Pipeline failed at stage={cfg.name}; see {state_path}")

    state["status"] = "success"
    _sync_runtime_artifacts(state, pipeline_dir)
    _save_state(state_path, state)
    return state_path


def main() -> None:
    args = parse_args()
    state_path = run_pipeline(args)
    print(f"pipeline_state={state_path}")
    state = _load_state(state_path)
    print(f"status={state.get('status', 'unknown')}")
    print("artifacts=" + json.dumps(state.get("artifacts", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
