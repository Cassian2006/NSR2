from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_closed_loop_pipeline import StageConfig, execute_stage, run_pipeline


def test_execute_stage_retry_then_success(tmp_path: Path) -> None:
    calls = {"n": 0}

    def runner(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(returncode=1, stdout="fail once", stderr="boom")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    state = {"stages": {}}
    state_path = tmp_path / "state.json"
    rec = execute_stage(
        state=state,
        stage=StageConfig(name="suggest", cmd=["python", "-V"], retries=2),
        cwd=tmp_path,
        state_path=state_path,
        resume=False,
        retry_backoff_sec=0.0,
        runner=runner,
    )
    assert rec["status"] == "success"
    assert int(rec["attempt"]) == 2
    assert calls["n"] == 2


def test_execute_stage_resume_skips_when_done(tmp_path: Path) -> None:
    state = {
        "stages": {
            "train": {
                "status": "success",
                "attempt": 1,
                "cmd": ["python", "scripts/train_unet_quick.py"],
            }
        }
    }
    state_path = tmp_path / "state.json"

    def bad_runner(*args, **kwargs):
        raise AssertionError("runner should not be called in resume skip")

    rec = execute_stage(
        state=state,
        stage=StageConfig(name="train", cmd=["python", "-V"], retries=1),
        cwd=tmp_path,
        state_path=state_path,
        resume=True,
        retry_backoff_sec=0.0,
        runner=bad_runner,
    )
    assert rec["status"] == "success"
    assert int(rec["attempt"]) == 1


def test_run_pipeline_resume_keeps_stage_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def runner(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def fake_build(args, pipeline_dir, context, **kwargs):
        return [
            StageConfig(name="qc", cmd=["python", "-V"], retries=0),
            StageConfig(name="suggest", cmd=["python", "-V"], retries=1),
        ]

    monkeypatch.setattr("scripts.run_closed_loop_pipeline._build_stage_configs", fake_build)
    monkeypatch.setattr("scripts.run_closed_loop_pipeline._discover_baseline_context", lambda repo_root: {})
    monkeypatch.setattr("scripts.run_closed_loop_pipeline._sync_runtime_artifacts", lambda state, pipeline_dir: None)

    args = Namespace(
        run_id="test_resume",
        resume=False,
        stages="qc,suggest",
        manifest="data/processed/unet_manifest_labeled.csv",
        annotation_root="data/processed/annotation_pack",
        top_k=10,
        batch_size=10,
        max_batches=1,
        train_epochs=1,
        train_steps=1,
        train_val_steps=1,
        train_batch_size=2,
        train_patch_size=64,
        train_loss_preset="caution_focus",
        planner_start="70.5,30.0",
        planner_goal="72.0,150.0",
        planner_list="astar",
        planner_timestamps=2,
        planner_corridor_bias=0.2,
        report_mode="static",
        report_planner="astar",
        report_infer_samples=1,
        retries_suggest=1,
        retries_train=1,
        retry_backoff_sec=0.0,
        out_root=str(tmp_path / "pipelines"),
        python_bin="python",
    )

    state_path_1 = run_pipeline(args, runner=runner)
    assert state_path_1.exists()
    assert calls["n"] == 2

    args.resume = True
    state_path_2 = run_pipeline(args, runner=runner)
    assert state_path_2 == state_path_1
    assert calls["n"] == 2
