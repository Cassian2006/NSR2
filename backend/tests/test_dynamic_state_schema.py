from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.dynamic_state import (
    build_dynamic_state_sequence,
    build_resume_context,
    load_dynamic_state_sequence,
    replay_dynamic_steps,
    save_dynamic_state_sequence,
    validate_dynamic_state_sequence,
)


def _settings(tmp_path: Path) -> Settings:
    project_root = tmp_path
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=project_root,
        data_root=data_root,
        outputs_root=outputs_root,
        allow_demo_fallback=False,
    )


def _sample_explain() -> dict:
    return {
        "planner": "dstar_lite_incremental",
        "distance_km": 3200.25,
        "distance_nm": 1728.4,
        "caution_len_km": 120.3,
        "route_cost_effective_km": 3310.5,
        "route_cost_risk_extra_km": 87.2,
        "risk_mode": "balanced",
        "risk_layer": "risk_mean",
        "risk_penalty_mean": 0.18,
        "risk_penalty_p90": 0.41,
        "blocked_ratio_last": 0.27,
        "risk_constraint_mode": "chance",
        "risk_constraint_metric_name": "chance_violation_ratio",
        "risk_constraint_metric": 0.11,
        "risk_constraint_satisfied": True,
        "dynamic_advance_steps": 8,
        "replan_runtime_ms_total": 45.2,
        "replan_runtime_ms_mean": 22.6,
        "dynamic_incremental_steps": 1,
        "dynamic_rebuild_steps": 1,
        "dynamic_replans": [
            {
                "step": 0,
                "timestamp": "2024-07-01_00",
                "runtime_ms": 10.5,
                "raw_points": 120,
                "smoothed_points": 48,
                "moved_edges": 8,
                "moved_distance_km": 120.0,
                "changed_blocked_cells": 0,
                "update_mode": "init",
                "smoothing_feasible": True,
                "smoothing_fallback_reason": "",
            },
            {
                "step": 1,
                "timestamp": "2024-07-01_03",
                "runtime_ms": 15.2,
                "raw_points": 110,
                "smoothed_points": 44,
                "moved_edges": 8,
                "moved_distance_km": 130.0,
                "changed_blocked_cells": 36,
                "update_mode": "incremental",
                "smoothing_feasible": True,
                "smoothing_fallback_reason": "",
            },
        ],
    }


def test_dynamic_state_sequence_roundtrip_and_resume(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sequence = build_dynamic_state_sequence(
        settings=settings,
        timestamps=["2024-07-01_00", "2024-07-01_03", "2024-07-01_06"],
        start={"lat": 70.5, "lon": 30.0},
        goal={"lat": 72.0, "lon": 150.0},
        policy={"planner": "dstar_lite", "risk_mode": "balanced", "corridor_bias": 0.2},
        explain=_sample_explain(),
        model_version="unet_v1",
    )
    issues = validate_dynamic_state_sequence(sequence)
    assert issues == []

    version_snapshot = sequence["version_snapshot"]
    assert "dataset_version" in version_snapshot
    assert version_snapshot["model_version"] == "unet_v1"
    assert "plan_version" in version_snapshot
    assert "eval_version" in version_snapshot
    assert version_snapshot["state_version"] == "state_v1"

    meta = save_dynamic_state_sequence(settings=settings, sequence=sequence)
    assert meta["sequence_id"]
    assert Path(meta["sequence_file"]).exists()
    assert Path(meta["checkpoint_file"]).exists()

    loaded = load_dynamic_state_sequence(settings=settings, sequence_id_or_path=meta["sequence_id"])
    assert loaded["sequence_id"] == meta["sequence_id"]
    assert len(loaded["steps"]) == 2

    resume = build_resume_context(settings=settings, sequence_id_or_path=meta["sequence_id"])
    assert resume["last_step_index"] == 1
    assert resume["next_step_index"] == 2
    assert resume["last_timestamp"] == "2024-07-01_03"


def test_dynamic_state_replay_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sequence = build_dynamic_state_sequence(
        settings=settings,
        timestamps=["2024-07-01_00", "2024-07-01_03", "2024-07-01_06"],
        start={"lat": 70.5, "lon": 30.0},
        goal={"lat": 72.0, "lon": 150.0},
        policy={"planner": "dstar_lite", "risk_mode": "balanced", "corridor_bias": 0.2},
        explain=_sample_explain(),
        model_version="unet_v1",
    )
    replay = list(replay_dynamic_steps(sequence, start_step=1))
    assert len(replay) == 1
    assert replay[0]["step_index"] == 1
    assert replay[0]["timestamp"] == "2024-07-01_03"
