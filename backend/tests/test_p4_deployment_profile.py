from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.core.p4_deployment_profile import (
    build_deployment_profile,
    deployment_profile_to_markdown,
    estimate_path_bytes,
)


def _fake_benchmark_payload() -> dict:
    return {
        "created_at": "2026-02-15T00:00:00+00:00",
        "summary": {
            "static:dstar_lite": {"avg_runtime_ms": 18000.0, "success_rate": 0.9},
            "dynamic:dstar_lite": {
                "avg_runtime_ms": 28000.0,
                "avg_replan_latency_ms": 4500.0,
                "success_rate": 0.85,
            },
            "static:astar": {"avg_runtime_ms": 16000.0},
        },
    }


def _fake_runtime_profile() -> dict:
    return {
        "status": "pass",
        "runtime_monitor": {
            "memory_peak_mb": 2200.0,
            "step_update_ms_mean": 4300.0,
        },
    }


def test_build_deployment_profile_contract() -> None:
    payload = build_deployment_profile(
        benchmark_payload=_fake_benchmark_payload(),
        runtime_profile_payload=_fake_runtime_profile(),
        dataset_summary={"sample_count": 100, "months": ["2024-07", "2024-08"]},
        dataset_bytes=12 * 1024**3,
        baseline_cpu_cores=8,
    )
    assert payload["profile_version"] == "p4_deployment_profile_v1"
    assert payload["observed_runtime"]["dynamic_runtime_ms"] > 0
    assert payload["resource_recommendation"]["minimal"]["cpu_cores"] >= 2
    assert payload["resource_recommendation"]["recommended"]["cpu_cores"] >= payload["resource_recommendation"]["minimal"]["cpu_cores"]
    assert "monthly_cost_estimate_usd" in payload

    md = deployment_profile_to_markdown(payload)
    assert "## Resource Recommendation" in md
    assert "## Monthly Cost Estimate (USD)" in md


def test_estimate_path_bytes(tmp_path: Path) -> None:
    d = tmp_path / "a"
    d.mkdir(parents=True, exist_ok=True)
    (d / "x.bin").write_bytes(b"12345")
    (d / "y.bin").write_bytes(b"67890")
    assert estimate_path_bytes(d) == 10


def test_run_p4_deployment_profile_script(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    repo_root = backend_root.parent
    script = backend_root / "scripts" / "run_p4_deployment_profile.py"

    bench_json = tmp_path / "bench.json"
    bench_json.write_text(json.dumps(_fake_benchmark_payload(), ensure_ascii=False), encoding="utf-8")

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "sample.bin").write_bytes(b"x" * 1024)

    out_dir = tmp_path / "release"
    env = os.environ.copy()
    env["NSR_DATA_ROOT"] = str(data_root)
    env["NSR_OUTPUTS_ROOT"] = str(tmp_path / "outputs")
    env["NSR_ALLOW_DEMO_FALLBACK"] = "false"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--benchmark-json",
            str(bench_json),
            "--out-dir",
            str(out_dir),
            "--dataset-root",
            str(data_root),
            "--baseline-cpu-cores",
            "8",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    generated = list(out_dir.glob("p4_deployment_profile_*.json"))
    assert generated, "deployment profile json not generated"
