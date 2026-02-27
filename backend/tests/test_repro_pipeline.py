from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_repro_pipeline_sample_mode_generates_summary(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    repo_root = backend_root.parent
    script = backend_root / "scripts" / "repro_pipeline.py"
    out = tmp_path / "repro_summary_test.json"

    env = os.environ.copy()
    env["NSR_DATA_ROOT"] = str(backend_root / "demo_data")
    env["NSR_OUTPUTS_ROOT"] = str(tmp_path / "outputs")
    env["NSR_ALLOW_DEMO_FALLBACK"] = "false"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sample-mode",
            "--allow-warn",
            "--out",
            str(out),
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sample_mode"] is True
    assert payload["overall_status"] in {"PASS", "WARN", "FAIL"}
    names = [stage["name"] for stage in payload["stages"]]
    assert names == ["contract", "manifest", "quality", "registry", "smoke_plan"]
    for stage in payload["stages"]:
        assert stage["status"] in {"PASS", "WARN", "FAIL"}

