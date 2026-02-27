from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.core.p4_release_gate import P4GateThresholds, build_p4_release_report


def _pass_payloads() -> dict:
    return {
        "protocol": {"protocol_hash": "abc123"},
        "repeatability": {"status": "PASS", "run_a_count": 4, "run_b_count": 4},
        "metrics": {"status": "PASS", "core_metric_ids": ["a", "b", "c"]},
        "significance": {"conclusions": [{"id": "c1"}], "pairwise_comparisons": []},
        "stratified": {"summary": {"status": "PASS", "ok_rows": 3}},
        "casebook": {"summary": {"total_rows": 10, "failed_count": 2, "case_count": 4}},
        "repro": {"summary": {"overall_status": "PASS", "count": 2, "pass_count": 1}},
        "deployment": {"profile_version": "p4_deployment_profile_v1", "resource_recommendation": {"recommended": {"cpu_cores": 4, "memory_gb": 8}}},
        "compliance": {
            "version": "compliance_v1",
            "notices": [
                {"id": "research_only"},
                {"id": "non_navigation_instruction"},
                {"id": "data_freshness_required"},
            ],
        },
        "report_template": {
            "template_version": "report_template_v1",
            "method": {},
            "data": {},
            "results": {},
            "statistics": {},
            "limitations": [],
            "reproducibility": {},
        },
    }


def test_build_p4_release_report_pass() -> None:
    p = _pass_payloads()
    report = build_p4_release_report(
        protocol_payload=p["protocol"],
        repeatability_payload=p["repeatability"],
        metric_payload=p["metrics"],
        significance_payload=p["significance"],
        stratified_payload=p["stratified"],
        casebook_payload=p["casebook"],
        repro_payload=p["repro"],
        deployment_payload=p["deployment"],
        compliance_payload=p["compliance"],
        report_template_payload=p["report_template"],
        thresholds=P4GateThresholds(),
    )
    assert report["status"] == "PASS"
    assert report["release_recommendation"] is True
    assert report["checks"]["deployment"]["status"] == "PASS"


def test_build_p4_release_report_fail_on_missing_critical() -> None:
    p = _pass_payloads()
    p["metrics"] = {"status": "FAIL", "core_metric_ids": []}
    report = build_p4_release_report(
        protocol_payload=p["protocol"],
        repeatability_payload=p["repeatability"],
        metric_payload=p["metrics"],
        significance_payload=p["significance"],
        stratified_payload=p["stratified"],
        casebook_payload=p["casebook"],
        repro_payload=p["repro"],
        deployment_payload=p["deployment"],
        compliance_payload=p["compliance"],
        report_template_payload=p["report_template"],
        thresholds=P4GateThresholds(),
    )
    assert report["status"] == "FAIL"
    assert "metrics" in report["stage_summary"]["blocking_stages"]


def test_p4_release_gate_script_with_explicit_artifacts(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    repo_root = backend_root.parent
    script = backend_root / "scripts" / "p4_release_gate.py"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    payloads = _pass_payloads()
    mapping = {
        "protocol-json": payloads["protocol"],
        "repeatability-json": payloads["repeatability"],
        "metric-json": payloads["metrics"],
        "significance-json": payloads["significance"],
        "stratified-json": payloads["stratified"],
        "casebook-json": payloads["casebook"],
        "repro-audit-json": payloads["repro"],
        "deployment-json": payloads["deployment"],
        "report-template-json": payloads["report_template"],
        "compliance-json": payloads["compliance"],
    }
    args: list[str] = [sys.executable, str(script), "--allow-warn"]
    for key, payload in mapping.items():
        file_path = artifacts_dir / f"{key}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        args.extend([f"--{key}", str(file_path)])

    out_json = tmp_path / "p4_gate.json"
    out_md = tmp_path / "p4_gate.md"
    bundle_dir = tmp_path / "bundle"
    args.extend(["--out-json", str(out_json), "--out-md", str(out_md), "--bundle-dir", str(bundle_dir)])

    env = os.environ.copy()
    env["NSR_DATA_ROOT"] = str(backend_root / "demo_data")
    env["NSR_OUTPUTS_ROOT"] = str(tmp_path / "outputs")
    env["NSR_ALLOW_DEMO_FALLBACK"] = "false"

    proc = subprocess.run(args, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists()
    assert out_md.exists()
    assert (bundle_dir / "manifest.json").exists()


def test_p4_release_gate_script_skip_quality(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    repo_root = backend_root.parent
    script = backend_root / "scripts" / "p4_release_gate.py"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    payloads = _pass_payloads()
    mapping = {
        "protocol-json": payloads["protocol"],
        "repeatability-json": payloads["repeatability"],
        "metric-json": payloads["metrics"],
        "significance-json": payloads["significance"],
        "stratified-json": payloads["stratified"],
        "casebook-json": payloads["casebook"],
        "repro-audit-json": payloads["repro"],
        "deployment-json": payloads["deployment"],
        "report-template-json": payloads["report_template"],
        "compliance-json": payloads["compliance"],
    }
    args: list[str] = [sys.executable, str(script), "--allow-warn", "--skip-quality"]
    for key, payload in mapping.items():
        file_path = artifacts_dir / f"{key}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        args.extend([f"--{key}", str(file_path)])

    out_json = tmp_path / "p4_gate_skip_quality.json"
    args.extend(["--out-json", str(out_json)])

    env = os.environ.copy()
    env["NSR_DATA_ROOT"] = str(backend_root / "demo_data")
    env["NSR_OUTPUTS_ROOT"] = str(tmp_path / "outputs")
    env["NSR_ALLOW_DEMO_FALLBACK"] = "false"

    proc = subprocess.run(args, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["data_description"]["quality"]
    assert str(summary.get("gate_status")) == "WARN"
    assert "skip" in str(summary.get("message", "")).lower()
