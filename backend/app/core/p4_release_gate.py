from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_status(value: Any, default: str = "FAIL") -> str:
    text = str(value or "").strip().upper()
    if text in {"PASS", "WARN", "FAIL"}:
        return text
    return default


@dataclass(frozen=True)
class P4GateThresholds:
    min_significance_conclusions: int = 1
    min_stratified_ok_rows: int = 1
    max_failure_ratio: float = 0.60
    min_repro_pass_ratio: float = 0.50
    min_recommended_cpu_cores: int = 2
    min_recommended_memory_gb: int = 4


def evaluate_protocol_stage(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "protocol_exists", "result": "FAIL", "message": "missing protocol payload"}]}
    protocol_hash = str(payload.get("protocol_hash", "") or payload.get("meta", {}).get("protocol_hash", ""))
    checks = [
        {"name": "protocol_exists", "result": "PASS", "message": "protocol payload available"},
        {"name": "protocol_hash", "result": "PASS" if bool(protocol_hash) else "FAIL", "message": f"protocol_hash={protocol_hash or 'missing'}"},
    ]
    status = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"status": status, "checks": checks}


def evaluate_repeatability_stage(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "repeatability_exists", "result": "FAIL", "message": "missing repeatability payload"}]}
    stage_status = _as_status(payload.get("status", "FAIL"))
    checks = [
        {"name": "repeatability_status", "result": stage_status, "message": f"status={payload.get('status', '')}"},
        {
            "name": "repeatability_case_count",
            "result": "PASS" if _safe_int(payload.get("run_a_count"), 0) > 0 and _safe_int(payload.get("run_b_count"), 0) > 0 else "FAIL",
            "message": f"run_a={payload.get('run_a_count', 0)}, run_b={payload.get('run_b_count', 0)}",
        },
    ]
    if stage_status == "FAIL" or any(c["result"] == "FAIL" for c in checks):
        status = "FAIL"
    elif stage_status == "WARN":
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}


def evaluate_metric_stage(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "metric_validation_exists", "result": "FAIL", "message": "missing metric completeness payload"}]}
    status = _as_status(payload.get("status", "FAIL"))
    checks = [
        {"name": "metric_validation_status", "result": status, "message": f"status={payload.get('status', '')}"},
        {
            "name": "core_metric_ids",
            "result": "PASS" if len(payload.get("core_metric_ids", []) if isinstance(payload.get("core_metric_ids"), list) else []) >= 3 else "FAIL",
            "message": f"core_metric_ids={len(payload.get('core_metric_ids', []) if isinstance(payload.get('core_metric_ids'), list) else [])}",
        },
    ]
    if status == "FAIL" or any(c["result"] == "FAIL" for c in checks):
        final = "FAIL"
    elif status == "WARN":
        final = "WARN"
    else:
        final = "PASS"
    return {"status": final, "checks": checks}


def evaluate_significance_stage(payload: dict[str, Any] | None, *, thresholds: P4GateThresholds) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "significance_exists", "result": "FAIL", "message": "missing significance payload"}]}
    conclusions = payload.get("conclusions", []) if isinstance(payload.get("conclusions"), list) else []
    checks = [
        {
            "name": "significance_conclusion_count",
            "result": "PASS" if len(conclusions) >= thresholds.min_significance_conclusions else "FAIL",
            "message": f"conclusions={len(conclusions)}, threshold>={thresholds.min_significance_conclusions}",
        },
        {
            "name": "significance_pairwise_exists",
            "result": "PASS" if isinstance(payload.get("pairwise_comparisons"), list) else "FAIL",
            "message": f"pairwise_comparisons_type={type(payload.get('pairwise_comparisons')).__name__}",
        },
    ]
    status = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"status": status, "checks": checks}


def evaluate_stratified_stage(payload: dict[str, Any] | None, *, thresholds: P4GateThresholds) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "stratified_exists", "result": "FAIL", "message": "missing stratified payload"}]}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    stage_status = _as_status(summary.get("status", "FAIL"))
    ok_rows = _safe_int(summary.get("ok_rows"), 0)
    checks = [
        {"name": "stratified_status", "result": stage_status, "message": f"status={summary.get('status', '')}"},
        {
            "name": "stratified_ok_rows",
            "result": "PASS" if ok_rows >= thresholds.min_stratified_ok_rows else "FAIL",
            "message": f"ok_rows={ok_rows}, threshold>={thresholds.min_stratified_ok_rows}",
        },
    ]
    if stage_status == "FAIL" or any(c["result"] == "FAIL" for c in checks):
        status = "FAIL"
    elif stage_status == "WARN":
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}


def evaluate_casebook_stage(payload: dict[str, Any] | None, *, thresholds: P4GateThresholds) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "casebook_exists", "result": "FAIL", "message": "missing failure casebook payload"}]}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    total_rows = max(1, _safe_int(summary.get("total_rows"), 0))
    failed_count = _safe_int(summary.get("failed_count"), 0)
    case_count = _safe_int(summary.get("case_count"), 0)
    failure_ratio = failed_count / float(total_rows)
    ratio_ok = failure_ratio <= thresholds.max_failure_ratio
    checks = [
        {"name": "casebook_case_count", "result": "PASS" if case_count > 0 else "FAIL", "message": f"case_count={case_count}"},
        {
            "name": "casebook_failure_ratio",
            "result": "PASS" if ratio_ok else "WARN",
            "message": f"failed/total={failed_count}/{total_rows}={failure_ratio:.4f}, threshold<={thresholds.max_failure_ratio:.4f}",
        },
    ]
    if any(c["result"] == "FAIL" for c in checks):
        status = "FAIL"
    elif any(c["result"] == "WARN" for c in checks):
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}


def evaluate_repro_stage(payload: dict[str, Any] | None, *, thresholds: P4GateThresholds) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "repro_audit_exists", "result": "FAIL", "message": "missing repro audit payload"}]}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    overall = _as_status(summary.get("overall_status", "FAIL"))
    total = max(1, _safe_int(summary.get("count"), 0))
    pass_ratio = _safe_int(summary.get("pass_count"), 0) / float(total)
    checks = [
        {"name": "repro_overall_status", "result": overall, "message": f"overall_status={summary.get('overall_status', '')}"},
        {
            "name": "repro_pass_ratio",
            "result": "PASS" if pass_ratio >= thresholds.min_repro_pass_ratio else "FAIL",
            "message": f"pass_ratio={pass_ratio:.4f}, threshold>={thresholds.min_repro_pass_ratio:.4f}",
        },
    ]
    if overall == "FAIL" or any(c["result"] == "FAIL" for c in checks):
        status = "FAIL"
    elif overall == "WARN":
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}


def evaluate_deployment_stage(payload: dict[str, Any] | None, *, thresholds: P4GateThresholds) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "deployment_profile_exists", "result": "FAIL", "message": "missing deployment profile payload"}]}
    rec = payload.get("resource_recommendation", {}) if isinstance(payload.get("resource_recommendation"), dict) else {}
    recommended = rec.get("recommended", {}) if isinstance(rec.get("recommended"), dict) else {}
    cpu = _safe_int(recommended.get("cpu_cores"), 0)
    mem = _safe_int(recommended.get("memory_gb"), 0)
    checks = [
        {"name": "deployment_profile_version", "result": "PASS" if bool(payload.get("profile_version")) else "FAIL", "message": f"profile_version={payload.get('profile_version', '')}"},
        {
            "name": "recommended_cpu",
            "result": "PASS" if cpu >= thresholds.min_recommended_cpu_cores else "FAIL",
            "message": f"cpu_cores={cpu}, threshold>={thresholds.min_recommended_cpu_cores}",
        },
        {
            "name": "recommended_memory",
            "result": "PASS" if mem >= thresholds.min_recommended_memory_gb else "FAIL",
            "message": f"memory_gb={mem}, threshold>={thresholds.min_recommended_memory_gb}",
        },
    ]
    status = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"status": status, "checks": checks}


def evaluate_compliance_stage(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "compliance_payload_exists", "result": "FAIL", "message": "missing compliance payload"}]}
    notices = payload.get("notices", []) if isinstance(payload.get("notices"), list) else []
    notice_ids = {str(item.get("id")) for item in notices if isinstance(item, dict)}
    required = {"research_only", "non_navigation_instruction", "data_freshness_required"}
    missing = sorted(required - notice_ids)
    checks = [
        {"name": "compliance_version", "result": "PASS" if bool(payload.get("version")) else "FAIL", "message": f"version={payload.get('version', '')}"},
        {"name": "compliance_required_notices", "result": "PASS" if not missing else "FAIL", "message": f"missing={missing}"},
    ]
    status = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"status": status, "checks": checks}


def evaluate_report_template_stage(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": [{"name": "report_template_exists", "result": "FAIL", "message": "missing report template payload"}]}
    required_sections = ["method", "data", "results", "statistics", "limitations", "reproducibility"]
    missing = [sec for sec in required_sections if sec not in payload]
    checks = [
        {
            "name": "report_template_version",
            "result": "PASS" if str(payload.get("template_version", "")).startswith("report_template_v1") else "FAIL",
            "message": f"template_version={payload.get('template_version', '')}",
        },
        {"name": "report_template_sections", "result": "PASS" if not missing else "FAIL", "message": f"missing={missing}"},
    ]
    status = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"status": status, "checks": checks}


def build_p4_release_report(
    *,
    protocol_payload: dict[str, Any] | None,
    repeatability_payload: dict[str, Any] | None,
    metric_payload: dict[str, Any] | None,
    significance_payload: dict[str, Any] | None,
    stratified_payload: dict[str, Any] | None,
    casebook_payload: dict[str, Any] | None,
    repro_payload: dict[str, Any] | None,
    deployment_payload: dict[str, Any] | None,
    compliance_payload: dict[str, Any] | None,
    report_template_payload: dict[str, Any] | None,
    thresholds: P4GateThresholds,
) -> dict[str, Any]:
    checks = {
        "protocol": evaluate_protocol_stage(protocol_payload),
        "repeatability": evaluate_repeatability_stage(repeatability_payload),
        "metrics": evaluate_metric_stage(metric_payload),
        "significance": evaluate_significance_stage(significance_payload, thresholds=thresholds),
        "stratified": evaluate_stratified_stage(stratified_payload, thresholds=thresholds),
        "casebook": evaluate_casebook_stage(casebook_payload, thresholds=thresholds),
        "repro": evaluate_repro_stage(repro_payload, thresholds=thresholds),
        "deployment": evaluate_deployment_stage(deployment_payload, thresholds=thresholds),
        "compliance": evaluate_compliance_stage(compliance_payload),
        "report_template": evaluate_report_template_stage(report_template_payload),
    }

    statuses = [str(v.get("status", "FAIL")).upper() for v in checks.values()]
    if any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif any(s == "WARN" for s in statuses):
        overall = "WARN"
    else:
        overall = "PASS"

    release_recommendation = overall == "PASS"
    return {
        "gate_version": "p4_release_gate_v1",
        "generated_at": _utc_now_iso(),
        "status": overall,
        "release_recommendation": release_recommendation,
        "checks": checks,
        "thresholds": {
            "min_significance_conclusions": thresholds.min_significance_conclusions,
            "min_stratified_ok_rows": thresholds.min_stratified_ok_rows,
            "max_failure_ratio": thresholds.max_failure_ratio,
            "min_repro_pass_ratio": thresholds.min_repro_pass_ratio,
            "min_recommended_cpu_cores": thresholds.min_recommended_cpu_cores,
            "min_recommended_memory_gb": thresholds.min_recommended_memory_gb,
        },
        "stage_summary": {
            "ready_for_public_demo": release_recommendation,
            "ready_for_paper_output": release_recommendation,
            "blocking_stages": [name for name, part in checks.items() if str(part.get("status", "")).upper() == "FAIL"],
            "warn_stages": [name for name, part in checks.items() if str(part.get("status", "")).upper() == "WARN"],
        },
    }


def collect_code_version(project_root: Path) -> dict[str, Any]:
    git_commit = "unknown"
    git_dirty = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or "unknown"
    except Exception:
        git_commit = "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        git_dirty = bool(dirty.strip())
    except Exception:
        git_dirty = None
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version.split()[0],
    }


def build_release_bundle(
    *,
    bundle_dir: Path,
    gate_report: dict[str, Any],
    artifact_paths: dict[str, str],
    code_version: dict[str, Any],
    data_description: dict[str, Any],
) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = bundle_dir / "artifacts"
    docs_dir = bundle_dir / "docs"
    scripts_dir = bundle_dir / "scripts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    for name, raw_path in artifact_paths.items():
        src = Path(raw_path)
        if not src.exists() or not src.is_file():
            continue
        dst = artifacts_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[name] = str(dst)

    reproduce_sh = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd \"$(dirname \"$0\")/../..\"",
            "python backend/scripts/repro_pipeline.py --sample-mode --allow-warn",
            "python backend/scripts/benchmark_planners.py --dynamic --timestamps 2 --dynamic-runs 1 --repeats 1",
            "python backend/scripts/run_p4_significance.py --mode dynamic --n-boot 400 --n-perm 1000",
            "python backend/scripts/run_p4_stratified_eval.py --risk-quantile 0.5 --distance-quantile 0.5",
            "python backend/scripts/run_p4_failure_casebook.py --runtime-degrade-pct 0.5 --cost-degrade-pct 0.15 --risk-degrade-pct 0.2",
            "python backend/scripts/run_p4_repro_audit.py --sample-mode --allow-warn",
            "python backend/scripts/run_p4_deployment_profile.py --baseline-cpu-cores 8",
            "python backend/scripts/p4_release_gate.py --allow-warn",
        ]
    ) + "\n"
    reproduce_ps1 = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "Set-Location (Join-Path $PSScriptRoot '..' '..')",
            "python backend/scripts/repro_pipeline.py --sample-mode --allow-warn",
            "python backend/scripts/benchmark_planners.py --dynamic --timestamps 2 --dynamic-runs 1 --repeats 1",
            "python backend/scripts/run_p4_significance.py --mode dynamic --n-boot 400 --n-perm 1000",
            "python backend/scripts/run_p4_stratified_eval.py --risk-quantile 0.5 --distance-quantile 0.5",
            "python backend/scripts/run_p4_failure_casebook.py --runtime-degrade-pct 0.5 --cost-degrade-pct 0.15 --risk-degrade-pct 0.2",
            "python backend/scripts/run_p4_repro_audit.py --sample-mode --allow-warn",
            "python backend/scripts/run_p4_deployment_profile.py --baseline-cpu-cores 8",
            "python backend/scripts/p4_release_gate.py --allow-warn",
        ]
    ) + "\n"
    (scripts_dir / "run_reproduce.sh").write_text(reproduce_sh, encoding="utf-8")
    (scripts_dir / "run_reproduce.ps1").write_text(reproduce_ps1, encoding="utf-8")

    readme_lines = [
        "# P4 Release Bundle",
        "",
        f"- generated_at: `{_utc_now_iso()}`",
        f"- gate_status: `{gate_report.get('status', 'FAIL')}`",
        f"- release_recommendation: `{gate_report.get('release_recommendation', False)}`",
        "",
        "## Code Version",
        f"- git_commit: `{code_version.get('git_commit', 'unknown')}`",
        f"- git_dirty: `{code_version.get('git_dirty', None)}`",
        f"- python_version: `{code_version.get('python_version', '')}`",
        "",
        "## Data Description",
        f"- dataset: `{json.dumps(data_description.get('dataset', {}), ensure_ascii=False)}`",
        f"- quality: `{json.dumps(data_description.get('quality', {}), ensure_ascii=False)}`",
        "",
        "## Artifacts",
    ]
    for name, path in sorted(copied.items()):
        readme_lines.append(f"- {name}: `{path}`")
    readme_lines.extend(
        [
            "",
            "## One-Click Reproduce",
            "- Linux/macOS: `bash scripts/run_reproduce.sh`",
            "- Windows PowerShell: `./scripts/run_reproduce.ps1`",
        ]
    )
    (docs_dir / "README_bundle.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    manifest = {
        "bundle_version": "p4_release_bundle_v1",
        "generated_at": _utc_now_iso(),
        "gate_status": gate_report.get("status", "FAIL"),
        "release_recommendation": gate_report.get("release_recommendation", False),
        "code_version": code_version,
        "data_description": data_description,
        "artifacts": copied,
        "docs": {"bundle_readme": str(docs_dir / "README_bundle.md")},
        "scripts": {
            "reproduce_sh": str(scripts_dir / "run_reproduce.sh"),
            "reproduce_ps1": str(scripts_dir / "run_reproduce.ps1"),
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
