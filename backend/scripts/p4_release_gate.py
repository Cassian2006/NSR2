from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.compliance import build_compliance_notices
from app.core.config import get_settings
from app.core.data_quality import build_data_quality_report
from app.core.dataset import DatasetService
from app.core.gallery import GalleryService
from app.core.p4_release_gate import (
    P4GateThresholds,
    build_p4_release_report,
    build_release_bundle,
    collect_code_version,
)
from app.core.report_template import build_report_template
from app.core.risk_report import build_risk_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P4 release gate and release bundle generator.")
    p.add_argument("--allow-warn", action="store_true", help="Return exit 0 when gate_status=WARN.")
    p.add_argument("--sample-mode", action="store_true", help="Use lighter defaults for smoke scenarios.")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--bundle-dir", default="")

    p.add_argument("--protocol-json", default="")
    p.add_argument("--repeatability-json", default="")
    p.add_argument("--metric-json", default="")
    p.add_argument("--significance-json", default="")
    p.add_argument("--stratified-json", default="")
    p.add_argument("--casebook-json", default="")
    p.add_argument("--repro-audit-json", default="")
    p.add_argument("--deployment-json", default="")
    p.add_argument("--report-template-json", default="")
    p.add_argument("--compliance-json", default="")
    p.add_argument("--run-missing", action="store_true", help="Try to generate missing report_template/compliance artifacts.")
    p.add_argument("--quality-sample-limit", type=int, default=80)
    p.add_argument("--skip-quality", action="store_true", help="Skip expensive quality scan and mark quality as WARN/skipped.")

    p.add_argument("--min-significance-conclusions", type=int, default=None)
    p.add_argument("--min-stratified-ok-rows", type=int, default=None)
    p.add_argument("--max-failure-ratio", type=float, default=None)
    p.add_argument("--min-repro-pass-ratio", type=float, default=None)
    p.add_argument("--min-recommended-cpu-cores", type=int, default=None)
    p.add_argument("--min-recommended-memory-gb", type=int, default=None)
    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_file(candidates: list[Path], pattern: str) -> Path | None:
    files: list[Path] = []
    for root in candidates:
        if root.exists():
            files.extend(sorted(root.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True))
    return files[0] if files else None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_path(cli_value: str, *, default_candidates: list[Path], pattern: str) -> Path | None:
    if cli_value.strip():
        p = Path(cli_value).resolve()
        return p if p.exists() else None
    return _latest_file(default_candidates, pattern)


def _quality_summary(
    *,
    settings,
    args: argparse.Namespace,
    output_roots: list[Path],
) -> dict[str, Any]:
    if args.skip_quality:
        return {
            "status": "warn",
            "gate_status": "WARN",
            "block_release": False,
            "message": "quality scan skipped by --skip-quality",
        }
    try:
        return build_data_quality_report(
            settings=settings,
            sample_limit=max(8, int(args.quality_sample_limit)),
        ).get("summary", {})
    except Exception as exc:
        # Fallback to latest cached quality report when runtime scan fails.
        latest_quality = _latest_file(output_roots + [settings.outputs_root / "qa"], "dataset_quality_*.json")
        if latest_quality is not None:
            cached = _load_json(latest_quality) or {}
            cached_summary = cached.get("summary", {}) if isinstance(cached, dict) else {}
            if isinstance(cached_summary, dict) and cached_summary:
                cached_summary = dict(cached_summary)
                cached_summary["message"] = f"loaded cached quality report due to runtime error: {exc.__class__.__name__}"
                cached_summary["cached_report"] = str(latest_quality)
                return cached_summary
        return {
            "status": "warn",
            "gate_status": "WARN",
            "block_release": False,
            "message": f"quality scan failed: {exc.__class__.__name__}",
        }


def _render_md(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", {}) if isinstance(payload.get("checks"), dict) else {}
    lines = [
        "# P4 Release Gate Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- gate_status: `{payload.get('status', 'FAIL')}`",
        f"- release_recommendation: `{payload.get('release_recommendation', False)}`",
        "",
        "## Stage Checks",
    ]
    for name, block in checks.items():
        if not isinstance(block, dict):
            continue
        lines.append(f"### {name}")
        lines.append(f"- status: `{block.get('status', 'FAIL')}`")
        for item in block.get("checks", []):
            lines.append(f"- `{item.get('name', '')}`: `{item.get('result', 'FAIL')}` ({item.get('message', '')})")
        lines.append("")
    summary = payload.get("stage_summary", {}) if isinstance(payload.get("stage_summary"), dict) else {}
    lines.append("## Decision")
    lines.append(f"- ready_for_public_demo: `{summary.get('ready_for_public_demo', False)}`")
    lines.append(f"- ready_for_paper_output: `{summary.get('ready_for_paper_output', False)}`")
    lines.append(f"- blocking_stages: `{summary.get('blocking_stages', [])}`")
    lines.append(f"- warn_stages: `{summary.get('warn_stages', [])}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _ensure_report_template_file(
    *,
    args: argparse.Namespace,
    settings,
    release_dir: Path,
) -> Path | None:
    if args.report_template_json.strip():
        p = Path(args.report_template_json).resolve()
        return p if p.exists() else None
    found = _latest_file([release_dir], "report_template_v1_*.json")
    if found is not None:
        return found
    if not args.run_missing:
        return None

    service = GalleryService()
    items = service.list_items()
    if not items:
        return None
    latest = items[0]
    risk_report = build_risk_report(latest)
    compliance = build_compliance_notices(
        settings=settings,
        context="export",
        timestamp=str(latest.get("timestamp", "")),
    )
    report = build_report_template(gallery_item=latest, risk_report=risk_report, compliance=compliance)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = release_dir / f"report_template_v1_{latest.get('id', 'latest')}_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _ensure_compliance_file(
    *,
    args: argparse.Namespace,
    settings,
    release_dir: Path,
    timestamp: str | None,
) -> Path | None:
    if args.compliance_json.strip():
        p = Path(args.compliance_json).resolve()
        return p if p.exists() else None
    found = _latest_file([release_dir], "compliance_notices_*.json")
    if found is not None:
        return found
    if not args.run_missing:
        return None
    payload = build_compliance_notices(settings=settings, context="export", timestamp=timestamp)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = release_dir / f"compliance_notices_{ts}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    settings = get_settings()
    release_dir = settings.outputs_root / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    output_roots = [release_dir, ROOT / "outputs" / "release", ROOT / "backend" / "outputs" / "release"]
    protocol_file = Path(args.protocol_json).resolve() if args.protocol_json else (release_dir / "p4_eval_protocol_v1.json")
    if not protocol_file.exists():
        protocol_file = _latest_file(output_roots, "p4_eval_protocol_v1*.json")

    repeatability_file = _resolve_path(args.repeatability_json, default_candidates=output_roots, pattern="p4_protocol_repeatability_*.json")
    metric_file = _resolve_path(args.metric_json, default_candidates=output_roots, pattern="p4_metric_completeness_*.json")
    significance_file = _resolve_path(args.significance_json, default_candidates=output_roots, pattern="p4_significance_*.json")
    stratified_file = _resolve_path(args.stratified_json, default_candidates=output_roots, pattern="p4_stratified_eval_*.json")
    casebook_file = _resolve_path(args.casebook_json, default_candidates=output_roots, pattern="failure_casebook_*.json")
    repro_file = _resolve_path(args.repro_audit_json, default_candidates=output_roots, pattern="p4_repro_audit_*.json")
    deployment_file = _resolve_path(args.deployment_json, default_candidates=output_roots, pattern="p4_deployment_profile_*.json")
    report_template_file = _ensure_report_template_file(args=args, settings=settings, release_dir=release_dir)
    report_template_payload = _load_json(report_template_file)
    report_ts = str(report_template_payload.get("timestamp", "")) if isinstance(report_template_payload, dict) else ""
    compliance_file = _ensure_compliance_file(args=args, settings=settings, release_dir=release_dir, timestamp=report_ts or None)

    protocol_payload = _load_json(protocol_file)
    repeatability_payload = _load_json(repeatability_file)
    metric_payload = _load_json(metric_file)
    significance_payload = _load_json(significance_file)
    stratified_payload = _load_json(stratified_file)
    casebook_payload = _load_json(casebook_file)
    repro_payload = _load_json(repro_file)
    deployment_payload = _load_json(deployment_file)
    compliance_payload = _load_json(compliance_file)

    thresholds = P4GateThresholds(
        min_significance_conclusions=int(args.min_significance_conclusions) if args.min_significance_conclusions is not None else 1,
        min_stratified_ok_rows=int(args.min_stratified_ok_rows) if args.min_stratified_ok_rows is not None else 1,
        max_failure_ratio=float(args.max_failure_ratio) if args.max_failure_ratio is not None else 0.60,
        min_repro_pass_ratio=float(args.min_repro_pass_ratio) if args.min_repro_pass_ratio is not None else (0.40 if args.sample_mode else 0.50),
        min_recommended_cpu_cores=int(args.min_recommended_cpu_cores) if args.min_recommended_cpu_cores is not None else 2,
        min_recommended_memory_gb=int(args.min_recommended_memory_gb) if args.min_recommended_memory_gb is not None else 4,
    )

    gate_report = build_p4_release_report(
        protocol_payload=protocol_payload,
        repeatability_payload=repeatability_payload,
        metric_payload=metric_payload,
        significance_payload=significance_payload,
        stratified_payload=stratified_payload,
        casebook_payload=casebook_payload,
        repro_payload=repro_payload,
        deployment_payload=deployment_payload,
        compliance_payload=compliance_payload,
        report_template_payload=report_template_payload,
        thresholds=thresholds,
    )

    ds = DatasetService()
    dataset_summary = ds.datasets_summary()
    quality_summary = _quality_summary(settings=settings, args=args, output_roots=output_roots)
    code_version = collect_code_version(settings.project_root)

    artifact_paths = {
        "protocol": str(protocol_file) if protocol_file else "",
        "repeatability": str(repeatability_file) if repeatability_file else "",
        "metric_validation": str(metric_file) if metric_file else "",
        "significance": str(significance_file) if significance_file else "",
        "stratified": str(stratified_file) if stratified_file else "",
        "failure_casebook": str(casebook_file) if casebook_file else "",
        "repro_audit": str(repro_file) if repro_file else "",
        "deployment_profile": str(deployment_file) if deployment_file else "",
        "report_template": str(report_template_file) if report_template_file else "",
        "compliance": str(compliance_file) if compliance_file else "",
    }
    artifact_paths = {k: v for k, v in artifact_paths.items() if v}

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (release_dir / f"p4_release_gate_{ts}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (release_dir / f"p4_release_gate_{ts}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    gate_report["inputs"] = {
        "artifact_paths": artifact_paths,
        "sample_mode": bool(args.sample_mode),
        "run_missing": bool(args.run_missing),
        "quality_sample_limit": int(args.quality_sample_limit),
        "skip_quality": bool(args.skip_quality),
    }
    gate_report["code_version"] = code_version
    gate_report["data_description"] = {
        "dataset": dataset_summary,
        "quality": quality_summary,
    }
    gate_report["generated_at"] = _utc_now_iso()
    out_json.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(gate_report), encoding="utf-8")

    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else (release_dir / f"p4_release_bundle_{ts}")
    manifest = build_release_bundle(
        bundle_dir=bundle_dir,
        gate_report=gate_report,
        artifact_paths=artifact_paths,
        code_version=code_version,
        data_description={"dataset": dataset_summary, "quality": quality_summary},
    )

    # Include gate report itself in bundle artifacts for direct audit traceability.
    gate_bundle_dst = bundle_dir / "artifacts" / out_json.name
    if out_json.resolve() != gate_bundle_dst.resolve():
        gate_bundle_dst.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    manifest["artifacts"]["p4_gate_report"] = str(gate_bundle_dst)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    gate_status = str(gate_report.get("status", "FAIL")).upper()
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"bundle_dir={bundle_dir}")
    print(f"gate_status={gate_status}")
    print(f"release_recommendation={gate_report.get('release_recommendation', False)}")

    if gate_status == "FAIL":
        raise SystemExit(2)
    if gate_status == "WARN" and not args.allow_warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
