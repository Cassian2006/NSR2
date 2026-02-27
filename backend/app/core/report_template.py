from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_report_template(
    *,
    gallery_item: dict[str, Any],
    risk_report: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    explain = gallery_item.get("explain", {}) if isinstance(gallery_item.get("explain"), dict) else {}
    action = gallery_item.get("action", {}) if isinstance(gallery_item.get("action"), dict) else {}
    policy = action.get("policy", {}) if isinstance(action.get("policy"), dict) else {}
    risk = risk_report.get("risk", {}) if isinstance(risk_report, dict) and isinstance(risk_report.get("risk"), dict) else {}
    candidate_cmp = (
        risk_report.get("candidate_comparison", {})
        if isinstance(risk_report, dict) and isinstance(risk_report.get("candidate_comparison"), dict)
        else {}
    )

    distance_km = _to_float(explain.get("distance_km", gallery_item.get("distance_km", 0.0)))
    caution_len_km = _to_float(explain.get("caution_len_km", gallery_item.get("caution_len_km", 0.0)))
    caution_ratio = (caution_len_km / distance_km) if distance_km > 1e-9 else 0.0

    limitations = [
        "Research-use output only; not a certified navigation instruction.",
        "Route quality depends on data freshness, model uncertainty, and source availability.",
        "Final operational decisions must follow captain judgment and official maritime regulations.",
    ]

    return {
        "template_version": "report_template_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gallery_id": str(gallery_item.get("id", "")),
        "timestamp": str(gallery_item.get("timestamp", "")),
        "method": {
            "objective": str(policy.get("objective", "shortest_distance_under_safety")),
            "planner": str(explain.get("planner", policy.get("planner", "astar"))),
            "blocked_sources": policy.get("blocked_sources", []),
            "caution_mode": str(explain.get("caution_mode", policy.get("caution_mode", "tie_breaker"))),
            "smoothing": bool(policy.get("smoothing", explain.get("smoothing", True))),
            "corridor_bias": _to_float(policy.get("corridor_bias", gallery_item.get("corridor_bias", 0.0))),
            "risk_mode": str(explain.get("risk_mode", policy.get("risk_mode", "balanced"))),
            "risk_constraint_mode": str(explain.get("risk_constraint_mode", policy.get("risk_constraint_mode", "none"))),
        },
        "data": {
            "dataset_timestamp": str(gallery_item.get("timestamp", "")),
            "start": gallery_item.get("start", {}),
            "goal": gallery_item.get("goal", {}),
            "model_version": str(gallery_item.get("model_version", "unet_v1")),
            "version_snapshot": gallery_item.get("version_snapshot"),
            "freshness": compliance.get("data_freshness") if isinstance(compliance, dict) else None,
            "source_credibility": compliance.get("source_credibility") if isinstance(compliance, dict) else None,
        },
        "results": {
            "distance_km": round(distance_km, 3),
            "distance_nm": round(_to_float(explain.get("distance_nm", 0.0)), 3),
            "caution_len_km": round(caution_len_km, 3),
            "caution_ratio": round(caution_ratio, 6),
            "corridor_alignment": round(_to_float(explain.get("corridor_alignment", 0.0)), 6),
            "route_points": int(_to_float(explain.get("smoothed_points", explain.get("route_points", 0)), 0)),
            "gallery_id": str(gallery_item.get("id", "")),
        },
        "statistics": {
            "risk_exposure": round(_to_float(risk.get("risk_exposure", explain.get("route_cost_risk_extra_km", 0.0))), 6),
            "high_risk_crossing_ratio": round(_to_float(risk.get("high_risk_crossing_ratio", explain.get("caution_cell_ratio", 0.0))), 6),
            "candidate_count": int(_to_float(candidate_cmp.get("count", 0), 0)),
            "candidate_ok_count": int(_to_float(candidate_cmp.get("ok_count", 0), 0)),
            "pareto_summary": candidate_cmp.get("pareto_summary"),
            "risk_constraint_satisfied": bool(explain.get("risk_constraint_satisfied", True)),
        },
        "limitations": limitations,
        "reproducibility": {
            "policy": policy,
            "explain_keys": sorted(list(explain.keys())),
            "route_geojson_available": isinstance(gallery_item.get("route_geojson"), dict),
            "artifact_refs": {
                "gallery_id": str(gallery_item.get("id", "")),
                "timestamp": str(gallery_item.get("timestamp", "")),
                "model_version": str(gallery_item.get("model_version", "unet_v1")),
            },
        },
        "compliance": compliance,
    }


def _flatten_rows(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_rows(child, child_prefix))
        return rows
    if isinstance(value, list):
        if not value:
            rows.append((prefix, "[]"))
            return rows
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            rows.extend(_flatten_rows(child, child_prefix))
        return rows
    rows.append((prefix, _safe_str(value)))
    return rows


def report_template_to_csv(report: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["section", "field", "value"])
    for section in ["method", "data", "results", "statistics", "limitations", "reproducibility"]:
        section_data = report.get(section)
        for field, value in _flatten_rows(section_data):
            writer.writerow([section, field, value])
    return output.getvalue()


def report_template_to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Report Template {report.get('template_version', 'report_template_v1')}")
    lines.append("")
    lines.append(f"- Gallery ID: {report.get('gallery_id', '')}")
    lines.append(f"- Timestamp: {report.get('timestamp', '')}")
    lines.append(f"- Generated At: {report.get('generated_at', '')}")
    lines.append("")

    section_titles = {
        "method": "方法",
        "data": "数据",
        "results": "结果",
        "statistics": "统计",
        "limitations": "局限",
        "reproducibility": "复现",
    }

    for key in ["method", "data", "results", "statistics", "limitations", "reproducibility"]:
        lines.append(f"## {section_titles[key]}")
        value = report.get(key)
        if isinstance(value, dict):
            for field, field_val in _flatten_rows(value):
                lines.append(f"- {field}: {_safe_str(field_val)}")
        elif isinstance(value, list):
            for item in value:
                lines.append(f"- {_safe_str(item)}")
        else:
            lines.append(f"- {_safe_str(value)}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
