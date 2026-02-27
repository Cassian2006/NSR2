from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from fastapi.testclient import TestClient


DEFAULT_METRIC_SPECS: dict[str, dict[str, dict[str, float]]] = {
    "plan": {
        "distance_km": {"abs_tol": 0.5, "rel_tol": 0.01},
        "caution_len_km": {"abs_tol": 0.5, "rel_tol": 0.05},
        "route_points": {"abs_tol": 8.0, "rel_tol": 0.08},
    },
    "plan_dynamic": {
        "distance_km": {"abs_tol": 1.0, "rel_tol": 0.02},
        "caution_len_km": {"abs_tol": 1.0, "rel_tol": 0.1},
        "route_points": {"abs_tol": 12.0, "rel_tol": 0.1},
        "replan_count": {"abs_tol": 1.0, "rel_tol": 0.3},
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def validate_snapshot_binding(snapshot: dict[str, Any]) -> dict[str, Any]:
    replay = snapshot.get("replay")
    if not isinstance(replay, dict):
        return {"ok": False, "reason": "missing_replay_block"}
    endpoint = replay.get("endpoint")
    payload = replay.get("payload")
    if not isinstance(endpoint, str) or not endpoint.strip():
        return {"ok": False, "reason": "missing_replay_endpoint"}
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "missing_replay_payload"}
    return {
        "ok": True,
        "endpoint": endpoint,
        "payload_fingerprint": _payload_fingerprint(payload),
    }


def _baseline_metrics(snapshot: dict[str, Any]) -> dict[str, float]:
    result = snapshot.get("result", {})
    if not isinstance(result, dict):
        result = {}
    return {
        "distance_km": _to_float(result.get("distance_km", 0.0)),
        "caution_len_km": _to_float(result.get("caution_len_km", 0.0)),
        "route_points": _to_float(result.get("route_points", 0.0)),
        "replan_count": _to_float(result.get("replan_count", 0.0)),
    }


def _response_metrics(kind: str, response_json: dict[str, Any]) -> dict[str, float]:
    explain = response_json.get("explain", {})
    if not isinstance(explain, dict):
        explain = {}
    route_geojson = response_json.get("route_geojson", {})
    if not isinstance(route_geojson, dict):
        route_geojson = {}
    geometry = route_geojson.get("geometry", {})
    if not isinstance(geometry, dict):
        geometry = {}
    coords = geometry.get("coordinates", [])
    route_points = len(coords) if isinstance(coords, list) else 0

    metrics = {
        "distance_km": _to_float(explain.get("distance_km", 0.0)),
        "caution_len_km": _to_float(explain.get("caution_len_km", 0.0)),
        "route_points": float(route_points),
        "replan_count": float(len(explain.get("dynamic_replans", []))) if isinstance(explain.get("dynamic_replans"), list) else 0.0,
    }
    if kind == "plan_dynamic" and metrics["replan_count"] <= 0.0:
        metrics["replan_count"] = _to_float(explain.get("replan_count", 0.0))
    return metrics


def compare_metric_drift(
    *,
    baseline: float,
    replayed: float,
    abs_tol: float,
    rel_tol: float,
    fail_scale: float = 2.0,
) -> dict[str, Any]:
    abs_err = abs(replayed - baseline)
    denom = max(abs(baseline), 1e-9)
    rel_err = abs_err / denom

    pass_abs = abs_err <= abs_tol
    pass_rel = rel_err <= rel_tol
    if pass_abs and pass_rel:
        status = "PASS"
    elif abs_err <= abs_tol * fail_scale and rel_err <= rel_tol * fail_scale:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "baseline": baseline,
        "replayed": replayed,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "status": status,
    }


def audit_snapshot_replay(
    *,
    snapshot: dict[str, Any],
    client: TestClient,
    metric_specs: dict[str, dict[str, dict[str, float]]] | None = None,
    fail_scale: float = 2.0,
) -> dict[str, Any]:
    snap_id = str(snapshot.get("snapshot_id", ""))
    kind = str(snapshot.get("snapshot_kind", ""))
    binding = validate_snapshot_binding(snapshot)
    if not binding.get("ok"):
        return {
            "snapshot_id": snap_id,
            "snapshot_kind": kind,
            "status": "FAIL",
            "binding": binding,
            "error": "snapshot binding invalid",
        }

    replay = snapshot["replay"]
    endpoint = str(replay["endpoint"])
    payload = replay["payload"]
    method = str(replay.get("method", "POST")).upper()

    response = client.request(method, endpoint, json=payload)
    if response.status_code != 200:
        return {
            "snapshot_id": snap_id,
            "snapshot_kind": kind,
            "status": "FAIL",
            "binding": binding,
            "replay": {
                "method": method,
                "endpoint": endpoint,
                "http_status": int(response.status_code),
                "response_excerpt": response.text[:800],
            },
            "error": "replay request failed",
        }

    body = response.json()
    base_metrics = _baseline_metrics(snapshot)
    replay_metrics = _response_metrics(kind, body)
    specs = (metric_specs or DEFAULT_METRIC_SPECS).get(kind, DEFAULT_METRIC_SPECS["plan"])

    comparisons: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    for metric, tol in specs.items():
        if metric not in base_metrics or metric not in replay_metrics:
            continue
        comp = compare_metric_drift(
            baseline=_to_float(base_metrics[metric]),
            replayed=_to_float(replay_metrics[metric]),
            abs_tol=float(tol.get("abs_tol", 0.0)),
            rel_tol=float(tol.get("rel_tol", 0.0)),
            fail_scale=fail_scale,
        )
        comparisons[metric] = comp
        statuses.append(str(comp["status"]))

    if not comparisons:
        status = "WARN"
    elif any(s == "FAIL" for s in statuses):
        status = "FAIL"
    elif any(s == "WARN" for s in statuses):
        status = "WARN"
    else:
        status = "PASS"

    return {
        "snapshot_id": snap_id,
        "snapshot_kind": kind,
        "status": status,
        "binding": binding,
        "replay": {
            "method": method,
            "endpoint": endpoint,
            "http_status": int(response.status_code),
        },
        "metrics": comparisons,
        "baseline_result": base_metrics,
        "replayed_result": replay_metrics,
        "audited_at": _utc_now_iso(),
    }


def summarize_repro_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [str(item.get("status", "FAIL")).upper() for item in results]
    if any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif any(s == "WARN" for s in statuses):
        overall = "WARN"
    else:
        overall = "PASS"
    return {
        "overall_status": overall,
        "count": len(results),
        "pass_count": sum(1 for s in statuses if s == "PASS"),
        "warn_count": sum(1 for s in statuses if s == "WARN"),
        "fail_count": sum(1 for s in statuses if s == "FAIL"),
    }


def repro_audit_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    runs = payload.get("runs", []) if isinstance(payload.get("runs"), list) else []
    lines: list[str] = []
    lines.append("# P4 Reproducibility Audit")
    lines.append("")
    lines.append(f"- Generated At: {payload.get('generated_at', '')}")
    lines.append(f"- Overall Status: {summary.get('overall_status', 'FAIL')}")
    lines.append(f"- Runs: {summary.get('count', 0)}")
    lines.append(f"- PASS/WARN/FAIL: {summary.get('pass_count', 0)}/{summary.get('warn_count', 0)}/{summary.get('fail_count', 0)}")
    lines.append("")
    lines.append("## Run Details")
    lines.append("")
    for run in runs:
        lines.append(f"### {run.get('snapshot_id', '')} ({run.get('snapshot_kind', '')})")
        lines.append(f"- Status: {run.get('status', 'FAIL')}")
        replay = run.get("replay", {})
        if isinstance(replay, dict):
            lines.append(f"- Replay: {replay.get('method', 'POST')} {replay.get('endpoint', '')} (HTTP {replay.get('http_status', '-')})")
        metrics = run.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            lines.append("- Drift:")
            for metric, detail in metrics.items():
                if not isinstance(detail, dict):
                    continue
                lines.append(
                    f"  - {metric}: baseline={detail.get('baseline')} replayed={detail.get('replayed')} "
                    f"abs_err={detail.get('abs_err')} rel_err={detail.get('rel_err')} status={detail.get('status')}"
                )
        if run.get("error"):
            lines.append(f"- Error: {run.get('error')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
