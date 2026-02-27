from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import get_dataset_service
from app.planning.router import PlanningError, plan_grid_route, plan_grid_route_dynamic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay one failure/degradation case from P4 casebook.")
    p.add_argument("--casebook-json", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--dry-run", action="store_true", help="Only print resolved replay payload.")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def _resolve_dynamic_timestamps(window_start: str, window_end: str) -> list[str]:
    all_ts = get_dataset_service().list_timestamps(month="all")
    if not all_ts:
        return []
    try:
        i0 = all_ts.index(window_start)
        i1 = all_ts.index(window_end)
    except ValueError:
        return []
    if i1 < i0:
        i0, i1 = i1, i0
    return all_ts[i0 : i1 + 1]


def main() -> None:
    args = parse_args()
    casebook_path = Path(args.casebook_json).resolve()
    payload = json.loads(casebook_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    target = next((x for x in cases if str(x.get("case_id", "")) == str(args.case_id)), None)
    if not isinstance(target, dict):
        raise SystemExit(f"Case not found: {args.case_id}")

    replay_hint = target.get("replay_hint", {}) if isinstance(target.get("replay_hint"), dict) else {}
    scenario_ref = replay_hint.get("scenario_ref", {}) if isinstance(replay_hint.get("scenario_ref"), dict) else {}
    policy = replay_hint.get("policy_defaults", {}) if isinstance(replay_hint.get("policy_defaults"), dict) else {}
    start = replay_hint.get("start", {}) if isinstance(replay_hint.get("start"), dict) else {}
    goal = replay_hint.get("goal", {}) if isinstance(replay_hint.get("goal"), dict) else {}

    mode = str(scenario_ref.get("mode", "static")).lower()
    replay_payload: dict[str, object] = {
        "mode": mode,
        "planner": str(policy.get("planner", target.get("planner", "astar"))),
        "start": {"lat": float(start.get("lat", 70.5)), "lon": float(start.get("lon", 30.0))},
        "goal": {"lat": float(goal.get("lat", 72.0)), "lon": float(goal.get("lon", 150.0))},
        "policy": policy,
    }

    if mode == "dynamic":
        ts = _resolve_dynamic_timestamps(str(scenario_ref.get("window_start", "")), str(scenario_ref.get("window_end", "")))
        replay_payload["timestamps"] = ts
        replay_payload["window_start"] = str(scenario_ref.get("window_start", ""))
        replay_payload["window_end"] = str(scenario_ref.get("window_end", ""))
    else:
        replay_payload["timestamp"] = str(scenario_ref.get("timestamp", ""))

    if args.dry_run:
        print(json.dumps({"case_id": args.case_id, "replay_payload": replay_payload}, ensure_ascii=False, indent=2))
        return

    settings = get_settings()
    result_obj: dict[str, object]
    try:
        if mode == "dynamic":
            timestamps = replay_payload.get("timestamps", [])
            if not isinstance(timestamps, list) or len(timestamps) < 2:
                raise SystemExit("Cannot replay dynamic case: resolved timestamps are insufficient.")
            result = plan_grid_route_dynamic(
                settings=settings,
                timestamps=[str(x) for x in timestamps],
                start=(float(replay_payload["start"]["lat"]), float(replay_payload["start"]["lon"])),  # type: ignore[index]
                goal=(float(replay_payload["goal"]["lat"]), float(replay_payload["goal"]["lon"])),  # type: ignore[index]
                model_version="unet_v1",
                corridor_bias=float(policy.get("corridor_bias", 0.2)),
                caution_mode=str(policy.get("caution_mode", "tie_breaker")),
                smoothing=bool(policy.get("smoothing", True)),
                blocked_sources=list(policy.get("blocked_sources", ["bathy", "unet_blocked"])),
                planner=str(policy.get("planner", "astar")),
                advance_steps=8,
            )
            explain = dict(result.explain)
            result_obj = {
                "status": "ok",
                "distance_km": float(explain.get("distance_km", 0.0)),
                "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                "risk_exposure": float(explain.get("route_cost_risk_extra_km", 0.0)),
                "replan_count": len(explain.get("dynamic_replans", [])),
            }
        else:
            timestamp = str(replay_payload.get("timestamp", ""))
            if not timestamp:
                raise SystemExit("Cannot replay static case: missing timestamp.")
            result = plan_grid_route(
                settings=settings,
                timestamp=timestamp,
                start=(float(replay_payload["start"]["lat"]), float(replay_payload["start"]["lon"])),  # type: ignore[index]
                goal=(float(replay_payload["goal"]["lat"]), float(replay_payload["goal"]["lon"])),  # type: ignore[index]
                model_version="unet_v1",
                corridor_bias=float(policy.get("corridor_bias", 0.2)),
                caution_mode=str(policy.get("caution_mode", "tie_breaker")),
                smoothing=bool(policy.get("smoothing", True)),
                blocked_sources=list(policy.get("blocked_sources", ["bathy", "unet_blocked"])),
                planner=str(policy.get("planner", "astar")),
            )
            explain = dict(result.explain)
            result_obj = {
                "status": "ok",
                "distance_km": float(explain.get("distance_km", 0.0)),
                "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                "risk_exposure": float(explain.get("route_cost_risk_extra_km", 0.0)),
            }
    except PlanningError as exc:
        result_obj = {"status": "fail", "error": str(exc)}

    output = {
        "case_id": args.case_id,
        "replay_payload": replay_payload,
        "result": result_obj,
    }
    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json={out}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

