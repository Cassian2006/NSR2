from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings
from app.core.latest import get_latest_meta


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _age_hours(iso_ts: str | None) -> float | None:
    ts = _parse_iso(iso_ts)
    if ts is None:
        return None
    return max(0.0, (_now() - ts).total_seconds() / 3600.0)


def _status_by_age(age: float | None, warn_hours: float, hard_hours: float) -> str:
    if age is None:
        return "unknown"
    if age <= warn_hours:
        return "fresh"
    if age <= hard_hours:
        return "stale"
    return "outdated"


def _degrade_actions(status: str) -> list[str]:
    if status == "fresh":
        return []
    if status == "stale":
        return ["switch_conservative", "raise_risk_weight"]
    if status == "outdated":
        return ["switch_conservative", "freeze_dynamic_replan", "mark_result_low_confidence"]
    return ["mark_result_unknown_freshness"]


def build_latest_slo_snapshot(*, settings: Settings, timestamp: str | None = None) -> dict[str, Any]:
    ice_warn = float(settings.latest_slo_ice_warn_hours)
    ice_hard = float(settings.latest_slo_ice_hard_hours)
    wave_warn = float(settings.latest_slo_wave_warn_hours)
    wave_hard = float(settings.latest_slo_wave_hard_hours)
    wind_warn = float(settings.latest_slo_wind_warn_hours)
    wind_hard = float(settings.latest_slo_wind_hard_hours)
    ais_warn = float(settings.latest_slo_ais_warn_hours)
    ais_hard = float(settings.latest_slo_ais_hard_hours)

    latest_meta = get_latest_meta(settings, timestamp) if timestamp else {}
    materialized_at = str(latest_meta.get("materialized_at", "")).strip() if isinstance(latest_meta, dict) else ""
    age = _age_hours(materialized_at or None)

    layers = {
        "ice": {
            "warn_hours": ice_warn,
            "hard_hours": ice_hard,
            "age_hours": age,
            "status": _status_by_age(age, ice_warn, ice_hard),
        },
        "wave": {
            "warn_hours": wave_warn,
            "hard_hours": wave_hard,
            "age_hours": age,
            "status": _status_by_age(age, wave_warn, wave_hard),
        },
        "wind": {
            "warn_hours": wind_warn,
            "hard_hours": wind_hard,
            "age_hours": age,
            "status": _status_by_age(age, wind_warn, wind_hard),
        },
        "ais": {
            "warn_hours": ais_warn,
            "hard_hours": ais_hard,
            "age_hours": None,
            "status": "unknown",
            "note": "AIS SLO threshold configured; runtime timestamp source not yet attached.",
        },
    }

    status_order = {"fresh": 0, "stale": 1, "outdated": 2, "unknown": 3}
    overall = max((v["status"] for v in layers.values()), key=lambda s: status_order.get(str(s), 99))

    return {
        "version": "latest_slo_v1",
        "timestamp": timestamp or "",
        "policy": {
            "warn_to_mode": settings.latest_slo_warn_mode,
            "hard_to_mode": settings.latest_slo_hard_mode,
            "warn_actions": _degrade_actions("stale"),
            "hard_actions": _degrade_actions("outdated"),
        },
        "overall_status": overall,
        "layers": layers,
        "source_meta_hint": {
            "materialized_at": materialized_at or None,
            "source": latest_meta.get("source") if isinstance(latest_meta, dict) else None,
        },
    }
