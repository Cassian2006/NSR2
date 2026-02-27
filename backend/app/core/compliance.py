from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from app.core.config import Settings
from app.core.latest import get_latest_meta
from app.core.latest_source_health import get_source_health_snapshot


ComplianceContext = Literal["workspace", "export"]


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _freshness_label(age_hours: float | None) -> str:
    if age_hours is None:
        return "unknown"
    if age_hours <= 24:
        return "fresh"
    if age_hours <= 72:
        return "stale"
    return "outdated"


def _build_data_freshness(settings: Settings, timestamp: str | None) -> dict[str, Any]:
    if not timestamp:
        return {
            "timestamp": "",
            "source": "unknown",
            "materialized_at": None,
            "age_hours": None,
            "status": "unknown",
            "hint": {
                "en": "No timestamp selected, freshness cannot be evaluated.",
                "zh": "未选择时间片，无法评估数据时效性。",
            },
        }

    meta = get_latest_meta(settings=settings, timestamp=timestamp)
    source = str(meta.get("source", "local_dataset"))
    materialized_at_raw = meta.get("materialized_at")
    materialized_at = str(materialized_at_raw) if materialized_at_raw else None
    materialized_dt = _parse_iso_datetime(materialized_at)
    age_hours = None
    if materialized_dt is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - materialized_dt).total_seconds() / 3600.0)
    status = _freshness_label(age_hours)

    status_hint_map = {
        "fresh": {
            "en": "Fresh data window. Still verify with official marine bulletins before navigation decisions.",
            "zh": "数据时效较新。用于航行决策前仍需核对官方海事通告。",
        },
        "stale": {
            "en": "Data may be stale for operational planning. Treat route result as exploratory only.",
            "zh": "数据可能已过时，不适合直接用于作业规划。请仅作探索性参考。",
        },
        "outdated": {
            "en": "Data is outdated. Do not use this result for operational routing.",
            "zh": "数据已明显过时。请勿将结果用于实际航线决策。",
        },
        "unknown": {
            "en": "Freshness metadata unavailable. Treat confidence as reduced.",
            "zh": "缺少时效元数据，可信度下降。",
        },
    }

    return {
        "timestamp": timestamp,
        "source": source,
        "materialized_at": materialized_at,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "status": status,
        "hint": status_hint_map.get(status, status_hint_map["unknown"]),
    }


def _build_source_credibility() -> dict[str, Any]:
    snapshot = get_source_health_snapshot()
    sources_raw = snapshot.get("sources")
    sources = sources_raw if isinstance(sources_raw, dict) else {}

    healthy = 0
    degraded = 0
    blocked = 0
    for _, source_state in sources.items():
        state = source_state if isinstance(source_state, dict) else {}
        if bool(state.get("circuit_open", False)):
            blocked += 1
        elif int(state.get("failure_count", 0)) > 0:
            degraded += 1
        else:
            healthy += 1

    if blocked > 0:
        level = "high_risk"
    elif degraded > 0:
        level = "medium_risk"
    else:
        level = "normal"

    level_hint = {
        "normal": {
            "en": "Source health is stable.",
            "zh": "数据源健康状态稳定。",
        },
        "medium_risk": {
            "en": "Some source failures detected. Validate critical route decisions manually.",
            "zh": "存在部分数据源失败，请人工复核关键决策。",
        },
        "high_risk": {
            "en": "Source circuit breaker active. Latest chain may be degraded.",
            "zh": "数据源熔断已触发，latest 链路可能降级。",
        },
    }

    return {
        "level": level,
        "summary": {
            "healthy": healthy,
            "degraded": degraded,
            "blocked": blocked,
        },
        "sources": sources,
        "updated_at": snapshot.get("updated_at"),
        "hint": level_hint[level],
    }


def build_compliance_notices(*, settings: Settings, context: ComplianceContext, timestamp: str | None = None) -> dict[str, Any]:
    notices = [
        {
            "id": "research_only",
            "severity": "high",
            "messages": {
                "en": "Research and educational use only. This system does not replace certified marine navigation systems.",
                "zh": "仅用于科研与教学。本系统不能替代经认证的航海导航系统。",
            },
        },
        {
            "id": "non_navigation_instruction",
            "severity": "high",
            "messages": {
                "en": "Output is not a navigation instruction. Final decisions must follow captain judgment and official regulations.",
                "zh": "输出不构成航行指令。最终决策需由船长判断并遵循官方规范。",
            },
        },
        {
            "id": "data_freshness_required",
            "severity": "medium",
            "messages": {
                "en": "Check data freshness and source health before interpreting route confidence.",
                "zh": "解读结果前请先检查数据时效与数据源健康状态。",
            },
        },
    ]

    return {
        "version": "compliance_v1",
        "context": context,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notices": notices,
        "data_freshness": _build_data_freshness(settings=settings, timestamp=timestamp),
        "source_credibility": _build_source_credibility(),
    }
