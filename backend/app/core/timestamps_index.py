from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


TS_FMT = "%Y-%m-%d_%H"


@dataclass(frozen=True)
class GapSegment:
    start: str
    end: str
    missing_count: int
    missing_hours: int

    def to_json(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "missing_count": self.missing_count,
            "missing_hours": self.missing_hours,
        }


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, TS_FMT).replace(tzinfo=timezone.utc)


def format_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(TS_FMT)


def infer_native_step_hours(timestamps: list[str]) -> int:
    if len(timestamps) < 2:
        return 1
    dt = sorted(parse_ts(ts) for ts in timestamps)
    deltas = []
    for i in range(1, len(dt)):
        delta_h = int((dt[i] - dt[i - 1]).total_seconds() // 3600)
        if delta_h > 0:
            deltas.append(delta_h)
    if not deltas:
        return 1
    deltas.sort()
    return int(deltas[len(deltas) // 2])


def _build_expected_axis(start: datetime, end: datetime, step_hours: int) -> list[datetime]:
    out: list[datetime] = []
    t = start
    step = timedelta(hours=step_hours)
    while t <= end:
        out.append(t)
        t += step
    return out


def _collect_gap_segments(missing: list[str], step_hours: int) -> list[GapSegment]:
    if not missing:
        return []
    missing_dt = sorted(parse_ts(ts) for ts in missing)
    gaps: list[GapSegment] = []

    block_start = missing_dt[0]
    block_prev = missing_dt[0]
    expected_step = timedelta(hours=step_hours)
    for cur in missing_dt[1:]:
        if cur - block_prev == expected_step:
            block_prev = cur
            continue
        count = int((block_prev - block_start).total_seconds() // 3600 // step_hours) + 1
        gaps.append(
            GapSegment(
                start=format_ts(block_start),
                end=format_ts(block_prev),
                missing_count=count,
                missing_hours=count * step_hours,
            )
        )
        block_start = cur
        block_prev = cur

    count = int((block_prev - block_start).total_seconds() // 3600 // step_hours) + 1
    gaps.append(
        GapSegment(
            start=format_ts(block_start),
            end=format_ts(block_prev),
            missing_count=count,
            missing_hours=count * step_hours,
        )
    )
    return gaps


def build_timestamps_index(
    *,
    source_timestamps: list[str],
    step_hours: int = 1,
    source_kind: str = "annotation_pack",
) -> dict[str, Any]:
    step_hours = max(1, int(step_hours))
    clean = sorted({ts for ts in source_timestamps})
    if not clean:
        return {
            "summary": {
                "status": "fail",
                "message": "No source timestamps.",
                "step_hours": step_hours,
                "source_count": 0,
                "expected_count": 0,
                "available_count": 0,
                "missing_count": 0,
                "coverage": 0.0,
                "native_step_hours": 1,
            },
            "timestamps": [],
            "missing_timestamps": [],
            "gaps": [],
            "months": [],
            "source_kind": source_kind,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    source_dt = sorted(parse_ts(ts) for ts in clean)
    start = source_dt[0]
    end = source_dt[-1]
    expected_axis = _build_expected_axis(start=start, end=end, step_hours=step_hours)
    expected_ts = [format_ts(t) for t in expected_axis]

    source_set = set(clean)
    available = [ts for ts in expected_ts if ts in source_set]
    missing = [ts for ts in expected_ts if ts not in source_set]
    gaps = _collect_gap_segments(missing=missing, step_hours=step_hours)

    expected_count = len(expected_ts)
    available_count = len(available)
    coverage = float(available_count / expected_count) if expected_count > 0 else 0.0
    status = "pass"
    if expected_count == 0:
        status = "fail"
    elif coverage < 0.95:
        status = "warn"
    if coverage < 0.75:
        status = "fail"

    return {
        "summary": {
            "status": status,
            "step_hours": step_hours,
            "source_count": len(clean),
            "expected_count": expected_count,
            "available_count": available_count,
            "missing_count": len(missing),
            "coverage": round(coverage, 6),
            "first_timestamp": clean[0],
            "last_timestamp": clean[-1],
            "native_step_hours": infer_native_step_hours(clean),
        },
        "timestamps": available,
        "missing_timestamps": missing,
        "gaps": [g.to_json() for g in gaps],
        "months": sorted({ts[:7] for ts in available}),
        "source_kind": source_kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
