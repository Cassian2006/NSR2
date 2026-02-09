from __future__ import annotations

import math
from dataclasses import dataclass


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass(frozen=True)
class PlanResult:
    route_geojson: dict
    explain: dict


def _interpolate(start: tuple[float, float], goal: tuple[float, float], steps: int) -> list[list[float]]:
    s_lat, s_lon = start
    g_lat, g_lon = goal
    coords: list[list[float]] = []
    for idx in range(steps + 1):
        t = idx / steps
        lat = s_lat + (g_lat - s_lat) * t
        lon = s_lon + (g_lon - s_lon) * t
        coords.append([lon, lat])
    return coords


def plan_simple_route(
    start: tuple[float, float],
    goal: tuple[float, float],
    corridor_bias: float,
    caution_mode: str,
    smoothing: bool,
) -> PlanResult:
    distance = haversine_km(start[0], start[1], goal[0], goal[1])
    steps = max(16, min(96, int(distance / 20)))
    coordinates = _interpolate(start, goal, steps)

    corridor_alignment = max(0.0, min(1.0, 0.45 + 0.9 * corridor_bias))
    caution_len = round(distance * 0.08, 3)
    explain = {
        "distance_km": round(distance, 3),
        "distance_nm": round(distance * 0.539957, 3),
        "caution_len_km": caution_len,
        "corridor_alignment": round(corridor_alignment, 3),
        "caution_mode": caution_mode,
        "smoothing": smoothing,
        "note": "Skeleton planner: straight-line baseline route for integration testing.",
    }

    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": explain,
    }
    return PlanResult(route_geojson=route_geojson, explain=explain)

