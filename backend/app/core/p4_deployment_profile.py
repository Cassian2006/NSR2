from __future__ import annotations

import math
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


def bytes_to_gb(value: int) -> float:
    return float(value) / float(1024**3)


def estimate_path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except OSError:
                continue
    return total


def extract_runtime_baseline(
    *,
    benchmark_payload: dict[str, Any] | None,
    runtime_profile_payload: dict[str, Any] | None,
) -> dict[str, float]:
    summary = benchmark_payload.get("summary", {}) if isinstance(benchmark_payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    dynamic_dstar = summary.get("dynamic:dstar_lite", {}) if isinstance(summary.get("dynamic:dstar_lite"), dict) else {}
    static_dstar = summary.get("static:dstar_lite", {}) if isinstance(summary.get("static:dstar_lite"), dict) else {}
    static_astar = summary.get("static:astar", {}) if isinstance(summary.get("static:astar"), dict) else {}

    runtime_monitor = runtime_profile_payload.get("runtime_monitor", {}) if isinstance(runtime_profile_payload, dict) else {}
    if not isinstance(runtime_monitor, dict):
        runtime_monitor = {}

    static_runtime_ms = _safe_float(dynamic_dstar.get("avg_runtime_ms"), 0.0)
    if static_runtime_ms <= 0.0:
        static_runtime_ms = _safe_float(static_dstar.get("avg_runtime_ms"), 0.0)
    if static_runtime_ms <= 0.0:
        static_runtime_ms = _safe_float(static_astar.get("avg_runtime_ms"), 0.0)

    dynamic_runtime_ms = _safe_float(dynamic_dstar.get("avg_runtime_ms"), static_runtime_ms)
    replan_latency_ms = _safe_float(dynamic_dstar.get("avg_replan_latency_ms"), 0.0)
    if replan_latency_ms <= 0.0:
        replan_latency_ms = _safe_float(runtime_monitor.get("step_update_ms_mean"), 0.0)
    memory_peak_mb = _safe_float(runtime_monitor.get("memory_peak_mb"), 0.0)

    return {
        "static_runtime_ms": static_runtime_ms,
        "dynamic_runtime_ms": dynamic_runtime_ms,
        "replan_latency_ms": replan_latency_ms,
        "memory_peak_mb": memory_peak_mb,
    }


def _estimate_plan_capacity_per_hour(*, runtime_ms: float, cpu_cores: int, utilization: float = 0.7) -> float:
    if runtime_ms <= 1e-9 or cpu_cores <= 0:
        return 0.0
    core_seconds_per_hour = 3600.0 * float(cpu_cores) * max(0.2, min(1.0, utilization))
    one_plan_core_seconds = runtime_ms / 1000.0
    if one_plan_core_seconds <= 1e-9:
        return 0.0
    return core_seconds_per_hour / one_plan_core_seconds


def recommend_specs(
    *,
    baseline_cpu_cores: int,
    baseline: dict[str, float],
    dataset_bytes: int,
    target_static_ms: float = 15000.0,
    target_dynamic_ms: float = 30000.0,
    memory_safety_factor: float = 1.8,
) -> dict[str, Any]:
    base_cpu = max(2, int(baseline_cpu_cores))
    observed_static = max(1.0, _safe_float(baseline.get("static_runtime_ms"), 25000.0))
    observed_dynamic = max(1.0, _safe_float(baseline.get("dynamic_runtime_ms"), observed_static))

    static_needed = base_cpu * (observed_static / max(1000.0, target_static_ms))
    dynamic_needed = base_cpu * (observed_dynamic / max(1000.0, target_dynamic_ms))
    min_cpu = max(2, int(math.ceil(max(static_needed, dynamic_needed))))
    rec_cpu = max(min_cpu + 2, int(math.ceil(min_cpu * 1.5)))

    peak_mb = max(1024.0, _safe_float(baseline.get("memory_peak_mb"), 0.0))
    if peak_mb <= 1024.0:
        # If runtime monitor cannot capture RSS in this environment, use conservative fallback.
        peak_mb = 3072.0
    min_mem_gb = max(4, int(math.ceil((peak_mb * memory_safety_factor) / 1024.0)))
    rec_mem_gb = max(min_mem_gb + 2, int(math.ceil(min_mem_gb * 1.5)))

    dataset_gb = bytes_to_gb(dataset_bytes)
    min_disk_gb = max(40, int(math.ceil(dataset_gb * 1.8 + 20.0)))
    rec_disk_gb = max(min_disk_gb + 30, int(math.ceil(min_disk_gb * 1.6)))

    return {
        "minimal": {
            "cpu_cores": min_cpu,
            "memory_gb": min_mem_gb,
            "disk_gb": min_disk_gb,
            "estimated_plan_per_hour": round(_estimate_plan_capacity_per_hour(runtime_ms=observed_static, cpu_cores=min_cpu), 2),
            "estimated_dynamic_plan_per_hour": round(_estimate_plan_capacity_per_hour(runtime_ms=observed_dynamic, cpu_cores=min_cpu), 2),
        },
        "recommended": {
            "cpu_cores": rec_cpu,
            "memory_gb": rec_mem_gb,
            "disk_gb": rec_disk_gb,
            "estimated_plan_per_hour": round(_estimate_plan_capacity_per_hour(runtime_ms=observed_static, cpu_cores=rec_cpu), 2),
            "estimated_dynamic_plan_per_hour": round(_estimate_plan_capacity_per_hour(runtime_ms=observed_dynamic, cpu_cores=rec_cpu), 2),
        },
    }


def _estimate_monthly_costs(specs: dict[str, Any]) -> dict[str, dict[str, float]]:
    min_spec = specs.get("minimal", {})
    rec_spec = specs.get("recommended", {})

    def _local_cost(cpu: int, mem_gb: int, disk_gb: int) -> float:
        hardware_usd = 1200.0
        amortized = hardware_usd / 36.0
        power_watts = 35.0 + cpu * 4.0 + mem_gb * 1.2
        electricity = (power_watts / 1000.0) * 24.0 * 30.0 * 0.16
        disk_extra = max(0.0, disk_gb - 256.0) * 0.05
        return amortized + electricity + disk_extra

    def _nas_cost(cpu: int, mem_gb: int, disk_gb: int) -> float:
        hardware_usd = 900.0
        amortized = hardware_usd / 48.0
        power_watts = 28.0 + cpu * 3.0 + mem_gb * 0.8
        electricity = (power_watts / 1000.0) * 24.0 * 30.0 * 0.16
        disk_extra = max(0.0, disk_gb - 512.0) * 0.02
        return amortized + electricity + disk_extra

    def _cloud_cost(cpu: int, mem_gb: int, disk_gb: int) -> float:
        cpu_hour = 0.05
        mem_hour = 0.007
        storage_month = 0.12
        compute = (cpu * cpu_hour + mem_gb * mem_hour) * 24.0 * 30.0
        storage = disk_gb * storage_month
        return compute + storage

    out: dict[str, dict[str, float]] = {"local": {}, "nas": {}, "cloud": {}}
    for tier_name, spec in (("minimal", min_spec), ("recommended", rec_spec)):
        cpu = _safe_int(spec.get("cpu_cores"), 2)
        mem = _safe_int(spec.get("memory_gb"), 4)
        disk = _safe_int(spec.get("disk_gb"), 40)
        out["local"][tier_name] = round(_local_cost(cpu, mem, disk), 2)
        out["nas"][tier_name] = round(_nas_cost(cpu, mem, disk), 2)
        out["cloud"][tier_name] = round(_cloud_cost(cpu, mem, disk), 2)
    return out


def build_deployment_profile(
    *,
    benchmark_payload: dict[str, Any] | None,
    runtime_profile_payload: dict[str, Any] | None,
    dataset_summary: dict[str, Any] | None,
    dataset_bytes: int,
    baseline_cpu_cores: int,
) -> dict[str, Any]:
    baseline = extract_runtime_baseline(
        benchmark_payload=benchmark_payload,
        runtime_profile_payload=runtime_profile_payload,
    )
    specs = recommend_specs(
        baseline_cpu_cores=baseline_cpu_cores,
        baseline=baseline,
        dataset_bytes=dataset_bytes,
    )
    costs = _estimate_monthly_costs(specs)

    sample_count = _safe_int((dataset_summary or {}).get("sample_count"), 0)
    months = (dataset_summary or {}).get("months", [])
    if not isinstance(months, list):
        months = []

    return {
        "profile_version": "p4_deployment_profile_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "sample_count": sample_count,
            "months": months,
            "dataset_footprint_gb": round(bytes_to_gb(dataset_bytes), 3),
            "benchmark_created_at": str((benchmark_payload or {}).get("created_at", "")),
            "runtime_profile_status": str((runtime_profile_payload or {}).get("status", "")),
            "baseline_cpu_cores": int(max(1, baseline_cpu_cores)),
        },
        "observed_runtime": baseline,
        "resource_recommendation": specs,
        "monthly_cost_estimate_usd": costs,
        "environment_notes": {
            "local": [
                "适合研发与快速迭代，网络依赖最低。",
                "硬件扩容周期较长，长期并发能力受限。",
            ],
            "nas": [
                "适合实验室内持续运行，数据落盘方便。",
                "需关注 Docker 资源限制与磁盘 I/O 瓶颈。",
            ],
            "cloud": [
                "弹性扩展和公网访问最方便。",
                "长期运行成本较高，需控制存储与实例规格。",
            ],
        },
        "deployment_recommendation": {
            "minimum_viable": {
                "cpu_cores": specs["minimal"]["cpu_cores"],
                "memory_gb": specs["minimal"]["memory_gb"],
                "disk_gb": specs["minimal"]["disk_gb"],
            },
            "recommended": {
                "cpu_cores": specs["recommended"]["cpu_cores"],
                "memory_gb": specs["recommended"]["memory_gb"],
                "disk_gb": specs["recommended"]["disk_gb"],
            },
        },
    }


def deployment_profile_to_markdown(profile: dict[str, Any]) -> str:
    inputs = profile.get("inputs", {}) if isinstance(profile.get("inputs"), dict) else {}
    runtime = profile.get("observed_runtime", {}) if isinstance(profile.get("observed_runtime"), dict) else {}
    rec = profile.get("resource_recommendation", {}) if isinstance(profile.get("resource_recommendation"), dict) else {}
    costs = profile.get("monthly_cost_estimate_usd", {}) if isinstance(profile.get("monthly_cost_estimate_usd"), dict) else {}
    lines: list[str] = []
    lines.append("# P4 Deployment Profile")
    lines.append("")
    lines.append(f"- profile_version: `{profile.get('profile_version', '')}`")
    lines.append(f"- generated_at: `{profile.get('generated_at', '')}`")
    lines.append(f"- sample_count: `{inputs.get('sample_count', 0)}`")
    lines.append(f"- dataset_footprint_gb: `{inputs.get('dataset_footprint_gb', 0)}`")
    lines.append(f"- benchmark_created_at: `{inputs.get('benchmark_created_at', '')}`")
    lines.append("")
    lines.append("## Observed Runtime")
    for key in ["static_runtime_ms", "dynamic_runtime_ms", "replan_latency_ms", "memory_peak_mb"]:
        lines.append(f"- {key}: `{runtime.get(key, 0)}`")
    lines.append("")
    lines.append("## Resource Recommendation")
    for tier in ["minimal", "recommended"]:
        tier_data = rec.get(tier, {})
        lines.append(f"### {tier}")
        lines.append(f"- cpu_cores: `{tier_data.get('cpu_cores', 0)}`")
        lines.append(f"- memory_gb: `{tier_data.get('memory_gb', 0)}`")
        lines.append(f"- disk_gb: `{tier_data.get('disk_gb', 0)}`")
        lines.append(f"- estimated_plan_per_hour: `{tier_data.get('estimated_plan_per_hour', 0)}`")
        lines.append(f"- estimated_dynamic_plan_per_hour: `{tier_data.get('estimated_dynamic_plan_per_hour', 0)}`")
        lines.append("")
    lines.append("## Monthly Cost Estimate (USD)")
    for env in ["local", "nas", "cloud"]:
        env_cost = costs.get(env, {})
        lines.append(f"- {env}: minimal=`{env_cost.get('minimal', 0)}`, recommended=`{env_cost.get('recommended', 0)}`")
    lines.append("")
    lines.append("## Deployment Guidance")
    rec_block = profile.get("deployment_recommendation", {})
    if isinstance(rec_block, dict):
        minv = rec_block.get("minimum_viable", {})
        recm = rec_block.get("recommended", {})
        lines.append(
            f"- minimum_viable: cpu={minv.get('cpu_cores', 0)}, mem={minv.get('memory_gb', 0)}GB, disk={minv.get('disk_gb', 0)}GB"
        )
        lines.append(
            f"- recommended: cpu={recm.get('cpu_cores', 0)}, mem={recm.get('memory_gb', 0)}GB, disk={recm.get('disk_gb', 0)}GB"
        )
    return "\n".join(lines).strip() + "\n"
