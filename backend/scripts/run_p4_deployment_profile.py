from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import DatasetService
from app.core.p4_deployment_profile import (
    build_deployment_profile,
    deployment_profile_to_markdown,
    estimate_path_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P4 deployment resource profile (local/NAS/cloud).")
    parser.add_argument("--benchmark-json", default="", help="Benchmark json path; default latest planner_benchmark_*.json")
    parser.add_argument("--runtime-profile-json", default="", help="Runtime profile json path; optional")
    parser.add_argument("--out-dir", default="", help="Output directory; default outputs/release")
    parser.add_argument("--dataset-root", default="", help="Dataset root for size estimation; default NSR_DATA_ROOT")
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include outputs_root footprint into disk estimation.",
    )
    parser.add_argument("--baseline-cpu-cores", type=int, default=0, help="Baseline CPU cores for scaling model.")
    return parser.parse_args()


def _latest_file(candidates: list[Path], pattern: str) -> Path | None:
    files: list[Path] = []
    for root in candidates:
        if root.exists():
            files.extend(sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))
    return files[0] if files else None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ds = DatasetService()

    benchmark_file = (
        Path(args.benchmark_json).resolve()
        if args.benchmark_json
        else _latest_file(
            [settings.outputs_root / "benchmarks", ROOT / "outputs" / "benchmarks", ROOT / "backend" / "outputs" / "benchmarks"],
            "planner_benchmark_*.json",
        )
    )
    if benchmark_file is None or not benchmark_file.exists():
        raise SystemExit("No benchmark json found. Please run benchmark_planners.py first or pass --benchmark-json.")

    runtime_profile_file = (
        Path(args.runtime_profile_json).resolve()
        if args.runtime_profile_json
        else _latest_file(
            [settings.outputs_root / "benchmarks", ROOT / "outputs" / "benchmarks", ROOT / "backend" / "outputs" / "benchmarks"],
            "dynamic_runtime_profile_*.json",
        )
    )

    benchmark_payload = _load_json(benchmark_file)
    runtime_profile_payload = _load_json(runtime_profile_file)
    dataset_summary = ds.datasets_summary()

    data_root = Path(args.dataset_root).resolve() if args.dataset_root else settings.data_root
    dataset_bytes = estimate_path_bytes(data_root)
    if args.include_outputs:
        dataset_bytes += estimate_path_bytes(settings.outputs_root)

    baseline_cpu_cores = int(args.baseline_cpu_cores) if int(args.baseline_cpu_cores) > 0 else (os.cpu_count() or 4)
    profile = build_deployment_profile(
        benchmark_payload=benchmark_payload,
        runtime_profile_payload=runtime_profile_payload,
        dataset_summary=dataset_summary,
        dataset_bytes=dataset_bytes,
        baseline_cpu_cores=baseline_cpu_cores,
    )
    profile["inputs"]["benchmark_json"] = str(benchmark_file)
    profile["inputs"]["runtime_profile_json"] = str(runtime_profile_file) if runtime_profile_file else ""
    profile["inputs"]["dataset_root"] = str(data_root)
    profile["inputs"]["include_outputs"] = bool(args.include_outputs)

    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "release")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"p4_deployment_profile_{stamp}.json"
    md_path = out_dir / f"p4_deployment_profile_{stamp}.md"
    json_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(deployment_profile_to_markdown(profile), encoding="utf-8")

    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"benchmark_json={benchmark_file}")
    print(f"runtime_profile_json={runtime_profile_file if runtime_profile_file else ''}")
    print(f"dataset_footprint_gb={profile['inputs'].get('dataset_footprint_gb', 0)}")


if __name__ == "__main__":
    main()
