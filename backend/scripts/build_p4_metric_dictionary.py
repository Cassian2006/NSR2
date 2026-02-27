from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.p4_metrics import build_benchmark_metric_bundle, metric_dictionary_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build P4 metric dictionary and optional benchmark-linked examples.")
    p.add_argument("--benchmark-json", default="", help="Optional benchmark json. Defaults to latest outputs/benchmarks/planner_benchmark_*.json.")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    return p.parse_args()


def _latest_benchmark_file(out_root: Path) -> Path | None:
    candidates = [
        out_root / "benchmarks",
        ROOT / "outputs" / "benchmarks",
    ]
    files: list[Path] = []
    for c in candidates:
        if c.exists():
            files.extend(sorted(c.glob("planner_benchmark_*.json")))
    if not files:
        return None
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return files[-1]


def _to_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# P4 Metric Dictionary")
    lines.append("")
    lines.append(f"- dictionary_version: `{payload.get('dictionary_version', '')}`")
    lines.append(f"- generated_at: `{payload.get('generated_at', '')}`")
    lines.append(f"- benchmark_source: `{payload.get('benchmark_source', '')}`")
    lines.append("")
    lines.append("## Core Metrics")
    for item in payload.get("dictionary", {}).get("core_metrics", []):
        lines.append(
            "- `{id}` | {title} | unit={unit} | dir={direction} | formula=`{formula}`".format(
                id=item.get("id", ""),
                title=item.get("title", ""),
                unit=item.get("unit", ""),
                direction=item.get("direction", ""),
                formula=item.get("formula", ""),
            )
        )
    lines.append("")
    lines.append("## Derived Metrics")
    for item in payload.get("dictionary", {}).get("derived_metrics", []):
        lines.append(
            "- `{id}` | {title} | unit={unit} | dir={direction} | formula=`{formula}`".format(
                id=item.get("id", ""),
                title=item.get("title", ""),
                unit=item.get("unit", ""),
                direction=item.get("direction", ""),
                formula=item.get("formula", ""),
            )
        )
    example = payload.get("example_bundle", {})
    if isinstance(example, dict) and example:
        lines.append("")
        lines.append("## Example Bundle")
        val = example.get("validation", {}) if isinstance(example.get("validation"), dict) else {}
        lines.append(f"- validation_status: `{val.get('status', '')}`")
        lines.append(f"- planner_count: `{len(example.get('core_by_planner', {}))}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / "p4_metric_dictionary_v1.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"p4_metric_dictionary_v1_{stamp}.md")

    benchmark_file: Path | None = None
    if args.benchmark_json:
        benchmark_file = Path(args.benchmark_json).resolve()
    else:
        benchmark_file = _latest_benchmark_file(settings.outputs_root)

    example_bundle: dict | None = None
    benchmark_source = ""
    if benchmark_file and benchmark_file.exists():
        payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        temporal_compare = payload.get("temporal_compare", {}) if isinstance(payload, dict) else {}
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        if isinstance(summary, dict) and isinstance(temporal_compare, dict) and isinstance(rows, list):
            example_bundle = build_benchmark_metric_bundle(summary=summary, temporal_compare=temporal_compare, rows=rows)
            benchmark_source = str(benchmark_file)

    out_payload = {
        "dictionary_version": "p4_metric_dictionary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_source": benchmark_source,
        "dictionary": metric_dictionary_payload(),
        "example_bundle": example_bundle or {},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(out_payload), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    if benchmark_source:
        print(f"benchmark_source={benchmark_source}")


if __name__ == "__main__":
    main()
