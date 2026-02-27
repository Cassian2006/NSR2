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
from app.core.p4_stratified import build_stratified_eval_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run P4 stratified evaluation (stable/volatile, risk, range).")
    p.add_argument("--benchmark-json", default="", help="Benchmark json path. Defaults to latest planner_benchmark_*.json.")
    p.add_argument("--risk-quantile", type=float, default=0.5)
    p.add_argument("--distance-quantile", type=float, default=0.5)
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--enforce", action="store_true", help="Exit 2 when stratified summary status is FAIL.")
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
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    thresholds = payload.get("thresholds", {}) if isinstance(payload.get("thresholds"), dict) else {}
    lines: list[str] = []
    lines.append("# P4 分层评估")
    lines.append("")
    lines.append(f"- status: `{summary.get('status', '')}`")
    lines.append(f"- total_rows: `{summary.get('total_rows', 0)}`")
    lines.append(f"- strata_count: `{summary.get('strata_count', 0)}`")
    lines.append(f"- risk_threshold: `{thresholds.get('risk_threshold', 0)}`")
    lines.append(f"- distance_threshold: `{thresholds.get('distance_threshold', 0)}`")
    lines.append("")

    lines.append("## 维度覆盖")
    cover = summary.get("dimension_coverage", {}) if isinstance(summary.get("dimension_coverage"), dict) else {}
    for dim, block in cover.items():
        if not isinstance(block, dict):
            continue
        lines.append(
            "- `{dim}`: ok={ok}, present={present}, missing={missing}".format(
                dim=dim,
                ok=block.get("ok", False),
                present=block.get("present", []),
                missing=block.get("missing", []),
            )
        )

    lines.append("")
    lines.append("## 分层摘要")
    lines.append("|strata|samples|success_rate|avg_cost|avg_risk|avg_safety|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload.get("strata_summary", [])[:50]:
        lines.append(
            "|{k}|{n}|{sr:.3f}|{c}|{r}|{s}|".format(
                k=row.get("strata_key", ""),
                n=int(row.get("sample_count", 0)),
                sr=float(row.get("success_rate", 0.0)),
                c=row.get("avg_total_cost_km"),
                r=row.get("avg_risk_exposure"),
                s=row.get("avg_route_safety"),
            )
        )

    lines.append("")
    lines.append("## 失败样本索引（前 30）")
    fail_rows = payload.get("failure_case_index", []) if isinstance(payload.get("failure_case_index"), list) else []
    if not fail_rows:
        lines.append("- none")
    else:
        for item in fail_rows[:30]:
            trace = item.get("trace", {}) if isinstance(item.get("trace"), dict) else {}
            lines.append(
                "- `{k}` | planner={p} | case={c} | detail={d}".format(
                    k=item.get("strata_key", ""),
                    p=trace.get("planner", ""),
                    c=trace.get("case_ref", ""),
                    d=trace.get("detail", ""),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    benchmark_file: Path | None = Path(args.benchmark_json).resolve() if args.benchmark_json else _latest_benchmark_file(settings.outputs_root)
    if benchmark_file is None or not benchmark_file.exists():
        raise SystemExit("No benchmark json found. Please run benchmark_planners.py first or pass --benchmark-json.")

    benchmark_payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
    report = build_stratified_eval_report(
        benchmark_payload=benchmark_payload,
        risk_quantile=float(args.risk_quantile),
        distance_quantile=float(args.distance_quantile),
    )
    report["benchmark_file"] = str(benchmark_file)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / f"p4_stratified_eval_{stamp}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"p4_stratified_eval_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"status={report.get('summary', {}).get('status', '')}")
    if args.enforce and str(report.get("summary", {}).get("status", "")).upper() == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

