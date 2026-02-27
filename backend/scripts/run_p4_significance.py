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
from app.core.p4_stats import build_significance_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run P4 statistical significance analysis for benchmark payload.")
    p.add_argument("--benchmark-json", default="", help="Benchmark json path. Defaults to latest planner_benchmark_*.json.")
    p.add_argument("--mode", default="dynamic", choices=["dynamic", "static"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
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
    lines.append("# P4 统计显著性分析")
    lines.append("")
    lines.append(f"- mode: `{payload.get('mode', '')}`")
    lines.append(f"- alpha: `{payload.get('alpha', '')}`")
    methods = payload.get("methods", {}) if isinstance(payload.get("methods"), dict) else {}
    ci = methods.get("ci", {}) if isinstance(methods.get("ci"), dict) else {}
    test = methods.get("test", {}) if isinstance(methods.get("test"), dict) else {}
    lines.append(f"- CI: `{ci.get('name', '')}` (n_boot={ci.get('n_boot', 0)})")
    lines.append(f"- Test: `{test.get('name', '')}` (n_perm={test.get('n_perm', 0)})")
    lines.append("")

    lines.append("## 关键结论")
    conclusions = payload.get("conclusions", []) if isinstance(payload.get("conclusions"), list) else []
    if conclusions:
        for item in conclusions[:20]:
            p_raw = item.get("p_value")
            p_text = f"{float(p_raw):.6f}" if p_raw is not None else "NA"
            lines.append(
                "- {statement}; p={p}; confidence={c}".format(
                    statement=item.get("statement", ""),
                    p=p_text,
                    c=item.get("confidence", "low"),
                )
            )
    else:
        lines.append("- 暂无显著结论（可能样本量不足或差异不显著）。")

    lines.append("")
    lines.append("## 指标置信区间")
    per_planner = payload.get("per_planner_metrics", {}) if isinstance(payload.get("per_planner_metrics"), dict) else {}
    for planner, block in per_planner.items():
        if not isinstance(block, dict):
            continue
        lines.append(f"### {planner}")
        for metric_name, ci in block.items():
            if not isinstance(ci, dict):
                continue
            lines.append(
                "- `{m}`: mean={mean}, ci=[{lo}, {hi}], n={n}".format(
                    m=metric_name,
                    mean=ci.get("mean"),
                    lo=ci.get("ci_low"),
                    hi=ci.get("ci_high"),
                    n=ci.get("n", 0),
                )
            )

    warns = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []
    lines.append("")
    lines.append("## 统计附录（告警）")
    if warns:
        for w in warns[:50]:
            lines.append(f"- {w}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    benchmark_file: Path | None = Path(args.benchmark_json).resolve() if args.benchmark_json else _latest_benchmark_file(settings.outputs_root)
    if benchmark_file is None or not benchmark_file.exists():
        raise SystemExit("No benchmark json found. Please run benchmark_planners.py first or pass --benchmark-json.")

    benchmark_payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
    report = build_significance_report(
        benchmark_payload=benchmark_payload,
        mode=str(args.mode),
        n_boot=max(200, int(args.n_boot)),
        n_perm=max(1000, int(args.n_perm)),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )
    report["benchmark_file"] = str(benchmark_file)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / f"p4_significance_{stamp}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"p4_significance_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"conclusion_count={len(report.get('conclusions', []))}")


if __name__ == "__main__":
    main()
