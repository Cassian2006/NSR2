from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.p4_protocol import build_p4_eval_protocol, write_protocol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze P4 evaluation protocol.")
    p.add_argument("--model-version", default="unet_v1")
    p.add_argument("--sample-mode", action="store_true")
    p.add_argument("--static-case-count", type=int, default=12)
    p.add_argument("--dynamic-case-count", type=int, default=8)
    p.add_argument("--dynamic-window", type=int, default=6)
    p.add_argument("--dynamic-advance-steps", type=int, default=12)
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    return p.parse_args()


def _to_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# P4 Eval Protocol")
    lines.append("")
    lines.append(f"- protocol_version: `{payload.get('protocol_version', '')}`")
    lines.append(f"- protocol_hash: `{payload.get('protocol_hash', '')}`")
    lines.append(f"- frozen_at: `{payload.get('frozen_at', '')}`")
    scope = payload.get("scope", {})
    lines.append(f"- timestamp_count: `{scope.get('timestamp_count', 0)}`")
    lines.append(f"- range: `{scope.get('timestamp_start', '')}` -> `{scope.get('timestamp_end', '')}`")
    lines.append(f"- months: `{scope.get('months', [])}`")
    lines.append("")
    lines.append("## Version Snapshot")
    for k, v in (payload.get("version_snapshot", {}) or {}).items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Scenarios")
    scenarios = payload.get("scenarios", {})
    static_cases = scenarios.get("static", []) if isinstance(scenarios, dict) else []
    dynamic_cases = scenarios.get("dynamic", []) if isinstance(scenarios, dict) else []
    lines.append(f"- static_case_count: `{len(static_cases)}`")
    lines.append(f"- dynamic_case_count: `{len(dynamic_cases)}`")
    lines.append("")
    lines.append("## Acceptance")
    rpt = ((payload.get("acceptance", {}) or {}).get("repeatability", {}) or {})
    lines.append(f"- metrics: `{rpt.get('metrics', [])}`")
    lines.append(f"- abs_tol: `{rpt.get('abs_tol', 0.0)}`")
    lines.append(f"- rel_tol: `{rpt.get('rel_tol', 0.0)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    static_n = 4 if args.sample_mode else int(args.static_case_count)
    dynamic_n = 3 if args.sample_mode else int(args.dynamic_case_count)
    dynamic_w = 3 if args.sample_mode else int(args.dynamic_window)
    advance_steps = 8 if args.sample_mode else int(args.dynamic_advance_steps)
    payload = build_p4_eval_protocol(
        settings=settings,
        model_version=str(args.model_version),
        static_case_count=static_n,
        dynamic_case_count=dynamic_n,
        dynamic_window=dynamic_w,
        dynamic_advance_steps=advance_steps,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / "p4_eval_protocol_v1.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"p4_eval_protocol_v1_{stamp}.md")
    write_protocol(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"protocol_hash={payload.get('protocol_hash', '')}")


if __name__ == "__main__":
    main()
