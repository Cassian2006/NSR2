from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.p4_repro_audit import audit_snapshot_replay, repro_audit_markdown, summarize_repro_audit
from app.core.run_snapshot import load_run_snapshot
from app.main import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P4 reproducibility audit from stored run snapshots.")
    parser.add_argument(
        "--snapshot-id",
        action="append",
        default=[],
        help="Snapshot id (or file path). Repeatable.",
    )
    parser.add_argument("--kind", default="", help="Filter snapshot kind when auto-selecting snapshots.")
    parser.add_argument("--latest-n", type=int, default=5, help="Number of latest snapshots to audit when --snapshot-id is not provided.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: outputs/release")
    parser.add_argument("--allow-warn", action="store_true", help="Exit 0 on WARN overall status.")
    parser.add_argument("--sample-mode", action="store_true", help="Use lighter selection for smoke checks.")
    return parser.parse_args()


def _snapshot_root() -> Path:
    settings = get_settings()
    root = settings.outputs_root / "repro" / "run_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_snapshot_by_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_latest_snapshots(*, kind: str, latest_n: int) -> list[Path]:
    root = _snapshot_root()
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[Path] = []
    for path in files:
        try:
            payload = _load_snapshot_by_path(path)
        except Exception:
            continue
        if kind and str(payload.get("snapshot_kind", "")) != kind:
            continue
        out.append(path)
        if len(out) >= latest_n:
            break
    return out


def _load_snapshots(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = get_settings()
    snapshots: list[dict[str, Any]] = []
    if args.snapshot_id:
        for sid in args.snapshot_id:
            snapshots.append(load_run_snapshot(settings=settings, snapshot_id_or_path=sid))
        return snapshots

    latest_n = 2 if args.sample_mode else max(1, int(args.latest_n))
    for path in _select_latest_snapshots(kind=str(args.kind or ""), latest_n=latest_n):
        snapshots.append(_load_snapshot_by_path(path))
    return snapshots


def main() -> None:
    args = parse_args()
    snapshots = _load_snapshots(args)
    if not snapshots:
        raise SystemExit("No snapshots found for reproducibility audit.")

    with TestClient(app) as client:
        runs = [audit_snapshot_replay(snapshot=snap, client=client) for snap in snapshots]

    summary = summarize_repro_audit(runs)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "runs": runs,
    }

    settings = get_settings()
    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "release")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"p4_repro_audit_{stamp}.json"
    md_path = out_dir / f"p4_repro_audit_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(repro_audit_markdown(payload), encoding="utf-8")

    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"overall_status={summary['overall_status']}")
    print(f"run_count={summary['count']}")

    status = str(summary["overall_status"]).upper()
    if status == "FAIL":
        raise SystemExit(2)
    if status == "WARN" and not args.allow_warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
