from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.data_manifest import build_data_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build data manifest.jsonl with file hash index.")
    p.add_argument(
        "--data-root",
        default="",
        help="Data root directory. Defaults to NSR_DATA_ROOT.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Manifest output path. Defaults to data/processed/manifest.jsonl.",
    )
    p.add_argument(
        "--state",
        default="",
        help="State index output path. Defaults to data/processed/manifest.state.json.",
    )
    p.add_argument(
        "--full-scan",
        action="store_true",
        help="Ignore incremental state and hash all files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    data_root = Path(args.data_root).resolve() if args.data_root else settings.data_root.resolve()
    manifest_path = (
        Path(args.out).resolve()
        if args.out
        else (data_root / "processed" / "manifest.jsonl").resolve()
    )
    state_path = (
        Path(args.state).resolve()
        if args.state
        else (data_root / "processed" / "manifest.state.json").resolve()
    )

    summary = build_data_manifest(
        data_root=data_root,
        manifest_path=manifest_path,
        state_path=state_path,
        full_scan=bool(args.full_scan),
    )
    summary_path = manifest_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"manifest={manifest_path}")
    print(f"state={state_path}")
    print(f"summary={summary_path}")
    print(
        "stats="
        + json.dumps(
            {
                "total_files": summary["total_files"],
                "hashed_files": summary["hashed_files"],
                "reused_hash_files": summary["reused_hash_files"],
                "removed_files": summary["removed_files"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
