from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.data_manifest import diff_manifests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two data manifest.jsonl files.")
    p.add_argument("--old", required=True, help="Old manifest path.")
    p.add_argument("--new", required=True, help="New manifest path.")
    p.add_argument(
        "--out",
        default="",
        help="Optional output json path for diff report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    old_manifest = Path(args.old).resolve()
    new_manifest = Path(args.new).resolve()
    if not old_manifest.exists():
        raise FileNotFoundError(f"old manifest not found: {old_manifest}")
    if not new_manifest.exists():
        raise FileNotFoundError(f"new manifest not found: {new_manifest}")

    report = diff_manifests(old_manifest=old_manifest, new_manifest=new_manifest)
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"out={out_path}")

    print(
        "diff="
        + json.dumps(
            {
                "old_count": report["old_count"],
                "new_count": report["new_count"],
                "added_count": report["added_count"],
                "removed_count": report["removed_count"],
                "changed_count": report["changed_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
