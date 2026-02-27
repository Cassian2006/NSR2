from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.run_snapshot import (
    load_run_snapshot,
    replay_entrypoint_for_snapshot,
    save_run_snapshot,
)
from app.core.versioning import build_version_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export or replay reproducible run snapshots.")
    p.add_argument("--kind", default="manual_export", help="Snapshot kind when creating new snapshot.")
    p.add_argument("--snapshot-id", default="", help="Snapshot id (or file path) to inspect/replay.")
    p.add_argument("--tag", action="append", default=[], help="Optional tag for created snapshot (repeatable).")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL used for replay endpoint calls.")
    p.add_argument("--print-replay-command", action="store_true", help="Print replay command for snapshot-id.")
    p.add_argument("--replay", action="store_true", help="Execute replay request for snapshot-id when possible.")
    return p.parse_args()


def _post_json(url: str, payload: dict) -> tuple[int, str]:
    req = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=90) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), text
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), body
    except URLError as exc:
        return 599, str(exc.reason)


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if args.snapshot_id:
        snapshot = load_run_snapshot(settings=settings, snapshot_id_or_path=args.snapshot_id)
        if args.print_replay_command:
            print(replay_entrypoint_for_snapshot(snapshot, base_url=args.base_url))
        if args.replay:
            replay = snapshot.get("replay")
            if not isinstance(replay, dict):
                raise SystemExit("snapshot has no replay block")
            endpoint = str(replay.get("endpoint", "")).strip()
            payload = replay.get("payload")
            if not endpoint or not isinstance(payload, dict):
                raise SystemExit("snapshot replay block missing endpoint/payload")
            url = args.base_url.rstrip("/") + endpoint
            status, body = _post_json(url, payload)
            print(f"replay_status={status}")
            print(f"replay_endpoint={url}")
            print(f"replay_response={body[:1200]}")
            if status >= 400:
                raise SystemExit(2)
        return

    version_snapshot = build_version_snapshot(settings=settings, model_version="manual")
    snap = save_run_snapshot(
        settings=settings,
        kind=args.kind,
        config={"entry": "scripts/export_run_snapshot.py"},
        result={"note": "manual snapshot export"},
        version_snapshot=version_snapshot,
        replay={"runner": "script.export_run_snapshot"},
        tags=list(args.tag),
    )
    print(f"snapshot_id={snap['snapshot_id']}")
    print(f"snapshot_file={snap['snapshot_file']}")
    print("tip=use --snapshot-id <id> --print-replay-command")


if __name__ == "__main__":
    main()

