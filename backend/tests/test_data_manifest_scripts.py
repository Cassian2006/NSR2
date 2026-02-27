from __future__ import annotations

import json
import sys
from pathlib import Path

from app.core.data_manifest import build_data_manifest, diff_manifests, load_manifest
from scripts import build_data_manifest as build_manifest_script
from scripts import diff_data_manifest as diff_manifest_script


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_manifest_incremental_reuses_hash(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_text(data_root / "raw" / "copernicus" / "2024-07-01_00.json", '{"v":1}')
    _write_text(data_root / "ais_heatmap" / "7d" / "2024-07-01_00.npy", "fake-npy")

    manifest = data_root / "processed" / "manifest.jsonl"
    state = data_root / "processed" / "manifest.state.json"

    first = build_data_manifest(data_root=data_root, manifest_path=manifest, state_path=state, full_scan=False)
    assert first["total_files"] == 2
    assert first["hashed_files"] == 2
    assert first["reused_hash_files"] == 0

    second = build_data_manifest(data_root=data_root, manifest_path=manifest, state_path=state, full_scan=False)
    assert second["total_files"] == 2
    assert second["hashed_files"] == 0
    assert second["reused_hash_files"] == 2

    _write_text(data_root / "raw" / "copernicus" / "2024-07-01_00.json", '{"v":2}')
    third = build_data_manifest(data_root=data_root, manifest_path=manifest, state_path=state, full_scan=False)
    assert third["total_files"] == 2
    assert third["hashed_files"] == 1
    assert third["reused_hash_files"] == 1

    rows = load_manifest(manifest)
    assert "raw/copernicus/2024-07-01_00.json" in rows
    assert rows["raw/copernicus/2024-07-01_00.json"]["timestamp"] == "2024-07-01_00"
    assert rows["ais_heatmap/7d/2024-07-01_00.npy"]["source"] == "ais_heatmap"


def test_diff_manifest_detects_add_remove_change(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_text(data_root / "raw" / "a.txt", "A1")
    _write_text(data_root / "raw" / "b.txt", "B1")

    old_manifest = tmp_path / "old.jsonl"
    old_state = tmp_path / "old.state.json"
    build_data_manifest(data_root=data_root, manifest_path=old_manifest, state_path=old_state, full_scan=True)

    _write_text(data_root / "raw" / "a.txt", "A2")
    (data_root / "raw" / "b.txt").unlink()
    _write_text(data_root / "raw" / "c.txt", "C1")
    new_manifest = tmp_path / "new.jsonl"
    new_state = tmp_path / "new.state.json"
    build_data_manifest(data_root=data_root, manifest_path=new_manifest, state_path=new_state, full_scan=True)

    diff = diff_manifests(old_manifest=old_manifest, new_manifest=new_manifest)
    assert diff["added_count"] == 1
    assert diff["removed_count"] == 1
    assert diff["changed_count"] == 1
    assert diff["added"] == ["raw/c.txt"]
    assert diff["removed"] == ["raw/b.txt"]
    assert diff["changed"][0]["path"] == "raw/a.txt"


def test_manifest_cli_scripts(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_text(data_root / "processed" / "annotation_pack" / "2024-07-01_00" / "meta.json", '{"ok":true}')

    out_manifest = tmp_path / "manifest_a.jsonl"
    state_path = tmp_path / "manifest_a.state.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_data_manifest.py",
            "--data-root",
            str(data_root),
            "--out",
            str(out_manifest),
            "--state",
            str(state_path),
        ],
    )
    build_manifest_script.main()
    assert out_manifest.exists()
    assert out_manifest.with_suffix(".summary.json").exists()

    _write_text(data_root / "processed" / "annotation_pack" / "2024-07-01_00" / "meta.json", '{"ok":false}')
    out_manifest2 = tmp_path / "manifest_b.jsonl"
    state_path2 = tmp_path / "manifest_b.state.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_data_manifest.py",
            "--data-root",
            str(data_root),
            "--out",
            str(out_manifest2),
            "--state",
            str(state_path2),
        ],
    )
    build_manifest_script.main()
    assert out_manifest2.exists()

    out_diff = tmp_path / "manifest_diff.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diff_data_manifest.py",
            "--old",
            str(out_manifest),
            "--new",
            str(out_manifest2),
            "--out",
            str(out_diff),
        ],
    )
    diff_manifest_script.main()
    report = json.loads(out_diff.read_text(encoding="utf-8"))
    assert report["changed_count"] == 1
