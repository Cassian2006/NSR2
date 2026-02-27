from __future__ import annotations

import json
import sys
from pathlib import Path

from app.core import config as config_module
from app.core import dataset as dataset_module
from app.core.timestamps_index import build_timestamps_index
from scripts import build_timestamps_index as build_index_script


def _make_ann_sample(root: Path, ts: str) -> None:
    folder = root / "processed" / "annotation_pack" / ts
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "x_stack.npy").write_bytes(b"x")
    (folder / "blocked_mask.npy").write_bytes(b"b")


def test_build_timestamps_index_reports_gap() -> None:
    report = build_timestamps_index(
        source_timestamps=["2024-07-01_00", "2024-07-01_01", "2024-07-01_03"],
        step_hours=1,
    )
    summary = report["summary"]
    assert summary["expected_count"] == 4
    assert summary["available_count"] == 3
    assert summary["missing_count"] == 1
    assert report["missing_timestamps"] == ["2024-07-01_02"]
    assert report["gaps"][0]["start"] == "2024-07-01_02"
    assert report["gaps"][0]["end"] == "2024-07-01_02"


def test_build_timestamps_index_supports_3h_downsample() -> None:
    report = build_timestamps_index(
        source_timestamps=[
            "2024-07-01_00",
            "2024-07-01_01",
            "2024-07-01_02",
            "2024-07-01_03",
            "2024-07-01_04",
            "2024-07-01_05",
            "2024-07-01_06",
        ],
        step_hours=3,
    )
    assert report["timestamps"] == ["2024-07-01_00", "2024-07-01_03", "2024-07-01_06"]
    assert report["summary"]["missing_count"] == 0
    assert report["summary"]["coverage"] == 1.0


def test_dataset_service_prefers_timestamps_index(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_ann_sample(data_root, "2024-07-01_00")
    _make_ann_sample(data_root, "2024-07-01_01")
    _make_ann_sample(data_root, "2024-07-01_02")

    idx_path = data_root / "processed" / "timestamps_index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(
        json.dumps(
            {
                "timestamps": ["2024-07-01_00", "2024-07-01_02"],
                "summary": {"status": "pass"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    dataset_module.get_dataset_service.cache_clear()
    try:
        service = dataset_module.get_dataset_service()
        assert service.list_source_timestamps(month="all") == [
            "2024-07-01_00",
            "2024-07-01_01",
            "2024-07-01_02",
        ]
        assert service.list_timestamps(month="all") == ["2024-07-01_00", "2024-07-01_02"]
    finally:
        dataset_module.get_dataset_service.cache_clear()
        config_module.get_settings.cache_clear()


def test_build_timestamps_index_script_writes_output(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_ann_sample(data_root, "2024-07-01_00")
    _make_ann_sample(data_root, "2024-07-01_03")
    _make_ann_sample(data_root, "2024-07-01_06")

    out_path = tmp_path / "timestamps_index.json"
    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    dataset_module.get_dataset_service.cache_clear()
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_timestamps_index.py",
                "--month",
                "all",
                "--step-hours",
                "3",
                "--out",
                str(out_path),
            ],
        )
        build_index_script.main()
    finally:
        dataset_module.get_dataset_service.cache_clear()
        config_module.get_settings.cache_clear()

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["step_hours"] == 3
    assert report["timestamps"] == ["2024-07-01_00", "2024-07-01_03", "2024-07-01_06"]
