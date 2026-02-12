from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import nas_runtime_healthcheck as healthcheck


def _load_single_report(out_dir: Path) -> dict[str, Any]:
    reports = sorted(out_dir.glob("nas_runtime_healthcheck_*.json"))
    assert reports, "expected at least one healthcheck json report"
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def test_nas_healthcheck_pass_writes_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_get_json(*, base_url: str, path: str, timeout_sec: float, retries: int) -> tuple[int, dict[str, Any]]:
        if path == "/healthz":
            return 200, {"status": "ok"}
        if path == "/v1/datasets":
            return 200, {"dataset": {"name": "NSR Dataset", "sample_count": 10}}
        if path == "/v1/timestamps":
            return 200, {"timestamps": ["2024-07-01-00:00", "2024-07-01-06:00"]}
        if path.startswith("/v1/layers?"):
            return 200, {
                "layers": [
                    {"id": "bathy", "available": True},
                    {"id": "ais_heatmap", "available": True},
                    {"id": "unet_pred", "available": True},
                    {"id": "unet_uncertainty", "available": True},
                    {"id": "ice", "available": True},
                    {"id": "wave", "available": True},
                    {"id": "wind", "available": True},
                ]
            }
        raise AssertionError(f"unexpected path: {path}")

    def fake_request(*, url: str, timeout_sec: float, retries: int) -> tuple[int, bytes, dict[str, str]]:
        assert "/v1/overlay/" in url
        return 200, b"\x89PNG" + b"x" * 2048, {"content-type": "image/png"}

    monkeypatch.setattr(healthcheck, "_get_json", fake_get_json)
    monkeypatch.setattr(healthcheck, "_request", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nas_runtime_healthcheck.py",
            "--base-url",
            "http://unit.test:8000",
            "--out-dir",
            str(tmp_path),
            "--probe-count",
            "4",
        ],
    )

    healthcheck.main()

    report = _load_single_report(tmp_path)
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["sample_count"] == 10
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["datasets"]["ok"] is True
    assert checks["layers_probe"]["ok"] is True


def test_nas_healthcheck_fails_on_low_sample_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_get_json(*, base_url: str, path: str, timeout_sec: float, retries: int) -> tuple[int, dict[str, Any]]:
        if path == "/healthz":
            return 200, {"status": "ok"}
        if path == "/v1/datasets":
            return 200, {"dataset": {"name": "NSR Dataset", "sample_count": 0}}
        if path == "/v1/timestamps":
            return 200, {"timestamps": []}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(healthcheck, "_get_json", fake_get_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nas_runtime_healthcheck.py",
            "--base-url",
            "http://unit.test:8000",
            "--out-dir",
            str(tmp_path),
            "--min-sample-count",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        healthcheck.main()
    assert exc.value.code == 2

    report = _load_single_report(tmp_path)
    assert report["summary"]["status"] == "fail"
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["datasets"]["ok"] is False

