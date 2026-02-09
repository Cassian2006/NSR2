from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings


TIMESTAMP_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}$")
UI_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}$")


@dataclass(frozen=True)
class SampleRecord:
    timestamp: str
    folder: Path
    files: dict[str, Path]


def normalize_timestamp(value: str) -> str:
    value = value.strip()
    if TIMESTAMP_DIR_RE.match(value):
        return value
    if UI_TIMESTAMP_RE.match(value):
        dt = datetime.strptime(value, "%Y-%m-%d-%H:%M")
        return dt.strftime("%Y-%m-%d_%H")
    raise ValueError(f"Unsupported timestamp format: {value}")


def ui_timestamp(value: str) -> str:
    dt = datetime.strptime(value, "%Y-%m-%d_%H")
    return dt.strftime("%Y-%m-%d-%H:%M")


def month_key(timestamp: str) -> str:
    return timestamp[:7]


class DatasetService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @lru_cache(maxsize=1)
    def _scan_samples(self) -> dict[str, SampleRecord]:
        records: dict[str, SampleRecord] = {}
        if not self.settings.processed_samples_root.exists():
            return records

        for meta_path in self.settings.processed_samples_root.rglob("meta.json"):
            folder = meta_path.parent
            ts = folder.name
            if not TIMESTAMP_DIR_RE.match(ts):
                continue
            files = {p.name: p for p in folder.glob("*.npy")}
            existing = records.get(ts)
            if existing and len(existing.files) >= len(files):
                continue
            records[ts] = SampleRecord(timestamp=ts, folder=folder, files=files)
        return records

    @lru_cache(maxsize=1)
    def _read_legacy_index(self) -> list[dict[str, Any]]:
        if not self.settings.dataset_index_path.exists():
            return []
        with self.settings.dataset_index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    def list_timestamps(self, month: str | None = None) -> list[str]:
        all_ts = sorted(self._scan_samples().keys())
        if month:
            return [ts for ts in all_ts if ts.startswith(month)]
        return all_ts

    def default_timestamp(self) -> str | None:
        all_ts = self.list_timestamps()
        if not all_ts:
            return None
        return all_ts[-1]

    def list_months(self) -> list[str]:
        return sorted({month_key(ts) for ts in self._scan_samples().keys()})

    def _legacy_entry_by_timestamp(self, timestamp: str) -> dict[str, Any] | None:
        for entry in self._read_legacy_index():
            if entry.get("sample") == timestamp:
                return entry
        return None

    def _layer_available_from_raw(self, layer: str, timestamp: str) -> bool:
        month = timestamp[5:7]
        if layer == "ice":
            p = self.settings.data_root / "raw" / "env_nc" / "ice_conc" / "2024" / month / "ice_conc.nc"
        elif layer == "wave":
            p = self.settings.data_root / "raw" / "env_nc" / "wave_hs" / "2024" / month / "wave_hs.nc"
        elif layer == "wind":
            p = self.settings.data_root / "raw" / "env_nc" / "wind_10m" / "2024" / month / "wind_10m.nc"
        else:
            return False
        return p.exists()

    def _find_local_heatmap(self, timestamp: str) -> Path | None:
        root = self.settings.ais_heatmap_root
        if not root.exists():
            return None
        for p in root.rglob(f"{timestamp}.npy"):
            return p
        return None

    def list_layers(self, timestamp: str) -> list[dict[str, Any]]:
        normalized = normalize_timestamp(timestamp)
        sample = self._scan_samples().get(normalized)
        legacy = self._legacy_entry_by_timestamp(normalized)
        local_heatmap = self._find_local_heatmap(normalized)
        pred_file = self.settings.pred_root / "unet_v1" / f"{normalized}.npy"
        legacy_bathy_exists = False
        if legacy and legacy.get("x_bathy"):
            legacy_bathy_exists = Path(str(legacy["x_bathy"])).exists()

        return [
            {
                "id": "bathy",
                "name": "Bathymetry",
                "available": bool(sample and "x_bathy.npy" in sample.files) or legacy_bathy_exists,
                "unit": "m",
            },
            {
                "id": "ais_heatmap",
                "name": "AIS Heatmap",
                "available": local_heatmap is not None,
                "unit": "score",
                "source": str(local_heatmap) if local_heatmap else "",
            },
            {
                "id": "unet_pred",
                "name": "U-Net Prediction",
                "available": pred_file.exists(),
                "unit": "class",
            },
            {
                "id": "ice",
                "name": "Ice Concentration",
                "available": self._layer_available_from_raw("ice", normalized),
                "unit": "%",
            },
            {
                "id": "wave",
                "name": "Wave Height",
                "available": self._layer_available_from_raw("wave", normalized),
                "unit": "m",
            },
            {
                "id": "wind",
                "name": "Wind 10m",
                "available": self._layer_available_from_raw("wind", normalized),
                "unit": "m/s",
            },
        ]

    def datasets_summary(self) -> dict[str, Any]:
        months = self.list_months()
        sample_count = len(self._scan_samples())
        return {
            "name": "NSR Dataset",
            "months": months,
            "sample_count": sample_count,
            "has_legacy_index": self.settings.dataset_index_path.exists(),
        }


@lru_cache(maxsize=1)
def get_dataset_service() -> DatasetService:
    return DatasetService()
