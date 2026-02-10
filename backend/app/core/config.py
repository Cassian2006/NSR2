from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NSR_", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[3]
    data_root: Path = project_root / "data"
    outputs_root: Path = project_root / "outputs"
    processed_samples_root: Path = data_root / "processed" / "samples"
    annotation_pack_root: Path = data_root / "processed" / "annotation_pack"
    dataset_index_path: Path = data_root / "processed" / "dataset" / "index.json"
    ais_heatmap_root: Path = data_root / "ais_heatmap"
    gallery_root: Path = outputs_root / "gallery"
    pred_root: Path = outputs_root / "pred"
    unet_default_summary: Path = outputs_root / "train_runs" / "unet_cycle_full_v1" / "summary.json"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    grid_lat_min: float = 60.0
    grid_lat_max: float = 86.0
    grid_lon_min: float = -180.0
    grid_lon_max: float = 180.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [v.strip() for v in self.cors_origins.split(",") if v.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.outputs_root.mkdir(parents=True, exist_ok=True)
    settings.gallery_root.mkdir(parents=True, exist_ok=True)
    (settings.gallery_root / "runs").mkdir(parents=True, exist_ok=True)
    (settings.gallery_root / "thumbs").mkdir(parents=True, exist_ok=True)
    settings.pred_root.mkdir(parents=True, exist_ok=True)
    settings.ais_heatmap_root.mkdir(parents=True, exist_ok=True)
    return settings
