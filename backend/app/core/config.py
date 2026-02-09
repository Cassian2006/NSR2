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
    dataset_index_path: Path = data_root / "processed" / "dataset" / "index.json"
    gallery_root: Path = outputs_root / "gallery"
    pred_root: Path = outputs_root / "pred"

    heatmap_root: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [v.strip() for v in self.cors_origins.split(",") if v.strip()]

    @property
    def heatmap_root_path(self) -> Path | None:
        if not self.heatmap_root.strip():
            return None
        return Path(self.heatmap_root)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.outputs_root.mkdir(parents=True, exist_ok=True)
    settings.gallery_root.mkdir(parents=True, exist_ok=True)
    (settings.gallery_root / "runs").mkdir(parents=True, exist_ok=True)
    (settings.gallery_root / "thumbs").mkdir(parents=True, exist_ok=True)
    settings.pred_root.mkdir(parents=True, exist_ok=True)
    return settings

