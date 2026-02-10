from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NSR_",
        extra="ignore",
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
    )

    project_root: Path = Path(__file__).resolve().parents[3]
    data_root: Path = project_root / "data"
    outputs_root: Path = project_root / "outputs"
    processed_samples_root: Path = data_root / "processed" / "samples"
    annotation_pack_root: Path = data_root / "processed" / "annotation_pack"
    env_grids_root: Path = data_root / "interim" / "env_grids"
    dataset_index_path: Path = data_root / "processed" / "dataset" / "index.json"
    ais_heatmap_root: Path = data_root / "ais_heatmap"
    gallery_root: Path = outputs_root / "gallery"
    pred_root: Path = outputs_root / "pred"
    frontend_dist_root: Path = project_root / "frontend" / "build"
    unet_default_summary: Path = outputs_root / "train_runs" / "unet_cycle_full_v1" / "summary.json"
    latest_snapshot_url_template: str = ""
    latest_snapshot_token: str = ""
    copernicus_username: str = ""
    copernicus_password: str = ""
    copernicus_ice_dataset_id: str = "cmems_mod_arc_phy_anfc_6km_detided_PT1H-i"
    copernicus_ice_var: str = "siconc"
    copernicus_ice_thick_var: str = "sithick"
    copernicus_wave_dataset_id: str = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
    copernicus_wave_var: str = "VHM0"
    copernicus_wind_dataset_id: str = "cmems_mod_arc_phy_anfc_6km_detided_PT1H-i"
    copernicus_wind_u_var: str = "vxo"
    copernicus_wind_v_var: str = "vyo"
    copernicus_request_timeout_sec: int = 180
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174,"
        "http://localhost:5178,http://127.0.0.1:5178,"
        "http://localhost:5179,http://127.0.0.1:5179"
    )
    cors_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
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
