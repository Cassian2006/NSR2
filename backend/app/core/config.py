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
    latest_root: Path = outputs_root / "latest"
    latest_progress_store_path: Path = latest_root / "progress_state.json"
    latest_source_health_path: Path = latest_root / "source_health.json"
    latest_progress_retention_hours: int = 72
    latest_progress_max_entries: int = 2000
    latest_plan_max_concurrent: int = 2
    latest_remote_retries: int = 3
    latest_remote_retry_backoff_sec: float = 1.5
    latest_source_failure_threshold: int = 3
    latest_source_cooldown_sec: int = 900
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
    allow_demo_fallback: bool = True
    grid_lat_min: float = 60.0
    grid_lat_max: float = 86.0
    grid_lon_min: float = -180.0
    grid_lon_max: float = 180.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [v.strip() for v in self.cors_origins.split(",") if v.strip()]


def _has_annotation_samples(annotation_pack_root: Path) -> bool:
    if not annotation_pack_root.exists() or not annotation_pack_root.is_dir():
        return False
    for folder in annotation_pack_root.iterdir():
        if not folder.is_dir():
            continue
        if (folder / "x_stack.npy").exists() and (folder / "blocked_mask.npy").exists():
            return True
    return False


def _has_any_heatmap_files(heatmap_root: Path) -> bool:
    if not heatmap_root.exists() or not heatmap_root.is_dir():
        return False
    return any(heatmap_root.rglob("*.npy"))


def _resolve_demo_data_root(project_root: Path) -> Path | None:
    candidates = [
        project_root / "backend" / "demo_data",
        project_root / "demo_data",
    ]
    for candidate in candidates:
        ann_root = candidate / "processed" / "annotation_pack"
        if _has_annotation_samples(ann_root):
            return candidate
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    provided = set(settings.model_fields_set)

    # Keep derived paths synced with data_root / outputs_root unless explicitly overridden.
    if "processed_samples_root" not in provided:
        settings.processed_samples_root = settings.data_root / "processed" / "samples"
    if "annotation_pack_root" not in provided:
        settings.annotation_pack_root = settings.data_root / "processed" / "annotation_pack"
    if "env_grids_root" not in provided:
        settings.env_grids_root = settings.data_root / "interim" / "env_grids"
    if "dataset_index_path" not in provided:
        settings.dataset_index_path = settings.data_root / "processed" / "dataset" / "index.json"
    if "ais_heatmap_root" not in provided:
        settings.ais_heatmap_root = settings.data_root / "ais_heatmap"

    if "gallery_root" not in provided:
        settings.gallery_root = settings.outputs_root / "gallery"
    if "pred_root" not in provided:
        settings.pred_root = settings.outputs_root / "pred"
    if "latest_root" not in provided:
        settings.latest_root = settings.outputs_root / "latest"
    if "latest_progress_store_path" not in provided:
        settings.latest_progress_store_path = settings.latest_root / "progress_state.json"
    if "latest_source_health_path" not in provided:
        settings.latest_source_health_path = settings.latest_root / "source_health.json"
    if "unet_default_summary" not in provided:
        settings.unet_default_summary = settings.outputs_root / "train_runs" / "unet_cycle_full_v1" / "summary.json"

    # Optional deployment safeguard:
    # when enabled, auto-fallback to bundled demo_data if configured roots are empty.
    if settings.allow_demo_fallback and not _has_annotation_samples(settings.annotation_pack_root):
        demo_root = _resolve_demo_data_root(settings.project_root)
        if demo_root is not None:
            settings.data_root = demo_root
            settings.processed_samples_root = demo_root / "processed" / "samples"
            settings.annotation_pack_root = demo_root / "processed" / "annotation_pack"
            settings.env_grids_root = demo_root / "interim" / "env_grids"
            settings.dataset_index_path = demo_root / "processed" / "dataset" / "index.json"
            settings.ais_heatmap_root = demo_root / "ais_heatmap"

    if settings.allow_demo_fallback and not _has_any_heatmap_files(settings.ais_heatmap_root):
        demo_root = _resolve_demo_data_root(settings.project_root)
        if demo_root is not None:
            settings.ais_heatmap_root = demo_root / "ais_heatmap"

    settings.outputs_root.mkdir(parents=True, exist_ok=True)
    settings.latest_root.mkdir(parents=True, exist_ok=True)
    settings.gallery_root.mkdir(parents=True, exist_ok=True)
    (settings.gallery_root / "runs").mkdir(parents=True, exist_ok=True)
    (settings.gallery_root / "thumbs").mkdir(parents=True, exist_ok=True)
    settings.pred_root.mkdir(parents=True, exist_ok=True)
    settings.ais_heatmap_root.mkdir(parents=True, exist_ok=True)
    return settings
