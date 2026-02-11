# Render Deployment Guide (Single Service)

This repo now supports deploying frontend + backend together as one Docker web service.

## 1. Create Service in Render
1. `New +` -> `Web Service`.
2. Connect this GitHub repo.
3. Render detects `Dockerfile` automatically.

## 2. Service Settings
- `Environment`: `Docker`
- `Root Directory`: leave empty (repo root)
- `Branch`: `main`
- `Region`: choose nearest to users
- `Instance Type`: free/starter is fine for first run

No `Build Command` or `Start Command` needed; Dockerfile handles both.

## 3. Environment Variables
Defaults in Docker image:

- `NSR_DATA_ROOT=/app/data`
- `NSR_OUTPUTS_ROOT=/app/outputs`
- `NSR_ALLOW_DEMO_FALLBACK=0`

This is production-first: service uses full dataset mount paths by default and will not silently fall back to demo data.

If you still want free-tier demo fallback behavior, set:
- `NSR_ALLOW_DEMO_FALLBACK=1`

Optional for same-domain deployment:
- `NSR_CORS_ORIGIN_REGEX=^https://.*\.onrender\.com$`
- `NSR_DISABLE_TORCH=1` (recommended for 512MB free-tier instances)

Usually `VITE_API_BASE_URL` is not required now:
- frontend default is same-origin `/v1`
- backend serves frontend static build directly

For `latest` live Copernicus pull, set:
- `NSR_COPERNICUS_USERNAME`
- `NSR_COPERNICUS_PASSWORD`
- `NSR_COPERNICUS_ICE_DATASET_ID`
- `NSR_COPERNICUS_WAVE_DATASET_ID`
- `NSR_COPERNICUS_WIND_DATASET_ID`
- Optional variable overrides:
  - `NSR_COPERNICUS_ICE_VAR` (default `ice_conc`)
  - `NSR_COPERNICUS_ICE_THICK_VAR` (default `ice_thick`)
  - `NSR_COPERNICUS_WAVE_VAR` (default `wave_hs`)
  - `NSR_COPERNICUS_WIND_U_VAR` (default `wind_u10`)
  - `NSR_COPERNICUS_WIND_V_VAR` (default `wind_v10`)
  - `NSR_LATEST_SNAPSHOT_URL_TEMPLATE` / `NSR_LATEST_SNAPSHOT_TOKEN` (fallback NPZ source)

## 4. Data Notes
- The Docker context excludes root `data/` and `outputs/` via `.dockerignore`.
- The image includes `backend/demo_data` as a demo subset (limited timestamps).
- For full dataset deployment, set:
  - `NSR_DATA_ROOT`
  - `NSR_OUTPUTS_ROOT`

### Render Layer Missing Troubleshooting
If base map appears but overlay layers are empty or `U-Net` shows as missing:
1. Open `GET /v1/datasets` and check `sample_count`:
   - `0` means your runtime dataset path is empty/unreachable.
   - `>0` means samples are discoverable from your configured dataset root.
2. Ensure `NSR_DATA_ROOT` points to a mounted disk path that contains:
   - `processed/annotation_pack/<timestamp>/x_stack.npy`
   - `processed/annotation_pack/<timestamp>/blocked_mask.npy`
3. Ensure `NSR_OUTPUTS_ROOT` points to writable persistent storage for:
   - `pred/unet_v1/*.npy`
4. If `torch` is unavailable in Render runtime:
   - backend now falls back to heuristic `unet_pred/unet_uncertainty` generation from environmental channels.
   - first tile request for `unet_pred` may be slower because it materializes cache on demand.
5. If Render reports `Ran out of memory (used over 512MB)`:
   - set `NSR_DISABLE_TORCH=1` to force lightweight heuristic inference path.
   - avoid enabling all heavy layers simultaneously on first load.

## 5. Health Check
- Path: `/healthz`

## 6. Local Validation Before Deploy
```bash
cd frontend && npm run build
cd ../backend && python -m pytest -q
```

## 7. Optional Dependencies For Live Copernicus
The latest-live path lazily imports `copernicusmarine` + `xarray`.
If not installed, app still boots, but `/v1/latest/plan` will fallback to local/snapshot.

Install when needed:
```bash
cd backend
pip install copernicusmarine xarray netCDF4
```
