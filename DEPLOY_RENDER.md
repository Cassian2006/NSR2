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
Optional for same-domain deployment:
- `NSR_CORS_ORIGIN_REGEX=^https://.*\.onrender\.com$`

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
- The Docker context excludes `data/` and `outputs/` via `.dockerignore`.
- Without mounted/provisioned dataset, API can start but map/planning data will be missing.
- If you have external data storage, set:
  - `NSR_DATA_ROOT`
  - `NSR_OUTPUTS_ROOT`

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
