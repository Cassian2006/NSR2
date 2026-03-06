# Render Deployment Guide

This repo supports two deployment shapes:

- `render.yaml`: Render blueprint with split services (`nsr2-api` + `nsr2-web`)
- `Dockerfile`: single-service container where the backend serves the built frontend

## 1. Recommended on Render: Blueprint

Use `render.yaml` when you want Render-managed API + static frontend services.

- API service: `backend/`, Python runtime
- Web service: `frontend/`, static build
- Frontend talks to backend through `VITE_API_BASE_URL` injected from the API service URL

This is the authoritative Render blueprint currently checked into the repo.

## 2. Alternative: Single-Service Docker

If you prefer one deployable artifact, use the repo-root `Dockerfile`.

- Backend serves the frontend build directly
- Frontend can use same-origin `/v1`
- No extra static hosting layer is required

Docker defaults are production-oriented:

- `NSR_DATA_ROOT=/app/data`
- `NSR_OUTPUTS_ROOT=/app/outputs`
- `NSR_ALLOW_DEMO_FALLBACK=0`

If you want bundled demo fallback in the single-container path, set:

- `NSR_ALLOW_DEMO_FALLBACK=1`

## 3. Data and Runtime Notes

- Root `data/` and `outputs/` are excluded from Docker build context via `.dockerignore`
- `backend/demo_data` stays in-repo for smoke tests and demo fallback
- For real deployments, point `NSR_DATA_ROOT` and `NSR_OUTPUTS_ROOT` to persistent storage

Optional environment variables:

- `NSR_CORS_ORIGIN_REGEX=^https://.*\.onrender\.com$`
- `NSR_DISABLE_TORCH=1` for low-memory instances
- `NSR_COPERNICUS_USERNAME`
- `NSR_COPERNICUS_PASSWORD`
- `NSR_COPERNICUS_ICE_DATASET_ID`
- `NSR_COPERNICUS_WAVE_DATASET_ID`
- `NSR_COPERNICUS_WIND_DATASET_ID`
- `NSR_LATEST_SNAPSHOT_URL_TEMPLATE`
- `NSR_LATEST_SNAPSHOT_TOKEN`

## 4. Local Validation Before Deploy

```bash
cd frontend && npm run build
cd ../backend && pip install -r requirements-dev.txt
python -m pytest -q tests/test_run_snapshot.py tests/test_data_quality_gate.py tests/test_datasets_registry_api.py
```

Optional extras for live Copernicus or ML/training flows:

```bash
cd backend
pip install -r requirements-optional.txt
```
