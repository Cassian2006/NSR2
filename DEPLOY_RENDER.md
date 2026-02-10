# Render Deployment Guide

## 1. Blueprint Deploy
1. Push this repo to GitHub.
2. In Render, use **New +** -> **Blueprint**.
3. Select this repository; Render will read `render.yaml` and create:
   - `nsr2-api` (FastAPI)
   - `nsr2-web` (web service with `runtime: static`)

## 2. Required Environment Variables

### Frontend (`nsr2-web`)
- `VITE_API_BASE_URL`
  - Example: `https://nsr2-api.onrender.com/v1`
  - Must point to your backend public URL.

### Backend (`nsr2-api`)
- `NSR_CORS_ORIGIN_REGEX`
  - Suggested: `^https://.*\.onrender\.com$`
- `NSR_CORS_ORIGINS`
  - Add explicit frontend domain if needed.
  - Example: `https://nsr2-web.onrender.com`

## 3. Data Path Notes
- Current backend defaults to reading data from repo path `data/`.
- This project usually keeps large data out of Git (`.gitignore`), so cloud deploy may not have full dataset.
- If you want full production data on Render, set:
  - `NSR_DATA_ROOT`
  - `NSR_OUTPUTS_ROOT`
  and mount/provision data accordingly.

## 4. Health and Validation
- Backend health: `GET /healthz`
- After deploy:
  1. Open frontend URL.
  2. Verify scenario/timestamp can load.
  3. Verify map layers and route planning API calls return 200.

## 5. Local Build Check Before Deploy
```bash
cd frontend && npm run build
cd ../backend && python -m pytest -q
```
