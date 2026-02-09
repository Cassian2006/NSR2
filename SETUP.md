# NSR2 Quick Start

## 1) Backend

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Optional external heatmap (before copying into `NSR2/data`):

```powershell
cd backend
copy .env.example .env
# edit NSR_HEATMAP_ROOT in .env
```

## 2) Frontend

```powershell
cd frontend
npm install
npm run dev
```

Default frontend expects backend at:

`http://127.0.0.1:8000/v1`

Override with env:

```powershell
setx VITE_API_BASE_URL "http://127.0.0.1:8000/v1"
```

## 3) Data notes

- Current source data is under `data/raw` and `data/processed`.
- Readme-standard folders are not yet materialized:
  - `data/env`
  - `data/bathy`
  - `data/ais_heatmap`
  - `data/unet_pred`
- Backend now scans `data/processed/samples/**/meta.json` timestamps directly and can run before full migration.
