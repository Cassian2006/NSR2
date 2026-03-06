# NSR2 Project README

Last updated: 2026-02-27  
Stack: React + Vite + FastAPI + Python

## 1. Project Purpose

NSR2 is a route-planning and risk-analysis system for Northern Sea Route scenarios.
It combines:

- Multi-source environmental grids (ice / wave / wind / bathymetry / AIS)
- U-Net caution and blocked-zone inference
- Static and dynamic replanning (A*, D* Lite, any-angle, hybrid)
- Explainability, gallery records, and exportable reports

## 2. Main Capabilities

- Route planning under safety constraints (`/v1/route/plan`)
- Dynamic timeline replanning (`/v1/route/plan/dynamic`)
- Latest snapshot planning workflow (`/v1/latest/plan`)
- Candidate-route Pareto comparison
- Risk report and report template export (JSON/CSV/Markdown)
- Compliance and data-freshness notices
- Gallery run history with soft-delete and restore

## 3. Backend API Overview

Core routes:

- `GET /healthz`
- `GET /v1/datasets`
- `GET /v1/datasets/quality`
- `GET /v1/timestamps`
- `GET /v1/layers`
- `POST /v1/infer`
- `POST /v1/route/plan`
- `POST /v1/route/plan/dynamic`
- `POST /v1/latest/plan`
- `GET /v1/latest/progress`
- `GET /v1/latest/runtime`
- `GET /v1/latest/sources/health`
- `GET /v1/gallery/list`
- `GET /v1/gallery/deleted`
- `POST /v1/gallery/{id}/restore`
- `DELETE /v1/gallery/{id}`
- `GET /v1/gallery/{id}/risk-report`
- `GET /v1/gallery/{id}/report-template`

## 4. Frontend Pages

- `ScenarioSelector`: timestamp / scenario selection
- `MapWorkspace`: planning controls, map layers, replay, explainability
- `ExportReport`: run history, risk summaries, exports, restore deleted runs
- `AnnotationWorkspace`: human-in-the-loop annotation patch workflow

## 5. Local Run

Backend:

```bash
cd backend
pip install -r requirements-dev.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend env:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/v1
```

Optional backend extras:

```bash
cd backend
pip install -r requirements-optional.txt
```

## 6. Notes

- Gallery deletion is soft-delete by default and can be restored from recycle list.
- The project includes many backend tests; run targeted suites first if full-suite runtime is long.
- `render.yaml` is the Render split-service blueprint; `Dockerfile` is the single-service container option.
- Keep `.env` and local dataset roots out of version control.
