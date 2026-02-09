from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_gallery import router as gallery_router
from app.api.routes_infer import router as infer_router
from app.api.routes_layers import router as layers_router
from app.api.routes_plan import router as plan_router
from app.core.config import get_settings
from app.core.dataset import get_dataset_service


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="NSR Route Planning API",
        version="0.1.0",
        description="Skeleton API for NSR route planning and gallery workflows.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(layers_router, prefix="/v1")
    app.include_router(infer_router, prefix="/v1")
    app.include_router(plan_router, prefix="/v1")
    app.include_router(gallery_router, prefix="/v1")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def root() -> dict:
        service = get_dataset_service()
        return {"service": "nsr-api", "latest_timestamp": service.default_timestamp()}

    return app


app = create_app()

