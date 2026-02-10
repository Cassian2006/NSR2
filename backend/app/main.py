from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_eval import router as eval_router
from app.api.routes_gallery import router as gallery_router
from app.api.routes_infer import router as infer_router
from app.api.routes_layers import router as layers_router
from app.api.routes_plan import router as plan_router
from app.core.config import get_settings
from app.core.errors import install_error_handlers


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
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    app.include_router(layers_router, prefix="/v1")
    app.include_router(infer_router, prefix="/v1")
    app.include_router(plan_router, prefix="/v1")
    app.include_router(gallery_router, prefix="/v1")
    app.include_router(eval_router, prefix="/v1")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    frontend_dist = settings.frontend_dist_root
    frontend_index = frontend_dist / "index.html"

    if frontend_index.exists():
        frontend_root = frontend_dist.resolve()

        @app.get("/", include_in_schema=False)
        def spa_index() -> FileResponse:
            return FileResponse(frontend_index)

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_assets(full_path: str) -> FileResponse:
            if full_path == "v1" or full_path.startswith("v1/"):
                raise HTTPException(status_code=404, detail="Not Found")
            requested = (frontend_dist / full_path).resolve()
            if str(requested).startswith(str(frontend_root)) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(frontend_index)
    else:
        @app.get("/")
        def root_fallback() -> dict:
            return {
                "service": "nsr-api",
                "frontend": "not_built",
                "hint": f"Build frontend to: {Path(frontend_dist).as_posix()}",
            }

    return app


app = create_app()
