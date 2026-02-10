from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except Exception:
        return "Request Error"


def _error_payload(*, code: str, status_code: int, message: str, detail: Any) -> dict[str, Any]:
    return {
        "code": code,
        "status": status_code,
        "message": message,
        "detail": detail,
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:  # noqa: ARG001
        detail = exc.detail
        if isinstance(detail, str):
            message = detail
        else:
            message = _status_phrase(exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                code="http_error",
                status_code=exc.status_code,
                message=message,
                detail=detail,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                code="validation_error",
                status_code=422,
                message="Request validation failed",
                detail=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                code="internal_error",
                status_code=500,
                message="Internal server error",
                detail=None,
            ),
        )

