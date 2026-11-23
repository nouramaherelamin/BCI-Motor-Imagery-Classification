"""Application factory and infrastructure wiring for the BCI API."""

from contextlib import asynccontextmanager
import logging
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .api.routes.v1 import router as v1_router
from .config import Settings
from .contracts import PredictorLoader
from .core.logging import configure_logging
from .services.examples import ExampleStore
from .services.model_manager import ModelLoadFailure, ModelManager

logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware:
    """Bound prediction JSON buffering and reject oversized bodies before parsing."""

    def __init__(
        self,
        app: Any,
        *,
        max_bytes: int,
        paths: tuple[str, ...] = ("/api/v1/predict", "/api/v1/predict/file"),
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.paths = paths

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[Any]]) -> None:
        if scope["type"] != "http" or scope.get("path") not in self.paths or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"error": {"code": "request_too_large", "message": "Request body exceeds the configured limit"}},
                )
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        sent = False

        async def replay_body() -> dict[str, Any]:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send)


def default_predictor_loader() -> PredictorLoader | None:
    """Return the project's real ML adapter (``ml.adapter``) if it is importable.

    The backend stays importable without the ML package (e.g. for contract tests); in that
    case a configured artifact is reported as a failed load instead of crashing the process.
    """
    try:
        from ml.adapter import BCIMotorImageryLoader
    except ImportError:
        logger.warning("ml_adapter_unavailable")
        return None
    return BCIMotorImageryLoader()


def create_app(
    settings: Settings | None = None,
    predictor_loader: PredictorLoader | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    models = ModelManager()
    examples = ExampleStore()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = settings
        application.state.model_manager = models
        application.state.examples = ExampleStore.from_directory(settings.examples_dir)
        if settings.model_artifact is not None:
            try:
                models.load(settings.model_artifact, predictor_loader)
                logger.info("model_loaded artifact=%s", settings.model_artifact.name)
            except ModelLoadFailure:
                # Keep the API process available so health/info report the failed state.
                logger.exception("model_startup_load_failed")
        else:
            logger.warning("model_not_configured set BCI_MODEL_ARTIFACT to enable predictions")
        yield

    production = settings.environment == "production"
    application = FastAPI(
        title="BCI Motor Imagery API",
        version="0.1.0",
        description=(
            "Classifies imagined LEFT vs RIGHT hand movement from one raw 10-second, "
            "4-channel (FZ, C3, CZ, C4) EEG trial sampled at 250 Hz, using the project's "
            "trained model and its exact training-time preprocessing."
        ),
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.model_manager = models
    application.state.examples = examples

    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
        )
    application.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo submitted sample values (which are large and may be non-finite).
        details = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Request validation failed", "details": details}},
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            error = exc.detail
        else:
            error = {"code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": error}, headers=exc.headers)

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_request_error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )

    application.include_router(v1_router)
    return application


app = create_app(predictor_loader=default_predictor_loader())
