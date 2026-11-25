"""FastAPI application of the inference service (Day 14).

Run it with ``make run-api`` / ``python -m src api`` / ``uvicorn src.api.main:app``.

Endpoints (OpenAPI docs at ``/docs``):

* ``POST /api/v1/predict``, ``POST /api/v1/batch-predict`` - score events
* ``GET  /api/v1/anomalies``, ``GET /api/v1/anomalies/{event_id}`` - recent alerts
* ``POST /api/v1/feedback``, ``GET /api/v1/feedback/stats`` - analyst feedback
* ``GET  /api/v1/feedback/analytics``, ``POST /api/v1/feedback/recalibrate`` - feedback loop
* ``GET  /api/v1/models``, ``/models/current``, ``POST /models/reload`` - model versions, hot swap
* ``GET  /health`` - liveness / readiness, ``GET /metrics`` - Prometheus

Middleware, outermost first: request id, metrics, rate limiting, CORS, GZip.

:func:`create_app` is a factory so tests (and later the streaming runner, Day 15) can inject a
config, a registry or a ready-made :class:`~src.api.service.InferenceService`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY, CollectorRegistry, make_asgi_app

from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.middleware.request_context import ApiMetrics, MetricsMiddleware, RequestIDMiddleware
from src.api.routers import detection, feedback, models
from src.api.schemas import HealthResponse
from src.api.service import InferenceService
from src.models.model_registry import ModelRegistry
from src.utils.config import AppConfig, get_config
from src.utils.logger import configure_logging, get_logger

API_VERSION = "0.1.0"
_log = get_logger(__name__)

_DESCRIPTION = """
Real-time anomaly detection for supply-chain streams (warehouse sensors, vehicle GPS,
inventory). Events are scored by an ensemble of a statistical detector, an Isolation Forest and
two LSTM models; the service falls back to the statistical detector when no trained model is
available. Alerts are enriched with a severity level and business-impact weighted score.

**Rate limits** (per client IP): `/predict` 100 requests/min, `/batch-predict` 10 requests/min.
Exceeding them returns `429` with a `Retry-After` header.
"""

_TAGS = [
    {"name": "detection", "description": "Score events and query alerts."},
    {"name": "feedback", "description": "Analyst verdicts on alerts."},
    {"name": "models", "description": "Model versions and hot reload."},
    {"name": "ops", "description": "Health and metrics."},
]


def create_app(
    config: AppConfig | None = None,
    *,
    registry: ModelRegistry | None = None,
    service: InferenceService | None = None,
    metrics_registry: CollectorRegistry | None = None,
    kafka_check: Callable[[], bool] | None = None,
    model_health_publisher: Callable[[dict[str, Any]], Any] | None = None,
    trust_forwarded_for: bool = False,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: Application config (``get_config()`` if omitted).
        registry: Model registry (``paths.model_registry`` of the config if omitted).
        service: A ready-made service (its models are loaded in the lifespan unless already).
        metrics_registry: Prometheus registry for the API series and ``/metrics`` (default:
            the process-wide registry, which also carries the Kafka consumer series).
        kafka_check: Optional callable reporting Kafka connectivity for ``/health``.
        model_health_publisher: Receives the drift alerts of the feedback loop (normally
            ``KafkaProducerManager.send_model_health``); keeps what the service already has if
            omitted.
        trust_forwarded_for: Rate-limit by ``X-Forwarded-For`` (only behind a trusted proxy).
    """
    cfg = config or get_config()
    reg = metrics_registry if metrics_registry is not None else REGISTRY
    api_metrics = ApiMetrics(reg)
    svc = service or InferenceService(cfg, registry)
    if model_health_publisher is not None:
        svc.model_health_publisher = model_health_publisher

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not svc.models.is_loaded:
            svc.start()
        _log.info("api_started", version=API_VERSION)
        try:
            yield
        finally:
            svc.stop()  # persists alerts and feedback
            _log.info("api_stopped")

    app = FastAPI(
        title="Supply Chain Anomaly Detection API",
        version=API_VERSION,
        description=_DESCRIPTION,
        openapi_tags=_TAGS,
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.service = svc
    app.state.api_metrics = api_metrics
    app.state.kafka_check = kafka_check

    # add_middleware wraps: the last one added is the outermost.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "Retry-After",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
        ],
    )
    app.add_middleware(
        RateLimitMiddleware,
        rules={
            "/api/v1/predict": cfg.api.predict_rate_limit_per_min,
            "/api/v1/batch-predict": cfg.api.batch_predict_rate_limit_per_min,
        },
        trust_forwarded_for=trust_forwarded_for,
    )
    app.add_middleware(MetricsMiddleware, metrics=api_metrics)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(detection.router)
    app.include_router(feedback.router)
    app.include_router(models.router)
    app.mount("/metrics", make_asgi_app(registry=reg))

    @app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Health check")
    def health(request: Request) -> HealthResponse:
        """Report liveness and readiness (always HTTP 200 while the process is up).

        ``status`` is ``degraded`` when the service can answer but is not at full strength:
        statistical fallback only, or Kafka unreachable.
        """
        service_: InferenceService = request.app.state.service
        checks: dict[str, str] = {}
        reasons: list[str] = []
        loaded = service_.models.is_loaded
        if not loaded:
            reasons.append("models not loaded")
        elif service_.models.active.degraded:
            reasons.append("trained models unavailable; serving statistical fallback")
        checks["models"] = "ok" if loaded and not service_.models.active.degraded else "degraded"
        check: Callable[[], bool] | None = request.app.state.kafka_check
        if check is not None:
            try:
                ok = bool(check())
            except Exception as error:  # noqa: BLE001 - a broken probe means "unreachable"
                _log.warning("kafka_health_check_failed", error=repr(error))
                ok = False
            checks["kafka"] = "ok" if ok else "unreachable"
            if not ok:
                reasons.append("kafka unreachable")
        return HealthResponse(
            status="degraded" if reasons else "ok",
            uptime_s=round(time.monotonic() - service_.started_at, 3),
            version=API_VERSION,
            models_loaded=loaded,
            degraded_reason="; ".join(reasons) or None,
            checks=checks,
            stored_anomalies=len(service_.anomaly_store),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={
                "detail": errors,
                "request_id": request.scope.get("state", {}).get("request_id"),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.scope.get("state", {}).get("request_id")
        _log.error("unhandled_error", path=request.url.path, error=repr(exc), exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": request_id},
        )

    return app


def _lazy_model_health_publisher(cfg: AppConfig) -> Callable[[dict[str, Any]], bool]:
    """Publish to the ``model-health`` topic through a producer created on first use.

    Drift alerts are rare, so the API process does not hold a Kafka producer until one is needed
    (and starting without a broker stays fine).
    """
    producer: Any = None
    lock = threading.Lock()

    def publish(payload: dict[str, Any]) -> bool:
        nonlocal producer
        with lock:
            if producer is None:
                from src.kafka.producer import KafkaProducerManager

                producer = KafkaProducerManager.from_config(cfg.kafka)
            sent = producer.send_model_health(payload)
            producer.flush(timeout=5.0)
        return bool(sent)

    return publish


def _default_app() -> FastAPI:
    cfg = get_config()
    configure_logging(cfg.logging.level, cfg.logging.format, cfg.logging.service_name)
    return create_app(cfg, model_health_publisher=_lazy_model_health_publisher(cfg))


#: ASGI entry point: ``uvicorn src.api.main:app``.
app: Any = _default_app()
