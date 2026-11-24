"""Request-ID and request-metrics middleware (Day 14), both pure ASGI.

* :class:`RequestIDMiddleware` gives every request an id: the client's ``X-Request-ID`` if it is
  short and made of safe characters, otherwise a fresh UUID4. The id is echoed in the response
  header, stored in ``scope["state"]["request_id"]`` and bound to the structlog context so every
  log line of the request carries it.
* :class:`MetricsMiddleware` records request count and latency in Prometheus, labelled by the
  *route template* (``/api/v1/anomalies/{event_id}``), never the raw path, to keep label
  cardinality bounded.
"""

from __future__ import annotations

import re
import time
import uuid

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.utils.logger import bind_context, clear_context

REQUEST_ID_HEADER = "X-Request-ID"
_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode()
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class RequestIDMiddleware:
    """Attach ``X-Request-ID`` to every request and response."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Resolve the id, bind it to logs, and add it to the response."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        supplied = next(
            (v.decode("latin-1") for k, v in scope["headers"] if k == _HEADER_BYTES), ""
        )
        request_id = supplied if _SAFE_ID.fullmatch(supplied) else str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        clear_context()
        bind_context(request_id=request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k != _HEADER_BYTES]
                headers.append((_HEADER_BYTES, request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            clear_context()


class ApiMetrics:
    """Prometheus series of the HTTP layer (prefixed ``scad_api_``)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Register the series on ``registry`` (the default one if omitted)."""
        reg = registry if registry is not None else REGISTRY
        self.requests = Counter(
            "scad_api_requests_total",
            "HTTP requests by method, route and status code.",
            ["method", "route", "status"],
            registry=reg,
        )
        self.latency = Histogram(
            "scad_api_request_duration_seconds",
            "HTTP request duration by route.",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=reg,
        )
        self.predictions = Counter(
            "scad_api_predictions_total",
            "Events scored by the API, by outcome.",
            ["outcome"],
            registry=reg,
        )


class MetricsMiddleware:
    """Record request count and latency per route template."""

    def __init__(self, app: ASGIApp, metrics: ApiMetrics) -> None:
        """Wrap ``app`` and report to ``metrics``."""
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the request and count it by final status."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = {"code": 500}

        async def send_status(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None)
            if template is None:
                path = scope["path"]
                template = path if path in ("/health", "/metrics") else "unmatched"
            method = scope["method"]
            self.metrics.requests.labels(method, template, str(status["code"])).inc()
            self.metrics.latency.labels(method, template).observe(time.perf_counter() - started)
