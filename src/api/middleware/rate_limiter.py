"""Token-bucket rate limiting per client IP and route (Day 14).

Each ``(client, rule)`` pair owns a bucket holding at most ``limit`` tokens that refills at
``limit / 60`` tokens per second, i.e. ``limit`` requests per minute with bursts up to ``limit``.
A request takes one token; with none left the middleware answers ``429`` with a ``Retry-After``
header (whole seconds until a token is available) without calling the application.

Implemented as a pure ASGI middleware (no ``BaseHTTPMiddleware``) so it adds almost no overhead
and cannot interfere with streaming responses.

The client address is the TCP peer. ``X-Forwarded-For`` is only honoured when
``trust_forwarded_for=True`` because the header is trivially spoofable unless a trusted proxy
overwrites it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_PRUNE_EVERY = 1024
_IDLE_PRUNE_S = 300.0


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """Thread-safe token buckets keyed by an arbitrary string."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Create the limiter; ``clock`` is injectable for tests."""
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def acquire(self, key: str, limit_per_min: int) -> tuple[bool, float, int]:
        """Try to take one token.

        Returns:
            ``(allowed, retry_after_s, remaining)``; ``retry_after_s`` is 0 when allowed.
        """
        capacity = float(limit_per_min)
        rate = capacity / 60.0
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % _PRUNE_EVERY == 0:
                self._prune(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = _Bucket(capacity, now)
            bucket.tokens = min(capacity, bucket.tokens + (now - bucket.updated) * rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0, int(bucket.tokens)
            return False, (1.0 - bucket.tokens) / rate, 0

    def _prune(self, now: float) -> None:
        stale = [k for k, b in self._buckets.items() if now - b.updated > _IDLE_PRUNE_S]
        for key in stale:
            del self._buckets[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)


class RateLimitMiddleware:
    """ASGI middleware enforcing per-IP limits on selected paths.

    Args:
        app: The wrapped ASGI application.
        rules: Exact request path to allowed requests per minute, e.g.
            ``{"/api/v1/predict": 100, "/api/v1/batch-predict": 10}``. Other paths are unlimited.
        limiter: Bucket store (a fresh one by default).
        trust_forwarded_for: Use the first ``X-Forwarded-For`` address as the client.
    """

    def __init__(
        self,
        app: ASGIApp,
        rules: Mapping[str, int],
        limiter: TokenBucketLimiter | None = None,
        trust_forwarded_for: bool = False,
    ) -> None:
        """Create the middleware (see the class docs)."""
        for path, limit in rules.items():
            if limit < 1:
                raise ValueError(f"rate limit for {path} must be >= 1")
        self.app = app
        self.rules = dict(rules)
        self.limiter = limiter or TokenBucketLimiter()
        self.trust_forwarded_for = trust_forwarded_for

    def _client(self, scope: Scope) -> str:
        if self.trust_forwarded_for:
            for name, value in scope.get("headers", []):
                if name == b"x-forwarded-for":
                    first = value.decode("latin-1").split(",")[0].strip()
                    if first:
                        return first
        client = scope.get("client")
        return str(client[0]) if client else "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Apply the rule for the request path, then call the application."""
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        limit = self.rules.get(scope["path"])
        if limit is None:
            await self.app(scope, receive, send)
            return

        allowed, retry_after, remaining = self.limiter.acquire(
            f"{self._client(scope)}|{scope['path']}", limit
        )
        if allowed:

            async def send_with_headers(message: Message) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-ratelimit-limit", str(limit).encode()))
                    headers.append((b"x-ratelimit-remaining", str(remaining).encode()))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_headers)
            return

        wait = max(1, math.ceil(retry_after))
        body = json.dumps(
            {"detail": f"Rate limit exceeded: {limit} requests/minute. Retry in {wait}s."}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(wait).encode()),
                    (b"x-ratelimit-limit", str(limit).encode()),
                    (b"x-ratelimit-remaining", b"0"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
