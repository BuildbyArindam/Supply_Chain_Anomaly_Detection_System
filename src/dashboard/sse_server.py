"""Server-Sent Events feed of new alerts: ``GET /api/v1/stream/alerts`` (Day 18).

:class:`AlertBroker` receives every alert the :class:`~src.api.anomaly_store.AnomalyStore` adds
(through ``store.subscribe``) and fans it out to the connected SSE clients, so a client sees an
alert within milliseconds of detection rather than at the next poll.

Wire format (``text/event-stream``)::

    retry: 3000
    event: hello     data: {"epoch": "...", "connections": 3, "server_time": "..."}
    id: <epoch>-<n>
    event: alert     data: <AnomalyEvent.to_dict() as JSON>
    : ping                                   (comment keep-alive every 15 s)

Reconnecting
------------
Every alert has an id ``<epoch>-<n>``; ``n`` counts up from 1 and ``epoch`` identifies the server
run. A browser's ``EventSource`` reconnects by itself and sends the last id as ``Last-Event-ID``
(``?last_event_id=`` works for clients that cannot set headers). The broker keeps the latest
``history_size`` alerts and replays those the client missed. When it cannot, because the server
restarted (other epoch) or the gap is older than the history, it first sends ``event: resync`` and
the client should re-read ``GET /api/v1/anomalies``.

Limits
------
* At most ``max_connections`` (default 50) streams at a time; the 51st gets HTTP 503 with
  ``Retry-After``.
* Every client has a bounded queue. A client too slow to keep up is sent ``event: overflow`` and
  disconnected; its reconnect replays what it missed. A slow client never blocks detection.
* :meth:`AlertBroker.close` ends all streams so that a server shutdown is not held up by them.

``publish`` is thread-safe and never blocks: the engine calls it from worker threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.api.anomaly_store import AnomalyStore
    from src.models.ensemble_detector import AnomalyEvent

_log = get_logger(__name__)

SEVERITIES: Final[tuple[str, ...]] = ("INFO", "WARNING", "CRITICAL", "EMERGENCY")
SeverityLiteral = Literal["INFO", "WARNING", "CRITICAL", "EMERGENCY"]
DEFAULT_MAX_CONNECTIONS: Final = 50
DEFAULT_HISTORY: Final = 1000
DEFAULT_QUEUE: Final = 256
DEFAULT_KEEPALIVE_S: Final = 15.0
RETRY_MS: Final = 3000
RETRY_AFTER_S: Final = 5


@dataclass(frozen=True, slots=True)
class _Message:
    seq: int
    rank: int
    data: str


class _Marker:
    """Queue sentinel (end of stream)."""

    def __init__(self, name: str) -> None:
        self.name = name


_CLOSE: Final = _Marker("close")
_OVERFLOW: Final = _Marker("overflow")


class _Subscriber:
    __slots__ = ("loop", "min_rank", "queue")

    def __init__(self, loop: asyncio.AbstractEventLoop, min_rank: int, size: int) -> None:
        self.loop = loop
        self.min_rank = min_rank
        self.queue: asyncio.Queue[_Message | _Marker] = asyncio.Queue(maxsize=size)


class ConnectionSlot:
    """One of the broker's connection slots; :meth:`release` is idempotent."""

    def __init__(self, broker: AlertBroker) -> None:
        self._broker = broker
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        """Give the slot back (safe to call more than once)."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._broker._release_slot()


class AlertBroker:
    """Fans alerts out to SSE clients; see the module docstring for the protocol."""

    def __init__(
        self,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        history_size: int = DEFAULT_HISTORY,
        queue_size: int = DEFAULT_QUEUE,
        keepalive_s: float = DEFAULT_KEEPALIVE_S,
    ) -> None:
        """Create the broker.

        Raises:
            ValueError: A size or interval is not positive.
        """
        if min(max_connections, history_size, queue_size) < 1 or keepalive_s <= 0:
            raise ValueError("max_connections, history_size, queue_size, keepalive_s must be > 0")
        self.max_connections = max_connections
        self.keepalive_s = keepalive_s
        self._queue_size = queue_size
        self.epoch = format(int(time.time() * 1000), "x")
        self._seq = 0
        self._history: deque[_Message] = deque(maxlen=history_size)
        self._subs: set[_Subscriber] = set()
        self._slots = 0
        self._closed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def connections(self) -> int:
        """Slots in use (streams open or about to open)."""
        return self._slots

    @property
    def published(self) -> int:
        """Alerts published since start."""
        return self._seq

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` was called."""
        return self._closed

    def try_acquire(self) -> ConnectionSlot | None:
        """Take a connection slot, or return ``None`` when all are in use (or closed)."""
        with self._lock:
            if self._closed or self._slots >= self.max_connections:
                return None
            self._slots += 1
        return ConnectionSlot(self)

    def _release_slot(self) -> None:
        with self._lock:
            self._slots = max(0, self._slots - 1)

    # ------------------------------------------------------------------ #
    # Publishing
    # ------------------------------------------------------------------ #
    def publish(self, event: AnomalyEvent | dict[str, Any]) -> int:
        """Broadcast one alert; thread-safe and non-blocking. Returns its sequence number."""
        payload = event if isinstance(event, dict) else event.to_dict()
        severity = str(payload.get("severity", "INFO")).upper()
        rank = SEVERITIES.index(severity) if severity in SEVERITIES else 0
        data = json.dumps(payload, default=str, separators=(",", ":"))
        with self._lock:
            self._seq += 1
            message = _Message(self._seq, rank, data)
            self._history.append(message)
            # Delivered under the lock so that two publisher threads cannot reorder messages.
            for sub in list(self._subs):
                if rank >= sub.min_rank:
                    self._deliver(sub, message)
        return message.seq

    def _deliver(self, sub: _Subscriber, item: _Message | _Marker) -> None:
        def put() -> None:
            try:
                sub.queue.put_nowait(item)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):  # raced with the consumer
                    sub.queue.get_nowait()
                sub.queue.put_nowait(_OVERFLOW)

        try:
            sub.loop.call_soon_threadsafe(put)
        except RuntimeError:  # the client's event loop is gone
            self._subs.discard(sub)

    def close(self) -> None:
        """End every stream and refuse new ones (call on shutdown)."""
        with self._lock:
            self._closed = True
            for sub in list(self._subs):
                self._deliver(sub, _CLOSE)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def _attach(
        self, loop: asyncio.AbstractEventLoop, min_rank: int, last_event_id: str | None
    ) -> tuple[_Subscriber, list[_Message], bool]:
        """Register a client and snapshot the alerts it missed, atomically."""
        sub = _Subscriber(loop, min_rank, self._queue_size)
        with self._lock:
            replay: list[_Message] = []
            resync = False
            if last_event_id:
                epoch, _, number = last_event_id.partition("-")
                try:
                    last = int(number)
                except ValueError:
                    last = -1
                oldest = self._history[0].seq if self._history else self._seq + 1
                if epoch != self.epoch or last < 0 or last > self._seq:
                    resync = True  # another server run, or a malformed id
                else:
                    resync = last + 1 < oldest  # part of the gap is no longer retained
                    replay = [m for m in self._history if m.seq > last and m.rank >= min_rank]
            if self._closed:
                sub.queue.put_nowait(_CLOSE)
            self._subs.add(sub)
        return sub, replay, resync

    def _detach(self, sub: _Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    def _frame(self, message: _Message) -> str:
        return f"id: {self.epoch}-{message.seq}\nevent: alert\ndata: {message.data}\n\n"

    async def stream(
        self,
        *,
        min_severity: str | None = None,
        last_event_id: str | None = None,
        slot: ConnectionSlot | None = None,
    ) -> AsyncIterator[str]:
        """Yield SSE frames until the client disconnects, overflows or the broker closes.

        Args:
            min_severity: Only alerts at or above this level.
            last_event_id: The client's last seen id (replay what it missed).
            slot: Connection slot to give back when the stream ends.
        """
        min_rank = SEVERITIES.index(min_severity) if min_severity in SEVERITIES else 0
        sub, replay, resync = self._attach(asyncio.get_running_loop(), min_rank, last_event_id)
        try:
            hello = {
                "epoch": self.epoch,
                "connections": self._slots,
                "server_time": datetime.now(UTC).isoformat(),
            }
            yield f"retry: {RETRY_MS}\nevent: hello\ndata: {json.dumps(hello)}\n\n"
            if resync:
                yield 'event: resync\ndata: {"reason": "history_unavailable"}\n\n'
            for message in replay:
                yield self._frame(message)
            while True:
                try:
                    item = await asyncio.wait_for(sub.queue.get(), timeout=self.keepalive_s)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                if item is _CLOSE:
                    return
                if item is _OVERFLOW:
                    yield 'event: overflow\ndata: {"reason": "client_too_slow"}\n\n'
                    return
                assert isinstance(item, _Message)
                yield self._frame(item)
        finally:
            self._detach(sub)
            if slot is not None:
                slot.release()


# --------------------------------------------------------------------------- #
# FastAPI wiring
# --------------------------------------------------------------------------- #
def build_stream_router() -> APIRouter:
    """Return the router with ``GET /api/v1/stream/alerts`` and ``GET /api/v1/stream/status``.

    The broker is read from ``request.app.state.alert_broker`` (see :func:`install_alert_stream`).
    """
    router = APIRouter(prefix="/api/v1/stream", tags=["stream"])

    def broker_of(request: Request) -> AlertBroker:
        broker = getattr(request.app.state, "alert_broker", None)
        if broker is None:
            raise HTTPException(503, "alert stream not initialised")
        return broker  # type: ignore[no-any-return]

    @router.get(
        "/alerts",
        summary="Live alerts (Server-Sent Events)",
        responses={
            200: {"content": {"text/event-stream": {}}, "description": "SSE stream of alerts"},
            503: {"description": "All connection slots are in use"},
        },
    )
    async def stream_alerts(
        request: Request,
        min_severity: Annotated[SeverityLiteral | None, Query(description="Lowest level")] = None,
        last_event_id: Annotated[str | None, Query(description="Replay from this id")] = None,
        last_event_id_header: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        """Stream every new alert as it is detected.

        Browsers reconnect automatically and resume from the last id they saw. At most
        ``dashboard.max_sse_connections`` streams are served at once.
        """
        broker = broker_of(request)
        slot = broker.try_acquire()
        if slot is None:
            raise HTTPException(
                503,
                "too many live connections" if not broker.closed else "shutting down",
                headers={"Retry-After": str(RETRY_AFTER_S)},
            )
        generator = broker.stream(
            min_severity=min_severity,
            last_event_id=last_event_id_header or last_event_id,
            slot=slot,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(slot.release),
        )

    @router.get("/status", summary="State of the alert stream")
    def stream_status(request: Request) -> dict[str, Any]:
        """Open connections, the limit and how many alerts were broadcast."""
        broker = broker_of(request)
        return {
            "epoch": broker.epoch,
            "connections": broker.connections,
            "max_connections": broker.max_connections,
            "published": broker.published,
            "closed": broker.closed,
        }

    return router


def install_alert_stream(
    app: FastAPI, store: AnomalyStore, *, max_connections: int = DEFAULT_MAX_CONNECTIONS
) -> AlertBroker:
    """Create a broker, feed it from ``store`` and mount the routes on ``app``.

    Returns:
        The broker; call :meth:`AlertBroker.close` on shutdown.
    """
    broker = AlertBroker(max_connections=max_connections)
    store.subscribe(broker.publish)
    app.state.alert_broker = broker
    app.include_router(build_stream_router())
    return broker
