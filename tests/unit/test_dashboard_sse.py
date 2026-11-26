"""Tests of the SSE alert feed: broker logic and a real server behind the API's middleware (Day 18).

The live tests run uvicorn on a free local port because ASGI test clients buffer a response until it
ends and an event stream never does. The app carries the same middleware stack as ``create_app``
(request id, metrics, rate limit, CORS, GZip), which is what could break streaming.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import CollectorRegistry

from src.api.anomaly_store import AnomalyStore
from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.middleware.request_context import ApiMetrics, MetricsMiddleware, RequestIDMiddleware
from src.dashboard.sse_server import AlertBroker, install_alert_stream
from src.models.ensemble_detector import AnomalyEvent, Severity

T0 = datetime(2026, 11, 26, 8, 0, tzinfo=UTC)


def event(i: int, severity: Severity = Severity.WARNING) -> AnomalyEvent:
    return AnomalyEvent(
        event_id=f"e{i}",
        entity_id=f"s{i}",
        entity_type="warehouse_sensor",
        timestamp=T0 + timedelta(seconds=i),
        anomaly_type="point",
        severity=severity,
        raw_value=float(i),
        expected_range=(2.0, 4.0),
        top_contributing_features=[],
        recommended_action="check",
    )


# --------------------------------------------------------------------------- #
# Broker (no network)
# --------------------------------------------------------------------------- #
async def collect(broker: AlertBroker, n: int, **kwargs: Any) -> list[str]:
    """Read ``n`` frames from a stream (with a safety timeout)."""
    frames: list[str] = []
    stream = broker.stream(**kwargs)
    try:
        async with asyncio.timeout(3):
            async for frame in stream:
                frames.append(frame)
                if len(frames) == n:
                    break
    finally:
        await stream.aclose()
    return frames


def test_broker_validates_arguments():
    with pytest.raises(ValueError):
        AlertBroker(max_connections=0)
    with pytest.raises(ValueError):
        AlertBroker(keepalive_s=0)


def test_publish_assigns_increasing_sequence_numbers():
    broker = AlertBroker()
    assert [broker.publish(event(i)) for i in range(3)] == [1, 2, 3]
    assert broker.published == 3


def test_stream_starts_with_retry_and_hello_then_delivers_live_alerts():
    async def scenario() -> list[str]:
        broker = AlertBroker()
        task = asyncio.create_task(collect(broker, 3))
        await asyncio.sleep(0.05)
        broker.publish(event(1, Severity.CRITICAL))
        broker.publish(event(2))
        return await task

    frames = asyncio.run(scenario())
    assert frames[0].startswith("retry: 3000\nevent: hello\n")
    assert "event: alert" in frames[1] and '"event_id":"e1"' in frames[1]
    assert '"severity":"CRITICAL"' in frames[1]
    assert frames[1].startswith("id: ") and "-1\n" in frames[1]
    assert "-2\n" in frames[2]


def test_new_client_gets_no_history_but_a_resuming_client_gets_what_it_missed():
    broker = AlertBroker()
    for i in range(1, 5):
        broker.publish(event(i))

    fresh = asyncio.run(collect(broker, 1))
    assert len(fresh) == 1 and "event: hello" in fresh[0]  # nothing replayed

    frames = asyncio.run(collect(broker, 3, last_event_id=f"{broker.epoch}-2"))
    assert "event: resync" not in "".join(frames)
    assert [json.loads(f.split("data: ")[1])["event_id"] for f in frames[1:]] == ["e3", "e4"]


@pytest.mark.parametrize("bad_id", ["other-epoch-3", "garbage", "-1"])
def test_unknown_epoch_or_malformed_id_asks_client_to_resync(bad_id):
    broker = AlertBroker()
    broker.publish(event(1))
    frames = asyncio.run(collect(broker, 2, last_event_id=bad_id))
    assert "event: resync" in frames[1]


def test_gap_older_than_history_asks_client_to_resync_and_replays_what_is_kept():
    broker = AlertBroker(history_size=2)
    for i in range(1, 6):
        broker.publish(event(i))
    frames = asyncio.run(collect(broker, 4, last_event_id=f"{broker.epoch}-1"))
    assert "event: resync" in frames[1]
    assert [json.loads(f.split("data: ")[1])["event_id"] for f in frames[2:]] == ["e4", "e5"]


def test_future_id_from_a_previous_run_asks_client_to_resync():
    broker = AlertBroker()
    frames = asyncio.run(collect(broker, 2, last_event_id=f"{broker.epoch}-99"))
    assert "event: resync" in frames[1]


def test_min_severity_filters_live_and_replayed_alerts():
    async def scenario() -> list[str]:
        broker = AlertBroker()
        broker.publish(event(1, Severity.INFO))
        broker.publish(event(2, Severity.CRITICAL))
        task = asyncio.create_task(
            collect(broker, 3, min_severity="CRITICAL", last_event_id=f"{broker.epoch}-0")
        )
        await asyncio.sleep(0.05)
        broker.publish(event(3, Severity.WARNING))  # filtered out
        broker.publish(event(4, Severity.EMERGENCY))
        return await task

    frames = asyncio.run(scenario())
    ids = [json.loads(f.split("data: ")[1])["event_id"] for f in frames[1:]]
    assert ids == ["e2", "e4"]


def test_slow_client_is_cut_off_with_overflow_event():
    async def scenario() -> list[str]:
        broker = AlertBroker(queue_size=2)
        stream = broker.stream()
        first = await stream.__anext__()  # hello; now registered but not reading
        for i in range(10):
            broker.publish(event(i))
        await asyncio.sleep(0.05)
        frames = [first]
        async with asyncio.timeout(3):
            async for frame in stream:
                frames.append(frame)
        return frames

    frames = asyncio.run(scenario())
    assert "event: overflow" in frames[-1]


def test_close_ends_streams_and_refuses_new_connections():
    async def scenario() -> list[str]:
        broker = AlertBroker()
        task = asyncio.create_task(collect(broker, 99))
        await asyncio.sleep(0.05)
        broker.close()
        return await task

    frames = asyncio.run(scenario())
    assert len(frames) == 1  # only the hello frame, then the stream ended
    broker = AlertBroker()
    broker.close()
    assert broker.try_acquire() is None


def test_keepalive_comment_is_sent_when_idle():
    broker = AlertBroker(keepalive_s=0.05)
    frames = asyncio.run(collect(broker, 2))
    assert frames[1] == ": ping\n\n"


def test_connection_slots_are_limited_and_release_is_idempotent():
    broker = AlertBroker(max_connections=2)
    a, b = broker.try_acquire(), broker.try_acquire()
    assert a and b and broker.try_acquire() is None and broker.connections == 2
    a.release()
    a.release()  # second call must not free a slot that is not ours
    assert broker.connections == 1
    assert broker.try_acquire() is not None
    assert broker.try_acquire() is None


def test_publish_from_many_threads_keeps_order_and_loses_nothing():
    async def scenario() -> list[int]:
        broker = AlertBroker(queue_size=1000)
        reader = asyncio.create_task(collect(broker, 201))
        await asyncio.sleep(0.05)

        def burst(start: int) -> None:
            for i in range(start, start + 50):
                broker.publish(event(i))

        threads = [threading.Thread(target=burst, args=(k * 50,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        frames = await reader
        return [int(f.split("\n")[0].split("-")[-1]) for f in frames[1:]]

    seqs = asyncio.run(scenario())
    assert seqs == list(range(1, 201))


def test_store_listener_feeds_the_broker_and_restored_alerts_are_not_broadcast(tmp_path):
    store = AnomalyStore()
    app = FastAPI()
    broker = install_alert_stream(app, store)
    store.add(event(1))
    assert broker.published == 1
    store.snapshot(tmp_path / "snap.json")
    fresh = AnomalyStore()
    fresh_broker = AlertBroker()
    fresh.subscribe(fresh_broker.publish)
    assert fresh.load_snapshot(tmp_path / "snap.json") == 1
    assert fresh_broker.published == 0  # restoring is not news


# --------------------------------------------------------------------------- #
# Live server behind the API's middleware
# --------------------------------------------------------------------------- #
class LiveServer:
    def __init__(self, max_connections: int = 3) -> None:
        self.store = AnomalyStore()
        self.app = FastAPI()
        self.app.add_middleware(GZipMiddleware, minimum_size=1024)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8050"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(RateLimitMiddleware, rules={"/api/v1/predict": 100})
        self.app.add_middleware(MetricsMiddleware, metrics=ApiMetrics(CollectorRegistry()))
        self.app.add_middleware(RequestIDMiddleware)
        self.broker = install_alert_stream(self.app, self.store, max_connections=max_connections)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1/stream"

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self.server.started

    def stop(self) -> None:
        self.broker.close()
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture
def live() -> Iterator[LiveServer]:
    server = LiveServer()
    server.start()
    yield server
    server.stop()


def frames_of(response: httpx.Response) -> Iterator[dict[str, str]]:
    """Parse SSE lines into frames (comments and blank separators handled)."""
    current: dict[str, str] = {}
    for line in response.iter_lines():
        if line == "":
            if current:
                yield current
            current = {}
        elif line.startswith(":"):
            yield {"comment": line[1:].strip()}
        else:
            key, _, value = line.partition(": ")
            current[key] = value


def open_stream(url: str, **kwargs: Any):
    client = httpx.Client(timeout=httpx.Timeout(5.0, read=5.0))
    return client, client.stream("GET", f"{url}/alerts", **kwargs)


def next_alert(frames: Iterator[dict[str, str]]) -> dict[str, str]:
    return next(f for f in frames if f.get("event") == "alert")


def test_live_alert_arrives_within_one_second_through_the_middleware(live):
    client, ctx = open_stream(live.url, headers={"Accept-Encoding": "gzip"})
    with client, ctx as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "content-encoding" not in response.headers  # GZip must not buffer the stream
        assert response.headers["cache-control"] == "no-cache"
        assert "x-request-id" in response.headers
        frames = frames_of(response)
        assert next(frames)["event"] == "hello"
        started = time.monotonic()
        live.store.add(event(7, Severity.CRITICAL))
        frame = next_alert(frames)
        assert time.monotonic() - started < 1.0
        assert json.loads(frame["data"])["event_id"] == "e7"
        assert frame["id"] == f"{live.broker.epoch}-1"


def test_live_reconnect_with_last_event_id_replays_missed_alerts(live):
    for i in range(1, 4):
        live.store.add(event(i))
    client, ctx = open_stream(live.url, headers={"Last-Event-ID": f"{live.broker.epoch}-1"})
    with client, ctx as response:
        frames = frames_of(response)
        assert next(frames)["event"] == "hello"
        got = [json.loads(next_alert(frames)["data"])["event_id"] for _ in range(2)]
    assert got == ["e2", "e3"]


def test_live_last_event_id_query_parameter_works_too(live):
    live.store.add(event(1))
    live.store.add(event(2))
    client, ctx = open_stream(live.url, params={"last_event_id": f"{live.broker.epoch}-1"})
    with client, ctx as response:
        frames = frames_of(response)
        next(frames)
        assert json.loads(next_alert(frames)["data"])["event_id"] == "e2"


def test_live_min_severity_query(live):
    client, ctx = open_stream(live.url, params={"min_severity": "EMERGENCY"})
    with client, ctx as response:
        frames = frames_of(response)
        next(frames)
        live.store.add(event(1, Severity.CRITICAL))
        live.store.add(event(2, Severity.EMERGENCY))
        assert json.loads(next_alert(frames)["data"])["event_id"] == "e2"


def test_live_invalid_min_severity_is_a_422(live):
    response = httpx.get(f"{live.url}/alerts", params={"min_severity": "LOUD"}, timeout=5)
    assert response.status_code == 422


def test_live_connection_limit_returns_503_with_retry_after_and_frees_on_disconnect(live):
    streams = []
    try:
        for _ in range(3):
            client, ctx = open_stream(live.url)
            response = ctx.__enter__()
            frames = frames_of(response)  # keep it: dropping the iterator closes the connection
            streams.append((client, ctx, frames))
            assert response.status_code == 200
            assert next(frames)["event"] == "hello"
        full = httpx.get(f"{live.url}/alerts", timeout=5)
        assert full.status_code == 503 and full.headers["retry-after"] == "5"
        status = httpx.get(f"{live.url}/status", timeout=5).json()
        assert status["connections"] == 3 and status["max_connections"] == 3

        client, ctx, _frames = streams.pop()
        ctx.__exit__(None, None, None)  # client goes away
        client.close()
        deadline = time.monotonic() + 5
        while live.broker.connections > 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert live.broker.connections == 2
        retry_client, retry_ctx = open_stream(live.url)
        with retry_client, retry_ctx as again:
            assert again.status_code == 200
    finally:
        for client, ctx, _frames in streams:
            ctx.__exit__(None, None, None)
            client.close()


def test_live_server_shutdown_ends_open_streams(live):
    client, ctx = open_stream(live.url)
    with client, ctx as response:
        frames = frames_of(response)
        next(frames)
        live.broker.close()
        assert list(frames) == []  # the stream ends instead of hanging


def test_live_cors_preflight_and_headers_allow_the_dashboard_origin(live):
    client, ctx = open_stream(live.url, headers={"Origin": "http://localhost:8050"})
    with client, ctx as response:
        assert response.headers["access-control-allow-origin"] == "http://localhost:8050"


def test_status_endpoint_reports_the_broker(live):
    live.store.add(event(1))
    body = httpx.get(f"{live.url}/status", timeout=5).json()
    assert body["published"] == 1 and body["connections"] == 0 and body["closed"] is False
