"""Unit tests of the dashboard data layer: sources, snapshot, queries, poller (Day 18)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.api.anomaly_store import AnomalyStore
from src.dashboard.data_service import (
    DashboardDataService,
    HttpSource,
    InProcessSource,
    severity_rank,
    slim_alert,
)
from src.dashboard.telemetry import all_entities, entity_timeseries
from src.kafka.feature_cache import FeatureCache
from src.kafka.stream_processor import WindowedAggregator
from src.models.ensemble_detector import AnomalyEvent, Severity

NOW = datetime(2026, 11, 26, 12, 30, tzinfo=UTC)


def alert(
    i: int,
    severity: str = "WARNING",
    minutes_ago: float = 5,
    *,
    resolved: bool = False,
    fp: bool = False,
    entity: str = "s1",
) -> dict[str, Any]:
    return {
        "event_id": f"e{i}",
        "entity_id": entity,
        "entity_type": "warehouse_sensor",
        "timestamp": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
        "anomaly_type": "point",
        "severity": severity,
        "raw_value": 9.0,
        "expected_range": [2.0, 4.0],
        "severity_score": 0.5,
        "is_resolved": resolved,
        "false_positive_marked": fp,
        "occurrence_count": 1,
        "escalated_from": None,
        "top_contributing_features": [{"name": "z", "contribution": 0.4}],
    }


class FakeSource:
    """A scriptable source: set the attributes, flip ``fail`` to make sections raise."""

    def __init__(self) -> None:
        self.alert_rows: list[dict[str, Any]] = []
        self.entity_rows: list[dict[str, Any]] = []
        self.health_row: dict[str, Any] = {
            "status": "ok",
            "models_loaded": True,
            "checks": {"models": "ok"},
        }
        self.feedback_row: dict[str, Any] = {}
        self.pipeline_row: dict[str, Any] = {}
        self.models_row: dict[str, Any] = {"registry": [], "current": {}}
        self.series_calls = 0
        self.fail: set[str] = set()
        self.closed = False

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail:
            raise ConnectionError(f"{name} down")

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        self._maybe_fail("alerts")
        return self.alert_rows[:limit]

    def alert(self, event_id: str) -> dict[str, Any] | None:
        return next((a for a in self.alert_rows if a["event_id"] == event_id), None)

    def entities(self) -> list[dict[str, Any]]:
        self._maybe_fail("entities")
        return self.entity_rows

    def timeseries(self, entity_id: str, lookback_seconds: float) -> dict[str, Any] | None:
        self.series_calls += 1
        self._maybe_fail("timeseries")
        return {"entity_id": entity_id, "points": []}

    def feedback_analytics(self) -> dict[str, Any]:
        self._maybe_fail("feedback")
        return self.feedback_row

    def pipeline(self) -> dict[str, Any]:
        self._maybe_fail("pipeline")
        return self.pipeline_row

    def health(self) -> dict[str, Any]:
        self._maybe_fail("health")
        return self.health_row

    def models(self) -> dict[str, Any]:
        self._maybe_fail("models")
        return self.models_row

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def service(source: FakeSource) -> DashboardDataService:
    return DashboardDataService(source, clock=lambda: NOW)


# --------------------------------------------------------------------------- #
# Snapshot and failure handling
# --------------------------------------------------------------------------- #
def test_refresh_builds_snapshot_and_bumps_version_only_on_change(source, service):
    source.alert_rows = [alert(1)]
    first = service.refresh()
    assert first.connected and len(first.alerts) == 1 and first.version == 1
    assert service.refresh().version == 1  # nothing changed
    source.alert_rows = [alert(1), alert(2)]
    assert service.refresh().version == 2


def test_failed_section_keeps_last_value_and_recovers(source, service):
    source.alert_rows = [alert(1)]
    service.refresh()
    source.fail = {"alerts"}
    snap = service.refresh()
    assert not snap.connected
    assert "alerts" in snap.errors and "ConnectionError" in snap.errors["alerts"]
    assert len(snap.alerts) == 1  # last good value is kept
    source.fail = set()
    snap = service.refresh()
    assert snap.connected and not snap.errors


def test_non_core_failure_keeps_connected(source, service):
    service.refresh()
    source.fail = {"models", "feedback"}
    snap = service.refresh()
    assert snap.connected
    assert set(snap.errors) == {"models", "feedback"}


def test_unreachable_source_gives_empty_but_valid_queries(source, service):
    source.fail = {"alerts", "health", "entities", "pipeline", "feedback", "models"}
    snap = service.refresh()
    assert not snap.connected
    assert service.get_active_anomalies() == []
    metrics = service.get_summary_metrics()
    assert metrics["connected"] is False and metrics["api_status"] == "unreachable"
    assert metrics["model_health_score"] == pytest.approx(0.4 * 0 + 0.25 + 0.20 + 0.15)
    assert len(service.get_alert_trend(6)) == 6
    assert service.get_model_performance()["models"] == []


def test_reference_now_follows_event_time_when_ahead_of_wall_clock(source, service):
    source.alert_rows = [{**alert(1), "timestamp": (NOW + timedelta(hours=3)).isoformat()}]
    snap = service.refresh()
    assert snap.reference_now == NOW + timedelta(hours=3)
    trend = service.get_alert_trend(2)
    assert trend[-1]["total"] == 1  # the future-dated alert is in the last bucket


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def test_active_anomalies_sorted_by_severity_then_recency_and_filtered(source, service):
    source.alert_rows = [
        alert(1, "WARNING", 1),
        alert(2, "CRITICAL", 30),
        alert(3, "CRITICAL", 10),
        alert(4, "EMERGENCY", 50, resolved=True),
        alert(5, "EMERGENCY", 20, fp=True),
        alert(6, "INFO", 2),
        alert(7, "EMERGENCY", 60 * 30),  # older than the 24 h active window
    ]
    service.refresh()
    ids = [a["event_id"] for a in service.get_active_anomalies()]
    assert ids == ["e3", "e2", "e1", "e6"]
    assert [a["event_id"] for a in service.get_active_anomalies(2)] == ["e3", "e2"]
    assert service.get_active_anomalies(0) == []


def test_summary_metrics_counts_and_rates(source, service):
    source.alert_rows = [
        alert(1, "CRITICAL", 10),
        alert(2, "EMERGENCY", 20, fp=True),  # a false positive is not active
        alert(3, "WARNING", 90),
        alert(4, "CRITICAL", 60 * 14),  # yesterday (UTC day) -> not today, still active
    ]
    source.feedback_row = {"windows": {"last_24h": {"total": 10, "false_positive_rate": 0.1}}}
    source.pipeline_row = {"latency": {"p99_ms": 120.0}, "latency_target_ms": 500.0}
    service.refresh()
    metrics = service.get_summary_metrics()
    assert metrics["total_alerts_today"] == 3
    assert metrics["alerts_last_hour"] == 2
    assert metrics["active_critical"] == 2
    assert metrics["fp_rate_24h"] == 0.1
    assert metrics["avg_latency_p99"] == 120.0
    assert metrics["connected"] is True and metrics["api_status"] == "ok"
    assert metrics["stored_alerts"] == 4


def test_summary_has_no_fp_rate_or_latency_without_data(source, service):
    service.refresh()
    metrics = service.get_summary_metrics()
    assert metrics["fp_rate_24h"] is None and metrics["avg_latency_p99"] is None
    assert metrics["model_health_score"] == pytest.approx(1.0)


def test_alert_trend_is_zero_filled_and_bucketed_by_severity(source, service):
    source.alert_rows = [
        alert(1, "CRITICAL", 5),  # 12:25 -> hour 12
        alert(2, "WARNING", 40),  # 11:50 -> hour 11
        alert(3, "WARNING", 45),
        alert(4, "INFO", 60 * 30),  # outside 24 h
    ]
    service.refresh()
    trend = service.get_alert_trend(24)
    assert len(trend) == 24
    assert trend[-1]["hour"] == "2026-11-26T12:00:00Z" and trend[-1]["CRITICAL"] == 1
    assert trend[-2]["WARNING"] == 2 and trend[-2]["total"] == 2
    assert sum(row["total"] for row in trend) == 3
    assert all(set(row) >= {"INFO", "WARNING", "CRITICAL", "EMERGENCY"} for row in trend)


@pytest.mark.parametrize(
    ("health", "feedback", "pipeline", "expected"),
    [
        ({"models_loaded": True, "checks": {"models": "degraded"}}, {}, {}, 0.8),
        ({"models_loaded": False}, {}, {}, 0.6),
        (
            {"models_loaded": True, "checks": {"models": "ok"}},
            {"windows": {"last_24h": {"total": 20, "false_positive_rate": 0.15}}},
            {},
            1 - 0.25 * 0.5,
        ),
        (
            {"models_loaded": True, "checks": {"models": "ok"}},
            {},
            {"latency": {"p99_ms": 750.0}, "latency_target_ms": 500.0},
            1 - 0.2 * 0.5,
        ),
        (
            {"models_loaded": True, "checks": {"models": "ok"}},
            {"drift": {"last_alert": {"detected_at": (NOW - timedelta(minutes=10)).isoformat()}}},
            {},
            0.85,
        ),
        (
            {"models_loaded": True, "checks": {"models": "ok"}},
            {"drift": {"last_alert": {"detected_at": (NOW - timedelta(hours=3)).isoformat()}}},
            {},
            1.0,
        ),
    ],
)
def test_model_health_score_components(source, service, health, feedback, pipeline, expected):
    source.health_row, source.feedback_row, source.pipeline_row = health, feedback, pipeline
    service.refresh()
    assert service.model_health()["score"] == pytest.approx(expected, abs=1e-3)


def test_model_performance_prefers_active_version_and_finds_ensemble(source, service):
    def version(v: str, active: bool, f1: float) -> dict[str, Any]:
        return {
            "version": v,
            "active": active,
            "created_at": "2026-11-25T08:00:00Z",
            "metrics": {"f1": f1},
        }

    source.models_row = {
        "registry": [
            {
                "model_type": "lstm_classifier",
                "versions": [version("v3", False, 0.9), version("v2", True, 0.8)],
            },
            {"model_type": "ensemble_meta_learner", "versions": [version("v1", False, 0.95)]},
            {"model_type": "empty", "versions": []},
        ],
        "current": {"degraded": False},
    }
    service.refresh()
    perf = service.get_model_performance()
    by_type = {m["model_type"]: m for m in perf["models"]}
    assert by_type["lstm_classifier"]["version"] == "v2" and by_type["lstm_classifier"]["active"]
    assert "empty" not in by_type
    assert perf["ensemble"] == {"f1": 0.95}
    assert perf["current"] == {"degraded": False}


def test_timeseries_is_cached_briefly_and_failures_give_none(source, service):
    assert service.get_sensor_timeseries("s1") == {"entity_id": "s1", "points": []}
    service.get_sensor_timeseries("s1")
    assert source.series_calls == 1
    service.get_sensor_timeseries("s1", 60)  # other lookback is another key
    assert source.series_calls == 2
    source.fail = {"timeseries"}
    assert service.get_sensor_timeseries("s2") is None


def test_get_alert_delegates_and_swallows_errors(source, service):
    source.alert_rows = [alert(1)]
    assert service.get_alert("e1")["event_id"] == "e1"
    assert service.get_alert("nope") is None

    def boom(_: str) -> None:
        raise RuntimeError("x")

    source.alert = boom  # type: ignore[method-assign]
    assert service.get_alert("e1") is None


def test_helpers():
    assert severity_rank("critical") == 2 and severity_rank("??") == -1
    slim = slim_alert(alert(1))
    assert "top_contributing_features" not in slim and slim["event_id"] == "e1"


def test_constructor_validates():
    with pytest.raises(ValueError):
        DashboardDataService(FakeSource(), poll_interval_s=0)
    with pytest.raises(ValueError):
        DashboardDataService(FakeSource(), alert_limit=0)


# --------------------------------------------------------------------------- #
# Poller thread
# --------------------------------------------------------------------------- #
def _wait_for(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_poller_picks_up_changes_and_stops_cleanly(source):
    service = DashboardDataService(source, poll_interval_s=0.05)
    service.start()
    service.start()  # idempotent
    try:
        assert _wait_for(lambda: service.snapshot().connected)
        source.alert_rows = [alert(1, "CRITICAL", 1)]
        assert _wait_for(lambda: len(service.snapshot().alerts) == 1)
        assert sum(t.name == "scad-dashboard-poller" for t in threading.enumerate()) == 1
    finally:
        service.stop()
    assert not service.running and source.closed


def test_poller_survives_outage_and_recovers(source):
    service = DashboardDataService(source, poll_interval_s=0.05)
    source.fail = {"alerts", "health"}
    service.start()
    try:
        assert _wait_for(lambda: "alerts" in service.snapshot().errors)
        assert not service.snapshot().connected and service.running
        source.fail = set()
        assert _wait_for(lambda: service.snapshot().connected, timeout=8.0)
    finally:
        service.stop()


# --------------------------------------------------------------------------- #
# Sources against the real store and cache
# --------------------------------------------------------------------------- #
def make_event(i: int, minutes: int, severity: Severity = Severity.WARNING) -> AnomalyEvent:
    when = NOW - timedelta(minutes=minutes)
    return AnomalyEvent(
        event_id=f"e{i}",
        entity_id="sensor-1",
        entity_type="warehouse_sensor",
        timestamp=when,
        anomaly_type="point",
        severity=severity,
        raw_value=9.0,
        expected_range=(2.0, 4.0),
        top_contributing_features=[],
        recommended_action="check",
    )


def feed(cache: FeatureCache, entity: str, values: list[float], start: float) -> None:
    aggregator = WindowedAggregator()
    ctx = {"zone": "A", "unit": "celsius", "acceptable_range": (2.0, 4.0), "warehouse_id": "w1"}
    for i, value in enumerate(values):
        outcome = aggregator.update(entity, "warehouse", start + i, {"value": value}, context=ctx)
        for window in outcome.features:
            cache.store_features(entity, window, 3600)


@pytest.fixture
def live() -> SimpleNamespace:
    cache = FeatureCache(clock=lambda: 0.0)
    feed(cache, "sensor-1", [3.0, 3.1, 3.2, 3.3, 9.0], start=NOW.timestamp() - 4)
    feed(cache, "sensor-2", [20.0, 20.5], start=NOW.timestamp() - 1)
    store = AnomalyStore()
    for i, minutes in enumerate((30, 20, 10)):
        store.add(make_event(i, minutes))
    return SimpleNamespace(anomaly_store=store, cache=cache)


def test_in_process_source_serves_alerts_entities_and_series(live):
    source = InProcessSource(live)
    alerts = source.alerts(2)
    assert [a["event_id"] for a in alerts] == ["e2", "e1"]  # newest first, limited
    assert alerts[0]["severity"] == "WARNING" and alerts[0]["expected_range"] == [2.0, 4.0]
    assert source.alert("e0")["event_id"] == "e0" and source.alert("zzz") is None

    entities = {e["entity_id"]: e for e in source.entities()}
    assert set(entities) == {"sensor-1", "sensor-2"}
    assert entities["sensor-1"]["value"] == 9.0 and entities["sensor-1"]["zone"] == "A"
    assert entities["sensor-1"]["expected_range"] == [2.0, 4.0]
    assert entities["sensor-1"]["unit"] == "celsius"

    series = source.timeseries("sensor-1", 300)
    assert [p["v"] for p in series["points"]] == [3.0, 3.1, 3.2, 3.3, 9.0]
    assert source.timeseries("ghost", 300) is None
    assert source.pipeline() == {}  # no engine


def test_timeseries_lookback_is_measured_from_newest_point(live):
    series = entity_timeseries(live.cache, "sensor-1", lookback_seconds=2)
    assert [p["v"] for p in series["points"]] == [3.2, 3.3, 9.0]
    assert series["lookback_seconds"] == 2


def test_dashboard_reads_do_not_count_as_cache_hits_or_misses(live):
    live.cache.get_features("sensor-1")  # one real hit
    live.cache.get_features("ghost")  # one real miss
    before = dict(live.cache.stats())
    all_entities(live.cache)
    entity_timeseries(live.cache, "sensor-1")
    entity_timeseries(live.cache, "ghost")
    assert live.cache.stats() == before
    assert live.cache.hit_ratio == 0.5


def test_peek_returns_oldest_first_and_respects_n(live):
    windows = live.cache.peek("sensor-1", 3)
    assert len(windows) == 3
    assert windows[0].window_end <= windows[-1].window_end
    assert live.cache.peek("sensor-1", 0) == [] and live.cache.peek("ghost", 5) == []


def test_service_over_in_process_source(live):
    live.models = SimpleNamespace(is_loaded=True, active=SimpleNamespace(degraded=True))
    live.feedback_processor = SimpleNamespace(analytics=lambda: {"windows": {}})
    source = InProcessSource(live)
    health = source.health()
    assert health["status"] == "degraded" and health["checks"] == {"models": "degraded"}
    assert health["stored_anomalies"] == 3

    # models() reuses the API router, which needs torch: cover the other sections here.
    source.models = lambda: {"registry": [], "current": {}}  # type: ignore[method-assign]
    service = DashboardDataService(source, clock=lambda: NOW)
    snap = service.refresh()
    assert snap.connected and len(snap.alerts) == 3 and len(snap.entities) == 2
    assert service.get_entities("warehouse") == list(snap.entities)
    assert service.get_entities("vehicle") == []
    assert service.get_sensor_timeseries("sensor-2")["points"][-1]["v"] == 20.5
    assert service.model_health()["components"]["availability"] == 0.5


# --------------------------------------------------------------------------- #
# HttpSource
# --------------------------------------------------------------------------- #
def make_http_source(handler) -> HttpSource:
    source = HttpSource("http://api.test")
    source._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://api.test"
    )
    return source


def test_http_source_maps_endpoints_and_tolerates_404():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            request.url.path + ("?" + request.url.query.decode() if request.url.query else "")
        )
        path = request.url.path
        if path == "/api/v1/anomalies":
            return httpx.Response(200, json={"total": 1, "items": [alert(1)]})
        if path == "/api/v1/anomalies/e1":
            return httpx.Response(200, json=alert(1))
        if path == "/api/v1/telemetry/entities":
            return httpx.Response(200, json={"count": 0, "items": []})
        if path == "/api/v1/models":
            return httpx.Response(200, json={"registry_path": "r", "models": [{"model_type": "m"}]})
        if path == "/api/v1/models/current":
            return httpx.Response(200, json={"degraded": False})
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"detail": "nope"})

    source = make_http_source(handler)
    assert source.alerts(5000)[0]["event_id"] == "e1"
    assert "limit=1000" in seen[0]  # capped at the API maximum
    assert source.alert("e1")["event_id"] == "e1" and source.alert("zzz") is None
    assert source.entities() == []
    assert source.timeseries("ghost", 60) is None
    assert source.pipeline() == {}  # the API runs without the streaming runner
    assert source.health() == {"status": "ok"}
    assert source.models() == {"registry": [{"model_type": "m"}], "current": {"degraded": False}}
    source.close()


def test_http_source_raises_on_server_error():
    source = make_http_source(lambda request: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        source.health()
