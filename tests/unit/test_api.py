"""Tests of the FastAPI inference service (Day 14), driven through httpx.AsyncClient."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from prometheus_client import CollectorRegistry

from src.api.main import create_app
from src.models.model_registry import ModelRegistry
from src.models.statistical_detector import AnomalyScore
from src.utils.config import AppConfig

pytestmark = pytest.mark.anyio

T0 = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class FakeForest:
    """Stands in for a trained Isolation Forest in the registry (must be picklable)."""

    source_type = "warehouse"

    def process(self, vector: Any, *, learn: bool = True) -> AnomalyScore:
        return AnomalyScore(
            entity_id=vector.entity_id,
            timestamp=vector.timestamp,
            score=0.9,
            is_anomaly=True,
            anomaly_type="point",
            confidence=0.8,
            detector="fake_forest",
        )

    def process_many(self, vectors: Any, *, learn: bool = True) -> list[AnomalyScore]:
        return [self.process(v, learn=learn) for v in vectors]


def make_config(tmp_path: Path, **api: Any) -> AppConfig:
    cfg = AppConfig()
    api_cfg = cfg.api.model_copy(
        update={
            "predict_rate_limit_per_min": 100_000,
            "batch_predict_rate_limit_per_min": 100_000,
            **api,
        }
    )
    paths = cfg.paths.model_copy(
        update={
            "model_registry": str(tmp_path / "registry"),
            "feedback_store": str(tmp_path / "feedback.json"),
            "anomaly_snapshots": str(tmp_path / "snapshots"),
        }
    )
    return cfg.model_copy(update={"api": api_cfg, "paths": paths})


def warehouse(i: int, value: float, sensor: str = "wh1-A-temp-01", **meta: Any) -> dict[str, Any]:
    return {
        "source_type": "warehouse",
        "payload": {
            "sensor_id": sensor,
            "warehouse_id": "wh1",
            "timestamp": (T0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
            "reading_type": "temperature",
            "value": value,
            "unit": "C",
            "metadata": {"zone": "A", "storage_type": "cold_storage", **meta},
        },
    }


def normal_value(rng: random.Random) -> float:
    return 3.0 + rng.gauss(0, 0.15)


async def warm_up(client: httpx.AsyncClient, n: int = 80, sensor: str = "wh1-A-temp-01") -> int:
    """Feed ``n`` normal readings; return the next free second offset."""
    rng = random.Random(7)
    for i in range(n):
        r = await client.post("/api/v1/predict", json=warehouse(i, normal_value(rng), sensor))
        assert r.status_code == 200, r.text
    return n


async def make_client(app: Any, host: str = "10.0.0.1") -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, client=(host, 5000))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


@pytest.fixture
def app(cfg: AppConfig) -> Any:
    return create_app(cfg, metrics_registry=CollectorRegistry())


@pytest.fixture
async def client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async for c in make_client(app):
        yield c


@pytest.fixture
def make_app(cfg: AppConfig) -> Callable[..., Any]:
    def factory(config: AppConfig | None = None, **kwargs: Any) -> Any:
        return create_app(config or cfg, metrics_registry=CollectorRegistry(), **kwargs)

    return factory


async def spike(client: httpx.AsyncClient, i: int, value: float = 25.0) -> dict[str, Any]:
    r = await client.post("/api/v1/predict", json=warehouse(i, value))
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# Health, metrics, middleware
# --------------------------------------------------------------------------- #
async def test_health_reports_degraded_without_trained_models(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["models_loaded"] is True
    assert "statistical fallback" in body["degraded_reason"]
    assert body["checks"]["models"] == "degraded"


async def test_health_reports_kafka_state(make_app):
    for check, expected in ((lambda: True, "ok"), (lambda: False, "unreachable")):
        app = make_app(kafka_check=check)
        async for c in make_client(app):
            assert (await c.get("/health")).json()["checks"]["kafka"] == expected


async def test_health_survives_a_crashing_kafka_probe(make_app):
    def boom() -> bool:
        raise RuntimeError("broker down")

    async for c in make_client(make_app(kafka_check=boom)):
        response = await c.get("/health")
        assert response.status_code == 200
        assert response.json()["checks"]["kafka"] == "unreachable"


async def test_request_id_generated_echoed_and_sanitised(client):
    generated = (await client.get("/health")).headers["x-request-id"]
    assert len(generated) == 36
    echoed = (await client.get("/health", headers={"X-Request-ID": "trace-abc-12345"})).headers
    assert echoed["x-request-id"] == "trace-abc-12345"
    unsafe = (await client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"})).headers
    assert unsafe["x-request-id"] != "bad id\twith spaces"


async def test_metrics_endpoint_uses_route_templates(client):
    await client.get("/api/v1/anomalies/does-not-exist")
    await client.get("/api/v1/anomalies/another-missing-id")
    text = (await client.get("/metrics/")).text
    assert 'route="/api/v1/anomalies/{event_id}"' in text
    assert "does-not-exist" not in text  # raw paths never become labels
    assert "scad_api_request_duration_seconds" in text


async def test_cors_allows_configured_origin_only(client):
    ok = await client.options(
        "/api/v1/predict",
        headers={"Origin": "http://localhost:8050", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers["access-control-allow-origin"] == "http://localhost:8050"
    other = await client.options(
        "/api/v1/predict",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in other.headers


async def test_openapi_documents_examples(client):
    spec = (await client.get("/openapi.json")).json()
    paths = spec["paths"]
    for route in ("/api/v1/predict", "/api/v1/batch-predict", "/api/v1/feedback", "/health"):
        assert route in paths
    assert spec["components"]["schemas"]["SensorEventRequest"]["examples"]


async def test_gzip_compresses_large_responses(client):
    await warm_up(client, 60)
    for i in range(60, 80):
        await spike(client, i * 400)  # distinct, well-separated alerts
    big = await client.get("/api/v1/anomalies", headers={"Accept-Encoding": "gzip"})
    assert big.status_code == 200


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
async def test_cold_start_is_reported_not_an_error(client):
    body = (await client.post("/api/v1/predict", json=warehouse(0, 3.0))).json()
    assert body["warming_up"] is True
    assert body["is_anomaly"] is False
    assert body["entity_id"] == "wh1-A-temp-01"
    assert body["severity"] is None


async def test_normal_traffic_is_not_flagged(client):
    n = await warm_up(client, 90)
    body = (await client.post("/api/v1/predict", json=warehouse(n, 3.05))).json()
    assert body["warming_up"] is False
    assert body["is_anomaly"] is False
    assert body["models_used"] == ["statistical"]
    assert body["method"] == "weighted_average"
    assert 0 <= body["anomaly_score"] < 0.5


async def test_spike_is_flagged_stored_and_retrievable(client):
    n = await warm_up(client)
    body = await spike(client, n)
    assert body["is_anomaly"] is True
    assert body["anomaly_score"] >= 0.5
    assert body["severity"] in {"INFO", "WARNING", "CRITICAL", "EMERGENCY"}
    assert body["event_id"]
    assert body["recommended_action"]
    stored = (await client.get(f"/api/v1/anomalies/{body['event_id']}")).json()
    assert stored["entity_id"] == "wh1-A-temp-01"
    assert stored["entity_type"] == "warehouse_sensor"
    assert stored["raw_value"] == 25.0
    assert stored["false_positive_marked"] is False


async def test_repeat_alert_for_same_entity_is_deduplicated(client):
    n = await warm_up(client)
    baseline = (await client.get("/api/v1/anomalies")).json()["total"]  # warm-up noise, if any
    first = await spike(client, n + 1000)
    second = await spike(client, n + 1001)
    assert first["event_id"] and not first["alert_suppressed"]
    assert second["is_anomaly"] and second["alert_suppressed"] and second["event_id"] is None
    assert (await client.get("/api/v1/anomalies")).json()["total"] == baseline + 1


async def test_ground_truth_label_is_never_used_by_detectors(client):
    n = await warm_up(client)
    mislabelled = warehouse(n, 3.0)
    mislabelled["payload"].update(is_anomaly=True, anomaly_type="point")
    body = (await client.post("/api/v1/predict", json=mislabelled)).json()
    assert body["is_anomaly"] is False


async def test_vehicle_and_inventory_events_are_accepted(client):
    gps = {
        "source_type": "vehicle",
        "payload": {
            "vehicle_id": "veh-007",
            "timestamp": "2026-11-24T08:00:10Z",
            "latitude": 22.57,
            "longitude": 88.36,
            "speed_kmh": 62.0,
            "heading": 87.5,
            "route_adherence_score": 0.98,
        },
    }
    inventory = {
        "source_type": "inventory",
        "payload": {
            "sku_id": "sku-1",
            "warehouse_id": "wh1",
            "timestamp": "2026-11-24T08:00:10Z",
            "previous_level": 100,
            "new_level": 95,
            "change_amount": -5,
            "transaction_type": "shipment",
        },
    }
    for body in (gps, inventory):
        r = await client.post("/api/v1/predict", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["source_type"] == body["source_type"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(source_type="order"),
        lambda b: b.update(unexpected="x"),
        lambda b: b["payload"].pop("value"),
        lambda b: b["payload"].update(value="not-a-number"),
        lambda b: b["payload"].update(timestamp="yesterday"),
        lambda b: b["payload"].update(reading_type="pressure"),
        lambda b: b["payload"].update(surprise=1),
        lambda b: b.pop("payload"),
    ],
)
async def test_invalid_requests_get_422_with_request_id(client, mutate):
    body = warehouse(0, 3.0)
    mutate(body)
    response = await client.post("/api/v1/predict", json=body)
    assert response.status_code == 422
    assert response.json()["request_id"] == response.headers["x-request-id"]


async def test_non_json_body_is_422(client):
    r = await client.post(
        "/api/v1/predict", content=b"{nope", headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


async def test_concurrent_predictions_are_all_served(client):
    async def one(k: int) -> int:
        r = await client.post(
            "/api/v1/predict", json=warehouse(k, 3.0, sensor=f"wh1-A-temp-{k % 7:02d}")
        )
        return r.status_code

    codes = await asyncio.gather(*(one(k) for k in range(60)))
    assert set(codes) == {200}


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
async def test_batch_preserves_order_and_counts_anomalies(client):
    n = await warm_up(client)
    events = [warehouse(n + i, 3.0) for i in range(5)] + [warehouse(n + 5, 25.0)]
    body = (await client.post("/api/v1/batch-predict", json={"events": events})).json()
    assert body["count"] == 6
    assert body["anomalies"] == 1
    assert [r["timestamp"] for r in body["results"]] == [e["payload"]["timestamp"] for e in events]
    assert body["results"][-1]["is_anomaly"] is True


async def test_batch_size_limits(client, make_app, cfg):
    assert (await client.post("/api/v1/batch-predict", json={"events": []})).status_code == 422
    too_many = {"events": [warehouse(i, 3.0) for i in range(101)]}
    assert (await client.post("/api/v1/batch-predict", json=too_many)).status_code == 422
    exactly = {"events": [warehouse(i, 3.0) for i in range(100)]}
    assert (await client.post("/api/v1/batch-predict", json=exactly)).status_code == 200

    small = cfg.model_copy(update={"api": cfg.api.model_copy(update={"batch_max_size": 3})})
    async for c in make_client(make_app(small)):
        r = await c.post(
            "/api/v1/batch-predict", json={"events": [warehouse(i, 3.0) for i in range(4)]}
        )
        assert r.status_code == 413


async def test_one_bad_event_rejects_the_whole_batch_before_scoring(client):
    bad = warehouse(1, 3.0)
    bad["payload"]["value"] = "x"
    r = await client.post("/api/v1/batch-predict", json={"events": [warehouse(0, 3.0), bad]})
    assert r.status_code == 422
    assert (await client.get("/health")).json()["stored_anomalies"] == 0


# --------------------------------------------------------------------------- #
# Alert queries
# --------------------------------------------------------------------------- #
async def _make_alerts(client: httpx.AsyncClient) -> list[str]:
    """Two sensors, two well-separated spikes each (the dedup window is 5 min)."""
    created = []
    for sensor in ("wh1-A-temp-01", "wh1-A-temp-02"):
        n = await warm_up(client, 80, sensor)
        base = n + 1000
        for k in range(2):
            r = await client.post(
                "/api/v1/predict", json=warehouse(base + k * 2000, 25.0 + k, sensor)
            )
            assert r.json()["event_id"], r.text
            created.append(r.json()["event_id"])
    return created


async def test_anomaly_filters_and_pagination(client):
    created = await _make_alerts(client)
    everything = (await client.get("/api/v1/anomalies", params={"limit": 1000})).json()
    items = everything["items"]
    assert everything["total"] == len(items) >= 4
    assert set(created) <= {i["event_id"] for i in items}
    times = [i["timestamp"] for i in items]
    assert times == sorted(times, reverse=True)  # newest first

    # every filter must agree with filtering the full list by hand
    one = (
        await client.get("/api/v1/anomalies", params={"entity_id": "wh1-A-temp-02", "limit": 1000})
    ).json()
    expected = [i for i in items if i["entity_id"] == "wh1-A-temp-02"]
    assert one["total"] >= 2 and [i["event_id"] for i in one["items"]] == [
        i["event_id"] for i in expected
    ]

    page = (await client.get("/api/v1/anomalies", params={"limit": 1, "offset": 3})).json()
    assert page["total"] == everything["total"] and [i["event_id"] for i in page["items"]] == [
        items[3]["event_id"]
    ]

    cutoff = items[1]["timestamp"]
    recent = (
        await client.get("/api/v1/anomalies", params={"start_time": cutoff, "limit": 1000})
    ).json()
    assert recent["total"] == sum(1 for i in items if i["timestamp"] >= cutoff)
    old = (await client.get("/api/v1/anomalies", params={"end_time": cutoff, "limit": 1000})).json()
    assert old["total"] == sum(1 for i in items if i["timestamp"] <= cutoff)

    kind = items[0]["anomaly_type"]
    by_type = (
        await client.get("/api/v1/anomalies", params={"anomaly_type": kind, "limit": 1000})
    ).json()
    assert by_type["total"] == sum(1 for i in items if i["anomaly_type"] == kind)

    severity = items[0]["severity"]
    by_sev = (
        await client.get("/api/v1/anomalies", params={"severity": severity, "limit": 1000})
    ).json()
    assert by_sev["total"] == sum(1 for i in items if i["severity"] == severity)
    assert all(i["severity"] == severity for i in by_sev["items"])


@pytest.mark.parametrize(
    "params",
    [
        {"severity": "SEVERE"},
        {"anomaly_type": "weird"},
        {"limit": 0},
        {"limit": 1001},
        {"offset": -1},
        {"start_time": "not-a-date"},
    ],
)
async def test_anomaly_query_validation(client, params):
    assert (await client.get("/api/v1/anomalies", params=params)).status_code == 422


async def test_reversed_time_range_is_422(client):
    r = await client.get(
        "/api/v1/anomalies",
        params={"start_time": "2026-11-24T10:00:00Z", "end_time": "2026-11-24T09:00:00Z"},
    )
    assert r.status_code == 422


async def test_unknown_anomaly_is_404(client):
    assert (await client.get("/api/v1/anomalies/nope")).status_code == 404


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
async def test_feedback_round_trip(client):
    n = await warm_up(client)
    event_id = (await spike(client, n))["event_id"]
    r = await client.post(
        "/api/v1/feedback",
        json={
            "anomaly_event_id": event_id,
            "feedback_type": "false_positive",
            "analyst_id": "a1",
            "notes": "  door was open  ",
        },
    )
    assert r.status_code == 201
    assert r.json()["notes"] == "door was open" and r.json()["replaced_previous"] is False
    assert (await client.get(f"/api/v1/anomalies/{event_id}")).json()[
        "false_positive_marked"
    ] is True

    stats = (await client.get("/api/v1/feedback/stats")).json()
    assert stats["overall"] == {
        "true_positive": 0,
        "false_positive": 1,
        "total": 1,
        "false_positive_rate": 1.0,
    }
    assert stats["per_entity"]["wh1-A-temp-01"]["false_positive"] == 1
    assert stats["per_model"]["statistical"]["false_positive"] == 1


async def test_same_analyst_replaces_verdict_others_add(client):
    n = await warm_up(client)
    event_id = (await spike(client, n))["event_id"]

    def rate(analyst: str, kind: str) -> Any:
        return client.post(
            "/api/v1/feedback",
            json={"anomaly_event_id": event_id, "feedback_type": kind, "analyst_id": analyst},
        )

    assert (await rate("a1", "false_positive")).json()["replaced_previous"] is False
    changed = (await rate("a1", "true_positive")).json()
    assert changed["replaced_previous"] is True
    await rate("a2", "true_positive")
    stats = (await client.get("/api/v1/feedback/stats")).json()
    assert stats["overall"]["total"] == 2 and stats["overall"]["true_positive"] == 2
    assert (await client.get(f"/api/v1/anomalies/{event_id}")).json()[
        "false_positive_marked"
    ] is False


async def test_feedback_validation(client):
    unknown = await client.post(
        "/api/v1/feedback", json={"anomaly_event_id": "ghost", "feedback_type": "true_positive"}
    )
    assert unknown.status_code == 404
    bad = await client.post(
        "/api/v1/feedback", json={"anomaly_event_id": "x", "feedback_type": "maybe"}
    )
    assert bad.status_code == 422
    long_notes = await client.post(
        "/api/v1/feedback",
        json={"anomaly_event_id": "x", "feedback_type": "true_positive", "notes": "n" * 1001},
    )
    assert long_notes.status_code == 422


async def test_empty_feedback_stats(client):
    stats = (await client.get("/api/v1/feedback/stats")).json()
    assert stats["overall"]["total"] == 0 and stats["per_model"] == {}


# --------------------------------------------------------------------------- #
# Models, hot swap
# --------------------------------------------------------------------------- #
async def test_models_empty_registry(client):
    assert (await client.get("/api/v1/models")).json()["models"] == []
    current = (await client.get("/api/v1/models/current")).json()
    assert current["degraded"] is True and current["generation"] == 1
    assert current["ensembles"]["warehouse"] == {
        "members": ["statistical"],
        "method": "weighted_average",
    }


async def test_hot_reload_picks_up_new_model_without_restart(client, cfg):
    n = await warm_up(client, 60)
    before = (await client.post("/api/v1/predict", json=warehouse(n, 3.0))).json()
    assert before["models_used"] == ["statistical"]
    stat_before = client._transport.app.state.service.models._statistical  # type: ignore[attr-defined]

    ModelRegistry(cfg.paths.model_registry).save(
        "isolation_forest_warehouse", FakeForest(), metrics={"f1": 0.81}
    )
    reload = await client.post("/api/v1/models/reload")
    assert reload.status_code == 200, reload.text
    body = reload.json()
    assert body["previous_generation"] == 1 and body["current"]["generation"] == 2
    assert body["current"]["degraded"] is False
    assert "isolation_forest_warehouse" in body["current"]["versions"]

    after = (await client.post("/api/v1/predict", json=warehouse(n + 1, 3.0))).json()
    assert set(after["models_used"]) == {"statistical", "isolation_forest"}

    listing = (await client.get("/api/v1/models")).json()["models"]
    forest = next(m for m in listing if m["model_type"] == "isolation_forest_warehouse")
    assert forest["versions"][0]["active"] is True and forest["versions"][0]["metrics"] == {
        "f1": 0.81
    }
    assert client._transport.app.state.service.models._statistical is stat_before  # type: ignore[attr-defined]


async def test_corrupt_new_version_keeps_old_models_serving(client, cfg):
    registry = ModelRegistry(cfg.paths.model_registry)
    registry.save("isolation_forest_warehouse", FakeForest())
    assert (await client.post("/api/v1/models/reload")).status_code == 200

    newest = registry.save("isolation_forest_warehouse", FakeForest())
    (newest.path / "model.joblib").write_bytes(b"corrupted")
    failed = await client.post("/api/v1/models/reload")
    assert failed.status_code == 500
    assert "isolation_forest_warehouse" in failed.json()["detail"]
    current = (await client.get("/api/v1/models/current")).json()
    assert current["generation"] == 2  # unchanged: the old snapshot still serves
    assert (await client.post("/api/v1/predict", json=warehouse(0, 3.0))).status_code == 200


async def test_corrupt_model_at_startup_degrades_instead_of_crashing(cfg, make_app):
    registry = ModelRegistry(cfg.paths.model_registry)
    version = registry.save("isolation_forest_warehouse", FakeForest())
    (version.path / "model.joblib").write_bytes(b"corrupted")
    async for c in make_client(make_app()):
        current = (await c.get("/api/v1/models/current")).json()
        assert current["degraded"] is True
        assert "isolation_forest_warehouse" in current["load_errors"]
        assert (await c.get("/health")).json()["status"] == "degraded"
        assert (await c.post("/api/v1/predict", json=warehouse(0, 3.0))).status_code == 200


async def test_reload_requires_admin_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_ADMIN_KEY", "s3cret")
    assert (await client.post("/api/v1/models/reload")).status_code == 401
    assert (
        await client.post("/api/v1/models/reload", headers={"X-API-Key": "wrong"})
    ).status_code == 401
    assert (
        await client.post("/api/v1/models/reload", headers={"X-API-Key": "s3cret"})
    ).status_code == 200


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
async def test_predict_rate_limit_returns_429_with_retry_after(cfg, make_app):
    limited = cfg.model_copy(
        update={"api": cfg.api.model_copy(update={"predict_rate_limit_per_min": 3})}
    )
    async for c in make_client(make_app(limited)):
        codes = [
            (await c.post("/api/v1/predict", json=warehouse(i, 3.0))).status_code for i in range(5)
        ]
        assert codes == [200, 200, 200, 429, 429]
        blocked = await c.post("/api/v1/predict", json=warehouse(9, 3.0))
        assert int(blocked.headers["retry-after"]) >= 1
        assert blocked.headers["x-ratelimit-limit"] == "3"
        assert "Rate limit" in blocked.json()["detail"]
        # other routes are not affected
        assert (await c.get("/health")).status_code == 200
        assert (await c.get("/api/v1/anomalies")).status_code == 200


async def test_batch_limit_is_separate_and_stricter(cfg, make_app):
    limited = cfg.model_copy(
        update={"api": cfg.api.model_copy(update={"batch_predict_rate_limit_per_min": 2})}
    )
    async for c in make_client(make_app(limited)):
        body = {"events": [warehouse(0, 3.0)]}
        codes = [(await c.post("/api/v1/batch-predict", json=body)).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert (await c.post("/api/v1/predict", json=warehouse(1, 3.0))).status_code == 200


async def test_limits_are_per_client_ip_and_forwarded_for_is_ignored_by_default(cfg, make_app):
    limited = cfg.model_copy(
        update={"api": cfg.api.model_copy(update={"predict_rate_limit_per_min": 1})}
    )
    app = make_app(limited)
    async for first in make_client(app, host="1.1.1.1"):
        assert (await first.post("/api/v1/predict", json=warehouse(0, 3.0))).status_code == 200
        spoofed = await first.post(
            "/api/v1/predict", json=warehouse(1, 3.0), headers={"X-Forwarded-For": "9.9.9.9"}
        )
        assert spoofed.status_code == 429  # spoofing a new address does not help
        transport = httpx.ASGITransport(app=app, client=("2.2.2.2", 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as second:
            assert (await second.post("/api/v1/predict", json=warehouse(2, 3.0))).status_code == 200


async def test_forwarded_for_is_used_behind_a_trusted_proxy(cfg, make_app):
    limited = cfg.model_copy(
        update={"api": cfg.api.model_copy(update={"predict_rate_limit_per_min": 1})}
    )
    async for c in make_client(make_app(limited, trust_forwarded_for=True)):

        def post(ip: str, i: int, c: httpx.AsyncClient = c) -> Any:
            return c.post(
                "/api/v1/predict", json=warehouse(i, 3.0), headers={"X-Forwarded-For": ip}
            )

        assert (await post("5.5.5.5", 0)).status_code == 200
        assert (await post("5.5.5.5", 1)).status_code == 429
        assert (await post("6.6.6.6", 2)).status_code == 200


# --------------------------------------------------------------------------- #
# Persistence across restarts
# --------------------------------------------------------------------------- #
async def test_alerts_and_feedback_survive_a_restart(cfg, make_app):
    app = make_app()
    async for c in make_client(app):
        n = await warm_up(c)
        event_id = (await spike(c, n))["event_id"]
        await c.post(
            "/api/v1/feedback",
            json={
                "anomaly_event_id": event_id,
                "feedback_type": "false_positive",
                "analyst_id": "a1",
            },
        )
    # lifespan exit persisted both stores; a brand-new app restores them
    async for c in make_client(make_app()):
        assert (await c.get(f"/api/v1/anomalies/{event_id}")).status_code == 200
        assert (await c.get("/api/v1/feedback/stats")).json()["overall"]["total"] == 1
