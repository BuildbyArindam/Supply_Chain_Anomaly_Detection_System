"""The inference API as one running service: every endpoint, driven over HTTP with httpx (Day 23).

The detailed per-endpoint behaviour is covered in ``tests/unit/test_api.py``. These tests exercise
the *workflow* an operator and the dashboard go through against one live app instance (lifespan
started, real scoring service, real feedback store on disk):

    health -> warm-up predictions -> anomaly -> list / filter / fetch -> feedback round-trip
    -> stats -> models -> hot reload -> metrics -> restart with persisted state

The Kafka-connected health check runs against the real broker when Docker is available.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from prometheus_client import CollectorRegistry

from src.api.main import create_app
from src.utils.config import AppConfig

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

T0 = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)
SENSOR = "wh1-A-temp-01"


def warehouse(i: int, value: float, sensor: str = SENSOR) -> dict[str, Any]:
    return {
        "source_type": "warehouse",
        "payload": {
            "sensor_id": sensor,
            "warehouse_id": "wh1",
            "timestamp": (T0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
            "reading_type": "temperature",
            "value": value,
            "unit": "C",
            "metadata": {"zone": "A", "storage_type": "cold_storage"},
        },
    }


def api_config(base: AppConfig, **api: Any) -> AppConfig:
    api_cfg = base.api.model_copy(
        update={
            "predict_rate_limit_per_min": 100_000,
            "batch_predict_rate_limit_per_min": 100_000,
            **api,
        }
    )
    return base.model_copy(update={"api": api_cfg})


async def serve(app: Any, host: str = "10.0.0.1") -> AsyncIterator[httpx.AsyncClient]:
    """Run the app's lifespan and yield a client bound to it."""
    transport = httpx.ASGITransport(app=app, client=(host, 5000))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


async def warm_up(client: httpx.AsyncClient, n: int = 80) -> int:
    rng = random.Random(7)
    for i in range(n):
        r = await client.post("/api/v1/predict", json=warehouse(i, 3.0 + rng.gauss(0, 0.15)))
        assert r.status_code == 200, r.text
    return n


async def test_operator_workflow_through_every_endpoint(app_config_in):
    cfg = api_config(app_config_in(AppConfig()))
    app = create_app(cfg, metrics_registry=CollectorRegistry())

    async for client in serve(app):
        # --- service is up (statistical fallback: no trained models in an empty registry) ----
        health = (await client.get("/health")).json()
        assert health["models_loaded"] is True and health["stored_anomalies"] == 0
        spec = (await client.get("/openapi.json")).json()
        for route in (
            "/api/v1/predict",
            "/api/v1/batch-predict",
            "/api/v1/anomalies",
            "/api/v1/anomalies/{event_id}",
            "/api/v1/feedback",
            "/api/v1/feedback/stats",
            "/api/v1/models",
            "/api/v1/models/reload",
            "/api/v1/models/current",
            "/health",
        ):
            assert route in spec["paths"], f"{route} missing from the OpenAPI spec"

        # --- score a normal stream, then a batch that ends in a spike ----------------------
        n = await warm_up(client)
        normal = (await client.post("/api/v1/predict", json=warehouse(n, 3.0))).json()
        assert normal["is_anomaly"] is False and normal["event_id"] is None

        batch = {"events": [warehouse(n + 1 + i, 3.0) for i in range(3)]}
        batch["events"].append(warehouse(n + 4, 25.0))
        body = (await client.post("/api/v1/batch-predict", json=batch)).json()
        assert body["count"] == 4 and body["anomalies"] == 1  # order kept, one anomaly counted
        assert [r["is_anomaly"] for r in body["results"]] == [False, False, False, True]
        spike = body["results"][-1]
        assert spike["event_id"] and spike["models_used"] == ["statistical"]
        event_id = spike["event_id"]

        # a second, well separated episode scored through /predict
        later = (await client.post("/api/v1/predict", json=warehouse(n + 600, 26.0))).json()
        assert later["is_anomaly"] is True and later["event_id"] != event_id

        # --- list, filter, fetch ------------------------------------------------------------
        listing = (await client.get("/api/v1/anomalies", params={"limit": 1000})).json()
        # the two injected spikes are there (a noisy warm-up reading may also have been flagged)
        spikes = [e for e in listing["items"] if e["raw_value"] > 20]
        assert {e["event_id"] for e in spikes} == {event_id, later["event_id"]}
        assert listing["total"] >= 2
        by_entity = (await client.get("/api/v1/anomalies", params={"entity_id": SENSOR})).json()
        assert by_entity["total"] == listing["total"]
        nobody = (await client.get("/api/v1/anomalies", params={"entity_id": "ghost"})).json()
        assert nobody["total"] == 0 and nobody["items"] == []
        window = (
            await client.get(
                "/api/v1/anomalies",
                params={"start_time": (T0 + timedelta(seconds=n + 300)).isoformat()},
            )
        ).json()
        assert window["total"] == 1  # only the later episode

        event = (await client.get(f"/api/v1/anomalies/{event_id}")).json()
        assert event["entity_id"] == SENSOR and event["raw_value"] == 25.0
        assert event["is_resolved"] is False and event["false_positive_marked"] is False
        assert event["severity"] in {"INFO", "WARNING", "CRITICAL", "EMERGENCY"}
        assert (await client.get("/api/v1/anomalies/does-not-exist")).status_code == 404

        # --- feedback round-trip ------------------------------------------------------------
        assert (await client.get("/api/v1/feedback/stats")).json()["overall"]["total"] == 0
        posted = await client.post(
            "/api/v1/feedback",
            json={
                "anomaly_event_id": event_id,
                "feedback_type": "false_positive",
                "analyst_id": "analyst-1",
                "notes": "door was open during stock take",
            },
        )
        assert posted.status_code == 201
        flagged = (await client.get(f"/api/v1/anomalies/{event_id}")).json()
        assert flagged["false_positive_marked"] is True  # the store reflects the verdict
        stats = (await client.get("/api/v1/feedback/stats")).json()
        assert stats["overall"]["false_positive"] == 1
        assert stats["overall"]["false_positive_rate"] == 1.0
        assert stats["per_entity"][SENSOR]["total"] == 1
        unknown = await client.post(
            "/api/v1/feedback", json={"anomaly_event_id": "ghost", "feedback_type": "true_positive"}
        )
        assert unknown.status_code == 404

        # --- models -------------------------------------------------------------------------
        assert (await client.get("/api/v1/models")).json()["models"] == []
        current = (await client.get("/api/v1/models/current")).json()
        assert current["degraded"] is True and current["generation"] == 1
        reload = (await client.post("/api/v1/models/reload")).json()
        assert reload["previous_generation"] == 1 and reload["current"]["generation"] == 2
        still = await client.post("/api/v1/predict", json=warehouse(n + 700, 3.0))
        assert still.status_code == 200  # serving continued across the hot reload

        # --- Prometheus metrics -------------------------------------------------------------
        metrics = await client.get("/metrics", follow_redirects=True)
        assert metrics.status_code == 200 and "text/plain" in metrics.headers["content-type"]
        assert "# HELP" in metrics.text  # Prometheus exposition format with the API's series

    # --- restart: alerts and feedback were persisted and come back ---------------------------
    async for client in serve(create_app(cfg, metrics_registry=CollectorRegistry())):
        restored = (await client.get(f"/api/v1/anomalies/{event_id}")).json()
        assert restored["entity_id"] == SENSOR
        assert (await client.get("/api/v1/feedback/stats")).json()["overall"]["total"] == 1


async def test_feedback_round_trip_moves_the_threshold_used_by_the_next_predict(
    app_config_in, tmp_path
):
    """false-positive verdicts -> ThresholdCalibrator raises the entity's threshold -> a marginal
    reading that was an anomaly before is judged normal on the next ``/predict``."""
    cfg = api_config(app_config_in(AppConfig()))
    app = create_app(cfg, metrics_registry=CollectorRegistry())
    marginal = warehouse(680, 3.4)  # statistical score ~0.56: above 0.5, below the raised 0.6

    async for client in serve(app):
        calibrator = app.state.service.calibrator
        n = await warm_up(client)
        spike = (await client.post("/api/v1/predict", json=warehouse(n, 25.0))).json()
        event_id = spike["event_id"]
        assert calibrator.get_adjusted_threshold(SENSOR, 0.5) == 0.5  # untouched so far

        # Verdicts: the first five stay inside the burst allowance, the sixth raises the threshold.
        adjustments = []
        for analyst in range(6):
            r = await client.post(
                "/api/v1/feedback",
                json={
                    "anomaly_event_id": event_id,
                    "feedback_type": "false_positive",
                    "analyst_id": f"analyst-{analyst}",
                },
            )
            assert r.status_code == 201, r.text
            adjustments.append(r.json()["threshold_adjustment"])
        assert adjustments[:5] == [0.0] * 5 and adjustments[5] == pytest.approx(0.1)

        # the calibrator inside the service was updated ...
        assert calibrator.get_adjusted_threshold(SENSOR, 0.5) == pytest.approx(0.6)
        analytics = (await client.get("/api/v1/feedback/analytics")).json()
        assert analytics["adjusted_entities"][SENSOR]["current_threshold_adjustment"] == 0.1

        # ... and the next prediction is decided against the new threshold
        after = (await client.post("/api/v1/predict", json=marginal)).json()
        assert 0.5 <= after["anomaly_score"] < 0.6  # the score itself is unchanged
        assert after["is_anomaly"] is False

    # Control: with fresh storage (no verdicts) the very same reading is an anomaly.
    fresh = api_config(app_config_in(AppConfig()))
    fresh = fresh.model_copy(
        update={
            "paths": fresh.paths.model_copy(
                update={
                    "model_registry": str(tmp_path / "control" / "registry"),
                    "feedback_store": str(tmp_path / "control" / "feedback.json"),
                    "anomaly_snapshots": str(tmp_path / "control" / "snapshots"),
                }
            )
        }
    )
    async for client in serve(create_app(fresh, metrics_registry=CollectorRegistry())):
        n = await warm_up(client)
        await client.post("/api/v1/predict", json=warehouse(n, 25.0))
        before = (await client.post("/api/v1/predict", json=marginal)).json()
        assert before["is_anomaly"] is True and 0.5 <= before["anomaly_score"] < 0.6


async def test_rate_limit_returns_429_with_retry_after(app_config_in):
    cfg = api_config(app_config_in(AppConfig()), predict_rate_limit_per_min=5)
    app = create_app(cfg, metrics_registry=CollectorRegistry())
    async for client in serve(app):
        codes = []
        for i in range(8):
            codes.append((await client.post("/api/v1/predict", json=warehouse(i, 3.0))).status_code)
        assert codes[:5] == [200] * 5 and set(codes[5:]) == {429}
        limited = await client.post("/api/v1/predict", json=warehouse(9, 3.0))
        assert int(limited.headers["retry-after"]) >= 1
        # the other client IP has its own bucket, and unlimited routes still answer
        assert (await client.get("/health")).status_code == 200


async def test_health_reports_the_real_kafka_connection(kafka, app_config_in):
    cfg = api_config(app_config_in(kafka.cfg))
    with kafka.producer() as producer:
        app = create_app(
            cfg,
            metrics_registry=CollectorRegistry(),
            kafka_check=lambda: producer.check_health(timeout=10).healthy,
        )
        async for client in serve(app):
            health = (await client.get("/health")).json()
            assert health["checks"]["kafka"] == "ok"
