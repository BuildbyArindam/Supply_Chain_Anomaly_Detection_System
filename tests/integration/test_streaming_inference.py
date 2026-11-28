"""Full streaming path: Kafka -> consumer group -> inference engine -> alerts topic (Day 23).

Labelled events (a normal warm-up, then injected spikes) are produced to the data topic of a real
Kafka broker (testcontainers, see ``conftest.py``); the :class:`StreamingInferenceEngine` consumes
them, scores them with the production scoring service and publishes alerts back to Kafka. The test
reads the ``anomaly-alerts`` topic and checks the alerts against the injected ground truth.

The service runs with an empty model registry, i.e. the statistical detector alone, the same
fallback the system serves when no trained models are available.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from src.api.service import InferenceService
from src.kafka.streaming_inference import (
    KafkaAlertPublisher,
    LatencyTracker,
    StreamingInferenceEngine,
)
from src.models.model_registry import ModelRegistry
from src.utils.metrics import StreamingMetrics

pytestmark = pytest.mark.integration

START = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)
WARMUP = 150
SPIKES = 6


def reading(sensor: str, i: int, value: float, *, anomaly: bool = False) -> dict[str, Any]:
    return {
        "source_type": "warehouse",
        "payload": {
            "sensor_id": sensor,
            "warehouse_id": "wh1",
            "timestamp": (START + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
            "reading_type": "temperature",
            "value": value,
            "unit": "°C",
            "is_anomaly": anomaly,
            "anomaly_type": "point" if anomaly else None,
            "metadata": {"zone": "A", "storage_type": "cold_storage"},
        },
    }


async def wait_until(predicate, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.mark.asyncio
async def test_injected_anomalies_are_published_to_the_alerts_topic(kafka, app_config_in, tmp_path):
    cfg = app_config_in(kafka.cfg)
    faulty, healthy = "wh1-A-temp-faulty", "wh1-A-temp-healthy"

    # --- labelled ground truth: two sensors, only the first one gets spikes -------------------
    events: list[dict[str, Any]] = []
    for i in range(WARMUP):
        wiggle = 0.05 * ((i * 7) % 5 - 2)  # deterministic, tiny noise around 3 degC
        events.append(reading(healthy, i, 3.0 + wiggle))
        events.append(reading(faulty, i, 3.0 + wiggle))
    spike_times = set()
    for k in range(SPIKES):
        i = WARMUP + k
        events.append(reading(faulty, i, 25.0 + k, anomaly=True))
        events.append(reading(healthy, i, 3.0))
        spike_times.add(START + timedelta(seconds=i))

    with kafka.producer() as data_producer:
        assert data_producer.send_batch(events) == len(events)
        assert data_producer.flush(60) == 0

    # --- the system under test ----------------------------------------------------------------
    service = InferenceService(cfg, ModelRegistry(tmp_path / "registry"))
    service.start()
    alert_producer = kafka.producer()
    metrics = StreamingMetrics(CollectorRegistry())
    engine = StreamingInferenceEngine.from_config(
        service,
        publisher=KafkaAlertPublisher(alert_producer),
        metrics=metrics,
        latency=LatencyTracker(0.5, window=len(events), metrics=metrics),
        snapshots=False,
    )
    await engine.start()
    try:
        engine.start_consuming(
            consumer_factory=kafka.consumer_factory, poll_timeout_s=0.2, lag_interval_s=0.5
        )
        assert await wait_until(
            lambda: engine.stats().processed >= len(events)
        ), f"engine processed {engine.stats().processed}/{len(events)} events"
        assert await engine.wait_idle(timeout_s=30)
        assert alert_producer.flush(30) == 0
    finally:
        assert await engine.stop(drain_timeout_s=30)
        alert_producer.close()

    # --- engine accounting: nothing lost, nothing failed --------------------------------------
    stats = engine.stats()
    assert stats.processed == len(events)
    assert stats.dropped == 0 and stats.failed == 0 and stats.rejected == 0

    # --- alerts on the Kafka topic vs the injected ground truth -------------------------------
    alerts = kafka.read("anomaly_alerts", at_least=1, timeout=60)
    assert alerts, "no alert reached the anomaly-alerts topic"
    assert stats.alerts_published == len(alerts)

    assert {a["entity_id"] for a in alerts} == {faulty}  # the healthy sensor never alerts
    flagged_times = {datetime.fromisoformat(a["timestamp"]) for a in alerts}
    assert flagged_times & spike_times, "no alert matches an injected spike"
    warmup_times = {START + timedelta(seconds=i) for i in range(WARMUP)}
    assert not flagged_times & warmup_times, "an alert fired during the normal warm-up"
    # the deduplicator folds the 6 consecutive spikes of one entity into (far) fewer alerts
    assert len(alerts) < SPIKES

    first = alerts[0]
    assert {"event_id", "severity", "anomaly_type", "recommended_action", "raw_value"} <= set(first)
    assert first["severity"] in {"WARNING", "CRITICAL", "EMERGENCY"}
    assert first["false_positive_marked"] is False
    assert len({a["event_id"] for a in alerts}) == len(alerts)  # unique ids

    # the same alerts are queryable through the API's store
    stored = service.anomaly_store.filter(entity_id=faulty)
    assert {e.event_id for e in stored} >= {a["event_id"] for a in alerts}

    latency = engine.latency.snapshot()
    assert latency.count == len(events)
