"""Tests of the real-time streaming inference engine (Day 15)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from src.api.anomaly_store import AnomalyStore
from src.api.service import InferenceService
from src.kafka.pipeline import DeadLetterPublisher
from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.kafka.schemas import KafkaMessage
from src.kafka.streaming_inference import (
    EngineStats,
    KafkaAlertPublisher,
    LatencyTracker,
    StreamingInferenceEngine,
    default_is_critical,
)
from src.models.ensemble_detector import AnomalyEvent, Severity
from src.models.model_registry import ModelRegistry
from src.utils.config import PROJECT_ROOT, AppConfig
from src.utils.metrics import StreamingMetrics
from tests.mock_kafka import MockBroker

T0 = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)
TOPICS = {
    "warehouse-sensors": 6,
    "vehicle-gps": 4,
    "inventory-events": 4,
    "anomaly-alerts": 3,
    "dlq-events": 1,
    "model-health": 1,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def iso(offset_s: float) -> str:
    return (T0 + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")


def make_config(tmp_path: Path, **consumer: Any) -> AppConfig:
    cfg = AppConfig()
    paths = cfg.paths.model_copy(
        update={
            "model_registry": str(tmp_path / "registry"),
            "feedback_store": str(tmp_path / "feedback.json"),
            "anomaly_snapshots": str(tmp_path / "snapshots"),
        }
    )
    kafka = cfg.kafka.model_copy(
        update={
            "consumer": cfg.kafka.consumer.model_copy(
                update={"auto_offset_reset": "earliest", "num_workers": 2, **consumer}
            )
        }
    )
    return cfg.model_copy(update={"paths": paths, "kafka": kafka})


def warehouse(
    i: int,
    value: float,
    sensor: str = "wh1-A-temp-01",
    *,
    cold: bool = True,
    produced_at: datetime | None = None,
) -> KafkaMessage:
    meta = {"zone": "A", "storage_type": "cold_storage" if cold else "dry_goods"}
    message = KafkaMessage.from_payload(
        "warehouse",
        {
            "sensor_id": sensor,
            "warehouse_id": "wh1",
            "timestamp": iso(i),
            "reading_type": "temperature",
            "value": value,
            "unit": "C",
            "is_anomaly": False,
            "anomaly_type": None,
            "metadata": meta,
        },
    )
    return message.model_copy(update={"produced_at": produced_at}) if produced_at else message


def vehicle(i: int, speed: float, vehicle_id: str = "veh-001") -> KafkaMessage:
    return KafkaMessage.from_payload(
        "vehicle",
        {
            "vehicle_id": vehicle_id,
            "timestamp": iso(i * 10),
            "latitude": 22.57,
            "longitude": 88.36,
            "speed_kmh": speed,
            "heading": 90.0,
            "is_anomaly": False,
            "anomaly_type": None,
            "route_adherence_score": 0.99,
            "metadata": {},
        },
    )


def order_event() -> KafkaMessage:
    return KafkaMessage.from_payload(
        "order",
        {
            "order_id": "ORD-1",
            "sku_id": "SKU-1",
            "warehouse_id": "wh1",
            "timestamp": iso(0),
            "status": "picked",
            "previous_status": "created",
            "quantity": 3,
            "stage_duration_s": 41.0,
            "expected_duration_s": 40.0,
            "is_anomaly": False,
            "anomaly_type": None,
            "metadata": {},
        },
    )


class ListPublisher:
    """Collects published alerts (thread-safe)."""

    def __init__(self, *, accept: bool = True, raises: bool = False) -> None:
        self.events: list[AnomalyEvent] = []
        self._accept, self._raises = accept, raises
        self._lock = threading.Lock()

    def publish(self, event: AnomalyEvent) -> bool:
        if self._raises:
            raise RuntimeError("producer exploded")
        with self._lock:
            self.events.append(event)
        return self._accept


class Rig:
    """An engine on a private metrics registry, with its service and publisher."""

    def __init__(self, tmp_path: Path, **engine_kwargs: Any) -> None:
        self.cfg = make_config(tmp_path)
        self.service = InferenceService(self.cfg, ModelRegistry(tmp_path / "registry"))
        self.service.start()
        self.metrics = StreamingMetrics(CollectorRegistry())
        self.publisher = engine_kwargs.pop("publisher", ListPublisher())
        engine_kwargs.setdefault("snapshots", False)
        self.engine = StreamingInferenceEngine.from_config(
            self.service,
            publisher=self.publisher,
            metrics=self.metrics,
            latency=LatencyTracker(metrics=self.metrics),
            **engine_kwargs,
        )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def baseline_then_spike(
    rig: Rig, sensor: str = "wh1-A-temp-01", n: int = 120
) -> list[KafkaMessage]:
    """A calm cold-storage sensor followed by a large spike."""
    calm = [warehouse(i, 3.0 + 0.05 * ((i * 7) % 5), sensor) for i in range(n)]
    return [*calm, warehouse(n, 40.0, sensor)]


# --------------------------------------------------------------------------- #
# LatencyTracker
# --------------------------------------------------------------------------- #
def test_latency_percentiles_and_clamping():
    tracker = LatencyTracker(metrics=StreamingMetrics(CollectorRegistry()))
    assert tracker.snapshot().count == 0
    for ms in range(1, 101):
        tracker.observe(ms / 1000.0)
    tracker.observe(-3.0)  # clock skew: clamped to zero, still counted
    snap = tracker.snapshot()
    assert snap.count == 101
    assert snap.max == pytest.approx(0.1)
    assert snap.p50 == pytest.approx(0.05, abs=0.002)
    assert snap.p95 == pytest.approx(0.095, abs=0.002)
    assert snap.p99 == pytest.approx(0.099, abs=0.002)
    assert snap.to_dict()["p99_ms"] == pytest.approx(snap.p99 * 1000.0)


def test_latency_window_keeps_only_recent_samples():
    tracker = LatencyTracker(window=10, metrics=StreamingMetrics(CollectorRegistry()))
    for _ in range(50):
        tracker.observe(5.0)
    for _ in range(10):
        tracker.observe(0.001)
    assert tracker.total == 60
    assert tracker.snapshot().p99 == pytest.approx(0.001)


def test_slo_check_flags_a_breach_and_sets_the_gauge():
    metrics = StreamingMetrics(CollectorRegistry())
    tracker = LatencyTracker(0.5, min_samples=10, metrics=metrics)
    for _ in range(9):
        tracker.observe(2.0)
    assert tracker.check_slo() is True  # too few samples to judge
    tracker.observe(2.0)
    assert tracker.check_slo() is False
    assert tracker.breaches == 1
    assert metrics.registry.get_sample_value("scad_detection_latency_p99_seconds") == 2.0
    healthy = LatencyTracker(0.5, min_samples=1, metrics=StreamingMetrics(CollectorRegistry()))
    healthy.observe(0.01)
    assert healthy.check_slo() is True


def test_latency_tracker_validates_arguments():
    with pytest.raises(ValueError):
        LatencyTracker(0.0)
    with pytest.raises(ValueError):
        LatencyTracker(window=0)


# --------------------------------------------------------------------------- #
# Criticality
# --------------------------------------------------------------------------- #
def test_only_cold_storage_temperature_is_critical():
    assert default_is_critical(warehouse(0, 3.0, cold=True)) is True
    assert default_is_critical(warehouse(0, 20.0, cold=False)) is False
    assert default_is_critical(vehicle(0, 60.0)) is False


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def test_engine_scores_events_and_publishes_the_alert(tmp_path):
    rig = Rig(tmp_path)
    messages = baseline_then_spike(rig)

    async def go() -> None:
        await rig.engine.start()
        for message in messages:
            assert await rig.engine.submit(message) == "accepted"
        assert await rig.engine.wait_idle()
        await rig.engine.stop()

    run(go())
    stats = rig.engine.stats()
    assert stats.processed == len(messages)
    assert stats.dropped == 0 and stats.sampled_out == 0 and stats.queue_depth == 0
    assert stats.alerts_published >= 1
    assert len(rig.publisher.events) == stats.alerts_published
    alert = rig.publisher.events[-1]
    assert alert.entity_id == "wh1-A-temp-01" and alert.raw_value == 40.0
    assert alert.severity >= Severity.WARNING
    # the alert is in the store the API reads, with the same id
    assert rig.service.anomaly_store.get(alert.event_id) is alert
    assert rig.engine.latency.total == len(messages)
    assert rig.metrics.registry.get_sample_value(
        "scad_inference_events_total", {"outcome": "processed"}
    ) == len(messages)


def test_engine_matches_the_rest_api_verdicts(tmp_path):
    """The engine and POST /predict share one scoring path: same events, same alerts."""
    stream_rig = Rig(tmp_path / "stream")
    api_service = InferenceService(
        make_config(tmp_path / "api"), ModelRegistry(tmp_path / "api" / "registry")
    )
    api_service.start()
    messages = baseline_then_spike(stream_rig)

    async def go() -> None:
        await stream_rig.engine.start()
        for message in messages:
            await stream_rig.engine.submit(message)
        await stream_rig.engine.stop()

    run(go())
    api_alerts = [
        a
        for a in (api_service.evaluate(m, m.payload.model_dump()).alert for m in messages)
        if a is not None
    ]
    streamed = stream_rig.publisher.events
    assert [(a.entity_id, a.timestamp, a.severity) for a in streamed] == [
        (a.entity_id, a.timestamp, a.severity) for a in api_alerts
    ]


def test_events_of_one_entity_are_scored_in_order(tmp_path):
    rig = Rig(tmp_path, num_workers=4)
    seen: dict[str, list[str]] = defaultdict(list)
    lock = threading.Lock()
    original = rig.service.evaluate

    def recording(message: KafkaMessage, raw: Any) -> Any:
        with lock:
            seen[message.entity_id].append(message.timestamp)
        return original(message, raw)

    rig.service.evaluate = recording  # type: ignore[method-assign]
    sensors = [f"wh1-A-temp-{n:02d}" for n in range(8)]

    async def go() -> None:
        await rig.engine.start()
        for i in range(60):
            for sensor in sensors:
                await rig.engine.submit(warehouse(i, 3.0, sensor))
        await rig.engine.stop()

    run(go())
    assert set(seen) == set(sensors)
    for timestamps in seen.values():
        assert len(timestamps) == 60
        assert timestamps == sorted(timestamps)


def test_replays_and_order_events_are_not_scored(tmp_path):
    rig = Rig(tmp_path)
    message = warehouse(0, 3.0)

    async def go() -> list[str]:
        await rig.engine.start()
        results = [
            await rig.engine.submit(message),
            await rig.engine.submit(message),  # at-least-once redelivery
            await rig.engine.submit(order_event()),
        ]
        await rig.engine.stop()
        return results

    assert run(go()) == ["accepted", "duplicate", "unsupported"]
    stats = rig.engine.stats()
    assert (stats.processed, stats.duplicates, stats.unsupported) == (1, 1, 1)
    assert stats.received == 3 and stats.dropped == 0


def test_scoring_failure_is_counted_and_does_not_stop_the_stream(tmp_path):
    rig = Rig(tmp_path, num_workers=1)
    original = rig.service.evaluate

    def flaky(message: KafkaMessage, raw: Any) -> Any:
        if message.entity_id == "bad-sensor":
            raise RuntimeError("model blew up")
        return original(message, raw)

    rig.service.evaluate = flaky  # type: ignore[method-assign]

    async def go() -> None:
        await rig.engine.start()
        await rig.engine.submit(warehouse(0, 3.0, "bad-sensor"))
        for i in range(5):
            await rig.engine.submit(warehouse(i, 3.0, "good-sensor"))
        await rig.engine.stop()

    run(go())
    stats = rig.engine.stats()
    assert (stats.processed, stats.failed, stats.dropped) == (5, 1, 1)


@pytest.mark.parametrize("publisher", [ListPublisher(accept=False), ListPublisher(raises=True)])
def test_publish_failure_is_counted_and_scoring_continues(tmp_path, publisher):
    rig = Rig(tmp_path, publisher=publisher)
    messages = baseline_then_spike(rig)

    async def go() -> None:
        await rig.engine.start()
        for message in messages:
            await rig.engine.submit(message)
        await rig.engine.stop()

    run(go())
    stats = rig.engine.stats()
    assert stats.processed == len(messages) and stats.failed == 0
    assert stats.alerts_published == 0 and stats.alert_publish_failures >= 1
    assert rig.service.anomaly_store.stats()["total"] >= 1  # still stored for the API


def test_engine_works_without_a_publisher(tmp_path):
    rig = Rig(tmp_path)
    rig.engine._publisher = None
    messages = baseline_then_spike(rig)

    async def go() -> None:
        await rig.engine.start()
        for message in messages:
            await rig.engine.submit(message)
        await rig.engine.stop()

    run(go())
    assert rig.engine.stats().processed == len(messages)
    assert rig.service.anomaly_store.stats()["total"] >= 1


# --------------------------------------------------------------------------- #
# Backpressure
# --------------------------------------------------------------------------- #
def slow_down(rig: Rig, delay_s: float) -> None:
    original = rig.service.evaluate

    def slow(message: KafkaMessage, raw: Any) -> Any:
        time.sleep(delay_s)
        return original(message, raw)

    rig.service.evaluate = slow  # type: ignore[method-assign]


def test_backpressure_samples_non_critical_events_but_never_cold_storage(tmp_path):
    rig = Rig(tmp_path, num_workers=1, backpressure_depth=20, sampling_every_n=5, batch_size=4)
    slow_down(rig, 0.003)
    cold_total = dry_total = 200

    async def go() -> None:
        await rig.engine.start()
        for i in range(cold_total):
            await rig.engine.submit(warehouse(i, 3.0, "cold-1", cold=True))
            await rig.engine.submit(warehouse(i, 20.0, "dry-1", cold=False))
        await rig.engine.stop()

    run(go())
    stats = rig.engine.stats()
    assert stats.backpressure_activations >= 1 and stats.peak_queue_depth > 20
    assert 0 < stats.sampled_out < dry_total  # thinned, not wiped out
    # nothing the engine accepted was lost; cold storage was never sampled
    assert stats.dropped == 0
    assert stats.processed + stats.sampled_out == cold_total + dry_total
    assert stats.processed >= cold_total
    # and the mode ended once the backlog was gone (hysteresis)
    assert stats.backpressure_active is False and stats.queue_depth == 0


def test_sampling_keeps_every_nth_event_per_entity(tmp_path):
    rig = Rig(tmp_path, num_workers=1, backpressure_depth=1, sampling_every_n=4, batch_size=1)
    slow_down(rig, 0.01)

    async def go() -> list[str]:
        await rig.engine.start()
        # Submitting never yields to the loop, so the queue depth cannot fall in between and the
        # mode stays on for the whole burst.
        for i in range(5):  # critical events push the depth past the threshold
            await rig.engine.submit(warehouse(i, 3.0, "cold-1", cold=True))
        results = [
            await rig.engine.submit(warehouse(i, 20.0, "dry-1", cold=False)) for i in range(40)
        ]
        await rig.engine.stop()
        return results

    results = run(go())
    # per-entity counter: events 0, 4, 8, ... are kept, the rest are thinned out
    assert [i for i, r in enumerate(results) if r == "accepted"] == list(range(0, 40, 4))
    assert results.count("sampled_out") == 30
    assert rig.engine.stats().sampled_out == 30


def test_full_queues_make_submit_wait_instead_of_dropping(tmp_path):
    rig = Rig(tmp_path, num_workers=1, backpressure_depth=1, max_queue=3, batch_size=1)
    slow_down(rig, 0.02)

    async def go() -> None:
        await rig.engine.start()
        started = time.monotonic()
        # all critical (never sampled); the 3-slot queue forces submit() to wait for the worker
        for i in range(15):
            assert await rig.engine.submit(warehouse(i, 3.0, "cold-1")) == "accepted"
        waited = time.monotonic() - started
        await rig.engine.stop()
        assert waited > 0.1  # submit really blocked on a slow worker

    run(go())
    stats = rig.engine.stats()
    assert stats.processed == 15 and stats.dropped == 0


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_stop_drains_the_queue_and_then_refuses_events(tmp_path):
    rig = Rig(tmp_path, num_workers=1, batch_size=2)
    slow_down(rig, 0.002)

    async def go() -> tuple[bool, str]:
        await rig.engine.start()
        for i in range(60):
            await rig.engine.submit(warehouse(i, 3.0))
        drained = await rig.engine.stop(drain_timeout_s=10.0)
        late = await rig.engine.submit(warehouse(99, 3.0))
        return drained, late

    drained, late = run(go())
    stats = rig.engine.stats()
    assert drained is True and stats.processed == 60 and stats.queue_depth == 0
    assert late == "closed" and stats.rejected == 1 and stats.dropped == 1


def test_drain_timeout_is_reported(tmp_path):
    rig = Rig(tmp_path, num_workers=1, batch_size=1)
    slow_down(rig, 0.05)

    async def go() -> bool:
        await rig.engine.start()
        for i in range(40):
            await rig.engine.submit(warehouse(i, 3.0))
        return await rig.engine.stop(drain_timeout_s=0.1)

    assert run(go()) is False


def test_engine_requires_loaded_models_and_a_single_start(tmp_path):
    cfg = make_config(tmp_path)
    service = InferenceService(cfg, ModelRegistry(tmp_path / "registry"))  # start() not called
    engine = StreamingInferenceEngine.from_config(
        service, snapshots=False, metrics=StreamingMetrics(CollectorRegistry())
    )

    async def not_loaded() -> None:
        await engine.start()

    with pytest.raises(RuntimeError, match="not loaded"):
        run(not_loaded())

    rig = Rig(tmp_path / "ok")

    async def twice() -> None:
        await rig.engine.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await rig.engine.start()
        finally:
            await rig.engine.stop()

    run(twice())


def test_submit_before_start_is_rejected_and_validation(tmp_path):
    rig = Rig(tmp_path)
    assert run(rig.engine.submit(warehouse(0, 3.0))) == "closed"
    with pytest.raises(RuntimeError):
        rig.engine.submit_threadsafe(warehouse(0, 3.0))
    for bad in (
        {"num_workers": 0},
        {"batch_size": 0},
        {"sampling_every_n": 0},
        {"scoring_threads": 0},
    ):
        with pytest.raises(ValueError):
            StreamingInferenceEngine.from_config(rig.service, **bad)
    with pytest.raises(ValueError):
        StreamingInferenceEngine.from_config(rig.service, monitor_interval_s=0)


def test_engine_snapshots_alerts_while_running_and_on_stop(tmp_path):
    snapshot = tmp_path / "alerts.json"
    rig = Rig(tmp_path, snapshots=True, snapshot_path=snapshot, snapshot_interval_s=0.05)
    messages = baseline_then_spike(rig)

    async def go() -> None:
        await rig.engine.start()
        for message in messages:
            await rig.engine.submit(message)
        await rig.engine.wait_idle()
        await asyncio.sleep(0.2)  # let the periodic snapshot fire
        assert snapshot.is_file()
        await rig.engine.stop()

    run(go())
    payload = json.loads(snapshot.read_text())
    assert len(payload["events"]) == rig.service.anomaly_store.stats()["total"] >= 1


# --------------------------------------------------------------------------- #
# Kafka end to end (in-memory broker)
# --------------------------------------------------------------------------- #
def make_producer(broker: MockBroker, cfg: AppConfig) -> KafkaProducerManager:
    return KafkaProducerManager.from_config(
        cfg.kafka,
        producer_factory=broker.producer_factory,
        background_polling=False,
        metrics=ProducerMetrics(CollectorRegistry()),
    )


def test_kafka_to_alerts_topic_round_trip(tmp_path):
    broker = MockBroker(TOPICS)
    rig = Rig(tmp_path)
    rig.engine._publisher = KafkaAlertPublisher(make_producer(broker, rig.cfg))
    data_producer = make_producer(broker, rig.cfg)
    messages = baseline_then_spike(rig)
    for message in messages:
        data_producer.send_event(
            {"source_type": "warehouse", "payload": message.payload.model_dump()}
        )
    data_producer.send_event(
        {"source_type": "vehicle", "payload": vehicle(0, 60.0).payload.model_dump()}
    )
    data_producer.flush()
    dlq = DeadLetterPublisher(broker.producer_factory({}), "dlq-events")

    async def go() -> None:
        await rig.engine.start()
        # a corrupt record must be dead-lettered, not stall the stream
        broker.append("warehouse-sensors", b"x", b"{not json")
        rig.engine.start_consuming(consumer_factory=broker.consumer_factory, dead_letter=dlq)
        deadline = time.monotonic() + 20
        while rig.engine.stats().processed < len(messages) + 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await rig.engine.stop()

    run(go())
    stats = rig.engine.stats()
    assert stats.processed == len(messages) + 1 and stats.dropped == 0
    published = broker.messages("anomaly-alerts")
    assert len(published) == stats.alerts_published >= 1
    alert = AnomalyEvent.from_dict(json.loads(published[-1].value))
    assert alert.entity_id == "wh1-A-temp-01" and alert.raw_value == 40.0
    assert published[-1].key == b"wh1-A-temp-01"
    dead = [json.loads(r.value) for r in broker.messages("dlq-events")]
    assert len(dead) == 1 and dead[0]["error"]["reason"] == "validation_error"
    assert rig.engine.consumer_group is None  # released by stop()


def test_consuming_requires_a_running_engine_and_only_once(tmp_path):
    rig = Rig(tmp_path)
    broker = MockBroker(TOPICS)
    with pytest.raises(RuntimeError, match="not running"):
        rig.engine.start_consuming(consumer_factory=broker.consumer_factory)

    async def go() -> None:
        await rig.engine.start()
        rig.engine.start_consuming(consumer_factory=broker.consumer_factory)
        try:
            with pytest.raises(RuntimeError, match="already consuming"):
                rig.engine.start_consuming(consumer_factory=broker.consumer_factory)
        finally:
            await rig.engine.stop()

    run(go())


# --------------------------------------------------------------------------- #
# Producer.send_alert and AnomalyStore snapshots
# --------------------------------------------------------------------------- #
def alert_dict(**overrides: Any) -> dict[str, Any]:
    event = AnomalyEvent(
        event_id="e-1",
        entity_id="wh1-A-temp-01",
        entity_type="warehouse_sensor",
        timestamp=T0,
        anomaly_type="point",
        severity=Severity.CRITICAL,
        raw_value=12.0,
        expected_range=(2.0, 4.0),
        top_contributing_features=[],
        recommended_action="check the compressor",
    )
    return {**event.to_dict(), **overrides}


def test_send_alert_publishes_json_keyed_by_entity():
    broker = MockBroker(TOPICS)
    producer = make_producer(broker, AppConfig())
    assert producer.send_alert(alert_dict()) is True
    producer.flush()
    record = broker.messages("anomaly-alerts")[0]
    assert record.key == b"wh1-A-temp-01"
    assert AnomalyEvent.from_dict(json.loads(record.value)).event_id == "e-1"


def test_send_alert_dead_letters_unusable_alerts():
    broker = MockBroker(TOPICS)
    producer = make_producer(broker, AppConfig())
    no_entity = alert_dict()
    del no_entity["entity_id"]
    assert producer.send_alert(no_entity) is False
    assert producer.send_alert(alert_dict(blob=object())) is False  # not JSON serialisable
    assert producer.send_alert(alert_dict(blob="x" * 2_000_000)) is False  # over the topic limit
    producer.flush()
    assert broker.messages("anomaly-alerts") == []
    reasons = sorted(json.loads(r.value)["error"]["reason"] for r in broker.messages("dlq-events"))
    assert reasons == ["message_too_large", "validation_error", "validation_error"]


def sample_event(n: int) -> AnomalyEvent:
    return AnomalyEvent.from_dict(alert_dict(event_id=f"e-{n}", entity_id=f"s-{n % 3}"))


def test_periodic_snapshots_write_and_stop_with_a_final_write(tmp_path):
    store = AnomalyStore(capacity=10)
    path = tmp_path / "nested" / "anomalies.json"
    store.start_periodic_snapshots(path, interval_s=0.05)
    store.add(sample_event(1))
    deadline = time.monotonic() + 3
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.is_file()
    store.add(sample_event(2))  # arrives after the last tick; the final snapshot must catch it
    store.stop_periodic_snapshots(final=True)
    restored = AnomalyStore(capacity=10)
    assert restored.load_snapshot(path) == 2
    store.stop_periodic_snapshots()  # idempotent


def test_periodic_snapshot_survives_a_failing_write_and_validates(tmp_path):
    store = AnomalyStore()
    with pytest.raises(ValueError):
        store.start_periodic_snapshots(tmp_path / "a.json", 0)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the snapshot directory should be")
    store.start_periodic_snapshots(blocker / "anomalies.json", interval_s=0.02)
    time.sleep(0.15)  # several failing ticks must not kill the thread
    thread = store._snapshot_thread
    assert thread is not None and thread.is_alive()
    store.stop_periodic_snapshots()
    assert thread is not None and not thread.is_alive()


# --------------------------------------------------------------------------- #
# Runner script
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def script():
    path = PROJECT_ROOT / "scripts" / "run_streaming_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_streaming_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_parser_defaults(script):
    args = script.build_parser().parse_args([])
    assert args.workers is None and args.no_api is False and args.drain_timeout == 30.0
    args = script.build_parser().parse_args(["--workers", "8", "--port", "8010", "--no-api"])
    assert (args.workers, args.port, args.no_api) == (8, 8010, True)


def test_runner_rejects_bad_arguments(script):
    assert script.main(["--workers", "0"]) == script.EXIT_USAGE
    assert script.main(["--nonsense"]) == script.EXIT_USAGE


def test_runner_stats_route_reports_the_engine(tmp_path, script):
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    rig = Rig(tmp_path)
    app = create_app(rig.cfg, service=rig.service, metrics_registry=CollectorRegistry())
    script._add_stats_route(app, rig.engine)
    with TestClient(app) as client:
        body = client.get("/api/v1/pipeline/stats").json()
    assert body["engine"]["processed"] == 0 and body["engine"]["dropped"] == 0
    assert body["latency_target_ms"] == 500.0 and body["consumer_lag"] is None
    assert isinstance(rig.engine.stats(), EngineStats)
