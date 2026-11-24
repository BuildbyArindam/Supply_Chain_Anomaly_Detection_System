"""Load test of the streaming inference engine (Day 15).

Replays 30 000 simulator events (warehouse sensors, GPS, inventory, with injected anomalies) into
the engine at 1000 events/second for 30 seconds and checks the Day 15 SLOs:

* p99 end-to-end latency (producer timestamp -> verdict / alert handed to the producer) < 500 ms
* zero events dropped: every injected event is scored (no failures, no refusals)
* sampling mode never kicks in at this rate (it would mean silently skipped events)

What this does and does not measure
-----------------------------------
It drives the engine directly (``submit``), so it measures queueing and scoring, not Kafka
transport. The service runs with an **empty model registry**, i.e. the statistical detector alone,
which is what the system serves when no trained model is available. Trained LSTM / Isolation Forest
members are much slower per event; their throughput has to be measured separately (Day 25).
The test is also sensitive to the machine: it asserts that the injector actually achieved the
target rate, so a starved CI runner fails loudly instead of passing on an easier workload.

Run it alone with ``pytest tests/test_streaming_performance.py -s`` (about 35 s); skip it with
``pytest -m "not performance"``.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from src.api.service import InferenceService
from src.data_simulation.orchestrator import SimulationOrchestrator
from src.kafka.pipeline import DeadLetterPublisher
from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.kafka.schemas import KafkaMessage
from src.kafka.streaming_inference import (
    KafkaAlertPublisher,
    LatencyTracker,
    StreamingInferenceEngine,
)
from src.models.ensemble_detector import AnomalyEvent
from src.models.model_registry import ModelRegistry
from src.utils.config import AppConfig
from src.utils.metrics import StreamingMetrics
from tests.mock_kafka import MockBroker

pytestmark = pytest.mark.performance

RATE_PER_S = 1000
DURATION_S = 30
TOTAL_EVENTS = RATE_PER_S * DURATION_S
P99_TARGET_S = 0.5
MIN_ACHIEVED_RATE = 0.95 * RATE_PER_S


class CollectingPublisher:
    """Counts published alerts (stands in for the Kafka producer)."""

    def __init__(self) -> None:
        self.alerts: list[AnomalyEvent] = []
        self._lock = threading.Lock()

    def publish(self, event: AnomalyEvent) -> bool:
        with self._lock:
            self.alerts.append(event)
        return True


@pytest.fixture(scope="module")
def stream() -> Iterator[list[KafkaMessage]]:
    """At least ``TOTAL_EVENTS`` realistic events in per-entity time order.

    The pre-built messages are test scaffolding that a real consumer never keeps, yet they are
    hundreds of thousands of live objects the cyclic garbage collector would rescan on every full
    collection. Left in, they add ~100 ms pauses that say nothing about the engine (measured:
    p99 ~90 ms with them scanned vs ~5 ms with the collector off), so they are frozen out of GC
    scans for the duration of the test. The engine's own state is still collected normally.
    """
    cfg = AppConfig()
    orchestrator = SimulationOrchestrator.from_config(cfg.simulation)
    events = orchestrator.run_batch(duration_s=2_400)
    messages = [
        KafkaMessage.from_payload(e["source_type"], e["payload"])
        for e in events
        if e["source_type"] != "order"
    ]
    assert len(messages) >= TOTAL_EVENTS, f"simulator produced only {len(messages)} events"
    del events
    gc.collect()
    gc.freeze()
    try:
        yield messages[:TOTAL_EVENTS]
    finally:
        gc.unfreeze()


def test_sustains_1000_events_per_second_within_the_latency_slo(
    tmp_path: Path, stream: list[KafkaMessage], record_property: pytest.RecordProperty
) -> None:
    cfg = AppConfig()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={
                    "model_registry": str(tmp_path / "registry"),
                    "feedback_store": str(tmp_path / "feedback.json"),
                    "anomaly_snapshots": str(tmp_path / "snapshots"),
                }
            )
        }
    )
    service = InferenceService(cfg, ModelRegistry(tmp_path / "registry"))
    service.start()
    metrics = StreamingMetrics(CollectorRegistry())
    publisher = CollectingPublisher()
    engine = StreamingInferenceEngine.from_config(
        service,
        publisher=publisher,
        metrics=metrics,
        latency=LatencyTracker(P99_TARGET_S, window=TOTAL_EVENTS, metrics=metrics),
        snapshots=False,
    )

    async def drive() -> float:
        await engine.start()
        started = time.perf_counter()
        sent = 0
        while sent < TOTAL_EVENTS:
            due = min(TOTAL_EVENTS, int((time.perf_counter() - started) * RATE_PER_S) + 1)
            while sent < due:
                # Stamp the event at injection time: that is when it "left the producer".
                message = stream[sent].model_copy(update={"produced_at": datetime.now(UTC)})
                assert await engine.submit(message) == "accepted"
                sent += 1
            await asyncio.sleep(0.002)
        injected_for = time.perf_counter() - started
        assert await engine.wait_idle(timeout_s=30), "queues did not drain after the load stopped"
        assert await engine.stop(drain_timeout_s=30)
        return injected_for

    injected_for = asyncio.run(drive())
    achieved = TOTAL_EVENTS / injected_for
    stats = engine.stats()
    latency = engine.latency.snapshot()
    record_property("achieved_events_per_s", round(achieved, 1))
    record_property("p50_ms", round(latency.p50 * 1000, 2))
    record_property("p95_ms", round(latency.p95 * 1000, 2))
    record_property("p99_ms", round(latency.p99 * 1000, 2))
    record_property("max_ms", round(latency.max * 1000, 2))
    record_property("peak_queue_depth", stats.peak_queue_depth)
    print(
        f"\nstreaming load test: {TOTAL_EVENTS} events in {injected_for:.1f}s "
        f"({achieved:.0f} ev/s), p50={latency.p50 * 1000:.1f}ms p95={latency.p95 * 1000:.1f}ms "
        f"p99={latency.p99 * 1000:.1f}ms max={latency.max * 1000:.1f}ms, "
        f"peak queue depth={stats.peak_queue_depth}, alerts={stats.alerts_published}"
    )

    # The injector really delivered the advertised load (otherwise the result proves nothing).
    assert achieved >= MIN_ACHIEVED_RATE, f"injector only managed {achieved:.0f} ev/s"
    # Zero events dropped: all of them were scored, none failed or were refused or skipped.
    assert stats.dropped == 0 and stats.failed == 0 and stats.rejected == 0
    assert stats.processed == TOTAL_EVENTS
    assert stats.sampled_out == 0 and stats.backpressure_activations == 0
    assert stats.queue_depth == 0
    # The SLO.
    assert latency.count == TOTAL_EVENTS
    assert latency.p99 < P99_TARGET_S, f"p99 {latency.p99 * 1000:.0f} ms >= 500 ms"
    assert engine.latency.check_slo() is True
    # The alert path ran: anomalies were found, published and stored for the API.
    assert stats.alerts_published > 0 and len(publisher.alerts) == stats.alerts_published
    assert service.anomaly_store.stats()["total"] > 0


def test_kafka_consumer_path_keeps_up_at_350_events_per_second(
    tmp_path: Path, stream: list[KafkaMessage]
) -> None:
    """Events travel producer -> in-memory broker -> consumer group -> engine -> alerts topic.

    Guards the SLO of the Kafka path itself (the other test calls ``submit`` directly). Note: with
    the mock producers' pollers left spinning on a 2-core machine this path stalls at about
    250 events/s; that spin is a mock artifact (a real poll blocks in C), so the test pumps the
    producers on a timer instead.
    """
    rate, seconds = 350, 8
    total = rate * seconds
    cfg = AppConfig()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={
                    "model_registry": str(tmp_path / "registry"),
                    "feedback_store": str(tmp_path / "feedback.json"),
                    "anomaly_snapshots": str(tmp_path / "snapshots"),
                }
            ),
            "kafka": cfg.kafka.model_copy(
                update={
                    "consumer": cfg.kafka.consumer.model_copy(
                        update={"auto_offset_reset": "earliest"}
                    )
                }
            ),
        }
    )
    service = InferenceService(cfg, ModelRegistry(tmp_path / "registry"))
    service.start()
    broker = MockBroker(
        {
            "warehouse-sensors": 6,
            "vehicle-gps": 4,
            "inventory-events": 4,
            "anomaly-alerts": 3,
            "dlq-events": 1,
            "model-health": 1,
        }
    )

    def producer() -> KafkaProducerManager:
        # background_polling=False: the mock's poll() returns at once, so a polling thread would
        # spin a core; a real librdkafka poll blocks in C.
        return KafkaProducerManager.from_config(
            cfg.kafka,
            producer_factory=broker.producer_factory,
            background_polling=False,
            metrics=ProducerMetrics(CollectorRegistry()),
        )

    data_p, alert_p = producer(), producer()
    stop_pump = threading.Event()

    def pump() -> None:
        while not stop_pump.is_set():
            data_p.pump()
            alert_p.pump()
            time.sleep(0.002)

    threading.Thread(target=pump, daemon=True).start()
    metrics = StreamingMetrics(CollectorRegistry())
    engine = StreamingInferenceEngine.from_config(
        service,
        publisher=KafkaAlertPublisher(alert_p),
        metrics=metrics,
        latency=LatencyTracker(P99_TARGET_S, window=total, metrics=metrics),
        snapshots=False,
    )

    def produce() -> None:
        started = time.perf_counter()
        for i in range(total):
            while (wait := started + i / rate - time.perf_counter()) > 0:
                time.sleep(min(wait, 0.002))
            message = stream[i]
            data_p.send_event(
                {"source_type": message.source_type, "payload": message.payload.model_dump()}
            )

    async def drive() -> None:
        await engine.start()
        engine.start_consuming(
            consumer_factory=broker.consumer_factory,
            dead_letter=DeadLetterPublisher(broker.producer_factory({}), "dlq-events"),
        )
        sender = threading.Thread(target=produce)
        sender.start()
        while sender.is_alive():
            await asyncio.sleep(0.05)
        deadline = time.monotonic() + 30
        while engine.stats().processed < total and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await engine.stop()

    try:
        asyncio.run(drive())
    finally:
        stop_pump.set()
    stats = engine.stats()
    latency = engine.latency.snapshot()
    assert stats.processed == total and stats.dropped == 0 and stats.sampled_out == 0
    assert latency.p99 < P99_TARGET_S, f"p99 {latency.p99 * 1000:.0f} ms"
