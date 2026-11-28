"""Kafka producer -> broker -> consumer group, against a real Kafka container (Day 23).

Runs on a ``confluentinc/cp-kafka`` KRaft broker started by testcontainers (see ``conftest.py``);
skipped when Docker is not available. Each test works on its own freshly created topics.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from src.kafka.consumer import ConsumedMessage, KafkaConsumerGroup, PermanentProcessingError
from src.kafka.pipeline import DeadLetterPublisher, build_pipeline
from src.utils.metrics import ConsumerMetrics

pytestmark = pytest.mark.integration

START = datetime(2026, 11, 20, 8, 0, 0, tzinfo=UTC)
N_EVENTS = 100


def iso(offset_s: float) -> str:
    return (START + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")


def warehouse(sensor: str, i: int, value: float) -> dict[str, Any]:
    return {
        "sensor_id": sensor,
        "warehouse_id": "wh1",
        "timestamp": iso(i),
        "reading_type": "temperature",
        "value": value,
        "unit": "°C",
        "is_anomaly": False,
        "anomaly_type": None,
        "metadata": {"zone": "A"},
    }


def gps(vehicle: str, i: int) -> dict[str, Any]:
    return {
        "vehicle_id": vehicle,
        "timestamp": iso(i * 10),
        "latitude": 51.9,
        "longitude": 4.47,
        "speed_kmh": 60.0 + i,
        "heading": 90.0,
        "is_anomaly": False,
        "anomaly_type": None,
        "route_adherence_score": 0.99,
        "metadata": {},
    }


def inventory(sku: str, i: int) -> dict[str, Any]:
    return {
        "sku_id": sku,
        "warehouse_id": "wh1",
        "timestamp": iso(i * 5),
        "previous_level": 400 - 10 * i,
        "new_level": 390 - 10 * i,
        "change_amount": -10,
        "transaction_type": "shipment",
        "is_anomaly": False,
        "anomaly_type": None,
        "metadata": {},
    }


def wait_for(predicate: Callable[[], bool], timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def make_group(env, topics: list[str], handler, **kwargs: Any) -> KafkaConsumerGroup:
    num_workers = kwargs.pop("num_workers", 2)
    kwargs.setdefault("poll_timeout_s", 0.2)
    kwargs.setdefault("lag_interval_s", 0.5)
    return KafkaConsumerGroup.from_config(
        env.cfg.kafka.bootstrap_servers,
        env.cfg.kafka.consumer.model_copy(update={"num_workers": num_workers}),
        topics,
        handler,
        consumer_factory=env.consumer_factory,
        **kwargs,
    )


def test_100_events_produced_are_all_consumed_exactly_once(kafka):
    received: list[str] = []
    lock = threading.Lock()

    def handler(message: ConsumedMessage) -> None:
        body = json.loads(message.value or b"{}")
        with lock:
            received.append(body["payload"]["timestamp"])

    topic = kafka.topic("warehouse_sensors")
    with kafka.producer() as producer:
        for i in range(N_EVENTS):
            assert producer.send_warehouse_event(warehouse(f"s{i % 10}", i, 3.0 + i / 1000))
        assert producer.flush(30) == 0  # everything delivered, nothing in flight

    group = make_group(kafka, [topic], handler)
    group.start()
    try:
        assert wait_for(
            lambda: group.processed_count >= N_EVENTS
        ), f"only {group.processed_count}/{N_EVENTS} events were consumed"
        assert wait_for(lambda: group.total_lag() == 0)
    finally:
        assert group.stop(timeout=30)

    assert len(received) == N_EVENTS
    assert sorted(received) == sorted(iso(i) for i in range(N_EVENTS))  # no loss, no duplicates
    assert group.dead_letter_count == 0
    assert kafka.read("dlq_events") == []  # nothing was dead-lettered


def test_each_source_is_routed_to_its_own_topic(kafka):
    with kafka.producer() as producer:
        for i in range(10):
            assert producer.send_warehouse_event(warehouse("wh1-A-temp-01", i, 3.0))
            assert producer.send_gps_event(gps("veh-001", i))
            assert producer.send_inventory_event(inventory("SKU-0001", i))
        assert producer.flush(30) == 0

    sensors = kafka.read("warehouse_sensors", at_least=10)
    vehicles = kafka.read("vehicle_gps", at_least=10)
    skus = kafka.read("inventory_events", at_least=10)

    assert {m["payload"]["sensor_id"] for m in sensors} == {"wh1-A-temp-01"} and len(sensors) == 10
    assert {m["payload"]["vehicle_id"] for m in vehicles} == {"veh-001"} and len(vehicles) == 10
    assert {m["payload"]["sku_id"] for m in skus} == {"SKU-0001"} and len(skus) == 10
    for message in sensors + vehicles + skus:  # the unified KafkaMessage wrapper
        assert {"message_id", "source_type", "schema_version", "payload"} <= set(message)
    assert len({m["message_id"] for m in sensors + vehicles + skus}) == 30


def test_events_of_one_entity_arrive_in_order(kafka):
    seen: dict[str, list[str]] = {}
    lock = threading.Lock()

    def handler(message: ConsumedMessage) -> None:
        payload = json.loads(message.value or b"{}")["payload"]
        with lock:
            seen.setdefault(payload["sensor_id"], []).append(payload["timestamp"])

    sensors = [f"s{k}" for k in range(8)]
    with kafka.producer() as producer:
        for i in range(30):
            for s in sensors:
                producer.send_warehouse_event(warehouse(s, i, 3.0))
        assert producer.flush(30) == 0

    group = make_group(kafka, [kafka.topic("warehouse_sensors")], handler, num_workers=3)
    group.start()
    try:
        assert wait_for(lambda: group.processed_count >= 30 * len(sensors))
    finally:
        group.stop(timeout=30)
    for s in sensors:  # keyed by entity -> one partition -> in order, whichever worker has it
        assert seen[s] == [iso(i) for i in range(30)]


def test_poison_message_is_dead_lettered_and_the_stream_continues(kafka):
    good: list[str] = []
    lock = threading.Lock()

    def handler(message: ConsumedMessage) -> None:
        try:
            body = json.loads(message.value or b"")
        except ValueError as exc:
            raise PermanentProcessingError(f"not JSON: {exc}") from exc
        with lock:
            good.append(body["payload"]["sensor_id"])

    topic = kafka.topic("warehouse_sensors")
    raw = kafka.raw_producer()
    raw.produce(topic, value=b"this is not json", key=b"bad-sensor")
    raw.flush(30)
    with kafka.producer() as producer:
        for i in range(5):
            producer.send_warehouse_event(warehouse("good", i, 3.0))
        assert producer.flush(30) == 0

    dlq_raw = kafka.raw_producer()
    dead_letters = DeadLetterPublisher(dlq_raw, kafka.topic("dlq_events"))
    group = make_group(kafka, [topic], handler, dead_letter=dead_letters)
    group.start()
    try:
        assert wait_for(lambda: len(good) == 5 and group.dead_letter_count == 1)
    finally:
        group.stop(timeout=30)
    dead_letters.flush()

    (letter,) = kafka.read("dlq_events", at_least=1)
    assert letter["original_topic"] == topic
    assert letter["original_payload"] == "this is not json"
    assert letter["error"]["reason"] == "validation_error"
    assert good == ["good"] * 5  # the poison message did not block the partition


def test_feature_pipeline_builds_windowed_features_from_the_live_stream(kafka):
    sensor = "wh1-A-temp-01"
    with kafka.producer() as producer:
        for i in range(120):
            producer.send_warehouse_event(warehouse(sensor, i, 3.0 + 0.01 * i))
        assert producer.flush(30) == 0

    runtime = build_pipeline(
        kafka.cfg,
        consumer_factory=kafka.consumer_factory,
        dlq_producer=kafka.raw_producer(),
        metrics=ConsumerMetrics(CollectorRegistry()),
        poll_timeout_s=0.2,
        lag_interval_s=0.5,
    )
    runtime.group.start()
    try:
        assert wait_for(lambda: runtime.group.processed_count >= 120)
    finally:
        runtime.group.stop()

    features = runtime.cache.get_features(sensor, 60)
    assert features is not None and features.count == 60
    assert features.signals["value"].rate_of_change == pytest.approx(0.01, rel=1e-6)
    assert runtime.cache.get_features(sensor, 900).count == 120


def test_producer_health_check_sees_the_broker_and_its_topics(kafka):
    with kafka.producer() as producer:
        health = producer.check_health(timeout=10)
    assert health.healthy and health.to_dict()["status"] == "ok"
    assert health.missing_topics == ()  # all six topics of this test exist
