import json
import logging
from collections import deque
from types import SimpleNamespace

import pytest
from confluent_kafka import KafkaError, KafkaException
from prometheus_client import CollectorRegistry

from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.utils.config import KafkaConfig, TopicConfig, TopicsConfig

WAREHOUSE = {
    "sensor_id": "wh1-A-temp-01",
    "warehouse_id": "wh1",
    "timestamp": "2026-11-20T08:00:00Z",
    "reading_type": "temperature",
    "value": 3.02,
    "unit": "°C",
    "is_anomaly": False,
    "anomaly_type": None,
    "metadata": {"zone": "A"},
}
GPS = {
    "vehicle_id": "veh-001",
    "timestamp": "2026-11-20T08:00:00Z",
    "latitude": 51.9,
    "longitude": 4.47,
    "speed_kmh": 63.4,
    "heading": 187.2,
    "is_anomaly": False,
    "anomaly_type": None,
    "route_adherence_score": 0.99,
    "metadata": {},
}
INVENTORY = {
    "sku_id": "SKU-0001",
    "warehouse_id": "wh1",
    "timestamp": "2026-11-20T08:00:05Z",
    "previous_level": 412,
    "new_level": 397,
    "change_amount": -15,
    "transaction_type": "shipment",
    "is_anomaly": False,
    "anomaly_type": None,
    "metadata": {},
}
ORDER = {
    "order_id": "ORD-000001",
    "sku_id": "SKU-0001",
    "warehouse_id": "wh1",
    "timestamp": "2026-11-20T08:00:09Z",
    "status": "picked",
    "previous_status": "created",
    "quantity": 3,
    "stage_duration_s": 41.2,
    "expected_duration_s": 40.0,
    "is_anomaly": False,
    "anomaly_type": None,
    "metadata": {},
}

# Exactly what librdkafka reports for a dead broker: a timeout flagged retriable()==False.
RETRIABLE = KafkaError(KafkaError._MSG_TIMED_OUT, "timed out", retriable=False)
FATAL_TOO_LARGE = KafkaError(KafkaError.MSG_SIZE_TOO_LARGE, "too large", retriable=False)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeProducer:
    """In-memory stand-in for confluent_kafka.Producer.

    ``produce`` queues a delivery report; ``poll`` delivers them. ``script`` maps a topic to
    a deque of outcomes consumed one per produce call: ``None`` = delivered, a ``KafkaError``
    = delivery report with that error, an exception instance = ``produce`` raises it.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.produced = []  # (topic, key, decoded_value) of every accepted produce() call
        self.calls = 0
        self.script = {}
        self._reports = deque()
        self.metadata = SimpleNamespace(
            topics={
                n: None
                for n in (
                    "warehouse-sensors",
                    "vehicle-gps",
                    "inventory-events",
                    "anomaly-alerts",
                    "dlq-events",
                    "model-health",
                )
            },
            brokers={1: object()},
        )
        self.list_topics_error = None

    def produce(self, topic, value=None, key=None, on_delivery=None, **_):
        self.calls += 1
        outcome = None
        queue = self.script.get(topic)
        if queue:
            outcome = queue.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        self.produced.append((topic, key, json.loads(value)))
        msg = SimpleNamespace(partition=lambda: 0, offset=lambda: len(self.produced))
        self._reports.append((on_delivery, outcome, msg))

    def poll(self, timeout=0):
        n = 0
        while self._reports:
            callback, err, msg = self._reports.popleft()
            callback(err, msg)
            n += 1
        return n

    def flush(self, timeout=None):
        self.poll()
        return 0

    def list_topics(self, timeout=None):
        if self.list_topics_error:
            raise self.list_topics_error
        return self.metadata

    # helpers
    def to(self, topic):
        return [(k, v) for t, k, v in self.produced if t == topic]


def make_topics(**overrides):
    return TopicsConfig(**overrides)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def registry():
    return CollectorRegistry()


@pytest.fixture
def fake():
    return FakeProducer()


@pytest.fixture
def manager(fake, clock, registry):
    mgr = KafkaProducerManager(
        "localhost:9092",
        make_topics(),
        max_retries=3,
        retry_backoff_ms=100,
        background_polling=False,
        producer_factory=lambda config: fake,
        metrics=ProducerMetrics(registry),
        clock=clock,
    )
    yield mgr
    mgr.close(timeout=0.1)


def counter(registry, name, **labels):
    return registry.get_sample_value(name, labels) or 0.0


# --------------------------------------------------------------------------- #
# Happy path and routing
# --------------------------------------------------------------------------- #
def test_convenience_methods_route_to_topics_with_entity_key(manager, fake, registry):
    assert manager.send_warehouse_event(WAREHOUSE)
    assert manager.send_gps_event(GPS)
    assert manager.send_inventory_event(INVENTORY)
    assert manager.send_order_event(ORDER)
    manager.pump()

    assert [(t, k) for t, k, _ in fake.produced] == [
        ("warehouse-sensors", "wh1-A-temp-01"),
        ("vehicle-gps", "veh-001"),
        ("inventory-events", "SKU-0001"),
        ("inventory-events", "ORD-000001"),
    ]
    envelope = fake.produced[0][2]
    assert envelope["source_type"] == "warehouse"
    assert envelope["payload"]["value"] == 3.02
    assert fake.produced[3][2]["source_type"] == "order"
    assert manager.in_flight == 0
    assert counter(registry, "scad_kafka_messages_sent_total", topic="warehouse-sensors") == 1
    assert counter(registry, "scad_kafka_messages_sent_total", topic="inventory-events") == 2


def test_send_event_accepts_orchestrator_envelope(manager, fake):
    event = {
        "source_type": "vehicle",
        "entity_id": "veh-001",
        "timestamp": GPS["timestamp"],
        "payload": GPS,
    }
    assert manager.send_event(event)
    manager.pump()
    assert fake.to("vehicle-gps")[0][0] == "veh-001"


def test_send_batch_counts_queued_events(manager, fake):
    events = [{"source_type": "warehouse", "payload": WAREHOUSE}] * 5
    assert manager.send_batch(events) == 5
    manager.pump()
    assert len(fake.to("warehouse-sensors")) == 5


def test_client_config_from_arguments(clock, registry):
    captured = {}

    def factory(config):
        captured.update(config)
        return FakeProducer()

    mgr = KafkaProducerManager(
        "broker:9092",
        make_topics(),
        acks="all",
        batch_size=32_768,
        linger_ms=25,
        compression_type="lz4",
        background_polling=False,
        producer_factory=factory,
        metrics=ProducerMetrics(registry),
        extra_config={"client.id": "custom"},
    )
    mgr.close(0.1)
    assert captured["bootstrap.servers"] == "broker:9092"
    assert captured["batch.size"] == 32_768
    assert captured["linger.ms"] == 25
    assert captured["compression.type"] == "lz4"
    assert captured["enable.idempotence"] is True
    assert captured["client.id"] == "custom"


def test_idempotence_off_unless_acks_all(registry):
    captured = {}
    KafkaProducerManager(
        "b:9092",
        make_topics(),
        acks="1",
        background_polling=False,
        producer_factory=lambda c: captured.update(c) or FakeProducer(),
        metrics=ProducerMetrics(registry),
    ).close(0.1)
    assert captured["enable.idempotence"] is False


def test_from_config_uses_yaml_values_and_overrides(registry):
    captured = {}
    cfg = KafkaConfig()
    mgr = KafkaProducerManager.from_config(
        cfg,
        linger_ms=50,
        background_polling=False,
        producer_factory=lambda c: captured.update(c) or FakeProducer(),
        metrics=ProducerMetrics(registry),
    )
    mgr.close(0.1)
    assert captured["bootstrap.servers"] == "localhost:9092"
    assert captured["batch.size"] == cfg.producer.batch_size
    assert captured["linger.ms"] == 50  # override wins


@pytest.mark.parametrize(
    "kwargs", [{"max_retries": -1}, {"retry_backoff_ms": -1}, {"batch_size": -1}, {"linger_ms": -1}]
)
def test_invalid_arguments(kwargs, registry):
    with pytest.raises(ValueError):
        KafkaProducerManager(
            "b:9092",
            make_topics(),
            background_polling=False,
            producer_factory=lambda c: FakeProducer(),
            metrics=ProducerMetrics(registry),
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Retries with exponential backoff
# --------------------------------------------------------------------------- #
def test_retry_with_exponential_backoff_then_success(manager, fake, clock, registry):
    fake.script["warehouse-sensors"] = deque([RETRIABLE, RETRIABLE, None])
    manager.send_warehouse_event(WAREHOUSE)

    # A failure is only seen on the next poll(), and the backoff counts from that moment.
    manager.pump()  # attempt 1 failed -> retry due 100 ms from now
    assert fake.calls == 1
    assert manager.in_flight == 1

    clock.advance(0.099)
    manager.pump()
    assert fake.calls == 1  # backoff not over yet

    clock.advance(0.002)
    manager.pump()  # attempt 2 is sent ...
    assert fake.calls == 2
    manager.pump()  # ... and its failure is seen -> next retry due in 200 ms

    clock.advance(0.199)
    manager.pump()
    assert fake.calls == 2  # doubled backoff not over yet

    clock.advance(0.002)
    manager.pump()  # attempt 3 is sent
    manager.pump()  # ... and delivered
    assert fake.calls == 3
    assert manager.in_flight == 0
    assert counter(registry, "scad_kafka_retries_total", topic="warehouse-sensors") == 2
    assert counter(registry, "scad_kafka_messages_sent_total", topic="warehouse-sensors") == 1
    assert counter(registry, "scad_kafka_messages_failed_total", topic="warehouse-sensors") == 0
    assert fake.to("dlq-events") == []


def test_retries_exhausted_routes_to_dlq_with_original_payload(manager, fake, clock, registry):
    fake.script["warehouse-sensors"] = deque([RETRIABLE] * 10)
    manager.send_warehouse_event(WAREHOUSE)

    for step in (0, 0.11, 0.21, 0.41, 0.1):
        clock.advance(step)
        manager.pump()
        manager.pump()

    attempts = [p for p in fake.script["warehouse-sensors"]]
    assert len(attempts) == 10 - 4  # 1 first attempt + 3 retries consumed
    ((key, dlq),) = fake.to("dlq-events")
    assert key == "wh1-A-temp-01"
    assert dlq["original_topic"] == "warehouse-sensors"
    assert dlq["attempts"] == 4
    assert dlq["error"]["reason"] == "delivery_failed"
    assert dlq["error"]["error_code"] == "_MSG_TIMED_OUT"
    assert dlq["error"]["retriable"] is True
    assert dlq["original_payload"]["payload"]["sensor_id"] == "wh1-A-temp-01"
    assert dlq["original_payload"]["source_type"] == "warehouse"
    assert manager.in_flight == 0
    assert counter(registry, "scad_kafka_retries_total", topic="warehouse-sensors") == 3
    assert counter(registry, "scad_kafka_messages_failed_total", topic="warehouse-sensors") == 1
    assert (
        counter(
            registry,
            "scad_kafka_dlq_messages_total",
            topic="warehouse-sensors",
            reason="delivery_failed",
        )
        == 1
    )
    assert counter(registry, "scad_kafka_messages_sent_total", topic="warehouse-sensors") == 0


def test_non_retriable_error_goes_straight_to_dlq(manager, fake, registry):
    fake.script["vehicle-gps"] = deque([FATAL_TOO_LARGE])
    manager.send_gps_event(GPS)
    manager.pump()
    manager.pump()

    assert fake.calls == 2  # the original attempt and the DLQ write; no retry
    ((_, dlq),) = fake.to("dlq-events")
    assert dlq["attempts"] == 1
    assert dlq["error"]["retriable"] is False
    assert dlq["error"]["error_code"] == "MSG_SIZE_TOO_LARGE"
    assert counter(registry, "scad_kafka_retries_total", topic="vehicle-gps") == 0


def test_max_retries_zero_dead_letters_on_first_failure(clock, registry):
    fake = FakeProducer()
    fake.script["warehouse-sensors"] = deque([RETRIABLE])
    mgr = KafkaProducerManager(
        "b:9092",
        make_topics(),
        max_retries=0,
        background_polling=False,
        producer_factory=lambda c: fake,
        metrics=ProducerMetrics(registry),
        clock=clock,
    )
    mgr.send_warehouse_event(WAREHOUSE)
    mgr.pump()
    mgr.pump()
    assert len(fake.to("dlq-events")) == 1
    mgr.close(0.1)


def test_full_local_queue_is_drained_and_retried_inline(manager, fake):
    fake.script["warehouse-sensors"] = deque([BufferError("queue full")])
    assert manager.send_warehouse_event(WAREHOUSE)
    manager.pump()
    assert len(fake.to("warehouse-sensors")) == 1
    assert fake.to("dlq-events") == []


def test_persistently_full_queue_falls_back_to_backoff_retry(manager, fake, clock):
    fake.script["warehouse-sensors"] = deque([BufferError("full"), BufferError("full")])
    manager.send_warehouse_event(WAREHOUSE)
    assert fake.produced == []  # both inline attempts failed
    clock.advance(0.11)
    manager.pump()
    manager.pump()
    assert len(fake.to("warehouse-sensors")) == 1
    assert manager.in_flight == 0


def test_produce_exception_is_dead_lettered(manager, fake, registry):
    err = KafkaException(KafkaError(KafkaError._UNKNOWN_TOPIC, "no such topic", retriable=False))
    fake.script["warehouse-sensors"] = deque([err])
    assert manager.send_warehouse_event(WAREHOUSE)
    manager.pump()
    ((_, dlq),) = fake.to("dlq-events")
    assert dlq["error"]["reason"] == "produce_failed"
    assert dlq["error"]["error_code"] == "_UNKNOWN_TOPIC"


def test_unexpected_produce_exception_does_not_crash_stream(manager, fake):
    fake.script["warehouse-sensors"] = deque([RuntimeError("librdkafka bug")])
    assert manager.send_warehouse_event(WAREHOUSE)
    manager.send_warehouse_event(WAREHOUSE)
    manager.pump()
    manager.pump()
    assert len(fake.to("warehouse-sensors")) == 1
    ((_, dlq),) = fake.to("dlq-events")
    assert dlq["error"]["error_type"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# Dead-letter queue
# --------------------------------------------------------------------------- #
def test_invalid_payload_goes_to_dlq_without_touching_topic(manager, fake, registry):
    bad = {**WAREHOUSE, "reading_type": "pressure"}
    assert manager.send_warehouse_event(bad) is False
    manager.pump()
    assert fake.to("warehouse-sensors") == []
    ((key, dlq),) = fake.to("dlq-events")
    assert key == "wh1-A-temp-01"
    assert dlq["error"]["reason"] == "validation_error"
    assert dlq["original_payload"]["reading_type"] == "pressure"
    assert dlq["original_topic"] == "warehouse-sensors"
    assert (
        counter(
            registry,
            "scad_kafka_dlq_messages_total",
            topic="warehouse-sensors",
            reason="validation_error",
        )
        == 1
    )


def test_oversized_message_goes_to_dlq(fake, clock, registry):
    topics = make_topics(
        warehouse_sensors=TopicConfig(
            name="warehouse-sensors", partitions=6, max_message_bytes=1024
        )
    )
    mgr = KafkaProducerManager(
        "b:9092",
        topics,
        background_polling=False,
        producer_factory=lambda c: fake,
        metrics=ProducerMetrics(registry),
        clock=clock,
    )
    big = {**WAREHOUSE, "metadata": {"blob": "x" * 5000}}
    assert mgr.send_warehouse_event(big) is False
    mgr.pump()
    assert fake.to("warehouse-sensors") == []
    ((_, dlq),) = fake.to("dlq-events")
    assert dlq["error"]["reason"] == "message_too_large"
    mgr.close(0.1)


def test_oversized_original_is_truncated_in_dlq(fake, clock, registry):
    topics = make_topics(
        dlq_events=TopicConfig(name="dlq-events", partitions=1, max_message_bytes=8192),
        warehouse_sensors=TopicConfig(
            name="warehouse-sensors", partitions=6, max_message_bytes=1024
        ),
    )
    mgr = KafkaProducerManager(
        "b:9092",
        topics,
        background_polling=False,
        producer_factory=lambda c: fake,
        metrics=ProducerMetrics(registry),
        clock=clock,
    )
    mgr.send_warehouse_event({**WAREHOUSE, "metadata": {"blob": "x" * 20_000}})
    mgr.pump()
    ((_, dlq),) = fake.to("dlq-events")
    assert dlq["payload_truncated"] is True
    assert isinstance(dlq["original_payload"], str)
    mgr.close(0.1)


def test_invalid_envelope_goes_to_dlq(manager, fake):
    assert manager.send_event({"source_type": "bogus", "payload": {}}) is False
    assert manager.send_event({"source_type": "warehouse", "payload": "nope"}) is False
    manager.pump()
    assert len(fake.to("dlq-events")) == 2


def test_failed_dlq_write_is_counted_and_logged(manager, fake, registry, caplog):
    caplog.set_level(logging.DEBUG)
    fake.script["dlq-events"] = deque([FATAL_TOO_LARGE])
    manager.send_warehouse_event({**WAREHOUSE, "reading_type": "pressure"})
    manager.pump()
    assert manager.in_flight == 0
    assert counter(registry, "scad_kafka_dlq_failures_total", topic="dlq-events") == 1
    assert any("kafka_dlq_write_failed_message_lost" in str(r.msg) for r in caplog.records)


def test_dlq_delivery_does_not_count_as_sent(manager, fake, registry):
    manager.send_warehouse_event({**WAREHOUSE, "reading_type": "pressure"})
    manager.pump()
    assert counter(registry, "scad_kafka_messages_sent_total", topic="dlq-events") == 0


# --------------------------------------------------------------------------- #
# Flush, close, background polling
# --------------------------------------------------------------------------- #
def test_flush_waits_for_everything(manager, fake):
    for _ in range(20):
        manager.send_warehouse_event(WAREHOUSE)
    assert manager.flush(timeout=2.0) == 0
    assert len(fake.to("warehouse-sensors")) == 20


def test_flush_reports_messages_still_in_flight(fake, registry):
    fake.script["warehouse-sensors"] = deque([RETRIABLE] * 5)
    mgr = KafkaProducerManager(
        "b:9092",
        make_topics(),
        retry_backoff_ms=60_000,  # retry will not be due within the test
        background_polling=False,
        producer_factory=lambda c: fake,
        metrics=ProducerMetrics(registry),
    )
    mgr.send_warehouse_event(WAREHOUSE)
    assert mgr.flush(timeout=0.2) == 1
    mgr.close(timeout=0.05)


def test_close_is_idempotent(manager):
    manager.send_warehouse_event(WAREHOUSE)
    assert manager.close(1.0) == 0
    assert manager.close(1.0) == 0


def test_background_thread_delivers_and_retries(registry):
    fake = FakeProducer()
    fake.script["warehouse-sensors"] = deque([RETRIABLE, None])
    with KafkaProducerManager(
        "b:9092",
        make_topics(),
        retry_backoff_ms=10,
        background_polling=True,
        producer_factory=lambda c: fake,
        metrics=ProducerMetrics(registry),
    ) as mgr:
        mgr.send_warehouse_event(WAREHOUSE)
        assert mgr.flush(timeout=5.0) == 0
    assert fake.calls == 2  # the failed first attempt and the successful retry
    assert counter(registry, "scad_kafka_retries_total", topic="warehouse-sensors") == 1
    assert counter(registry, "scad_kafka_messages_sent_total", topic="warehouse-sensors") == 1


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #
def test_health_ok(manager):
    health = manager.check_health()
    assert health.healthy
    assert health.brokers == 1
    assert health.missing_topics == ()
    assert health.to_dict()["status"] == "ok"


def test_health_reports_missing_topics(manager, fake):
    del fake.metadata.topics["dlq-events"]
    health = manager.check_health()
    assert not health.healthy
    assert health.missing_topics == ("dlq-events",)
    assert "dlq-events" in health.error
    assert health.to_dict()["status"] == "unavailable"


def test_health_reports_connection_failure(manager, fake):
    fake.list_topics_error = KafkaException(KafkaError(KafkaError._TRANSPORT, "all brokers down"))
    health = manager.check_health()
    assert not health.healthy
    assert "all brokers down" in health.error


def test_health_reports_no_brokers(manager, fake):
    fake.metadata.brokers = {}
    health = manager.check_health()
    assert not health.healthy
    assert "no brokers" in health.error


def test_topic_for_and_unknown_source(manager):
    assert manager.topic_for("vehicle") == "vehicle-gps"
    assert manager.topic_for("order") == "inventory-events"
    with pytest.raises(KeyError):
        manager.topic_for("bogus")


def test_librdkafka_timeout_flag_does_not_disable_retries():
    """Regression: a real delivery timeout reports retriable()==False yet must be retried."""
    assert RETRIABLE.retriable() is False
    from src.kafka.producer import _describe_error

    assert _describe_error(RETRIABLE)[3] is True
    assert _describe_error(KafkaError(KafkaError._TRANSPORT, "down"))[3] is True
    assert _describe_error(FATAL_TOO_LARGE)[3] is False
    assert _describe_error(KafkaError(KafkaError._MSG_TIMED_OUT, "x", fatal=True))[3] is False
