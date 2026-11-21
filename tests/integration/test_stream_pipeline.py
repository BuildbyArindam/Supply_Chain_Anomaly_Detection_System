"""End to end: KafkaProducerManager -> mock broker -> consumer group -> aggregator -> cache."""

import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from src.kafka.feature_cache import FeatureCache
from src.kafka.pipeline import (
    DeadLetterPublisher,
    StreamingFeaturePipeline,
    build_pipeline,
)
from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.kafka.stream_processor import WindowedAggregator
from src.utils.config import AppConfig
from src.utils.metrics import ConsumerMetrics
from tests.mock_kafka import MockBroker

START = datetime(2026, 11, 20, 8, 0, 0, tzinfo=UTC)
TOPICS = {
    "warehouse-sensors": 6,
    "vehicle-gps": 4,
    "inventory-events": 4,
    "anomaly-alerts": 3,
    "dlq-events": 1,
    "model-health": 1,
}


def iso(offset_s):
    return (START + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")


def warehouse(sensor, offset_s, value, anomaly=None):
    return {
        "sensor_id": sensor,
        "warehouse_id": "wh1",
        "timestamp": iso(offset_s),
        "reading_type": "temperature",
        "value": value,
        "unit": "°C",
        "is_anomaly": anomaly is not None,
        "anomaly_type": anomaly,
        "metadata": {"zone": "A"},
    }


def gps(vehicle, offset_s, speed):
    return {
        "vehicle_id": vehicle,
        "timestamp": iso(offset_s),
        "latitude": 51.9,
        "longitude": 4.47,
        "speed_kmh": speed,
        "heading": 90.0,
        "is_anomaly": False,
        "anomaly_type": None,
        "route_adherence_score": 0.99,
        "metadata": {},
    }


def inventory(sku, offset_s, previous, new):
    return {
        "sku_id": sku,
        "warehouse_id": "wh1",
        "timestamp": iso(offset_s),
        "previous_level": previous,
        "new_level": new,
        "change_amount": new - previous,
        "transaction_type": "shipment",
        "is_anomaly": False,
        "anomaly_type": None,
        "metadata": {},
    }


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def broker():
    return MockBroker(TOPICS)


@pytest.fixture
def cfg():
    return AppConfig.model_validate(
        {"kafka": {"consumer": {"auto_offset_reset": "earliest", "num_workers": 3}}}
    )


@pytest.fixture
def producer(broker, cfg):
    mgr = KafkaProducerManager(
        "localhost:9092",
        cfg.kafka.topics,
        background_polling=False,
        producer_factory=broker.producer_factory,
        metrics=ProducerMetrics(CollectorRegistry()),
    )
    yield mgr
    mgr.close(timeout=0.1)


@pytest.fixture
def registry():
    return CollectorRegistry()


@pytest.fixture
def runtime(broker, cfg, registry):
    rt = build_pipeline(
        cfg,
        consumer_factory=broker.consumer_factory,
        dlq_producer=broker.producer_factory({}),
        metrics=ConsumerMetrics(registry),
        poll_timeout_s=0.01,
        retry_backoff_s=0.001,
        lag_interval_s=0.0,
    )
    yield rt
    rt.group.stop()


def run_until_drained(runtime, broker, topics, timeout=10.0):
    runtime.group.start()
    assert wait_for(
        lambda: all(broker.lag(runtime.group._config["group.id"], t) == 0 for t in topics),
        timeout,
    ), "consumer did not drain the topics"


def sample(registry, name, **labels):
    return registry.get_sample_value(name, labels) or 0.0


def test_all_three_sources_flow_into_the_cache_with_correct_features(runtime, broker, producer):
    for i in range(120):  # 2 minutes of 1 Hz readings
        producer.send_warehouse_event(warehouse("wh1-A-temp-01", i, 3.0 + 0.01 * i))
    for i in range(12):
        producer.send_gps_event(gps("veh-001", i * 10, 60.0 + i))
    for i in range(10):
        producer.send_inventory_event(inventory("SKU-0001", i * 5, 400 - 10 * i, 390 - 10 * i))
    producer.flush(1)

    run_until_drained(runtime, broker, ["warehouse-sensors", "vehicle-gps", "inventory-events"])
    cache = runtime.cache

    latest = cache.get_features("wh1-A-temp-01", 60)
    assert latest.count == 60 and latest.window_end.isoformat().startswith("2026-11-20T08:01:59")
    assert latest.signals["value"].rate_of_change == pytest.approx(0.01, rel=1e-6)
    assert latest.context["unit"] == "°C"
    assert cache.get_features("wh1-A-temp-01", 900).count == 120
    assert set(cache.get_all_windows("wh1-A-temp-01")) == {60, 300, 900}

    vehicle = cache.get_features("veh-001", 60)
    assert vehicle.signals["speed_kmh"].last == 71.0 and vehicle.count == 6
    sku = cache.get_features("SKU-0001", 60)
    assert sku.signals["change_amount"].mean == -10.0
    assert sku.signals["new_level"].last == 300.0


def test_history_gives_lstm_ready_sequences_in_event_order(runtime, broker, producer):
    for i in range(90):
        producer.send_warehouse_event(warehouse("s1", i, float(i)))
    producer.flush(1)
    run_until_drained(runtime, broker, ["warehouse-sensors"])

    seq = runtime.cache.get_history("s1", 60, window_seconds=60)
    assert len(seq) == 60
    ends = [f.window_end for f in seq]
    assert ends == sorted(ends) and ends[-1] == START + timedelta(seconds=89)
    assert [f.signals["value"].last for f in seq] == [float(i) for i in range(30, 90)]


def test_each_entity_stays_in_order_across_parallel_workers(runtime, broker, producer):
    sensors = [f"s{k}" for k in range(12)]
    for i in range(60):
        for s in sensors:
            producer.send_warehouse_event(warehouse(s, i, float(i)))
    producer.flush(2)
    run_until_drained(runtime, broker, ["warehouse-sensors"])
    for s in sensors:
        f = runtime.cache.get_features(s, 60)
        assert f.count == 60 and f.signals["value"].last == 59.0
        assert f.signals["value"].rate_of_change == pytest.approx(1.0)
    assert runtime.pipeline.aggregator.entity_count == 12


def test_late_events_are_accepted_within_30s_and_dropped_beyond(
    runtime, broker, producer, registry
):
    producer.send_warehouse_event(warehouse("s1", 0, 1.0))
    producer.send_warehouse_event(warehouse("s1", 100, 2.0))
    producer.send_warehouse_event(warehouse("s1", 80, 3.0))  # 20 s late: accepted
    producer.send_warehouse_event(warehouse("s1", 60, 4.0))  # 40 s late: dropped
    producer.send_warehouse_event(warehouse("s1", 101, 5.0))
    producer.flush(1)
    run_until_drained(runtime, broker, ["warehouse-sensors"])
    assert sample(registry, "scad_pipeline_events_total", outcome="late_accepted") == 1
    assert sample(registry, "scad_pipeline_events_total", outcome="dropped_late") == 1
    f = runtime.cache.get_features("s1", 60)
    assert f.count == 3 and f.signals["value"].max == 5.0


def test_malformed_messages_go_to_the_dlq_and_do_not_block_the_stream(
    runtime, broker, producer, registry
):
    broker.append("warehouse-sensors", b"s1", b"this is not json")
    broker.append("warehouse-sensors", b"s1", json.dumps({"hello": "world"}).encode())
    broker.append("warehouse-sensors", b"s1", None)
    producer.send_warehouse_event(warehouse("s1", 0, 1.0))
    producer.flush(1)
    run_until_drained(runtime, broker, ["warehouse-sensors"])
    runtime.dead_letter.flush()

    assert runtime.cache.get_features("s1", 60).count == 1
    dlq = [json.loads(r.value) for r in broker.messages("dlq-events")]
    assert len(dlq) == 3
    assert {d["error"]["reason"] for d in dlq} == {"validation_error"}
    assert {d["original_topic"] for d in dlq} == {"warehouse-sensors"}
    assert any(d["original_payload"] == "this is not json" for d in dlq)
    assert (
        sample(
            registry,
            "scad_messages_consumed_total",
            topic="warehouse-sensors",
            status="dead_lettered",
        )
        == 3
    )


def test_replayed_messages_are_not_double_counted(runtime, broker, producer, registry):
    for i in range(10):
        producer.send_warehouse_event(warehouse("s1", i, float(i)))
    producer.flush(1)
    for record in list(broker.messages("warehouse-sensors")):  # at-least-once redelivery
        broker.append("warehouse-sensors", record.key, record.value)
    run_until_drained(runtime, broker, ["warehouse-sensors"])
    assert runtime.cache.get_features("s1", 60).count == 10
    assert sample(registry, "scad_pipeline_events_total", outcome="duplicate") == 10


def test_order_events_on_the_inventory_topic_are_skipped(runtime, broker, producer, registry):
    producer.send_order_event(
        {
            "order_id": "ORD-1",
            "sku_id": "SKU-1",
            "warehouse_id": "wh1",
            "timestamp": iso(0),
            "status": "created",
            "quantity": 1,
            "is_anomaly": False,
            "anomaly_type": None,
            "metadata": {},
        }
    )
    producer.send_inventory_event(inventory("SKU-1", 1, 10, 9))
    producer.flush(1)
    run_until_drained(runtime, broker, ["inventory-events"])
    assert sample(registry, "scad_pipeline_events_total", outcome="unsupported") == 1
    assert runtime.cache.get_features("SKU-1", 60).count == 1
    assert runtime.cache.get_features("ORD-1", 60) is None


def test_metrics_cover_throughput_latency_age_lag_and_cache(runtime, broker, producer, registry):
    for i in range(50):
        producer.send_warehouse_event(warehouse("s1", i, float(i)))
    producer.flush(1)
    run_until_drained(runtime, broker, ["warehouse-sensors"])

    assert (
        sample(registry, "scad_messages_consumed_total", topic="warehouse-sensors", status="ok")
        == 50
    )
    assert sample(registry, "scad_processing_latency_seconds_count", source_type="warehouse") == 50
    assert sample(registry, "scad_processing_latency_seconds_sum", source_type="warehouse") > 0
    assert sample(registry, "scad_event_age_seconds_count") == 50
    assert sample(registry, "scad_feature_cache_entities") == 1
    assert sample(registry, "scad_aggregator_entities") == 1

    runtime.cache.get_features("s1")
    runtime.cache.get_features("missing")
    assert sample(registry, "scad_cache_hit_ratio") == pytest.approx(0.5)
    assert runtime.group.total_lag() == 0


def test_graceful_shutdown_commits_everything_processed(runtime, broker, producer):
    for i in range(30):
        producer.send_warehouse_event(warehouse("s1", i, float(i)))
    producer.flush(1)
    run_until_drained(runtime, broker, ["warehouse-sensors"])
    assert runtime.group.stop()
    assert broker.lag("anomaly-detector", "warehouse-sensors") == 0
    assert not runtime.group.is_running


def test_restarted_group_resumes_from_committed_offsets(broker, cfg, producer):
    def build():
        reg = CollectorRegistry()
        return build_pipeline(
            cfg,
            consumer_factory=broker.consumer_factory,
            dlq_producer=broker.producer_factory({}),
            metrics=ConsumerMetrics(reg),
            poll_timeout_s=0.01,
            lag_interval_s=0.0,
        )

    for i in range(20):
        producer.send_warehouse_event(warehouse("s1", i, float(i)))
    producer.flush(1)
    first = build()
    run_until_drained(first, broker, ["warehouse-sensors"])
    first.group.stop()
    for i in range(20, 30):
        producer.send_warehouse_event(warehouse("s1", i, float(i)))
    producer.flush(1)
    second = build()
    run_until_drained(second, broker, ["warehouse-sensors"])
    second.group.stop()
    assert second.group.processed_count == 10  # only the new messages


def test_pipeline_handle_can_be_driven_directly_without_a_consumer():
    registry = CollectorRegistry()
    cache = FeatureCache()
    pipeline = StreamingFeaturePipeline(
        WindowedAggregator((60,), 30), cache, metrics=ConsumerMetrics(registry)
    )
    from src.kafka.consumer import ConsumedMessage
    from src.kafka.schemas import KafkaMessage

    value = KafkaMessage.from_payload("warehouse", warehouse("s1", 0, 2.0)).to_json_bytes()
    pipeline.handle(ConsumedMessage("warehouse-sensors", 0, 0, "s1", value))
    assert cache.get_features("s1", 60).signals["value"].mean == 2.0


def test_dead_letter_publisher_marks_handler_failures_as_processing_failed(broker):
    from src.kafka.consumer import ConsumedMessage

    producer = broker.producer_factory({})
    DeadLetterPublisher(producer, "dlq-events")(
        ConsumedMessage("vehicle-gps", 1, 7, "veh-1", b"{}"), RuntimeError("db down"), 4
    )
    doc = json.loads(broker.messages("dlq-events")[0].value)
    assert doc["error"]["reason"] == "processing_failed" and doc["attempts"] == 4
    assert doc["original_key"] == "veh-1"
