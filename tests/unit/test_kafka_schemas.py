import json
import uuid

import pytest
from pydantic import ValidationError

from src.data_simulation import SimulationOrchestrator
from src.kafka.schemas import (
    PAYLOAD_MODELS,
    SCHEMA_VERSION,
    DLQErrorInfo,
    DLQMessage,
    GPSReading,
    InventoryEvent,
    KafkaMessage,
    WarehouseReading,
)
from src.utils.config import SimulationConfig

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


def test_envelope_fields_and_json_roundtrip():
    message = KafkaMessage.from_payload("warehouse", WAREHOUSE)
    assert isinstance(message.message_id, uuid.UUID)
    assert message.schema_version == SCHEMA_VERSION
    assert message.entity_id == "wh1-A-temp-01"
    assert message.timestamp == WAREHOUSE["timestamp"]

    raw = message.to_json_bytes()
    doc = json.loads(raw)
    assert set(doc) == {
        "message_id",
        "source_type",
        "schema_version",
        "entity_id",
        "timestamp",
        "produced_at",
        "payload",
    }
    assert doc["produced_at"].endswith("Z")
    assert KafkaMessage.model_validate_json(raw) == message


@pytest.mark.parametrize(
    ("source", "record", "model"),
    [
        ("warehouse", WAREHOUSE, WarehouseReading),
        ("vehicle", GPS, GPSReading),
        ("inventory", INVENTORY, InventoryEvent),
    ],
)
def test_payload_parsed_with_model_matching_source(source, record, model):
    message = KafkaMessage.from_payload(source, record)
    assert isinstance(message.payload, model)
    assert isinstance(KafkaMessage.model_validate_json(message.to_json_bytes()).payload, model)


def test_unique_message_ids():
    ids = {KafkaMessage.from_payload("warehouse", WAREHOUSE).message_id for _ in range(50)}
    assert len(ids) == 50


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        KafkaMessage.from_payload("warehouse", {**WAREHOUSE, "surprise": 1})


def test_anomaly_label_must_be_consistent():
    with pytest.raises(ValidationError, match="contradicts"):
        KafkaMessage.from_payload("warehouse", {**WAREHOUSE, "is_anomaly": True})
    ok = KafkaMessage.from_payload(
        "warehouse", {**WAREHOUSE, "is_anomaly": True, "anomaly_type": "point"}
    )
    assert ok.payload.anomaly_type == "point"


def test_inventory_levels_must_add_up():
    with pytest.raises(ValidationError, match="change_amount"):
        KafkaMessage.from_payload("inventory", {**INVENTORY, "change_amount": -99})


@pytest.mark.parametrize(
    ("record", "problem"),
    [
        ({**GPS, "latitude": 91.0}, "latitude"),
        ({**GPS, "speed_kmh": -1.0}, "speed_kmh"),
        ({**GPS, "route_adherence_score": 1.5}, "route_adherence_score"),
    ],
)
def test_gps_range_checks(record, problem):
    with pytest.raises(ValidationError, match=problem):
        KafkaMessage.from_payload("vehicle", record)


def test_bad_timestamp_is_rejected():
    with pytest.raises(ValidationError, match="ISO 8601"):
        KafkaMessage.from_payload("warehouse", {**WAREHOUSE, "timestamp": "yesterday"})


def test_missing_required_field_is_rejected():
    record = {k: v for k, v in WAREHOUSE.items() if k != "value"}
    with pytest.raises(ValidationError):
        KafkaMessage.from_payload("warehouse", record)


def test_payload_must_match_source_type():
    with pytest.raises(ValidationError, match="needs a"):
        KafkaMessage(
            source_type="vehicle",
            entity_id="x",
            timestamp=WAREHOUSE["timestamp"],
            payload=WarehouseReading.model_validate(WAREHOUSE),
        )


def test_unknown_source_type_is_rejected():
    with pytest.raises(KeyError):
        KafkaMessage.from_payload("bogus", WAREHOUSE)  # type: ignore[arg-type]


def test_dlq_message_serialises():
    dlq = DLQMessage(
        original_topic="warehouse-sensors",
        original_key="k",
        attempts=4,
        error=DLQErrorInfo(
            reason="delivery_failed", error_type="KafkaError", message="boom", retriable=True
        ),
        original_payload={"a": 1},
    )
    doc = json.loads(dlq.to_json_bytes())
    assert doc["original_payload"] == {"a": 1}
    assert doc["error"]["reason"] == "delivery_failed"
    assert doc["failed_at"].endswith("Z")


def test_every_simulator_record_matches_its_schema():
    """Contract test: the schemas accept exactly what the Day 2/3 simulators emit."""
    orchestrator = SimulationOrchestrator.from_config(SimulationConfig(seed=7), include_orders=True)
    seen = set()
    for event in orchestrator.run_batch(1800):
        message = KafkaMessage.from_payload(event["source_type"], event["payload"])
        assert message.entity_id == event["entity_id"]
        assert message.timestamp == event["timestamp"]
        seen.add(event["source_type"])
    assert seen == set(PAYLOAD_MODELS)
