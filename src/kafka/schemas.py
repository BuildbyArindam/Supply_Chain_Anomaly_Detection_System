"""Message schemas for the Kafka layer (Day 4).

Two levels of model live here:

* **Payload models** - one Pydantic v2 model per simulator record, field for field
  the dicts produced by the Day 2/3 simulators (:class:`WarehouseReading`,
  :class:`GPSReading`, :class:`InventoryEvent`, :class:`OrderEvent`). They forbid
  unknown fields on purpose: if a simulator starts emitting a new field, the
  affected events land in the dead-letter queue instead of silently flowing
  downstream with a schema nobody reviewed. Update the model and bump
  :data:`SCHEMA_VERSION` when that happens.
* **The wire envelope** - :class:`KafkaMessage`, the single JSON document that is
  actually written to a topic::

      {
          "message_id": "0b0e...",                 # UUID, unique per message
          "source_type": "warehouse",              # warehouse | vehicle | inventory | order
          "schema_version": "1.0",
          "entity_id": "wh1-A-temp-01",            # also the Kafka partition key
          "timestamp": "2026-11-20T08:00:00Z",     # event time (from the simulator)
          "produced_at": "2026-11-20T08:00:00.123456Z",   # when the producer built it
          "payload": {...}                         # one of the payload models above
      }

  ``entity_id`` is the partition key, so every reading of one sensor / vehicle / SKU
  goes to one partition and stays in order - which the windowed features of Day 5
  rely on.

Consumers can rebuild the typed object with ``KafkaMessage.model_validate_json(raw)``:
the ``payload`` is parsed with the model that matches ``source_type``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any, Final, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Final[str] = "1.0"

SourceType = Literal["warehouse", "vehicle", "inventory", "order"]
SOURCE_TYPES: Final[tuple[SourceType, ...]] = ("warehouse", "vehicle", "inventory", "order")


def _check_timestamp(value: str) -> str:
    """Accept ISO 8601 strings (``Z`` or an offset) and return them unchanged."""
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"timestamp must be ISO 8601, got {value!r}") from exc
    return value


class _Payload(BaseModel):
    """Shared behaviour of all simulator records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str
    is_anomaly: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_iso(cls, value: str) -> str:
        return _check_timestamp(value)

    @model_validator(mode="after")
    def _label_is_consistent(self) -> Self:
        """``is_anomaly`` is the ground truth and must agree with ``anomaly_type``."""
        anomaly_type = getattr(self, "anomaly_type", None)
        if self.is_anomaly != (anomaly_type is not None):
            raise ValueError(
                f"is_anomaly={self.is_anomaly} contradicts anomaly_type={anomaly_type!r}"
            )
        return self


class WarehouseReading(_Payload):
    """One temperature / humidity reading (Day 2, ``WarehouseSimulator``)."""

    sensor_id: str = Field(min_length=1)
    warehouse_id: str = Field(min_length=1)
    reading_type: Literal["temperature", "humidity"]
    value: float
    unit: str
    anomaly_type: Literal["point", "contextual", "collective"] | None = None


class GPSReading(_Payload):
    """One GPS fix of a vehicle (Day 3, ``GPSSimulator``)."""

    vehicle_id: str = Field(min_length=1)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    speed_kmh: float = Field(ge=0.0)
    heading: float = Field(ge=0.0, lt=360.0)
    route_adherence_score: float = Field(ge=0.0, le=1.0)
    anomaly_type: Literal["point", "collective"] | None = None


class InventoryEvent(_Payload):
    """One stock-level change of a SKU (Day 3, ``InventorySimulator``)."""

    sku_id: str = Field(min_length=1)
    warehouse_id: str = Field(min_length=1)
    previous_level: int = Field(ge=0)
    new_level: int = Field(ge=0)
    change_amount: int
    transaction_type: Literal["shipment", "restock", "return", "adjustment"]
    anomaly_type: Literal["point", "contextual"] | None = None

    @model_validator(mode="after")
    def _change_matches_levels(self) -> Self:
        if self.new_level - self.previous_level != self.change_amount:
            raise ValueError(
                f"change_amount {self.change_amount} != new_level {self.new_level} "
                f"- previous_level {self.previous_level}"
            )
        return self


class OrderEvent(_Payload):
    """One order-lifecycle step (Day 3, ``OrderEventSimulator``)."""

    order_id: str = Field(min_length=1)
    sku_id: str = Field(min_length=1)
    warehouse_id: str = Field(min_length=1)
    status: str
    previous_status: str | None = None
    quantity: int = Field(ge=1)
    stage_duration_s: float | None = Field(default=None, ge=0.0)
    expected_duration_s: float | None = Field(default=None, ge=0.0)
    anomaly_type: Literal["point"] | None = None


Payload = WarehouseReading | GPSReading | InventoryEvent | OrderEvent

#: Payload model and entity-id field for every source type.
PAYLOAD_MODELS: Final[dict[str, type[_Payload]]] = {
    "warehouse": WarehouseReading,
    "vehicle": GPSReading,
    "inventory": InventoryEvent,
    "order": OrderEvent,
}
ENTITY_ID_FIELDS: Final[dict[str, str]] = {
    "warehouse": "sensor_id",
    "vehicle": "vehicle_id",
    "inventory": "sku_id",
    "order": "order_id",
}


class KafkaMessage(BaseModel):
    """The envelope written to Kafka: routing metadata plus one typed payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_type: SourceType
    schema_version: str = SCHEMA_VERSION
    entity_id: str = Field(min_length=1)
    timestamp: str
    produced_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: Payload

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_iso(cls, value: str) -> str:
        return _check_timestamp(value)

    @field_validator("payload", mode="before")
    @classmethod
    def _parse_payload_by_source(cls, value: Any, info: ValidationInfo) -> Any:
        """Validate a dict payload with the model that matches ``source_type``."""
        source = info.data.get("source_type")
        if isinstance(value, dict) and source in PAYLOAD_MODELS:
            return PAYLOAD_MODELS[source].model_validate(value)
        return value

    @model_validator(mode="after")
    def _payload_matches_source(self) -> Self:
        expected = PAYLOAD_MODELS[self.source_type]
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"source_type {self.source_type!r} needs a {expected.__name__} payload, "
                f"got {type(self.payload).__name__}"
            )
        return self

    @field_serializer("produced_at")
    def _serialize_produced_at(self, value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------ #
    # Construction and serialisation
    # ------------------------------------------------------------------ #
    @classmethod
    def from_payload(cls, source_type: SourceType, record: dict[str, Any]) -> KafkaMessage:
        """Validate a raw simulator record and wrap it.

        Args:
            source_type: Which simulator produced ``record``.
            record: The simulator's dict, unchanged.

        Raises:
            pydantic.ValidationError: The record does not match its schema.
            KeyError: ``source_type`` is not one of :data:`SOURCE_TYPES`.
        """
        payload = cast(Payload, PAYLOAD_MODELS[source_type].model_validate(record))
        return cls(
            source_type=source_type,
            entity_id=str(record[ENTITY_ID_FIELDS[source_type]]),
            timestamp=payload.timestamp,
            payload=payload,
        )

    def to_json_bytes(self) -> bytes:
        """Serialise to compact UTF-8 JSON, the Kafka message value."""
        return self.model_dump_json().encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible dict form of the message."""
        return json.loads(self.model_dump_json())


class DLQErrorInfo(BaseModel):
    """Why a message ended up in the dead-letter queue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: Literal["validation_error", "message_too_large", "delivery_failed", "produce_failed"]
    error_type: str
    message: str
    error_code: str | None = None
    retriable: bool = False


class DLQMessage(BaseModel):
    """Dead-letter record: the original payload plus everything needed to replay it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dlq_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    original_topic: str
    original_key: str | None = None
    source_type: str | None = None
    attempts: int = Field(ge=0)
    error: DLQErrorInfo
    #: Parsed JSON when the original was JSON, otherwise the raw text.
    original_payload: Any
    payload_truncated: bool = False

    @field_serializer("failed_at")
    def _serialize_failed_at(self, value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def to_json_bytes(self) -> bytes:
        """Serialise to compact UTF-8 JSON, the DLQ message value."""
        return self.model_dump_json().encode("utf-8")
