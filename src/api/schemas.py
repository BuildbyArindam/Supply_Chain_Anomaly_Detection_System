"""Pydantic v2 request / response models of the inference API (Day 14).

Requests reuse the Day 4 payload models (:mod:`src.kafka.schemas`) so the API accepts exactly the
records the simulators and Kafka topics carry. Only two things differ for API clients:

* ``is_anomaly`` (the simulator's *ground-truth* label) may be omitted. Detectors never read it,
  it is only kept so labelled data can be replayed through the API.
* ``entity_id`` is derived from the payload (``sensor_id`` / ``vehicle_id`` / ``sku_id``).

Every model carries example values that show up in the OpenAPI docs (``/docs``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.kafka.schemas import ENTITY_ID_FIELDS, PAYLOAD_MODELS, KafkaMessage, Payload

#: Source types that can be scored (order events carry no detectable signal).
ScoredSource = Literal["warehouse", "vehicle", "inventory"]
SeverityName = Literal["INFO", "WARNING", "CRITICAL", "EMERGENCY"]
AnomalyTypeName = Literal["point", "contextual", "collective"]
FeedbackType = Literal["false_positive", "true_positive", "severity_adjustment"]

MAX_BATCH_SIZE: Final = 100

_EXAMPLE_WAREHOUSE: Final[dict[str, Any]] = {
    "source_type": "warehouse",
    "payload": {
        "sensor_id": "wh1-A-temp-01",
        "warehouse_id": "wh1",
        "timestamp": "2026-11-24T08:00:00Z",
        "reading_type": "temperature",
        "value": 11.7,
        "unit": "C",
        "metadata": {"zone": "A", "storage_type": "cold_storage"},
    },
}
_EXAMPLE_VEHICLE: Final[dict[str, Any]] = {
    "source_type": "vehicle",
    "payload": {
        "vehicle_id": "veh-007",
        "timestamp": "2026-11-24T08:00:10Z",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "speed_kmh": 142.0,
        "heading": 87.5,
        "route_adherence_score": 0.31,
    },
}


class _ApiModel(BaseModel):
    """Base class: strict about unknown fields so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
class SensorEventRequest(_ApiModel):
    """One raw event (warehouse reading, GPS fix or inventory change) to score."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [_EXAMPLE_WAREHOUSE, _EXAMPLE_VEHICLE]},
    )

    source_type: ScoredSource = Field(description="Which simulator schema ``payload`` follows.")
    payload: dict[str, Any] = Field(
        description="The Day 2/3 record. ``is_anomaly`` is optional (ground truth, never used)."
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        """Normalise ``payload`` with the matching Kafka payload model (defaults, types)."""
        data = dict(self.payload)
        data.setdefault("is_anomaly", data.get("anomaly_type") is not None)
        data.setdefault("anomaly_type", None)
        try:
            model = PAYLOAD_MODELS[self.source_type].model_validate(data)
        except ValueError as exc:
            # Re-raise as ValueError so FastAPI reports a clean 422 with the field errors.
            raise ValueError(f"invalid {self.source_type} payload: {exc}") from exc
        self.payload = model.model_dump(mode="json")
        return self

    @property
    def entity_id(self) -> str:
        """Sensor, vehicle or SKU id of the event."""
        return str(self.payload[ENTITY_ID_FIELDS[self.source_type]])

    @property
    def timestamp(self) -> str:
        """Event time as given by the client (ISO 8601)."""
        return str(self.payload["timestamp"])

    def to_raw_event(self) -> dict[str, Any]:
        """Return the simulator-style dict the detectors consume."""
        return dict(self.payload)

    def to_kafka_message(self) -> KafkaMessage:
        """Wrap the event in the Day 4 envelope (what the stream processor ingests)."""
        return KafkaMessage(
            source_type=self.source_type,
            entity_id=self.entity_id,
            timestamp=self.timestamp,
            payload=cast(Payload, PAYLOAD_MODELS[self.source_type].model_validate(self.payload)),
        )


class AnomalyDetectionResponse(_ApiModel):
    """The verdict for one event."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "entity_id": "wh1-A-temp-01",
                    "source_type": "warehouse",
                    "timestamp": "2026-11-24T08:00:00Z",
                    "is_anomaly": True,
                    "anomaly_score": 0.93,
                    "confidence": 0.86,
                    "anomaly_type": "point",
                    "severity": "EMERGENCY",
                    "severity_score": 0.97,
                    "event_id": "5f0c3a9e-2d57-4b8d-9d1e-0a8e0e7c5d11",
                    "alert_suppressed": False,
                    "method": "weighted_average",
                    "models_used": ["statistical"],
                    "individual_scores": {"statistical": 0.93},
                    "recommended_action": "IMMEDIATE ACTION: inspect the cold-storage unit.",
                    "warming_up": False,
                    "processing_ms": 3.2,
                }
            ]
        },
    )

    entity_id: str
    source_type: ScoredSource
    timestamp: str
    is_anomaly: bool
    anomaly_score: float = Field(ge=0, le=1, description="Ensemble score; 0.5 is the boundary.")
    confidence: float = Field(ge=0, le=1)
    anomaly_type: AnomalyTypeName | None = None
    severity: SeverityName | None = Field(
        default=None, description="Only set when ``is_anomaly`` is true."
    )
    severity_score: float | None = Field(default=None, ge=0, le=1)
    event_id: str | None = Field(
        default=None, description="Id of the stored alert (see GET /api/v1/anomalies/{event_id})."
    )
    alert_suppressed: bool = Field(
        default=False, description="True when an alert for this entity is already open (dedup)."
    )
    method: str = Field(description="``meta_learner``, ``weighted_average`` or ``none``.")
    models_used: list[str] = Field(default_factory=list)
    individual_scores: dict[str, float] = Field(default_factory=dict)
    recommended_action: str | None = None
    warming_up: bool = Field(
        default=False, description="No detector had enough history yet (cold start)."
    )
    ingest_status: str = Field(
        default="emitted", description="What the stream aggregator did with the event."
    )
    processing_ms: float = Field(ge=0)


class BatchPredictRequest(_ApiModel):
    """Up to 100 events, scored in the order given."""

    events: list[SensorEventRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"events": [_EXAMPLE_WAREHOUSE, _EXAMPLE_VEHICLE]}]},
    )


class BatchPredictResponse(_ApiModel):
    """One response per request event, in request order."""

    count: int
    anomalies: int
    results: list[AnomalyDetectionResponse]


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
class AnomalyEventModel(_ApiModel):
    """The canonical alert record (see ``AnomalyEvent.to_dict``)."""

    event_id: str
    entity_id: str
    entity_type: str
    timestamp: datetime
    anomaly_type: AnomalyTypeName | None
    severity: SeverityName
    raw_value: float | None
    expected_range: tuple[float, float] | None
    top_contributing_features: list[dict[str, Any]]
    recommended_action: str
    is_resolved: bool
    false_positive_marked: bool
    severity_score: float
    ensemble_score: float
    confidence: float
    business_impact: float
    individual_scores: dict[str, float]
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    escalated_from: SeverityName | None = None


class AnomalyListResponse(_ApiModel):
    """A page of alerts, newest first."""

    total: int = Field(description="Matches before ``limit`` / ``offset``.")
    limit: int
    offset: int
    items: list[AnomalyEventModel]


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(_ApiModel):
    """An analyst's verdict on one alert."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "anomaly_event_id": "5f0c3a9e-2d57-4b8d-9d1e-0a8e0e7c5d11",
                    "feedback_type": "false_positive",
                    "analyst_id": "analyst-42",
                    "notes": "Door was open during stock-take.",
                }
            ]
        },
    )

    anomaly_event_id: str = Field(min_length=1, max_length=64)
    feedback_type: FeedbackType
    analyst_id: str = Field(default="anonymous", min_length=1, max_length=64)
    notes: str | None = Field(default=None, max_length=1000)
    corrected_severity: SeverityName | None = Field(
        default=None, description="The right severity; required for ``severity_adjustment``."
    )
    missed_detection: bool = Field(
        default=False,
        description=(
            "``true_positive`` only: the system did NOT alert on a real anomaly. Needs "
            "``entity_id`` when ``anomaly_event_id`` is not a stored alert."
        ),
    )
    entity_id: str | None = Field(
        default=None, min_length=1, max_length=128, description="Only for a missed detection."
    )

    @field_validator("notes")
    @classmethod
    def _strip_notes(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @model_validator(mode="after")
    def _check_combination(self) -> Self:
        if self.feedback_type == "severity_adjustment":
            if self.corrected_severity is None:
                raise ValueError("corrected_severity is required for a severity_adjustment")
        elif self.corrected_severity is not None:
            raise ValueError("corrected_severity only applies to a severity_adjustment")
        if self.missed_detection and self.feedback_type != "true_positive":
            raise ValueError("missed_detection only applies to a true_positive")
        return self


class FeedbackResponse(_ApiModel):
    """The stored verdict."""

    feedback_id: str
    anomaly_event_id: str
    entity_id: str
    feedback_type: FeedbackType
    analyst_id: str
    notes: str | None
    timestamp: datetime
    replaced_previous: bool = Field(
        description="True when this analyst had already rated the alert (verdict replaced)."
    )
    corrected_severity: SeverityName | None = None
    missed_detection: bool = False
    threshold_adjustment: float | None = Field(
        default=None,
        description="The entity's decision-threshold adjustment after this verdict (labels only).",
    )
    drift_alert_id: str | None = Field(
        default=None, description="Set when this verdict made the drift detector fire."
    )


class FeedbackGroupStats(_ApiModel):
    """True / false positive tallies plus the derived false-positive rate."""

    true_positive: int = 0
    false_positive: int = 0
    total: int = 0
    false_positive_rate: float = 0.0


class FeedbackStatsResponse(_ApiModel):
    """Feedback statistics overall, per model that flagged the alert, and per entity."""

    overall: FeedbackGroupStats
    per_model: dict[str, FeedbackGroupStats]
    per_entity: dict[str, FeedbackGroupStats]
    per_severity: dict[str, FeedbackGroupStats]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ModelVersionInfo(_ApiModel):
    """One registry version of a model type."""

    version: str
    created_at: datetime
    metrics: dict[str, Any]
    hyperparams: dict[str, Any]
    active: bool = Field(description="Currently serving traffic.")


class ModelTypeInfo(_ApiModel):
    """A model type and its versions, newest first."""

    model_type: str
    versions: list[ModelVersionInfo]


class ModelListResponse(_ApiModel):
    """All models in the registry."""

    registry_path: str
    models: list[ModelTypeInfo]


class EnsembleInfo(_ApiModel):
    """How one source type is scored."""

    members: list[str]
    method: Literal["meta_learner", "weighted_average"]


class CurrentModelsResponse(_ApiModel):
    """The active model configuration."""

    generation: int = Field(description="Incremented by every successful (re)load.")
    loaded_at: datetime
    degraded: bool = Field(description="Only the statistical fallback is serving.")
    versions: dict[str, str] = Field(description="Registry version of each loaded model type.")
    ensembles: dict[str, EnsembleInfo]
    load_errors: dict[str, str] = Field(description="Models that exist but failed to load.")


class ReloadResponse(_ApiModel):
    """Result of ``POST /api/v1/models/reload``."""

    reloaded: bool
    previous_generation: int
    current: CurrentModelsResponse


# --------------------------------------------------------------------------- #
# Health / errors
# --------------------------------------------------------------------------- #
class HealthResponse(_ApiModel):
    """Liveness and readiness summary."""

    status: Literal["ok", "degraded"]
    uptime_s: float
    version: str
    models_loaded: bool
    degraded_reason: str | None = None
    checks: dict[str, str] = Field(default_factory=dict)
    stored_anomalies: int = 0


class ErrorResponse(_ApiModel):
    """Error body of non-validation failures."""

    detail: str
    request_id: str | None = None
