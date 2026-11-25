"""Feedback loop through the service, the HTTP API and Kafka (Day 16)."""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from prometheus_client import CollectorRegistry

from src.api.main import create_app
from src.api.schemas import FeedbackRequest
from src.api.service import EventNotFoundError, InferenceService
from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.models.ensemble_detector import EnsembleAnomalyScore
from src.models.statistical_detector import AnomalyScore
from src.utils.config import AppConfig
from tests.mock_kafka import MockBroker

pytestmark = pytest.mark.anyio

T0 = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)
SENSOR = "wh1-A-temp-01"
TOPICS = {
    "warehouse-sensors": 6,
    "vehicle-gps": 4,
    "inventory-events": 4,
    "anomaly-alerts": 3,
    "dlq-events": 1,
    "model-health": 1,
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    api_cfg = cfg.api.model_copy(
        update={"predict_rate_limit_per_min": 100_000, "batch_predict_rate_limit_per_min": 100_000}
    )
    paths = cfg.paths.model_copy(
        update={
            "model_registry": str(tmp_path / "registry"),
            "feedback_store": str(tmp_path / "feedback_store.json"),
            "anomaly_snapshots": str(tmp_path / "snapshots"),
        }
    )
    return cfg.model_copy(update={"api": api_cfg, "paths": paths})


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


async def make_client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, client=("10.0.0.1", 5000))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


@pytest.fixture
def make_app(cfg: AppConfig) -> Callable[..., Any]:
    def factory(**kwargs: Any) -> Any:
        return create_app(cfg, metrics_registry=CollectorRegistry(), **kwargs)

    return factory


@pytest.fixture
async def client(make_app: Callable[..., Any]) -> AsyncIterator[httpx.AsyncClient]:
    async for c in make_client(make_app()):
        yield c


async def alert_id(client: httpx.AsyncClient) -> str:
    """Warm the detector up, then spike it; return the stored alert's id."""
    rng = random.Random(7)
    for i in range(80):
        r = await client.post("/api/v1/predict", json=warehouse(i, 3.0 + rng.gauss(0, 0.15)))
        assert r.status_code == 200, r.text
    r = await client.post("/api/v1/predict", json=warehouse(80, 25.0))
    body = r.json()
    assert body["event_id"], body
    return str(body["event_id"])


def verdict(event_id: str, kind: str = "false_positive", analyst: str = "a1", **extra: Any):
    return {"anomaly_event_id": event_id, "feedback_type": kind, "analyst_id": analyst, **extra}


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "body",
    [
        {"feedback_type": "severity_adjustment"},  # needs corrected_severity
        {"feedback_type": "severity_adjustment", "corrected_severity": "SEVERE"},
        {"feedback_type": "false_positive", "corrected_severity": "INFO"},
        {"feedback_type": "false_positive", "missed_detection": True},
        {"feedback_type": "true_positive", "entity_id": ""},
        {"feedback_type": "maybe"},
    ],
)
async def test_invalid_feedback_requests_are_422(client, body):
    r = await client.post("/api/v1/feedback", json={"anomaly_event_id": "x", **body})
    assert r.status_code == 422


async def test_unknown_alert_is_404_unless_it_is_a_missed_detection_with_an_entity(client):
    assert (await client.post("/api/v1/feedback", json=verdict("ghost"))).status_code == 404
    no_entity = verdict("ghost", "true_positive", missed_detection=True)
    assert (await client.post("/api/v1/feedback", json=no_entity)).status_code == 404
    ok = verdict("ghost", "true_positive", missed_detection=True, entity_id=SENSOR)
    r = await client.post("/api/v1/feedback", json=ok)
    assert r.status_code == 201
    assert r.json()["entity_id"] == SENSOR and r.json()["missed_detection"] is True


# --------------------------------------------------------------------------- #
# Verdicts reach the loop
# --------------------------------------------------------------------------- #
async def test_false_positive_is_stored_with_the_alerts_meta_features(client, make_app):
    app = client._transport.app  # the app under test
    event_id = await alert_id(client)
    r = await client.post("/api/v1/feedback", json=verdict(event_id))
    assert r.status_code == 201
    body = r.json()
    assert body["threshold_adjustment"] == 0.0 and body["drift_alert_id"] is None
    assert body["replaced_previous"] is False

    stored = app.state.service.feedback_events.all()
    assert len(stored) == 1 and stored[0].entity_id == SENSOR
    assert stored[0].original_severity in {"INFO", "WARNING", "CRITICAL", "EMERGENCY"}
    features = stored[0].meta_features
    assert features is not None and len(features) == 10
    assert features[0] is not None and features[0] > 0.5  # the statistical detector's score
    assert features[1:4] == (None, None, None)  # no trained members in this test setup


async def test_burst_of_false_positives_raises_the_entity_threshold(client):
    event_id = await alert_id(client)
    adjustments = []
    for n in range(6):
        r = await client.post("/api/v1/feedback", json=verdict(event_id, analyst=f"analyst-{n}"))
        assert r.status_code == 201, r.text
        adjustments.append(r.json()["threshold_adjustment"])
    assert adjustments[:5] == [0.0] * 5
    assert adjustments[5] == pytest.approx(0.1)
    analytics = (await client.get("/api/v1/feedback/analytics")).json()
    assert analytics["adjusted_entities"][SENSOR]["current_threshold_adjustment"] == 0.1
    assert analytics["overall"]["false_positive"] == 6


async def test_missed_detections_lower_the_entity_threshold(client):
    adjustments = []
    for n in range(4):
        r = await client.post(
            "/api/v1/feedback",
            json=verdict(f"missed-{n}", "true_positive", missed_detection=True, entity_id=SENSOR),
        )
        adjustments.append(r.json()["threshold_adjustment"])
    assert adjustments == [0.0, 0.0, 0.0, pytest.approx(-0.05)]


async def test_changed_verdict_replaces_without_double_counting(client):
    event_id = await alert_id(client)
    first = (await client.post("/api/v1/feedback", json=verdict(event_id))).json()
    second = (await client.post("/api/v1/feedback", json=verdict(event_id, "true_positive"))).json()
    assert first["replaced_previous"] is False and second["replaced_previous"] is True
    analytics = (await client.get("/api/v1/feedback/analytics")).json()
    assert analytics["overall"]["total"] == 1 and analytics["overall"]["true_positive"] == 1
    alert = (await client.get(f"/api/v1/anomalies/{event_id}")).json()
    assert alert["false_positive_marked"] is False  # the flag follows the latest verdict


async def test_severity_adjustment_is_recorded_and_does_not_touch_the_labels(client):
    event_id = await alert_id(client)
    await client.post("/api/v1/feedback", json=verdict(event_id, "false_positive"))
    r = await client.post(
        "/api/v1/feedback",
        json=verdict(event_id, "severity_adjustment", corrected_severity="INFO"),
    )
    assert r.status_code == 201
    assert r.json()["corrected_severity"] == "INFO" and r.json()["threshold_adjustment"] is None
    analytics = (await client.get("/api/v1/feedback/analytics")).json()
    assert analytics["severity_adjustments"]["total"] == 1
    assert analytics["overall"]["total"] == 1  # still just the one false positive
    stats = (await client.get("/api/v1/feedback/stats")).json()  # Day 14 endpoint unaffected
    assert stats["overall"]["total"] == 1 and stats["overall"]["false_positive"] == 1


async def test_analytics_endpoint_reports_calibration_and_drift_state(client):
    analytics = (await client.get("/api/v1/feedback/analytics")).json()
    assert analytics["overall"]["total"] == 0
    assert analytics["drift"] == {
        "statistic": 0.0,
        "threshold": 8.0,
        "alerts": 0,
        "last_alert": None,
    }
    assert analytics["last_recalibration"] is None


async def test_recalibrate_endpoint_reports_why_it_skipped(client):
    r = await client.post("/api/v1/feedback/recalibrate")
    assert r.status_code == 200
    assert r.json()["status"] == "skipped" and "need 20" in r.json()["reason"]
    analytics = (await client.get("/api/v1/feedback/analytics")).json()
    assert analytics["last_recalibration"]["status"] == "skipped"


async def test_recalibrate_endpoint_requires_the_admin_key_when_one_is_set(client, monkeypatch):
    monkeypatch.setenv("API_ADMIN_KEY", "s3cret")
    assert (await client.post("/api/v1/feedback/recalibrate")).status_code == 401
    ok = await client.post("/api/v1/feedback/recalibrate", headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


# --------------------------------------------------------------------------- #
# Threshold calibration changes the decision
# --------------------------------------------------------------------------- #
def ensemble_score(score: float, *, is_anomaly: bool) -> EnsembleAnomalyScore:
    member = AnomalyScore(
        entity_id=SENSOR,
        timestamp=T0,
        score=score,
        is_anomaly=is_anomaly,
        anomaly_type="contextual",
        confidence=0.5,
        detector="fake",
    )
    return EnsembleAnomalyScore(
        entity_id=SENSOR,
        timestamp=T0,
        ensemble_score=score,
        individual_scores={"statistical": score},
        predicted_anomaly_type="point" if is_anomaly else None,
        confidence=abs(2 * score - 1),
        is_anomaly=is_anomaly,
        members={"statistical": member},
        details={},
    )


def test_a_raised_threshold_suppresses_marginal_alerts_only(cfg):
    service = InferenceService(cfg)
    for i in range(6):
        service.calibrator.update(SENSOR, True, timestamp=T0 + timedelta(seconds=i))
    marginal = ensemble_score(0.55, is_anomaly=True)
    service._apply_calibration(SENSOR, marginal)
    assert marginal.is_anomaly is False and marginal.predicted_anomaly_type is None
    assert marginal.details["suppressed_by_calibration"] is True
    assert marginal.details["calibrated_threshold"] == pytest.approx(0.6)
    assert marginal.ensemble_score == 0.55  # the score itself is untouched

    clear = ensemble_score(0.9, is_anomaly=True)
    service._apply_calibration(SENSOR, clear)
    assert clear.is_anomaly is True

    other = ensemble_score(0.55, is_anomaly=True)  # another entity keeps the 0.5 boundary
    service._apply_calibration("someone-else", other)
    assert other.is_anomaly is True and "calibrated_threshold" not in other.details


def test_a_lowered_threshold_lets_marginal_events_through(cfg):
    service = InferenceService(cfg)
    for _ in range(4):
        service.calibrator.update(SENSOR, False, missed=True)
    near_miss = ensemble_score(0.47, is_anomaly=False)
    service._apply_calibration(SENSOR, near_miss)
    assert near_miss.is_anomaly is True and near_miss.details["raised_by_calibration"] is True
    assert near_miss.predicted_anomaly_type == "contextual"  # borrowed from a member's vote
    below = ensemble_score(0.40, is_anomaly=False)
    service._apply_calibration(SENSOR, below)
    assert below.is_anomaly is False


async def test_calibration_applies_to_live_predictions(make_app):
    app = make_app()
    async for c in make_client(app):
        service: InferenceService = app.state.service
        rng = random.Random(7)
        for i in range(80):
            await c.post("/api/v1/predict", json=warehouse(i, 3.0 + rng.gauss(0, 0.15)))
        # ensemble scores of a mild and a gross deviation under the default 0.5 boundary
        mild = (await c.post("/api/v1/predict", json=warehouse(80, 3.7))).json()
        assert mild["is_anomaly"] is True and 0.5 <= mild["anomaly_score"] < 0.95
        for i in range(60):  # rejected over and over: the threshold climbs to its cap
            service.calibrator.update(SENSOR, True, timestamp=T0 + timedelta(seconds=i))
        cap = service.calibrator.get_adjusted_threshold(SENSOR, 0.5)
        assert cap == pytest.approx(0.8)
        again = (await c.post("/api/v1/predict", json=warehouse(81, 3.7))).json()
        assert again["is_anomaly"] == (again["anomaly_score"] >= cap)
        gross = (await c.post("/api/v1/predict", json=warehouse(82, 40.0))).json()
        assert gross["anomaly_score"] >= cap and gross["is_anomaly"] is True  # real ones pass


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
async def test_verdicts_and_threshold_adjustments_survive_a_restart(cfg, make_app):
    async for c in make_client(make_app()):
        event_id = await alert_id(c)
        for n in range(6):
            await c.post("/api/v1/feedback", json=verdict(event_id, analyst=f"analyst-{n}"))
    stored = json.loads(Path(cfg.paths.feedback_store).read_text())
    assert stored["version"] == 2 and len(stored["events"]) == 6
    assert stored["events"][0]["meta_features"] is not None

    app = make_app()
    async for c in make_client(app):
        service: InferenceService = app.state.service
        assert service.calibrator.get_adjustment(SENSOR) == pytest.approx(0.1)
        assert len(service.feedback_events) == 6
        stats = (await c.get("/api/v1/feedback/stats")).json()  # Day 14 stats rebuilt too
        assert stats["overall"]["false_positive"] == 6
        assert stats["per_entity"][SENSOR]["total"] == 6


async def test_a_day_14_feedback_file_is_still_readable(cfg, make_app):
    Path(cfg.paths.feedback_store).write_text(
        json.dumps(
            {
                "version": 1,
                "feedback": [
                    {
                        "feedback_id": "f-1",
                        "anomaly_event_id": "old-alert",
                        "entity_id": SENSOR,
                        "feedback_type": "false_positive",
                        "analyst_id": "a1",
                        "severity": "WARNING",
                        "flagged_by": ["statistical"],
                        "notes": None,
                        "timestamp": "2026-11-20T08:00:00+00:00",
                    }
                ],
            }
        )
    )
    async for c in make_client(make_app()):
        stats = (await c.get("/api/v1/feedback/stats")).json()
        assert stats["overall"]["false_positive"] == 1
        assert (await c.get("/api/v1/feedback/analytics")).json()["overall"]["total"] == 1


def test_record_feedback_raises_for_an_unknown_alert(cfg):
    service = InferenceService(cfg)
    with pytest.raises(EventNotFoundError):
        service.record_feedback(
            FeedbackRequest(anomaly_event_id="ghost", feedback_type="false_positive")
        )


# --------------------------------------------------------------------------- #
# Kafka: the model-health topic
# --------------------------------------------------------------------------- #
def make_producer(broker: MockBroker, cfg: AppConfig) -> KafkaProducerManager:
    return KafkaProducerManager.from_config(
        cfg.kafka,
        producer_factory=broker.producer_factory,
        background_polling=False,
        metrics=ProducerMetrics(CollectorRegistry()),
    )


def test_send_model_health_publishes_json_to_the_model_health_topic():
    broker = MockBroker(TOPICS)
    producer = make_producer(broker, AppConfig())
    assert producer.send_model_health({"event_type": "ModelDriftAlert", "alert_id": "d-1"}) is True
    producer.flush()
    [record] = broker.messages("model-health")
    assert record.key == b"ModelDriftAlert"
    assert json.loads(record.value) == {"event_type": "ModelDriftAlert", "alert_id": "d-1"}
    assert broker.messages("anomaly-alerts") == []


def test_send_model_health_dead_letters_unusable_events():
    broker = MockBroker(TOPICS)
    producer = make_producer(broker, AppConfig())
    assert producer.send_model_health({"event_type": "x", "blob": object()}) is False
    assert producer.send_model_health({"event_type": "x", "blob": "y" * 2_000_000}) is False
    producer.flush()
    assert broker.messages("model-health") == []
    reasons = sorted(json.loads(r.value)["error"]["reason"] for r in broker.messages("dlq-events"))
    assert reasons == ["message_too_large", "validation_error"]


def test_drift_in_the_feedback_reaches_the_model_health_topic(cfg):
    broker = MockBroker(TOPICS)
    producer = make_producer(broker, cfg)
    service = InferenceService(cfg)
    service.model_health_publisher = producer.send_model_health

    # a calm stretch, then analysts reject nearly everything: the false-positive rate jumps
    alert_ids = []
    for n in range(120):
        is_fp = n >= 40
        event = service.feedback_processor.process(
            _event(n, "false_positive" if is_fp else "true_positive")
        )
        if event.drift_alert:
            alert_ids.append(event.drift_alert.alert_id)
    producer.flush()

    messages = [json.loads(r.value) for r in broker.messages("model-health")]
    assert alert_ids and [m["alert_id"] for m in messages] == alert_ids
    assert messages[0]["event_type"] == "ModelDriftAlert"
    assert messages[0]["recommended_action"] == "full_retrain"
    assert messages[0]["recent_fp_rate"] > messages[0]["baseline_fp_rate"]


def _event(n: int, kind: str):
    from src.models.feedback_loop import FeedbackEvent

    return FeedbackEvent.create(
        f"alert-{n}",
        f"analyst-{n}",
        kind,
        entity_id=f"e{n % 9}",
        timestamp=T0 + timedelta(seconds=n),
    )


async def test_create_app_hands_the_publisher_to_the_service(make_app):
    sent: list[dict[str, Any]] = []
    app = make_app(model_health_publisher=sent.append)
    async for _ in make_client(app):
        service: InferenceService = app.state.service
        service._publish_model_health({"event_type": "ModelDriftAlert"})
    assert sent == [{"event_type": "ModelDriftAlert"}]
