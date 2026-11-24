"""Prediction and alert-query endpoints (Day 14)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.dependencies import ConfigDep, ServiceDep
from src.api.schemas import (
    AnomalyDetectionResponse,
    AnomalyEventModel,
    AnomalyListResponse,
    AnomalyTypeName,
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    SensorEventRequest,
    SeverityName,
)
from src.models.ensemble_detector import Severity
from src.utils.logger import get_logger

router = APIRouter(prefix="/api/v1", tags=["detection"])
_log = get_logger(__name__)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _count(request: Request, outcome: str, n: int = 1) -> None:
    metrics = getattr(request.app.state, "api_metrics", None)
    if metrics is not None:
        metrics.predictions.labels(outcome).inc(n)


@router.post(
    "/predict",
    response_model=AnomalyDetectionResponse,
    summary="Score one event",
    responses={429: {"model": ErrorResponse, "description": "Rate limit exceeded"}},
)
def predict(
    body: SensorEventRequest, service: ServiceDep, request: Request
) -> AnomalyDetectionResponse:
    """Score a single warehouse / vehicle / inventory event.

    The event updates the entity's rolling windows, so scoring is stateful: send an entity's
    events in time order. Anomalies are also stored and returned by ``GET /anomalies``.
    """
    result = service.predict(body)
    _count(request, "anomaly" if result.is_anomaly else "normal")
    return result


@router.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
    summary="Score up to 100 events",
    responses={
        413: {"model": ErrorResponse, "description": "Batch larger than the configured maximum"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
def batch_predict(
    body: BatchPredictRequest, service: ServiceDep, config: ConfigDep, request: Request
) -> BatchPredictResponse:
    """Score events in the order given (results keep that order)."""
    if len(body.events) > config.api.batch_max_size:
        raise HTTPException(
            413,
            f"batch of {len(body.events)} exceeds the maximum of {config.api.batch_max_size}",
        )
    results = [service.predict(event) for event in body.events]
    flagged = sum(r.is_anomaly for r in results)
    _count(request, "anomaly", flagged)
    _count(request, "normal", len(results) - flagged)
    return BatchPredictResponse(count=len(results), anomalies=flagged, results=results)


@router.get("/anomalies", response_model=AnomalyListResponse, summary="List recent alerts")
def list_anomalies(
    service: ServiceDep,
    entity_id: Annotated[str | None, Query(description="Exact entity id")] = None,
    severity: Annotated[SeverityName | None, Query()] = None,
    anomaly_type: Annotated[AnomalyTypeName | None, Query()] = None,
    start_time: Annotated[datetime | None, Query(description="ISO 8601, inclusive")] = None,
    end_time: Annotated[datetime | None, Query(description="ISO 8601, inclusive")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AnomalyListResponse:
    """Alerts from the in-memory store, newest first, with optional filters."""
    if start_time and end_time and _aware(end_time) < _aware(start_time):
        raise HTTPException(422, "end_time is before start_time")
    matches = service.anomaly_store.filter(
        entity_id=entity_id,
        severity=Severity[severity] if severity else None,
        anomaly_type=anomaly_type,
        start_time=start_time,
        end_time=end_time,
    )
    page = matches[offset : offset + limit]
    return AnomalyListResponse(
        total=len(matches),
        limit=limit,
        offset=offset,
        items=[AnomalyEventModel.model_validate(e.to_dict()) for e in page],
    )


@router.get(
    "/anomalies/{event_id}",
    response_model=AnomalyEventModel,
    summary="Get one alert",
    responses={404: {"model": ErrorResponse, "description": "Unknown or evicted alert"}},
)
def get_anomaly(event_id: str, service: ServiceDep) -> AnomalyEventModel:
    """Return the full alert record, including top contributing features when available."""
    event = service.anomaly_store.get(event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"anomaly {event_id!r} not found")
    return AnomalyEventModel.model_validate(event.to_dict())
