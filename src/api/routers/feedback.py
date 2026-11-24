"""Analyst feedback endpoints (Day 14; the recalibration logic arrives on Day 16)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.dependencies import ServiceDep
from src.api.schemas import (
    ErrorResponse,
    FeedbackGroupStats,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from src.api.service import EventNotFoundError

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Mark an alert as a false or true positive",
    responses={404: {"model": ErrorResponse, "description": "Unknown or evicted alert"}},
)
def submit_feedback(body: FeedbackRequest, service: ServiceDep) -> FeedbackResponse:
    """Record an analyst verdict.

    A second verdict from the same analyst on the same alert replaces the first
    (``replaced_previous`` is then true). The alert's ``false_positive_marked`` flag follows
    the latest verdict.
    """
    try:
        record, replaced = service.record_feedback(body)
    except EventNotFoundError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"anomaly {body.anomaly_event_id!r} not found"
        ) from error
    return FeedbackResponse(
        feedback_id=record.feedback_id,
        anomaly_event_id=record.anomaly_event_id,
        entity_id=record.entity_id,
        feedback_type=record.feedback_type,  # type: ignore[arg-type]
        analyst_id=record.analyst_id,
        notes=record.notes,
        timestamp=record.timestamp,
        replaced_previous=replaced,
    )


@router.get("/stats", response_model=FeedbackStatsResponse, summary="Feedback statistics")
def feedback_stats(service: ServiceDep) -> FeedbackStatsResponse:
    """Return true / false positive tallies overall, per flagging detector and per entity."""
    raw = service.feedback_store.stats()
    return FeedbackStatsResponse(
        overall=FeedbackGroupStats(**raw["overall"]),
        per_model={k: FeedbackGroupStats(**v) for k, v in raw["per_model"].items()},
        per_entity={k: FeedbackGroupStats(**v) for k, v in raw["per_entity"].items()},
        per_severity={k: FeedbackGroupStats(**v) for k, v in raw["per_severity"].items()},
    )
