"""Analyst feedback endpoints (Day 14 verdicts, Day 16 calibration, recalibration and drift)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import ServiceDep, require_admin
from src.api.schemas import (
    ErrorResponse,
    FeedbackGroupStats,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from src.api.service import EventNotFoundError
from src.models.feedback_loop import FeedbackValidationError

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Give feedback on an alert",
    responses={404: {"model": ErrorResponse, "description": "Unknown or evicted alert"}},
)
def submit_feedback(body: FeedbackRequest, service: ServiceDep) -> FeedbackResponse:
    """Record an analyst verdict and learn from it.

    * ``false_positive`` / ``true_positive``: the alert's ``false_positive_marked`` flag follows
      the latest verdict, the entity's decision threshold is recalibrated (a burst of false
      positives raises it, missed detections lower it) and the verdict feeds the periodic
      ensemble recalibration and the drift detector.
    * ``severity_adjustment``: records the severity the analyst thinks is right
      (``corrected_severity``).
    * ``missed_detection``: with ``true_positive``, reports a real anomaly the system did not alert
      on; pass ``entity_id`` when there is no stored alert to point at.

    A second verdict of the same kind from the same analyst on the same alert replaces the first
    (``replaced_previous`` is then true) and is not counted twice by the calibrators.
    """
    try:
        outcome = service.record_feedback(body)
    except EventNotFoundError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"anomaly {body.anomaly_event_id!r} not found"
        ) from error
    except FeedbackValidationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    event = outcome.event
    return FeedbackResponse(
        feedback_id=event.event_id,
        anomaly_event_id=event.anomaly_event_id,
        entity_id=event.entity_id,
        feedback_type=event.feedback_type,  # type: ignore[arg-type]
        analyst_id=event.analyst_id,
        notes=event.notes,
        timestamp=event.timestamp,
        replaced_previous=outcome.replaced,
        corrected_severity=event.corrected_severity,  # type: ignore[arg-type]
        missed_detection=event.missed,
        threshold_adjustment=outcome.threshold_adjustment,
        drift_alert_id=outcome.drift_alert_id,
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


@router.get("/analytics", summary="Feedback analytics, calibration state and drift")
def feedback_analytics(service: ServiceDep) -> dict[str, Any]:
    """Summarise the feedback loop.

    False-positive rates (overall, last hour, last 24 h), the noisiest entities, entities whose
    threshold was adjusted, severity corrections, the latest ensemble recalibration (weights
    before / after, estimated false-positive reduction) and the drift detector's state.
    """
    return service.feedback_processor.analytics()


@router.get("/timeline", summary="Daily false-positive rate and feedback volume")
def feedback_timeline(
    service: ServiceDep, days: Annotated[int, Query(ge=1, le=90)] = 7
) -> dict[str, Any]:
    """Per-day false-positive rates, the day x hour feedback volume and per-entity tallies.

    Days are UTC calendar days ending today; a day without verdicts has a ``null`` rate. Feeds the
    dashboard's feedback analytics.
    """
    return service.feedback_processor.timeline(days)


@router.post(
    "/recalibrate",
    summary="Run the ensemble recalibration now",
    dependencies=[Depends(require_admin)],
)
def recalibrate_now(service: ServiceDep) -> dict[str, Any]:
    """Recalibrate the meta-learner and LSTM temperature from the unused verdicts right now.

    Normally this runs every 30 minutes or after 50 new verdicts. The minimum-data checks still
    apply: with too few labelled verdicts the report says ``skipped`` and why.
    """
    return service.recalibrator.run_once(force=True).to_dict()
