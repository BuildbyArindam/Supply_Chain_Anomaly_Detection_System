"""Model listing, hot reload and active-configuration endpoints (Day 14).

Day 21 adds the retraining view the dashboard's model-health page reads
(``GET /api/v1/models/retraining``) and the two admin actions on a challenger that is waiting for
a human decision (``POST /api/v1/models/promote`` and ``/reject``).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import ServiceDep, require_admin
from src.api.schemas import (
    CurrentModelsResponse,
    EnsembleInfo,
    ErrorResponse,
    ModelListResponse,
    ModelTypeInfo,
    ModelVersionInfo,
    ReloadResponse,
)
from src.api.service import ActiveModels, InferenceService, ReloadError
from src.utils.logger import get_logger

router = APIRouter(prefix="/api/v1/models", tags=["models"])
_log = get_logger(__name__)


def _current(active: ActiveModels) -> CurrentModelsResponse:
    return CurrentModelsResponse(
        generation=active.generation,
        loaded_at=active.loaded_at,
        degraded=active.degraded,
        versions=dict(active.versions),
        ensembles={
            source: EnsembleInfo(
                members=ensemble.available_members,
                method="meta_learner" if ensemble.uses_meta_learner else "weighted_average",
            )
            for source, ensemble in active.ensembles.items()
        },
        load_errors=dict(active.load_errors),
    )


@router.get("", response_model=ModelListResponse, summary="List registry models and versions")
def list_models(service: ServiceDep) -> ModelListResponse:
    """Every model type in the registry with its versions (newest first) and saved metrics."""
    active_versions = dict(service.models.active.versions)
    models = []
    for model_type in service.registry.list_model_types():
        versions = [
            ModelVersionInfo(
                version=v.version,
                created_at=v.created_at,
                metrics=dict(v.metrics),
                hyperparams=dict(v.hyperparams),
                active=active_versions.get(model_type) == v.version,
            )
            for v in service.registry.list_versions(model_type)
        ]
        models.append(ModelTypeInfo(model_type=model_type, versions=versions))
    return ModelListResponse(registry_path=str(service.registry.root), models=models)


@router.get("/current", response_model=CurrentModelsResponse, summary="Active configuration")
def current_models(service: ServiceDep) -> CurrentModelsResponse:
    """Which models serve traffic, how the ensemble combines them, and any load errors."""
    return _current(service.models.active)


@router.post(
    "/reload",
    response_model=ReloadResponse,
    summary="Hot-reload models from the registry",
    dependencies=[Depends(require_admin)],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
        500: {"model": ErrorResponse, "description": "Reload failed; old models keep serving"},
    },
)
def reload_models(service: ServiceDep) -> ReloadResponse:
    """Rebuild the ensembles from the newest registry versions and swap them in.

    Requests in flight finish on the old models. If loading fails the old models keep serving.
    Protected by ``X-API-Key`` when the ``API_ADMIN_KEY`` environment variable is set.
    """
    previous = service.models.active.generation
    try:
        fresh = service.models.reload()
    except ReloadError as error:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(error)) from error
    return ReloadResponse(reloaded=True, previous_generation=previous, current=_current(fresh))


# --------------------------------------------------------------------------- #
# Retraining view and challenger review (Day 21)
# --------------------------------------------------------------------------- #
def retraining_overview(service: InferenceService) -> dict[str, Any]:
    """Retraining history, rollbacks and the challenger waiting for review.

    Returns:
        ``{"enabled", "runs", "rollbacks", "watching", "challenger"}``. ``enabled`` is false when
        no retraining orchestrator is attached to this process (then ``runs`` is empty).
        ``runs`` are the audit-log rows of ``RetrainingResult.to_dict()``, oldest first.
        ``challenger`` is ``None`` unless a :class:`~src.api.service.ChallengerReview` is attached
        and holds one.
    """
    challenger: dict[str, Any] | None = None
    review = service.challenger_review
    if review is not None:
        try:
            challenger = review.pending()
        except Exception as error:  # noqa: BLE001 - a broken hook must not hide the history
            _log.warning("challenger_pending_failed", error=repr(error))
    orchestrator = service.retraining
    if orchestrator is None:
        return {
            "enabled": False,
            "runs": [],
            "rollbacks": [],
            "watching": [],
            "challenger": challenger,
        }
    rollback = getattr(orchestrator, "rollback", None)
    return {
        "enabled": True,
        "runs": orchestrator.audit.entries("retraining_result", limit=100),
        "rollbacks": orchestrator.audit.entries("model_rollback", limit=50),
        "watching": rollback.watching if rollback is not None else [],
        "challenger": challenger,
    }


def review_challenger(
    service: InferenceService, action: Literal["promote", "reject"]
) -> dict[str, Any]:
    """Promote or reject the waiting challenger.

    Raises:
        HTTPException: 404 when no challenger is waiting, 409 when the hook refuses the action.
    """
    review = service.challenger_review
    if review is None or review.pending() is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no challenger is waiting for review")
    try:
        result = review.promote() if action == "promote" else review.reject()
    except ValueError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    _log.info("challenger_reviewed", action=action)
    return {"action": action, **result}


@router.get("/retraining", summary="Retraining history and the challenger awaiting review")
def retraining(service: ServiceDep) -> dict[str, Any]:
    """Return what the dashboard's model-health page shows about continuous retraining."""
    return retraining_overview(service)


_REVIEW_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
    404: {"model": ErrorResponse, "description": "No challenger is waiting for review"},
    409: {"model": ErrorResponse, "description": "The challenger could not be promoted"},
}


@router.post(
    "/promote",
    summary="Promote the challenger model",
    dependencies=[Depends(require_admin)],
    responses=_REVIEW_RESPONSES,
)
def promote_challenger(service: ServiceDep) -> dict[str, Any]:
    """Make the waiting challenger the serving model (protected by ``X-API-Key`` when set)."""
    return review_challenger(service, "promote")


@router.post(
    "/reject",
    summary="Reject the challenger model",
    dependencies=[Depends(require_admin)],
    responses=_REVIEW_RESPONSES,
)
def reject_challenger(service: ServiceDep) -> dict[str, Any]:
    """Discard the waiting challenger; the champion keeps serving."""
    return review_challenger(service, "reject")
