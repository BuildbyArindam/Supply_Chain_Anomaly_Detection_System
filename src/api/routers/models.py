"""Model listing, hot reload and active-configuration endpoints (Day 14)."""

from __future__ import annotations

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
from src.api.service import ActiveModels, ReloadError

router = APIRouter(prefix="/api/v1/models", tags=["models"])


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
