"""Live readings from the feature cache for the dashboard (Day 18).

``GET /api/v1/telemetry/entities`` returns the latest reading of every live sensor, vehicle and
SKU; ``GET /api/v1/telemetry/{entity_id}`` returns its recent readings. Reads do not count as
feature-cache hits or misses.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import ServiceDep
from src.api.schemas import ErrorResponse
from src.dashboard.telemetry import all_entities, entity_timeseries

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])


@router.get("/entities", summary="Latest reading of every live entity")
def list_entities(service: ServiceDep) -> dict[str, Any]:
    """Latest plotted signal, unit, zone and healthy range per entity, sorted by id."""
    items = all_entities(service.cache)
    return {"count": len(items), "items": items}


@router.get(
    "/{entity_id}",
    summary="Recent readings of one entity",
    responses={404: {"model": ErrorResponse, "description": "Unknown or expired entity"}},
)
def get_timeseries(
    entity_id: str,
    service: ServiceDep,
    lookback_seconds: Annotated[float, Query(ge=1, le=3600)] = 300.0,
) -> dict[str, Any]:
    """Return the readings of the last ``lookback_seconds`` (event time), oldest first."""
    series = entity_timeseries(service.cache, entity_id, lookback_seconds)
    if series is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no live data for {entity_id!r}")
    return series
