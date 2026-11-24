"""FastAPI dependencies shared by the routers (Day 14)."""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from src.api.service import InferenceService
from src.utils.config import AppConfig

ADMIN_KEY_ENV = "API_ADMIN_KEY"


def get_service(request: Request) -> InferenceService:
    """Return the :class:`InferenceService` created by the app lifespan."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "service not initialised")
    return service


def get_config(request: Request) -> AppConfig:
    """Return the application config the app was created with."""
    return request.app.state.config


def require_admin(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Protect privileged endpoints with ``X-API-Key`` when ``$API_ADMIN_KEY`` is set.

    With the variable unset (local development) the endpoint is open.
    """
    expected = os.environ.get(ADMIN_KEY_ENV)
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")


ServiceDep = Annotated[InferenceService, Depends(get_service)]
ConfigDep = Annotated[AppConfig, Depends(get_config)]
