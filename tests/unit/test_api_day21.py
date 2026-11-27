"""Tests of the Day 21 API additions: retraining overview and challenger review helpers.

These import the API package, which needs PyTorch, so they are skipped where it is missing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from fastapi import HTTPException  # noqa: E402

from src.api.routers.models import retraining_overview, review_challenger  # noqa: E402
from tests.unit.test_dashboard_day21 import CHALLENGER, RUN_1  # noqa: E402


class Review:
    def __init__(self, waiting: bool = True) -> None:
        self.waiting, self.calls = waiting, []

    def pending(self):
        return CHALLENGER if self.waiting else None

    def promote(self):
        self.calls.append("promote")
        return {"version": "v2"}

    def reject(self):
        self.calls.append("reject")
        return {}


def fake_service(**attrs: Any) -> Any:
    return SimpleNamespace(retraining=None, challenger_review=None, **attrs)


def test_retraining_overview_without_an_orchestrator():
    out = retraining_overview(fake_service())
    assert out["enabled"] is False and out["runs"] == [] and out["challenger"] is None
    review = Review()
    assert retraining_overview(fake_service(challenger_review=review))["challenger"] == CHALLENGER


def test_retraining_overview_reads_the_audit_log():
    audit = SimpleNamespace(
        entries=lambda kind, limit: [RUN_1] if kind == "retraining_result" else []
    )
    service = fake_service()
    service.retraining = SimpleNamespace(audit=audit, rollback=SimpleNamespace(watching=["m"]))
    out = retraining_overview(service)
    assert out["enabled"] and out["runs"] == [RUN_1] and out["watching"] == ["m"]


def test_review_challenger_endpoint_logic():
    review = Review()
    out = review_challenger(fake_service(challenger_review=review), "promote")
    assert out == {"action": "promote", "version": "v2"} and review.calls == ["promote"]
    for service in (fake_service(), fake_service(challenger_review=Review(waiting=False))):
        with pytest.raises(HTTPException) as err:
            review_challenger(service, "reject")
        assert err.value.status_code == 404
