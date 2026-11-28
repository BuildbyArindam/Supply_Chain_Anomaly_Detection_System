"""SeverityScorer: all four levels trigger at the configured thresholds (Day 23).

Cold-storage temperature has impact 1.0, is not time-amplified and has no asset value in these
tests, so the final severity score equals the ensemble score and the boundaries can be probed
exactly. The other tests isolate one factor at a time (impact, business hours, asset value).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.models.ensemble_detector import (
    IMPACT_COLD_STORAGE,
    IMPACT_DEFAULT,
    IMPACT_INVENTORY,
    IMPACT_VEHICLE,
    EntityContext,
    Severity,
    SeverityScorer,
)
from src.utils.config import get_config

COLD = EntityContext(entity_type="warehouse_sensor", impact_key=IMPACT_COLD_STORAGE)
NIGHT = datetime(2026, 3, 10, 3, 0, tzinfo=UTC)
NOON = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def scorer() -> SeverityScorer:
    return SeverityScorer()


# --------------------------------------------------------------------------- #
# The four levels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("ensemble", "level"),
    [
        (0.0, Severity.INFO),
        (0.299, Severity.INFO),
        (0.30, Severity.WARNING),
        (0.599, Severity.WARNING),
        (0.60, Severity.CRITICAL),
        (0.849, Severity.CRITICAL),
        (0.85, Severity.EMERGENCY),
        (1.0, Severity.EMERGENCY),
    ],
)
def test_levels_switch_exactly_at_the_thresholds(scorer, ensemble, level):
    result = scorer.score(ensemble, COLD, timestamp=NIGHT)
    assert result.level is level
    assert result.score == pytest.approx(ensemble)


def test_levels_are_ordered_and_escalate_with_a_cap():
    assert Severity.INFO < Severity.WARNING < Severity.CRITICAL < Severity.EMERGENCY
    assert Severity.INFO.escalated() is Severity.WARNING
    assert Severity.WARNING.escalated(2) is Severity.EMERGENCY
    assert Severity.EMERGENCY.escalated(3) is Severity.EMERGENCY


def test_thresholds_come_from_the_config():
    scorer = SeverityScorer.from_config(get_config().alerts)
    assert scorer.thresholds == (0.3, 0.6, 0.85)
    assert scorer.impact_weights[IMPACT_COLD_STORAGE] == 1.0
    assert scorer.impact_weights[IMPACT_VEHICLE] == 0.8
    assert scorer.impact_weights[IMPACT_INVENTORY] == 0.7


def test_custom_thresholds_move_the_boundaries():
    scorer = SeverityScorer(thresholds=(0.1, 0.2, 0.3))
    assert scorer.score(0.15, COLD, timestamp=NIGHT).level is Severity.WARNING
    assert scorer.score(0.25, COLD, timestamp=NIGHT).level is Severity.CRITICAL
    assert scorer.score(0.35, COLD, timestamp=NIGHT).level is Severity.EMERGENCY


# --------------------------------------------------------------------------- #
# Business impact, time sensitivity, asset value
# --------------------------------------------------------------------------- #
def test_business_impact_weights_scale_and_cap_the_score(scorer):
    at_night = {
        key: scorer.score(0.9, EntityContext(impact_key=key), timestamp=NIGHT)
        for key in (IMPACT_COLD_STORAGE, IMPACT_VEHICLE, IMPACT_INVENTORY, IMPACT_DEFAULT)
    }
    assert at_night[IMPACT_COLD_STORAGE].score == pytest.approx(0.9)
    assert at_night[IMPACT_VEHICLE].score == pytest.approx(0.72)
    assert at_night[IMPACT_INVENTORY].score == pytest.approx(0.63)
    assert at_night[IMPACT_DEFAULT].score == pytest.approx(0.45)
    levels = [r.level for r in at_night.values()]
    assert levels == [Severity.EMERGENCY, Severity.CRITICAL, Severity.CRITICAL, Severity.WARNING]
    assert at_night[IMPACT_VEHICLE].business_impact == 0.8


def test_unknown_impact_key_falls_back_to_the_default_weight(scorer):
    result = scorer.score(0.8, EntityContext(impact_key="something_new"), timestamp=NIGHT)
    assert result.business_impact == scorer.impact_weights[IMPACT_DEFAULT]


def test_business_hours_boost_applies_to_time_sensitive_entities_only(scorer):
    vehicle = EntityContext(impact_key=IMPACT_VEHICLE)
    day = scorer.score(0.7, vehicle, timestamp=NOON)
    night = scorer.score(0.7, vehicle, timestamp=NIGHT)
    assert day.components["time_factor"] == scorer.business_hours_boost
    assert night.components["time_factor"] == 1.0
    assert day.score == pytest.approx(night.score * scorer.business_hours_boost)
    assert (night.level, day.level) == (Severity.WARNING, Severity.CRITICAL)

    # cold-chain temperature is the same at any hour
    assert (
        scorer.score(0.7, COLD, timestamp=NOON).score
        == scorer.score(0.7, COLD, timestamp=NIGHT).score
    )


@pytest.mark.parametrize(("hour", "inside"), [(7, False), (8, True), (17, True), (18, False)])
def test_business_hours_window_is_start_inclusive_end_exclusive(scorer, hour, inside):
    assert scorer.is_business_hours(datetime(2026, 3, 10, hour, 30, tzinfo=UTC)) is inside


def test_asset_value_raises_severity_up_to_the_reference_value(scorer):
    def asset(value):
        ctx = EntityContext(impact_key=IMPACT_COLD_STORAGE, asset_value=value)
        return scorer.score(0.5, ctx, timestamp=NIGHT)

    assert asset(None).components["asset_factor"] == 1.0
    assert asset(0).components["asset_factor"] == 1.0
    half = asset(scorer.asset_reference_value / 2)
    full = asset(scorer.asset_reference_value)
    huge = asset(100 * scorer.asset_reference_value)
    assert half.components["asset_factor"] == pytest.approx(1 + scorer.asset_gain / 2)
    assert full.components["asset_factor"] == pytest.approx(1 + scorer.asset_gain)
    assert huge.score == pytest.approx(full.score)  # saturates
    assert asset(None).score < half.score < full.score


def test_score_is_clipped_to_the_unit_interval(scorer):
    ctx = EntityContext(impact_key=IMPACT_COLD_STORAGE, asset_value=1e9)
    assert scorer.score(1.0, ctx, timestamp=NOON).score == 1.0
    assert scorer.score(-0.5, COLD, timestamp=NOON).score == 0.0


def test_result_carries_every_factor_and_an_action(scorer):
    result = scorer.score(0.9, COLD, timestamp=NIGHT, anomaly_type="collective")
    assert set(result.components) == {
        "ensemble_score",
        "business_impact",
        "time_factor",
        "asset_factor",
    }
    assert result.components["ensemble_score"] == 0.9
    assert result.recommended_action
    info = scorer.score(0.1, COLD, timestamp=NIGHT, anomaly_type="collective")
    assert info.recommended_action != result.recommended_action  # urgency differs by level


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"thresholds": (0.6, 0.3, 0.85)},
        {"thresholds": (0.0, 0.5, 0.9)},
        {"thresholds": (0.3, 0.6, 1.0)},
        {"impact_weights": {IMPACT_COLD_STORAGE: 1.0}},  # no default entry
        {"impact_weights": {IMPACT_DEFAULT: 1.5}},
        {"business_hours_boost": 0.9},
        {"asset_gain": -0.1},
        {"asset_reference_value": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SeverityScorer(**kwargs)
