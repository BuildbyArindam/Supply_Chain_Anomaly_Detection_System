"""Unit tests for src/models/statistical_detector.py."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.models.statistical_detector import (
    AnomalyResult,
    CUSUMDetector,
    IQRDetector,
    RunningStats,
    UnifiedStatisticalDetector,
    ZScoreDetector,
    event_to_observations,
    parse_timestamp,
)

T0 = datetime(2026, 1, 1, 8, 0, 0)


def ts(i: int, step: int = 10) -> datetime:
    return T0 + timedelta(seconds=step * i)


def feed(det, values, entity="s1", start=0, step=10):
    return [det.update(entity, v, ts(start + i, step)) for i, v in enumerate(values)]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------- helpers
def test_running_stats_matches_numpy(rng):
    x = rng.normal(5, 2, 500)
    rs = RunningStats()
    for v in x:
        rs.fit_update(float(v))
    assert rs.mean == pytest.approx(x.mean())
    assert rs.std == pytest.approx(x.std(ddof=1))
    assert rs.transform(7.0) == pytest.approx((7.0 - x.mean()) / x.std(ddof=1))


def test_running_stats_std_floor_when_constant():
    rs = RunningStats(min_std=1e-3)
    for _ in range(10):
        rs.fit_update(1.0)
    assert rs.std == 1e-3


def test_parse_timestamp_variants():
    assert parse_timestamp("2026-01-01T03:04:05Z").hour == 3
    assert parse_timestamp("2026-01-01T03:04:05").minute == 4
    assert parse_timestamp(T0) is T0
    assert parse_timestamp(0).year == 1970
    with pytest.raises(TypeError):
        parse_timestamp(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- z-score
def test_zscore_warmup_never_flags():
    det = ZScoreDetector(min_samples=30)
    out = feed(det, [1.0] * 10 + [100.0])
    assert all(not o.is_anomaly and o.details["warming_up"] for o in out)


def test_zscore_flags_spike_and_not_noise(rng):
    det = ZScoreDetector()
    out = feed(det, rng.normal(3, 0.4, 2000))
    assert sum(o.is_anomaly for o in out[200:]) / 1800 < 0.02
    spike = det.update("s1", 8.0, ts(2001))
    assert spike.is_anomaly and spike.anomaly_type == "point"
    assert spike.score == 1.0 and spike.confidence > 0.5
    assert spike.details["ratio"] > 2


def test_zscore_score_is_half_at_threshold():
    det = ZScoreDetector(threshold=3.0, contextual=False)
    for i, v in enumerate([0.0, 1.0] * 50):
        det.update("e", v, ts(i))
    st = det._states["e"]
    boundary = st.g.mean + 3.0 * st.g.std
    out = det.update("e", boundary + 1e-6, ts(200))
    assert out.score == pytest.approx(0.5, abs=1e-3)
    assert out.is_anomaly


def test_zscore_freezes_baseline_during_anomaly(rng):
    det = ZScoreDetector(contextual=False)
    feed(det, rng.normal(0, 1, 300))
    std_before = det._states["s1"].g.std
    feed(det, [50.0] * 20, start=300)
    assert det._states["s1"].g.std == pytest.approx(std_before)


def test_zscore_resumes_learning_after_max_frozen_run(rng):
    det = ZScoreDetector(contextual=False, max_frozen_run=20)
    feed(det, rng.normal(0, 1, 300))
    out = feed(det, rng.normal(30, 1, 400), start=300)
    assert out[0].is_anomaly
    assert not out[-1].is_anomaly  # accepted the new regime


def test_zscore_contextual_hour_baseline_catches_off_hours_value(rng):
    """Value is unremarkable globally but extreme for the 02:00 baseline."""
    det = ZScoreDetector(threshold=3.0, min_context_samples=30)
    day = datetime(2026, 1, 1)
    # night (02:00-03:00) around 0, day (10:00-11:00) around 6
    for i in range(300):
        det.update("s", rng.normal(0, 0.5), day + timedelta(hours=2, seconds=i))
        det.update("s", rng.normal(6, 0.5), day + timedelta(hours=10, seconds=i))
    night = det.update("s", 3.0, day + timedelta(hours=2, minutes=30))
    assert night.is_anomaly and night.anomaly_type == "contextual"
    assert abs(night.details["z_global"]) < 3.0
    daytime = det.update("s", 3.0, day + timedelta(hours=10, minutes=30))
    assert daytime.is_anomaly  # 3.0 is also far below the daytime mean
    nocontext = ZScoreDetector(contextual=False)
    for i in range(300):
        nocontext.update("s", rng.normal(0, 0.5), day + timedelta(hours=2, seconds=i))
        nocontext.update("s", rng.normal(6, 0.5), day + timedelta(hours=10, seconds=i))
    assert not nocontext.update("s", 3.0, day + timedelta(hours=2, minutes=30)).is_anomaly


def test_zscore_zone_hour_and_entity_threshold_precedence():
    det = ZScoreDetector(threshold=3.0, zone_thresholds={"A": 2.0}, hour_thresholds={3: 4.0})
    assert det.threshold_for("e", None, 0) == 3.0
    assert det.threshold_for("e", "A", 0) == 2.0
    assert det.threshold_for("e", "A", 3) == 4.0  # hour beats zone
    det.adjust_threshold("e", 0.5)
    assert det.threshold_for("e", "A", 0) == 2.5
    assert det.threshold_for("other", "A", 0) == 2.0


def test_zscore_threshold_adjustment_changes_decision(rng):
    det = ZScoreDetector(contextual=False)
    feed(det, rng.normal(0, 1, 500))
    mid = det.update("s1", 3.5, ts(600))
    assert mid.is_anomaly
    det.adjust_threshold("s1", 1.0)
    assert not det.update("s1", 3.5, ts(601)).is_anomaly


def test_zscore_non_finite_is_ignored():
    det = ZScoreDetector()
    out = det.update("e", float("nan"), ts(0))
    assert not out.is_anomaly and out.details["skipped"] == "non_finite_value"
    assert "e" not in det._states


def test_zscore_entities_are_independent(rng):
    det = ZScoreDetector()
    feed(det, rng.normal(0, 1, 200), entity="a")
    feed(det, rng.normal(100, 1, 200), entity="b")
    assert not det.update("b", 100.5, ts(300)).is_anomaly
    assert det.update("a", 100.5, ts(300)).is_anomaly


def test_zscore_reset():
    det = ZScoreDetector()
    feed(det, [1.0, 2.0] * 30)
    det.reset("s1")
    assert "s1" not in det._states
    det.reset()
    assert not det._states


def test_zscore_scaler_factory_is_used():
    calls = []

    class Fake:
        def fit_update(self, x):
            calls.append(x)

        def transform(self, x):
            return 10.0

    det = ZScoreDetector(min_samples=1, contextual=False, scaler_factory=Fake)
    det.update("e", 1.0, ts(0))
    out = det.update("e", 2.0, ts(1))
    assert out.is_anomaly and out.details["z_global"] == 10.0


# ---------------------------------------------------------------- IQR
def test_iqr_flags_spike_not_noise(rng):
    det = IQRDetector()
    out = feed(det, rng.normal(3, 0.4, 3000))
    assert sum(o.is_anomaly for o in out[100:]) / 2900 < 0.01
    spike = det.update("s1", 8.0, ts(3001))
    assert spike.is_anomaly and spike.anomaly_type == "point"


def test_iqr_flags_low_side(rng):
    det = IQRDetector()
    feed(det, rng.normal(3, 0.4, 500))
    assert det.update("s1", -2.0, ts(600)).is_anomaly


def test_iqr_seasonal_bins_detect_contextual(rng):
    det = IQRDetector(window=500)
    day = datetime(2026, 1, 1)
    for i in range(500):
        det.update("s", rng.normal(0, 0.5), day + timedelta(hours=2, seconds=i))
        det.update("s", rng.normal(6, 0.5), day + timedelta(hours=10, seconds=i))
    out = det.update("s", 3.0, day + timedelta(hours=2, minutes=30))
    assert out.is_anomaly and out.anomaly_type == "contextual"


def test_iqr_window_is_bounded(rng):
    det = IQRDetector(window=200)
    feed(det, rng.normal(0, 1, 1000))
    assert len(det._states["s1"].glob.buf) <= 200


def test_iqr_warmup_and_validation():
    det = IQRDetector(min_samples=30)
    assert feed(det, [1.0] * 5)[-1].details["warming_up"]
    with pytest.raises(ValueError):
        IQRDetector(window=10, min_samples=30)
    with pytest.raises(ValueError):
        IQRDetector(bucket_hours=5)


def test_iqr_constant_series_does_not_crash():
    det = IQRDetector()
    out = feed(det, [5.0] * 200)
    assert not any(o.is_anomaly for o in out)
    assert det.update("s1", 5.5, ts(300)).is_anomaly  # any deviation from a flat line is extreme


# ---------------------------------------------------------------- CUSUM
def test_cusum_detects_gradual_drift_before_it_is_large(rng):
    det = CUSUMDetector()
    feed(det, rng.normal(3, 0.4, 300))
    drift = 3 + np.linspace(0, 3, 30) + rng.normal(0, 0.4, 30)  # 0 -> +7.5 sigma
    out = feed(det, drift, start=300)
    first = next(i for i, o in enumerate(out) if o.is_anomaly)
    assert first < 25
    assert out[first].details["direction"] == "up"
    assert out[first].anomaly_type == "collective"
    assert out[first].details["new_alarm"]


def test_cusum_detects_downward_drift(rng):
    det = CUSUMDetector()
    feed(det, rng.normal(3, 0.4, 300))
    out = feed(det, 3 - np.linspace(0, 3, 30), start=300)
    flagged = [o for o in out if o.is_anomaly]
    assert flagged and flagged[0].details["direction"] == "down"


def test_cusum_quiet_on_stationary_noise(rng):
    det = CUSUMDetector()
    out = feed(det, rng.normal(3, 0.4, 20000))
    assert sum(o.is_anomaly for o in out[100:]) / 19900 < 0.01


def test_cusum_cooldown_holds_alarm_then_releases(rng):
    det = CUSUMDetector(cooldown_readings=5)
    feed(det, rng.normal(0, 1, 200))
    out = feed(det, [6.0] * 3 + list(rng.normal(0, 1, 30)), start=200)
    flags = [o.is_anomaly for o in out]
    alarm = flags.index(True)
    assert out[alarm].details["new_alarm"]
    assert all(flags[alarm : alarm + 6])  # alarm + 5 held readings
    assert out[alarm + 1].details["held"] and not out[alarm + 1].details["new_alarm"]
    assert not any(flags[alarm + 8 :])


def test_cusum_baseline_not_polluted_by_drift(rng):
    det = CUSUMDetector()
    feed(det, rng.normal(0, 1, 300))
    mean_before = det._states["s1"].stats.mean
    feed(det, np.linspace(0, 10, 40), start=300)
    assert abs(det._states["s1"].stats.mean - mean_before) < 0.5


def test_cusum_validation_and_non_finite():
    with pytest.raises(ValueError):
        CUSUMDetector(h=0)
    assert CUSUMDetector().update("e", math.inf, ts(0)).details["skipped"] == "non_finite_value"


# ---------------------------------------------------------------- unified
def test_unified_weights_are_normalised():
    u = UnifiedStatisticalDetector(z_weight=2, iqr_weight=1, cusum_weight=1)
    assert sum(u.weights.values()) == pytest.approx(1.0)
    assert u.weights["zscore"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        UnifiedStatisticalDetector(z_weight=0, iqr_weight=0, cusum_weight=0)
    with pytest.raises(ValueError):
        UnifiedStatisticalDetector(max_blend=1.5)


def test_unified_spike_result_contents(rng):
    u = UnifiedStatisticalDetector()
    for i, v in enumerate(rng.normal(3, 0.4, 500)):
        u.detect("s1", float(v), ts(i))
    res = u.detect("s1", 9.0, ts(600))
    assert isinstance(res, AnomalyResult)
    assert res.is_anomaly and res.anomaly_type == "point"
    assert "zscore" in res.contributing_detectors
    assert res.severity_score >= 0.5
    assert set(res.detector_scores) == {"zscore", "iqr", "cusum"}
    d = res.to_dict()
    assert d["detector_scores"]["zscore"]["detector"] == "zscore"


def test_unified_severity_is_consistent_with_flag(rng):
    u = UnifiedStatisticalDetector()
    for i, v in enumerate(rng.normal(3, 0.4, 3000)):
        res = u.detect("s1", float(v), ts(i))
        assert (res.severity_score >= 0.5) == res.is_anomaly
        assert 0.0 <= res.severity_score <= 1.0
        if not res.is_anomaly:
            assert res.anomaly_type is None and res.contributing_detectors == []


def test_unified_drift_typed_collective(rng):
    u = UnifiedStatisticalDetector()
    for i, v in enumerate(rng.normal(3, 0.4, 400)):
        u.detect("s1", float(v), ts(i))
    results = [u.detect("s1", float(v), ts(400 + i)) for i, v in enumerate(3 + np.linspace(0, 2.4, 30))]
    flagged = [r for r in results if r.is_anomaly]
    assert flagged
    assert "cusum" in flagged[0].contributing_detectors or flagged[0].anomaly_type == "point"
    assert any(r.anomaly_type == "collective" for r in flagged)


def test_unified_vote_threshold_requires_agreement(rng):
    or_rule = UnifiedStatisticalDetector(vote_threshold=0.25)
    and_rule = UnifiedStatisticalDetector(vote_threshold=0.99)
    base = rng.normal(3, 0.4, 600)
    for i, v in enumerate(base):
        or_rule.detect("s", float(v), ts(i))
        and_rule.detect("s", float(v), ts(i))
    # moderate excursion: Z fires, CUSUM (needs accumulation) does not on a single reading
    value = 3 + 4.0 * 0.4
    a = or_rule.detect("s", value, ts(700))
    b = and_rule.detect("s", value, ts(700))
    assert a.is_anomaly
    assert not b.is_anomaly


def test_unified_cold_start_is_quiet():
    u = UnifiedStatisticalDetector()
    res = u.detect("new", 123.0, ts(0))
    assert not res.is_anomaly and res.details["warming_up"]


# ---------------------------------------------------------------- events
def test_event_to_observations_schemas():
    wh = {"sensor_id": "S1", "timestamp": "2026-01-01T00:00:00", "value": 3.1, "metadata": {"zone": "A"}}
    (o,) = event_to_observations(wh)
    assert (o.entity_id, o.value, o.zone) == ("S1", 3.1, "A")

    gps = {"vehicle_id": "V1", "timestamp": "2026-01-01T00:00:00", "speed_kmh": 55.0, "route_adherence_score": 0.9}
    obs = event_to_observations(gps)
    assert [x.entity_id for x in obs] == ["V1:speed", "V1:adherence"]

    inv = {"sku_id": "K1", "warehouse_id": "W1", "timestamp": "2026-01-01T00:00:00", "change_amount": -4}
    assert event_to_observations(inv)[0].entity_id == "K1@W1"

    with pytest.raises(ValueError):
        event_to_observations({"timestamp": "2026-01-01T00:00:00"})


def test_detect_event_returns_most_severe_metric(rng):
    u = UnifiedStatisticalDetector()
    for i in range(300):
        u.detect_event(
            {
                "vehicle_id": "V1",
                "timestamp": ts(i).isoformat(),
                "speed_kmh": float(rng.normal(60, 3)),
                "route_adherence_score": float(rng.normal(0.95, 0.01)),
            }
        )
    res = u.detect_event(
        {"vehicle_id": "V1", "timestamp": ts(400).isoformat(), "speed_kmh": 60.0, "route_adherence_score": 0.2}
    )
    assert res.is_anomaly
    assert res.details["metric"] == "route_adherence" and res.details["n_metrics"] == 2
    assert res.entity_id == "V1:adherence"


# ---------------------------------------------------------------- CUSUM seasonality
def _diurnal_series(rng, days, step=60, amp=2.0, sd=0.3):
    n = int(days * 86400 / step)
    hours = (np.arange(n) * step / 3600.0) % 24
    return 5 + amp * np.sin(2 * np.pi * (hours - 8) / 24) + rng.normal(0, sd, n), n


def test_cusum_seasonal_ignores_daily_cycle_but_global_does_not(rng):
    x, n = _diurnal_series(rng, 4)
    seasonal, plain = CUSUMDetector(), CUSUMDetector(seasonal=False)
    s_flags = [seasonal.update("e", float(v), ts(i, 60)).is_anomaly for i, v in enumerate(x)]
    p_flags = [plain.update("e", float(v), ts(i, 60)).is_anomaly for i, v in enumerate(x)]
    day4 = slice(3 * 1440, n)
    assert sum(s_flags[day4]) / 1440 < 0.02
    assert sum(p_flags[day4]) / 1440 > 0.10  # the cycle looks like permanent drift


def test_cusum_seasonal_still_catches_drift_on_cyclic_signal(rng):
    x, n = _diurnal_series(rng, 4)
    det = CUSUMDetector()
    for i, v in enumerate(x[: 3 * 1440]):
        det.update("e", float(v), ts(i, 60))
    start = 3 * 1440 + 600
    drifted = x.copy()
    drifted[start : start + 60] += np.linspace(0, 1.5, 60)  # 5 sigma ramp
    out = [det.update("e", float(drifted[i]), ts(i, 60)) for i in range(3 * 1440, start + 80)]
    assert any(o.is_anomaly for o in out[600:])
    assert not any(o.is_anomaly for o in out[300:560])


def test_cusum_seasonal_warms_up_per_bucket():
    det = CUSUMDetector(min_bucket_samples=30)
    out = feed(det, [1.0] * 10)
    assert all(o.details["warming_up"] for o in out)


def test_cusum_bucket_hours_validation():
    with pytest.raises(ValueError):
        CUSUMDetector(bucket_hours=5)


def test_production_factory_uses_tuned_settings():
    u = UnifiedStatisticalDetector.production()
    assert u.zscore.threshold == 3.5 and u.iqr.multiplier == 2.5
    assert (u.cusum.k, u.cusum.h) == (0.75, 10.0) and u.vote_threshold == 0.25
