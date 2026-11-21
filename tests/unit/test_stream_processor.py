import math
import random

import numpy as np
import pytest

from src.kafka.schemas import KafkaMessage
from src.kafka.stream_processor import (
    SignalStats,
    WindowedAggregator,
    WindowedFeatures,
    parse_event_time,
)

T0 = parse_event_time("2026-11-20T08:00:00Z")


def feed(agg, entity, ts, value, source="warehouse", **kw):
    return agg.update(entity, source, ts, {"value": value}, **kw)


def window_of(outcome, seconds):
    return next(f for f in outcome.features if f.window_seconds == seconds)


def reference_stats(points, anchor, span):
    inside = [(t, v) for t, v in points if anchor - span < t <= anchor]
    ts = np.array([t for t, _ in inside])
    vs = np.array([v for _, v in inside])
    slope = float(np.polyfit(ts - ts[0], vs, 1)[0]) if len(inside) > 1 else 0.0
    return {
        "count": len(inside),
        "mean": float(vs.mean()),
        "std": float(vs.std(ddof=1)) if len(inside) > 1 else 0.0,
        "min": float(vs.min()),
        "max": float(vs.max()),
        "p95": float(np.percentile(vs, 95)),
        "slope": slope,
        "last": float(vs[-1]),
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_matches_numpy_for_every_window_over_a_long_noisy_stream():
    rng = random.Random(7)
    agg = WindowedAggregator((60, 300, 900), 30)
    points = []
    for i in range(1500):  # 25 minutes at 1 Hz: all windows fill and slide
        ts = T0 + i
        value = 4.0 + 0.002 * i + rng.gauss(0, 0.4)
        points.append((ts, value))
        out = feed(agg, "s1", ts, value)
    for span in (60, 300, 900):
        feats = window_of(out, span)
        ref = reference_stats(points, ts, span)
        s = feats.signals["value"]
        assert feats.count == ref["count"]
        assert s.mean == pytest.approx(ref["mean"], abs=1e-9)
        assert s.std == pytest.approx(ref["std"], abs=1e-9)
        assert s.min == ref["min"] and s.max == ref["max"]
        assert s.percentile_95 == pytest.approx(ref["p95"], abs=1e-9)
        assert s.rate_of_change == pytest.approx(ref["slope"], abs=1e-9)
        assert s.last == ref["last"]


def test_rate_of_change_is_signal_units_per_second():
    agg = WindowedAggregator((60,), 30)
    for i in range(30):
        out = feed(agg, "s1", T0 + i, 10.0 + 0.5 * i)  # +0.5 per second
    assert window_of(out, 60).signals["value"].rate_of_change == pytest.approx(0.5)


def test_single_point_and_constant_signal_have_zero_spread_and_slope():
    agg = WindowedAggregator((60,), 30)
    first = window_of(feed(agg, "s1", T0, 7.0), 60).signals["value"]
    assert (first.mean, first.std, first.rate_of_change, first.percentile_95) == (
        7.0,
        0.0,
        0.0,
        7.0,
    )
    for i in range(1, 20):
        out = feed(agg, "s1", T0 + i, 7.0)
    flat = window_of(out, 60).signals["value"]
    assert flat.std == 0.0 and flat.rate_of_change == pytest.approx(0.0, abs=1e-12)


def test_window_boundary_is_half_open():
    agg = WindowedAggregator((60,), 30)
    feed(agg, "s1", T0, 1.0)
    out = feed(agg, "s1", T0 + 60, 2.0)  # the point at exactly anchor-60 falls out
    assert window_of(out, 60).count == 1
    out = feed(agg, "s2", T0, 1.0)
    out = feed(agg, "s2", T0 + 59.999, 2.0)
    assert window_of(out, 60).count == 2


def test_numerical_drift_stays_negligible_with_large_offsets_and_many_evictions():
    agg = WindowedAggregator((60,), 30)
    rng = random.Random(1)
    points = []
    for i in range(6000):  # 10x the refresh interval in evictions
        value = 1.0e6 + rng.gauss(0, 0.01)
        points.append((T0 + i, value))
        out = feed(agg, "s1", T0 + i, value)
    ref = reference_stats(points, T0 + 5999, 60)
    s = window_of(out, 60).signals["value"]
    assert s.mean == pytest.approx(ref["mean"], rel=1e-12)
    assert s.std == pytest.approx(ref["std"], rel=1e-6)


def test_multi_signal_entities_get_one_stats_block_per_signal():
    agg = WindowedAggregator((60,), 30)
    out = agg.update("veh-1", "vehicle", T0, {"speed_kmh": 60.0, "route_adherence_score": 1.0})
    out = agg.update("veh-1", "vehicle", T0 + 10, {"speed_kmh": 80.0, "route_adherence_score": 0.5})
    sig = window_of(out, 60).signals
    assert sig["speed_kmh"].mean == 70.0 and sig["route_adherence_score"].mean == 0.75


# --------------------------------------------------------------------------- #
# Windows, warm-up, entities
# --------------------------------------------------------------------------- #
def test_emits_one_feature_set_per_window_with_event_time_bounds():
    agg = WindowedAggregator((60, 300, 900), 30)
    out = feed(agg, "s1", T0, 1.0)
    assert out.status == "emitted"
    assert [f.window_seconds for f in out.features] == [60, 300, 900]
    f = window_of(out, 300)
    assert (f.window_end - f.window_start).total_seconds() == 300
    assert f.window_end.timestamp() == T0 and f.entity_id == "s1"


def test_warm_up_reported_through_coverage():
    agg = WindowedAggregator((60,), 30)
    out = feed(agg, "s1", T0, 1.0)
    assert not window_of(out, 60).is_warm
    for i in range(1, 70):
        out = feed(agg, "s1", T0 + i, 1.0)
    f = window_of(out, 60)
    assert f.is_warm and f.coverage_seconds == pytest.approx(59.0)


def test_entities_are_independent():
    agg = WindowedAggregator((60,), 30)
    feed(agg, "a", T0, 1.0)
    out = feed(agg, "b", T0 + 5, 100.0)
    assert window_of(out, 60).count == 1 and window_of(out, 60).signals["value"].mean == 100.0
    assert agg.entity_count == 2


def test_context_and_labels_are_passed_through():
    agg = WindowedAggregator((60,), 30)
    out = feed(agg, "s1", T0, 9.0, context={"zone": "A"}, is_anomaly=True, anomaly_type="point")
    f = out.features[0]
    assert f.context == {"zone": "A"} and f.label_is_anomaly and f.label_anomaly_type == "point"


# --------------------------------------------------------------------------- #
# Watermark / late events
# --------------------------------------------------------------------------- #
def test_late_event_within_tolerance_is_merged_without_emitting():
    agg = WindowedAggregator((60,), 30)
    feed(agg, "s1", T0, 1.0)
    feed(agg, "s1", T0 + 40, 3.0)
    late = feed(agg, "s1", T0 + 20, 2.0)  # 20 s late, tolerance is 30 s
    assert late.status == "late_accepted" and late.features == ()
    out = feed(agg, "s1", T0 + 41, 4.0)
    assert window_of(out, 60).count == 4
    assert window_of(out, 60).signals["value"].mean == pytest.approx(2.5)


def test_event_beyond_watermark_is_dropped_and_changes_nothing():
    agg = WindowedAggregator((60,), 30)
    feed(agg, "s1", T0, 1.0)
    feed(agg, "s1", T0 + 100, 5.0)
    assert feed(agg, "s1", T0 + 69, 99.0).status == "dropped_late"  # 31 s late
    out = feed(agg, "s1", T0 + 101, 5.0)
    assert window_of(out, 60).signals["value"].max == 5.0


def test_event_exactly_at_the_watermark_is_accepted():
    agg = WindowedAggregator((60,), 30)
    feed(agg, "s1", T0 + 100, 5.0)
    assert feed(agg, "s1", T0 + 70, 1.0).status == "late_accepted"


def test_late_event_outside_the_window_is_accepted_but_ignored():
    agg = WindowedAggregator((10,), 30)  # watermark is wider than the window
    feed(agg, "s1", T0 + 100, 5.0)
    assert feed(agg, "s1", T0 + 85, 99.0).status == "late_accepted"
    assert window_of(feed(agg, "s1", T0 + 101, 6.0), 10).signals["value"].max == 6.0


def test_out_of_order_arrival_gives_same_features_as_sorted_arrival():
    rng = random.Random(3)
    events = [(T0 + i, rng.gauss(0, 1)) for i in range(120)]
    shuffled = list(events)
    for i in range(0, 120, 6):  # swap neighbours: at most 1 s out of order
        shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
    a, b = WindowedAggregator((60,), 30), WindowedAggregator((60,), 30)
    for ts, v in events:
        out_a = feed(a, "s", ts, v)
    for ts, v in shuffled:
        feed(b, "s", ts, v)
    out_b = feed(b, "s", T0 + 120, 0.0)
    out_a = feed(a, "s", T0 + 120, 0.0)
    sa, sb = window_of(out_a, 60).signals["value"], window_of(out_b, 60).signals["value"]
    assert sa.mean == pytest.approx(sb.mean) and sa.std == pytest.approx(sb.std)
    assert sa.rate_of_change == pytest.approx(sb.rate_of_change)


def test_zero_tolerance_rejects_any_out_of_order_event():
    agg = WindowedAggregator((60,), 0)
    feed(agg, "s1", T0 + 10, 1.0)
    assert feed(agg, "s1", T0 + 9, 1.0).status == "dropped_late"
    assert feed(agg, "s1", T0 + 10, 2.0).status == "emitted"  # equal timestamps are in order


# --------------------------------------------------------------------------- #
# Validation, bounds, expiry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_invalid(bad):
    agg = WindowedAggregator((60,), 30)
    assert feed(agg, "s1", T0, bad).status == "invalid"
    assert agg.entity_count == 0


def test_signal_set_change_is_invalid():
    agg = WindowedAggregator((60,), 30)
    feed(agg, "s1", T0, 1.0)
    assert agg.update("s1", "warehouse", T0 + 1, {"other": 1.0}).status == "invalid"


def test_constructor_validation():
    for kwargs in ({"window_seconds": ()}, {"window_seconds": (0,)}, {"watermark_tolerance_s": -1}):
        with pytest.raises(ValueError):
            WindowedAggregator(**kwargs)
    with pytest.raises(ValueError):
        WindowedAggregator((60,), 30, max_points_per_window=1)
    assert WindowedAggregator((900, 60, 60)).window_seconds == (60, 900)


def test_max_points_bounds_memory_and_stats_stay_consistent():
    agg = WindowedAggregator((900,), 30, max_points_per_window=50)
    for i in range(200):
        out = feed(agg, "s1", T0 + i, float(i))
    f = window_of(out, 900)
    assert f.count == 50 and f.signals["value"].min == 150.0


def test_evict_idle_removes_silent_entities_only():
    agg = WindowedAggregator((60, 900), 30)
    feed(agg, "silent", T0, 1.0)
    feed(agg, "alive", T0 + 2000, 1.0)
    assert agg.evict_idle() == 1
    assert agg.entity_count == 1
    assert agg.evict_idle() == 0


def test_max_entities_drops_least_recently_updated():
    agg = WindowedAggregator((60,), 30, max_entities=2)
    for i, name in enumerate("abc"):
        feed(agg, name, T0 + i, 1.0)
    assert agg.entity_count == 2
    assert window_of(feed(agg, "a", T0 + 10, 2.0), 60).count == 1  # "a" was dropped, starts fresh


# --------------------------------------------------------------------------- #
# Envelopes and serialisation
# --------------------------------------------------------------------------- #
WAREHOUSE = {
    "sensor_id": "wh1-A-temp-01",
    "warehouse_id": "wh1",
    "timestamp": "2026-11-20T08:00:00Z",
    "reading_type": "temperature",
    "value": 3.0,
    "unit": "°C",
    "is_anomaly": False,
    "anomaly_type": None,
    "metadata": {"zone": "A"},
}
ORDER = {
    "order_id": "ORD-1",
    "sku_id": "SKU-1",
    "warehouse_id": "wh1",
    "timestamp": "2026-11-20T08:00:09Z",
    "status": "picked",
    "quantity": 1,
    "is_anomaly": False,
    "anomaly_type": None,
}


def test_process_message_handles_every_supported_source():
    agg = WindowedAggregator((60,), 30)
    msg = KafkaMessage.from_payload("warehouse", WAREHOUSE)
    out = agg.process_message(msg)
    f = out.features[0]
    assert out.status == "emitted" and f.entity_id == "wh1-A-temp-01"
    assert f.context["unit"] == "°C" and f.context["zone"] == "A"
    assert f.window_end.isoformat().startswith("2026-11-20T08:00:00")


def test_order_events_are_unsupported():
    out = WindowedAggregator((60,), 30).process_message(KafkaMessage.from_payload("order", ORDER))
    assert out.status == "unsupported" and out.features == ()


def test_features_round_trip_through_dict():
    agg = WindowedAggregator((60,), 30)
    f = window_of(
        feed(agg, "s1", T0, 3.5, context={"zone": "A"}, is_anomaly=True, anomaly_type="point"), 60
    )
    again = WindowedFeatures.from_dict(f.to_dict())
    assert again == f
    assert SignalStats.from_dict(f.signals["value"].to_dict()) == f.signals["value"]
