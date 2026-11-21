"""Tests for src/models/feature_engineering.py (Day 6)."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from src.kafka.feature_cache import FeatureCache
from src.kafka.schemas import KafkaMessage
from src.kafka.stream_processor import WindowedAggregator
from src.models.feature_engineering import (
    EntityHistory,
    FeatureEngineer,
    FeaturePipeline,
    NoDataError,
    OnlineScaler,
    SequenceBuilder,
    acceptable_range,
    bundles_from_features,
    feature_names,
    fill_missing,
)

START = datetime(2026, 11, 20, 8, 0, 0, tzinfo=UTC)
ENTITY = "wh1-A-temp-01"
_RNG = np.random.default_rng(0)


def warehouse_ctx(**over):
    ctx = {
        "zone": "A",
        "storage_type": "cold_storage",
        "acceptable_range": [2.0, 4.0],
        "last_calibration_date": "2026-08-14",
        "battery_level": 90.0,
        "episode_id": None,
        "episode_step": None,
    }
    ctx.update(over)
    return ctx


class Rig:
    """Real aggregator + cache + pipeline, fed like the Day 5 stream pipeline does."""

    def __init__(self, store=None):
        self.agg = WindowedAggregator()
        self.cache = store if store is not None else FeatureCache()
        self.pipe = FeaturePipeline(self.cache)
        self.emitted = []
        self.raws = []

    def feed(self, entity, source, n, signals_fn, ctx, *, start=START, label=False, offset=0):
        step = {"warehouse": 1.0, "vehicle": 10.0, "inventory": 5.0}[source]
        for i in range(offset, offset + n):
            ts = (start + timedelta(seconds=i * step)).timestamp()
            out = self.agg.update(
                entity, source, ts, signals_fn(i), context=ctx, is_anomaly=label,
                anomaly_type="point" if label else None,
            )  # fmt: skip
            for f in out.features:
                self.cache.store_features(entity, f)
                self.emitted.append(f)
            vec = self.pipe.process(entity)
            self.raws.append(vec.raw if vec else None)


def stable_temp(i):
    return {"value": 3.0 + 0.2 * math.sin(i / 7.0) + float(_RNG.normal(0, 0.05))}


@pytest.fixture
def rig():
    return Rig()


def feats(rig, entity=ENTITY):
    return rig.pipe.process(entity, update_scaler=False).as_dict()


# --------------------------------------------------------------------------- #
# OnlineScaler
# --------------------------------------------------------------------------- #
class TestOnlineScaler:
    def test_matches_numpy_mean_and_sample_std(self):
        x = np.random.default_rng(1).normal(5, 3, size=(200, 4))
        scaler = OnlineScaler(4, clip=None)
        for row in x:
            scaler.fit_update(row)
        np.testing.assert_allclose(scaler.mean, x.mean(axis=0))
        np.testing.assert_allclose(scaler.std, x.std(axis=0, ddof=1))
        np.testing.assert_allclose(scaler.transform(x[0]), (x[0] - x.mean(0)) / x.std(0, ddof=1))

    def test_batch_update_equals_sequential_and_chunked(self):
        x = np.random.default_rng(2).normal(size=(100, 3))
        seq, batch, chunks = OnlineScaler(3), OnlineScaler(3), OnlineScaler(3)
        for row in x:
            seq.fit_update(row)
        batch.fit_update(x)
        for part in np.array_split(x, 7):
            chunks.fit_update(part)
        for other in (batch, chunks):
            np.testing.assert_allclose(other.mean, seq.mean)
            np.testing.assert_allclose(other.std, seq.std)
            np.testing.assert_allclose(other.count, seq.count)

    def test_nan_is_skipped_per_feature(self):
        scaler = OnlineScaler(2)
        scaler.fit_update([1.0, np.nan])
        scaler.fit_update([3.0, 10.0])
        scaler.fit_update([np.nan, 20.0])
        assert scaler.count.tolist() == [2, 2]
        assert scaler.mean.tolist() == [2.0, 15.0]
        out = scaler.transform([np.nan, 15.0])
        assert math.isnan(out[0]) and out[1] == 0.0

    def test_first_observation_and_constant_feature_stay_finite(self):
        scaler = OnlineScaler(2)
        assert scaler.transform([1.0, 2.0]).tolist() == [1.0, 2.0]  # nothing seen: centred on 0
        scaler.fit_update([5.0, 7.0])
        assert scaler.transform([5.0, 7.0]).tolist() == [0.0, 0.0]
        for _ in range(10):
            scaler.fit_update([5.0, 7.0])
        out = scaler.transform([6.0, 7.0])  # zero variance: centred, not divided by ~0
        assert out.tolist() == [1.0, 0.0]

    def test_output_is_clipped(self):
        scaler = OnlineScaler(1, clip=10.0)
        scaler.fit_update(np.arange(100.0).reshape(-1, 1))
        assert scaler.transform([1e9])[0] == 10.0
        assert scaler.transform([-1e9])[0] == -10.0

    def test_json_round_trip(self, tmp_path):
        scaler = OnlineScaler(3, clip=5.0)
        scaler.fit_update(np.random.default_rng(3).normal(size=(50, 3)))
        path = tmp_path / "nested" / "scaler.json"
        scaler.save(path)
        loaded = OnlineScaler.load(path)
        x = np.array([0.3, -1.0, 2.0])
        np.testing.assert_array_equal(loaded.transform(x), scaler.transform(x))
        assert loaded.clip == 5.0
        loaded.fit_update(x)  # still usable after loading
        assert loaded.count[0] == 51

    def test_rejects_wrong_width_and_bad_size(self):
        scaler = OnlineScaler(3)
        with pytest.raises(ValueError):
            scaler.fit_update([1.0, 2.0])
        with pytest.raises(ValueError):
            scaler.transform(np.zeros((4, 2)))
        with pytest.raises(ValueError):
            OnlineScaler(0)


# --------------------------------------------------------------------------- #
# EntityHistory
# --------------------------------------------------------------------------- #
class TestEntityHistory:
    def test_time_since_last_anomaly_is_as_of(self):
        h = EntityHistory()
        assert h.time_since_last_anomaly("e", 1000.0) is None
        h.record_anomaly("e", 900.0)
        h.record_anomaly("e", 500.0)
        assert h.time_since_last_anomaly("e", 1000.0) == 100.0
        assert h.time_since_last_anomaly("e", 600.0) == 100.0
        assert h.time_since_last_anomaly("e", 400.0) is None

    def test_observe_is_idempotent_and_rate_is_flagged_share(self):
        h = EntityHistory()
        t0 = START.timestamp()
        for i in range(10):
            assert h.observe("e", t0 + i)
        assert not h.observe("e", t0 + 9)  # replay of the same event
        h.record_anomaly("e", t0 + 3)
        assert h.anomaly_rate("e", t0 + 10) == pytest.approx(0.1)
        assert h.anomaly_rate("nobody", t0) == 0.0

    def test_rate_forgets_after_seven_days(self):
        h = EntityHistory()
        t0 = START.timestamp()
        h.observe("e", t0)
        h.record_anomaly("e", t0)
        h.observe("e", t0 + 8 * 86_400)
        assert h.anomaly_rate("e", t0 + 8 * 86_400) == 0.0

    def test_reliability_drops_with_false_positives_only(self):
        h = EntityHistory()
        assert h.reliability("e") == 1.0
        h.record_feedback("e", false_positive=False)
        h.record_feedback("e", false_positive=False)
        assert h.reliability("e") == 1.0
        h.record_feedback("e", false_positive=True)
        assert h.reliability("e") == pytest.approx(1 - 1 / 5)
        assert h.reliability("other") == 1.0

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            EntityHistory(rate_days=0)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class TestSchema:
    @pytest.mark.parametrize("source", ["warehouse", "vehicle", "inventory"])
    def test_at_least_25_unique_features(self, source):
        names = feature_names(source)
        assert len(names) >= 25
        assert len(set(names)) == len(names)

    def test_zone_vocabulary_is_configurable(self):
        assert "zone_D" in feature_names("warehouse", ("A", "D"))
        assert not any(n.startswith("zone_") for n in feature_names("vehicle"))

    def test_unknown_source_raises(self):
        with pytest.raises(KeyError):
            feature_names("order")

    def test_acceptable_range_rules(self):
        assert acceptable_range("warehouse", "value", {"acceptable_range": [2, 4]}) == (2.0, 4.0)
        assert acceptable_range("warehouse", "value", {}) is None
        assert acceptable_range("warehouse", "value", {"acceptable_range": [4, 2]}) is None
        assert acceptable_range("vehicle", "speed_kmh", {"cruise_speed_kmh": 50}) == (30.0, 60.0)
        assert acceptable_range("vehicle", "route_adherence_score", {}) == (0.2, 1.0)
        ctx = {"min_stock": 50, "max_stock": 1000}
        assert acceptable_range("inventory", "new_level", ctx) == (50.0, 1000.0)
        assert acceptable_range("inventory", "change_amount", ctx) is None


# --------------------------------------------------------------------------- #
# Feature values
# --------------------------------------------------------------------------- #
class TestWarehouseFeatures:
    def test_vector_matches_schema_and_is_finite_when_warm(self, rig):
        rig.feed(ENTITY, "warehouse", 400, stable_temp, warehouse_ctx())
        vec = rig.pipe.process(ENTITY, update_scaler=False)
        assert vec.names == tuple(feature_names("warehouse"))
        assert vec.raw.shape == (len(vec.names),)
        assert np.isfinite(vec.raw).all()
        assert vec.scaled.shape == vec.raw.shape

    def test_point_spike_above_range(self, rig):
        rig.feed(ENTITY, "warehouse", 300, stable_temp, warehouse_ctx())
        rig.feed(ENTITY, "warehouse", 1, lambda i: {"value": 8.0}, warehouse_ctx(), offset=300)
        f = feats(rig)
        assert f["value__last"] == 8.0
        assert f["value__dev_signed"] == pytest.approx((8.0 - 4.0) / 2.0)
        assert f["value__margin_high"] == pytest.approx(-2.0)
        assert f["value__margin_low"] == pytest.approx(3.0)
        assert f["value__z_baseline"] > 10
        assert f["value__z_window"] > 10

    def test_value_below_range_is_negative_and_inside_is_zero(self, rig):
        rig.feed(ENTITY, "warehouse", 50, stable_temp, warehouse_ctx())
        assert feats(rig)["value__dev_signed"] == 0.0
        rig.feed(ENTITY, "warehouse", 1, lambda i: {"value": 0.0}, warehouse_ctx(), offset=50)
        assert feats(rig)["value__dev_signed"] == pytest.approx(-1.0)

    def test_cold_start_has_no_baseline_but_no_crash(self, rig):
        rig.feed(ENTITY, "warehouse", 3, stable_temp, warehouse_ctx())
        f = feats(rig)
        assert f["value__z_baseline"] == 0.0
        assert f["value__z_window"] == 0.0
        assert f["value__level_shift"] == 0.0
        assert f["value__vol_ratio"] == 1.0
        assert f["warmup_ratio"] < 0.01
        assert f["value__std_short"] >= 0.0

    def test_first_event_ever(self, rig):
        rig.feed(ENTITY, "warehouse", 1, stable_temp, warehouse_ctx())
        f = feats(rig)
        assert f["value__roc_short"] == 0.0 and f["value__std_short"] == 0.0
        assert f["log_count_short"] == pytest.approx(math.log1p(1))

    def test_first_and_second_derivative(self, rig):
        # linear ramp: slope constant, so acceleration ~ 0
        rig.feed(ENTITY, "warehouse", 400, lambda i: {"value": 0.01 * i}, warehouse_ctx())
        linear = feats(rig)
        assert linear["value__roc_short"] == pytest.approx(0.01, rel=1e-3)
        assert linear["value__roc_medium"] == pytest.approx(0.01, rel=1e-3)
        assert abs(linear["value__accel"]) < 1e-6
        # convex ramp: the short window is steeper than the medium one
        rig2 = Rig()
        rig2.feed(ENTITY, "warehouse", 400, lambda i: {"value": 1e-4 * i * i}, warehouse_ctx())
        assert feats(rig2)["value__accel"] > 0

    def test_cyclical_encoding(self, rig):
        six = datetime(2026, 11, 20, 6, 0, 0, tzinfo=UTC)  # a Friday
        rig.feed(ENTITY, "warehouse", 1, stable_temp, warehouse_ctx(), start=six)
        f = feats(rig)
        assert f["hour_sin"] == pytest.approx(1.0) and f["hour_cos"] == pytest.approx(0.0, abs=1e-9)
        assert f["minute_sin"] == pytest.approx(0.0, abs=1e-9) and f["minute_cos"] == 1.0
        assert f["dow_sin"] == pytest.approx(math.sin(2 * math.pi * 4 / 7))
        assert f["dow_cos"] == pytest.approx(math.cos(2 * math.pi * 4 / 7))

    def test_hour_encoding_is_continuous_across_midnight(self, rig):
        late = datetime(2026, 11, 20, 23, 59, 0, tzinfo=UTC)
        rig.feed(ENTITY, "warehouse", 120, stable_temp, warehouse_ctx(), start=late)
        f = feats(rig)  # 00:01 next day
        assert f["hour_cos"] == pytest.approx(1.0, abs=1e-3)

    def test_sensor_age_zone_and_missing_calibration(self, rig):
        rig.feed(ENTITY, "warehouse", 5, stable_temp, warehouse_ctx())
        f = feats(rig)
        assert f["sensor_age_days"] == 98.0  # 2026-08-14 -> 2026-11-20
        assert (f["zone_A"], f["zone_B"], f["zone_C"]) == (1.0, 0.0, 0.0)
        other = Rig()
        ctx = warehouse_ctx(zone="Z", last_calibration_date=None)
        other.feed("s2", "warehouse", 5, stable_temp, ctx)
        g = feats(other, "s2")
        assert math.isnan(g["sensor_age_days"])
        assert (g["zone_A"], g["zone_B"], g["zone_C"]) == (0.0, 0.0, 0.0)

    def test_missing_range_gives_nan_not_zero(self, rig):
        ctx = {"zone": "A"}
        rig.feed(ENTITY, "warehouse", 30, stable_temp, ctx)
        f = feats(rig)
        assert math.isnan(f["value__dev_signed"])
        assert math.isnan(f["value__margin_low"]) and math.isnan(f["value__margin_high"])

    def test_future_calibration_date_is_not_negative(self, rig):
        rig.feed(
            ENTITY, "warehouse", 2, stable_temp, warehouse_ctx(last_calibration_date="2030-01-01")
        )
        assert feats(rig)["sensor_age_days"] == 0.0


class TestHistoryDrivenFeatures:
    def test_time_since_last_anomaly_uses_detector_flags(self, rig):
        rig.feed(ENTITY, "warehouse", 100, stable_temp, warehouse_ctx())
        assert feats(rig)["time_since_last_anomaly_log"] == pytest.approx(math.log1p(86_400))
        last = START + timedelta(seconds=99)
        rig.pipe.record_anomaly(ENTITY, last - timedelta(seconds=60))
        assert feats(rig)["time_since_last_anomaly_log"] == pytest.approx(math.log1p(60))

    def test_anomaly_rate_counts_each_event_once(self, rig):
        rig.feed(ENTITY, "warehouse", 10, stable_temp, warehouse_ctx())
        rig.pipe.record_anomaly(ENTITY, START + timedelta(seconds=3))
        for _ in range(5):  # re-processing the same event must not count it again
            rig.pipe.process(ENTITY)
        assert feats(rig)["anomaly_rate_7d"] == pytest.approx(0.1)

    def test_reliability_follows_feedback(self, rig):
        rig.feed(ENTITY, "warehouse", 5, stable_temp, warehouse_ctx())
        assert feats(rig)["reliability"] == 1.0
        rig.pipe.record_feedback(ENTITY, false_positive=True)
        assert feats(rig)["reliability"] == pytest.approx(1 - 1 / 3)


class TestNoLabelLeakage:
    def test_ground_truth_never_changes_the_vector(self):
        clean, tainted = Rig(), Rig()
        clean.feed(
            ENTITY, "warehouse", 100, lambda i: {"value": 3.0 + 0.01 * (i % 5)}, warehouse_ctx()
        )
        leak_ctx = warehouse_ctx(episode_id="ep-9", episode_step=3, anomaly_subtype="drift")
        tainted.feed(
            ENTITY,
            "warehouse",
            100,
            lambda i: {"value": 3.0 + 0.01 * (i % 5)},
            leak_ctx,
            label=True,
        )
        np.testing.assert_array_equal(clean.raws[-1], tainted.raws[-1])

    def test_no_feature_is_named_after_labels(self):
        for source in ("warehouse", "vehicle", "inventory"):
            assert not any(
                bad in n for n in feature_names(source) for bad in ("label", "episode", "subtype")
            )


class TestOtherSources:
    def test_vehicle(self, rig):
        ctx = {
            "route_id": "R1",
            "driver_id": "d1",
            "cruise_speed_kmh": 65.0,
            "distance_from_route_m": 3,
        }
        sig = lambda i: {"speed_kmh": 65.0 + (i % 3), "route_adherence_score": 0.99}  # noqa: E731
        rig.feed("veh-001", "vehicle", 100, sig, ctx)
        rig.feed(
            "veh-001",
            "vehicle",
            1,
            lambda i: {"speed_kmh": 130.0, "route_adherence_score": 0.3},
            ctx,
            offset=100,
        )
        f = feats(rig, "veh-001")
        assert list(f) == feature_names("vehicle")
        assert f["speed_kmh__dev_signed"] == pytest.approx((130 - 78) / (78 - 39))
        assert f["route_adherence_score__dev_signed"] == 0.0  # 0.3 is inside (0.2, 1.0)
        assert not any(k.startswith(("zone_", "sensor_age")) for k in f)

    def test_inventory(self, rig):
        ctx = {"zone": "B", "min_stock": 50, "max_stock": 1000, "reorder_point": 288}
        sig = lambda i: {"new_level": 400.0, "change_amount": float(-1 - i % 3)}  # noqa: E731
        rig.feed("SKU-1", "inventory", 60, sig, ctx)
        rig.feed(
            "SKU-1",
            "inventory",
            1,
            lambda i: {"new_level": 20.0, "change_amount": -380.0},
            ctx,
            offset=60,
        )
        f = feats(rig, "SKU-1")
        assert list(f) == feature_names("inventory")
        assert f["new_level__dev_signed"] == pytest.approx((20 - 50) / 950)
        assert f["stock_vs_reorder"] == pytest.approx((20 - 288) / 950)
        assert f["zone_B"] == 1.0
        assert "change_amount__dev_signed" not in f


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #
class TestSequenceBuilder:
    def test_shape_dtype_and_equals_online_vector(self, rig):
        rig.feed(ENTITY, "warehouse", 200, stable_temp, warehouse_ctx())
        seq = rig.pipe.build_sequence(ENTITY, lookback=60)
        n = len(feature_names("warehouse"))
        assert seq.shape == (60, n) and seq.dtype == torch.float32
        assert torch.isfinite(seq).all()
        online = rig.pipe.process(ENTITY, update_scaler=False).scaled
        np.testing.assert_allclose(seq[-1].numpy(), np.nan_to_num(online), rtol=1e-4, atol=1e-5)

    def test_cold_start_pads_the_front_with_zeros(self, rig):
        rig.feed(ENTITY, "warehouse", 10, stable_temp, warehouse_ctx())
        seq, mask = rig.pipe.sequences.build_sequence_with_mask(ENTITY, lookback=60)
        assert seq.shape[0] == 60
        assert mask.tolist() == [False] * 50 + [True] * 10
        assert torch.count_nonzero(seq[:50]) == 0
        assert torch.isfinite(seq).all()

    def test_lookback_shorter_than_history_takes_the_newest(self, rig):
        rig.feed(ENTITY, "warehouse", 100, stable_temp, warehouse_ctx())
        long_seq = rig.pipe.build_sequence(ENTITY, lookback=60)
        short_seq, mask = rig.pipe.sequences.build_sequence_with_mask(ENTITY, lookback=5)
        assert short_seq.shape[0] == 5 and mask.all()
        torch.testing.assert_close(short_seq, long_seq[-5:])

    def test_lookback_longer_than_cache_history(self):
        rig = Rig(FeatureCache(max_history=20))
        rig.feed(ENTITY, "warehouse", 50, stable_temp, warehouse_ctx())
        _, mask = rig.pipe.sequences.build_sequence_with_mask(ENTITY, lookback=60)
        assert mask.sum() == 20 and not mask[:40].any()

    def test_unknown_entity(self, rig):
        with pytest.raises(NoDataError):
            rig.pipe.build_sequence("ghost", lookback=10)
        seq, mask = rig.pipe.sequences.build_sequence_with_mask("ghost", 10, "warehouse")
        assert seq.shape == (10, len(feature_names("warehouse")))
        assert torch.count_nonzero(seq) == 0 and not mask.any()

    def test_invalid_lookback(self, rig):
        with pytest.raises(ValueError):
            rig.pipe.build_sequence(ENTITY, lookback=0)

    def test_building_never_updates_the_scaler(self, rig):
        rig.feed(ENTITY, "warehouse", 80, stable_temp, warehouse_ctx())
        before = rig.pipe.scalers["warehouse"].count.copy()
        rig.pipe.build_sequence(ENTITY, 60)
        np.testing.assert_array_equal(rig.pipe.scalers["warehouse"].count, before)

    def test_nan_features_are_forward_filled(self, rig):
        idx = feature_names("warehouse").index("sensor_age_days")
        rig.feed(ENTITY, "warehouse", 30, stable_temp, warehouse_ctx())
        rig.feed(
            ENTITY,
            "warehouse",
            30,
            stable_temp,
            warehouse_ctx(last_calibration_date=None),
            offset=30,
        )
        rig.pipe.scalers["warehouse"].mean[idx] = 50.0  # make the known value non-zero when scaled
        seq = rig.pipe.build_sequence(ENTITY, lookback=60)
        column = seq[:, idx]
        assert column[29] == 10.0  # (98 - 50) / 1, clipped
        assert torch.all(column[30:] == 10.0)  # carried forward over the unknown tail
        assert torch.isfinite(seq).all()

    def test_leading_nan_falls_back_to_zero(self, rig):
        idx = feature_names("warehouse").index("sensor_age_days")
        rig.feed(ENTITY, "warehouse", 30, stable_temp, warehouse_ctx(last_calibration_date=None))
        rig.feed(ENTITY, "warehouse", 30, stable_temp, warehouse_ctx(), offset=30)
        rig.pipe.scalers["warehouse"].mean[idx] = 50.0
        column = rig.pipe.build_sequence(ENTITY, lookback=60)[:, idx]
        assert torch.all(column[:30] == 0.0) and torch.all(column[30:] == 10.0)

    def test_step_with_missing_window_is_forward_filled(self, rig):
        rig.feed(ENTITY, "warehouse", 80, stable_temp, warehouse_ctx())
        broken = FeatureCache()
        dropped_end = None
        for f in rig.emitted:
            if f.window_seconds == 900 and f.window_end == START + timedelta(seconds=50):
                dropped_end = f.window_end
                continue
            broken.store_features(ENTITY, f)
        assert dropped_end is not None
        builder = SequenceBuilder(broken, rig.pipe.engineer, rig.pipe.scalers)
        seq, mask = builder.build_sequence_with_mask(ENTITY, lookback=60)
        pos = 50 - 20  # shortest history covers events 20..79
        assert not mask[pos] and mask.sum() == 59
        torch.testing.assert_close(seq[pos], seq[pos - 1])

    def test_all_steps_incomplete_acts_like_no_data(self):
        cache = FeatureCache()
        agg = WindowedAggregator()
        out = agg.update(ENTITY, "warehouse", START.timestamp(), {"value": 3.0}, context={})
        cache.store_features(ENTITY, out.features[0])  # only the 60 s window
        pipe = FeaturePipeline(cache)
        with pytest.raises(NoDataError):
            pipe.build_sequence(ENTITY, 5)


class TestFillMissing:
    def test_forward_then_zero(self):
        m = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan], [4.0, 5.0]])
        out = fill_missing(m)
        assert out[:, 0].tolist() == [0.0, 2.0, 2.0, 4.0]
        assert out[:, 1].tolist() == [1.0, 1.0, 1.0, 5.0]
        assert fill_missing(np.full((3, 1), np.nan)).tolist() == [[0.0]] * 3
        assert np.isnan(m).any()  # input is not modified


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class TestFeaturePipeline:
    def test_unknown_entity_returns_none(self, rig):
        assert rig.pipe.process("ghost") is None
        assert rig.pipe.bundle("ghost") is None

    def test_process_updates_scaler_unless_told_not_to(self, rig):
        rig.feed(ENTITY, "warehouse", 10, stable_temp, warehouse_ctx())
        scaler = rig.pipe.scalers["warehouse"]
        assert scaler.count.max() == 10
        rig.pipe.process(ENTITY, update_scaler=False)
        assert scaler.count.max() == 10

    def test_offline_vectors_equal_online_vectors(self, rig):
        rig.feed(ENTITY, "warehouse", 120, stable_temp, warehouse_ctx())
        bundles = bundles_from_features(rig.emitted)
        assert len(bundles) == 120
        offline = rig.pipe.vectorize_bundles(bundles)["warehouse"]
        online = np.vstack(rig.raws)
        np.testing.assert_allclose(offline, online, equal_nan=True)

    def test_vectorize_bundles_can_fit_scalers(self):
        rig = Rig()
        rig.feed(ENTITY, "warehouse", 50, stable_temp, warehouse_ctx())
        fresh = FeaturePipeline(FeatureCache())
        fresh.vectorize_bundles(bundles_from_features(rig.emitted), update_scaler=True)
        assert fresh.scalers["warehouse"].count.max() == 50

    def test_bundles_drop_incomplete_groups(self):
        agg = WindowedAggregator()
        full = agg.update(ENTITY, "warehouse", START.timestamp(), {"value": 3.0}, context={})
        part = agg.update(ENTITY, "warehouse", START.timestamp() + 1, {"value": 3.0}, context={})
        items = [*full.features, *part.features[:2]]
        assert len(bundles_from_features(items)) == 1

    def test_scaler_persistence(self, rig, tmp_path):
        rig.feed(ENTITY, "warehouse", 60, stable_temp, warehouse_ctx())
        rig.pipe.save_scalers(tmp_path)
        fresh = FeaturePipeline(FeatureCache())
        assert fresh.load_scalers(tmp_path) == 3
        x = rig.raws[-1]
        np.testing.assert_array_equal(
            fresh.scalers["warehouse"].transform(x), rig.pipe.scalers["warehouse"].transform(x)
        )
        assert FeaturePipeline(FeatureCache()).load_scalers(tmp_path / "empty") == 0

    def test_load_rejects_changed_schema(self, rig, tmp_path):
        rig.pipe.save_scalers(tmp_path)
        other = FeaturePipeline(FeatureCache(), zones=("A", "B", "C", "D"))
        with pytest.raises(ValueError):
            other.load_scalers(tmp_path)

    def test_from_config(self):
        from src.utils.config import load_config

        pipe = FeaturePipeline.from_config(load_config(environ={}), FeatureCache())
        assert pipe.window_seconds == (60, 300, 900)
        assert pipe.engineer.zones == ("A", "B", "C")

    def test_engineer_names_cover_computed_features(self, rig):
        rig.feed(ENTITY, "warehouse", 30, stable_temp, warehouse_ctx())
        engineer: FeatureEngineer = rig.pipe.engineer
        computed = engineer.compute(rig.pipe.bundle(ENTITY))
        assert set(computed) == set(engineer.names("warehouse"))


# --------------------------------------------------------------------------- #
# Real simulators end to end
# --------------------------------------------------------------------------- #
def test_real_simulator_output_for_all_sources():
    from src.data_simulation.orchestrator import SimulationOrchestrator
    from src.utils.config import load_config

    cfg = load_config(environ={})
    sim = SimulationOrchestrator.from_config(cfg.simulation, start_time=START, include_orders=False)
    rig = Rig()
    seen = {}
    for event in sim.run_batch(600):
        message = KafkaMessage.from_payload(event["source_type"], event["payload"])
        outcome = rig.agg.process_message(message)
        for f in outcome.features:
            rig.cache.store_features(f.entity_id, f)
        if outcome.features:
            vec = rig.pipe.process(message.entity_id)
            assert vec is not None and vec.raw.shape == (len(feature_names(vec.source_type)),)
            seen[message.entity_id] = vec
    assert {v.source_type for v in seen.values()} == {"warehouse", "vehicle", "inventory"}
    for source in ("warehouse", "vehicle", "inventory"):
        vecs = [v for v in seen.values() if v.source_type == source]
        assert all(np.isfinite(np.nan_to_num(v.scaled)).all() for v in vecs)
    sensor = next(e for e, v in seen.items() if v.source_type == "warehouse")
    seq = rig.pipe.build_sequence(sensor, lookback=60)
    assert seq.shape == (60, len(feature_names("warehouse"))) and torch.isfinite(seq).all()
