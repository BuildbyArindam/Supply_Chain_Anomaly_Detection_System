import json
import statistics
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta, timezone
from itertools import islice

import pytest

from src.data_simulation import warehouse_simulator as ws
from src.data_simulation.warehouse_simulator import (
    ANOMALY_TYPES,
    WarehouseSensor,
    WarehouseSimulator,
    build_default_sensors,
)
from src.utils.config import load_config

NIGHT = datetime(2026, 3, 10, 1, 0, 0, tzinfo=UTC)  # inside 00:00-05:00
NOON = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
REF_DATE = date(2026, 3, 10)
NO_ANOMALIES = {
    "point_anomaly_rate": 0.0,
    "contextual_anomaly_rate": 0.0,
    "collective_anomaly_rate": 0.0,
}

COLD_TEMP = WarehouseSensor(
    sensor_id="t1",
    warehouse_id="wh1",
    location="A",
    sensor_type="temperature",
    normal_mean=3.0,
    normal_std=0.25,
    acceptable_range=(2.0, 4.0),
)
HUMIDITY = WarehouseSensor(
    sensor_id="h1",
    warehouse_id="wh1",
    location="B",
    sensor_type="humidity",
    normal_mean=45.0,
    normal_std=2.5,
    acceptable_range=(35.0, 55.0),
)


def make_sim(sensors=None, *, start=NIGHT, seed=7, quiet=True, **kwargs):
    if quiet:
        kwargs = {**NO_ANOMALIES, **kwargs}
    return WarehouseSimulator(
        sensors if sensors is not None else [COLD_TEMP], start_time=start, seed=seed, **kwargs
    )


def run(sim, ticks):
    return list(sim.simulate_all_sensors(n_ticks=ticks))


# --------------------------------------------------------------------------- #
# WarehouseSensor
# --------------------------------------------------------------------------- #
def sensor(**overrides):
    fields = {
        "sensor_id": "s",
        "warehouse_id": "w",
        "location": "A",
        "sensor_type": "temperature",
        "normal_mean": 3.0,
        "normal_std": 0.25,
        "acceptable_range": (2.0, 4.0),
    }
    return WarehouseSensor(**{**fields, **overrides})


def test_sensor_defaults_and_properties():
    s = sensor()
    assert s.unit == "°C"
    assert s.span == 2.0
    assert s.edge_margin == pytest.approx(0.04)
    assert s.firmware_version == "1.0.0"
    assert s.battery_level == 100.0
    assert s.last_calibration_date is None
    assert s.storage_type == "ambient"
    assert HUMIDITY.unit == "%RH"


def test_sensor_normalises_range_to_float_tuple_and_is_frozen():
    s = sensor(acceptable_range=[2, 4])
    assert s.acceptable_range == (2.0, 4.0)
    with pytest.raises(AttributeError):
        s.normal_mean = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"sensor_id": ""},
        {"warehouse_id": ""},
        {"location": ""},
        {"sensor_type": "pressure"},
        {"normal_std": 0.0},
        {"normal_std": -1.0},
        {"acceptable_range": (4.0, 2.0)},
        {"acceptable_range": (2.0, 2.0)},
        {"normal_mean": 5.0},
        {"battery_level": 101.0},
        {"battery_level": -1.0},
        {"sensor_type": "humidity", "normal_mean": 50.0, "acceptable_range": (40.0, 120.0)},
        {"acceptable_range": (-60.0, 4.0), "normal_mean": 0.0},
    ],
)
def test_sensor_validation(overrides):
    with pytest.raises(ValueError):
        sensor(**overrides)


def test_sensor_without_room_for_anomalies_rejected():
    with pytest.raises(ValueError, match="no room"):
        sensor(
            sensor_type="humidity",
            normal_mean=50.0,
            acceptable_range=(0.0, 100.0),
            normal_std=5.0,
        )


def test_spike_feasibility_flags():
    near_top = sensor(
        sensor_type="humidity", normal_mean=90.0, normal_std=2.0, acceptable_range=(80.0, 99.9)
    )
    assert not near_top.can_spike_up
    assert near_top.can_spike_down


# --------------------------------------------------------------------------- #
# build_default_sensors
# --------------------------------------------------------------------------- #
def test_default_fleet_shape_and_ids():
    fleet = build_default_sensors(seed=1, reference_date=REF_DATE)
    assert len(fleet) == 2 * 3 * 2
    ids = [s.sensor_id for s in fleet]
    assert len(set(ids)) == len(ids)
    assert "wh1-A-temp-01" in ids
    assert "wh2-C-hum-01" in ids
    assert {s.warehouse_id for s in fleet} == {"wh1", "wh2"}
    assert {s.location for s in fleet} == {"A", "B", "C"}


def test_default_fleet_zone_profiles():
    fleet = {s.sensor_id: s for s in build_default_sensors(seed=1, reference_date=REF_DATE)}
    cold = fleet["wh1-A-temp-01"]
    dry = fleet["wh1-B-temp-01"]
    assert cold.storage_type == "cold_storage"
    assert cold.acceptable_range == (2.0, 4.0)
    assert dry.storage_type == "dry_goods"
    assert dry.acceptable_range == (18.0, 22.0)
    assert fleet["wh1-A-hum-01"].sensor_type == "humidity"


def test_default_fleet_metadata_is_realistic_and_reproducible():
    a = build_default_sensors(seed=3, reference_date=REF_DATE)
    b = build_default_sensors(seed=3, reference_date=REF_DATE)
    assert a == b
    for s in a:
        assert s.firmware_version
        assert 55.0 <= s.battery_level <= 100.0
        assert s.last_calibration_date is not None
        assert 5 <= (REF_DATE - s.last_calibration_date).days <= 240


def test_default_fleet_extra_sensors_and_zones_cycle_profiles():
    fleet = build_default_sensors(
        n_warehouses=1, zones=["A", "B", "C", "D"], sensors_per_zone=4, reference_date=REF_DATE
    )
    assert len(fleet) == 16
    assert [s.sensor_id for s in fleet if s.location == "A"] == [
        "wh1-A-temp-01",
        "wh1-A-hum-01",
        "wh1-A-temp-02",
        "wh1-A-hum-02",
    ]
    assert next(s for s in fleet if s.location == "D").storage_type == "cold_storage"


def test_default_fleet_uses_today_when_no_reference_date():
    fleet = build_default_sensors(n_warehouses=1, sensors_per_zone=1)
    assert all(s.last_calibration_date < datetime.now(UTC).date() for s in fleet)


@pytest.mark.parametrize("kwargs", [{"n_warehouses": 0}, {"sensors_per_zone": 0}, {"zones": []}])
def test_default_fleet_validation(kwargs):
    with pytest.raises(ValueError):
        build_default_sensors(**kwargs)


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_anomaly_rate": -0.1},
        {"contextual_anomaly_rate": 1.5},
        {"collective_anomaly_rate": 2.0},
        {"point_duration_range": (0, 3)},
        {"contextual_duration_range": (5, 2)},
        {"collective_length_range": (10, 5)},
        {"off_hours_start": 24},
        {"off_hours_end": -1},
        {"reading_interval_s": 0},
        {"reading_interval_s": -1.0},
        {"point_anomaly_rate": 1.0, "point_duration_range": (1, 1), "collective_anomaly_rate": 1.0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        WarehouseSimulator([COLD_TEMP], **kwargs)


def test_constructor_rejects_empty_and_duplicate_sensors():
    with pytest.raises(ValueError, match="at least one"):
        WarehouseSimulator([])
    with pytest.raises(ValueError, match="duplicate"):
        WarehouseSimulator([COLD_TEMP, COLD_TEMP])


def test_default_fleet_when_no_sensors_given():
    sim = WarehouseSimulator(seed=1, start_time=NIGHT)
    assert len(sim.sensors) == 12
    assert sim.get_sensor("wh1-A-temp-01").storage_type == "cold_storage"


def test_start_time_normalisation():
    naive = WarehouseSimulator([COLD_TEMP], start_time=datetime(2026, 3, 10, 1, 0, 0))
    assert naive.current_time == NIGHT
    ist = timezone(timedelta(hours=5, minutes=30))
    shifted = WarehouseSimulator([COLD_TEMP], start_time=datetime(2026, 3, 10, 6, 30, tzinfo=ist))
    assert shifted.current_time == NIGHT
    now = WarehouseSimulator([COLD_TEMP])
    assert abs((datetime.now(UTC) - now.current_time).total_seconds()) < 5


def test_unknown_sensor_raises_key_error():
    sim = make_sim()
    with pytest.raises(KeyError, match="nope"):
        sim.get_sensor("nope")
    with pytest.raises(KeyError):
        sim.inject_point_anomaly("nope")


# --------------------------------------------------------------------------- #
# Reading schema and clock
# --------------------------------------------------------------------------- #
def test_reading_schema_matches_spec():
    sim = make_sim([COLD_TEMP, HUMIDITY], seed=1)
    temp, hum = sim.tick()
    expected_keys = {
        "sensor_id",
        "warehouse_id",
        "timestamp",
        "reading_type",
        "value",
        "unit",
        "is_anomaly",
        "anomaly_type",
        "metadata",
    }
    assert set(temp) == expected_keys
    assert (temp["sensor_id"], temp["warehouse_id"]) == ("t1", "wh1")
    assert temp["reading_type"] == "temperature" and temp["unit"] == "°C"
    assert hum["reading_type"] == "humidity" and hum["unit"] == "%RH"
    assert temp["timestamp"] == "2026-03-10T01:00:00Z"
    assert isinstance(temp["value"], float)
    assert temp["is_anomaly"] is False and temp["anomaly_type"] is None


def test_metadata_contents():
    detailed = sensor(
        sensor_id="m1",
        firmware_version="2.4.0",
        battery_level=80.0,
        last_calibration_date=date(2026, 1, 15),
        storage_type="cold_storage",
    )
    (reading,) = make_sim([detailed]).tick()
    meta = reading["metadata"]
    assert meta["zone"] == "A"
    assert meta["storage_type"] == "cold_storage"
    assert meta["firmware_version"] == "2.4.0"
    assert meta["last_calibration_date"] == "2026-01-15"
    assert meta["acceptable_range"] == [2.0, 4.0]
    assert meta["episode_id"] is None and meta["episode_step"] is None
    assert 79.9 < meta["battery_level"] <= 80.0
    (no_calibration,) = make_sim([COLD_TEMP]).tick()
    assert no_calibration["metadata"]["last_calibration_date"] is None


def test_readings_are_json_serialisable():
    sim = make_sim([COLD_TEMP, HUMIDITY], seed=2)
    sim.inject_collective_anomaly("t1", length=3)
    payload = json.dumps(run(sim, 5))
    assert "collective" in payload


def test_timestamps_advance_one_second_per_sensor_and_stay_ordered():
    sim = make_sim([COLD_TEMP, HUMIDITY])
    readings = run(sim, 10)
    assert len(readings) == 20
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in readings]
    assert stamps == sorted(stamps)
    per_sensor = [r for r in readings if r["sensor_id"] == "t1"]
    gaps = {
        (datetime.fromisoformat(b["timestamp"]) - datetime.fromisoformat(a["timestamp"])).seconds
        for a, b in zip(per_sensor, per_sensor[1:], strict=False)
    }
    assert gaps == {1}
    assert sim.current_time == NIGHT + timedelta(seconds=10)


def test_custom_reading_interval():
    sim = make_sim(reading_interval_s=0.5)
    readings = run(sim, 4)
    assert [r["timestamp"] for r in readings] == [
        "2026-03-10T01:00:00Z",
        "2026-03-10T01:00:00.500000Z",
        "2026-03-10T01:00:01Z",
        "2026-03-10T01:00:01.500000Z",
    ]
    assert sim.reading_interval_s == 0.5


def test_simulate_all_sensors_arguments():
    sim = make_sim([COLD_TEMP, HUMIDITY])
    assert len(list(sim.simulate_all_sensors(n_ticks=3))) == 6
    assert len(list(sim.simulate_all_sensors(duration_s=4))) == 8
    assert len(list(sim.simulate_all_sensors(duration_s=2.2))) == 6  # rounded up to whole ticks
    assert list(sim.simulate_all_sensors(n_ticks=0)) == []
    assert list(sim.simulate_all_sensors(duration_s=0)) == []
    with pytest.raises(ValueError, match="either"):
        list(sim.simulate_all_sensors(n_ticks=1, duration_s=1))
    with pytest.raises(ValueError, match="n_ticks"):
        list(sim.simulate_all_sensors(n_ticks=-1))
    with pytest.raises(ValueError, match="duration_s"):
        list(sim.simulate_all_sensors(duration_s=-1))


def test_simulate_all_sensors_is_endless_by_default():
    sim = make_sim([COLD_TEMP, HUMIDITY])
    assert len(list(islice(sim.simulate_all_sensors(), 500))) == 500


def test_is_off_hours_window_and_wraparound():
    sim = make_sim()
    assert sim.is_off_hours(datetime(2026, 3, 10, 0, 0, tzinfo=UTC))
    assert sim.is_off_hours(datetime(2026, 3, 10, 4, 59, 59, tzinfo=UTC))
    assert not sim.is_off_hours(datetime(2026, 3, 10, 5, 0, tzinfo=UTC))
    assert not sim.is_off_hours(datetime(2026, 3, 10, 23, 0, tzinfo=UTC))
    wrapped = make_sim(off_hours_start=22, off_hours_end=6)
    assert wrapped.is_off_hours(datetime(2026, 3, 10, 23, 0, tzinfo=UTC))
    assert wrapped.is_off_hours(datetime(2026, 3, 10, 2, 0, tzinfo=UTC))
    assert not wrapped.is_off_hours(datetime(2026, 3, 10, 12, 0, tzinfo=UTC))
    empty = make_sim(off_hours_start=3, off_hours_end=3)
    assert not empty.is_off_hours(datetime(2026, 3, 10, 3, 0, tzinfo=UTC))


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_same_stream_different_seed_differs():
    kwargs = {"quiet": False, "start": NIGHT}
    a = run(make_sim([COLD_TEMP, HUMIDITY], seed=11, **kwargs), 2000)
    b = run(make_sim([COLD_TEMP, HUMIDITY], seed=11, **kwargs), 2000)
    c = run(make_sim([COLD_TEMP, HUMIDITY], seed=12, **kwargs), 2000)
    assert a == b
    assert a != c


def test_adding_a_sensor_does_not_change_other_sensors():
    alone = [r for r in run(make_sim([COLD_TEMP], quiet=False), 500)]
    together = [
        r for r in run(make_sim([COLD_TEMP, HUMIDITY], quiet=False), 500) if r["sensor_id"] == "t1"
    ]
    assert alone == together


def test_unseeded_runs_differ():
    a = run(make_sim(seed=None), 50)
    b = run(make_sim(seed=None), 50)
    assert [r["value"] for r in a] != [r["value"] for r in b]


# --------------------------------------------------------------------------- #
# Normal behaviour
# --------------------------------------------------------------------------- #
def test_normal_readings_stay_in_range_and_unlabelled():
    sim = make_sim([COLD_TEMP, HUMIDITY], start=NOON)
    for reading in run(sim, 5000):
        low, high = sim.get_sensor(reading["sensor_id"]).acceptable_range
        assert low <= reading["value"] <= high
        assert reading["is_anomaly"] is False
        assert reading["anomaly_type"] is None


def test_normal_readings_are_gaussian_around_the_mean_at_night():
    values = [r["value"] for r in run(make_sim([HUMIDITY], seed=3), 8000)]
    assert statistics.mean(values) == pytest.approx(45.0, abs=0.15)
    assert statistics.stdev(values) == pytest.approx(2.5, rel=0.1)


def test_business_hours_are_slightly_warmer_than_night():
    night = [r["value"] for r in run(make_sim([HUMIDITY], start=NIGHT, seed=4), 4000)]
    peak = datetime(2026, 3, 10, 14, 0, tzinfo=UTC)
    day = [r["value"] for r in run(make_sim([HUMIDITY], start=peak, seed=4), 4000)]
    assert statistics.mean(day) > statistics.mean(night) + 0.5


def test_zero_rates_produce_no_anomalies_over_a_day():
    sim = make_sim([COLD_TEMP], start=datetime(2026, 3, 10, 0, 0, tzinfo=UTC))
    assert not any(r["is_anomaly"] for r in run(sim, 86_400))


def test_battery_drains_slowly_and_never_goes_negative():
    low_battery = sensor(sensor_id="b1", battery_level=0.0005)
    levels = [r["metadata"]["battery_level"] for r in run(make_sim([COLD_TEMP]), 100)]
    assert levels == sorted(levels, reverse=True)
    assert levels[0] > levels[-1] > 99.98
    drained = [r["metadata"]["battery_level"] for r in run(make_sim([low_battery]), 20)]
    assert drained[-1] == 0.0
    assert min(drained) >= 0.0


# --------------------------------------------------------------------------- #
# Point anomalies
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [COLD_TEMP, HUMIDITY])
@pytest.mark.parametrize("direction", [1, -1])
def test_point_anomaly_leaves_acceptable_range_for_requested_duration(target, direction):
    sim = make_sim([target])
    episode_id = sim.inject_point_anomaly(target.sensor_id, duration=4, direction=direction)
    readings = run(sim, 8)
    low, high = target.acceptable_range
    anomalous, normal = readings[:4], readings[4:]
    for step, reading in enumerate(anomalous):
        assert reading["is_anomaly"] and reading["anomaly_type"] == "point"
        assert reading["metadata"]["episode_id"] == episode_id
        assert reading["metadata"]["episode_step"] == step
        assert reading["value"] > high if direction > 0 else reading["value"] < low
    assert all(not r["is_anomaly"] and low <= r["value"] <= high for r in normal)


def test_point_anomaly_random_direction_and_duration():
    sim = make_sim()
    directions = set()
    for seed_offset in range(40):
        s = make_sim(seed=seed_offset)
        s.inject_point_anomaly("t1")
        readings = [r for r in run(s, 10) if r["is_anomaly"]]
        assert 1 <= len(readings) <= 5
        directions.add(readings[0]["value"] > 4.0)
    assert directions == {True, False}
    del sim


def test_point_anomaly_respects_physical_bounds():
    near_top = sensor(
        sensor_id="p1",
        sensor_type="humidity",
        normal_mean=90.0,
        normal_std=2.0,
        acceptable_range=(80.0, 99.0),
    )
    for seed in range(30):
        sim = make_sim([near_top], seed=seed)
        sim.inject_point_anomaly("p1", duration=3)
        for reading in run(sim, 3):
            assert 0.0 <= reading["value"] <= 100.0
            assert reading["value"] < 80.0 or reading["value"] > 99.0


def test_point_anomaly_clamped_at_physical_ceiling():
    near_top = sensor(
        sensor_id="p1",
        sensor_type="humidity",
        normal_mean=90.0,
        normal_std=2.0,
        acceptable_range=(80.0, 99.0),
    )
    sim = make_sim([near_top])
    sim.inject_point_anomaly("p1", duration=5, direction=1)
    assert all(99.0 < r["value"] <= 100.0 for r in run(sim, 5))


def test_point_anomaly_impossible_direction_rejected():
    near_top = sensor(
        sensor_id="p1",
        sensor_type="humidity",
        normal_mean=90.0,
        normal_std=2.0,
        acceptable_range=(80.0, 99.9),
    )
    with pytest.raises(ValueError, match="physical bounds"):
        make_sim([near_top]).inject_point_anomaly("p1", direction=1)
    sim = make_sim([near_top])
    sim.inject_point_anomaly("p1", duration=1)  # random direction must pick the feasible side
    assert run(sim, 1)[0]["value"] < 80.0


@pytest.mark.parametrize(
    "bad", [{"duration": 0}, {"duration": -2}, {"direction": 0}, {"direction": 2}]
)
def test_point_anomaly_argument_validation(bad):
    with pytest.raises(ValueError):
        make_sim().inject_point_anomaly("t1", **bad)


# --------------------------------------------------------------------------- #
# Contextual anomalies
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", [1, -1])
def test_contextual_anomaly_is_in_range_but_shifted_off_hours(direction):
    sim = make_sim([HUMIDITY], seed=5)
    episode_id = sim.inject_contextual_anomaly("h1", duration=300, direction=direction)
    readings = run(sim, 300)
    low, high = HUMIDITY.acceptable_range
    assert all(r["anomaly_type"] == "contextual" for r in readings)
    assert all(r["metadata"]["episode_id"] == episode_id for r in readings)
    assert all(low < r["value"] < high for r in readings)  # not catchable by a range check
    shift = statistics.mean(r["value"] for r in readings) - HUMIDITY.normal_mean
    assert shift * direction > 1.0  # clearly displaced for the quiet hours
    baseline = [r["value"] for r in run(make_sim([HUMIDITY], seed=6), 300)]
    assert statistics.stdev(r["value"] for r in readings) > statistics.stdev(baseline) * 0.9


def test_contextual_anomaly_requires_off_hours():
    sim = make_sim(start=NOON)
    with pytest.raises(ValueError, match="off-hours"):
        sim.inject_contextual_anomaly("t1")
    no_window = make_sim(off_hours_start=2, off_hours_end=2)
    with pytest.raises(ValueError, match="off-hours"):
        no_window.inject_contextual_anomaly("t1")


def test_contextual_anomaly_stops_when_off_hours_window_closes():
    start = datetime(2026, 3, 10, 4, 59, 57, tzinfo=UTC)
    sim = make_sim(start=start)
    sim.inject_contextual_anomaly("t1", duration=20)
    flags = [r["anomaly_type"] for r in run(sim, 8)]
    assert flags == ["contextual"] * 3 + [None] * 5  # 04:59:57, :58, :59 then business hours


def test_contextual_anomaly_works_across_midnight_window():
    start = datetime(2026, 3, 10, 23, 59, 58, tzinfo=UTC)
    sim = make_sim(start=start, off_hours_start=22, off_hours_end=6)
    sim.inject_contextual_anomaly("t1", duration=6)
    assert [r["anomaly_type"] for r in run(sim, 6)] == ["contextual"] * 6


def test_contextual_anomaly_argument_validation():
    with pytest.raises(ValueError):
        make_sim().inject_contextual_anomaly("t1", duration=0)
    with pytest.raises(ValueError):
        make_sim().inject_contextual_anomaly("t1", direction=3)


# --------------------------------------------------------------------------- #
# Collective anomalies
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", [1, -1])
def test_collective_anomaly_drifts_gradually(direction):
    sim = make_sim([HUMIDITY], seed=8)
    episode_id = sim.inject_collective_anomaly("h1", length=30, direction=direction)
    readings = run(sim, 40)
    drift, after = readings[:30], readings[30:]
    assert all(r["anomaly_type"] == "collective" for r in drift)
    assert all(r["metadata"]["episode_id"] == episode_id for r in drift)
    assert all(r["anomaly_type"] is None for r in after)
    values = [r["value"] for r in drift]
    first, last = statistics.mean(values[:10]), statistics.mean(values[-10:])
    assert (last - first) * direction > 3.0
    assert abs(statistics.mean(r["value"] for r in after) - 45.0) < 2.5  # back to normal


def test_collective_drift_is_monotone_on_average():
    sim = make_sim([HUMIDITY], seed=9)
    sim.inject_collective_anomaly("h1", length=30, direction=1)
    values = [r["value"] for r in run(sim, 30)]
    thirds = [statistics.mean(values[i : i + 10]) for i in (0, 10, 20)]
    assert thirds[0] < thirds[1] < thirds[2]


def test_collective_anomaly_length_and_default_direction():
    ups = 0
    for seed in range(60):
        sim = make_sim([COLD_TEMP], seed=seed)
        sim.inject_collective_anomaly("t1")
        labelled = [r for r in run(sim, 40) if r["is_anomaly"]]
        assert 10 <= len(labelled) <= 30
        ups += labelled[-1]["value"] > labelled[0]["value"]
    assert ups > 35  # temperature degradation mostly warms up
    humidity_ups = 0
    for seed in range(60):
        sim = make_sim([HUMIDITY], seed=seed)
        sim.inject_collective_anomaly("h1", length=30)
        values = [r["value"] for r in run(sim, 30)]
        humidity_ups += values[-1] > values[0]
    assert 10 < humidity_ups < 50  # both directions occur


def test_collective_anomaly_clamped_to_physical_range():
    wet = sensor(
        sensor_id="w1",
        sensor_type="humidity",
        normal_mean=90.0,
        normal_std=5.0,
        acceptable_range=(70.0, 99.0),
    )
    sim = make_sim([wet], seed=1)
    sim.inject_collective_anomaly("w1", length=30, direction=1)
    assert all(0.0 <= r["value"] <= 100.0 for r in run(sim, 30))


def test_collective_anomaly_argument_validation():
    with pytest.raises(ValueError):
        make_sim().inject_collective_anomaly("t1", length=0)
    with pytest.raises(ValueError):
        make_sim().inject_collective_anomaly("t1", direction=-3)


# --------------------------------------------------------------------------- #
# Episode semantics
# --------------------------------------------------------------------------- #
def test_manual_injection_replaces_running_episode():
    sim = make_sim()
    first = sim.inject_collective_anomaly("t1", length=20)
    run(sim, 3)
    second = sim.inject_point_anomaly("t1", duration=2)
    readings = run(sim, 3)
    assert first != second
    assert [r["anomaly_type"] for r in readings] == ["point", "point", None]
    assert readings[0]["metadata"]["episode_id"] == second


def test_episode_ids_are_unique_and_sequential_per_sensor():
    sim = make_sim([COLD_TEMP, HUMIDITY])
    ids = [
        sim.inject_point_anomaly("t1", duration=1),
        sim.inject_point_anomaly("t1", duration=1),
        sim.inject_collective_anomaly("h1", length=1),
    ]
    assert ids == ["t1#1", "t1#2", "h1#1"]


def test_injection_only_affects_target_sensor():
    sim = make_sim([COLD_TEMP, HUMIDITY])
    sim.inject_point_anomaly("t1", duration=3)
    readings = run(sim, 3)
    assert {r["sensor_id"] for r in readings if r["is_anomaly"]} == {"t1"}


def test_logging_does_not_break_injection(capsys):
    import logging

    from src.utils.logger import configure_logging

    configure_logging(level="DEBUG", fmt="json")
    sim = make_sim()
    sim.inject_point_anomaly("t1", duration=1)
    logging.getLogger().handlers.clear()
    assert "anomaly_episode_started" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Random injection at configured rates
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full_day():
    """One simulated day for six sensors with the default anomaly rates."""
    fleet = build_default_sensors(n_warehouses=1, reference_date=REF_DATE)
    sim = WarehouseSimulator(
        fleet, start_time=datetime(2026, 3, 10, 0, 0, tzinfo=UTC), seed=42, **{}
    )
    return sim, list(sim.simulate_all_sensors(duration_s=86_400))


def test_default_rates_hit_targets_over_a_day(full_day):
    _, readings = full_day
    total = len(readings)
    share = {
        kind: sum(r["anomaly_type"] == kind for r in readings) / total for kind in ANOMALY_TYPES
    }
    assert share["point"] == pytest.approx(0.02, rel=0.2)
    assert share["contextual"] == pytest.approx(0.01, rel=0.25)
    assert share["collective"] == pytest.approx(0.005, rel=0.3)


def test_labels_are_consistent(full_day):
    _, readings = full_day
    for r in readings:
        assert r["is_anomaly"] == (r["anomaly_type"] is not None)
        assert (r["metadata"]["episode_id"] is None) == (not r["is_anomaly"])
        assert r["anomaly_type"] in (None, *ANOMALY_TYPES)


def test_anomaly_signatures_in_random_stream(full_day):
    sim, readings = full_day
    for r in readings:
        low, high = sim.get_sensor(r["sensor_id"]).acceptable_range
        if r["anomaly_type"] == "point":
            assert r["value"] < low or r["value"] > high
        elif r["anomaly_type"] == "contextual":
            assert low < r["value"] < high
            assert datetime.fromisoformat(r["timestamp"]).hour < 5
        elif r["anomaly_type"] is None:
            assert low <= r["value"] <= high


def test_episode_lengths_follow_configured_ranges(full_day):
    _, readings = full_day
    episodes = defaultdict(list)
    for r in readings:
        if r["is_anomaly"]:
            episodes[r["metadata"]["episode_id"]].append(r)
    bounds = {"point": (1, 5), "contextual": (1, 10), "collective": (10, 30)}
    seen = set()
    for steps in episodes.values():
        kind = steps[0]["anomaly_type"]
        seen.add(kind)
        assert {s["anomaly_type"] for s in steps} == {kind}
        assert [s["metadata"]["episode_step"] for s in steps] == list(range(len(steps)))
        low, high = bounds[kind]
        # an episode cut off by the end of the run or the quiet window may be short
        assert len(steps) <= high
        if len(steps) < low:
            last = steps[-1]["timestamp"]
            assert kind == "contextual" or last.startswith("2026-03-10T23:59:59")
    assert seen == set(ANOMALY_TYPES)


def test_contextual_anomalies_only_appear_in_off_hours_even_with_custom_window():
    sim = make_sim(
        [COLD_TEMP, HUMIDITY],
        start=datetime(2026, 3, 10, 0, 0, tzinfo=UTC),
        quiet=False,
        contextual_anomaly_rate=0.02,
        off_hours_start=20,
        off_hours_end=23,
    )
    readings = run(sim, 86_400)
    hours = {
        datetime.fromisoformat(r["timestamp"]).hour
        for r in readings
        if r["anomaly_type"] == "contextual"
    }
    assert hours and hours <= {20, 21, 22}


def test_contextual_rate_with_no_window_is_ignored():
    sim = make_sim(
        start=datetime(2026, 3, 10, 0, 0, tzinfo=UTC),
        quiet=False,
        contextual_anomaly_rate=0.5,
        point_anomaly_rate=0.0,
        collective_anomaly_rate=0.0,
        off_hours_start=4,
        off_hours_end=4,
    )
    assert not any(r["is_anomaly"] for r in run(sim, 5_000))


# --------------------------------------------------------------------------- #
# Config integration
# --------------------------------------------------------------------------- #
def test_from_config_uses_project_config():
    cfg = load_config(environ={})
    sim = WarehouseSimulator.from_config(cfg.simulation.warehouse, seed=cfg.simulation.seed)
    wh = cfg.simulation.warehouse
    assert len(sim.sensors) == wh.n_warehouses * len(wh.zones) * wh.sensors_per_zone
    assert sim.reading_interval_s == wh.reading_interval_s
    assert len(run(sim, 3)) == 3 * len(sim.sensors)


def test_from_config_with_custom_sensors_and_start_time():
    cfg = load_config(environ={})
    sim = WarehouseSimulator.from_config(
        cfg.simulation.warehouse, seed=1, start_time=NIGHT, sensors=[COLD_TEMP]
    )
    assert sim.sensors == (COLD_TEMP,)
    assert sim.current_time == NIGHT


def test_from_config_matches_direct_construction():
    cfg = load_config(environ={}).simulation.warehouse
    a = WarehouseSimulator.from_config(cfg, seed=5, start_time=NIGHT)
    fleet = build_default_sensors(
        cfg.n_warehouses, cfg.zones, cfg.sensors_per_zone, seed=5, reference_date=NIGHT.date()
    )
    b = WarehouseSimulator(
        fleet,
        point_anomaly_rate=cfg.point_anomaly_rate,
        contextual_anomaly_rate=cfg.contextual_anomaly_rate,
        collective_anomaly_rate=cfg.collective_anomaly_rate,
        point_duration_range=cfg.point_duration_range,
        collective_length_range=cfg.collective_length_range,
        off_hours_start=cfg.off_hours_start,
        off_hours_end=cfg.off_hours_end,
        reading_interval_s=cfg.reading_interval_s,
        start_time=NIGHT,
        seed=5,
    )
    assert run(a, 300) == run(b, 300)


def test_package_exports():
    from src import data_simulation

    assert data_simulation.WarehouseSimulator is ws.WarehouseSimulator
    assert set(data_simulation.__all__) == {
        "WarehouseReading",
        "WarehouseSensor",
        "WarehouseSimulator",
        "build_default_sensors",
    }


def test_sample_within_falls_back_to_clamped_mean():
    import random

    rng = random.Random(0)
    # a distribution that essentially never lands inside the window forces the fallback
    assert ws._sample_within(rng, 100.0, 0.001, 0.0, 10.0) == 10.0
