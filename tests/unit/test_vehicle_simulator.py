import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from src.data_simulation.vehicle_simulator import (
    ANOMALY_SUBTYPES,
    EXCESSIVE_SPEED,
    GRADUAL_DELAY,
    ROUTE_DEVIATION,
    SUDDEN_STOP,
    GPSSimulator,
    VehicleTracker,
    build_default_vehicles,
)
from src.utils.config import VehicleSimConfig

START = datetime(2026, 3, 10, 8, 0, 0, tzinfo=UTC)
QUIET = {"anomaly_rate": 0.0}
LONG_ROUTE = ((51.9244, 4.4777), (51.5719, 4.7683), (51.2194, 4.4025), (50.8503, 4.3517))
# A short out-and-back route (~3 km) so a vehicle turns around within a few ticks.
SHORT_ROUTE = ((51.0, 4.0), (51.0, 4.02), (51.015, 4.04))


def tracker(**overrides):
    fields = {
        "vehicle_id": "v1",
        "driver_id": "d1",
        "route_id": "r1",
        "origin": "A",
        "destination": "B",
        "waypoints": LONG_ROUTE,
        "cruise_speed_kmh": 65.0,
    }
    return VehicleTracker(**{**fields, **overrides})


def make_sim(vehicles=None, *, quiet=True, seed=7, **kwargs):
    if quiet:
        kwargs = {**QUIET, **kwargs}
    return GPSSimulator(
        vehicles if vehicles is not None else [tracker()], start_time=START, seed=seed, **kwargs
    )


def run(sim, ticks):
    return list(sim.simulate_all_vehicles(n_ticks=ticks))


def dense_route_distance_m(waypoints, lat, lon):
    """Independent check: haversine distance to a densely sampled route."""

    def haversine(a, b):
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dphi, dlam = p2 - p1, math.radians(b[1] - a[1])
        h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        return 2 * 6_371_008.8 * math.asin(math.sqrt(h))

    best = math.inf
    for (la0, lo0), (la1, lo1) in zip(waypoints, waypoints[1:], strict=False):
        for i in range(401):
            f = i / 400
            best = min(best, haversine((la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0)), (lat, lon)))
    return best


# --------------------------------------------------------------------------- #
# VehicleTracker
# --------------------------------------------------------------------------- #
def test_tracker_normalises_waypoints_and_is_frozen():
    t = tracker(waypoints=[[51.0, 4.0], [51.1, 4.1]])
    assert t.waypoints == ((51.0, 4.0), (51.1, 4.1))
    with pytest.raises(AttributeError):
        t.vehicle_id = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"vehicle_id": ""},
        {"driver_id": ""},
        {"route_id": ""},
        {"origin": ""},
        {"destination": ""},
        {"waypoints": ((51.0, 4.0),)},
        {"waypoints": ((95.0, 4.0), (51.0, 4.0))},
        {"waypoints": ((51.0, 4.0), (51.0, 200.0))},
        {"waypoints": ((51.0, 4.0), (51.0, 4.0), (51.1, 4.1))},  # repeated waypoint
        {"waypoints": ((51.0, 4.0), (51.001, 4.001))},  # route shorter than 1 km
        {"cruise_speed_kmh": 10.0},
        {"cruise_speed_kmh": 150.0},
    ],
)
def test_tracker_validation(overrides):
    with pytest.raises(ValueError):
        tracker(**overrides)


# --------------------------------------------------------------------------- #
# Default fleet
# --------------------------------------------------------------------------- #
def test_default_fleet_shape_and_ids():
    fleet = build_default_vehicles(7, seed=1)
    assert [v.vehicle_id for v in fleet][:2] == ["veh-001", "veh-002"]
    assert len({v.vehicle_id for v in fleet}) == 7
    assert fleet[5].route_id == fleet[0].route_id  # routes cycle after five
    assert all(55.0 <= v.cruise_speed_kmh <= 80.0 for v in fleet)


def test_default_fleet_deterministic_and_validated():
    assert build_default_vehicles(5, seed=3) == build_default_vehicles(5, seed=3)
    with pytest.raises(ValueError):
        build_default_vehicles(0)


# --------------------------------------------------------------------------- #
# Normal behaviour
# --------------------------------------------------------------------------- #
def test_reading_schema_and_json_serialisable():
    reading = run(make_sim(), 1)[0]
    assert set(reading) == {
        "vehicle_id",
        "timestamp",
        "latitude",
        "longitude",
        "speed_kmh",
        "heading",
        "is_anomaly",
        "anomaly_type",
        "route_adherence_score",
        "metadata",
    }
    assert reading["vehicle_id"] == "v1"
    assert reading["timestamp"] == "2026-03-10T08:00:00Z"
    assert reading["is_anomaly"] is False
    assert reading["anomaly_type"] is None
    json.dumps(reading)


def test_one_reading_per_vehicle_every_ten_seconds():
    sim = make_sim(build_default_vehicles(3, seed=1))
    readings = run(sim, 4)
    assert len(readings) == 12
    stamps = sorted({r["timestamp"] for r in readings})
    assert stamps == [
        "2026-03-10T08:00:00Z",
        "2026-03-10T08:00:10Z",
        "2026-03-10T08:00:20Z",
        "2026-03-10T08:00:30Z",
    ]
    assert sim.current_time == START + timedelta(seconds=40)
    assert sim.reading_interval_s == 10.0


def test_normal_readings_stay_on_route_and_in_speed_band():
    sim = make_sim()
    for r in run(sim, 3000):
        assert not r["is_anomaly"]
        assert 0.6 * 65.0 <= r["speed_kmh"] <= 1.2 * 65.0
        assert r["metadata"]["distance_from_route_m"] < 30.0
        assert r["route_adherence_score"] > 0.99
        assert 0.0 <= r["heading"] < 360.0
        assert sim.distance_from_route_m("v1", r["latitude"], r["longitude"]) < 30.0


def test_vehicle_progresses_along_route_at_roughly_cruise_speed():
    readings = run(make_sim(), 30)
    assert all(a["metadata"]["trip"] == readings[0]["metadata"]["trip"] for a in readings[:5])
    moved = {r["metadata"]["progress_m"] for r in readings}
    assert len(moved) > 25  # position keeps changing


def test_vehicle_turns_around_at_the_end_of_the_route():
    sim = make_sim([tracker(waypoints=SHORT_ROUTE, cruise_speed_kmh=90.0)])
    readings = run(sim, 120)
    assert {r["metadata"]["trip"] for r in readings} >= {1, 2, 3}
    by_trip: dict[int, list[float]] = {}
    for r in readings:
        by_trip.setdefault(r["metadata"]["trip"], []).append(r["heading"])
    # Consecutive legs travel in opposite directions (headings ~180 degrees apart).
    mean = {trip: sum(h) / len(h) for trip, h in by_trip.items()}
    for trip in (2, 3):
        assert 120.0 < abs(mean[trip] - mean[trip - 1]) < 240.0


def test_deterministic_for_a_seed_and_different_across_seeds():
    a = run(make_sim(quiet=False, seed=5), 300)
    b = run(make_sim(quiet=False, seed=5), 300)
    c = run(make_sim(quiet=False, seed=6), 300)
    assert a == b
    assert a != c


def test_adding_a_vehicle_does_not_change_the_others():
    fleet = build_default_vehicles(3, seed=1)
    solo = [r for r in run(make_sim(fleet[:2], quiet=False), 200) if r["vehicle_id"] == "veh-001"]
    trio = [r for r in run(make_sim(fleet, quiet=False), 200) if r["vehicle_id"] == "veh-001"]
    assert solo == trio


def test_zero_gps_noise_puts_vehicle_exactly_on_route():
    sim = make_sim(gps_noise_m=0.0)
    assert all(r["metadata"]["distance_from_route_m"] < 0.5 for r in run(sim, 50))


# --------------------------------------------------------------------------- #
# Route deviation (point)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", [1, -1])
def test_route_deviation_leaves_the_corridor(side):
    sim = make_sim()
    run(sim, 3)
    episode_id = sim.inject_route_deviation("v1", duration=4, side=side)
    readings = run(sim, 8)
    flagged = [r for r in readings if r["is_anomaly"]]
    assert len(flagged) == 4
    assert all(r["is_anomaly"] for r in readings[:4])
    assert not any(r["is_anomaly"] for r in readings[4:])
    for r in flagged:
        meta = r["metadata"]
        assert r["anomaly_type"] == "point"
        assert meta["anomaly_subtype"] == ROUTE_DEVIATION
        assert meta["episode_id"] == episode_id
        assert meta["distance_from_route_m"] > 500.0
        assert r["route_adherence_score"] < 0.25
        # Independent geometry (haversine, dense sampling) agrees it is off-corridor.
        assert dense_route_distance_m(LONG_ROUTE, r["latitude"], r["longitude"]) > 500.0
    assert [r["metadata"]["episode_step"] for r in flagged] == [0, 1, 2, 3]
    back_on = readings[4]
    assert back_on["metadata"]["distance_from_route_m"] < 30.0


def test_route_deviation_sides_are_opposite():
    left, right = make_sim(seed=1), make_sim(seed=1)
    left.inject_route_deviation("v1", duration=1, side=1)
    right.inject_route_deviation("v1", duration=1, side=-1)
    a, b = run(left, 1)[0], run(right, 1)[0]
    # Same seed => same on-route position; the two offsets are mirror images about it,
    # so their midpoint is back on the route and they are >1 km apart.
    mid_lat, mid_lon = (a["latitude"] + b["latitude"]) / 2, (a["longitude"] + b["longitude"]) / 2
    assert left.distance_from_route_m("v1", mid_lat, mid_lon) < 30.0
    assert a["metadata"]["distance_from_route_m"] > 500.0
    assert b["metadata"]["distance_from_route_m"] > 500.0
    assert (a["latitude"], a["longitude"]) != (b["latitude"], b["longitude"])


def test_route_deviation_label_matches_geometry_near_sharp_corner():
    # A hairpin route makes naive perpendicular offsets land close to another segment.
    hairpin = ((51.0, 4.0), (51.05, 4.0), (51.0, 4.003))
    sim = make_sim([tracker(waypoints=hairpin, cruise_speed_kmh=30.0)], seed=2)
    for _ in range(40):
        sim.inject_route_deviation("v1", duration=2)
        for r in run(sim, 2):
            assert sim.distance_from_route_m("v1", r["latitude"], r["longitude"]) > 500.0


# --------------------------------------------------------------------------- #
# Speed anomalies (point)
# --------------------------------------------------------------------------- #
def test_sudden_stop_halts_the_vehicle():
    sim = make_sim()
    run(sim, 2)
    sim.inject_speed_anomaly("v1", duration=3, mode="stop")
    before = run(sim, 1)  # still inside the 3-reading episode
    stopped = before + run(sim, 2)
    assert all(r["is_anomaly"] and r["anomaly_type"] == "point" for r in stopped)
    assert all(r["metadata"]["anomaly_subtype"] == SUDDEN_STOP for r in stopped)
    assert all(r["speed_kmh"] <= 2.0 for r in stopped)
    positions = [r["metadata"]["progress_m"] for r in stopped]
    assert max(positions) - min(positions) < 10.0
    assert run(sim, 1)[0]["speed_kmh"] >= 0.6 * 65.0  # moving again


def test_excessive_speed_is_far_above_normal():
    sim = make_sim()
    sim.inject_speed_anomaly("v1", duration=5, mode="excessive")
    readings = run(sim, 5)
    assert all(r["metadata"]["anomaly_subtype"] == EXCESSIVE_SPEED for r in readings)
    assert all(r["speed_kmh"] >= 1.45 * 65.0 for r in readings)
    assert all(r["metadata"]["distance_from_route_m"] < 30.0 for r in readings)


def test_speed_anomaly_random_mode_covers_both_kinds():
    sim = make_sim()
    kinds = set()
    for _ in range(40):
        sim.inject_speed_anomaly("v1", duration=1)
        kinds.add(run(sim, 1)[0]["metadata"]["anomaly_subtype"])
    assert kinds == {SUDDEN_STOP, EXCESSIVE_SPEED}


# --------------------------------------------------------------------------- #
# Gradual delay (collective)
# --------------------------------------------------------------------------- #
def test_gradual_delay_is_collective_and_slows_progressively():
    sim = make_sim(seed=3)
    sim.inject_gradual_delay("v1", length=24)
    readings = run(sim, 30)
    delayed = readings[:24]
    assert all(r["anomaly_type"] == "collective" for r in delayed)
    assert all(r["metadata"]["anomaly_subtype"] == GRADUAL_DELAY for r in delayed)
    assert not any(r["is_anomaly"] for r in readings[24:])
    third = 8
    early = sum(r["speed_kmh"] for r in delayed[:third]) / third
    late = sum(r["speed_kmh"] for r in delayed[-third:]) / third
    assert late < 0.75 * early
    # The first reading is still indistinguishable from normal driving.
    assert delayed[0]["speed_kmh"] >= 0.6 * 65.0
    assert all(r["metadata"]["distance_from_route_m"] < 30.0 for r in delayed)


def test_gradual_delay_makes_the_vehicle_fall_behind():
    slow, normal = make_sim(seed=4), make_sim(seed=4)
    slow.inject_gradual_delay("v1", length=30)
    run(slow, 40)
    run(normal, 40)
    assert slow._state("v1").travelled < normal._state("v1").travelled


# --------------------------------------------------------------------------- #
# Injection plumbing
# --------------------------------------------------------------------------- #
def test_injection_replaces_a_running_episode_and_returns_unique_ids():
    sim = make_sim()
    first = sim.inject_gradual_delay("v1", length=20)
    run(sim, 2)
    second = sim.inject_speed_anomaly("v1", duration=2, mode="stop")
    assert first != second
    readings = run(sim, 2)
    assert {r["metadata"]["episode_id"] for r in readings} == {second}


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.inject_route_deviation("nope"),
        lambda s: s.inject_speed_anomaly("nope"),
        lambda s: s.inject_gradual_delay("nope"),
        lambda s: s.get_vehicle("nope"),
        lambda s: s.distance_from_route_m("nope", 0.0, 0.0),
    ],
)
def test_unknown_vehicle_raises_key_error(call):
    with pytest.raises(KeyError):
        call(make_sim())


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.inject_route_deviation("v1", duration=0),
        lambda s: s.inject_route_deviation("v1", side=2),
        lambda s: s.inject_speed_anomaly("v1", duration=-1),
        lambda s: s.inject_speed_anomaly("v1", mode="fly"),
        lambda s: s.inject_gradual_delay("v1", length=0),
    ],
)
def test_injection_argument_validation(call):
    with pytest.raises(ValueError):
        call(make_sim())


# --------------------------------------------------------------------------- #
# Random anomalies and ground truth
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def noisy_run():
    sim = GPSSimulator(build_default_vehicles(5, seed=1), start_time=START, seed=42)
    return sim, run(sim, 20_000)


def test_anomaly_rate_close_to_configured(noisy_run):
    _, readings = noisy_run
    rate = sum(r["is_anomaly"] for r in readings) / len(readings)
    assert 0.0165 <= rate <= 0.0235  # target 2 % (realised ~ rate / (1 + rate))


def test_anomaly_mix_roughly_matches_configuration(noisy_run):
    _, readings = noisy_run
    counts = Counter(r["metadata"]["anomaly_subtype"] for r in readings if r["is_anomaly"])
    assert set(counts) == set(ANOMALY_SUBTYPES)
    total = sum(counts.values())
    assert 0.30 <= counts[ROUTE_DEVIATION] / total <= 0.50
    assert 0.20 <= (counts[SUDDEN_STOP] + counts[EXCESSIVE_SPEED]) / total <= 0.40
    assert 0.20 <= counts[GRADUAL_DELAY] / total <= 0.40


def test_labels_are_exact_ground_truth(noisy_run):
    sim, readings = noisy_run
    for r in readings:
        meta = r["metadata"]
        assert r["is_anomaly"] == (r["anomaly_type"] is not None)
        assert (meta["anomaly_subtype"] is None) == (not r["is_anomaly"])
        deviating = meta["anomaly_subtype"] == ROUTE_DEVIATION
        off_route = sim.distance_from_route_m(r["vehicle_id"], r["latitude"], r["longitude"])
        # is_anomaly for deviations is exactly "outside the 500 m corridor".
        assert (off_route > 500.0) == deviating
        if meta["anomaly_subtype"] in (SUDDEN_STOP, EXCESSIVE_SPEED):
            cruise = meta["cruise_speed_kmh"]
            assert not 0.6 * cruise <= r["speed_kmh"] <= 1.2 * cruise
        if not r["is_anomaly"]:
            cruise = meta["cruise_speed_kmh"]
            assert 0.6 * cruise <= r["speed_kmh"] <= 1.2 * cruise


def test_episode_lengths_respect_configured_ranges(noisy_run):
    _, readings = noisy_run
    lengths: dict[str, int] = Counter()
    subtype: dict[str, str] = {}
    for r in readings:
        eid = r["metadata"]["episode_id"]
        if eid:
            lengths[eid] += 1
            subtype[eid] = r["metadata"]["anomaly_subtype"]
    bounds = {
        ROUTE_DEVIATION: (3, 12),
        SUDDEN_STOP: (2, 6),
        EXCESSIVE_SPEED: (2, 6),
        GRADUAL_DELAY: (12, 30),
    }
    for eid, n in lengths.items():
        low, high = bounds[subtype[eid]]
        assert low <= n <= high


def test_zero_anomaly_rate_produces_no_anomalies():
    assert not any(r["is_anomaly"] for r in run(make_sim(), 2000))


# --------------------------------------------------------------------------- #
# Construction and streaming arguments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"deviation_threshold_m": 0},
        {"anomaly_rate": 1.5},
        {"anomaly_rate": -0.1},
        {"anomaly_mix": (0.5, 0.5)},
        {"anomaly_mix": (0.5, 0.5, 0.5)},
        {"anomaly_mix": (-0.2, 0.7, 0.5)},
        {"deviation_length_range": (0, 3)},
        {"speed_length_range": (5, 2)},
        {"delay_length_range": (0, 0)},
        {"gps_noise_m": -1.0},
        {"reading_interval_s": 0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        GPSSimulator([tracker()], start_time=START, **kwargs)


def test_empty_and_duplicate_fleets_rejected():
    with pytest.raises(ValueError):
        GPSSimulator([], start_time=START)
    with pytest.raises(ValueError, match="duplicate"):
        GPSSimulator([tracker(), tracker()], start_time=START)


def test_naive_start_time_is_treated_as_utc_and_default_is_now():
    sim = GPSSimulator([tracker()], start_time=datetime(2026, 1, 1, 12, 0, 0), anomaly_rate=0.0)
    assert sim.current_time == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    now = GPSSimulator([tracker()]).current_time
    assert abs((datetime.now(UTC) - now).total_seconds()) < 5


def test_simulate_by_duration_and_argument_errors():
    sim = make_sim(build_default_vehicles(2, seed=1))
    assert len(list(sim.simulate_all_vehicles(duration_s=35))) == 2 * 4  # rounds up to 4 ticks
    assert list(sim.simulate_all_vehicles(n_ticks=0)) == []
    with pytest.raises(ValueError):
        list(sim.simulate_all_vehicles(n_ticks=1, duration_s=10))
    with pytest.raises(ValueError):
        list(sim.simulate_all_vehicles(duration_s=-1))
    with pytest.raises(ValueError):
        list(sim.simulate_all_vehicles(n_ticks=-1))


def test_endless_stream_can_be_sliced():
    from itertools import islice

    sim = make_sim()
    assert len(list(islice(sim.simulate_all_vehicles(), 25))) == 25


def test_from_config_uses_config_values():
    cfg = VehicleSimConfig(n_vehicles=3, reading_interval_s=5.0, route_deviation_m=300.0)
    sim = GPSSimulator.from_config(cfg, seed=1, start_time=START)
    assert len(sim.vehicles) == 3
    assert sim.reading_interval_s == 5.0
    assert sim.deviation_threshold_m == 300.0
    custom = [tracker()]
    assert GPSSimulator.from_config(cfg, vehicles=custom, start_time=START).vehicles == tuple(
        custom
    )


def test_get_vehicle_returns_definition():
    sim = make_sim()
    assert sim.get_vehicle("v1").route_id == "r1"
    assert sim.vehicles[0].vehicle_id == "v1"
