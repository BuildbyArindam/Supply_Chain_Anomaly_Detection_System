import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from itertools import islice

import pytest

import src.data_simulation as data_simulation
from src.data_simulation import (
    GPSSimulator,
    InventorySimulator,
    OrderEventSimulator,
    SimulationOrchestrator,
    WarehouseSimulator,
)
from src.data_simulation.orchestrator import SOURCE_TYPES
from src.utils.config import SimulationConfig, load_config

START = datetime(2026, 3, 10, 8, 0, 0, tzinfo=UTC)


class FakeWallClock:
    """Deterministic stand-in for time.monotonic / time.sleep."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make(*, orders=False, seed=42, **kwargs):
    return SimulationOrchestrator.from_config(
        SimulationConfig(seed=seed), start_time=START, include_orders=orders, **kwargs
    )


def parse(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# Stream contents
# --------------------------------------------------------------------------- #
def test_envelope_schema_and_source_tags():
    events = make().run_batch(60)
    assert {e["source_type"] for e in events} == {"warehouse", "vehicle", "inventory"}
    id_keys = {"warehouse": "sensor_id", "vehicle": "vehicle_id", "inventory": "sku_id"}
    for e in events:
        assert set(e) == {"source_type", "entity_id", "timestamp", "payload"}
        assert e["entity_id"] == e["payload"][id_keys[e["source_type"]]]
        assert e["timestamp"] == e["payload"]["timestamp"]
        assert "is_anomaly" in e["payload"]
    json.dumps(events[:50])


def test_stream_is_in_strict_timestamp_order_across_sources():
    events = make(orders=True).run_batch(1800)
    stamps = [parse(e["timestamp"]) for e in events]
    assert stamps == sorted(stamps)
    assert {e["source_type"] for e in events} == set(SOURCE_TYPES)


def test_duration_bounds_and_per_source_cadence():
    orchestrator = make()
    events = orchestrator.run_batch(60)
    assert all(START <= parse(e["timestamp"]) < START + timedelta(seconds=60) for e in events)
    counts = Counter(e["source_type"] for e in events)
    assert counts["warehouse"] == 12 * 60  # 12 sensors, 1 s interval
    assert counts["vehicle"] == 5 * 6  # 5 vehicles, 10 s interval
    assert counts["inventory"] > 0
    vehicle_stamps = {e["timestamp"] for e in events if e["source_type"] == "vehicle"}
    assert len(vehicle_stamps) == 6
    inventory_stamps = {e["timestamp"] for e in events if e["source_type"] == "inventory"}
    assert inventory_stamps <= {
        (START + timedelta(seconds=5 * k)).isoformat().replace("+00:00", "Z") for k in range(12)
    }


def test_zero_duration_yields_nothing_and_negative_is_rejected_eagerly():
    orchestrator = make()
    assert orchestrator.run_batch(0) == []
    with pytest.raises(ValueError):
        orchestrator.stream(-1)  # raised when called, not on first next()
    with pytest.raises(ValueError):
        orchestrator.run_batch(-5)


def test_deterministic_for_a_seed_and_different_across_seeds():
    assert make(seed=1).run_batch(120) == make(seed=1).run_batch(120)
    assert make(seed=1).run_batch(120) != make(seed=2).run_batch(120)


def test_stream_continues_where_it_stopped():
    orchestrator = make()
    first = orchestrator.run_batch(30)
    second = orchestrator.run_batch(30)
    whole = make().run_batch(60)
    assert first + second == whole
    assert orchestrator.current_time == START + timedelta(seconds=60)


def test_endless_stream_can_be_sliced_with_islice():
    events = list(islice(make().stream(throttle=False), 500))
    assert len(events) == 500


def test_stats_match_what_was_emitted():
    orchestrator = make(orders=True)
    events = orchestrator.run_batch(1800)
    stats = orchestrator.stats
    assert set(stats) == set(SOURCE_TYPES)
    for source in SOURCE_TYPES:
        mine = [e for e in events if e["source_type"] == source]
        assert stats[source]["events"] == len(mine) > 0
        assert stats[source]["anomalies"] == sum(e["payload"]["is_anomaly"] for e in mine)
    stats["warehouse"]["events"] = -1  # returned copy, not the live counters
    assert orchestrator.stats["warehouse"]["events"] > 0


# --------------------------------------------------------------------------- #
# Speed multiplier / pacing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("speed", "expected_wall_s"), [(1.0, 60.0), (10.0, 6.0), (60.0, 1.0)])
def test_throttled_stream_paces_against_the_wall_clock(speed, expected_wall_s):
    wall = FakeWallClock()
    orchestrator = make(speed_multiplier=speed, clock=wall.clock, sleep=wall.sleep)
    assert orchestrator.speed_multiplier == speed
    events = list(orchestrator.stream(60))
    assert events
    # Waiting stops at the last event (~59 s into the 60 s window).
    assert wall.now - 1000.0 == pytest.approx(59.0 / speed, abs=1e-6)
    assert wall.now - 1000.0 <= expected_wall_s
    assert all(s > 0 for s in wall.slept)


def test_event_is_released_when_its_simulated_time_is_due():
    wall = FakeWallClock()
    orchestrator = make(speed_multiplier=10.0, clock=wall.clock, sleep=wall.sleep)
    for event in orchestrator.stream(30):
        elapsed_wall = wall.now - 1000.0
        elapsed_sim = (parse(event["timestamp"]) - START).total_seconds()
        assert elapsed_wall == pytest.approx(elapsed_sim / 10.0, abs=1e-6)


def test_unthrottled_stream_never_sleeps():
    wall = FakeWallClock()
    orchestrator = make(speed_multiplier=1.0, clock=wall.clock, sleep=wall.sleep)
    list(orchestrator.stream(120, throttle=False))
    orchestrator.run_batch(120)
    assert wall.slept == []


def test_slow_consumer_is_not_slept_at_extra():
    wall = FakeWallClock()
    orchestrator = make(speed_multiplier=10.0, clock=wall.clock, sleep=wall.sleep)
    iterator = orchestrator.stream(20)
    next(iterator)
    wall.now += 1000.0  # consumer stalls far longer than the simulation advances
    wall.slept.clear()
    for _ in iterator:
        pass
    assert wall.slept == []  # already behind schedule, so it just catches up


@pytest.mark.parametrize("speed", [0, -1.0])
def test_speed_multiplier_must_be_positive(speed):
    with pytest.raises(ValueError):
        make(speed_multiplier=speed)


def test_default_speed_comes_from_config():
    cfg = SimulationConfig(seed=1, speed_multiplier=10.0)
    assert SimulationOrchestrator.from_config(cfg, start_time=START).speed_multiplier == 10.0


# --------------------------------------------------------------------------- #
# Manual injection through the orchestrator
# --------------------------------------------------------------------------- #
def test_injection_takes_effect_on_the_very_next_tick_mid_stream():
    orchestrator = make()
    stream = orchestrator.stream(throttle=False)
    # Consume everything up to (and including) the first warehouse tick.
    seen = [next(stream) for _ in range(12)]
    assert all(e["source_type"] in {"warehouse", "vehicle", "inventory"} for e in seen)
    sensor_id = orchestrator.warehouse.sensors[0].sensor_id
    episode_id = orchestrator.warehouse.inject_point_anomaly(sensor_id, duration=2)
    later = [e for e in islice(stream, 200) if e["entity_id"] == sensor_id]
    assert later[0]["payload"]["is_anomaly"] is True
    assert later[0]["payload"]["metadata"]["episode_id"] == episode_id
    assert later[0]["timestamp"] > seen[0]["timestamp"]


def test_gps_and_inventory_injections_appear_in_the_stream():
    orchestrator = make()
    orchestrator.vehicles.inject_route_deviation("veh-001", duration=2)
    orchestrator.inventory.inject_sudden_drop("SKU-0001")
    events = orchestrator.run_batch(30)
    deviation = [e for e in events if e["entity_id"] == "veh-001" and e["payload"]["is_anomaly"]]
    drop = [e for e in events if e["entity_id"] == "SKU-0001" and e["payload"]["is_anomaly"]]
    assert deviation and deviation[0]["payload"]["metadata"]["anomaly_subtype"] == "route_deviation"
    assert drop and drop[0]["payload"]["metadata"]["anomaly_subtype"] == "sudden_drop"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_from_config_builds_all_simulators_from_config_values():
    cfg = SimulationConfig.model_validate(
        {
            "seed": 3,
            "warehouse": {"n_warehouses": 1, "zones": ["A", "B"], "sensors_per_zone": 2},
            "vehicle": {"n_vehicles": 2},
            "inventory": {"n_skus": 6},
        }
    )
    orchestrator = SimulationOrchestrator.from_config(cfg, start_time=START)
    assert isinstance(orchestrator.warehouse, WarehouseSimulator)
    assert isinstance(orchestrator.vehicles, GPSSimulator)
    assert isinstance(orchestrator.inventory, InventorySimulator)
    assert orchestrator.orders is None
    assert len(orchestrator.warehouse.sensors) == 4
    assert len(orchestrator.vehicles.vehicles) == 2
    assert len(orchestrator.inventory.items) == 6
    # SKUs live in the same warehouses and zones as the sensors.
    assert {i.warehouse_id for i in orchestrator.inventory.items} == {"wh1"}
    assert {i.zone for i in orchestrator.inventory.items} <= {"A", "B"}
    assert orchestrator.current_time == START


def test_all_sources_share_the_start_time_even_without_one():
    orchestrator = SimulationOrchestrator.from_config(SimulationConfig(seed=1))
    start = orchestrator.warehouse.current_time
    assert orchestrator.vehicles.current_time == start
    assert orchestrator.inventory.current_time == start
    assert abs((datetime.now(UTC) - start).total_seconds()) < 5


def test_orders_enabled_by_config_or_argument():
    cfg = SimulationConfig.model_validate({"seed": 1, "order": {"enabled": True}})
    with_orders = SimulationOrchestrator.from_config(cfg, start_time=START)
    assert isinstance(with_orders.orders, OrderEventSimulator)
    assert "order" in with_orders.stats
    forced_off = SimulationOrchestrator.from_config(cfg, start_time=START, include_orders=False)
    assert forced_off.orders is None
    assert "order" not in forced_off.stats
    assert make(orders=True).orders is not None


def test_order_events_use_the_inventory_catalogue():
    orchestrator = make(orders=True)
    skus = {i.sku_id for i in orchestrator.inventory.items}
    orders = [e for e in orchestrator.run_batch(900) if e["source_type"] == "order"]
    assert orders and {e["payload"]["sku_id"] for e in orders} <= skus
    assert all(e["entity_id"] == e["payload"]["order_id"] for e in orders)


def test_manual_construction_from_ready_made_simulators():
    warehouse = WarehouseSimulator(start_time=START, seed=1)
    vehicles = GPSSimulator(start_time=START, seed=1)
    inventory = InventorySimulator(start_time=START, seed=1)
    orchestrator = SimulationOrchestrator(warehouse, vehicles, inventory)
    assert orchestrator.warehouse is warehouse
    assert orchestrator.vehicles is vehicles
    assert orchestrator.inventory is inventory
    assert len(orchestrator.run_batch(10)) > 0


def test_simulators_with_different_start_times_are_merged_by_timestamp():
    warehouse = WarehouseSimulator(start_time=START + timedelta(seconds=30), seed=1)
    vehicles = GPSSimulator(start_time=START, seed=1)
    inventory = InventorySimulator(start_time=START + timedelta(seconds=10), seed=1)
    orchestrator = SimulationOrchestrator(warehouse, vehicles, inventory)
    events = orchestrator.run_batch(40)
    stamps = [parse(e["timestamp"]) for e in events]
    assert stamps == sorted(stamps)
    first_of = {}
    for e in events:
        first_of.setdefault(e["source_type"], parse(e["timestamp"]))
    assert first_of["vehicle"] == START
    assert first_of["inventory"] >= START + timedelta(seconds=10)
    assert first_of["warehouse"] == START + timedelta(seconds=30)


def test_config_yaml_defaults_are_loadable_and_orders_off_by_default():
    cfg = load_config(environ={})
    assert cfg.simulation.order.enabled is False
    assert cfg.simulation.inventory.transaction_probability == 0.25
    orchestrator = SimulationOrchestrator.from_config(cfg.simulation, start_time=START)
    assert orchestrator.orders is None


def test_public_exports():
    for name in data_simulation.__all__:
        assert hasattr(data_simulation, name)
    for name in ("GPSSimulator", "InventorySimulator", "OrderEventSimulator", "VehicleTracker"):
        assert name in data_simulation.__all__
