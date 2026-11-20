import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from itertools import islice

import pytest

from src.data_simulation.inventory_simulator import (
    ADJUSTMENT,
    FULFILLMENT_DELAY,
    GHOST_STOCK,
    INVENTORY_ANOMALY_SUBTYPES,
    ORDER_STATUSES,
    RESTOCK,
    SHIPMENT,
    SUDDEN_DROP,
    TRANSACTION_TYPES,
    InventoryItem,
    InventorySimulator,
    OrderEventSimulator,
    build_default_items,
)
from src.utils.config import InventorySimConfig, OrderSimConfig

START = datetime(2026, 3, 10, 8, 0, 0, tzinfo=UTC)


def item(**overrides):
    fields = {
        "sku_id": "S1",
        "warehouse_id": "wh1",
        "zone": "A",
        "min_stock": 50,
        "max_stock": 1000,
        "reorder_point": 288,
        "initial_stock": 600,
    }
    return InventoryItem(**{**fields, **overrides})


def make_sim(items=None, *, quiet=True, seed=7, **kwargs):
    if quiet:
        kwargs = {"anomaly_rate": 0.0, **kwargs}
    return InventorySimulator(
        items if items is not None else [item()], start_time=START, seed=seed, **kwargs
    )


def run(sim, ticks):
    return list(sim.simulate_all_skus(n_ticks=ticks))


def parse(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# InventoryItem
# --------------------------------------------------------------------------- #
def test_item_derived_properties():
    i = item()
    assert i.stock_range == 950
    assert i.max_shipment == 38
    assert i.restock_quantity == 332
    assert i.anomaly_floor == 190


def test_tiny_item_has_sane_derived_values():
    i = item(min_stock=0, max_stock=10, reorder_point=3, initial_stock=None)
    assert i.max_shipment == 1
    assert i.restock_quantity >= 1
    assert i.anomaly_floor == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"sku_id": ""},
        {"warehouse_id": ""},
        {"zone": ""},
        {"min_stock": -1},
        {"max_stock": 50},
        {"max_stock": 40},
        {"reorder_point": 49},
        {"reorder_point": 1000},
        {"initial_stock": 49},
        {"initial_stock": 1001},
    ],
)
def test_item_validation(overrides):
    with pytest.raises(ValueError):
        item(**overrides)


def test_item_is_frozen():
    with pytest.raises(AttributeError):
        item().min_stock = 1  # type: ignore[misc]


def test_default_catalogue():
    items = build_default_items(10, n_warehouses=2, zones=("A", "B"), seed=1)
    assert [i.sku_id for i in items][:2] == ["SKU-0001", "SKU-0002"]
    assert {i.warehouse_id for i in items} == {"wh1", "wh2"}
    assert {i.zone for i in items} == {"A", "B"}
    assert all(i.min_stock <= i.reorder_point < i.max_stock for i in items)
    assert items == build_default_items(10, n_warehouses=2, zones=("A", "B"), seed=1)


@pytest.mark.parametrize("kwargs", [{"n_skus": 0}, {"n_warehouses": 0}, {"zones": ()}])
def test_default_catalogue_validation(kwargs):
    with pytest.raises(ValueError):
        build_default_items(**kwargs)


# --------------------------------------------------------------------------- #
# Normal behaviour
# --------------------------------------------------------------------------- #
def test_event_schema_and_json_serialisable():
    events = run(make_sim(transaction_probability=1.0), 3)
    assert len(events) == 3
    first = events[0]
    assert set(first) == {
        "sku_id",
        "warehouse_id",
        "timestamp",
        "previous_level",
        "new_level",
        "change_amount",
        "transaction_type",
        "is_anomaly",
        "anomaly_type",
        "metadata",
    }
    assert first["timestamp"] == "2026-03-10T08:00:00Z"
    assert first["is_anomaly"] is False and first["anomaly_type"] is None
    assert first["metadata"]["sequence"] == 1
    json.dumps(first)


def test_ticks_advance_the_clock_by_the_event_interval():
    sim = make_sim(transaction_probability=1.0, event_interval_s=5.0)
    stamps = [e["timestamp"] for e in run(sim, 3)]
    assert stamps == ["2026-03-10T08:00:00Z", "2026-03-10T08:00:05Z", "2026-03-10T08:00:10Z"]
    assert sim.current_time == START + timedelta(seconds=15)
    assert sim.event_interval_s == 5.0


def test_levels_chain_and_change_amounts_add_up():
    sim = make_sim(transaction_probability=1.0)
    start_level = sim.stock_level("S1")
    events = run(sim, 2000)
    assert events[0]["previous_level"] == start_level
    for prev, cur in zip(events, events[1:], strict=False):
        assert cur["previous_level"] == prev["new_level"]
    for e in events:
        assert e["new_level"] - e["previous_level"] == e["change_amount"] != 0
    assert sim.stock_level("S1") == events[-1]["new_level"]


def test_normal_stock_stays_in_band_and_sawtooths():
    it = item()
    sim = make_sim([it], transaction_probability=1.0)
    events = run(sim, 20_000)
    assert all(it.min_stock <= e["new_level"] <= it.max_stock for e in events)
    types = Counter(e["transaction_type"] for e in events)
    assert set(types) == set(TRANSACTION_TYPES)
    assert types[SHIPMENT] > 10 * types[RESTOCK] > 0
    for e in events:
        if e["transaction_type"] == SHIPMENT:
            assert -it.max_shipment <= e["change_amount"] <= -1
        if e["transaction_type"] == RESTOCK:
            assert e["previous_level"] <= it.reorder_point  # only restocks when low
            assert e["change_amount"] <= round(it.restock_quantity * 1.1) + 1


def test_transaction_probability_controls_volume():
    sim = make_sim(build_default_items(20, seed=1), transaction_probability=0.25)
    events = run(sim, 2000)
    assert 0.22 <= len(events) / (2000 * 20) <= 0.28


def test_deterministic_and_independent_per_sku():
    a = run(make_sim(quiet=False, seed=5), 500)
    assert a == run(make_sim(quiet=False, seed=5), 500)
    assert a != run(make_sim(quiet=False, seed=6), 500)
    items = build_default_items(3, seed=1)
    solo = [e for e in run(make_sim(items[:2], quiet=False), 400) if e["sku_id"] == "SKU-0001"]
    trio = [e for e in run(make_sim(items, quiet=False), 400) if e["sku_id"] == "SKU-0001"]
    assert solo == trio


# --------------------------------------------------------------------------- #
# Sudden drop (point)
# --------------------------------------------------------------------------- #
def test_sudden_drop_is_a_large_unexplained_decrease():
    sim = make_sim(transaction_probability=0.01)  # injection must not depend on probability
    level = sim.stock_level("S1")
    episode_id = sim.inject_sudden_drop("S1")
    event = run(sim, 1)[0]
    meta = event["metadata"]
    assert event["is_anomaly"] and event["anomaly_type"] == "point"
    assert event["transaction_type"] == ADJUSTMENT
    assert meta["anomaly_subtype"] == SUDDEN_DROP and meta["episode_id"] == episode_id
    assert event["previous_level"] == level
    assert event["change_amount"] <= -item().anomaly_floor
    assert event["new_level"] == level + event["change_amount"] >= 0
    # Far bigger than any normal shipment.
    assert -event["change_amount"] > 4 * item().max_shipment


def test_sudden_drop_manual_magnitude_and_clamping():
    sim = make_sim()
    sim.inject_sudden_drop("S1", magnitude=123)
    assert run(sim, 1)[0]["change_amount"] == -123
    level = sim.stock_level("S1")
    sim.inject_sudden_drop("S1", magnitude=level + 5000)
    event = run(sim, 1)[0]
    assert event["new_level"] == 0 and event["change_amount"] == -level


def test_sudden_drop_on_low_stock_wipes_remaining_stock():
    sim = make_sim([item(initial_stock=60)])
    sim.inject_sudden_drop("S1")
    event = run(sim, 1)[0]
    assert event["is_anomaly"] and event["new_level"] == 0


def test_stock_recovers_by_restocking_after_a_drop():
    sim = make_sim(transaction_probability=1.0)
    sim.inject_sudden_drop("S1", magnitude=10_000)  # empties the shelf
    events = run(sim, 5)
    assert events[0]["new_level"] == 0
    assert events[1]["transaction_type"] == RESTOCK and not events[1]["is_anomaly"]


# --------------------------------------------------------------------------- #
# Ghost stock (contextual)
# --------------------------------------------------------------------------- #
def test_ghost_stock_looks_like_a_restock_but_is_labelled_contextual():
    sim = make_sim(transaction_probability=0.01)
    level = sim.stock_level("S1")
    episode_id = sim.inject_ghost_stock("S1")
    event = run(sim, 1)[0]
    assert event["is_anomaly"] and event["anomaly_type"] == "contextual"
    assert event["transaction_type"] == RESTOCK
    assert event["metadata"]["anomaly_subtype"] == GHOST_STOCK
    assert event["metadata"]["episode_id"] == episode_id
    assert event["previous_level"] == level > item().reorder_point  # nowhere near low stock
    assert event["change_amount"] >= item().anomaly_floor
    assert event["new_level"] <= item().max_stock


def test_ghost_stock_manual_magnitude_clamped_to_capacity():
    sim = make_sim([item(initial_stock=900)])
    sim.inject_ghost_stock("S1", magnitude=5000)
    event = run(sim, 1)[0]
    assert event["new_level"] == 1000 and event["change_amount"] == 100
    sim2 = make_sim()
    sim2.inject_ghost_stock("S1", magnitude=17)
    assert run(sim2, 1)[0]["change_amount"] == 17


# --------------------------------------------------------------------------- #
# Injection plumbing
# --------------------------------------------------------------------------- #
def test_injection_ids_are_unique_and_latest_injection_wins():
    sim = make_sim()
    first = sim.inject_sudden_drop("S1")
    second = sim.inject_ghost_stock("S1")
    assert first != second
    events = run(sim, 1)
    assert events[0]["metadata"]["episode_id"] == second
    assert events[0]["metadata"]["anomaly_subtype"] == GHOST_STOCK


def test_injection_only_applies_once():
    sim = make_sim(transaction_probability=1.0)
    sim.inject_sudden_drop("S1")
    events = run(sim, 5)
    assert [e["is_anomaly"] for e in events] == [True, False, False, False, False]


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.inject_sudden_drop("nope"),
        lambda s: s.inject_ghost_stock("nope"),
        lambda s: s.get_item("nope"),
        lambda s: s.stock_level("nope"),
    ],
)
def test_unknown_sku_raises_key_error(call):
    with pytest.raises(KeyError):
        call(make_sim())


def test_injection_argument_validation():
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.inject_sudden_drop("S1", magnitude=0)
    with pytest.raises(ValueError):
        sim.inject_ghost_stock("S1", magnitude=-3)
    empty = make_sim([item(initial_stock=50)])
    empty.inject_sudden_drop("S1", magnitude=10_000)
    run(empty, 1)
    with pytest.raises(ValueError, match="no stock"):
        empty.inject_sudden_drop("S1")
    full = make_sim([item(initial_stock=1000)])
    with pytest.raises(ValueError, match="max_stock"):
        full.inject_ghost_stock("S1")


# --------------------------------------------------------------------------- #
# Random anomalies and ground truth
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def noisy_run():
    sim = InventorySimulator(
        build_default_items(50, seed=1), start_time=START, seed=42, anomaly_rate=0.01
    )
    return sim, run(sim, 6000)


def test_anomaly_rate_close_to_configured(noisy_run):
    _, events = noisy_run
    rate = sum(e["is_anomaly"] for e in events) / len(events)
    assert 0.007 <= rate <= 0.0125


def test_both_anomaly_kinds_appear_in_configured_proportion(noisy_run):
    _, events = noisy_run
    counts = Counter(e["metadata"]["anomaly_subtype"] for e in events if e["is_anomaly"])
    assert set(counts) == set(INVENTORY_ANOMALY_SUBTYPES)
    share = counts[SUDDEN_DROP] / sum(counts.values())
    assert 0.5 <= share <= 0.7  # configured 60 / 40


def test_labels_are_exact_ground_truth(noisy_run):
    sim, events = noisy_run
    for e in events:
        meta = e["metadata"]
        it = sim.get_item(e["sku_id"])
        assert e["is_anomaly"] == (e["anomaly_type"] is not None)
        assert (meta["anomaly_subtype"] is None) == (not e["is_anomaly"])
        assert e["new_level"] - e["previous_level"] == e["change_amount"]
        if meta["anomaly_subtype"] == SUDDEN_DROP:
            assert e["change_amount"] <= -it.anomaly_floor
        elif meta["anomaly_subtype"] == GHOST_STOCK:
            assert e["change_amount"] >= it.anomaly_floor
            assert e["previous_level"] > it.reorder_point
        elif e["transaction_type"] == SHIPMENT:
            assert -it.max_shipment <= e["change_amount"] < 0
        # No normal event ever lands outside the band.
        if not e["is_anomaly"]:
            assert 0 <= e["new_level"] <= it.max_stock


def test_events_are_time_ordered_and_sequences_increase(noisy_run):
    _, events = noisy_run
    stamps = [parse(e["timestamp"]) for e in events]
    assert stamps == sorted(stamps)
    seq: dict[str, int] = defaultdict(int)
    for e in events:
        assert e["metadata"]["sequence"] == seq[e["sku_id"]] + 1
        seq[e["sku_id"]] += 1


def test_zero_anomaly_rate_produces_no_anomalies():
    assert not any(e["is_anomaly"] for e in run(make_sim(transaction_probability=1.0), 3000))


# --------------------------------------------------------------------------- #
# Construction and streaming arguments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"transaction_probability": 0.0},
        {"transaction_probability": 1.5},
        {"anomaly_rate": -0.1},
        {"anomaly_rate": 1.1},
        {"anomaly_mix": (0.5,)},
        {"anomaly_mix": (0.7, 0.7)},
        {"anomaly_mix": (-0.1, 1.1)},
        {"event_interval_s": 0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        InventorySimulator([item()], start_time=START, **kwargs)


def test_empty_and_duplicate_catalogues_rejected():
    with pytest.raises(ValueError):
        InventorySimulator([], start_time=START)
    with pytest.raises(ValueError, match="duplicate"):
        InventorySimulator([item(), item()], start_time=START)


def test_initial_stock_respected_or_chosen_within_band():
    assert make_sim([item(initial_stock=321)]).stock_level("S1") == 321
    sim = make_sim(build_default_items(30, seed=2), seed=3)
    for it in sim.items:
        assert it.reorder_point <= sim.stock_level(it.sku_id) <= it.max_stock


def test_simulate_by_duration_and_argument_errors():
    sim = make_sim(transaction_probability=1.0)
    assert len(list(sim.simulate_all_skus(duration_s=12))) == 3  # rounds up to 3 ticks
    assert list(sim.simulate_all_skus(n_ticks=0)) == []
    with pytest.raises(ValueError):
        list(sim.simulate_all_skus(n_ticks=1, duration_s=5))
    with pytest.raises(ValueError):
        list(sim.simulate_all_skus(duration_s=-1))
    with pytest.raises(ValueError):
        list(sim.simulate_all_skus(n_ticks=-1))
    assert len(list(islice(sim.simulate_all_skus(), 7))) == 7


def test_naive_start_time_is_utc_and_default_is_now():
    sim = InventorySimulator([item()], start_time=datetime(2026, 1, 1, 12, 0, 0))
    assert sim.current_time == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert abs((datetime.now(UTC) - InventorySimulator([item()]).current_time).total_seconds()) < 5


def test_from_config_uses_config_values():
    cfg = InventorySimConfig(n_skus=8, event_interval_s=2.0, transaction_probability=0.5)
    sim = InventorySimulator.from_config(
        cfg, n_warehouses=3, zones=("X",), seed=1, start_time=START
    )
    assert len(sim.items) == 8
    assert sim.event_interval_s == 2.0
    assert {i.warehouse_id for i in sim.items} == {"wh1", "wh2", "wh3"}
    assert {i.zone for i in sim.items} == {"X"}
    custom = [item()]
    assert InventorySimulator.from_config(cfg, items=custom, start_time=START).items == tuple(
        custom
    )
    assert sim.get_item("SKU-0001").sku_id == "SKU-0001"


# =========================================================================== #
# OrderEventSimulator
# =========================================================================== #
def make_orders(*, seed=3, **kwargs):
    return OrderEventSimulator(
        [item(sku_id="A"), item(sku_id="B", warehouse_id="wh2")],
        start_time=START,
        seed=seed,
        **kwargs,
    )


@pytest.fixture(scope="module")
def order_run():
    sim = make_orders(seed=11, arrival_rate_per_s=0.5, anomaly_rate=0.02)
    # 6 simulated hours, so almost every order started in the first hours is delivered.
    return sim, list(sim.simulate(duration_s=6 * 3600))


def test_order_event_schema_and_json():
    event = next(make_orders().simulate(n_events=1))
    assert set(event) == {
        "order_id",
        "sku_id",
        "warehouse_id",
        "timestamp",
        "status",
        "previous_status",
        "quantity",
        "stage_duration_s",
        "expected_duration_s",
        "is_anomaly",
        "anomaly_type",
        "metadata",
    }
    assert event["status"] == "created" and event["previous_status"] is None
    assert event["stage_duration_s"] is None and event["expected_duration_s"] is None
    assert event["order_id"] == "ORD-000001"
    json.dumps(event)


def test_events_are_globally_time_ordered(order_run):
    _, events = order_run
    stamps = [parse(e["timestamp"]) for e in events]
    assert stamps == sorted(stamps)
    assert stamps[0] >= START


def test_each_order_walks_the_lifecycle_in_order(order_run):
    _, events = order_run
    by_order: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_order[e["order_id"]].append(e)
    complete = [evs for evs in by_order.values() if len(evs) == len(ORDER_STATUSES)]
    assert len(complete) > 1000
    for evs in complete:
        assert [e["status"] for e in evs] == list(ORDER_STATUSES)
        assert [e["previous_status"] for e in evs] == [None, *ORDER_STATUSES[:-1]]
        assert len({(e["sku_id"], e["warehouse_id"], e["quantity"]) for e in evs}) == 1
        for prev, cur in zip(evs, evs[1:], strict=False):
            gap = (parse(cur["timestamp"]) - parse(prev["timestamp"])).total_seconds()
            assert gap == pytest.approx(cur["stage_duration_s"], abs=0.01)


def test_delay_anomalies_are_exact_ground_truth(order_run):
    _, events = order_run
    transitions = [e for e in events if e["status"] != "created"]
    medians = (90.0, 60.0, 120.0, 600.0)
    for e in transitions:
        ratio = e["stage_duration_s"] / e["expected_duration_s"]
        assert e["expected_duration_s"] == medians[ORDER_STATUSES.index(e["status"]) - 1]
        if e["is_anomaly"]:
            assert 4.99 <= ratio <= 12.01
            assert e["anomaly_type"] == "point"
            assert e["metadata"]["anomaly_subtype"] == FULFILLMENT_DELAY
            assert e["metadata"]["episode_id"].startswith(e["order_id"])
        else:
            assert ratio <= 3.5 + 1e-6
            assert e["anomaly_type"] is None and e["metadata"]["anomaly_subtype"] is None
    assert not any(e["is_anomaly"] for e in events if e["status"] == "created")


def test_at_most_one_delayed_stage_per_order(order_run):
    _, events = order_run
    delayed = Counter(e["order_id"] for e in events if e["is_anomaly"])
    assert delayed and max(delayed.values()) == 1


def test_order_anomaly_rate_and_arrival_rate(order_run):
    sim, events = order_run
    rate = sum(e["is_anomaly"] for e in events) / len(events)
    assert 0.015 <= rate <= 0.026  # configured 2 % of events
    assert sim.orders_created / (6 * 3600) == pytest.approx(0.5, rel=0.05)
    assert sim.in_flight > 0


def test_order_stream_deterministic_and_seed_sensitive():
    a = list(make_orders(seed=1).simulate(n_events=300))
    assert a == list(make_orders(seed=1).simulate(n_events=300))
    assert a != list(make_orders(seed=2).simulate(n_events=300))


def test_next_event_time_matches_the_next_event():
    sim = make_orders()
    for _ in range(200):
        due = sim.next_event_time
        assert parse(sim.next_event()["timestamp"]) == due


def test_simulate_bounds_and_argument_errors():
    sim = make_orders()
    events = list(sim.simulate(duration_s=300))
    assert events and all(parse(e["timestamp"]) < START + timedelta(seconds=300) for e in events)
    assert len(list(make_orders().simulate(n_events=25))) == 25
    assert list(make_orders().simulate(n_events=0)) == []
    with pytest.raises(ValueError):
        list(make_orders().simulate(n_events=1, duration_s=1))
    with pytest.raises(ValueError):
        list(make_orders().simulate(n_events=-1))
    with pytest.raises(ValueError):
        list(make_orders().simulate(duration_s=-1))


def test_zero_anomaly_rate_and_zero_sigma_give_exact_medians():
    sim = make_orders(anomaly_rate=0.0, stage_sigma=0.0)
    transitions = [e for e in sim.simulate(duration_s=3600) if e["status"] != "created"]
    assert transitions
    assert all(not e["is_anomaly"] for e in transitions)
    assert all(e["stage_duration_s"] == e["expected_duration_s"] for e in transitions)


def test_full_anomaly_rate_is_capped_at_one_delay_per_order():
    sim = make_orders(anomaly_rate=1.0)
    events = list(sim.simulate(duration_s=4 * 3600))
    by_order = Counter(e["order_id"] for e in events if e["is_anomaly"])
    assert max(by_order.values()) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arrival_rate_per_s": 0},
        {"anomaly_rate": 1.5},
        {"anomaly_rate": -0.1},
        {"stage_median_s": (1.0, 2.0, 3.0)},
        {"stage_median_s": (1.0, 2.0, 3.0, 0.0)},
        {"stage_sigma": -0.1},
    ],
)
def test_order_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        make_orders(**kwargs)


def test_order_empty_catalogue_rejected_and_default_catalogue_used():
    with pytest.raises(ValueError):
        OrderEventSimulator([], start_time=START)
    sim = OrderEventSimulator(start_time=START, seed=1)
    assert next(sim.simulate(n_events=1))["sku_id"].startswith("SKU-")


def test_order_from_config():
    cfg = OrderSimConfig(enabled=True, arrival_rate_per_s=1.0, anomaly_rate=0.0)
    sim = OrderEventSimulator.from_config(cfg, items=[item()], seed=1, start_time=START)
    events = list(sim.simulate(duration_s=600))
    assert 450 < sim.orders_created < 750
    assert not any(e["is_anomaly"] for e in events)
