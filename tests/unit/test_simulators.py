"""Injection-rate and cross-simulator checks for the three data simulators (Day 23).

The per-module test files (``test_warehouse_simulator.py`` ...) cover each simulator's behaviour in
detail. This file asserts the property the rest of the system depends on: the share of anomalous
events the simulators emit matches the configured injection rates, so the labelled data used to
train and evaluate every detector has the class balance the configuration promises.

All runs are seeded, so the counts are deterministic: a failure means the simulators changed, not
that a random draw was unlucky.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import pytest

from src.data_simulation import (
    GPSSimulator,
    InventorySimulator,
    SimulationOrchestrator,
    WarehouseSimulator,
    build_default_items,
    build_default_sensors,
    build_default_vehicles,
)
from src.utils.config import get_config

START = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
DAY_S = 86_400
ANOMALY_TYPES = ("point", "contextual", "collective")

#: Allowed relative deviation of the realised rate from the configured one.
TOLERANCE = 0.10
#: The inventory simulator cannot inject a drop on an almost empty SKU or ghost stock on a full
#: one, so its realised rate is documented to be "slightly below the target" (about 11 % here).
INVENTORY_TOLERANCE = 0.15

WAREHOUSE_RATES = {"point": 0.02, "contextual": 0.01, "collective": 0.005}
GPS_RATE = 0.02
INVENTORY_RATE = 0.01


# --------------------------------------------------------------------------- #
# Realised anomaly shares over multi-day runs (counted, never stored)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def warehouse_shares() -> dict[str, float]:
    """Share of readings per label over three simulated days of the default fleet."""
    sim = WarehouseSimulator(
        build_default_sensors(seed=1),
        point_anomaly_rate=WAREHOUSE_RATES["point"],
        contextual_anomaly_rate=WAREHOUSE_RATES["contextual"],
        collective_anomaly_rate=WAREHOUSE_RATES["collective"],
        start_time=START,
        seed=1,
    )
    counts: Counter[str | None] = Counter()
    total = 0
    for reading in sim.simulate_all_sensors(duration_s=3 * DAY_S):
        total += 1
        counts[reading["anomaly_type"]] += 1
    return {kind: counts[kind] / total for kind in ANOMALY_TYPES}


@pytest.fixture(scope="module")
def gps_shares() -> dict[str, float]:
    """Share of anomalous readings and the labelled sub-types over a simulated week."""
    sim = GPSSimulator(
        build_default_vehicles(seed=1), anomaly_rate=GPS_RATE, start_time=START, seed=1
    )
    counts: Counter[str | None] = Counter()
    total = 0
    for reading in sim.simulate_all_vehicles(duration_s=7 * DAY_S):
        total += 1
        counts[reading["anomaly_type"]] += 1
    anomalous = total - counts[None]
    shares = {"anomaly": anomalous / total}
    shares.update({kind: counts[kind] / total for kind in ANOMALY_TYPES})
    return shares


@pytest.fixture(scope="module")
def inventory_shares() -> dict[str, float]:
    """Share of anomalous events over three simulated days of the default catalogue."""
    sim = InventorySimulator(
        build_default_items(seed=1), anomaly_rate=INVENTORY_RATE, start_time=START, seed=1
    )
    counts: Counter[str | None] = Counter()
    total = 0
    for event in sim.simulate_all_skus(duration_s=3 * DAY_S):
        total += 1
        counts[event["anomaly_type"]] += 1
    return {"anomaly": (total - counts[None]) / total}


@pytest.mark.parametrize("kind", ANOMALY_TYPES)
def test_warehouse_rate_per_type_is_within_10_percent(warehouse_shares, kind):
    assert warehouse_shares[kind] == pytest.approx(WAREHOUSE_RATES[kind], rel=TOLERANCE)


def test_warehouse_rates_keep_their_ordering(warehouse_shares):
    assert (
        warehouse_shares["point"] > warehouse_shares["contextual"] > warehouse_shares["collective"]
    )


def test_gps_anomaly_rate_is_within_10_percent(gps_shares):
    assert gps_shares["anomaly"] == pytest.approx(GPS_RATE, rel=TOLERANCE)


def test_gps_emits_every_anomaly_type_it_is_configured_for(gps_shares):
    assert gps_shares["point"] > 0
    assert gps_shares["collective"] > 0  # gradual delay
    assert sum(gps_shares[kind] for kind in ANOMALY_TYPES) == pytest.approx(gps_shares["anomaly"])


def test_inventory_anomaly_rate_is_close_to_configured_and_never_above(inventory_shares):
    realised = inventory_shares["anomaly"]
    assert realised == pytest.approx(INVENTORY_RATE, rel=INVENTORY_TOLERANCE)
    assert realised <= INVENTORY_RATE * (1 + TOLERANCE)


def test_doubling_the_warehouse_point_rate_doubles_the_share():
    """The rate parameter scales the output linearly (one simulated day, point anomalies only)."""

    def share(rate: float) -> float:
        sim = WarehouseSimulator(
            build_default_sensors(seed=2),
            point_anomaly_rate=rate,
            contextual_anomaly_rate=0.0,
            collective_anomaly_rate=0.0,
            start_time=START,
            seed=2,
        )
        readings = list(sim.simulate_all_sensors(duration_s=DAY_S))
        return sum(r["anomaly_type"] == "point" for r in readings) / len(readings)

    base, doubled = share(0.02), share(0.04)
    assert base == pytest.approx(0.02, rel=TOLERANCE)
    assert doubled == pytest.approx(2 * base, rel=TOLERANCE)


# --------------------------------------------------------------------------- #
# Reproducibility and orchestrator accounting
# --------------------------------------------------------------------------- #
def _fingerprint(events: list[dict]) -> list[tuple]:
    return [(e["source_type"], e["entity_id"], e["timestamp"]) for e in events[:500]]


def test_same_seed_gives_the_same_stream_and_a_different_seed_does_not():
    cfg = get_config().simulation

    def run(seed: int) -> list[dict]:
        orch = SimulationOrchestrator.from_config(
            cfg.model_copy(update={"seed": seed}), start_time=START, include_orders=False
        )
        return list(orch.stream(600, throttle=False))

    first, again, other = run(11), run(11), run(12)
    assert first == again
    assert [e["payload"].get("value") for e in first[:200]] != [
        e["payload"].get("value") for e in other[:200]
    ]


def test_orchestrator_stats_match_the_events_it_yielded():
    cfg = get_config().simulation.model_copy(update={"seed": 5})
    orch = SimulationOrchestrator.from_config(cfg, start_time=START, include_orders=False)
    events = orch.run_batch(2 * 3600)

    by_source: Counter[str] = Counter(e["source_type"] for e in events)
    anomalies: Counter[str] = Counter(
        e["source_type"] for e in events if e["payload"]["is_anomaly"]
    )
    assert set(by_source) == {"warehouse", "vehicle", "inventory"}
    for source, totals in orch.stats.items():
        assert totals["events"] == by_source[source]
        assert totals["anomalies"] == anomalies[source]
    # the merged stream is in time order
    stamps = [e["timestamp"] for e in events]
    assert stamps == sorted(stamps)


def test_every_source_labels_anomalies_consistently():
    cfg = get_config().simulation.model_copy(update={"seed": 6})
    orch = SimulationOrchestrator.from_config(cfg, start_time=START, include_orders=False)
    for event in orch.run_batch(3600):
        payload = event["payload"]
        assert payload["is_anomaly"] == (payload["anomaly_type"] is not None)
        assert payload["anomaly_type"] in (None, *ANOMALY_TYPES)
