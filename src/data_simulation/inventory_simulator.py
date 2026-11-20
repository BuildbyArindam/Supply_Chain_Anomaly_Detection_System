"""Inventory stock-level and order-fulfilment simulation (Day 3).

Two streams live here because orders are what move stock:

* :class:`InventorySimulator` emits stock-change events per SKU. Normal stock
  follows a saw-tooth: small **shipments** drain it, a lot-sized **restock** tops
  it up once it falls to the reorder point, with occasional small **returns** and
  cycle-count **adjustments**. Two anomalies are injected, both single events:

  * **sudden_drop** (``point``) - a large unexplained decrease (theft, system
    error), booked as an ``adjustment``. It is 5-12x larger than any shipment.
  * **ghost_stock** (``contextual``) - stock appears without a physical arrival,
    booked as a ``restock``. It looks exactly like a legitimate restock; only the
    *context* (stock was nowhere near the reorder point) gives it away.

* :class:`OrderEventSimulator` emits the order lifecycle
  (created -> picked -> packed -> shipped -> delivered) with realistic stage
  durations. The anomaly is a **fulfillment_delay** (``point``): one stage takes
  5-12x its normal time.

Normal events never leave the ``[min_stock, max_stock]`` band (except while the
stock is recovering from an injected drop) and never use the unusual sizes, so
``is_anomaly`` is the exact ground truth. Stock events look like::

    {
        "sku_id": "SKU-0001", "warehouse_id": "wh1", "timestamp": "2026-11-20T08:00:05Z",
        "previous_level": 412, "new_level": 397, "change_amount": -15,
        "transaction_type": "shipment",        # shipment | restock | return | adjustment
        "is_anomaly": False, "anomaly_type": None,        # None | point | contextual
        "metadata": {"zone": "A", "min_stock": 50, "max_stock": 1000, "reorder_point": 288,
                     "sequence": 17, "anomaly_subtype": None, "episode_id": None,
                     "episode_step": None},
    }

Example::

    from src.data_simulation.inventory_simulator import InventorySimulator

    sim = InventorySimulator(seed=42)                    # 50 SKUs over 2 warehouses
    events = list(sim.simulate_all_skus(duration_s=60))

    sim.inject_sudden_drop("SKU-0001")                   # force a theft on the next tick

Both simulators are deterministic for a given ``seed`` (each SKU has its own
random stream) and depend only on the standard library and the project logger.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, TypedDict

from src.data_simulation._common import (
    default_start,
    isoformat,
    to_utc,
    validate_mix,
    validate_positive,
    validate_rate,
)
from src.data_simulation.warehouse_simulator import CONTEXTUAL, POINT, AnomalyType
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config import InventorySimConfig, OrderSimConfig

SHIPMENT: Final = "shipment"
RESTOCK: Final = "restock"
RETURN: Final = "return"
ADJUSTMENT: Final = "adjustment"
TRANSACTION_TYPES: Final[tuple[str, ...]] = (SHIPMENT, RESTOCK, RETURN, ADJUSTMENT)

SUDDEN_DROP: Final = "sudden_drop"
GHOST_STOCK: Final = "ghost_stock"
FULFILLMENT_DELAY: Final = "fulfillment_delay"
INVENTORY_ANOMALY_SUBTYPES: Final[tuple[str, ...]] = (SUDDEN_DROP, GHOST_STOCK)

ORDER_STATUSES: Final[tuple[str, ...]] = ("created", "picked", "packed", "shipped", "delivered")

# Sizes are fractions of an item's (max_stock - min_stock) range.
_SHIPMENT_FRACTION: Final = 0.04  # largest normal shipment
_RESTOCK_FRACTION: Final = 0.35  # restock lot size
_RESTOCK_JITTER: Final = (0.9, 1.1)
_ANOMALY_FLOOR_SHIPMENTS: Final = 5  # anomalies move at least this many max-shipments
_DROP_LEVEL_FRACTION: Final = (0.2, 0.5)  # sudden drop as a share of the current level
_GHOST_RANGE_FRACTION: Final = 0.45  # largest ghost, as a share of the range

_RESTOCK_WHEN_LOW_PROBABILITY: Final = 0.25  # per transaction once at/below the reorder point
_RETURN_PROBABILITY: Final = 0.03
_ADJUSTMENT_PROBABILITY: Final = 0.02
_SMALL_CHANGE_MAX: Final = 3  # units, for returns and adjustments

# Orders: normal stage durations are lognormal, capped so that a delayed stage
# (5-12x the median) can never be mistaken for a normal one.
_NORMAL_DURATION_CAP: Final = 3.5
_DELAY_MULTIPLIER: Final = (5.0, 12.0)
_MAX_RESAMPLES: Final = 25
_STAGES: Final = len(ORDER_STATUSES) - 1
_EVENTS_PER_ORDER: Final = len(ORDER_STATUSES)

_CAPACITIES: Final[tuple[int, ...]] = (200, 500, 1000, 2000)


class InventoryEvent(TypedDict):
    """One labelled stock-change event (the schema Day 4's Kafka models mirror)."""

    sku_id: str
    warehouse_id: str
    timestamp: str
    previous_level: int
    new_level: int
    change_amount: int
    transaction_type: str
    is_anomaly: bool
    anomaly_type: AnomalyType | None
    metadata: dict[str, Any]


class OrderEvent(TypedDict):
    """One order-lifecycle event: an order reaching a new status."""

    order_id: str
    sku_id: str
    warehouse_id: str
    timestamp: str
    status: str
    previous_status: str | None
    quantity: int
    stage_duration_s: float | None
    expected_duration_s: float | None
    is_anomaly: bool
    anomaly_type: AnomalyType | None
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Item definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InventoryItem:
    """Static description of one stocked SKU.

    Attributes:
        sku_id: Unique id, e.g. ``"SKU-0001"``.
        warehouse_id: Warehouse holding the stock, e.g. ``"wh1"``.
        zone: Zone inside the warehouse (``"A"``, ``"B"``, ``"C"``).
        min_stock: Lowest normal stock level (units).
        max_stock: Highest normal stock level (units).
        reorder_point: Level at or below which restocks are triggered.
        initial_stock: Starting level; chosen at random inside the band if omitted.
    """

    sku_id: str
    warehouse_id: str
    zone: str
    min_stock: int
    max_stock: int
    reorder_point: int
    initial_stock: int | None = None

    def __post_init__(self) -> None:
        """Validate the definition."""
        for name in ("sku_id", "warehouse_id", "zone"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.min_stock < 0:
            raise ValueError(f"min_stock must be >= 0, got {self.min_stock}")
        if self.max_stock <= self.min_stock:
            raise ValueError(
                f"max_stock must exceed min_stock, got {self.min_stock} / {self.max_stock}"
            )
        if not self.min_stock <= self.reorder_point < self.max_stock:
            raise ValueError(
                f"reorder_point must satisfy min_stock <= reorder_point < max_stock, "
                f"got {self.reorder_point}"
            )
        if self.initial_stock is not None and not (
            self.min_stock <= self.initial_stock <= self.max_stock
        ):
            raise ValueError(
                f"initial_stock {self.initial_stock} must lie within "
                f"[{self.min_stock}, {self.max_stock}]"
            )

    @property
    def stock_range(self) -> int:
        """Width of the normal stock band."""
        return self.max_stock - self.min_stock

    @property
    def max_shipment(self) -> int:
        """Largest normal shipment (units)."""
        return max(1, round(_SHIPMENT_FRACTION * self.stock_range))

    @property
    def restock_quantity(self) -> int:
        """Typical restock lot size (units)."""
        return max(1, round(_RESTOCK_FRACTION * self.stock_range))

    @property
    def anomaly_floor(self) -> int:
        """Smallest size of a randomly injected anomaly (units)."""
        return max(_ANOMALY_FLOOR_SHIPMENTS * self.max_shipment, 3)


def build_default_items(
    n_skus: int = 50,
    *,
    n_warehouses: int = 2,
    zones: Sequence[str] = ("A", "B", "C"),
    seed: int | None = None,
) -> list[InventoryItem]:
    """Build a realistic SKU catalogue spread across warehouses and zones.

    Capacities are drawn from 200 / 500 / 1000 / 2000 units, the minimum stock is
    5 % of capacity and the reorder point sits 25 % of the way up the band. Ids
    look like ``SKU-0001``.

    Args:
        n_skus: Number of SKUs.
        n_warehouses: Warehouses (``wh1``, ``wh2``, ...) the SKUs are spread over.
        zones: Zone names inside every warehouse.
        seed: Seeds the capacity choices.

    Raises:
        ValueError: A count is not positive or ``zones`` is empty.
    """
    if n_skus < 1:
        raise ValueError(f"n_skus must be >= 1, got {n_skus}")
    if n_warehouses < 1:
        raise ValueError(f"n_warehouses must be >= 1, got {n_warehouses}")
    if not zones:
        raise ValueError("zones must not be empty")
    rng = random.Random(seed)
    items: list[InventoryItem] = []
    for index in range(n_skus):
        capacity = rng.choice(_CAPACITIES)
        minimum = round(0.05 * capacity)
        items.append(
            InventoryItem(
                sku_id=f"SKU-{index + 1:04d}",
                warehouse_id=f"wh{index % n_warehouses + 1}",
                zone=zones[(index // n_warehouses) % len(zones)],
                min_stock=minimum,
                max_stock=capacity,
                reorder_point=minimum + round(0.25 * (capacity - minimum)),
            )
        )
    return items


# --------------------------------------------------------------------------- #
# Inventory simulator
# --------------------------------------------------------------------------- #
@dataclass
class _PendingAnomaly:
    """A manually injected anomaly waiting for the SKU's next tick."""

    subtype: str
    magnitude: int | None
    episode_id: str


@dataclass
class _SkuState:
    """Mutable per-SKU simulation state."""

    item: InventoryItem
    rng: random.Random
    level: int
    sequence: int = 0
    anomalies_started: int = 0
    pending: _PendingAnomaly | None = None


class InventorySimulator:
    """Generate labelled stock-change events for a catalogue of SKUs.

    On every tick (``event_interval_s``, default 5 s) each SKU transacts with
    probability ``transaction_probability``; a transaction is a shipment, restock,
    return or adjustment - or, at ``anomaly_rate``, an anomaly. All events of one
    tick share the tick's timestamp.

    ``anomaly_rate`` is the approximate fraction of **all** events labelled
    anomalous, split between sudden drops and ghost stock by ``anomaly_mix``. An
    anomaly needs a big enough stock level (drops) or enough free capacity (ghost
    stock), so the realised fraction is slightly below the target.
    """

    def __init__(
        self,
        items: Sequence[InventoryItem] | None = None,
        *,
        transaction_probability: float = 0.25,
        anomaly_rate: float = 0.01,
        anomaly_mix: Sequence[float] = (0.6, 0.4),
        event_interval_s: float = 5.0,
        start_time: datetime | None = None,
        seed: int | None = None,
    ) -> None:
        """Create a simulator.

        Args:
            items: SKUs to simulate. Defaults to :func:`build_default_items`.
            transaction_probability: Chance per SKU per tick that it transacts.
            anomaly_rate: Target fraction of events labelled anomalous.
            anomaly_mix: Shares of ``(sudden_drop, ghost_stock)`` among anomalies;
                must sum to 1.
            event_interval_s: Simulated seconds between ticks.
            start_time: Simulated clock start (naive values are UTC). Defaults to now.
            seed: Makes the stream reproducible. ``None`` is non-deterministic.

        Raises:
            ValueError: Invalid probabilities, mix, interval, items or start time.
        """
        self._p_transaction = validate_rate("transaction_probability", transaction_probability)
        if self._p_transaction == 0.0:
            raise ValueError("transaction_probability must be greater than 0")
        rate = validate_rate("anomaly_rate", anomaly_rate)
        mix = validate_mix("anomaly_mix", anomaly_mix, 2)
        self._p_drop = rate * mix[0]
        self._p_ghost = rate * mix[1]
        self._interval = validate_positive("event_interval_s", event_interval_s)
        self._start = to_utc(start_time) if start_time else default_start()
        self._tick_index = 0
        self._seed = seed
        self._log = get_logger(__name__)

        catalogue = list(items) if items is not None else build_default_items(seed=seed)
        if not catalogue:
            raise ValueError("at least one item is required")
        self._states: dict[str, _SkuState] = {}
        for item in catalogue:
            if item.sku_id in self._states:
                raise ValueError(f"duplicate sku_id {item.sku_id!r}")
            rng = self._make_rng(item.sku_id)
            self._states[item.sku_id] = _SkuState(
                item=item, rng=rng, level=self._initial_level(item, rng)
            )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: InventorySimConfig,
        *,
        n_warehouses: int = 2,
        zones: Sequence[str] = ("A", "B", "C"),
        seed: int | None = None,
        start_time: datetime | None = None,
        items: Sequence[InventoryItem] | None = None,
    ) -> InventorySimulator:
        """Build a simulator from ``config.simulation.inventory``.

        Args:
            cfg: The ``simulation.inventory`` section of the project config.
            n_warehouses: Warehouses to spread SKUs over (usually the warehouse config's).
            zones: Zone names (usually the warehouse config's).
            seed: Usually ``config.simulation.seed``.
            start_time: Simulated clock start (defaults to now, UTC).
            items: Custom catalogue; defaults to ``cfg.n_skus`` built-in SKUs.
        """
        catalogue = (
            list(items)
            if items is not None
            else build_default_items(cfg.n_skus, n_warehouses=n_warehouses, zones=zones, seed=seed)
        )
        return cls(
            catalogue,
            transaction_probability=cfg.transaction_probability,
            anomaly_rate=cfg.anomaly_rate,
            event_interval_s=cfg.event_interval_s,
            start_time=start_time,
            seed=seed,
        )

    def _make_rng(self, sku_id: str) -> random.Random:
        """Return the SKU's private random stream (stable across catalogue changes)."""
        if self._seed is None:
            return random.Random()
        return random.Random(f"{self._seed}:{sku_id}")

    @staticmethod
    def _initial_level(item: InventoryItem, rng: random.Random) -> int:
        if item.initial_stock is not None:
            return item.initial_stock
        upper = max(item.reorder_point, item.min_stock + round(0.7 * item.stock_range))
        return rng.randint(item.reorder_point, upper)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def items(self) -> tuple[InventoryItem, ...]:
        """All simulated SKUs, in tick order."""
        return tuple(state.item for state in self._states.values())

    @property
    def event_interval_s(self) -> float:
        """Simulated seconds between ticks."""
        return self._interval

    @property
    def current_time(self) -> datetime:
        """Simulated timestamp of the *next* tick."""
        return self._time_at(self._tick_index)

    def get_item(self, sku_id: str) -> InventoryItem:
        """Return a SKU definition by id.

        Raises:
            KeyError: Unknown SKU id.
        """
        return self._state(sku_id).item

    def stock_level(self, sku_id: str) -> int:
        """Return the SKU's current simulated stock level.

        Raises:
            KeyError: Unknown SKU id.
        """
        return self._state(sku_id).level

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def tick(self) -> list[InventoryEvent]:
        """Advance the clock one interval and return the events of that tick.

        Only SKUs that transact (or have a pending injected anomaly) produce an
        event, so the list can be shorter than the catalogue - or empty.
        """
        stamp = isoformat(self._time_at(self._tick_index))
        self._tick_index += 1
        events: list[InventoryEvent] = []
        for state in self._states.values():
            event = self._step(state, stamp)
            if event is not None:
                events.append(event)
        return events

    def simulate_all_skus(
        self, n_ticks: int | None = None, duration_s: float | None = None
    ) -> Iterator[InventoryEvent]:
        """Yield stock events tick by tick, in timestamp order.

        With neither argument the stream is endless.

        Args:
            n_ticks: Number of ticks to simulate.
            duration_s: Simulated seconds to cover (rounded up to whole ticks).

        Raises:
            ValueError: Both arguments given, or a negative value.
        """
        if n_ticks is not None and duration_s is not None:
            raise ValueError("pass either n_ticks or duration_s, not both")
        if duration_s is not None:
            if duration_s < 0:
                raise ValueError(f"duration_s must be >= 0, got {duration_s}")
            n_ticks = math.ceil(duration_s / self._interval)
        if n_ticks is not None and n_ticks < 0:
            raise ValueError(f"n_ticks must be >= 0, got {n_ticks}")
        produced = 0
        while n_ticks is None or produced < n_ticks:
            yield from self.tick()
            produced += 1

    # ------------------------------------------------------------------ #
    # Anomaly injection
    # ------------------------------------------------------------------ #
    def inject_sudden_drop(self, sku_id: str, magnitude: int | None = None) -> str:
        """Force a large unexplained stock decrease at the SKU's next tick.

        The event is booked as an ``adjustment`` and labelled ``point``. The SKU
        transacts on the next tick regardless of ``transaction_probability``.
        Replaces a pending injection on the same SKU.

        Args:
            sku_id: Target SKU.
            magnitude: Units lost (clamped to the current stock). Random if omitted:
                20-50 % of the current level, but at least ``anomaly_floor`` units -
                or everything, when less than that is on hand.

        Returns:
            The anomaly's episode id (reported in the event's metadata).

        Raises:
            KeyError: Unknown SKU id.
            ValueError: ``magnitude`` is not positive, or the SKU has no stock to lose.
        """
        state = self._state(sku_id)
        self._validate_magnitude(magnitude)
        if state.level <= 0:
            raise ValueError(f"{sku_id} has no stock to lose")
        return self._queue(state, SUDDEN_DROP, magnitude)

    def inject_ghost_stock(self, sku_id: str, magnitude: int | None = None) -> str:
        """Force stock to appear without a physical arrival at the SKU's next tick.

        The event is booked as a ``restock`` and labelled ``contextual``. The SKU
        transacts on the next tick regardless of ``transaction_probability``.
        Replaces a pending injection on the same SKU.

        Args:
            sku_id: Target SKU.
            magnitude: Units added (clamped so the level does not exceed
                ``max_stock``). Random if omitted: ``anomaly_floor`` up to 45 % of the
                stock range, within the free capacity.

        Returns:
            The anomaly's episode id (reported in the event's metadata).

        Raises:
            KeyError: Unknown SKU id.
            ValueError: ``magnitude`` is not positive, or the SKU is already full.
        """
        state = self._state(sku_id)
        self._validate_magnitude(magnitude)
        if state.item.max_stock - state.level <= 0:
            raise ValueError(f"{sku_id} is already at or above max_stock")
        return self._queue(state, GHOST_STOCK, magnitude)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _state(self, sku_id: str) -> _SkuState:
        try:
            return self._states[sku_id]
        except KeyError:
            raise KeyError(f"unknown sku_id {sku_id!r}") from None

    def _time_at(self, tick_index: int) -> datetime:
        return self._start + timedelta(seconds=tick_index * self._interval)

    @staticmethod
    def _validate_magnitude(magnitude: int | None) -> None:
        if magnitude is not None and magnitude < 1:
            raise ValueError(f"magnitude must be >= 1, got {magnitude}")

    def _queue(self, state: _SkuState, subtype: str, magnitude: int | None) -> str:
        state.anomalies_started += 1
        episode_id = f"{state.item.sku_id}#{state.anomalies_started}"
        state.pending = _PendingAnomaly(subtype, magnitude, episode_id)
        self._log.debug(
            "anomaly_episode_started",
            sku_id=state.item.sku_id,
            anomaly_subtype=subtype,
            episode_id=episode_id,
            manual=True,
        )
        return episode_id

    def _step(self, state: _SkuState, stamp: str) -> InventoryEvent | None:
        rng = state.rng
        pending, state.pending = state.pending, None
        episode_id: str | None = None
        subtype: str | None = None
        outcome: tuple[int, str] | None = None

        if pending is not None:
            subtype, episode_id = pending.subtype, pending.episode_id
            outcome = self._anomaly_change(state, subtype, pending.magnitude)
        else:
            if rng.random() >= self._p_transaction:
                return None
            draw = rng.random()
            if draw < self._p_drop:
                candidate: str | None = SUDDEN_DROP
            elif draw < self._p_drop + self._p_ghost:
                candidate = GHOST_STOCK
            else:
                candidate = None
            if candidate is not None:
                outcome = self._anomaly_change(state, candidate, None, random_start=True)
                if outcome is not None:
                    subtype = candidate
                    state.anomalies_started += 1
                    episode_id = f"{state.item.sku_id}#{state.anomalies_started}"

        if outcome is None:
            subtype = None
            episode_id = None
            outcome = self._normal_change(state)
        change, transaction_type = outcome
        return self._emit(state, stamp, change, transaction_type, subtype, episode_id)

    def _emit(
        self,
        state: _SkuState,
        stamp: str,
        change: int,
        transaction_type: str,
        subtype: str | None,
        episode_id: str | None,
    ) -> InventoryEvent:
        item = state.item
        previous = state.level
        state.level = previous + change
        state.sequence += 1
        anomaly_type: AnomalyType | None = None
        if subtype == SUDDEN_DROP:
            anomaly_type = POINT
        elif subtype == GHOST_STOCK:
            anomaly_type = CONTEXTUAL
        return {
            "sku_id": item.sku_id,
            "warehouse_id": item.warehouse_id,
            "timestamp": stamp,
            "previous_level": previous,
            "new_level": state.level,
            "change_amount": change,
            "transaction_type": transaction_type,
            "is_anomaly": anomaly_type is not None,
            "anomaly_type": anomaly_type,
            "metadata": {
                "zone": item.zone,
                "min_stock": item.min_stock,
                "max_stock": item.max_stock,
                "reorder_point": item.reorder_point,
                "sequence": state.sequence,
                "anomaly_subtype": subtype,
                "episode_id": episode_id,
                "episode_step": 0 if subtype is not None else None,
            },
        }

    def _anomaly_change(
        self,
        state: _SkuState,
        subtype: str,
        magnitude: int | None,
        *,
        random_start: bool = False,
    ) -> tuple[int, str] | None:
        """Return ``(change, transaction_type)`` for an anomaly, or ``None`` if infeasible.

        Randomly started anomalies are skipped (``None``) when the SKU cannot
        absorb one cleanly; manual injections were validated up front and are
        clamped to what the SKU can take.
        """
        item, rng, level = state.item, state.rng, state.level
        floor = item.anomaly_floor
        if subtype == SUDDEN_DROP:
            if random_start and level < floor:
                return None
            if magnitude is None:
                share = rng.uniform(*_DROP_LEVEL_FRACTION)
                size = min(max(round(level * share), floor), level)
            else:
                size = min(magnitude, level)
            return (-size, ADJUSTMENT) if size >= 1 else None
        headroom = item.max_stock - level
        if random_start and (headroom < floor or level <= item.reorder_point):
            return None
        if magnitude is None:
            upper = min(headroom, max(floor, round(_GHOST_RANGE_FRACTION * item.stock_range)))
            size = rng.randint(min(floor, upper), upper) if upper >= 1 else 0
        else:
            size = min(magnitude, headroom)
        return (size, RESTOCK) if size >= 1 else None

    def _normal_change(self, state: _SkuState) -> tuple[int, str]:
        """Return a normal ``(change, transaction_type)`` that keeps stock in its band."""
        item, rng, level = state.item, state.rng, state.level
        if level <= item.reorder_point and (
            level <= item.min_stock or rng.random() < _RESTOCK_WHEN_LOW_PROBABILITY
        ):
            lot = max(1, round(item.restock_quantity * rng.uniform(*_RESTOCK_JITTER)))
            return min(lot, item.max_stock - level), RESTOCK
        roll = rng.random()
        if roll < _RETURN_PROBABILITY and level < item.max_stock:
            return min(rng.randint(1, _SMALL_CHANGE_MAX), item.max_stock - level), RETURN
        if roll < _RETURN_PROBABILITY + _ADJUSTMENT_PROBABILITY:
            delta = rng.randint(1, _SMALL_CHANGE_MAX) * rng.choice((-1, 1))
            delta = min(max(delta, item.min_stock - level), item.max_stock - level)
            if delta != 0:
                return delta, ADJUSTMENT
        shipment = min(rng.randint(1, item.max_shipment), level - item.min_stock)
        return -shipment, SHIPMENT


# --------------------------------------------------------------------------- #
# Order fulfilment simulator
# --------------------------------------------------------------------------- #
class OrderEventSimulator:
    """Generate order-lifecycle events with realistic fulfilment timestamps.

    Orders arrive as a Poisson process (``arrival_rate_per_s``). Each order walks
    through created -> picked -> packed -> shipped -> delivered; stage durations are
    lognormal around per-stage medians. Unlike the tick-based simulators this one is
    *event driven*: events come out one at a time in strict timestamp order, so
    ``next_event_time`` is always the timestamp of the next event.

    One order in ``anomaly_rate * 5`` gets a **fulfillment_delay** - a single stage
    takes 5-12x its median - so about ``anomaly_rate`` of all events are anomalous
    (capped at every order being delayed). Normal stage durations are truncated at
    3.5x the median, which keeps the labels exact. Only the delayed transition
    event is flagged; later stages of the same order are normal.
    """

    def __init__(
        self,
        items: Sequence[InventoryItem] | None = None,
        *,
        arrival_rate_per_s: float = 0.2,
        anomaly_rate: float = 0.02,
        stage_median_s: Sequence[float] = (90.0, 60.0, 120.0, 600.0),
        stage_sigma: float = 0.35,
        start_time: datetime | None = None,
        seed: int | None = None,
    ) -> None:
        """Create a simulator.

        Args:
            items: SKUs that orders are placed for. Defaults to :func:`build_default_items`.
            arrival_rate_per_s: Mean new orders per simulated second.
            anomaly_rate: Target fraction of events labelled anomalous.
            stage_median_s: Median seconds of the four stages
                (created->picked, picked->packed, packed->shipped, shipped->delivered).
            stage_sigma: Lognormal shape of the stage durations (0 = exactly the median).
            start_time: Simulated clock start (naive values are UTC). Defaults to now.
            seed: Makes the stream reproducible. ``None`` is non-deterministic.

        Raises:
            ValueError: Invalid rates, medians, sigma, items or start time.
        """
        self._rate = validate_positive("arrival_rate_per_s", arrival_rate_per_s)
        self._p_delay = min(1.0, validate_rate("anomaly_rate", anomaly_rate) * _EVENTS_PER_ORDER)
        medians = tuple(float(m) for m in stage_median_s)
        if len(medians) != _STAGES or any(m <= 0 for m in medians):
            raise ValueError(f"stage_median_s needs {_STAGES} positive values, got {medians}")
        self._medians = medians
        if stage_sigma < 0:
            raise ValueError(f"stage_sigma must be >= 0, got {stage_sigma}")
        self._sigma = float(stage_sigma)
        self._start = to_utc(start_time) if start_time else default_start()
        self._log = get_logger(__name__)

        catalogue = list(items) if items is not None else build_default_items(seed=seed)
        if not catalogue:
            raise ValueError("at least one item is required")
        self._items = catalogue
        self._rng = random.Random() if seed is None else random.Random(f"{seed}:orders")
        self._heap: list[tuple[datetime, int, OrderEvent]] = []
        self._counter = 0
        self._orders = 0
        self._next_arrival = self._start + timedelta(seconds=self._rng.expovariate(self._rate))

    @classmethod
    def from_config(
        cls,
        cfg: OrderSimConfig,
        *,
        items: Sequence[InventoryItem] | None = None,
        seed: int | None = None,
        start_time: datetime | None = None,
    ) -> OrderEventSimulator:
        """Build a simulator from ``config.simulation.order``.

        Args:
            cfg: The ``simulation.order`` section of the project config.
            items: SKUs to order; defaults to the built-in catalogue.
            seed: Usually ``config.simulation.seed``.
            start_time: Simulated clock start (defaults to now, UTC).
        """
        return cls(
            items,
            arrival_rate_per_s=cfg.arrival_rate_per_s,
            anomaly_rate=cfg.anomaly_rate,
            start_time=start_time,
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    @property
    def next_event_time(self) -> datetime:
        """Timestamp of the event :meth:`next_event` will return."""
        if self._heap and self._heap[0][0] <= self._next_arrival:
            return self._heap[0][0]
        return self._next_arrival

    @property
    def orders_created(self) -> int:
        """Number of orders created so far."""
        return self._orders

    @property
    def in_flight(self) -> int:
        """Number of events still scheduled for orders that are already created."""
        return len(self._heap)

    def next_event(self) -> OrderEvent:
        """Return the next order event in timestamp order (the stream never ends)."""
        if self._heap and self._heap[0][0] <= self._next_arrival:
            return heapq.heappop(self._heap)[2]
        return self._create_order()

    def simulate(
        self, n_events: int | None = None, duration_s: float | None = None
    ) -> Iterator[OrderEvent]:
        """Yield order events in timestamp order.

        With neither argument the stream is endless. ``duration_s`` counts from the
        simulator's start time and stops before the first event at or after it.

        Args:
            n_events: Number of events to yield.
            duration_s: Simulated seconds (from ``start_time``) to cover.

        Raises:
            ValueError: Both arguments given, or a negative value.
        """
        if n_events is not None and duration_s is not None:
            raise ValueError("pass either n_events or duration_s, not both")
        if n_events is not None and n_events < 0:
            raise ValueError(f"n_events must be >= 0, got {n_events}")
        if duration_s is not None and duration_s < 0:
            raise ValueError(f"duration_s must be >= 0, got {duration_s}")
        end = self._start + timedelta(seconds=duration_s) if duration_s is not None else None
        produced = 0
        while n_events is None or produced < n_events:
            if end is not None and self.next_event_time >= end:
                return
            yield self.next_event()
            produced += 1

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _stage_duration(self, stage: int, *, delayed: bool) -> float:
        rng, median = self._rng, self._medians[stage]
        if delayed:
            return median * rng.uniform(*_DELAY_MULTIPLIER)
        for _ in range(_MAX_RESAMPLES):
            duration = median * math.exp(rng.gauss(0.0, self._sigma))
            if duration <= _NORMAL_DURATION_CAP * median:
                return duration
        return median

    def _create_order(self) -> OrderEvent:
        rng = self._rng
        created_at = self._next_arrival
        self._next_arrival = created_at + timedelta(seconds=rng.expovariate(self._rate))
        self._orders += 1
        order_id = f"ORD-{self._orders:06d}"
        item = rng.choice(self._items)
        quantity = rng.randint(1, item.max_shipment)
        delayed_stage = rng.randrange(_STAGES) if rng.random() < self._p_delay else None

        def event(
            status_index: int,
            moment: datetime,
            duration: float | None,
            expected: float | None,
            delayed: bool,
        ) -> OrderEvent:
            return {
                "order_id": order_id,
                "sku_id": item.sku_id,
                "warehouse_id": item.warehouse_id,
                "timestamp": isoformat(moment),
                "status": ORDER_STATUSES[status_index],
                "previous_status": ORDER_STATUSES[status_index - 1] if status_index else None,
                "quantity": quantity,
                "stage_duration_s": None if duration is None else round(duration, 3),
                "expected_duration_s": expected,
                "is_anomaly": delayed,
                "anomaly_type": POINT if delayed else None,
                "metadata": {
                    "zone": item.zone,
                    "anomaly_subtype": FULFILLMENT_DELAY if delayed else None,
                    "episode_id": f"{order_id}#1" if delayed else None,
                    "episode_step": 0 if delayed else None,
                },
            }

        moment = created_at
        for stage in range(_STAGES):
            delayed = stage == delayed_stage
            duration = self._stage_duration(stage, delayed=delayed)
            moment = moment + timedelta(seconds=duration)
            self._counter += 1
            heapq.heappush(
                self._heap,
                (
                    moment,
                    self._counter,
                    event(stage + 1, moment, duration, self._medians[stage], delayed),
                ),
            )
        return event(0, created_at, None, None, False)
