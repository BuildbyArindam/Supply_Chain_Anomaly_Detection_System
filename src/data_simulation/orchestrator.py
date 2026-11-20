"""Unified simulation stream across all data sources (Day 3).

:class:`SimulationOrchestrator` coordinates the warehouse, GPS and inventory
simulators (and, optionally, the order-fulfilment simulator) and merges their
output into one stream in strict timestamp order. Each event is wrapped in a small
envelope that carries the ``source_type`` tag - the contract the Kafka producer
(Day 4) routes on::

    {
        "source_type": "warehouse",            # warehouse | vehicle | inventory | order
        "entity_id": "wh1-A-temp-01",          # sensor_id / vehicle_id / sku_id / order_id
        "timestamp": "2026-11-20T08:00:00Z",   # same as payload["timestamp"]
        "payload": {...},                      # the simulator's own record, unchanged
    }

The sources tick at different rates (warehouse 1 s, GPS 10 s, inventory 5 s by
default). The orchestrator keeps one "next event time" per source and always
advances the source that is due first, so nothing is generated ahead of time:
manual ``inject_*`` calls on any simulator take effect at that source's very next
tick, even in the middle of a stream.

``speed_multiplier`` paces the stream against the wall clock: ``1.0`` is real time,
``10.0`` is ten times faster. Pass ``throttle=False`` (or use :meth:`run_batch`)
to generate as fast as the CPU allows, e.g. for EDA and model training.

Example::

    from src.data_simulation.orchestrator import SimulationOrchestrator
    from src.utils.config import get_config

    orchestrator = SimulationOrchestrator.from_config(get_config().simulation)
    for event in orchestrator.stream(duration_s=60):     # one real minute, paced
        producer.send(event)

    history = orchestrator.run_batch(duration_s=3600)    # one simulated hour, instantly
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

from src.data_simulation._common import validate_positive
from src.data_simulation.inventory_simulator import (
    InventorySimulator,
    OrderEventSimulator,
)
from src.data_simulation.vehicle_simulator import GPSSimulator
from src.data_simulation.warehouse_simulator import WarehouseSimulator
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config import SimulationConfig

SourceType = Literal["warehouse", "vehicle", "inventory", "order"]
SOURCE_TYPES: Final[tuple[SourceType, ...]] = ("warehouse", "vehicle", "inventory", "order")


class SimulationEvent(TypedDict):
    """Envelope around one simulator record, tagged with its source."""

    source_type: SourceType
    entity_id: str
    timestamp: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class _Source:
    """Adapter that gives every simulator the same two-method interface."""

    source_type: SourceType
    entity_key: str
    next_time: Callable[[], datetime]
    emit: Callable[[], Sequence[Mapping[str, Any]]]


class SimulationOrchestrator:
    """Merge several simulators into one time-ordered, optionally paced stream."""

    def __init__(
        self,
        warehouse: WarehouseSimulator,
        vehicles: GPSSimulator,
        inventory: InventorySimulator,
        orders: OrderEventSimulator | None = None,
        *,
        speed_multiplier: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create an orchestrator from ready-made simulators.

        Args:
            warehouse: Warehouse sensor simulator.
            vehicles: GPS vehicle simulator.
            inventory: Inventory stock simulator.
            orders: Optional order-fulfilment simulator.
            speed_multiplier: Simulated seconds per wall-clock second when throttled
                (``1.0`` real time, ``10.0`` ten times faster).
            clock: Monotonic wall-clock source in seconds (injectable for tests).
            sleep: Function used to wait for the wall clock (injectable for tests).

        Raises:
            ValueError: ``speed_multiplier`` is not positive.
        """
        self._speed = validate_positive("speed_multiplier", speed_multiplier)
        self._clock = clock
        self._sleep = sleep
        self._log = get_logger(__name__)
        self._warehouse = warehouse
        self._vehicles = vehicles
        self._inventory = inventory
        self._orders = orders
        self._sources: list[_Source] = [
            _Source("warehouse", "sensor_id", lambda: warehouse.current_time, warehouse.tick),
            _Source("vehicle", "vehicle_id", lambda: vehicles.current_time, vehicles.tick),
            _Source("inventory", "sku_id", lambda: inventory.current_time, inventory.tick),
        ]
        if orders is not None:
            self._sources.append(
                _Source(
                    "order",
                    "order_id",
                    lambda: orders.next_event_time,
                    lambda: [orders.next_event()],
                )
            )
        self._counts: dict[SourceType, dict[str, int]] = {
            source.source_type: {"events": 0, "anomalies": 0} for source in self._sources
        }

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: SimulationConfig,
        *,
        start_time: datetime | None = None,
        include_orders: bool | None = None,
        speed_multiplier: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> SimulationOrchestrator:
        """Build every simulator from ``config.simulation`` and wire them together.

        All simulators share ``start_time`` and ``cfg.seed``. SKUs are spread over the
        same warehouses and zones as the warehouse sensors.

        Args:
            cfg: The ``simulation`` section of the project config.
            start_time: Simulated clock start (defaults to now, UTC).
            include_orders: Add the order-fulfilment stream. Defaults to ``cfg.order.enabled``.
            speed_multiplier: Overrides ``cfg.speed_multiplier``.
            clock: Wall-clock source (injectable for tests).
            sleep: Wait function (injectable for tests).
        """
        seed = cfg.seed
        warehouse = WarehouseSimulator.from_config(cfg.warehouse, seed=seed, start_time=start_time)
        # Share one resolved start time so every source begins at the same instant.
        start = warehouse.current_time
        vehicles = GPSSimulator.from_config(cfg.vehicle, seed=seed, start_time=start)
        inventory = InventorySimulator.from_config(
            cfg.inventory,
            n_warehouses=cfg.warehouse.n_warehouses,
            zones=cfg.warehouse.zones,
            seed=seed,
            start_time=start,
        )
        want_orders = cfg.order.enabled if include_orders is None else include_orders
        orders = (
            OrderEventSimulator.from_config(
                cfg.order, items=list(inventory.items), seed=seed, start_time=start
            )
            if want_orders
            else None
        )
        return cls(
            warehouse,
            vehicles,
            inventory,
            orders,
            speed_multiplier=(
                speed_multiplier if speed_multiplier is not None else cfg.speed_multiplier
            ),
            clock=clock,
            sleep=sleep,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def warehouse(self) -> WarehouseSimulator:
        """The warehouse sensor simulator (use its ``inject_*`` methods)."""
        return self._warehouse

    @property
    def vehicles(self) -> GPSSimulator:
        """The GPS vehicle simulator (use its ``inject_*`` methods)."""
        return self._vehicles

    @property
    def inventory(self) -> InventorySimulator:
        """The inventory simulator (use its ``inject_*`` methods)."""
        return self._inventory

    @property
    def orders(self) -> OrderEventSimulator | None:
        """The order-fulfilment simulator, if enabled."""
        return self._orders

    @property
    def speed_multiplier(self) -> float:
        """Simulated seconds per wall-clock second when the stream is throttled."""
        return self._speed

    @property
    def current_time(self) -> datetime:
        """Simulated timestamp of the next event the stream will produce."""
        return min(source.next_time() for source in self._sources)

    @property
    def stats(self) -> dict[str, dict[str, int]]:
        """Events and ground-truth anomalies emitted so far, per source type."""
        return {name: dict(counts) for name, counts in self._counts.items()}

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def stream(
        self, duration_s: float | None = None, *, throttle: bool = True
    ) -> Iterator[SimulationEvent]:
        """Yield events from all sources in timestamp order.

        Events are generated lazily, one source tick at a time. Calling ``stream``
        again continues where the previous call stopped. To stop after a number of
        events instead of a duration, wrap the stream in :func:`itertools.islice`.

        Args:
            duration_s: Simulated seconds to cover, counted from the first event.
                Events at or after that instant are not produced. ``None`` streams forever.
            throttle: Sleep so that simulated time advances at ``speed_multiplier``
                times wall-clock time. ``False`` generates as fast as possible.

        Raises:
            ValueError: ``duration_s`` is negative.
        """
        if duration_s is not None and duration_s < 0:
            raise ValueError(f"duration_s must be >= 0, got {duration_s}")
        return self._stream(duration_s, throttle)

    def run_batch(self, duration_s: float) -> list[SimulationEvent]:
        """Generate ``duration_s`` simulated seconds as fast as possible and return them all.

        Handy for EDA and model training. Memory grows with the duration: one
        simulated hour with the default configuration is roughly 50 000 events.

        Raises:
            ValueError: ``duration_s`` is negative.
        """
        return list(self.stream(duration_s, throttle=False))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _next_source(self) -> _Source:
        """Return the source whose next event is due first (ties go to the earliest source)."""
        return min(self._sources, key=lambda source: source.next_time())

    def _stream(self, duration_s: float | None, throttle: bool) -> Iterator[SimulationEvent]:
        origin = self.current_time
        end = origin + timedelta(seconds=duration_s) if duration_s is not None else None
        wall_start = self._clock()
        emitted = 0
        self._log.info(
            "simulation_stream_started",
            start=origin.isoformat(),
            duration_s=duration_s,
            speed_multiplier=self._speed,
            throttled=throttle,
            sources=[source.source_type for source in self._sources],
        )
        try:
            while True:
                source = self._next_source()
                moment = source.next_time()
                if end is not None and moment >= end:
                    return
                if throttle:
                    due = wall_start + (moment - origin).total_seconds() / self._speed
                    delay = due - self._clock()
                    if delay > 0:
                        self._sleep(delay)
                counts = self._counts[source.source_type]
                for record in source.emit():
                    counts["events"] += 1
                    counts["anomalies"] += bool(record["is_anomaly"])
                    emitted += 1
                    yield {
                        "source_type": source.source_type,
                        "entity_id": str(record[source.entity_key]),
                        "timestamp": str(record["timestamp"]),
                        "payload": dict(record),
                    }
        finally:
            self._log.info("simulation_stream_stopped", events=emitted, totals=self.stats)
