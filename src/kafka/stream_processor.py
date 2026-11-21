"""Sliding-window aggregation of the event streams (Day 5).

:class:`WindowedAggregator` keeps, for every entity (sensor / vehicle / SKU) and every
configured window length (1, 5 and 15 minutes by default), a sliding window over the
entity's recent readings and turns each new event into one :class:`WindowedFeatures` per
window length.

Event time, not wall time
-------------------------
Windows are anchored on the *event timestamps*, so a 10x fast-forward simulation or a
replay of old data produces the same features as real time. The anchor of an entity is the
newest event time seen for it; a window covers ``(anchor - W, anchor]``.

Late events and the watermark
-----------------------------
The watermark of an entity is ``anchor - watermark_tolerance_s``:

* ``ts >= anchor``                  in order: merged into every window, features emitted
* ``watermark <= ts < anchor``      late but tolerated: merged into the windows (kept in time
  order) but nothing is emitted; the next in-order event reflects it
* ``ts < watermark``                too late: dropped and reported as ``dropped_late``

Windows only move forward, so a point is evicted once it falls behind ``anchor - W`` and can
never be needed again. Entities that stop sending are aged out with :meth:`evict_idle`.

Statistics are maintained incrementally (running sums, a sorted value list), so one event
costs roughly O(log n) plus a memmove instead of a pass over the window. The running sums
are re-centred and recomputed exactly every few hundred evictions to stop floating point
drift. Per window and signal:

* ``mean``, ``std`` (sample, ``ddof=1``; ``0`` for a single point), ``min``, ``max``
* ``rate_of_change``: least-squares slope in signal units per second (``0`` below 2 points)
* ``percentile_95``: linear interpolation, like ``numpy.percentile``
* ``last``: value of the newest point

Ground-truth labels of the triggering event are carried along as ``label_*`` fields. They
exist only so offline evaluation and supervised training can join features to labels; they
must never be used as model inputs.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_left, insort
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Literal

from src.kafka.schemas import GPSReading, InventoryEvent, KafkaMessage, WarehouseReading

DEFAULT_WINDOWS: Final[tuple[int, ...]] = (60, 300, 900)
#: Signals aggregated per source type, in a fixed order (Day 6 flattens them in this order).
SIGNALS_BY_SOURCE: Final[dict[str, tuple[str, ...]]] = {
    "warehouse": ("value",),
    "vehicle": ("speed_kmh", "route_adherence_score"),
    "inventory": ("new_level", "change_amount"),
}
_REFRESH_EVERY: Final[int] = 512  # evictions between exact recomputations of the sums

Outcome = Literal["emitted", "late_accepted", "dropped_late", "invalid", "unsupported"]


# --------------------------------------------------------------------------- #
# Output types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SignalStats:
    """Statistics of one signal over one window."""

    mean: float
    std: float
    min: float
    max: float
    rate_of_change: float
    percentile_95: float
    last: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly dict."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "rate_of_change": self.rate_of_change,
            "percentile_95": self.percentile_95,
            "last": self.last,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> SignalStats:
        """Rebuild from :meth:`to_dict` output."""
        return cls(**{name: float(data[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class WindowedFeatures:
    """Features of one entity over one window, produced for one triggering event.

    Attributes:
        entity_id: Sensor, vehicle or SKU id.
        source_type: ``warehouse``, ``vehicle`` or ``inventory``.
        window_seconds: Window length.
        window_start: ``window_end - window_seconds`` (UTC).
        window_end: Event time of the triggering event (UTC).
        count: Points inside the window.
        coverage_seconds: Time spanned by those points; below ``window_seconds`` while the
            window is still warming up (cold start).
        signals: Statistics per signal name (see :data:`SIGNALS_BY_SOURCE`).
        context: Descriptive fields of the newest event (zone, unit, acceptable range, ...).
        label_is_anomaly: Ground truth of the triggering event. Evaluation only.
        label_anomaly_type: Ground-truth anomaly type of the triggering event. Evaluation only.
    """

    entity_id: str
    source_type: str
    window_seconds: int
    window_start: datetime
    window_end: datetime
    count: int
    coverage_seconds: float
    signals: dict[str, SignalStats]
    context: dict[str, Any] = field(default_factory=dict)
    label_is_anomaly: bool = False
    label_anomaly_type: str | None = None

    @property
    def is_warm(self) -> bool:
        """True once the window spans at least 90% of its length."""
        return self.coverage_seconds >= 0.9 * self.window_seconds

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (what a Redis-backed cache would store)."""
        return {
            "entity_id": self.entity_id,
            "source_type": self.source_type,
            "window_seconds": self.window_seconds,
            "window_start": self.window_start.isoformat().replace("+00:00", "Z"),
            "window_end": self.window_end.isoformat().replace("+00:00", "Z"),
            "count": self.count,
            "coverage_seconds": self.coverage_seconds,
            "signals": {name: stats.to_dict() for name, stats in self.signals.items()},
            "context": dict(self.context),
            "label_is_anomaly": self.label_is_anomaly,
            "label_anomaly_type": self.label_anomaly_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WindowedFeatures:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            entity_id=data["entity_id"],
            source_type=data["source_type"],
            window_seconds=int(data["window_seconds"]),
            window_start=parse_event_datetime(data["window_start"]),
            window_end=parse_event_datetime(data["window_end"]),
            count=int(data["count"]),
            coverage_seconds=float(data["coverage_seconds"]),
            signals={n: SignalStats.from_dict(s) for n, s in data["signals"].items()},
            context=dict(data.get("context", {})),
            label_is_anomaly=bool(data.get("label_is_anomaly", False)),
            label_anomaly_type=data.get("label_anomaly_type"),
        )


@dataclass(frozen=True, slots=True)
class UpdateOutcome:
    """What the aggregator did with one event."""

    status: Outcome
    features: tuple[WindowedFeatures, ...] = ()


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def parse_event_datetime(value: str) -> datetime:
    """Parse an ISO 8601 string (``Z`` or offset; naive means UTC) to an aware UTC datetime."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def parse_event_time(value: str) -> float:
    """Parse an ISO 8601 string to UTC epoch seconds."""
    return parse_event_datetime(value).timestamp()


def _from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


# --------------------------------------------------------------------------- #
# One sliding window over one signal
# --------------------------------------------------------------------------- #
class _SlidingWindow:
    """Time-ordered points plus incrementally maintained statistics."""

    __slots__ = (
        "_n_evictions",
        "_stt",
        "_st",
        "_stx",
        "_sx",
        "_sxx",
        "_tr",
        "_xr",
        "max_points",
        "points",
        "sorted_values",
        "span",
    )

    def __init__(self, span_s: float, max_points: int) -> None:
        self.span = span_s
        self.max_points = max_points
        self.points: list[tuple[float, float]] = []  # (ts, value), ascending by ts
        self.sorted_values: list[float] = []
        self._reset_sums(0.0, 0.0)
        self._n_evictions = 0

    def _reset_sums(self, tr: float, xr: float) -> None:
        self._tr, self._xr = tr, xr
        self._st = self._stt = self._sx = self._sxx = self._stx = 0.0

    def _accumulate(self, ts: float, x: float, sign: float) -> None:
        dt, dx = ts - self._tr, x - self._xr
        self._st += sign * dt
        self._stt += sign * dt * dt
        self._sx += sign * dx
        self._sxx += sign * dx * dx
        self._stx += sign * dt * dx

    def _recompute(self) -> None:
        if not self.points:
            self._reset_sums(0.0, 0.0)
        else:
            self._reset_sums(self.points[0][0], self.points[0][1])
            for ts, x in self.points:
                self._accumulate(ts, x, 1.0)
        self._n_evictions = 0

    def add(self, ts: float, x: float) -> None:
        """Insert a point (kept in time order)."""
        if not self.points:
            self._reset_sums(ts, x)
        insort(self.points, (ts, x))
        insort(self.sorted_values, x)
        self._accumulate(ts, x, 1.0)
        if len(self.points) > self.max_points:
            self._remove_oldest()

    def _remove_oldest(self) -> None:
        ts, x = self.points.pop(0)
        del self.sorted_values[bisect_left(self.sorted_values, x)]
        self._accumulate(ts, x, -1.0)
        self._n_evictions += 1
        if not self.points or self._n_evictions >= _REFRESH_EVERY:
            self._recompute()

    def evict(self, anchor: float) -> None:
        """Drop points at or before ``anchor - span``."""
        cutoff = anchor - self.span
        while self.points and self.points[0][0] <= cutoff:
            self._remove_oldest()

    def accepts(self, ts: float, anchor: float) -> bool:
        """Return whether ``ts`` falls inside the window ending at ``anchor``."""
        return ts > anchor - self.span

    def stats(self) -> SignalStats:
        """Statistics of the current contents (the window must not be empty)."""
        n = len(self.points)
        mean = self._xr + self._sx / n
        std = 0.0
        slope = 0.0
        if n > 1:
            std = math.sqrt(max((self._sxx - self._sx * self._sx / n) / (n - 1), 0.0))
            denom = n * self._stt - self._st * self._st
            if denom > 1e-9:
                slope = (n * self._stx - self._st * self._sx) / denom
        values = self.sorted_values
        pos = 0.95 * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        p95 = values[lo] + (values[hi] - values[lo]) * (pos - lo)
        return SignalStats(
            mean=mean,
            std=std,
            min=values[0],
            max=values[-1],
            rate_of_change=slope,
            percentile_95=p95,
            last=self.points[-1][1],
        )


@dataclass(slots=True)
class _EntityState:
    source_type: str
    anchor: float
    windows: dict[str, list[_SlidingWindow]]  # signal -> one window per configured length


# --------------------------------------------------------------------------- #
# Extraction from Kafka envelopes
# --------------------------------------------------------------------------- #
def extract_signals(message: KafkaMessage) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Return ``(signals, context)`` for an envelope, or ``None`` if it is not aggregated.

    Order events return ``None``: they are one-off per order id, so a per-entity window is
    meaningless.
    """
    payload = message.payload
    if isinstance(payload, WarehouseReading):
        return (
            {"value": payload.value},
            {
                **payload.metadata,
                "warehouse_id": payload.warehouse_id,
                "reading_type": payload.reading_type,
                "unit": payload.unit,
            },
        )
    if isinstance(payload, GPSReading):
        return (
            {
                "speed_kmh": payload.speed_kmh,
                "route_adherence_score": payload.route_adherence_score,
            },
            {
                **payload.metadata,
                "latitude": payload.latitude,
                "longitude": payload.longitude,
                "heading": payload.heading,
            },
        )
    if isinstance(payload, InventoryEvent):
        return (
            {"new_level": float(payload.new_level), "change_amount": float(payload.change_amount)},
            {
                **payload.metadata,
                "warehouse_id": payload.warehouse_id,
                "transaction_type": payload.transaction_type,
            },
        )
    return None


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
class WindowedAggregator:
    """Per-entity sliding windows over several window lengths. Thread-safe."""

    def __init__(
        self,
        window_seconds: Sequence[int] = DEFAULT_WINDOWS,
        watermark_tolerance_s: float = 30.0,
        *,
        max_points_per_window: int = 5_000,
        max_entities: int = 50_000,
    ) -> None:
        """Create the aggregator.

        Args:
            window_seconds: Window lengths in seconds (deduplicated and sorted).
            watermark_tolerance_s: How far behind an entity's newest event a late event may
                be and still be accepted.
            max_points_per_window: Safety bound on memory; the oldest point is dropped when
                a window would exceed it.
            max_entities: Entities kept at once; the least recently updated one is dropped
                when exceeded.

        Raises:
            ValueError: A window length is not positive, or a bound is out of range.
        """
        windows = sorted(set(window_seconds))
        if not windows or windows[0] <= 0:
            raise ValueError(f"window_seconds must be positive integers, got {window_seconds}")
        if watermark_tolerance_s < 0:
            raise ValueError("watermark_tolerance_s must be >= 0")
        if max_points_per_window < 2 or max_entities < 1:
            raise ValueError("max_points_per_window must be >= 2 and max_entities >= 1")
        self.window_seconds: tuple[int, ...] = tuple(windows)
        self.watermark_tolerance_s = float(watermark_tolerance_s)
        self._max_points = max_points_per_window
        self._max_entities = max_entities
        self._states: OrderedDict[str, _EntityState] = OrderedDict()
        self._global_max_ts = -math.inf
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._states)

    @property
    def entity_count(self) -> int:
        """Number of entities with live windows."""
        return len(self._states)

    # ------------------------------------------------------------------ #
    def process_message(self, message: KafkaMessage) -> UpdateOutcome:
        """Aggregate one Kafka envelope (see :func:`extract_signals`)."""
        extracted = extract_signals(message)
        if extracted is None:
            return UpdateOutcome("unsupported")
        signals, context = extracted
        payload = message.payload
        try:
            event_time = parse_event_time(message.timestamp)
        except ValueError:
            return UpdateOutcome("invalid")
        return self.update(
            message.entity_id,
            message.source_type,
            event_time,
            signals,
            context=context,
            is_anomaly=payload.is_anomaly,
            anomaly_type=getattr(payload, "anomaly_type", None),
        )

    def update(
        self,
        entity_id: str,
        source_type: str,
        event_time: float,
        signals: Mapping[str, float],
        *,
        context: Mapping[str, Any] | None = None,
        is_anomaly: bool = False,
        anomaly_type: str | None = None,
    ) -> UpdateOutcome:
        """Merge one event into the entity's windows.

        Args:
            entity_id: Sensor, vehicle or SKU id.
            source_type: ``warehouse``, ``vehicle`` or ``inventory``.
            event_time: Event time as UTC epoch seconds.
            signals: Signal name to value. Every event of an entity must carry the same names.
            context: Descriptive fields of this event, passed through to the output.
            is_anomaly: Ground-truth label (evaluation only).
            anomaly_type: Ground-truth anomaly type (evaluation only).

        Returns:
            :class:`UpdateOutcome`; ``features`` holds one item per window length when the
            status is ``emitted``.
        """
        if not math.isfinite(event_time) or not signals:
            return UpdateOutcome("invalid")
        if any(not math.isfinite(v) for v in signals.values()):
            return UpdateOutcome("invalid")

        with self._lock:
            state = self._states.get(entity_id)
            if state is None:
                state = _EntityState(
                    source_type=source_type,
                    anchor=event_time,
                    windows={
                        name: [_SlidingWindow(w, self._max_points) for w in self.window_seconds]
                        for name in signals
                    },
                )
                self._states[entity_id] = state
                if len(self._states) > self._max_entities:
                    self._states.popitem(last=False)
            elif set(signals) != set(state.windows):
                return UpdateOutcome("invalid")
            self._states.move_to_end(entity_id)

            if event_time < state.anchor - self.watermark_tolerance_s:
                return UpdateOutcome("dropped_late")

            in_order = event_time >= state.anchor
            if in_order:
                state.anchor = event_time
            self._global_max_ts = max(self._global_max_ts, state.anchor)

            for name, value in signals.items():
                for window in state.windows[name]:
                    if window.accepts(event_time, state.anchor):
                        window.add(event_time, float(value))
                    if in_order:
                        window.evict(state.anchor)
            if not in_order:
                return UpdateOutcome("late_accepted")
            return UpdateOutcome(
                "emitted",
                self._build_features(
                    entity_id, state, dict(context or {}), is_anomaly, anomaly_type
                ),
            )

    def _build_features(
        self,
        entity_id: str,
        state: _EntityState,
        context: dict[str, Any],
        is_anomaly: bool,
        anomaly_type: str | None,
    ) -> tuple[WindowedFeatures, ...]:
        first_signal = next(iter(state.windows.values()))
        out: list[WindowedFeatures] = []
        for idx, span in enumerate(self.window_seconds):
            points = first_signal[idx].points
            out.append(
                WindowedFeatures(
                    entity_id=entity_id,
                    source_type=state.source_type,
                    window_seconds=span,
                    window_start=_from_epoch(state.anchor - span),
                    window_end=_from_epoch(state.anchor),
                    count=len(points),
                    coverage_seconds=state.anchor - points[0][0],
                    signals={name: wins[idx].stats() for name, wins in state.windows.items()},
                    context=context,
                    label_is_anomaly=is_anomaly,
                    label_anomaly_type=anomaly_type,
                )
            )
        return tuple(out)

    # ------------------------------------------------------------------ #
    def evict_idle(self, now: float | None = None) -> int:
        """Forget entities that sent nothing for longer than the largest window plus tolerance.

        This is what ages out a sensor that dropped off the network. ``now`` is event time
        in epoch seconds; by default the newest event time seen across all entities.

        Returns:
            The number of entities removed.
        """
        with self._lock:
            reference = self._global_max_ts if now is None else now
            if not math.isfinite(reference):
                return 0
            horizon = reference - (self.window_seconds[-1] + self.watermark_tolerance_s)
            stale = [eid for eid, st in self._states.items() if st.anchor < horizon]
            for eid in stale:
                del self._states[eid]
            return len(stale)

    def reset(self) -> None:
        """Drop all state."""
        with self._lock:
            self._states.clear()
            self._global_max_ts = -math.inf
