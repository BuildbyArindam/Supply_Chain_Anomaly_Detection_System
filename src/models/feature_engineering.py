"""Feature engineering and preprocessing: windowed features to ML-ready tensors (Day 6).

The Day 5 stream processor turns every event into one :class:`WindowedFeatures` per window
length (1, 5 and 15 minutes by default) and the :class:`FeatureCache` keeps their history.
This module turns those into fixed-width numeric vectors and LSTM sequences::

    FeatureCache ──> WindowBundle ──> FeatureEngineer ──> raw vector ──> OnlineScaler
    (3 windows,      (aligned by       TimeSeriesFeatures                      │
     per entity)      window_end)      + TabularContextFeatures                ▼
                                                                 SequenceBuilder
                                                                 (lookback, n_features)

Design decisions
----------------
* **One schema per source type.** Warehouse, vehicle and inventory events carry different
  signals, so each source type has its own named feature list
  (:func:`feature_names`) and its own :class:`OnlineScaler`. Train one model per source
  type, or pad to a common width yourself.
* **Features are pure functions of the cached windows** (plus :class:`EntityHistory`). The
  "rolling historical baseline" of the z-score is the longest window (15 min by default).
  Because of this, :class:`SequenceBuilder` can rebuild from the cache exactly the vectors
  the online path computed, so there is no training / serving skew.
* **No label leakage.** ``label_*`` fields of :class:`WindowedFeatures` and the simulator
  ground-truth metadata (``episode_id``, ``episode_step``, ``anomaly_subtype``) are never
  read. "Time since last anomaly", the 7-day anomaly rate and the reliability score come
  from what the *detectors flagged* and what *analysts reported*, recorded in
  :class:`EntityHistory`.
* **Unknown is NaN, not zero.** A feature that cannot be computed (a sensor without a
  calibration date, a range that is not in the context) is ``NaN`` in the raw vector. The
  scaler skips NaNs; :class:`SequenceBuilder` forward-fills them from the previous step and
  falls back to ``0.0`` (the mean of a standardised feature) when nothing precedes.
* **Time is UTC**, like the simulators.
"""

from __future__ import annotations

import json
import math
import threading
from bisect import bisect_right, insort
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch

from src.kafka.feature_cache import FeatureStore
from src.kafka.stream_processor import DEFAULT_WINDOWS, SIGNALS_BY_SOURCE, WindowedFeatures

if TYPE_CHECKING:
    from src.utils.config import AppConfig

DEFAULT_ZONES: Final[tuple[str, ...]] = ("A", "B", "C")
#: Z-scores and ratios are clipped to this magnitude so one extreme event cannot dominate.
Z_CLIP: Final[float] = 20.0
RATIO_CAP: Final[float] = 20.0
#: Windows with fewer points than this give no baseline (the z-score is reported as 0).
MIN_BASELINE_COUNT: Final[int] = 20
#: Seconds reported as "time since last anomaly" when the entity never had one.
NEVER_ANOMALOUS_S: Final[float] = 86_400.0
#: Vehicle speeds stay in ``(0.6, 1.2) x cruise`` when healthy (see ``GPSSimulator``).
VEHICLE_SPEED_BAND: Final[tuple[float, float]] = (0.6, 1.2)
#: ``route_adherence_score`` at the corridor edge: ``1 / (1 + (d / (0.5 * d_max)) ** 2)`` = 0.2.
ADHERENCE_RANGE: Final[tuple[float, float]] = (0.2, 1.0)

#: Per-window statistic names of :class:`SignalStats` used here.
_EPS: Final[float] = 1e-9
_SIGNAL_FEATURES: Final[tuple[str, ...]] = (
    "last",
    "mean_short",
    "std_short",
    "range_short",
    "p95_short",
    "roc_short",
    "roc_medium",
    "accel",
    "level_shift",
    "vol_ratio",
    "z_window",
    "z_baseline",
)
_TEMPORAL_FEATURES: Final[tuple[str, ...]] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "minute_sin",
    "minute_cos",
)
_ENTITY_FEATURES: Final[tuple[str, ...]] = (
    "time_since_last_anomaly_log",
    "warmup_ratio",
    "log_count_short",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else math.nan


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _zscore(x: float, mean: float, std: float, count: int) -> float:
    """Return ``(x - mean) / std``, ``0.0`` when the window is too small or constant."""
    if count < MIN_BASELINE_COUNT:
        return 0.0
    scale = max(std, _EPS + 1e-6 * abs(mean))
    return _clip((x - mean) / scale, Z_CLIP)


def _ratio(num: float, den: float) -> float:
    """Return ``num / den`` capped at :data:`RATIO_CAP`; ``1.0`` when both are ~0."""
    if den < _EPS:
        return 1.0 if num < _EPS else RATIO_CAP
    return min(num / den, RATIO_CAP)


def _parse_date(value: Any, as_of: datetime) -> float:
    """Return days between an ISO date and ``as_of`` (never negative), or ``NaN``."""
    if not isinstance(value, str | date):
        return math.nan
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(value[:10])
    except ValueError:
        return math.nan
    return max((as_of.date() - parsed).days, 0.0)


def _epoch(moment: datetime) -> float:
    return moment.timestamp()


def acceptable_range(
    source_type: str, signal: str, context: Mapping[str, Any]
) -> tuple[float, float] | None:
    """Return the healthy ``(low, high)`` of a signal, or ``None`` if it has none.

    * warehouse ``value``: the sensor's ``acceptable_range``
    * vehicle ``speed_kmh``: ``0.6 - 1.2 x cruise_speed_kmh``; ``route_adherence_score``:
      ``0.2 - 1.0`` (0.2 is the corridor edge)
    * inventory ``new_level``: ``min_stock - max_stock``
    * inventory ``change_amount``: none
    """
    try:
        if source_type == "warehouse" and signal == "value":
            low, high = context["acceptable_range"]
        elif source_type == "vehicle" and signal == "speed_kmh":
            cruise = float(context["cruise_speed_kmh"])
            low, high = VEHICLE_SPEED_BAND[0] * cruise, VEHICLE_SPEED_BAND[1] * cruise
        elif source_type == "vehicle" and signal == "route_adherence_score":
            low, high = ADHERENCE_RANGE
        elif source_type == "inventory" and signal == "new_level":
            low, high = context["min_stock"], context["max_stock"]
        else:
            return None
        low, high = float(low), float(high)
    except (KeyError, TypeError, ValueError):
        return None
    return (low, high) if math.isfinite(low) and math.isfinite(high) and high > low else None


_RANGED_SIGNALS: Final[dict[str, tuple[str, ...]]] = {
    "warehouse": ("value",),
    "vehicle": ("speed_kmh", "route_adherence_score"),
    "inventory": ("new_level",),
}


def feature_names(source_type: str, zones: Sequence[str] = DEFAULT_ZONES) -> list[str]:
    """Return the ordered feature names of a source type (the columns of every vector).

    Raises:
        KeyError: ``source_type`` is not ``warehouse``, ``vehicle`` or ``inventory``.
    """
    signals = SIGNALS_BY_SOURCE[source_type]
    ranged = _RANGED_SIGNALS[source_type]
    names: list[str] = []
    for signal in signals:
        names += [f"{signal}__{f}" for f in _SIGNAL_FEATURES]
        if signal in ranged:
            names.append(f"{signal}__dev_signed")
    names += list(_ENTITY_FEATURES) + list(_TEMPORAL_FEATURES)
    for signal in ranged:
        names += [f"{signal}__margin_low", f"{signal}__margin_high"]
    if source_type == "warehouse":
        names.append("sensor_age_days")
    if source_type == "inventory":
        names.append("stock_vs_reorder")
    if source_type in ("warehouse", "inventory"):
        names += [f"zone_{z}" for z in zones]
    names += ["anomaly_rate_7d", "reliability"]
    return names


# --------------------------------------------------------------------------- #
# Aligned windows of one entity at one moment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class WindowBundle:
    """The :class:`WindowedFeatures` of every window length for one entity and one event.

    Attributes:
        windows: Window length in seconds to features, all with the same ``window_end``.
    """

    entity_id: str
    source_type: str
    window_end: datetime
    windows: Mapping[int, WindowedFeatures]

    @property
    def short(self) -> WindowedFeatures:
        """Shortest window."""
        return self.windows[min(self.windows)]

    @property
    def long(self) -> WindowedFeatures:
        """Longest window (the rolling baseline)."""
        return self.windows[max(self.windows)]

    @property
    def medium(self) -> WindowedFeatures:
        """Middle window; the longest one if fewer than three are configured."""
        keys = sorted(self.windows)
        return self.windows[keys[len(keys) // 2]] if len(keys) >= 3 else self.long

    @property
    def context(self) -> Mapping[str, Any]:
        """Descriptive fields of the newest event."""
        return self.short.context


def bundles_from_features(
    features: Iterable[WindowedFeatures], window_seconds: Sequence[int] = DEFAULT_WINDOWS
) -> list[WindowBundle]:
    """Group features by entity and ``window_end`` into complete bundles, oldest first.

    Incomplete groups (a window length is missing) are dropped. Handy for offline training:
    run the aggregator over a simulation, collect its output, call this.
    """
    needed = set(window_seconds)
    groups: dict[tuple[str, datetime], dict[int, WindowedFeatures]] = defaultdict(dict)
    for item in features:
        if item.window_seconds in needed:
            groups[(item.entity_id, item.window_end)][item.window_seconds] = item
    bundles = [
        WindowBundle(eid, next(iter(w.values())).source_type, end, dict(sorted(w.items())))
        for (eid, end), w in groups.items()
        if set(w) == needed
    ]
    bundles.sort(key=lambda b: (b.window_end, b.entity_id))
    return bundles


# --------------------------------------------------------------------------- #
# What detectors and analysts told us about each entity
# --------------------------------------------------------------------------- #
class EntityHistory:
    """Per-entity record of flagged anomalies, event volume and analyst feedback. Thread-safe.

    Feeds three features: time since last anomaly, the rolling 7-day anomaly rate and the
    reliability score. Only *detector output* and *feedback* belong here, never ground-truth
    labels. Event volume is kept in hourly buckets, so memory is bounded (168 per entity).
    """

    _BUCKET_S: Final = 3_600.0
    _MAX_ANOMALY_TIMES: Final = 10_000

    def __init__(self, *, rate_days: int = 7, prior_pseudo_count: float = 2.0) -> None:
        """Create the history.

        Args:
            rate_days: Length of the rolling anomaly-rate window.
            prior_pseudo_count: Strength of the optimistic prior of the reliability score
                (an entity without feedback scores ``1.0``).

        Raises:
            ValueError: An argument is not positive.
        """
        if rate_days < 1 or prior_pseudo_count <= 0:
            raise ValueError("rate_days and prior_pseudo_count must be positive")
        self._rate_s = rate_days * 86_400.0
        self._prior = prior_pseudo_count
        self._anomaly_times: dict[str, list[float]] = defaultdict(list)
        self._buckets: dict[str, dict[int, list[int]]] = defaultdict(dict)
        self._last_seen: dict[str, float] = {}
        self._feedback: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [n, false_pos]
        self._lock = threading.RLock()

    def observe(self, entity_id: str, timestamp: float) -> bool:
        """Count one processed event. Idempotent: an event not newer than the last is ignored.

        Returns:
            Whether the event was counted.
        """
        with self._lock:
            if timestamp <= self._last_seen.get(entity_id, -math.inf):
                return False
            self._last_seen[entity_id] = timestamp
            buckets = self._buckets[entity_id]
            buckets.setdefault(int(timestamp // self._BUCKET_S), [0, 0])[0] += 1
            oldest = int((timestamp - self._rate_s) // self._BUCKET_S)
            for stale in [b for b in buckets if b < oldest]:
                del buckets[stale]
            return True

    def record_anomaly(self, entity_id: str, timestamp: float) -> None:
        """Record that a detector flagged ``entity_id`` at ``timestamp`` (epoch seconds)."""
        with self._lock:
            times = self._anomaly_times[entity_id]
            insort(times, timestamp)
            if len(times) > self._MAX_ANOMALY_TIMES:
                del times[: len(times) - self._MAX_ANOMALY_TIMES]
            self._buckets[entity_id].setdefault(int(timestamp // self._BUCKET_S), [0, 0])[1] += 1

    def record_feedback(self, entity_id: str, *, false_positive: bool) -> None:
        """Record one analyst verdict on an alert of this entity."""
        with self._lock:
            counts = self._feedback[entity_id]
            counts[0] += 1
            counts[1] += int(false_positive)

    def time_since_last_anomaly(self, entity_id: str, as_of: float) -> float | None:
        """Seconds from the last anomaly at or before ``as_of`` to ``as_of``; ``None`` if none."""
        with self._lock:
            times = self._anomaly_times.get(entity_id)
            if not times:
                return None
            idx = bisect_right(times, as_of)
            return as_of - times[idx - 1] if idx else None

    def anomaly_rate(self, entity_id: str, as_of: float) -> float:
        """Return the flagged share of events in the window ending at ``as_of`` (``0.0`` if none).

        Hour granularity: the bucket containing ``as_of`` is counted whole.
        """
        with self._lock:
            buckets = self._buckets.get(entity_id)
            if not buckets:
                return 0.0
            first = int((as_of - self._rate_s) // self._BUCKET_S)
            last = int(as_of // self._BUCKET_S)
            events = sum(c[0] for b, c in buckets.items() if first <= b <= last)
            flagged = sum(c[1] for b, c in buckets.items() if first <= b <= last)
            return min(flagged / events, 1.0) if events else 0.0

    def reliability(self, entity_id: str) -> float:
        """``1 - false_positive_rate`` of the entity's feedback, smoothed towards ``1.0``."""
        with self._lock:
            n, fp = self._feedback.get(entity_id, (0, 0))
        return 1.0 - fp / (n + self._prior)


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
class TimeSeriesFeatures:
    """Signal dynamics and calendar encodings computed from a :class:`WindowBundle`.

    Per signal (``short`` / ``medium`` / ``long`` are the shortest, middle and longest window):

    ===============  =================================================================
    ``last``         newest value
    ``mean_short``   mean over the short window
    ``std_short``    std over the short window
    ``range_short``  max - min over the short window
    ``p95_short``    95th percentile over the short window
    ``roc_short``    first derivative: least-squares slope of the short window
    ``roc_medium``   slope of the medium window
    ``accel``        second derivative: slope change per second between window centres
    ``level_shift``  short mean vs long mean, in long-window stds
    ``vol_ratio``    short std / long std
    ``z_window``     ``last`` vs the medium window (local z-score)
    ``z_baseline``   ``last`` vs the long window: the entity's rolling historical baseline
    ``dev_signed``   distance outside the acceptable range / range width (0 inside; positive
                     above, negative below); only for signals that have a range
    ===============  =================================================================

    Per entity: ``time_since_last_anomaly_log`` (``log1p`` of seconds, capped at 24 h),
    ``warmup_ratio`` (how much of the longest window is filled) and ``log_count_short``
    (``log1p`` of the event count, a message-rate signal that drops on sensor dropout).
    Calendar: sin / cos of hour of day, day of week and minute of hour.
    """

    def __init__(self, history: EntityHistory | None = None) -> None:
        """Create the feature group.

        Args:
            history: Source of "time since last anomaly"; ``None`` means never anomalous.
        """
        self.history = history

    def compute(self, bundle: WindowBundle) -> dict[str, float]:
        """Return the time-series features of one bundle, in schema order."""
        out: dict[str, float] = {}
        short, medium, long_ = bundle.short, bundle.medium, bundle.long
        ranged = _RANGED_SIGNALS[bundle.source_type]
        for signal in SIGNALS_BY_SOURCE[bundle.source_type]:
            s, m, lg = short.signals[signal], medium.signals[signal], long_.signals[signal]
            dt = (medium.window_seconds - short.window_seconds) / 2.0
            accel = (s.rate_of_change - m.rate_of_change) / dt if dt > 0 else 0.0
            p = f"{signal}__"
            out[p + "last"] = s.last
            out[p + "mean_short"] = s.mean
            out[p + "std_short"] = s.std
            out[p + "range_short"] = s.max - s.min
            out[p + "p95_short"] = s.percentile_95
            out[p + "roc_short"] = s.rate_of_change
            out[p + "roc_medium"] = m.rate_of_change
            out[p + "accel"] = accel
            out[p + "level_shift"] = (
                _zscore(s.mean, lg.mean, lg.std, long_.count) if long_ is not short else 0.0
            )
            out[p + "vol_ratio"] = (
                _ratio(s.std, lg.std) if long_.count >= MIN_BASELINE_COUNT else 1.0
            )
            out[p + "z_window"] = _zscore(s.last, m.mean, m.std, medium.count)
            out[p + "z_baseline"] = _zscore(s.last, lg.mean, lg.std, long_.count)
            if signal in ranged:
                rng = acceptable_range(bundle.source_type, signal, bundle.context)
                out[p + "dev_signed"] = _signed_deviation(s.last, rng)

        ts = _epoch(bundle.window_end)
        since = self.history.time_since_last_anomaly(bundle.entity_id, ts) if self.history else None
        capped = NEVER_ANOMALOUS_S if since is None else min(max(since, 0.0), NEVER_ANOMALOUS_S)
        out["time_since_last_anomaly_log"] = math.log1p(capped)
        out["warmup_ratio"] = min(long_.coverage_seconds / long_.window_seconds, 1.0)
        out["log_count_short"] = math.log1p(short.count)
        out.update(_cyclical(bundle.window_end))
        return out


def _signed_deviation(value: float, rng: tuple[float, float] | None) -> float:
    if rng is None:
        return math.nan
    low, high = rng
    width = high - low
    if value > high:
        return (value - high) / width
    if value < low:
        return (value - low) / width
    return 0.0


def _cyclical(moment: datetime) -> dict[str, float]:
    hour = moment.hour + moment.minute / 60.0
    minute = moment.minute + moment.second / 60.0
    angles = {
        "hour": 2 * math.pi * hour / 24.0,
        "dow": 2 * math.pi * moment.weekday() / 7.0,
        "minute": 2 * math.pi * minute / 60.0,
    }
    out: dict[str, float] = {}
    for name, angle in angles.items():
        out[f"{name}_sin"] = math.sin(angle)
        out[f"{name}_cos"] = math.cos(angle)
    return out


class TabularContextFeatures:
    """Static and slowly-changing context of an entity.

    * ``{signal}__margin_low`` / ``margin_high``: distance of the newest value from each bound
      of its acceptable range, in range widths. Positive inside, negative once outside, so
      small values mean "close to the threshold".
    * ``sensor_age_days`` (warehouse): days since ``last_calibration_date``; ``NaN`` if unknown.
    * ``stock_vs_reorder`` (inventory): ``(new_level - reorder_point) / (max - min)``.
    * ``zone_<Z>``: one-hot warehouse zone; all zeros for an unknown zone.
    * ``anomaly_rate_7d``: share of this entity's events flagged in the last 7 days.
    * ``reliability``: ``1 - false positive rate`` from analyst feedback (``1.0`` without any).
    """

    def __init__(
        self, history: EntityHistory | None = None, zones: Sequence[str] = DEFAULT_ZONES
    ) -> None:
        """Create the feature group.

        Args:
            history: Source of anomaly rate and reliability; ``None`` gives ``0.0`` / ``1.0``.
            zones: Zone vocabulary of the one-hot encoding.
        """
        self.history = history
        self.zones = tuple(zones)

    def compute(self, bundle: WindowBundle) -> dict[str, float]:
        """Return the tabular features of one bundle, in schema order."""
        out: dict[str, float] = {}
        ctx, source = bundle.context, bundle.source_type
        short = bundle.short
        for signal in _RANGED_SIGNALS[source]:
            last = short.signals[signal].last
            rng = acceptable_range(source, signal, ctx)
            if rng is None:
                out[f"{signal}__margin_low"] = out[f"{signal}__margin_high"] = math.nan
            else:
                width = rng[1] - rng[0]
                out[f"{signal}__margin_low"] = _clip((last - rng[0]) / width, Z_CLIP)
                out[f"{signal}__margin_high"] = _clip((rng[1] - last) / width, Z_CLIP)
        if source == "warehouse":
            out["sensor_age_days"] = _parse_date(
                ctx.get("last_calibration_date"), bundle.window_end
            )
        if source == "inventory":
            out["stock_vs_reorder"] = _stock_vs_reorder(short.signals["new_level"].last, ctx)
        if source in ("warehouse", "inventory"):
            zone = ctx.get("zone")
            for z in self.zones:
                out[f"zone_{z}"] = 1.0 if zone == z else 0.0
        ts = _epoch(bundle.window_end)
        out["anomaly_rate_7d"] = (
            self.history.anomaly_rate(bundle.entity_id, ts) if self.history else 0.0
        )
        out["reliability"] = self.history.reliability(bundle.entity_id) if self.history else 1.0
        return out


def _stock_vs_reorder(level: float, ctx: Mapping[str, Any]) -> float:
    try:
        reorder, low, high = (
            float(ctx["reorder_point"]),
            float(ctx["min_stock"]),
            float(ctx["max_stock"]),
        )
    except (KeyError, TypeError, ValueError):
        return math.nan
    return _clip((level - reorder) / (high - low), Z_CLIP) if high > low else math.nan


class FeatureEngineer:
    """Combines :class:`TimeSeriesFeatures` and :class:`TabularContextFeatures` into vectors.

    The vector layout is :func:`feature_names` (30 to 45 features depending on the source).
    """

    def __init__(
        self, history: EntityHistory | None = None, zones: Sequence[str] = DEFAULT_ZONES
    ) -> None:
        """Create the engineer.

        Args:
            history: Shared :class:`EntityHistory`; created if ``None``.
            zones: Zone vocabulary of the one-hot encoding.
        """
        self.history = history if history is not None else EntityHistory()
        self.zones = tuple(zones)
        self.time_series = TimeSeriesFeatures(self.history)
        self.tabular = TabularContextFeatures(self.history, self.zones)

    def names(self, source_type: str) -> list[str]:
        """Return the feature names of ``source_type``."""
        return feature_names(source_type, self.zones)

    def compute(self, bundle: WindowBundle) -> dict[str, float]:
        """Return all features of a bundle as a name-to-value dict (``NaN`` = unknown)."""
        features = {**self.time_series.compute(bundle), **self.tabular.compute(bundle)}
        return {k: _finite_or_nan(float(v)) for k, v in features.items()}

    def vectorize(self, bundle: WindowBundle) -> np.ndarray:
        """Return the raw (unscaled) feature vector, ``float64`` of shape ``(n_features,)``."""
        features = self.compute(bundle)
        return np.array([features[n] for n in self.names(bundle.source_type)], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Online standardisation
# --------------------------------------------------------------------------- #
class OnlineScaler:
    """Per-feature running mean and std (Welford's algorithm); NaN-aware and serialisable.

    ``fit_update`` accepts one observation ``(n_features,)`` or a batch ``(n, n_features)``
    (merged with Chan's parallel update). NaNs are skipped per feature. ``transform``
    standardises with the current statistics and leaves NaNs in place. Features with fewer
    than two observations or no variance are only centred, never divided by ~0. Output is
    clipped to ``±clip`` so a single extreme anomaly cannot blow up an LSTM input.
    """

    def __init__(
        self, n_features: int, *, clip: float | None = 10.0, min_std: float = 1e-8
    ) -> None:
        """Create an empty scaler.

        Args:
            n_features: Width of the vectors.
            clip: Magnitude limit of standardised values; ``None`` for no clipping.
            min_std: Standard deviations below this are treated as ``1.0``.

        Raises:
            ValueError: ``n_features`` is not positive.
        """
        if n_features < 1:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.clip = clip
        self.min_std = min_std
        self.count = np.zeros(n_features)
        self.mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._lock = threading.Lock()

    def fit_update(self, x: np.ndarray | Sequence[float]) -> None:
        """Update the statistics with one observation or a batch.

        Raises:
            ValueError: The last dimension is not ``n_features``.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.n_features or arr.ndim > 2:
            raise ValueError(f"expected (..., {self.n_features}) input, got {arr.shape}")
        batch = arr.reshape(-1, self.n_features)
        valid = np.isfinite(batch)
        n_b = valid.sum(axis=0).astype(np.float64)
        if not n_b.any():
            return
        safe = np.where(valid, batch, 0.0)
        mean_b = np.divide(safe.sum(axis=0), n_b, out=np.zeros(self.n_features), where=n_b > 0)
        m2_b = (np.where(valid, batch - mean_b, 0.0) ** 2).sum(axis=0)
        with self._lock:
            n_a = self.count
            total = n_a + n_b
            delta = mean_b - self.mean
            frac = np.divide(n_b, total, out=np.zeros(self.n_features), where=total > 0)
            self._m2 = self._m2 + m2_b + delta**2 * n_a * frac
            self.mean = self.mean + delta * frac
            self.count = total

    @property
    def std(self) -> np.ndarray:
        """Sample standard deviation per feature (``1.0`` where undefined or ~0)."""
        with self._lock:
            var = np.divide(
                self._m2, self.count - 1, out=np.zeros(self.n_features), where=self.count > 1
            )
        std = np.sqrt(np.maximum(var, 0.0))
        return np.where(std < self.min_std, 1.0, std)

    def transform(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        """Standardise ``x`` (shape ``(..., n_features)``); NaNs stay NaN.

        Raises:
            ValueError: The last dimension is not ``n_features``.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.n_features:
            raise ValueError(f"expected (..., {self.n_features}) input, got {arr.shape}")
        with self._lock:
            mean = self.mean.copy()
        out = (arr - mean) / self.std
        return out if self.clip is None else np.clip(out, -self.clip, self.clip)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        with self._lock:
            return {
                "n_features": self.n_features,
                "clip": self.clip,
                "min_std": self.min_std,
                "count": self.count.tolist(),
                "mean": self.mean.tolist(),
                "m2": self._m2.tolist(),
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OnlineScaler:
        """Rebuild from :meth:`to_dict` output."""
        scaler = cls(int(data["n_features"]), clip=data["clip"], min_std=float(data["min_std"]))
        scaler.count = np.asarray(data["count"], dtype=np.float64)
        scaler.mean = np.asarray(data["mean"], dtype=np.float64)
        scaler._m2 = np.asarray(data["m2"], dtype=np.float64)
        return scaler

    def load_state(self, other: OnlineScaler) -> None:
        """Copy the statistics of ``other`` into this scaler (same width required)."""
        if other.n_features != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {other.n_features}")
        with self._lock:
            self.count, self.mean, self._m2 = (
                other.count.copy(),
                other.mean.copy(),
                other._m2.copy(),
            )

    def save(self, path: str | Path) -> None:
        """Write the snapshot as JSON (atomically, via a temp file)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> OnlineScaler:
        """Read a snapshot written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# LSTM sequences
# --------------------------------------------------------------------------- #
class NoDataError(LookupError):
    """The feature store has nothing for an entity and its source type is unknown."""


def fill_missing(matrix: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down each column; NaNs with nothing before them become ``0.0``."""
    out = matrix.copy()
    for col in range(out.shape[1]):
        values = out[:, col]
        valid = np.isfinite(values)
        if not valid.any():
            out[:, col] = 0.0
            continue
        idx = np.where(valid, np.arange(len(values)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = values[idx]
        filled[: int(np.argmax(valid))] = 0.0
        out[:, col] = filled
    return out


class SequenceBuilder:
    """Builds ``(lookback, n_features)`` LSTM inputs from the feature store.

    For each of the last ``lookback`` events it fetches every window length from the store,
    aligns them on ``window_end``, computes the raw vector, standardises it and fills gaps:

    * a step whose windows are not all available is forward-filled from the step before;
    * NaN features are forward-filled per feature;
    * anything with nothing before it, including the padding in front of a cold-started
      entity with fewer than ``lookback`` events, is ``0.0`` (the standardised mean).

    The scaler is only read, never updated.
    """

    def __init__(
        self,
        store: FeatureStore,
        engineer: FeatureEngineer,
        scalers: Mapping[str, OnlineScaler],
        window_seconds: Sequence[int] = DEFAULT_WINDOWS,
    ) -> None:
        """Create the builder.

        Args:
            store: Where the Day 5 pipeline writes windowed features.
            engineer: Computes the raw vectors.
            scalers: One scaler per source type.
            window_seconds: Window lengths the store holds (must match the aggregator).
        """
        self.store = store
        self.engineer = engineer
        self.scalers = scalers
        self.window_seconds = tuple(sorted(set(window_seconds)))

    def build_sequence(
        self, entity_id: str, lookback: int = 60, source_type: str | None = None
    ) -> torch.Tensor:
        """Return a ``float32`` tensor of shape ``(lookback, n_features)``, oldest step first.

        Raises:
            ValueError: ``lookback`` is not positive.
            NoDataError: The store has no data for the entity and ``source_type`` is ``None``.
        """
        return self.build_sequence_with_mask(entity_id, lookback, source_type)[0]

    def build_sequence_with_mask(
        self, entity_id: str, lookback: int = 60, source_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like :meth:`build_sequence`, plus a bool mask of shape ``(lookback,)``.

        The mask is ``True`` for steps built from real data and ``False`` for padding and for
        steps that were forward-filled because their windows were incomplete.
        """
        if lookback < 1:
            raise ValueError("lookback must be positive")
        bundles = self._bundles(entity_id, lookback)
        real_bundles = [b for b in bundles if b is not None]
        source = real_bundles[-1].source_type if real_bundles else source_type
        if source is None:
            raise NoDataError(f"no features for entity {entity_id!r} and no source_type given")
        n_features = len(self.engineer.names(source))
        scaler = self.scalers[source]

        steps = self._step_matrix(bundles, n_features, scaler) if real_bundles else None
        matrix = np.zeros((lookback, n_features))
        mask = np.zeros(lookback, dtype=bool)
        if steps is not None:
            raw, real = steps
            filled = fill_missing(raw)
            n = len(filled)
            matrix[lookback - n :] = filled
            mask[lookback - n :] = real
        return torch.from_numpy(matrix.astype(np.float32)), torch.from_numpy(mask)

    def _bundles(self, entity_id: str, lookback: int) -> list[WindowBundle | None]:
        """Complete bundles of the last events; ``None`` marks a step with missing windows."""
        per_window = {
            w: self.store.get_history(entity_id, lookback, w) for w in self.window_seconds
        }
        shortest = per_window[self.window_seconds[0]]
        by_end = {w: {f.window_end: f for f in items} for w, items in per_window.items()}
        out: list[WindowBundle | None] = []
        for item in shortest:
            found = {w: by_end[w].get(item.window_end) for w in self.window_seconds}
            if all(v is not None for v in found.values()):
                windows = {w: v for w, v in found.items() if v is not None}
                out.append(WindowBundle(entity_id, item.source_type, item.window_end, windows))
            else:
                out.append(None)
        return out

    def _step_matrix(
        self, bundles: list[WindowBundle | None], n_features: int, scaler: OnlineScaler
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.full((len(bundles), n_features), np.nan)
        real = np.zeros(len(bundles), dtype=bool)
        for i, bundle in enumerate(bundles):
            if bundle is not None:
                raw[i] = self.engineer.vectorize(bundle)
                real[i] = True
        return scaler.transform(raw), real


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FeatureVector:
    """One processed event: raw and standardised vectors with their names."""

    entity_id: str
    source_type: str
    timestamp: datetime
    names: tuple[str, ...]
    raw: np.ndarray
    scaled: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, float]:
        """Return the raw features as a name-to-value dict."""
        return dict(zip(self.names, self.raw.tolist(), strict=True))


class FeaturePipeline:
    """Runs the whole chain: store -> bundles -> features -> scaling -> sequences.

    Online use (Day 15)::

        cache.store_features(...)                       # written by the Day 5 pipeline
        vec = pipeline.process("wh1-A-temp-01")         # latest vector, updates the scaler
        seq = pipeline.build_sequence("wh1-A-temp-01")  # (60, n_features) for the LSTMs
        pipeline.record_anomaly("wh1-A-temp-01", ts)    # when a detector flags it

    Offline use: :func:`bundles_from_features` then :meth:`vectorize_bundles`.
    """

    def __init__(
        self,
        store: FeatureStore,
        *,
        window_seconds: Sequence[int] = DEFAULT_WINDOWS,
        zones: Sequence[str] = DEFAULT_ZONES,
        history: EntityHistory | None = None,
        scalers: Mapping[str, OnlineScaler] | None = None,
        scaler_clip: float | None = 10.0,
    ) -> None:
        """Create the pipeline.

        Args:
            store: Feature store written by the stream pipeline.
            window_seconds: Window lengths the store holds.
            zones: Zone vocabulary of the one-hot encoding.
            history: Shared entity history; created if ``None``.
            scalers: Existing scalers per source type (for example loaded from disk); missing
                source types get a fresh one.
            scaler_clip: Output clip of newly created scalers.
        """
        self.store = store
        self.window_seconds = tuple(sorted(set(window_seconds)))
        self.history = history if history is not None else EntityHistory()
        self.engineer = FeatureEngineer(self.history, zones)
        self.scalers: dict[str, OnlineScaler] = {
            src: OnlineScaler(len(self.engineer.names(src)), clip=scaler_clip)
            for src in SIGNALS_BY_SOURCE
        }
        if scalers:
            self.scalers.update(scalers)
        self.sequences = SequenceBuilder(store, self.engineer, self.scalers, self.window_seconds)

    @classmethod
    def from_config(cls, cfg: AppConfig, store: FeatureStore, **kwargs: Any) -> FeaturePipeline:
        """Create the pipeline with window lengths and zones from the project config."""
        return cls(
            store,
            window_seconds=cfg.features.window_seconds,
            zones=cfg.simulation.warehouse.zones,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    def bundle(self, entity_id: str) -> WindowBundle | None:
        """Return the latest complete bundle of an entity, or ``None`` (unknown / cold)."""
        latest = {w: self.store.get_features(entity_id, w) for w in self.window_seconds}
        windows = {w: f for w, f in latest.items() if f is not None}
        if len(windows) != len(self.window_seconds):
            return None
        ends = {f.window_end for f in windows.values()}
        if len(ends) != 1:
            return None
        first = windows[self.window_seconds[0]]
        return WindowBundle(entity_id, first.source_type, first.window_end, dict(windows))

    def process(self, entity_id: str, *, update_scaler: bool = True) -> FeatureVector | None:
        """Compute the newest feature vector of an entity.

        Also counts the event for the 7-day anomaly rate (once per event, however often this
        is called) and, unless ``update_scaler`` is false, updates the source type's scaler.

        Returns:
            The vector, or ``None`` if the store has no complete bundle for the entity.
        """
        bundle = self.bundle(entity_id)
        if bundle is None:
            return None
        raw = self.engineer.vectorize(bundle)
        scaler = self.scalers[bundle.source_type]
        if update_scaler:
            scaler.fit_update(raw)
        self.history.observe(entity_id, _epoch(bundle.window_end))
        return FeatureVector(
            entity_id=entity_id,
            source_type=bundle.source_type,
            timestamp=bundle.window_end,
            names=tuple(self.engineer.names(bundle.source_type)),
            raw=raw,
            scaled=scaler.transform(raw),
        )

    def vectorize_bundles(
        self, bundles: Iterable[WindowBundle], *, update_scaler: bool = False
    ) -> dict[str, np.ndarray]:
        """Raw vectors of many bundles (offline), stacked per source type in input order.

        With ``update_scaler`` the scalers are fitted on the result, which is how training
        data prepares them.
        """
        rows: dict[str, list[np.ndarray]] = defaultdict(list)
        for bundle in bundles:
            rows[bundle.source_type].append(self.engineer.vectorize(bundle))
        stacked = {src: np.vstack(items) for src, items in rows.items()}
        if update_scaler:
            for src, matrix in stacked.items():
                self.scalers[src].fit_update(matrix)
        return stacked

    def build_sequence(
        self, entity_id: str, lookback: int = 60, source_type: str | None = None
    ) -> torch.Tensor:
        """Return the standardised ``(lookback, n_features)`` LSTM input of an entity."""
        return self.sequences.build_sequence(entity_id, lookback, source_type)

    # ------------------------------------------------------------------ #
    def record_anomaly(self, entity_id: str, timestamp: float | datetime) -> None:
        """Tell the pipeline a detector flagged ``entity_id`` (epoch seconds or datetime)."""
        ts = _epoch(timestamp) if isinstance(timestamp, datetime) else float(timestamp)
        self.history.record_anomaly(entity_id, ts)

    def record_feedback(self, entity_id: str, *, false_positive: bool) -> None:
        """Tell the pipeline an analyst judged an alert of ``entity_id``."""
        self.history.record_feedback(entity_id, false_positive=false_positive)

    def save_scalers(self, directory: str | Path) -> None:
        """Write one ``scaler_<source>.json`` per source type into ``directory``."""
        for src, scaler in self.scalers.items():
            scaler.save(Path(directory) / f"scaler_{src}.json")

    def load_scalers(self, directory: str | Path) -> int:
        """Load the scalers found in ``directory``; return how many were loaded."""
        loaded = 0
        for src in self.scalers:
            path = Path(directory) / f"scaler_{src}.json"
            if path.exists():
                self.scalers[src].load_state(OnlineScaler.load(path))
                loaded += 1
        return loaded


__all__ = [
    "DEFAULT_ZONES",
    "EntityHistory",
    "FeatureEngineer",
    "FeaturePipeline",
    "FeatureVector",
    "NoDataError",
    "OnlineScaler",
    "SequenceBuilder",
    "TabularContextFeatures",
    "TimeSeriesFeatures",
    "WindowBundle",
    "acceptable_range",
    "bundles_from_features",
    "feature_names",
    "fill_missing",
]
