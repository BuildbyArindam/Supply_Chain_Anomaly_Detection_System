"""Statistical anomaly detection baseline (Day 8).

Three classic streaming detectors plus a combiner, all operating on one scalar
observation at a time so they can sit directly behind the Kafka consumer /
FeatureCache without any batch refit:

* :class:`ZScoreDetector`  - adaptive mean/std (Welford / ``OnlineScaler``),
  optional per-hour contextual baselines and per-zone / per-hour thresholds.
  Targets point anomalies (and contextual ones via the hourly baseline).
* :class:`IQRDetector`     - rolling Tukey fences, with a separate window per
  hour-of-day bucket (seasonal adjustment). Robust to heavy tails.
* :class:`CUSUMDetector`   - two-sided tabular CUSUM for slow drift
  (collective anomalies such as equipment degradation).
* :class:`UnifiedStatisticalDetector` - runs the three independent detectors on
  every observation and merges them into one :class:`AnomalyResult`.

Score convention (shared by every detector)
-------------------------------------------
Each detector computes a *ratio* that equals ``1.0`` exactly at its decision
boundary (``|z| / threshold``, distance-to-fence / fence-distance,
``max(S+, S-) / h``). The reported ``score`` is ``clip(ratio / 2, 0, 1)``, so
``score == 0.5`` is the boundary, ``score >= 0.5`` means "anomalous" and scores
from different detectors are directly comparable. The unclipped ratio is kept in
``details["ratio"]`` for threshold sweeps.

Design notes
------------
* Detectors are stateful and keyed by ``entity_id``; they are not thread-safe.
  Use one instance per consumer worker (or guard with a lock).
* While an entity is flagged, its baseline statistics are *frozen* so anomalies
  do not inflate the std or drag the mean. After ``max_frozen_run`` consecutive
  flags the baseline resumes learning (accepts a genuine regime change).
* Until an entity has ``min_samples`` observations the detectors return
  ``score=0`` / ``is_anomaly=False`` with ``details["warming_up"] = True``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NamedTuple

import numpy as np

try:  # project logger (Day 1); fall back to stdlib so the module is importable alone
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - depends on project layout
    import logging

    logger = logging.getLogger(__name__)

__all__ = [
    "AnomalyScore",
    "AnomalyResult",
    "Observation",
    "RunningStats",
    "onlinescaler_factory",
    "parse_timestamp",
    "event_to_observations",
    "ZScoreDetector",
    "IQRDetector",
    "CUSUMDetector",
    "UnifiedStatisticalDetector",
]

Timestamp = str | datetime | int | float


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyScore:
    """Output of a single detector for one observation.

    ``anomaly_type`` is the detector's native type ('point', 'contextual' or
    'collective') and is populated even when ``is_anomaly`` is False; consumers
    must check ``is_anomaly`` first.
    """

    entity_id: str
    timestamp: datetime
    score: float
    is_anomaly: bool
    anomaly_type: str | None
    confidence: float
    detector: str = ""
    raw_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "entity_id": self.entity_id,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "confidence": self.confidence,
            "detector": self.detector,
            "raw_value": self.raw_value,
            "details": dict(self.details),
        }


@dataclass
class AnomalyResult:
    """Merged output of :class:`UnifiedStatisticalDetector`."""

    entity_id: str
    timestamp: datetime
    severity_score: float
    is_anomaly: bool
    anomaly_type: str | None
    confidence: float
    contributing_detectors: list[str]
    detector_scores: dict[str, AnomalyScore]
    raw_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "entity_id": self.entity_id,
            "timestamp": self.timestamp.isoformat(),
            "severity_score": self.severity_score,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "confidence": self.confidence,
            "contributing_detectors": list(self.contributing_detectors),
            "detector_scores": {k: v.to_dict() for k, v in self.detector_scores.items()},
            "raw_value": self.raw_value,
            "details": dict(self.details),
        }


class Observation(NamedTuple):
    """One scalar series sample extracted from a raw simulator event."""

    entity_id: str
    value: float
    timestamp: Timestamp
    zone: str | None
    metric: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_timestamp(ts: Timestamp) -> datetime:
    """Convert an ISO-8601 string, epoch seconds or ``datetime`` to ``datetime``.

    The hour-of-day used for seasonal logic is the hour *as written* in the
    timestamp (no timezone conversion), matching how the simulators define
    off-hours.
    """
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=UTC)
    if isinstance(ts, str):
        text = ts.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    raise TypeError(f"Unsupported timestamp type: {type(ts).__name__}")


def _score_from_ratio(ratio: float) -> float:
    return float(min(1.0, max(0.0, ratio / 2.0)))


def _confidence(ratio: float, n: int, min_samples: int) -> float:
    """Distance from the decision boundary, damped while the baseline is young."""
    maturity = min(1.0, n / float(max(1, 4 * min_samples)))
    return float(min(1.0, abs(ratio - 1.0)) * maturity)


class RunningStats:
    """Welford online mean/std for a single feature.

    Mirrors the ``fit_update(x)`` / ``transform(x)`` interface of the project's
    ``OnlineScaler`` so either can back :class:`ZScoreDetector`.
    """

    __slots__ = ("n", "mean", "_m2", "min_std")

    def __init__(self, min_std: float = 1e-6) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.min_std = min_std

    @property
    def std(self) -> float:
        """Sample standard deviation, floored at ``min_std``."""
        if self.n < 2:
            return self.min_std
        return max(math.sqrt(self._m2 / (self.n - 1)), self.min_std)

    def fit_update(self, x: float) -> None:
        """Fold one observation into the running statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    def transform(self, x: float) -> float:
        """Standardise ``x`` with the current statistics."""
        return (x - self.mean) / self.std


class _OnlineScalerAdapter:
    """Wraps the project's feature-engineering ``OnlineScaler`` for one feature."""

    def __init__(self, scaler: Any) -> None:
        self._scaler = scaler

    def fit_update(self, x: float) -> None:
        self._scaler.fit_update(np.array([x], dtype=float))

    def transform(self, x: float) -> float:
        out = self._scaler.transform(np.array([x], dtype=float))
        return float(np.asarray(out, dtype=float).ravel()[0])


def onlinescaler_factory(**scaler_kwargs: Any) -> Callable[[], Any]:
    """Return a ``scaler_factory`` backed by ``feature_engineering.OnlineScaler``.

    Example::

        ZScoreDetector(scaler_factory=onlinescaler_factory(n_features=1))

    The keyword arguments are forwarded to ``OnlineScaler``. Imported lazily so
    this module has no hard dependency on the Day 6 code.
    """
    from src.models.feature_engineering import OnlineScaler  # type: ignore

    def _factory() -> _OnlineScalerAdapter:
        return _OnlineScalerAdapter(OnlineScaler(**scaler_kwargs))

    return _factory


def _default_scaler_factory() -> RunningStats:
    return RunningStats()


def event_to_observations(event: Mapping[str, Any]) -> list[Observation]:
    """Turn a raw simulator event (Day 2/3 schema) into scalar observations.

    * warehouse sensor: ``sensor_id`` / ``value``
    * vehicle GPS:      ``{vehicle_id}:speed`` / ``speed_kmh`` and, when present,
      ``{vehicle_id}:adherence`` / ``route_adherence_score``
    * inventory:        ``{sku_id}@{warehouse_id}`` / ``change_amount``
    """
    ts = event["timestamp"]
    if "sensor_id" in event:
        meta = event.get("metadata") or {}
        zone = event.get("zone") or meta.get("zone") or meta.get("location")
        return [Observation(str(event["sensor_id"]), float(event["value"]), ts, zone, "value")]
    if "vehicle_id" in event:
        vid = str(event["vehicle_id"])
        obs = [Observation(f"{vid}:speed", float(event["speed_kmh"]), ts, None, "speed_kmh")]
        if event.get("route_adherence_score") is not None:
            obs.append(
                Observation(
                    f"{vid}:adherence",
                    float(event["route_adherence_score"]),
                    ts,
                    None,
                    "route_adherence",
                )
            )
        return obs
    if "sku_id" in event:
        eid = f"{event['sku_id']}@{event.get('warehouse_id', '')}"
        return [
            Observation(eid, float(event["change_amount"]), ts, event.get("zone"), "change_amount")
        ]
    raise ValueError("Event has none of sensor_id / vehicle_id / sku_id")


def _warming(entity_id: str, ts: datetime, x: float, name: str, n: int, atype: str) -> AnomalyScore:
    return AnomalyScore(
        entity_id=entity_id,
        timestamp=ts,
        score=0.0,
        is_anomaly=False,
        anomaly_type=atype,
        confidence=0.0,
        detector=name,
        raw_value=x,
        details={"warming_up": True, "ratio": 0.0, "n_samples": n},
    )


# --------------------------------------------------------------------------- #
# Z-score
# --------------------------------------------------------------------------- #
class _ZState:
    __slots__ = ("g", "g_n", "ctx", "frozen_run")

    def __init__(self, scaler: Any) -> None:
        self.g = scaler
        self.g_n = 0
        self.ctx: dict[int, list[Any]] = {}  # hour -> [scaler, n]
        self.frozen_run = 0


class ZScoreDetector:
    """Adaptive Z-score detector with optional contextual (hour-of-day) baseline.

    Parameters
    ----------
    threshold:
        Decision threshold in sigmas (default 3.0).
    min_samples / min_context_samples:
        Observations required before the entity-level / hour-level baseline is
        trusted.
    contextual:
        Keep a separate baseline per (entity, hour-of-day). An observation that
        is normal overall but extreme *for its hour* is labelled ``contextual``.
    zone_thresholds / hour_thresholds:
        Absolute sigma overrides. Precedence: hour override, then zone override,
        then ``threshold``; the entity offset from :meth:`adjust_threshold`
        (feedback hook) is added last.
    scaler_factory:
        Zero-arg callable returning an object with ``fit_update(x)`` and
        ``transform(x)``. Defaults to :class:`RunningStats`; use
        :func:`onlinescaler_factory` to back it with ``OnlineScaler``.
    freeze_on_anomaly / max_frozen_run:
        Skip baseline updates while flagged, up to ``max_frozen_run``
        consecutive flags.
    """

    name = "zscore"

    def __init__(
        self,
        threshold: float = 3.0,
        min_samples: int = 30,
        contextual: bool = True,
        min_context_samples: int = 30,
        zone_thresholds: Mapping[str, float] | None = None,
        hour_thresholds: Mapping[int, float] | None = None,
        scaler_factory: Callable[[], Any] | None = None,
        freeze_on_anomaly: bool = True,
        max_frozen_run: int = 300,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.contextual = contextual
        self.min_context_samples = int(min_context_samples)
        self.zone_thresholds: dict[str, float] = dict(zone_thresholds or {})
        self.hour_thresholds: dict[int, float] = dict(hour_thresholds or {})
        self._factory = scaler_factory or _default_scaler_factory
        self.freeze_on_anomaly = freeze_on_anomaly
        self.max_frozen_run = int(max_frozen_run)
        self._states: dict[str, _ZState] = {}
        self._offsets: dict[str, float] = {}

    # -- threshold management ------------------------------------------------
    def threshold_for(
        self, entity_id: str, zone: str | None = None, hour: int | None = None
    ) -> float:
        """Effective sigma threshold for an entity at a given zone / hour."""
        thr = self.threshold
        if zone is not None and zone in self.zone_thresholds:
            thr = self.zone_thresholds[zone]
        if hour is not None and hour in self.hour_thresholds:
            thr = self.hour_thresholds[hour]
        return max(0.5, thr + self._offsets.get(entity_id, 0.0))

    def adjust_threshold(self, entity_id: str, delta: float) -> None:
        """Shift one entity's threshold by ``delta`` sigmas (feedback-loop hook)."""
        self._offsets[entity_id] = self._offsets.get(entity_id, 0.0) + float(delta)

    def reset(self, entity_id: str | None = None) -> None:
        """Forget learned state for one entity, or for all of them."""
        if entity_id is None:
            self._states.clear()
            self._offsets.clear()
        else:
            self._states.pop(entity_id, None)
            self._offsets.pop(entity_id, None)

    # -- streaming API ---------------------------------------------------------
    def update(
        self, entity_id: str, value: float, timestamp: Timestamp, zone: str | None = None
    ) -> AnomalyScore:
        """Score one observation, then learn from it."""
        ts = parse_timestamp(timestamp)
        x = float(value)
        if not math.isfinite(x):
            return AnomalyScore(
                entity_id,
                ts,
                0.0,
                False,
                "point",
                0.0,
                self.name,
                None,
                {"skipped": "non_finite_value"},
            )
        st = self._states.get(entity_id)
        if st is None:
            st = self._states[entity_id] = _ZState(self._factory())
        hour = ts.hour
        thr = self.threshold_for(entity_id, zone, hour)

        z_g: float | None = None
        z_c: float | None = None
        if st.g_n >= self.min_samples:
            v = st.g.transform(x)
            z_g = v if math.isfinite(v) else None
        ctx = st.ctx.get(hour) if self.contextual else None
        if ctx is not None and ctx[1] >= self.min_context_samples:
            v = ctx[0].transform(x)
            z_c = v if math.isfinite(v) else None

        warming = z_g is None and z_c is None
        ratio_g = abs(z_g) / thr if z_g is not None else 0.0
        ratio_c = abs(z_c) / thr if z_c is not None else 0.0
        ratio = max(ratio_g, ratio_c)
        flagged = (not warming) and ratio >= 1.0
        atype = "contextual" if (flagged and ratio_g < 1.0 <= ratio_c) else "point"

        # learn
        if flagged and self.freeze_on_anomaly:
            st.frozen_run += 1
        else:
            st.frozen_run = 0
        if (not flagged) or (not self.freeze_on_anomaly) or st.frozen_run > self.max_frozen_run:
            st.g.fit_update(x)
            st.g_n += 1
            if self.contextual:
                if ctx is None:
                    ctx = st.ctx[hour] = [self._factory(), 0]
                ctx[0].fit_update(x)
                ctx[1] += 1

        if warming:
            return _warming(entity_id, ts, x, self.name, st.g_n, "point")
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=ts,
            score=_score_from_ratio(ratio),
            is_anomaly=flagged,
            anomaly_type=atype,
            confidence=_confidence(ratio, st.g_n, self.min_samples),
            detector=self.name,
            raw_value=x,
            details={
                "ratio": ratio,
                "z_global": z_g,
                "z_context": z_c,
                "threshold": thr,
                "hour": hour,
                "n_samples": st.g_n,
                "warming_up": False,
            },
        )

    detect = update


# --------------------------------------------------------------------------- #
# IQR
# --------------------------------------------------------------------------- #
class _Window:
    """Bounded window with cached quartiles (refreshed every ``refresh_every`` adds)."""

    __slots__ = ("buf", "since", "q1", "med", "q3")

    def __init__(self, maxlen: int) -> None:
        self.buf: deque[float] = deque(maxlen=maxlen)
        self.since = 0
        self.q1: float | None = None
        self.med = 0.0
        self.q3 = 0.0

    def add(self, x: float, refresh_every: int, min_n: int) -> None:
        self.buf.append(x)
        self.since += 1
        if len(self.buf) >= min_n and (self.q1 is None or self.since >= refresh_every):
            arr = np.fromiter(self.buf, dtype=float, count=len(self.buf))
            self.q1, self.med, self.q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
            self.since = 0

    @property
    def ready(self) -> bool:
        return self.q1 is not None

    def ratio(self, x: float, k: float) -> float:
        """Distance from the median in units of the median-to-fence distance."""
        assert self.q1 is not None
        iqr = max(self.q3 - self.q1, 1e-9)
        if x >= self.med:
            return (x - self.med) / max(self.q3 + k * iqr - self.med, 1e-9)
        return (self.med - x) / max(self.med - (self.q1 - k * iqr), 1e-9)


class _IQRState:
    __slots__ = ("glob", "bins", "frozen_run")

    def __init__(self, window: int, bin_window: int) -> None:
        self.glob = _Window(window)
        self.bins: dict[int, _Window] = {}
        self.frozen_run = 0


class IQRDetector:
    """Rolling-IQR (Tukey fence) detector with hour-of-day seasonal bins.

    An observation is compared with the entity's rolling global window (all
    hours) and with the window of its own hour bucket. Outside the global fences
    -> ``point``; inside the global fences but outside the seasonal fences ->
    ``contextual``.

    Parameters
    ----------
    window:
        Rolling window size per entity (and per hour bin), default 1000.
    multiplier:
        Fence multiplier ``k`` (``Q1 - k*IQR`` / ``Q3 + k*IQR``). 1.5 is Tukey's
        "outlier", 3.0 "far out"; 2.5 is the default here to keep false alarms
        manageable at 1 reading/s.
    bucket_hours:
        Width of the seasonal bins in hours (1 = one bin per hour).
    refresh_every:
        Recompute cached quartiles every N accepted samples (speed knob).
    """

    name = "iqr"

    def __init__(
        self,
        window: int = 1000,
        multiplier: float = 2.5,
        min_samples: int = 30,
        min_bin_samples: int = 30,
        bucket_hours: int = 1,
        refresh_every: int = 20,
        freeze_on_anomaly: bool = True,
        max_frozen_run: int = 300,
    ) -> None:
        if window < min_samples:
            raise ValueError("window must be >= min_samples")
        if 24 % bucket_hours != 0:
            raise ValueError("bucket_hours must divide 24")
        self.window = int(window)
        self.multiplier = float(multiplier)
        self.min_samples = int(min_samples)
        self.min_bin_samples = int(min_bin_samples)
        self.bucket_hours = int(bucket_hours)
        self.refresh_every = int(refresh_every)
        self.freeze_on_anomaly = freeze_on_anomaly
        self.max_frozen_run = int(max_frozen_run)
        self._states: dict[str, _IQRState] = {}

    def reset(self, entity_id: str | None = None) -> None:
        """Forget learned state for one entity, or for all of them."""
        if entity_id is None:
            self._states.clear()
        else:
            self._states.pop(entity_id, None)

    def update(
        self, entity_id: str, value: float, timestamp: Timestamp, zone: str | None = None
    ) -> AnomalyScore:
        """Score one observation, then learn from it. ``zone`` is unused."""
        ts = parse_timestamp(timestamp)
        x = float(value)
        if not math.isfinite(x):
            return AnomalyScore(
                entity_id,
                ts,
                0.0,
                False,
                "point",
                0.0,
                self.name,
                None,
                {"skipped": "non_finite_value"},
            )
        st = self._states.get(entity_id)
        if st is None:
            st = self._states[entity_id] = _IQRState(self.window, self.window)
        bucket = ts.hour // self.bucket_hours
        binw = st.bins.get(bucket)

        ratio_g = st.glob.ratio(x, self.multiplier) if st.glob.ready else None
        ratio_s = binw.ratio(x, self.multiplier) if (binw is not None and binw.ready) else None
        warming = ratio_g is None and ratio_s is None
        rg = ratio_g or 0.0
        rs = ratio_s or 0.0
        ratio = max(rg, rs)
        flagged = (not warming) and ratio >= 1.0
        atype = "contextual" if (rs >= 1.0 > rg) else "point"

        if flagged and self.freeze_on_anomaly:
            st.frozen_run += 1
        else:
            st.frozen_run = 0
        if (not flagged) or (not self.freeze_on_anomaly) or st.frozen_run > self.max_frozen_run:
            st.glob.add(x, self.refresh_every, self.min_samples)
            if binw is None:
                binw = st.bins[bucket] = _Window(self.window)
            binw.add(x, self.refresh_every, self.min_bin_samples)

        n = len(st.glob.buf)
        if warming:
            return _warming(entity_id, ts, x, self.name, n, "point")
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=ts,
            score=_score_from_ratio(ratio),
            is_anomaly=flagged,
            anomaly_type=atype,
            confidence=_confidence(ratio, n, self.min_samples),
            detector=self.name,
            raw_value=x,
            details={
                "ratio": ratio,
                "ratio_global": ratio_g,
                "ratio_seasonal": ratio_s,
                "bucket": bucket,
                "n_samples": n,
                "warming_up": False,
            },
        )

    detect = update


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #
class _CUSUMState:
    __slots__ = ("stats", "buckets", "s_pos", "s_neg", "hold_left")

    def __init__(self, min_std: float) -> None:
        self.stats = RunningStats(min_std)
        self.buckets: dict[int, RunningStats] = {}
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.hold_left = 0


class CUSUMDetector:
    """Two-sided tabular CUSUM on standardised observations.

    ``S+ = max(0, S+ + z - k)`` and ``S- = max(0, S- - z - k)`` with ``z``
    standardised by an in-control baseline. An alarm fires when either statistic
    exceeds ``h``.

    After an alarm the statistics are reset and the detector stays in alarm
    (``is_anomaly=True``) for ``cooldown_readings`` further readings, so one
    drift episode is reported as one contiguous incident. ``details["new_alarm"]``
    is True only for the first alarm of an incident, which is what an alert
    deduplicator should key on. A drift that continues re-alarms and extends the
    hold.

    The baseline mean/std only learns while the statistics are low
    (``max(S)/h < baseline_update_ratio``) and not in alarm, so slow drift is not
    absorbed into the baseline.

    Seasonality
    -----------
    With ``seasonal=True`` (default) ``z`` is computed against an hour-of-day
    baseline: the mean is linearly interpolated between neighbouring hour-bucket
    centres and the std comes from the current bucket. Without this, any signal
    with a daily cycle looks like a permanent drift to CUSUM (in the Day 8
    evaluation this alone produced ~30 false alarm incidents per entity-day).
    A bucket needs ``min_bucket_samples`` readings before it is trusted; until
    then the detector reports ``warming_up``. Use ``seasonal=False`` for signals
    with no daily pattern or when history is shorter than a few hours.

    Parameters
    ----------
    k:
        Slack in sigma units (half the shift you want to detect; 0.5 targets a
        1-sigma shift).
    h:
        Decision threshold in sigma units. 8.0 gave ~0.1% flagged readings on
        stationary Gaussian noise (including the cooldown hold); h=5 flagged
        ~4% because each false alarm is held for ``cooldown_readings`` readings.
    """

    name = "cusum"

    def __init__(
        self,
        k: float = 0.5,
        h: float = 8.0,
        cooldown_readings: int = 10,
        min_samples: int = 50,
        baseline_update_ratio: float = 0.5,
        min_std: float = 1e-6,
        seasonal: bool = True,
        bucket_hours: int = 1,
        min_bucket_samples: int = 30,
    ) -> None:
        if k < 0 or h <= 0:
            raise ValueError("require k >= 0 and h > 0")
        if 24 % bucket_hours != 0:
            raise ValueError("bucket_hours must divide 24")
        self.k = float(k)
        self.h = float(h)
        self.cooldown_readings = int(cooldown_readings)
        self.min_samples = int(min_samples)
        self.baseline_update_ratio = float(baseline_update_ratio)
        self.min_std = float(min_std)
        self.seasonal = bool(seasonal)
        self.bucket_hours = int(bucket_hours)
        self.min_bucket_samples = int(min_bucket_samples)
        self._states: dict[str, _CUSUMState] = {}

    def reset(self, entity_id: str | None = None) -> None:
        """Forget learned state for one entity, or for all of them."""
        if entity_id is None:
            self._states.clear()
        else:
            self._states.pop(entity_id, None)

    def _seasonal_baseline(
        self, st: _CUSUMState, ts: datetime
    ) -> tuple[float, float, RunningStats] | None:
        """(mean, std, current bucket) for ``ts``, or None if the bucket is not trusted yet."""
        bh = self.bucket_hours
        nb = 24 // bh
        hf = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
        cur = st.buckets.get(int(ts.hour // bh))
        if cur is None or cur.n < self.min_bucket_samples:
            return None
        pos = hf / bh - 0.5
        i0 = math.floor(pos)
        frac = pos - i0
        b0, b1 = st.buckets.get(i0 % nb), st.buckets.get((i0 + 1) % nb)
        if (
            b0 is not None
            and b1 is not None
            and b0.n >= self.min_bucket_samples
            and b1.n >= self.min_bucket_samples
        ):
            mean = (1.0 - frac) * b0.mean + frac * b1.mean
        else:
            mean = cur.mean
        return mean, cur.std, cur

    def update(
        self, entity_id: str, value: float, timestamp: Timestamp, zone: str | None = None
    ) -> AnomalyScore:
        """Score one observation, then learn from it. ``zone`` is unused."""
        ts = parse_timestamp(timestamp)
        x = float(value)
        if not math.isfinite(x):
            return AnomalyScore(
                entity_id,
                ts,
                0.0,
                False,
                "collective",
                0.0,
                self.name,
                None,
                {"skipped": "non_finite_value"},
            )
        st = self._states.get(entity_id)
        if st is None:
            st = self._states[entity_id] = _CUSUMState(self.min_std)
        stats = st.stats
        bucket: RunningStats | None = None
        if self.seasonal:
            bidx = int(ts.hour // self.bucket_hours)
            bucket = st.buckets.get(bidx)
            if bucket is None:
                bucket = st.buckets[bidx] = RunningStats(self.min_std)
            base = self._seasonal_baseline(st, ts)
            if base is None:
                bucket.fit_update(x)
                stats.fit_update(x)
                return _warming(entity_id, ts, x, self.name, bucket.n, "collective")
            mean_b, std_b, _ = base
            z = (x - mean_b) / std_b
        else:
            if stats.n < self.min_samples:
                stats.fit_update(x)
                return _warming(entity_id, ts, x, self.name, stats.n, "collective")
            z = stats.transform(x)
        st.s_pos = max(0.0, st.s_pos + z - self.k)
        st.s_neg = max(0.0, st.s_neg - z - self.k)
        ratio = max(st.s_pos, st.s_neg) / self.h
        direction = "up" if st.s_pos >= st.s_neg else "down"
        s_pos, s_neg = st.s_pos, st.s_neg

        new_alarm = False
        held = False
        if ratio >= 1.0:
            new_alarm = st.hold_left <= 0
            st.s_pos = st.s_neg = 0.0
            st.hold_left = self.cooldown_readings
            flagged = True
            if new_alarm:
                logger.debug(
                    f"CUSUM alarm entity={entity_id} direction={direction} ratio={ratio:.2f}"
                )
        elif st.hold_left > 0:
            st.hold_left -= 1
            flagged = True
            held = True
            ratio = max(ratio, 1.0)
        else:
            flagged = False
            if ratio < self.baseline_update_ratio:
                stats.fit_update(x)
                if bucket is not None:
                    bucket.fit_update(x)

        return AnomalyScore(
            entity_id=entity_id,
            timestamp=ts,
            score=_score_from_ratio(ratio),
            is_anomaly=flagged,
            anomaly_type="collective",
            confidence=_confidence(ratio, stats.n, self.min_samples),
            detector=self.name,
            raw_value=x,
            details={
                "ratio": ratio,
                "z": z,
                "s_pos": s_pos,
                "s_neg": s_neg,
                "direction": direction,
                "new_alarm": new_alarm,
                "held": held,
                "n_samples": stats.n,
                "warming_up": False,
            },
        )

    detect = update


# --------------------------------------------------------------------------- #
# Unified detector
# --------------------------------------------------------------------------- #
class UnifiedStatisticalDetector:
    """Combine Z-score, IQR and CUSUM into one result per observation.

    The three detectors are independent (each keeps its own state) and are all
    evaluated for every observation.

    Merging rules
    -------------
    * ``weighted = sum(w_i * score_i)`` with weights normalised to sum to 1.
    * ``severity_score = max_blend * max_i(score_i) + (1 - max_blend) * weighted``.
      Blending in the max stops a single confident detector (e.g. CUSUM on a
      drift) from being diluted by the other two.
    * ``is_anomaly`` is a weighted vote: ``sum(w_i for flagged detectors) >=
      vote_threshold``. The default 0.25 is below every default weight, i.e. an
      OR rule; raise it towards 0.5 for majority voting or ~1.0 for AND.
    * When ``is_anomaly`` is True ``severity_score`` is floored at 0.5, so
      ``severity_score >= 0.5`` iff ``is_anomaly``.
    * ``anomaly_type`` comes from the flagged detector with the highest
      weighted score.
    """

    name = "statistical_unified"

    def __init__(
        self,
        z_weight: float = 0.4,
        iqr_weight: float = 0.3,
        cusum_weight: float = 0.3,
        vote_threshold: float = 0.25,
        max_blend: float = 0.5,
        zscore: ZScoreDetector | None = None,
        iqr: IQRDetector | None = None,
        cusum: CUSUMDetector | None = None,
    ) -> None:
        total = z_weight + iqr_weight + cusum_weight
        if min(z_weight, iqr_weight, cusum_weight) < 0 or total <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        if not 0.0 <= max_blend <= 1.0:
            raise ValueError("max_blend must be within [0, 1]")
        self.weights = {
            "zscore": z_weight / total,
            "iqr": iqr_weight / total,
            "cusum": cusum_weight / total,
        }
        self.vote_threshold = float(vote_threshold)
        self.max_blend = float(max_blend)
        self.zscore = zscore or ZScoreDetector()
        self.iqr = iqr or IQRDetector()
        self.cusum = cusum or CUSUMDetector()

    @classmethod
    def production(cls) -> UnifiedStatisticalDetector:
        """Return the settings recommended by ``notebooks/02_statistical_baseline_eval.ipynb``.

        Tuned on one simulated day and validated on a second; chosen to maximise
        F1 subject to <= 3 false-alarm incidents per entity per day. Re-tune on
        real data before relying on the exact numbers.
        """
        return cls(
            zscore=ZScoreDetector(threshold=3.5),
            iqr=IQRDetector(multiplier=2.5),
            cusum=CUSUMDetector(k=0.75, h=10.0),
            vote_threshold=0.25,
        )

    def reset(self, entity_id: str | None = None) -> None:
        """Forget learned state in all three detectors."""
        self.zscore.reset(entity_id)
        self.iqr.reset(entity_id)
        self.cusum.reset(entity_id)

    def detect(
        self, entity_id: str, value: float, timestamp: Timestamp, zone: str | None = None
    ) -> AnomalyResult:
        """Run all three detectors on one observation and merge the outcome."""
        scores: dict[str, AnomalyScore] = {
            "zscore": self.zscore.update(entity_id, value, timestamp, zone),
            "iqr": self.iqr.update(entity_id, value, timestamp, zone),
            "cusum": self.cusum.update(entity_id, value, timestamp, zone),
        }
        ts = scores["zscore"].timestamp
        w = self.weights
        weighted = sum(w[n] * s.score for n, s in scores.items())
        max_score = max(s.score for s in scores.values())
        severity = self.max_blend * max_score + (1.0 - self.max_blend) * weighted

        flagged = [n for n, s in scores.items() if s.is_anomaly]
        vote = sum(w[n] for n in flagged)
        is_anomaly = bool(flagged) and vote >= self.vote_threshold - 1e-12
        if is_anomaly:
            severity = max(severity, 0.5)
            lead = max(flagged, key=lambda n: w[n] * scores[n].score)
            anomaly_type: str | None = scores[lead].anomaly_type
        else:
            severity = min(severity, 0.4999)
            anomaly_type = None

        conf_pool = (
            [scores[n].confidence for n in flagged]
            if flagged
            else [s.confidence for s in scores.values()]
        )
        confidence = float(np.mean(conf_pool)) if conf_pool else 0.0
        return AnomalyResult(
            entity_id=entity_id,
            timestamp=ts,
            severity_score=float(min(1.0, severity)),
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=confidence,
            contributing_detectors=flagged if is_anomaly else [],
            detector_scores=scores,
            raw_value=scores["zscore"].raw_value,
            details={
                "weighted_score": weighted,
                "vote": vote,
                "warming_up": all(s.details.get("warming_up", False) for s in scores.values()),
            },
        )

    def detect_event(self, event: Mapping[str, Any]) -> AnomalyResult:
        """Detect on a raw simulator event; returns the most severe metric's result.

        Vehicle events produce two observations (speed and route adherence); the
        result with the highest severity is returned and its metric name is
        stored in ``details["metric"]``.
        """
        results = []
        for obs in event_to_observations(event):
            res = self.detect(obs.entity_id, obs.value, obs.timestamp, obs.zone)
            res.details["metric"] = obs.metric
            results.append(res)
        best = max(results, key=lambda r: r.severity_score)
        best.details["n_metrics"] = len(results)
        return best
