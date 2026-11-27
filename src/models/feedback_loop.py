"""Analyst feedback loop and false-positive reduction (Day 16).

Analysts mark alerts as false / true positives (or correct their severity). This module turns
those verdicts into three kinds of learning, from cheapest and fastest to heaviest:

1. :class:`ThresholdCalibrator` -- per entity, immediately. An entity that produces a burst of
   false positives gets a higher decision threshold; one with missed anomalies gets a lower one.
2. :class:`EnsembleRecalibrator` -- every 30 minutes or 50 new verdicts, in the background. The
   LightGBM meta-learner keeps its trees and boosts a few more on the labelled verdicts (no full
   retrain), and the LSTM classifier's temperature is re-fitted. A challenger that does not beat
   the current model on held-out verdicts is rejected.
3. :class:`FeedbackDriftDetector` -- a Page-Hinkley test on the false-positive stream. A sustained
   rise means the models no longer fit the data: a :class:`ModelDriftAlert` goes to the
   ``model-health`` topic and a full retrain is due (Day 17).

:class:`FeedbackProcessor` is the front door: validate, store (atomic JSON), then feed all three.

Threshold convention
--------------------
Ensemble scores are on the 0.5-boundary scale (:data:`~src.models.ensemble_detector.BOUNDARY`),
so a per-entity threshold is ``0.5 + adjustment``; an event is anomalous iff its ensemble score
reaches it.

Known limits (by design, to keep the loop simple and safe)
----------------------------------------------------------
* Only alerted events get verdicts, so the labelled rows are biased towards high scores. That is
  why the meta-learner is *continued* from its trained trees instead of refit from scratch, and
  why the threshold can only move a bounded step per run.
* A changed verdict (same analyst, same alert) replaces the stored one and is used for every
  later refit, but the running EMAs / burst counters of the calibrator cannot un-count the first
  verdict (the same convention as the Day 14 history counters).
* The classifier temperature is fitted on the binary anomaly score, an approximation of the
  3-class softmax temperature; the step is clamped to a factor of two.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.models.ensemble_detector import (
    LSTM_CLASSIFIER,
    META_FEATURE_NAMES,
    N_MEMBERS,
    MetaLearner,
    best_f1_threshold,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.models.ensemble_detector import EnsembleDetector
    from src.models.model_registry import ModelRegistry
    from src.utils.config import FeedbackConfig

_log = get_logger(__name__)

FEEDBACK_TYPES: Final[tuple[str, ...]] = ("false_positive", "true_positive", "severity_adjustment")
LABEL_TYPES: Final[tuple[str, ...]] = ("false_positive", "true_positive")
SEVERITIES: Final[tuple[str, ...]] = ("INFO", "WARNING", "CRITICAL", "EMERGENCY")
MAX_NOTES_CHARS: Final = 1000
MIN_THRESHOLD: Final = 0.05
MAX_THRESHOLD: Final = 0.95
_CLASSIFIER_SCORE_COLUMN: Final = N_MEMBERS - 1  # lstm_classifier is the last member
_STORE_VERSION: Final = 2


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


class FeedbackValidationError(ValueError):
    """A feedback event is malformed (unknown type, missing field, out-of-range value)."""


# --------------------------------------------------------------------------- #
# Feedback event
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeedbackEvent:
    """One analyst verdict on one alert.

    Attributes:
        event_id: Id of this feedback event.
        anomaly_event_id: The alert the verdict is about.
        analyst_id: Who gave it.
        feedback_type: ``false_positive``, ``true_positive`` or ``severity_adjustment``.
        corrected_severity: The analyst's severity (``severity_adjustment`` only).
        notes: Free text.
        timestamp: When the verdict was given (UTC).
        entity_id: Sensor / vehicle / SKU of the alert (the unit of threshold calibration).
        missed: A ``true_positive`` for an anomaly the system failed to alert on (a *missed* TP).
        original_severity: The severity the alert carried (for severity-adjustment analytics).
        meta_features: The meta-learner input row of the alert (``None`` = missing value);
            what the recalibrator trains on. Absent for verdicts on unknown alerts.
    """

    event_id: str
    anomaly_event_id: str
    analyst_id: str
    feedback_type: str
    corrected_severity: str | None = None
    notes: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    entity_id: str = ""
    missed: bool = False
    original_severity: str | None = None
    meta_features: tuple[float | None, ...] | None = None

    @classmethod
    def create(
        cls,
        anomaly_event_id: str,
        analyst_id: str,
        feedback_type: str,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create an event with a fresh id (and ``timestamp`` = now unless given)."""
        return cls(
            event_id=str(uuid.uuid4()),
            anomaly_event_id=anomaly_event_id,
            analyst_id=analyst_id,
            feedback_type=feedback_type,
            **kwargs,
        )

    @property
    def is_label(self) -> bool:
        """Whether the verdict is a false / true positive (not a severity correction)."""
        return self.feedback_type in LABEL_TYPES

    @property
    def is_false_positive(self) -> bool:
        """Whether the analyst called the alert a false positive."""
        return self.feedback_type == "false_positive"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "event_id": self.event_id,
            "anomaly_event_id": self.anomaly_event_id,
            "analyst_id": self.analyst_id,
            "feedback_type": self.feedback_type,
            "corrected_severity": self.corrected_severity,
            "notes": self.notes,
            "timestamp": _aware(self.timestamp).isoformat(),
            "entity_id": self.entity_id,
            "missed": self.missed,
            "original_severity": self.original_severity,
            "meta_features": None if self.meta_features is None else list(self.meta_features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeedbackEvent:
        """Rebuild an event from :meth:`to_dict` output.

        Also reads the Day 14 ``FeedbackRecord`` layout (``feedback_id``, ``severity``).
        """
        features = data.get("meta_features")
        return cls(
            event_id=str(data.get("event_id") or data["feedback_id"]),
            anomaly_event_id=str(data["anomaly_event_id"]),
            analyst_id=str(data["analyst_id"]),
            feedback_type=str(data["feedback_type"]),
            corrected_severity=data.get("corrected_severity"),
            notes=data.get("notes"),
            timestamp=_aware(datetime.fromisoformat(str(data["timestamp"]))),
            entity_id=str(data.get("entity_id", "")),
            missed=bool(data.get("missed", False)),
            original_severity=data.get("original_severity") or data.get("severity") or None,
            meta_features=None if features is None else tuple(features),
        )


def validate_feedback(event: FeedbackEvent) -> None:
    """Check a feedback event; raise :class:`FeedbackValidationError` on the first problem."""
    if event.feedback_type not in FEEDBACK_TYPES:
        raise FeedbackValidationError(
            f"feedback_type must be one of {list(FEEDBACK_TYPES)}, got {event.feedback_type!r}"
        )
    for name in ("anomaly_event_id", "analyst_id", "entity_id"):
        if not str(getattr(event, name)).strip():
            raise FeedbackValidationError(f"{name} is required")
    if event.notes is not None and len(event.notes) > MAX_NOTES_CHARS:
        raise FeedbackValidationError(f"notes are limited to {MAX_NOTES_CHARS} characters")
    if event.feedback_type == "severity_adjustment":
        if event.corrected_severity not in SEVERITIES:
            raise FeedbackValidationError(
                f"corrected_severity must be one of {list(SEVERITIES)} for a severity_adjustment"
            )
    elif event.corrected_severity is not None:
        raise FeedbackValidationError("corrected_severity only applies to severity_adjustment")
    if event.missed and event.feedback_type != "true_positive":
        raise FeedbackValidationError("only a true_positive can be a missed detection")
    if event.meta_features is not None:
        if len(event.meta_features) != len(META_FEATURE_NAMES):
            raise FeedbackValidationError(
                f"meta_features must have {len(META_FEATURE_NAMES)} values"
            )
        if any(v is not None and not math.isfinite(v) for v in event.meta_features):
            raise FeedbackValidationError("meta_features must be finite or null")


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class FeedbackStore:
    """Thread-safe store of feedback events, persisted to JSON (``feedback_store.json``).

    One verdict per ``(alert, analyst, kind)``: a later label from the same analyst on the same
    alert replaces the earlier one (analysts change their minds); a severity correction is a
    separate kind and never replaces a label. Every stored event gets a monotonically increasing
    sequence number, which the recalibrator uses to consume each verdict once.

    Writes are atomic (temp file + ``os.replace``): a crash never leaves a half-written file.
    """

    def __init__(self, path: str | Path | None = None, *, autosave: bool = False) -> None:
        """Create an empty store.

        Args:
            path: JSON file for :meth:`save` / :meth:`load` (``None``: memory only).
            autosave: Persist after every :meth:`add` (a failed write is logged, not raised).
        """
        self.path = Path(path) if path is not None else None
        self._autosave = autosave
        self._rows: dict[tuple[str, str, str], tuple[int, FeedbackEvent]] = {}
        self._seq = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def attach(self, path: str | Path, *, autosave: bool = True) -> None:
        """Bind the store to ``path`` (and persist after every :meth:`add` if ``autosave``)."""
        self.path = Path(path)
        self._autosave = autosave

    @property
    def total_added(self) -> int:
        """Events ever added (the last sequence number)."""
        with self._lock:
            return self._seq

    @staticmethod
    def _key(event: FeedbackEvent) -> tuple[str, str, str]:
        kind = "label" if event.is_label else "severity"
        return (event.anomaly_event_id, event.analyst_id, kind)

    def add(self, event: FeedbackEvent) -> FeedbackEvent | None:
        """Store ``event``; return the verdict it replaced, if any."""
        with self._lock:
            self._seq += 1
            previous = self._rows.get(self._key(event))
            self._rows[self._key(event)] = (self._seq, event)
            if self._autosave and self.path is not None:
                self._save_quietly()
        return previous[1] if previous else None

    def all(self) -> list[FeedbackEvent]:
        """Every stored event, oldest first."""
        with self._lock:
            rows = [event for _, event in self._rows.values()]
        return sorted(rows, key=lambda e: e.timestamp)

    def labelled_since(self, seq: int = 0) -> list[tuple[int, FeedbackEvent]]:
        """Return false / true positive verdicts numbered above ``seq`` (oldest first)."""
        with self._lock:
            rows = [(s, e) for s, e in self._rows.values() if e.is_label and s > seq]
        return sorted(rows, key=lambda row: row[0])

    def for_alert(self, anomaly_event_id: str) -> list[FeedbackEvent]:
        """All verdicts on one alert (oldest first)."""
        return [e for e in self.all() if e.anomaly_event_id == anomaly_event_id]

    # ------------------------------------------------------------------ #
    def _save_quietly(self) -> None:
        try:
            self.save()
        except OSError as error:
            _log.error("feedback_store_save_failed", path=str(self.path), error=repr(error))

    def save(self, path: str | Path | None = None) -> int:
        """Write every event as JSON, atomically; return the count.

        Raises:
            ValueError: No path was given here or at construction.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save the feedback store to")
        with self._lock:
            rows = sorted(self._rows.values(), key=lambda row: row[0])
            payload = {
                "version": _STORE_VERSION,
                "seq": self._seq,
                "events": [{**event.to_dict(), "seq": seq} for seq, event in rows],
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return len(rows)

    def load(self, path: str | Path | None = None) -> int:
        """Load events written by :meth:`save` (or by the Day 14 store); return the count.

        A missing or unreadable file loads nothing; a bad row is skipped and logged.
        """
        source = Path(path) if path is not None else self.path
        if source is None or not source.is_file():
            return 0
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _log.warning("feedback_store_unreadable", path=str(source), error=repr(error))
            return 0
        raw = data.get("events")
        if raw is None:  # Day 14 layout: {"version": 1, "feedback": [...]}, no sequence numbers
            raw = sorted(data.get("feedback", []), key=lambda r: str(r.get("timestamp", "")))
        loaded = 0
        with self._lock:
            for row in raw:
                try:
                    event = FeedbackEvent.from_dict(row)
                    validate_feedback(event)
                except (KeyError, ValueError, TypeError) as error:
                    _log.warning("feedback_event_skipped", error=repr(error))
                    continue
                self._seq = max(self._seq + 1, int(row.get("seq", 0)))
                self._rows[self._key(event)] = (self._seq, event)
                loaded += 1
            self._seq = max(self._seq, int(data.get("seq", 0)))
        return loaded


# --------------------------------------------------------------------------- #
# Per-entity threshold calibration
# --------------------------------------------------------------------------- #
@dataclass
class EntityCalibration:
    """Calibration state of one entity.

    Attributes:
        fp_rate: EMA of "the verdict was a false positive" (0 until a verdict arrives).
        tp_rate: EMA of "the verdict was a true positive".
        adjustment: Added to the base threshold (positive: fewer alerts).
        n_fp: False positives seen.
        n_tp: True positives seen.
        missed_tps: Missed true positives since the last lowering.
        fp_times: Epoch seconds of the false positives inside the burst window.
    """

    fp_rate: float = 0.0
    tp_rate: float = 0.0
    adjustment: float = 0.0
    n_fp: int = 0
    n_tp: int = 0
    missed_tps: int = 0
    fp_times: deque[float] = field(default_factory=deque, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot."""
        return {
            "fp_rate": round(self.fp_rate, 6),
            "tp_rate": round(self.tp_rate, 6),
            "current_threshold_adjustment": round(self.adjustment, 6),
            "n_fp": self.n_fp,
            "n_tp": self.n_tp,
            "missed_tps": self.missed_tps,
            "recent_fps": len(self.fp_times),
        }


class ThresholdCalibrator:
    """Per-entity decision thresholds learned from analyst verdicts.

    * Every verdict updates the entity's EMAs: ``rate = (1 - alpha) * rate + alpha * outcome``.
    * More than ``fp_burst_count`` false positives inside ``fp_burst_window_minutes`` raises the
      entity's threshold by ``fp_threshold_raise`` (and starts a fresh count, so a continuing
      burst raises it again only after another ``fp_burst_count + 1`` false positives).
    * More than ``missed_tp_count`` missed true positives lower it by
      ``missed_tp_threshold_lower`` (and restart the missed count).
    * The total adjustment stays within ``[-max_lower, +max_raise]`` so a noisy analyst cannot mute
      an entity completely.

    Thread-safe.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        fp_burst_count: int = 5,
        fp_burst_window_minutes: float = 60.0,
        fp_threshold_raise: float = 0.1,
        missed_tp_count: int = 3,
        missed_tp_threshold_lower: float = 0.05,
        max_raise: float = 0.3,
        max_lower: float = 0.2,
    ) -> None:
        """Create a calibrator (all entities start unadjusted)."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if fp_burst_count < 1 or missed_tp_count < 1:
            raise ValueError("burst counts must be at least 1")
        if min(fp_threshold_raise, missed_tp_threshold_lower, max_raise, max_lower) < 0:
            raise ValueError("threshold steps and caps must be non-negative")
        self.alpha = alpha
        self.fp_burst_count = fp_burst_count
        self.fp_burst_window_s = fp_burst_window_minutes * 60.0
        self.fp_threshold_raise = fp_threshold_raise
        self.missed_tp_count = missed_tp_count
        self.missed_tp_threshold_lower = missed_tp_threshold_lower
        self.max_raise = max_raise
        self.max_lower = max_lower
        self._entities: dict[str, EntityCalibration] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg: FeedbackConfig) -> ThresholdCalibrator:
        """Create the calibrator from ``config.feedback``."""
        return cls(
            alpha=cfg.ema_alpha,
            fp_burst_count=cfg.fp_burst_count,
            fp_burst_window_minutes=cfg.fp_burst_window_minutes,
            fp_threshold_raise=cfg.fp_threshold_raise,
            missed_tp_count=cfg.missed_tp_count,
            missed_tp_threshold_lower=cfg.missed_tp_threshold_lower,
            max_raise=cfg.max_threshold_raise,
            max_lower=cfg.max_threshold_lower,
        )

    def update(
        self,
        entity_id: str,
        is_fp: bool,
        *,
        missed: bool = False,
        timestamp: datetime | None = None,
    ) -> EntityCalibration:
        """Fold one verdict into the entity's state; return the (live) state.

        Args:
            entity_id: The entity the alert belonged to.
            is_fp: ``True`` for a false positive, ``False`` for a true positive.
            missed: The true positive was an anomaly that was *not* alerted on.
            timestamp: When the verdict was given (default now); drives the burst window.

        Raises:
            ValueError: ``missed`` together with ``is_fp``.
        """
        if missed and is_fp:
            raise ValueError("a missed detection is a true positive, not a false positive")
        now = _aware(timestamp or _utcnow()).timestamp()
        with self._lock:
            state = self._entities.setdefault(entity_id, EntityCalibration())
            a = self.alpha
            state.fp_rate = (1 - a) * state.fp_rate + a * float(is_fp)
            state.tp_rate = (1 - a) * state.tp_rate + a * float(not is_fp)
            if is_fp:
                state.n_fp += 1
                state.fp_times.append(now)
                horizon = now - self.fp_burst_window_s
                while state.fp_times and state.fp_times[0] < horizon:
                    state.fp_times.popleft()
                if len(state.fp_times) > self.fp_burst_count:
                    state.adjustment = min(
                        state.adjustment + self.fp_threshold_raise, self.max_raise
                    )
                    state.fp_times.clear()
                    _log.warning(
                        "entity_threshold_raised",
                        entity_id=entity_id,
                        adjustment=round(state.adjustment, 4),
                    )
            else:
                state.n_tp += 1
                if missed:
                    state.missed_tps += 1
                    if state.missed_tps > self.missed_tp_count:
                        state.adjustment = max(
                            state.adjustment - self.missed_tp_threshold_lower, -self.max_lower
                        )
                        state.missed_tps = 0
                        _log.warning(
                            "entity_threshold_lowered",
                            entity_id=entity_id,
                            adjustment=round(state.adjustment, 4),
                        )
            return state

    def get_adjustment(self, entity_id: str) -> float:
        """Return the entity's current adjustment (0 for an unknown entity)."""
        with self._lock:
            state = self._entities.get(entity_id)
            return state.adjustment if state else 0.0

    def get_adjusted_threshold(self, entity_id: str, base_threshold: float) -> float:
        """Return ``base_threshold + adjustment`` for the entity, kept inside ``[0.05, 0.95]``."""
        adjusted = base_threshold + self.get_adjustment(entity_id)
        return float(min(max(adjusted, MIN_THRESHOLD), MAX_THRESHOLD))

    def state(self, entity_id: str) -> dict[str, Any] | None:
        """Snapshot of one entity's state (``None`` if it never received feedback)."""
        with self._lock:
            state = self._entities.get(entity_id)
            return state.to_dict() if state else None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Snapshot of every entity's state."""
        with self._lock:
            return {k: v.to_dict() for k, v in sorted(self._entities.items())}

    def rebuild(self, events: Iterable[FeedbackEvent]) -> int:
        """Replay stored verdicts (oldest first) after a restart; return how many were applied."""
        applied = 0
        for event in sorted(events, key=lambda e: e.timestamp):
            if event.is_label and event.entity_id:
                self.update(
                    event.entity_id,
                    event.is_false_positive,
                    missed=event.missed,
                    timestamp=event.timestamp,
                )
                applied += 1
        return applied


# --------------------------------------------------------------------------- #
# Ensemble recalibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecalibrationReport:
    """Outcome of one :meth:`EnsembleRecalibrator.run_once`.

    Attributes:
        status: ``applied`` (the new meta-learner / temperature is serving), ``rejected`` (the
            challenger did not beat the current model on held-out verdicts) or ``skipped``.
        reason: Why it was skipped / rejected (empty when applied).
        n_feedback_events_used: Labelled verdicts consumed by this run.
        n_train: Rows the new trees were boosted on.
        n_validation: Held-out rows the challenger was judged on.
        old_weights / new_weights: Normalised gain share of each meta-feature before / after.
        old_threshold / new_threshold: The meta-learner's operating threshold (probability).
        old_log_loss / new_log_loss: Held-out log loss of the current model / challenger.
        estimated_fp_reduction: Share of held-out false positives the new model + threshold would
            no longer alert on (``None`` if there were none).
        tp_retention: Share of held-out true positives that are still alerted on.
        temperature_old / temperature_new: Mean LSTM-classifier temperature before / after.
        registry_version: Registry version the new meta-learner was saved as (if persisted).
        ran_at: When the run finished (UTC).
        duration_s: Wall time of the run.
    """

    status: Literal["applied", "rejected", "skipped"]
    reason: str = ""
    n_feedback_events_used: int = 0
    n_train: int = 0
    n_validation: int = 0
    old_weights: dict[str, float] = field(default_factory=dict)
    new_weights: dict[str, float] = field(default_factory=dict)
    old_threshold: float | None = None
    new_threshold: float | None = None
    old_log_loss: float | None = None
    new_log_loss: float | None = None
    estimated_fp_reduction: float | None = None
    tp_retention: float | None = None
    temperature_old: float | None = None
    temperature_new: float | None = None
    registry_version: str | None = None
    ran_at: datetime = field(default_factory=_utcnow)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "status": self.status,
            "reason": self.reason,
            "n_feedback_events_used": self.n_feedback_events_used,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "old_weights": self.old_weights,
            "new_weights": self.new_weights,
            "old_threshold": self.old_threshold,
            "new_threshold": self.new_threshold,
            "old_log_loss": self.old_log_loss,
            "new_log_loss": self.new_log_loss,
            "estimated_fp_reduction": self.estimated_fp_reduction,
            "tp_retention": self.tp_retention,
            "temperature_old": self.temperature_old,
            "temperature_new": self.temperature_new,
            "registry_version": self.registry_version,
            "ran_at": self.ran_at.isoformat(),
            "duration_s": round(self.duration_s, 4),
        }


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-4, 1.0 - 1e-4)
    result: np.ndarray = np.log(clipped / (1.0 - clipped))
    return result


class EnsembleRecalibrator:
    """Re-optimises the ensemble from analyst verdicts, without a full retrain.

    **Meta-learner.** The labelled verdicts (false positive = 0, true positive = 1, each with the
    meta-feature row the alert had) are split into a training and a held-out part. The current
    LightGBM stacker is *continued* (:meth:`MetaLearner.continue_fit`: its trees are kept, a few
    more are boosted on the training part, weighted by recency and class balance). The challenger
    replaces the current stacker only if its held-out log loss is no worse; its operating
    threshold moves towards the F1-optimal one by at most ``max_threshold_shift``. The new
    stacker is shared by every source-type ensemble, saved as a new registry version (so a model
    reload keeps it) and swapped in with one reference assignment.

    **LSTM classifier.** A temperature on the classifier's anomaly logit is fitted on the same
    verdicts (:class:`~src.models.lstm_classifier.TemperatureScaler`) and multiplied into the
    temperature of every classifier detector.

    Each verdict is consumed once: the run remembers the last sequence number it used. It runs
    when ``every_minutes`` have passed or ``after_events`` new verdicts arrived
    (:meth:`maybe_run`, driven by the background thread of :meth:`start` or by hand).
    """

    def __init__(
        self,
        ensembles: Callable[[], Mapping[str, EnsembleDetector]],
        store: FeedbackStore,
        *,
        registry: ModelRegistry | None = None,
        every_minutes: float = 30.0,
        after_events: int = 50,
        min_events: int = 20,
        min_per_class: int = 3,
        n_new_trees: int = 30,
        half_life_hours: float = 72.0,
        max_threshold_shift: float = 0.1,
        validation_fraction: float = 0.25,
        seed: int = 42,
        poll_interval_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Create the recalibrator (nothing runs until :meth:`run_once` / :meth:`start`).

        Args:
            ensembles: Returns the serving ensembles by source type (called on every run, so a
                hot model reload is picked up).
            store: Where the verdicts are.
            registry: Where a recalibrated meta-learner is saved (``None``: not persisted).
            every_minutes: Time trigger.
            after_events: Count trigger (verdicts since the last run).
            min_events: Labelled verdicts needed to refit.
            min_per_class: Verdicts of each class needed to refit.
            n_new_trees: Trees boosted on top of the current stacker.
            half_life_hours: Age at which a verdict's training weight halves.
            max_threshold_shift: Largest move of the stacker threshold per run.
            validation_fraction: Share of verdicts held out to judge the challenger.
            seed: Seed of the split.
            poll_interval_s: How often the background thread checks the triggers.
            clock: Monotonic clock of the time trigger (injectable).
            now: Wall clock used for the recency weights (injectable).
        """
        self._ensembles = ensembles
        self.store = store
        self.registry = registry
        self.every_s = every_minutes * 60.0
        self.after_events = after_events
        self.min_events = min_events
        self.min_per_class = min_per_class
        self.n_new_trees = n_new_trees
        self.half_life_hours = half_life_hours
        self.max_threshold_shift = max_threshold_shift
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.poll_interval_s = poll_interval_s
        self._clock = clock
        self._now = now
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cursor = 0  # last consumed sequence number
        self._last_attempt = clock()
        self._added_at_last_attempt = store.total_added
        self.history: deque[RecalibrationReport] = deque(maxlen=20)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(
        cls,
        cfg: FeedbackConfig,
        ensembles: Callable[[], Mapping[str, EnsembleDetector]],
        store: FeedbackStore,
        **kwargs: Any,
    ) -> EnsembleRecalibrator:
        """Create the recalibrator from ``config.feedback``."""
        return cls(
            ensembles,
            store,
            every_minutes=cfg.recalibrate_every_minutes,
            after_events=cfg.recalibrate_after_events,
            min_events=cfg.recalibrate_min_events,
            n_new_trees=cfg.recalibrate_new_trees,
            max_threshold_shift=cfg.recalibrate_max_threshold_shift,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Triggers and scheduling
    # ------------------------------------------------------------------ #
    @property
    def new_events(self) -> int:
        """Verdicts added since the last attempt."""
        return self.store.total_added - self._added_at_last_attempt

    def is_due(self) -> bool:
        """Whether the count or the time trigger has fired."""
        if self.new_events <= 0:
            return False
        return (
            self.new_events >= self.after_events
            or self._clock() - self._last_attempt >= self.every_s
        )

    def notify(self) -> None:
        """Tell the background thread a verdict arrived (cheap; wakes it if the count is due)."""
        if self.new_events >= self.after_events:
            self._wake.set()

    def maybe_run(self) -> RecalibrationReport | None:
        """Run once if a trigger fired; ``None`` if nothing was due."""
        return self.run_once() if self.is_due() else None

    def start(self) -> None:
        """Start the background thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ensemble-recalibrator", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread (waits for a run in progress, up to ``timeout``)."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.maybe_run()
            except Exception as error:  # noqa: BLE001 - the loop must survive a bad run
                _log.error("recalibration_crashed", error=repr(error), exc_info=error)

    # ------------------------------------------------------------------ #
    # The run
    # ------------------------------------------------------------------ #
    def run_once(self, *, force: bool = False) -> RecalibrationReport:
        """Recalibrate from the verdicts not used yet and return a report.

        Args:
            force: Run even if no trigger fired (the minimum-data checks still apply).
        """
        with self._run_lock:
            started = time.perf_counter()
            with self._state_lock:
                self._last_attempt = self._clock()
                self._added_at_last_attempt = self.store.total_added
            report = self._recalibrate()
            report = _with_duration(report, time.perf_counter() - started)
            self.history.append(report)
            _log.info(
                "ensemble_recalibration",
                status=report.status,
                reason=report.reason,
                n_feedback_events_used=report.n_feedback_events_used,
                new_vs_old_weights={"old": report.old_weights, "new": report.new_weights},
                old_threshold=report.old_threshold,
                new_threshold=report.new_threshold,
                estimated_fp_reduction=report.estimated_fp_reduction,
                tp_retention=report.tp_retention,
                temperature_old=report.temperature_old,
                temperature_new=report.temperature_new,
            )
            return report

    def _recalibrate(self) -> RecalibrationReport:
        rows = [
            (seq, e)
            for seq, e in self.store.labelled_since(self._cursor)
            if e.meta_features is not None
        ]
        if len(rows) < self.min_events:
            return RecalibrationReport(
                "skipped", f"{len(rows)} usable verdicts, need {self.min_events}"
            )
        y = np.array([not e.is_false_positive for _, e in rows], dtype=bool)
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if min(n_pos, n_neg) < self.min_per_class:
            return RecalibrationReport(
                "skipped",
                f"need {self.min_per_class} verdicts of each class, have {n_pos} true / "
                f"{n_neg} false positives",
            )
        features = np.array(
            [[np.nan if v is None else v for v in e.meta_features or ()] for _, e in rows],
            dtype=np.float64,
        )
        now = self._now()
        age_h = np.array([max((now - _aware(e.timestamp)).total_seconds(), 0.0) for _, e in rows])
        recency = 0.5 ** (age_h / 3600.0 / self.half_life_hours)
        balance = np.where(y, 0.5 / n_pos, 0.5 / n_neg) * len(y)  # each class sums to n / 2
        weight = recency * balance

        ensembles = dict(self._ensembles())
        champion = next(
            (
                ens.meta_learner
                for ens in ensembles.values()
                if ens.meta_learner is not None and ens.meta_learner.is_fitted
            ),
            None,
        )
        temp_old = self._mean_temperature(ensembles)
        report = self._recalibrate_meta(champion, features, y, weight, len(rows))
        temp_new = self._recalibrate_temperature(ensembles, features, y)
        self._cursor = max(seq for seq, _ in rows)  # every usable verdict is consumed once
        return replace(
            report,
            temperature_old=temp_old,
            temperature_new=temp_old if temp_new is None else temp_new,
        )

    def _recalibrate_meta(
        self,
        champion: MetaLearner | None,
        features: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        n_rows: int,
    ) -> RecalibrationReport:
        if champion is None:
            return RecalibrationReport(
                "skipped",
                "no fitted meta-learner (weighted-average fallback is serving)",
                n_feedback_events_used=n_rows,
            )
        idx = np.arange(len(y))
        n_val = int(round(self.validation_fraction * len(y)))
        if n_val >= 2 and min(int(y.sum()), int((~y).sum())) >= 2:
            train_idx, val_idx = train_test_split(
                idx, test_size=n_val, stratify=y, random_state=self.seed
            )
        else:
            train_idx, val_idx = idx, idx[:0]
        try:
            challenger = champion.continue_fit(
                features[train_idx],
                y[train_idx],
                sample_weight=weight[train_idx],
                n_estimators=self.n_new_trees,
            )
        except (ValueError, RuntimeError) as error:
            return RecalibrationReport(
                "skipped", f"refit failed: {error}", n_feedback_events_used=n_rows
            )

        judge = val_idx if len(val_idx) else idx
        old_p = champion.predict_proba(features[judge])
        new_p = challenger.predict_proba(features[judge])
        labels = y[judge]
        old_ll = float(log_loss(labels, np.clip(old_p, 1e-6, 1 - 1e-6), labels=[False, True]))
        new_ll = float(log_loss(labels, np.clip(new_p, 1e-6, 1 - 1e-6), labels=[False, True]))

        # Move the operating threshold towards the F1-optimal one, at most `max_threshold_shift`.
        candidate = best_f1_threshold(y, challenger.predict_proba(features))
        shift = float(
            np.clip(
                candidate - champion.threshold, -self.max_threshold_shift, self.max_threshold_shift
            )
        )
        challenger.threshold = float(np.clip(champion.threshold + shift, 0.01, 0.99))

        old_flag = old_p >= champion.threshold
        new_flag = new_p >= challenger.threshold
        negatives, positives = ~labels, labels
        fp_old = int((old_flag & negatives).sum())
        fp_new = int((new_flag & negatives).sum())
        estimated_fp_reduction = (fp_old - fp_new) / fp_old if fp_old else None
        tp_old = int((old_flag & positives).sum())
        tp_new = int((new_flag & positives).sum())
        tp_retention = tp_new / tp_old if tp_old else None

        base = RecalibrationReport(
            "applied",
            n_feedback_events_used=n_rows,
            n_train=len(train_idx),
            n_validation=len(val_idx),
            old_weights=_round(champion.feature_importance()),
            new_weights=_round(challenger.feature_importance()),
            old_threshold=round(champion.threshold, 6),
            new_threshold=round(challenger.threshold, 6),
            old_log_loss=round(old_ll, 6),
            new_log_loss=round(new_ll, 6),
            estimated_fp_reduction=_maybe_round(estimated_fp_reduction),
            tp_retention=_maybe_round(tp_retention),
        )
        if new_ll > old_ll + 1e-9:
            return replace(
                base, status="rejected", reason="challenger is worse on held-out verdicts"
            )
        version = self._publish(challenger)
        return replace(base, registry_version=version)

    def _publish(self, challenger: MetaLearner) -> str | None:
        """Swap ``challenger`` into every ensemble that had a stacker and persist it."""
        ensembles = self._ensembles()
        targets = [e for e in ensembles.values() if e.meta_learner is not None]
        for ensemble in targets:
            ensemble.meta_learner = challenger  # one reference assignment: the hot swap
        if self.registry is None or not targets:
            return None
        try:
            saved = targets[0].save_meta_learner(
                self.registry, extra={"source": "feedback_recalibration"}
            )
        except Exception as error:  # noqa: BLE001 - serving is fine; only persistence failed
            _log.warning("recalibrated_meta_learner_not_saved", error=repr(error))
            return None
        return str(saved.version)

    @staticmethod
    def _classifiers(ensembles: Mapping[str, EnsembleDetector]) -> list[Any]:
        found: dict[int, Any] = {}
        for ensemble in ensembles.values():
            member = ensemble.members.get(LSTM_CLASSIFIER)
            if member is not None and hasattr(member, "temperature"):
                found[id(member)] = member
        return list(found.values())

    def _mean_temperature(self, ensembles: Mapping[str, EnsembleDetector]) -> float | None:
        detectors = self._classifiers(ensembles)
        if not detectors:
            return None
        return round(float(np.mean([d.temperature.temperature for d in detectors])), 6)

    def _recalibrate_temperature(
        self, ensembles: Mapping[str, EnsembleDetector], features: np.ndarray, y: np.ndarray
    ) -> float | None:
        """Fit a temperature on the classifier's anomaly logit; return the new mean ``T``."""
        from src.models.lstm_classifier import TemperatureScaler

        detectors = self._classifiers(ensembles)
        if not detectors:
            return None
        score = features[:, _CLASSIFIER_SCORE_COLUMN]
        known = ~np.isnan(score)
        labels = y[known]
        if known.sum() < self.min_events or len(np.unique(labels)) < 2:
            return None
        logits = np.column_stack([np.zeros(int(known.sum())), _logit(score[known])])
        scaler = TemperatureScaler()
        factor = float(np.clip(scaler.calibrate(logits, labels.astype(int)), 0.5, 2.0))
        for detector in detectors:
            scaled = float(np.clip(detector.temperature.temperature * factor, 0.05, 20.0))
            detector.temperature.temperature = scaled
            detector.temperature.is_fitted = True
        return self._mean_temperature(ensembles)


def _round(values: Mapping[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in values.items()}


def _maybe_round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _with_duration(report: RecalibrationReport, seconds: float) -> RecalibrationReport:
    return replace(report, duration_s=seconds, ran_at=_utcnow())


# --------------------------------------------------------------------------- #
# Drift detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelDriftAlert:
    """Published to the ``model-health`` topic when the feedback pattern has shifted.

    Attributes:
        alert_id: UUID of the alert.
        detected_at: When the test fired (UTC).
        direction: ``increase`` (false positives rising) or ``decrease`` (falling).
        statistic: The Page-Hinkley statistic at the alarm.
        threshold: The alarm level it crossed.
        n_samples: Verdicts since the test last reset.
        baseline_fp_rate: False-positive rate before the estimated change point.
        recent_fp_rate: False-positive rate since the estimated change point.
        recommended_action: What to do about it.
        message: Human-readable summary.
    """

    alert_id: str
    detected_at: datetime
    direction: str
    statistic: float
    threshold: float
    n_samples: int
    baseline_fp_rate: float
    recent_fp_rate: float
    recommended_action: str = "full_retrain"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (the ``model-health`` message)."""
        return {
            "event_type": "ModelDriftAlert",
            "alert_id": self.alert_id,
            "detected_at": self.detected_at.isoformat(),
            "detector": "page_hinkley",
            "direction": self.direction,
            "statistic": round(self.statistic, 6),
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "baseline_fp_rate": round(self.baseline_fp_rate, 6),
            "recent_fp_rate": round(self.recent_fp_rate, 6),
            "recommended_action": self.recommended_action,
            "message": self.message,
        }


class FeedbackDriftDetector:
    """Page-Hinkley test on the stream of verdicts (1 = false positive, 0 = true positive).

    The test accumulates ``x_t - mean_t - delta`` and alarms when the sum climbs more than
    ``threshold`` above its running minimum, i.e. when the false-positive rate has risen and
    stayed up. With ``direction="decrease"`` / ``"both"`` it also watches for a sustained fall
    (analysts suddenly confirming almost everything, which can mean the alerting went blind).
    Tuning (measured on simulated verdict streams, 40 runs each): with ``delta=0.1`` and
    ``threshold=8`` a stable stream raises about 0.1 false alarms per 1000 verdicts at worst (a 50%
    false-positive rate; fewer at lower rates), while a rise from 30% to 60% is flagged after a
    median of ~38 verdicts and from 30% to 90% after ~15. ``delta`` should be about a third of the
    smallest change in rate worth reacting to. Smaller values false-alarm: ``delta=0.005`` (the
    first default) fired on stable data after a few hundred verdicts, and ``delta=0.05`` still
    fired about once per 1300 verdicts at a 50% rate. A larger ``threshold`` trades delay for
    fewer false alarms.

    After an alarm the test resets and needs ``min_samples`` fresh verdicts before it can fire
    again, which doubles as the cool-down.

    Thread-safe. ``publisher`` receives :meth:`ModelDriftAlert.to_dict` (normally
    ``KafkaProducerManager.send_model_health``); its failures are logged, never raised.
    """

    def __init__(
        self,
        *,
        delta: float = 0.1,
        threshold: float = 8.0,
        min_samples: int = 30,
        direction: Literal["increase", "decrease", "both"] = "increase",
        publisher: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Create the detector (see the class docs for the parameters)."""
        if threshold <= 0 or delta < 0 or min_samples < 2:
            raise ValueError("threshold must be > 0, delta >= 0 and min_samples >= 2")
        self.delta = delta
        self.threshold = threshold
        self.min_samples = min_samples
        self.direction = direction
        self.publisher = publisher
        self.alerts: list[ModelDriftAlert] = []
        self._lock = threading.Lock()
        self._reset()

    @classmethod
    def from_config(cls, cfg: FeedbackConfig, **kwargs: Any) -> FeedbackDriftDetector:
        """Create the detector from ``config.feedback``."""
        return cls(
            delta=cfg.drift_delta,
            threshold=cfg.drift_threshold,
            min_samples=cfg.drift_min_samples,
            **kwargs,
        )

    def _reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._up = 0.0
        self._up_min = 0.0
        self._down = 0.0
        self._down_min = 0.0
        self._sum = 0.0  # verdicts that were false positives, since the reset
        # (count, false positives) at the minimum of each statistic: the estimated change point
        self._up_origin = (0, 0.0)
        self._down_origin = (0, 0.0)

    @property
    def statistic(self) -> float:
        """Current Page-Hinkley statistic (the larger of the watched directions)."""
        with self._lock:
            return self._statistic()

    def _statistic(self) -> float:
        up = self._up - self._up_min
        down = self._down - self._down_min
        if self.direction == "increase":
            return up
        if self.direction == "decrease":
            return down
        return max(up, down)

    def update(self, is_fp: bool) -> ModelDriftAlert | None:
        """Feed one verdict; return the alert if the test fired (and publish it)."""
        x = float(is_fp)
        with self._lock:
            self._n += 1
            self._mean += (x - self._mean) / self._n
            self._up += x - self._mean - self.delta
            self._sum += x
            if self._up <= self._up_min:
                self._up_min, self._up_origin = self._up, (self._n, self._sum)
            self._down += self._mean - x - self.delta
            if self._down <= self._down_min:
                self._down_min, self._down_origin = self._down, (self._n, self._sum)
            if self._n < self.min_samples or self._statistic() <= self.threshold:
                return None
            up = self._up - self._up_min
            down = self._down - self._down_min
            if self.direction == "both":
                direction = "increase" if up >= down else "decrease"
            else:
                direction = self.direction
            origin_n, origin_sum = self._up_origin if direction == "increase" else self._down_origin
            if origin_n > 0:  # rate before the estimated change point vs since it
                baseline = origin_sum / origin_n
                recent = (self._sum - origin_sum) / (self._n - origin_n)
            else:  # the change is as old as the test: no earlier baseline exists
                baseline = recent = self._mean
            alert = ModelDriftAlert(
                alert_id=str(uuid.uuid4()),
                detected_at=_utcnow(),
                direction=direction,
                statistic=self._statistic(),
                threshold=self.threshold,
                n_samples=self._n,
                baseline_fp_rate=baseline,
                recent_fp_rate=recent,
                message=(
                    f"analyst feedback shifted: false-positive rate "
                    f"{'rose' if direction == 'increase' else 'fell'} to {recent:.0%} "
                    f"(baseline {baseline:.0%}); the models need a full retrain"
                ),
            )
            self.alerts.append(alert)
            del self.alerts[:-100]
            self._reset()
        self._publish(alert)
        return alert

    def _publish(self, alert: ModelDriftAlert) -> None:
        fields = {k: v for k, v in alert.to_dict().items() if k != "event_type"}
        _log.warning("model_drift_detected", **fields)
        if self.publisher is None:
            return
        try:
            self.publisher(alert.to_dict())
        except Exception as error:  # noqa: BLE001 - a broken broker must not fail the feedback call
            _log.error("model_drift_publish_failed", error=repr(error))


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcessResult:
    """What :meth:`FeedbackProcessor.process` did with one verdict.

    Attributes:
        event: The stored event.
        replaced: The verdict this one replaced (same analyst, same alert, same kind), if any.
        threshold_adjustment: The entity's adjustment after the verdict (labels only).
        drift_alert: The drift alert this verdict triggered, if any.
    """

    event: FeedbackEvent
    replaced: FeedbackEvent | None
    threshold_adjustment: float | None
    drift_alert: ModelDriftAlert | None


class FeedbackProcessor:
    """Front door of the feedback loop: validate, store, calibrate, watch for drift.

    ``process`` runs synchronously and cheaply (the heavy refit happens on the recalibrator's own
    thread, which ``process`` only nudges). A changed verdict replaces the stored one and does
    not count twice in the calibrator or the drift test.
    """

    def __init__(
        self,
        store: FeedbackStore,
        calibrator: ThresholdCalibrator,
        *,
        recalibrator: EnsembleRecalibrator | None = None,
        drift_detector: FeedbackDriftDetector | None = None,
    ) -> None:
        """Wire the components (``recalibrator`` / ``drift_detector`` are optional)."""
        self.store = store
        self.calibrator = calibrator
        self.recalibrator = recalibrator
        self.drift_detector = drift_detector

    def process(self, event: FeedbackEvent) -> ProcessResult:
        """Validate, store and learn from one verdict.

        Raises:
            FeedbackValidationError: The event is malformed (nothing is stored).
        """
        validate_feedback(event)
        replaced = self.store.add(event)
        adjustment: float | None = None
        alert: ModelDriftAlert | None = None
        if event.is_label:
            if replaced is None:
                state = self.calibrator.update(
                    event.entity_id,
                    event.is_false_positive,
                    missed=event.missed,
                    timestamp=event.timestamp,
                )
                adjustment = state.adjustment
                if self.drift_detector is not None:
                    alert = self.drift_detector.update(event.is_false_positive)
            else:
                adjustment = self.calibrator.get_adjustment(event.entity_id)
            if self.recalibrator is not None:
                self.recalibrator.notify()
        _log.info(
            "feedback_processed",
            anomaly_event_id=event.anomaly_event_id,
            entity_id=event.entity_id,
            feedback_type=event.feedback_type,
            replaced=replaced is not None,
            threshold_adjustment=adjustment,
        )
        return ProcessResult(event, replaced, adjustment, alert)

    def restore(self) -> int:
        """Rebuild the calibrator from the stored verdicts (after :meth:`FeedbackStore.load`)."""
        return self.calibrator.rebuild(self.store.all())

    # ------------------------------------------------------------------ #
    def analytics(self, now: datetime | None = None) -> dict[str, Any]:
        """Summarise the feedback: rates, trend, noisy entities, severity corrections, models."""
        moment = _aware(now or _utcnow())
        events = self.store.all()
        labels = [e for e in events if e.is_label]

        def rate(rows: list[FeedbackEvent]) -> dict[str, Any]:
            fp = sum(e.is_false_positive for e in rows)
            return {
                "total": len(rows),
                "false_positive": fp,
                "true_positive": len(rows) - fp,
                "false_positive_rate": fp / len(rows) if rows else 0.0,
            }

        per_entity: dict[str, list[FeedbackEvent]] = {}
        for e in labels:
            per_entity.setdefault(e.entity_id, []).append(e)
        calibration = self.calibrator.snapshot()
        noisy = sorted(
            (
                {"entity_id": k, **rate(v), **(calibration.get(k) or {})}
                for k, v in per_entity.items()
            ),
            key=lambda row: (-row["false_positive"], row["entity_id"]),
        )[:10]

        ups = downs = 0
        for e in events:
            if e.feedback_type == "severity_adjustment" and e.original_severity in SEVERITIES:
                delta = SEVERITIES.index(str(e.corrected_severity)) - SEVERITIES.index(
                    str(e.original_severity)
                )
                ups += delta > 0
                downs += delta < 0
        windows = {
            name: rate([e for e in labels if moment - _aware(e.timestamp) <= span])
            for name, span in (("last_hour", timedelta(hours=1)), ("last_24h", timedelta(hours=24)))
        }
        recalibrator = self.recalibrator
        drift = self.drift_detector
        return {
            "overall": rate(labels),
            "windows": windows,
            "missed_detections": sum(e.missed for e in labels),
            "severity_adjustments": {
                "total": sum(e.feedback_type == "severity_adjustment" for e in events),
                "upgrades": ups,
                "downgrades": downs,
            },
            "analysts": len({e.analyst_id for e in events}),
            "noisiest_entities": noisy,
            "adjusted_entities": {
                k: v for k, v in calibration.items() if v["current_threshold_adjustment"] != 0
            },
            "last_recalibration": (
                recalibrator.history[-1].to_dict()
                if recalibrator is not None and recalibrator.history
                else None
            ),
            "drift": (
                {
                    "statistic": round(drift.statistic, 6),
                    "threshold": drift.threshold,
                    "alerts": len(drift.alerts),
                    "last_alert": drift.alerts[-1].to_dict() if drift.alerts else None,
                }
                if drift is not None
                else None
            ),
        }

    def timeline(self, days: int = 7, now: datetime | None = None) -> dict[str, Any]:
        """Daily false-positive rates, the day x hour verdict volume and per-entity tallies.

        Feeds the dashboard's feedback analytics (Day 21). Days are UTC calendar days and the last
        one is the day of ``now``; a day without verdicts is kept with a ``None`` rate so a chart
        shows a gap, not a false "0 % false positives".

        Args:
            days: Window length in days (1-90).
            now: Reference time (default: the wall clock).

        Returns:
            ``{"days", "start", "end", "daily", "heatmap", "entities"}``:

            * ``daily``: one row per day, ``{"date", "total", "false_positive",
              "false_positive_rate"}`` (labels only).
            * ``heatmap``: the non-empty ``{"date", "hour", "count"}`` cells; counts *every*
              feedback event (including severity adjustments), because it shows analyst activity.
            * ``entities``: per-entity label tallies in the window, highest false-positive rate
              first.
        """
        span = max(1, min(int(days), 90))
        today = _aware(now or _utcnow()).astimezone(UTC).date()
        first = today - timedelta(days=span - 1)
        daily = {
            first + timedelta(days=i): [0, 0] for i in range(span)
        }  # [labels, false positives]
        cells: dict[tuple[str, int], int] = {}
        per_entity: dict[str, list[int]] = {}
        for event in self.store.all():
            when = _aware(event.timestamp).astimezone(UTC)
            if not first <= when.date() <= today:
                continue
            key = (when.date().isoformat(), when.hour)
            cells[key] = cells.get(key, 0) + 1
            if not event.is_label:
                continue
            row = daily[when.date()]
            row[0] += 1
            row[1] += int(event.is_false_positive)
            if event.entity_id:
                tally = per_entity.setdefault(event.entity_id, [0, 0])
                tally[0] += 1
                tally[1] += int(event.is_false_positive)
        entities = sorted(
            (
                {
                    "entity_id": name,
                    "total": total,
                    "false_positive": fp,
                    "false_positive_rate": fp / total,
                }
                for name, (total, fp) in per_entity.items()
            ),
            key=lambda row: (-row["false_positive_rate"], -row["false_positive"], row["entity_id"]),
        )
        return {
            "days": span,
            "start": first.isoformat(),
            "end": today.isoformat(),
            "daily": [
                {
                    "date": day.isoformat(),
                    "total": total,
                    "false_positive": fp,
                    "false_positive_rate": fp / total if total else None,
                }
                for day, (total, fp) in daily.items()
            ],
            "heatmap": [
                {"date": day, "hour": hour, "count": count}
                for (day, hour), count in sorted(cells.items())
            ],
            "entities": entities,
        }


__all__ = [
    "FEEDBACK_TYPES",
    "SEVERITIES",
    "EnsembleRecalibrator",
    "EntityCalibration",
    "FeedbackDriftDetector",
    "FeedbackEvent",
    "FeedbackProcessor",
    "FeedbackStore",
    "FeedbackValidationError",
    "ModelDriftAlert",
    "ProcessResult",
    "RecalibrationReport",
    "ThresholdCalibrator",
    "validate_feedback",
]
