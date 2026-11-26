"""Continuous retraining pipeline (Day 17).

Day 16 keeps the models *calibrated* (thresholds, stacker trees, classifier temperature). This
module handles the heavier case where calibration is not enough and the LSTM weights themselves
must move. It decides *when* to retrain, retrains *safely*, and can *undo* a bad promotion::

    RetrainingTrigger --RetrainingRequest--> RetrainingOrchestrator
                                               1. DataPreparation        feedback + stream data
                                               2. IncrementalFineTuner   replay buffer, lr 1e-5
                                               3. EnsembleRecalibrator   re-fit the meta-learner
                                               4. ChampionChallenger     held-out F1 comparison
                                               5. promote / discard      hot swap through a slot
                                               6. RetrainingAuditLog     one JSON line per run
                                             RollbackManager watches every promotion afterwards

Every :class:`RetrainingRequest`, :class:`RetrainingResult` and :class:`RollbackEvent` is published
to the ``model-health`` topic (``KafkaProducerManager.send_model_health``), next to the
:class:`~src.models.feedback_loop.ModelDriftAlert` that Day 16 already publishes there.

Design decisions
----------------
* **Windows, not events.** The LSTMs consume ``(seq_len, n_features)`` windows, but a feedback
  verdict only stores the ensemble's meta-feature row. :class:`DataPreparation` therefore takes a
  ``window_lookup`` that turns a verdict back into its window (normally a read from the feature
  cache by entity and time); verdicts it cannot resolve are counted and skipped, never guessed.
* **Champions are never mutated.** The fine-tuner trains a deep copy. The champion keeps serving
  until the challenger has beaten it on a held-out set; promotion is one reference assignment on a
  :class:`ModelSlot` (the same hot-swap idiom as :class:`EnsembleRecalibrator`).
* **A fine-tuned classifier is promoted on F1, the autoencoder likewise.** The classifier is
  scored by anomaly F1 at its decision threshold. The autoencoder has no labels at training time,
  so each model is scored with a threshold fitted to its *own* errors on the normal test windows
  (mean + 3 std, the Day 10 rule), which keeps the comparison fair.
* **Rollback uses the analysts' verdicts.** After a promotion, the share of false positives among
  new verdicts is compared with the share before it. A relative rise above ``rollback_error_spike``
  (20%) inside ``rollback_window_minutes`` (60) restores the previous model.

Known limits
------------
* A verdict for a true positive carries no anomaly type, so its class defaults to ``point_class``
  unless the lookup supplies one.
* Rollback needs ``min_observations`` verdicts after the promotion; a quiet hour is not evidence.
* The audit log is append-only JSON lines and is only read back on :meth:`RetrainingAuditLog.load`.
"""

from __future__ import annotations

import copy
import json
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.models.lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderTrainer,
    reconstruction_errors,
)
from src.models.lstm_classifier import (
    DEFAULT_CLASS_NAMES,
    LSTMClassifier,
    LSTMClassifierTrainer,
    SequenceDataset,
    predict_logits,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.models.feedback_loop import (
        EnsembleRecalibrator,
        FeedbackDriftDetector,
        FeedbackEvent,
        FeedbackStore,
        RecalibrationReport,
    )
    from src.models.model_registry import ModelRegistry
    from src.utils.config import RetrainingConfig

_log = get_logger(__name__)

NORMAL: Final = 0
TRIGGER_DRIFT: Final = "drift_alert"
TRIGGER_FEEDBACK: Final = "feedback_volume"
TRIGGER_SCHEDULED: Final = "scheduled"
TRIGGER_MANUAL: Final = "manual"
OUTCOME_PROMOTED: Final = "promoted"
OUTCOME_REJECTED: Final = "rejected"
OUTCOME_SKIPPED: Final = "skipped"
OUTCOME_FAILED: Final = "failed"
_MIN_SPLIT_ROWS: Final = 5


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


Publisher = Callable[[dict[str, Any]], Any]


def _safe_publish(publisher: Publisher | None, event: dict[str, Any]) -> bool:
    """Publish ``event`` and swallow failures: a broken broker must not stop a retrain."""
    if publisher is None:
        return False
    try:
        publisher(event)
    except Exception as error:  # noqa: BLE001 - publishing is best effort
        _log.error(
            "model_health_publish_failed", event_type=event.get("event_type"), error=repr(error)
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabeledWindows:
    """Labelled LSTM input windows.

    Attributes:
        x: ``float32`` windows, shape ``(n, seq_len, n_features)``.
        y: Class index per window; ``0`` is normal, anything else an anomaly.
    """

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float32)
        y = np.asarray(self.y, dtype=np.int64)
        if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x must be (n, seq_len, n_features) and y (n,) with the same n")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @classmethod
    def empty(cls, seq_len: int, n_features: int) -> LabeledWindows:
        """Return a set with no windows (of the given window shape)."""
        return cls(np.empty((0, seq_len, n_features), np.float32), np.empty(0, np.int64))

    @classmethod
    def from_sequences(cls, sequences: Any) -> LabeledWindows:
        """Wrap anything with ``x`` and ``y`` arrays (for example ``LabeledSequences``)."""
        return cls(np.asarray(sequences.x), np.asarray(sequences.y))

    @classmethod
    def concat(cls, parts: Sequence[LabeledWindows]) -> LabeledWindows:
        """Stack several sets (empty ones are ignored).

        Raises:
            ValueError: No parts, or the window shapes differ.
        """
        if not parts:
            raise ValueError("nothing to concatenate")
        kept = [p for p in parts if len(p)] or [parts[0]]
        shapes = {p.x.shape[1:] for p in kept}
        if len(shapes) != 1:
            raise ValueError(f"window shapes differ: {sorted(shapes)}")
        return cls(np.concatenate([p.x for p in kept]), np.concatenate([p.y for p in kept]))

    def subset(self, index: np.ndarray) -> LabeledWindows:
        """Return the windows selected by a boolean mask or an index array."""
        return LabeledWindows(self.x[index], self.y[index])

    @property
    def anomalous(self) -> np.ndarray:
        """Boolean mask of the anomalous windows."""
        return self.y != NORMAL

    def normal(self) -> LabeledWindows:
        """Only the normal windows."""
        return self.subset(self.y == NORMAL)

    def class_counts(self, n_classes: int) -> dict[int, int]:
        """Windows per class index."""
        counts = np.bincount(self.y, minlength=n_classes) if len(self) else np.zeros(n_classes, int)
        return {i: int(c) for i, c in enumerate(counts)}


def _split(
    data: LabeledWindows, fraction: float, seed: int
) -> tuple[LabeledWindows, LabeledWindows]:
    """Split into ``(rest, held_out)``, stratified when every class has at least two windows."""
    n_out = max(1, int(round(fraction * len(data))))
    if len(data) < _MIN_SPLIT_ROWS or n_out >= len(data):
        raise ValueError(
            f"need at least {_MIN_SPLIT_ROWS} windows to hold some out, have {len(data)}"
        )
    index = np.arange(len(data))
    counts = np.bincount(data.y)
    stratify = (
        data.y if (counts[counts > 0] >= 2).all() and n_out >= len(counts[counts > 0]) else None
    )
    try:
        rest, out = train_test_split(index, test_size=n_out, stratify=stratify, random_state=seed)
    except ValueError:
        rest, out = train_test_split(index, test_size=n_out, random_state=seed)
    return data.subset(np.sort(rest)), data.subset(np.sort(out))


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrainingRequest:
    """A trigger fired: please retrain.

    Attributes:
        request_id: UUID of the request.
        reason: ``drift_alert``, ``feedback_volume``, ``scheduled`` or ``manual``.
        requested_at: When the trigger fired (UTC).
        details: Trigger-specific facts (alert ids, event counts, merged reasons).
    """

    request_id: str
    reason: str
    requested_at: datetime
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, reason: str, **details: Any) -> RetrainingRequest:
        """Create a request with a fresh id, stamped now."""
        return cls(str(uuid.uuid4()), reason, _utcnow(), dict(details))

    def to_dict(self) -> dict[str, Any]:
        """Return the ``model-health`` message."""
        return {
            "event_type": "RetrainingRequest",
            "request_id": self.request_id,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "details": self.details,
        }


@dataclass(frozen=True)
class ChallengerReport:
    """Held-out comparison of a challenger with the champion it would replace.

    Attributes:
        champion_f1: Anomaly F1 of the serving model.
        challenger_f1: Anomaly F1 of the fine-tuned model.
        improvement: ``challenger_f1 - champion_f1``.
        promoted: ``improvement`` exceeds the required margin (the challenger should replace it).
        model_name: Which model this is about.
        n_test: Held-out windows used.
        min_improvement: The margin that was required.
    """

    champion_f1: float
    challenger_f1: float
    improvement: float
    promoted: bool
    model_name: str = ""
    n_test: int = 0
    min_improvement: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "model_name": self.model_name,
            "champion_f1": _round(self.champion_f1),
            "challenger_f1": _round(self.challenger_f1),
            "improvement": _round(self.improvement),
            "promoted": self.promoted,
            "n_test": self.n_test,
            "min_improvement": self.min_improvement,
        }


@dataclass(frozen=True)
class RetrainingResult:
    """Outcome of one retraining run (the audit-log row and the ``model-health`` message).

    Attributes:
        run_id: UUID of the run.
        request_id: The request that started it.
        trigger_reason: Why it ran.
        outcome: ``promoted``, ``rejected``, ``skipped`` or ``failed``.
        reason: Why it was skipped / rejected / failed (empty otherwise).
        started_at / finished_at: Run boundaries (UTC).
        data_size: New labelled windows used for fine-tuning.
        n_feedback_windows / n_stream_windows: Where those windows came from.
        n_replay_windows: Old windows mixed in against forgetting.
        training_time_s: Wall time of the fine-tuning step.
        champion_f1 / challenger_f1 / improvement: The primary model's comparison.
        models: Per-model :class:`ChallengerReport` dicts.
        registry_versions: Registry version of each promoted model.
        recalibration: ``status`` of the meta-learner recalibration (``None`` if not run).
    """

    run_id: str
    request_id: str
    trigger_reason: str
    outcome: str
    reason: str = ""
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime = field(default_factory=_utcnow)
    data_size: int = 0
    n_feedback_windows: int = 0
    n_stream_windows: int = 0
    n_replay_windows: int = 0
    training_time_s: float = 0.0
    champion_f1: float | None = None
    challenger_f1: float | None = None
    improvement: float | None = None
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    registry_versions: dict[str, str] = field(default_factory=dict)
    recalibration: str | None = None

    @property
    def promoted(self) -> bool:
        """Whether any model was promoted."""
        return self.outcome == OUTCOME_PROMOTED

    def to_dict(self) -> dict[str, Any]:
        """Return the ``model-health`` message / audit-log row."""
        return {
            "event_type": "RetrainingResult",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "trigger_reason": self.trigger_reason,
            "outcome": self.outcome,
            "reason": self.reason,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "data_size": self.data_size,
            "n_feedback_windows": self.n_feedback_windows,
            "n_stream_windows": self.n_stream_windows,
            "n_replay_windows": self.n_replay_windows,
            "training_time_s": round(self.training_time_s, 4),
            "champion_f1": _round(self.champion_f1),
            "challenger_f1": _round(self.challenger_f1),
            "improvement": _round(self.improvement),
            "models": self.models,
            "registry_versions": self.registry_versions,
            "recalibration": self.recalibration,
        }


@dataclass(frozen=True)
class RollbackEvent:
    """A promoted model was withdrawn because analysts' verdicts got worse.

    Attributes:
        rollback_id: UUID of the event.
        model_name: The model that was rolled back.
        run_id: The retraining run that promoted it.
        rolled_back_at: When (UTC).
        baseline_error_rate: False-positive share of the verdicts before the promotion.
        observed_error_rate: False-positive share since the promotion.
        n_observations: Verdicts since the promotion.
        reason: Human-readable summary.
    """

    rollback_id: str
    model_name: str
    run_id: str | None
    rolled_back_at: datetime
    baseline_error_rate: float
    observed_error_rate: float
    n_observations: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the ``model-health`` message / audit-log row."""
        return {
            "event_type": "ModelRollback",
            "rollback_id": self.rollback_id,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "rolled_back_at": self.rolled_back_at.isoformat(),
            "baseline_error_rate": _round(self.baseline_error_rate),
            "observed_error_rate": _round(self.observed_error_rate),
            "n_observations": self.n_observations,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
class RetrainingAuditLog:
    """Append-only history of retraining runs and rollbacks (JSON lines).

    Every row has a ``kind`` (``retraining_result`` or ``model_rollback``). Rows are also kept in
    memory (the newest ``max_memory``) so the dashboard needs no file access. A failing write is
    logged and never raised: auditing must not stop a retrain. Thread-safe.
    """

    def __init__(self, path: str | Path | None = None, *, max_memory: int = 500) -> None:
        """Create the log (``path=None`` keeps it in memory only)."""
        self.path = Path(path) if path is not None else None
        self._rows: deque[dict[str, Any]] = deque(maxlen=max_memory)
        self._lock = threading.Lock()

    def record(self, kind: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Append one row and return it."""
        row = {"kind": kind, "logged_at": _utcnow().isoformat(), **entry}
        with self._lock:
            self._rows.append(row)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, default=str) + "\n")
                except OSError as error:
                    _log.error(
                        "retraining_audit_write_failed", path=str(self.path), error=repr(error)
                    )
        return row

    def entries(self, kind: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Return rows oldest first, optionally of one ``kind`` and only the newest ``limit``."""
        with self._lock:
            rows = [r for r in self._rows if kind is None or r.get("kind") == kind]
        return rows[-limit:] if limit else rows

    def load(self) -> int:
        """Read the file back into memory (after a restart); bad lines are skipped.

        Returns:
            Rows loaded.
        """
        if self.path is None or not self.path.exists():
            return 0
        loaded: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    loaded.append(row)
        with self._lock:
            self._rows.clear()
            self._rows.extend(loaded)
        return len(loaded)


# --------------------------------------------------------------------------- #
# Triggers
# --------------------------------------------------------------------------- #
class RetrainingTrigger:
    """Watches three conditions and emits a :class:`RetrainingRequest` when one fires.

    1. **Drift** -- the :class:`FeedbackDriftDetector` has raised a new ``ModelDriftAlert``.
    2. **Volume** -- more than ``feedback_events_trigger`` verdicts were stored since the last
       request.
    3. **Schedule** -- ``scheduled_interval_hours`` passed since the last request.

    :meth:`check` is cheap and side-effect free apart from the request; :meth:`start` polls it
    every ``check_interval_minutes``. **Deduplication:** while a retrain runs (between the request
    and :meth:`mark_finished`) a firing condition is *queued*, not emitted; the queue holds one
    request (further reasons are merged into it) and :meth:`mark_finished` releases it.
    """

    def __init__(
        self,
        store: FeedbackStore,
        *,
        drift_detector: FeedbackDriftDetector | None = None,
        feedback_events_trigger: int = 500,
        scheduled_interval_hours: float = 24.0,
        check_interval_minutes: float = 15.0,
        publisher: Publisher | None = None,
        on_request: Callable[[RetrainingRequest], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the trigger.

        Args:
            store: Feedback store whose growth is counted.
            drift_detector: Source of drift alerts (``None``: that trigger is off).
            feedback_events_trigger: New verdicts that fire the volume trigger.
            scheduled_interval_hours: Maximum time between retrains.
            check_interval_minutes: Polling period of the background thread.
            publisher: Receives every emitted request (``KafkaProducerManager.send_model_health``).
            on_request: Called by :meth:`poll` / the thread with each emitted request.
            clock: Monotonic clock of the schedule (injectable).

        Raises:
            ValueError: A threshold or interval is not positive.
        """
        if (
            feedback_events_trigger < 1
            or scheduled_interval_hours <= 0
            or check_interval_minutes <= 0
        ):
            raise ValueError("trigger count and intervals must be positive")
        self.store = store
        self.drift_detector = drift_detector
        self.feedback_events_trigger = feedback_events_trigger
        self.scheduled_interval_s = scheduled_interval_hours * 3600.0
        self.check_interval_s = check_interval_minutes * 60.0
        self.publisher = publisher
        self.on_request = on_request
        self._clock = clock
        self._lock = threading.RLock()
        self._running = False
        self._queued: RetrainingRequest | None = None
        self._events_baseline = store.total_added
        self._last_request_at = clock()
        self._seen_alerts: set[str] = {a.alert_id for a in self._alerts()}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(
        cls, cfg: RetrainingConfig, store: FeedbackStore, **kwargs: Any
    ) -> RetrainingTrigger:
        """Create the trigger from ``config.retraining``."""
        return cls(
            store,
            feedback_events_trigger=cfg.feedback_events_trigger,
            scheduled_interval_hours=cfg.scheduled_interval_hours,
            check_interval_minutes=cfg.check_interval_minutes,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        """Whether a retrain is in progress (new firings are queued)."""
        with self._lock:
            return self._running

    @property
    def queued(self) -> RetrainingRequest | None:
        """The request waiting for the running retrain to finish, if any."""
        with self._lock:
            return self._queued

    def _alerts(self) -> list[Any]:
        return list(self.drift_detector.alerts) if self.drift_detector is not None else []

    def _fired(self) -> tuple[list[str], dict[str, Any]]:
        reasons: list[str] = []
        details: dict[str, Any] = {}
        fresh = [a for a in self._alerts() if a.alert_id not in self._seen_alerts]
        if fresh:
            self._seen_alerts.update(a.alert_id for a in fresh)
            reasons.append(TRIGGER_DRIFT)
            details["drift_alert_ids"] = [a.alert_id for a in fresh]
        new_events = self.store.total_added - self._events_baseline
        if new_events > self.feedback_events_trigger:
            reasons.append(TRIGGER_FEEDBACK)
            details["new_feedback_events"] = new_events
        elapsed = self._clock() - self._last_request_at
        if elapsed >= self.scheduled_interval_s:
            reasons.append(TRIGGER_SCHEDULED)
            details["hours_since_last_retrain"] = round(elapsed / 3600.0, 3)
        return reasons, details

    def _consume(self) -> None:
        """Reset the volume and schedule counters: this request accounts for them."""
        self._events_baseline = self.store.total_added
        self._last_request_at = self._clock()

    def check(self) -> RetrainingRequest | None:
        """Evaluate the conditions; return a request to act on now, or ``None``.

        A condition that fires while a retrain is running is queued (one slot, reasons merged)
        and returned later by :meth:`mark_finished`.
        """
        with self._lock:
            reasons, details = self._fired()
            if not reasons:
                return None
            self._consume()
            if self._running:
                self._queue(reasons, details)
                return None
            request = RetrainingRequest.create(reasons[0], reasons=reasons, **details)
            self._running = True
        _safe_publish(self.publisher, request.to_dict())
        _log.info("retraining_requested", reason=request.reason, details=request.details)
        return request

    def _queue(self, reasons: list[str], details: dict[str, Any]) -> None:
        if self._queued is None:
            self._queued = RetrainingRequest.create(reasons[0], reasons=reasons, **details)
        else:
            merged = list(dict.fromkeys([*self._queued.details.get("reasons", []), *reasons]))
            self._queued = RetrainingRequest(
                self._queued.request_id,
                self._queued.reason,
                self._queued.requested_at,
                {**self._queued.details, **details, "reasons": merged},
            )
        _log.info("retraining_queued", reasons=self._queued.details.get("reasons"))

    def trigger_manually(self, **details: Any) -> RetrainingRequest | None:
        """Request a retrain now (queued if one is running). Returns the request if emitted."""
        with self._lock:
            self._consume()
            if self._running:
                self._queue([TRIGGER_MANUAL], details)
                return None
            request = RetrainingRequest.create(TRIGGER_MANUAL, reasons=[TRIGGER_MANUAL], **details)
            self._running = True
        _safe_publish(self.publisher, request.to_dict())
        return request

    def mark_started(self) -> None:
        """Declare that a retrain is running (idempotent; resets the counters if it was idle)."""
        with self._lock:
            if not self._running:
                self._running = True
                self._consume()

    def mark_finished(self) -> RetrainingRequest | None:
        """Declare the retrain finished; return the queued request (and publish it), if any."""
        with self._lock:
            self._running = False
            queued, self._queued = self._queued, None
            if queued is not None:
                self._running = True
        if queued is not None:
            _safe_publish(self.publisher, queued.to_dict())
        return queued

    # ------------------------------------------------------------------ #
    def poll(self) -> RetrainingRequest | None:
        """:meth:`check` once and hand any request to ``on_request``."""
        request = self.check()
        if request is not None and self.on_request is not None:
            self.on_request(request)
        return request

    def start(self) -> None:
        """Poll every ``check_interval_minutes`` on a background thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="retraining-trigger", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.check_interval_s):
            try:
                self.poll()
            except Exception as error:  # noqa: BLE001 - the loop must survive a bad poll
                _log.error("retraining_trigger_failed", error=repr(error), exc_info=error)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
WindowLookup = Callable[["FeedbackEvent"], "tuple[np.ndarray, int | None] | None"]


@dataclass(frozen=True)
class PreparedData:
    """Output of :meth:`DataPreparation.prepare`.

    Attributes:
        new: Fresh labelled windows to fine-tune on (feedback + stream, test part removed).
        replay: Old windows to mix in against catastrophic forgetting.
        test: Held-out windows that judge champion and challenger (never trained on).
        n_feedback: Windows that came from analyst verdicts.
        n_stream: Windows that came from recent streaming data.
        n_unresolved: Verdicts whose window could not be found (skipped).
    """

    new: LabeledWindows
    replay: LabeledWindows | None
    test: LabeledWindows
    n_feedback: int = 0
    n_stream: int = 0
    n_unresolved: int = 0


class DataPreparation:
    """Step 1: gather the labelled windows a retrain needs.

    * **Feedback** -- every labelled verdict in the :class:`FeedbackStore`, resolved to its window
      by ``window_lookup(event) -> (window, class | None)`` (``None`` if it is gone). A false
      positive is a *normal* window; a true positive is the looked-up class, else ``point_class``.
    * **Stream** -- ``stream_source()`` returns recent labelled windows (for example the ground
      truth of the simulator, or windows from confirmed incidents).
    * **Replay** -- ``replay_source()`` returns old training windows.
    * **Test** -- ``holdout_source()`` returns a fixed held-out set; without one a stratified
      ``test_fraction`` of the new data is set aside.
    """

    def __init__(
        self,
        store: FeedbackStore,
        *,
        window_lookup: WindowLookup | None = None,
        stream_source: Callable[[], LabeledWindows | None] | None = None,
        replay_source: Callable[[], LabeledWindows | None] | None = None,
        holdout_source: Callable[[], LabeledWindows | None] | None = None,
        test_fraction: float = 0.2,
        point_class: int = 1,
        seed: int = 42,
    ) -> None:
        """Create the preparation step (every source is optional)."""
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")
        self.store = store
        self.window_lookup = window_lookup
        self.stream_source = stream_source
        self.replay_source = replay_source
        self.holdout_source = holdout_source
        self.test_fraction = test_fraction
        self.point_class = point_class
        self.seed = seed

    def _feedback_windows(self) -> tuple[LabeledWindows | None, int]:
        if self.window_lookup is None:
            return None, 0
        windows: list[np.ndarray] = []
        labels: list[int] = []
        unresolved = 0
        for _, event in self.store.labelled_since(0):
            found = self.window_lookup(event)
            if found is None:
                unresolved += 1
                continue
            window, anomaly_class = found
            windows.append(np.asarray(window, dtype=np.float32))
            labels.append(
                NORMAL if event.is_false_positive else (anomaly_class or self.point_class)
            )
        if not windows:
            return None, unresolved
        return LabeledWindows(np.stack(windows), np.asarray(labels)), unresolved

    def prepare(self) -> PreparedData:
        """Collect, merge and split the data.

        Raises:
            ValueError: There is no data at all, or window shapes disagree.
        """
        feedback, unresolved = self._feedback_windows()
        stream = self.stream_source() if self.stream_source is not None else None
        parts = [p for p in (feedback, stream) if p is not None and len(p)]
        if not parts:
            raise ValueError("no labelled windows: no resolvable feedback and no stream data")
        new = LabeledWindows.concat(parts)
        holdout = self.holdout_source() if self.holdout_source is not None else None
        if holdout is not None and len(holdout):
            test = holdout
        else:
            new, test = _split(new, self.test_fraction, self.seed)
        replay = self.replay_source() if self.replay_source is not None else None
        return PreparedData(
            new=new,
            replay=replay if replay is not None and len(replay) else None,
            test=test,
            n_feedback=0 if feedback is None else len(feedback),
            n_stream=0 if stream is None else len(stream),
            n_unresolved=unresolved,
        )


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FineTuneReport:
    """What :meth:`IncrementalFineTuner.fine_tune` did.

    Attributes:
        model_name: ``lstm_classifier`` or ``lstm_autoencoder``.
        n_new / n_replay: Fresh and replayed windows in the training mix.
        n_train / n_val: Split of that mix.
        epochs_run: Epochs actually trained (early stopping may end sooner).
        learning_rate: The (low) fine-tuning rate.
        duration_s: Wall time.
    """

    model_name: str
    n_new: int
    n_replay: int
    n_train: int
    n_val: int
    epochs_run: int
    learning_rate: float
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "model_name": self.model_name,
            "n_new": self.n_new,
            "n_replay": self.n_replay,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "epochs_run": self.epochs_run,
            "learning_rate": self.learning_rate,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass(frozen=True)
class FineTuneResult:
    """A fine-tuned copy of a model and the report of how it was made."""

    model: Any
    report: FineTuneReport


def _class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights; a class without samples gets weight 1 (it has no loss term)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = int((counts > 0).sum())
    weights = np.ones(n_classes)
    weights[counts > 0] = counts.sum() / (present * counts[counts > 0])
    return weights


class IncrementalFineTuner:
    """Step 2: fine-tune a copy of an LSTM on new data without forgetting the old.

    The training mix is ``replay_ratio`` old windows and ``1 - replay_ratio`` new ones (default
    50 / 50, so the old distribution keeps a voice) and the learning rate is a hundredth of the
    initial one (``1e-5`` against ``1e-3``). Training stops after ``max_epochs`` (10) or earlier
    when the validation score stalls. The model passed in is **never modified**.

    * :class:`LSTMClassifier` -- trained on all classes with class-weighted cross-entropy.
    * :class:`LSTMAutoencoder` -- trained on the *normal* windows only (it learns normality).
    """

    def __init__(
        self,
        *,
        learning_rate: float = 1.0e-5,
        max_epochs: int = 10,
        replay_ratio: float = 0.5,
        validation_fraction: float = 0.2,
        batch_size: int = 64,
        patience: int = 3,
        seed: int = 42,
    ) -> None:
        """Create the fine-tuner.

        Raises:
            ValueError: ``replay_ratio`` is not in ``[0, 1)`` or another value is out of range.
        """
        if not 0.0 <= replay_ratio < 1.0:
            raise ValueError("replay_ratio must be in [0, 1)")
        if learning_rate <= 0 or max_epochs < 1 or batch_size < 1 or patience < 1:
            raise ValueError("learning_rate, max_epochs, batch_size and patience must be positive")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.replay_ratio = replay_ratio
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: RetrainingConfig, **overrides: Any) -> IncrementalFineTuner:
        """Create the fine-tuner from ``config.retraining``."""
        params: dict[str, Any] = {
            "learning_rate": cfg.finetune_learning_rate,
            "max_epochs": cfg.finetune_max_epochs,
            "replay_ratio": min(cfg.replay_buffer_ratio, 0.99),
        }
        params.update(overrides)
        return cls(**params)

    def build_mix(
        self, new: LabeledWindows, replay: LabeledWindows | None
    ) -> tuple[LabeledWindows, int]:
        """Return ``(training mix, replayed count)`` with ``replay_ratio`` old windows.

        If the replay buffer is smaller than needed, all of it is used (and the ratio is lower).
        """
        if replay is None or len(replay) == 0 or self.replay_ratio == 0.0:
            return new, 0
        wanted = int(round(len(new) * self.replay_ratio / (1.0 - self.replay_ratio)))
        n_old = min(wanted, len(replay))
        if n_old < wanted:
            _log.warning("replay_buffer_smaller_than_ratio", wanted=wanted, available=len(replay))
        rng = np.random.default_rng(self.seed)
        chosen = np.sort(rng.choice(len(replay), size=n_old, replace=False))
        return LabeledWindows.concat([new, replay.subset(chosen)]), n_old

    def fine_tune(
        self,
        model: Any,
        new_labeled_data: LabeledWindows,
        replay_data: LabeledWindows | None = None,
    ) -> FineTuneResult:
        """Fine-tune a deep copy of ``model``.

        Args:
            model: An :class:`LSTMClassifier` or :class:`LSTMAutoencoder` (left untouched).
            new_labeled_data: Fresh labelled windows.
            replay_data: Old windows to mix in (``None``: new data only).

        Raises:
            TypeError: Unsupported model type.
            ValueError: Too little data or a feature-count mismatch.
        """
        if not isinstance(model, LSTMClassifier | LSTMAutoencoder):
            raise TypeError(f"cannot fine-tune a {type(model).__name__}")
        started = time.perf_counter()
        is_classifier = isinstance(model, LSTMClassifier)
        new = new_labeled_data if is_classifier else new_labeled_data.normal()
        old = replay_data if (is_classifier or replay_data is None) else replay_data.normal()
        if len(new) == 0:
            raise ValueError("no usable windows to fine-tune on")
        mix, n_old = self.build_mix(new, old)
        train, val = _split(mix, self.validation_fraction, self.seed)
        tuned = copy.deepcopy(model)
        device = next(tuned.parameters()).device
        if isinstance(tuned, LSTMClassifier):
            names = (
                DEFAULT_CLASS_NAMES
                if tuned.n_classes == len(DEFAULT_CLASS_NAMES)
                else ("normal", *(f"anomaly_{i}" for i in range(1, tuned.n_classes)))
            )
            trainer = LSTMClassifierTrainer(
                tuned,
                class_names=names,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                num_workers=0,
                pin_memory=False,
                device=device,
                log_dir=None,
                seed=self.seed,
            )
            history = trainer.fit(
                SequenceDataset(train.x, train.y),
                SequenceDataset(val.x, val.y),
                class_weights=_class_weights(train.y, tuned.n_classes),
            )
            name = "lstm_classifier"
        else:
            ae_trainer = LSTMAutoencoderTrainer(
                tuned,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                device=device,
                log_dir=None,
                seed=self.seed,
            )
            history = ae_trainer.fit(train.x, val.x)
            name = "lstm_autoencoder"
        report = FineTuneReport(
            model_name=name,
            n_new=len(new),
            n_replay=n_old,
            n_train=len(train),
            n_val=len(val),
            epochs_run=int(history.epochs_run),
            learning_rate=self.learning_rate,
            duration_s=time.perf_counter() - started,
        )
        _log.info("incremental_finetune_done", **report.to_dict())
        return FineTuneResult(tuned, report)


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
Scorer = Callable[[Any, LabeledWindows], float]


def classifier_anomaly_f1(model: Any, test: LabeledWindows, *, threshold: float = 0.5) -> float:
    """Anomaly F1 of an :class:`LSTMClassifier` (``1 - P(normal) >= threshold`` is an alert)."""
    logits, _ = predict_logits(model, test.x)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predicted = (1.0 - probability[:, NORMAL]) >= threshold
    return float(f1_score(test.anomalous, predicted, zero_division=0))


def autoencoder_anomaly_f1(model: Any, test: LabeledWindows, *, sigma: float = 3.0) -> float:
    """Anomaly F1 of an :class:`LSTMAutoencoder` at its own normal errors' mean + 3 std.

    Fitting the threshold per model keeps the comparison fair: a model with a lower error scale is
    not penalised for it.
    """
    errors = reconstruction_errors(model, test.x, mc_passes=1).error
    normal = errors[~test.anomalous]
    if len(normal) < 2:
        return 0.0
    threshold = float(normal.mean() + sigma * normal.std())
    return float(f1_score(test.anomalous, errors > threshold, zero_division=0))


class ChampionChallenger:
    """Step 4: judge a challenger against the serving model on held-out data.

    The challenger is promoted only if ``challenger_f1 > champion_f1 + min_improvement`` (0.02):
    a margin guards against promoting noise. :meth:`evaluate` only *measures*; :meth:`promote`
    makes the challenger the champion and keeps the old one (:attr:`previous_champion`) for a
    rollback, :meth:`discard` drops it.
    """

    def __init__(
        self,
        champion_model: Any,
        challenger_model: Any | None = None,
        *,
        scorer: Scorer = classifier_anomaly_f1,
        min_improvement: float = 0.02,
        name: str = "lstm_classifier",
    ) -> None:
        """Hold the two models (the challenger may be set later with :meth:`set_challenger`)."""
        if min_improvement < 0:
            raise ValueError("min_improvement must be >= 0")
        self.champion_model = champion_model
        self.challenger_model = challenger_model
        self.previous_champion: Any | None = None
        self.scorer = scorer
        self.min_improvement = min_improvement
        self.name = name

    def set_challenger(self, model: Any) -> None:
        """Replace the challenger."""
        self.challenger_model = model

    def evaluate(
        self, test_data: LabeledWindows, challenger_model: Any | None = None
    ) -> ChallengerReport:
        """Score both models on ``test_data``.

        Raises:
            ValueError: No challenger, or an empty / single-class test set (F1 would be
                meaningless).
        """
        if challenger_model is not None:
            self.challenger_model = challenger_model
        if self.challenger_model is None:
            raise ValueError("no challenger model to evaluate")
        if len(test_data) == 0 or not test_data.anomalous.any():
            raise ValueError("the test set needs windows and at least one anomaly")
        champion_f1 = float(self.scorer(self.champion_model, test_data))
        challenger_f1 = float(self.scorer(self.challenger_model, test_data))
        improvement = round(challenger_f1 - champion_f1, 10)
        report = ChallengerReport(
            champion_f1=champion_f1,
            challenger_f1=challenger_f1,
            improvement=improvement,
            promoted=improvement > round(self.min_improvement, 10),
            model_name=self.name,
            n_test=len(test_data),
            min_improvement=self.min_improvement,
        )
        _log.info("champion_challenger_evaluated", **report.to_dict())
        return report

    def promote(self) -> Any:
        """Make the challenger the champion; return the model it replaced."""
        if self.challenger_model is None:
            raise ValueError("no challenger to promote")
        self.previous_champion = self.champion_model
        self.champion_model, self.challenger_model = self.challenger_model, None
        return self.previous_champion

    def discard(self) -> None:
        """Drop the challenger."""
        self.challenger_model = None


# --------------------------------------------------------------------------- #
# Serving slots and rollback
# --------------------------------------------------------------------------- #
class ModelSlot(Protocol):
    """Where a serving model lives; promotion and rollback are one :meth:`set` call."""

    def get(self) -> Any:
        """Return the model that is serving."""

    def set(self, model: Any) -> None:
        """Atomically make ``model`` the serving model."""


class SimpleModelSlot:
    """A slot holding one model (tests, scripts)."""

    def __init__(self, model: Any) -> None:
        """Hold ``model``."""
        self.model = model

    def get(self) -> Any:
        """Return the model."""
        return self.model

    def set(self, model: Any) -> None:
        """Replace the model."""
        self.model = model


class DetectorModelSlot:
    """Slot over detectors that share a model (their ``.model`` attribute is swapped).

    ``detectors`` is called on every access, so a hot model reload is picked up. The swap is one
    reference assignment per detector: a window being scored finishes on the old model.
    """

    def __init__(self, detectors: Callable[[], Iterable[Any]]) -> None:
        """Create the slot from a function returning the detectors."""
        self._detectors = detectors

    def get(self) -> Any:
        """Return the model of the first detector.

        Raises:
            LookupError: There is no detector.
        """
        for detector in self._detectors():
            return detector.model
        raise LookupError("no detector to read a model from")

    def set(self, model: Any) -> None:
        """Assign ``model`` to every detector."""
        for detector in self._detectors():
            detector.model = model


@dataclass
class _Watch:
    name: str
    slot: ModelSlot
    previous_model: Any
    run_id: str | None
    promoted_at: datetime
    seq: int
    baseline: float


class RollbackManager:
    """Withdraws a promoted model when analysts' verdicts get clearly worse.

    After :meth:`watch`, :meth:`check` compares the false-positive share of the verdicts stored
    *since the promotion* with the share of the (up to ``baseline_observations``) verdicts before
    it. If at least ``min_observations`` verdicts exist and the new share exceeds
    ``baseline * (1 + error_spike)`` (``error_spike`` = 0.20, so +20% relative; with a baseline of
    zero: above ``error_spike`` itself), the previous model is put back, a :class:`RollbackEvent`
    is published and the watch ends. A watch that stays quiet for ``window_minutes`` (60) expires:
    the promotion is considered stable.
    """

    def __init__(
        self,
        store: FeedbackStore,
        *,
        error_spike: float = 0.20,
        window_minutes: float = 60.0,
        min_observations: int = 20,
        baseline_observations: int = 200,
        publisher: Publisher | None = None,
        on_rollback: Callable[[RollbackEvent], Any] | None = None,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Create the manager (nothing is watched until :meth:`watch`)."""
        if error_spike <= 0 or window_minutes <= 0 or min_observations < 1:
            raise ValueError("error_spike, window_minutes and min_observations must be positive")
        self.store = store
        self.error_spike = error_spike
        self.window_s = window_minutes * 60.0
        self.min_observations = min_observations
        self.baseline_observations = baseline_observations
        self.publisher = publisher
        self.on_rollback = on_rollback
        self._now = now
        self._watches: dict[str, _Watch] = {}
        self.events: list[RollbackEvent] = []
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls, cfg: RetrainingConfig, store: FeedbackStore, **kwargs: Any
    ) -> RollbackManager:
        """Create the manager from ``config.retraining``."""
        return cls(
            store,
            error_spike=cfg.rollback_error_spike,
            window_minutes=cfg.rollback_window_minutes,
            **kwargs,
        )

    @property
    def watching(self) -> list[str]:
        """Names of the models under watch."""
        with self._lock:
            return list(self._watches)

    def _fp_rate(self, rows: Sequence[Any]) -> tuple[float, int]:
        if not rows:
            return 0.0, 0
        false_positives = sum(e.is_false_positive for _, e in rows)
        return false_positives / len(rows), len(rows)

    def watch(
        self, name: str, slot: ModelSlot, previous_model: Any, run_id: str | None = None
    ) -> None:
        """Start watching the promotion of ``name`` (``previous_model`` is what to restore)."""
        rows = self.store.labelled_since(0)
        baseline, _ = self._fp_rate(rows[-self.baseline_observations :])
        seq = rows[-1][0] if rows else 0
        with self._lock:
            self._watches[name] = _Watch(
                name, slot, previous_model, run_id, self._now(), seq, baseline
            )
        _log.info("rollback_watch_started", model=name, baseline_error_rate=round(baseline, 4))

    def check(self) -> list[RollbackEvent]:
        """Evaluate every watch; roll back the ones that spiked and expire the quiet ones."""
        fired: list[RollbackEvent] = []
        with self._lock:
            watches = list(self._watches.values())
        for watch in watches:
            rows = self.store.labelled_since(watch.seq)
            observed, n = self._fp_rate(rows)
            limit = (
                watch.baseline * (1.0 + self.error_spike)
                if watch.baseline > 0
                else self.error_spike
            )
            if n >= self.min_observations and observed > limit:
                fired.append(self._roll_back(watch, observed, n))
            elif (self._now() - watch.promoted_at).total_seconds() >= self.window_s:
                with self._lock:
                    self._watches.pop(watch.name, None)
                _log.info("rollback_watch_expired", model=watch.name, observations=n)
        return fired

    def rollback_now(self, name: str, reason: str = "manual rollback") -> RollbackEvent | None:
        """Restore the previous model of ``name`` immediately (``None`` if it is not watched)."""
        with self._lock:
            watch = self._watches.get(name)
        if watch is None:
            return None
        observed, n = self._fp_rate(self.store.labelled_since(watch.seq))
        return self._roll_back(watch, observed, n, reason)

    def _roll_back(
        self, watch: _Watch, observed: float, n: int, reason: str | None = None
    ) -> RollbackEvent:
        watch.slot.set(watch.previous_model)
        with self._lock:
            self._watches.pop(watch.name, None)
        event = RollbackEvent(
            rollback_id=str(uuid.uuid4()),
            model_name=watch.name,
            run_id=watch.run_id,
            rolled_back_at=self._now(),
            baseline_error_rate=watch.baseline,
            observed_error_rate=observed,
            n_observations=n,
            reason=reason
            or (
                f"false-positive share rose from {watch.baseline:.0%} to {observed:.0%} over "
                f"{n} verdicts after the promotion; restored the previous model"
            ),
        )
        self.events.append(event)
        del self.events[:-100]
        _log.warning("model_rolled_back", **event.to_dict())
        _safe_publish(self.publisher, event.to_dict())
        if self.on_rollback is not None:
            try:
                self.on_rollback(event)
            except Exception as error:  # noqa: BLE001 - the rollback already happened
                _log.error("rollback_callback_failed", error=repr(error))
        return event


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class ModelStage:
    """One retrainable model: where it serves, how it is scored, where it is archived.

    Attributes:
        name: ``lstm_classifier`` or ``lstm_autoencoder`` (also the metrics key).
        slot: Serving slot (promotion and rollback go through it).
        scorer: Held-out score used by the champion / challenger comparison.
        registry_model_type: Model-registry family a promoted model is saved under
            (``None``: not archived).
    """

    name: str
    slot: ModelSlot
    scorer: Scorer = classifier_anomaly_f1
    registry_model_type: str | None = None


class _Recalibrator(Protocol):
    def run_once(self, *, force: bool = False) -> RecalibrationReport: ...


class RetrainingOrchestrator:
    """Runs the six-step retraining pipeline for one :class:`RetrainingRequest` at a time.

    1. :class:`DataPreparation` -- feedback-labelled windows + recent stream data.
    2. :class:`IncrementalFineTuner` -- fine-tune a copy of every stage's model.
    3. :class:`EnsembleRecalibrator` -- re-fit the meta-learner (it validates itself and keeps the
       current stacker if the new one is worse).
    4. :class:`ChampionChallenger` -- compare each challenger with its champion on held-out data.
    5. Promote a stage whose challenger F1 is more than ``min_f1_improvement`` higher (swap through
       its slot, archive in the registry, start a rollback watch); discard the rest.
    6. Log a :class:`RetrainingResult` to the audit log and the ``model-health`` topic.

    A failure anywhere ends the run as ``failed`` and leaves every champion serving. Requests can be
    handled synchronously (:meth:`on_retrain_request`) or queued for a worker thread
    (:meth:`submit` + :meth:`start`), the way :class:`RetrainingTrigger` hands them over.
    """

    def __init__(
        self,
        data: DataPreparation,
        fine_tuner: IncrementalFineTuner,
        stages: Sequence[ModelStage],
        *,
        recalibrator: _Recalibrator | EnsembleRecalibrator | None = None,
        rollback: RollbackManager | None = None,
        trigger: RetrainingTrigger | None = None,
        audit: RetrainingAuditLog | None = None,
        registry: ModelRegistry | None = None,
        publisher: Publisher | None = None,
        min_f1_improvement: float = 0.02,
        min_new_windows: int = 20,
    ) -> None:
        """Wire the pipeline.

        Args:
            data: Step 1.
            fine_tuner: Step 2.
            stages: The models to retrain; the first is the *primary* one whose F1 is reported at
                the top level of the result.
            recalibrator: Step 3 (``None``: skipped).
            rollback: Watches promotions (``None``: promotions are not watched).
            trigger: Told when a run starts and ends so duplicate requests queue.
            audit: Audit log (a new in-memory one if ``None``).
            registry: Archives promoted models.
            publisher: Receives every :class:`RetrainingResult`.
            min_f1_improvement: Margin a challenger must beat its champion by.
            min_new_windows: Fewer new windows than this and fine-tuning is skipped.

        Raises:
            ValueError: No stages.
        """
        if not stages:
            raise ValueError("at least one model stage is required")
        self.data = data
        self.fine_tuner = fine_tuner
        self.stages = list(stages)
        self.recalibrator = recalibrator
        self.rollback = rollback
        self.trigger = trigger
        self.audit = audit if audit is not None else RetrainingAuditLog()
        self.registry = registry
        self.publisher = publisher
        self.min_f1_improvement = min_f1_improvement
        self.min_new_windows = min_new_windows
        self.history: deque[RetrainingResult] = deque(maxlen=100)
        self._run_lock = threading.Lock()
        self._queue: queue.Queue[RetrainingRequest | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        if rollback is not None and rollback.on_rollback is None:
            rollback.on_rollback = lambda event: self.audit.record(
                "model_rollback", event.to_dict()
            )

    @classmethod
    def from_config(
        cls,
        cfg: RetrainingConfig,
        data: DataPreparation,
        stages: Sequence[ModelStage],
        **kwargs: Any,
    ) -> RetrainingOrchestrator:
        """Create the orchestrator (and its fine-tuner) from ``config.retraining``."""
        fine_tuner = kwargs.pop("fine_tuner", None) or IncrementalFineTuner.from_config(cfg)
        return cls(data, fine_tuner, stages, min_f1_improvement=cfg.min_f1_improvement, **kwargs)

    # ------------------------------------------------------------------ #
    # Queueing
    # ------------------------------------------------------------------ #
    def submit(self, request: RetrainingRequest) -> None:
        """Queue a request for the worker thread (start it with :meth:`start`)."""
        self._queue.put(request)

    def start(self) -> None:
        """Start the worker thread that handles queued requests (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, name="retraining-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker after the current run."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._queue.put(None)
            thread.join(timeout)
        self._thread = None

    def _worker(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return
            try:
                self.on_retrain_request(request)
            except Exception as error:  # noqa: BLE001 - on_retrain_request already records failures
                _log.error("retraining_worker_failed", error=repr(error), exc_info=error)

    # ------------------------------------------------------------------ #
    # The pipeline
    # ------------------------------------------------------------------ #
    def on_retrain_request(self, request: RetrainingRequest) -> RetrainingResult:
        """Run the pipeline for ``request`` and return its result (never raises)."""
        with self._run_lock:
            if self.trigger is not None:
                self.trigger.mark_started()
            started_at = _utcnow()
            run_id = str(uuid.uuid4())
            _log.info("retraining_started", run_id=run_id, reason=request.reason)
            try:
                result = self._run(request, run_id, started_at)
            except Exception as error:  # noqa: BLE001 - champions keep serving, the run is logged
                _log.error("retraining_failed", run_id=run_id, error=repr(error), exc_info=error)
                result = RetrainingResult(
                    run_id=run_id,
                    request_id=request.request_id,
                    trigger_reason=request.reason,
                    outcome=OUTCOME_FAILED,
                    reason=f"{type(error).__name__}: {error}",
                    started_at=started_at,
                    finished_at=_utcnow(),
                )
            self.history.append(result)
            self.audit.record("retraining_result", result.to_dict())
            _safe_publish(self.publisher, result.to_dict())
            _log.info(
                "retraining_finished", run_id=run_id, outcome=result.outcome, reason=result.reason
            )
            queued = self.trigger.mark_finished() if self.trigger is not None else None
        if queued is not None:
            self.submit(queued)
        return result

    def _run(
        self, request: RetrainingRequest, run_id: str, started_at: datetime
    ) -> RetrainingResult:
        base = {
            "run_id": run_id,
            "request_id": request.request_id,
            "trigger_reason": request.reason,
            "started_at": started_at,
        }
        # Step 1: data
        try:
            prepared = self.data.prepare()
        except ValueError as error:
            recalibration = self._recalibrate()
            return RetrainingResult(
                **base,
                outcome=OUTCOME_SKIPPED,
                reason=f"no training data: {error}",
                finished_at=_utcnow(),
                recalibration=recalibration,
            )
        counts = {
            "data_size": len(prepared.new),
            "n_feedback_windows": prepared.n_feedback,
            "n_stream_windows": prepared.n_stream,
            "n_replay_windows": 0 if prepared.replay is None else len(prepared.replay),
        }
        if len(prepared.new) < self.min_new_windows:
            recalibration = self._recalibrate()
            return RetrainingResult(
                **base,
                **counts,
                outcome=OUTCOME_SKIPPED,
                reason=f"{len(prepared.new)} new windows, need {self.min_new_windows}",
                finished_at=_utcnow(),
                recalibration=recalibration,
            )
        # Step 2: fine-tune a challenger per stage
        trained = time.perf_counter()
        contenders: dict[str, ChampionChallenger] = {}
        for stage in self.stages:
            champion = stage.slot.get()
            tuned = self.fine_tuner.fine_tune(champion, prepared.new, prepared.replay)
            contenders[stage.name] = ChampionChallenger(
                champion,
                tuned.model,
                scorer=stage.scorer,
                min_improvement=self.min_f1_improvement,
                name=stage.name,
            )
        training_time = time.perf_counter() - trained
        # Step 3: recalibrate the meta-learner
        recalibration = self._recalibrate()
        # Step 4 + 5: compare, promote or discard
        reports: dict[str, ChallengerReport] = {}
        versions: dict[str, str] = {}
        for stage in self.stages:
            contender = contenders[stage.name]
            report = contender.evaluate(prepared.test)
            reports[stage.name] = report
            if not report.promoted:
                contender.discard()
                continue
            candidate = contender.challenger_model
            version = self._archive(stage, candidate, report, request, run_id)
            previous = contender.promote()
            stage.slot.set(candidate)
            if version is not None:
                versions[stage.name] = version
            if self.rollback is not None:
                self.rollback.watch(stage.name, stage.slot, previous, run_id)
            _log.info("model_promoted", model=stage.name, improvement=report.improvement)
        primary = reports[self.stages[0].name]
        promoted = [name for name, r in reports.items() if r.promoted]
        return RetrainingResult(
            **base,
            **counts,
            outcome=OUTCOME_PROMOTED if promoted else OUTCOME_REJECTED,
            reason="" if promoted else "no challenger beat its champion by the required margin",
            finished_at=_utcnow(),
            training_time_s=training_time,
            champion_f1=primary.champion_f1,
            challenger_f1=primary.challenger_f1,
            improvement=primary.improvement,
            models={name: r.to_dict() for name, r in reports.items()},
            registry_versions=versions,
            recalibration=recalibration,
        )

    def _recalibrate(self) -> str | None:
        """Step 3: re-fit the meta-learner; a failure here must not stop the LSTM promotion."""
        if self.recalibrator is None:
            return None
        try:
            return str(self.recalibrator.run_once(force=True).status)
        except Exception as error:  # noqa: BLE001 - the recalibrator keeps the current stacker
            _log.error("retraining_recalibration_failed", error=repr(error))
            return "failed"

    def _archive(
        self,
        stage: ModelStage,
        model: Any,
        report: ChallengerReport,
        request: RetrainingRequest,
        run_id: str,
    ) -> str | None:
        if self.registry is None or stage.registry_model_type is None:
            return None
        try:
            saved = self.registry.save(
                stage.registry_model_type,
                model,
                metrics={"f1": report.challenger_f1, "champion_f1": report.champion_f1},
                hyperparams={"finetune_learning_rate": self.fine_tuner.learning_rate},
                extra={
                    "source": "continuous_retraining",
                    "run_id": run_id,
                    "trigger": request.reason,
                },
            )
        except Exception as error:  # noqa: BLE001 - serving is fine; only persistence failed
            _log.warning("promoted_model_not_archived", model=stage.name, error=repr(error))
            return None
        return str(saved.version)


__all__ = [
    "ChallengerReport",
    "ChampionChallenger",
    "DataPreparation",
    "DetectorModelSlot",
    "FineTuneReport",
    "FineTuneResult",
    "IncrementalFineTuner",
    "LabeledWindows",
    "ModelSlot",
    "ModelStage",
    "PreparedData",
    "RetrainingAuditLog",
    "RetrainingOrchestrator",
    "RetrainingRequest",
    "RetrainingResult",
    "RetrainingTrigger",
    "RollbackEvent",
    "RollbackManager",
    "SimpleModelSlot",
    "autoencoder_anomaly_f1",
    "classifier_anomaly_f1",
]
