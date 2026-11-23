"""Evaluation framework and metrics suite for every detector (Day 13).

One reusable harness judges the four Day 8-11 detectors and the Day 12 ensemble the same way:

* :class:`AnomalyDetectionEvaluator` -- turns ``(y_true, scores)`` into a :class:`MetricsReport`
  (17 headline metrics, per-anomaly-type breakdown, event-level detection and point-adjusted F1),
  sweeps the decision threshold (:class:`ThresholdSweep`) and breaks the results down per entity.
* :class:`ModelLeaderboard` -- collects reports, ranks them by weighted F1, answers "which model
  under these constraints?" and exports CSV + HTML.
* :class:`EvaluationReporter` -- renders one self-contained HTML report per model (Jinja2 + plotly:
  metrics, confusion matrix, ROC / PR curves, threshold sweep, per-entity heatmap). The page has a
  print stylesheet, so "Print to PDF" in any browser gives the PDF version.

Conventions
-----------
* **Scores** are ``0..1`` and ``score >= threshold`` means "flagged"; the default threshold is the
  decision boundary every detector shares (``BOUNDARY = 0.5``, see ``ensemble_detector``). ``NaN``
  scores ("no opinion", for example a detector that is still warming up) count as 0 and are
  reported in ``n_missing_scores``.
* **Events.** Consecutive anomalous rows of one entity (and one anomaly type) form one *anomaly
  event*; rows are ordered by timestamp inside each entity when timestamps are given, otherwise in
  the order supplied. ``max_gap_s`` additionally splits an event across a time gap.
* **MTTD** is the mean number of rows (timesteps) between an event's first row and the first
  flagged row inside it, over detected events only; ``mttd_seconds`` is the same in event time.
  Missed events are not in the mean -- read it next to ``event_recall``.
* **Point-adjusted (PA) F1** -- a hit anywhere in an event counts the whole event as detected. It is
  the usual time-series-anomaly-detection metric, but it is generous: a long event with one lucky
  flag scores perfect recall. It is therefore always reported *next to* the plain F1 and the
  event recall, never instead of them.
* **Per-type precision / recall / F1** are one-vs-normal when only a binary decision exists: the
  recall of type *t* is the share of *t* rows flagged and its precision uses all false alarms on
  normal rows, so a type's precision answers "how clean is this detector when it is judged on
  this type". Pass ``pred_types`` to :meth:`AnomalyDetectionEvaluator.compute_metrics` to score a
  typed prediction (the type counts as correct only when it matches) instead.
* **Macro / weighted** averages run over the anomaly types present in the ground truth; weighted
  uses the number of rows per type. Without type labels every anomaly is one ``"anomaly"`` class,
  so macro = weighted = binary F1.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.models.ensemble_detector import BOUNDARY
from src.utils.logger import get_logger

if TYPE_CHECKING:
    import plotly.graph_objects as go

_log = get_logger(__name__)

NORMAL_LABEL: Final = "normal"
ANOMALY_LABEL: Final = "anomaly"
DEFAULT_REPORT_DIR: Final = Path("reports")
#: Okabe-Ito colour-blind-safe palette used by every figure and the HTML pages.
PALETTE: Final[dict[str, str]] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#6B7280",
}
#: Columns where a smaller value is better (the leaderboard highlights the minimum).
LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {"fpr", "fnr", "mttd", "mttd_s", "latency_ms"}
)  # fmt: skip


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _ratio(numerator: float, denominator: float, undefined: float = 0.0) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else undefined


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _is_nan(value: Any) -> bool:
    return isinstance(value, float | np.floating) and math.isnan(float(value))


def _clean(value: Any) -> Any:
    """Make a value JSON-safe (``NaN`` -> ``None``, numpy scalars -> Python)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    """Jinja filter: fixed-point number, an en dash for missing values."""
    if value is None or _is_nan(value):
        return "–"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | np.integer):
        return f"{int(value):,}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# Results containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TypeMetrics:
    """Quality of one anomaly type (see the module docstring for the definitions).

    Attributes:
        type: Anomaly type (``point`` / ``contextual`` / ``collective`` / ``anomaly``).
        support: Ground-truth rows of this type.
        tp: Rows of this type that were flagged (and typed correctly when types were predicted).
        precision: ``tp / (tp + false alarms)``.
        recall: ``tp / support``.
        f1: Harmonic mean of the two.
        events: Anomaly events of this type.
        events_detected: Events with at least one flagged row.
        mttd: Mean timesteps to detect over the detected events of this type (``NaN`` if none).
    """

    type: str
    support: int
    tp: int
    precision: float
    recall: float
    f1: float
    events: int
    events_detected: int
    mttd: float


@dataclass(frozen=True)
class MetricsReport:
    """Every metric of one scorer at one threshold.

    Attributes:
        threshold: Decision threshold the confusion counts belong to.
        n_samples: Rows evaluated.
        n_anomalies: Ground-truth anomalous rows.
        n_flagged: Rows with ``score >= threshold``.
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives.
        precision: ``tp / (tp + fp)`` (0 if nothing was flagged).
        recall: ``tp / (tp + fn)``.
        f1: Binary F1 of the anomaly class.
        specificity: ``tn / (tn + fp)``.
        false_positive_rate: ``fp / (fp + tn)``.
        false_negative_rate: ``fn / (fn + tp)``.
        accuracy: Share of correct decisions (misleading at 1-3% prevalence; kept for completeness).
        balanced_accuracy: Mean of recall and specificity.
        mcc: Matthews correlation coefficient.
        roc_auc: Area under the ROC curve (``NaN`` with a single class).
        average_precision: Area under the precision-recall curve (AUC-PR).
        macro_precision: Unweighted mean over anomaly types.
        macro_recall: Unweighted mean over anomaly types.
        macro_f1: Unweighted mean over anomaly types.
        weighted_precision: Row-weighted mean over anomaly types.
        weighted_recall: Row-weighted mean over anomaly types.
        weighted_f1: Row-weighted mean over anomaly types (the leaderboard's ranking metric).
        n_events: Anomaly events in the ground truth.
        n_events_detected: Events with at least one flagged row.
        event_recall: ``n_events_detected / n_events``.
        mttd: Mean time to detect in timesteps (detected events only).
        mttd_seconds: The same in event time (``NaN`` without timestamps).
        pa_precision: Point-adjusted precision.
        pa_recall: Point-adjusted recall.
        pa_f1: Point-adjusted F1.
        per_type: :class:`TypeMetrics` per anomaly type.
        n_missing_scores: ``NaN`` scores that were treated as 0.
    """

    threshold: float
    n_samples: int
    n_anomalies: int
    n_flagged: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    balanced_accuracy: float
    mcc: float
    roc_auc: float
    average_precision: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    n_events: int
    n_events_detected: int
    event_recall: float
    mttd: float
    mttd_seconds: float
    pa_precision: float
    pa_recall: float
    pa_f1: float
    per_type: dict[str, TypeMetrics] = field(default_factory=dict)
    n_missing_scores: int = 0

    @property
    def auc_roc(self) -> float:
        """Alias of :attr:`roc_auc`."""
        return self.roc_auc

    @property
    def auc_pr(self) -> float:
        """Alias of :attr:`average_precision`."""
        return self.average_precision

    def scalars(self) -> dict[str, float]:
        """Return the headline metrics as a flat ``name -> float`` mapping."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "roc_auc": self.roc_auc,
            "auc_pr": self.average_precision,
            "fpr": self.false_positive_rate,
            "fnr": self.false_negative_rate,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "mttd": self.mttd,
            "event_recall": self.event_recall,
            "pa_precision": self.pa_precision,
            "pa_recall": self.pa_recall,
            "pa_f1": self.pa_f1,
        }

    def per_type_frame(self) -> pd.DataFrame:
        """Return :attr:`per_type` as a frame indexed by anomaly type."""
        columns = [f for f in TypeMetrics.__dataclass_fields__ if f != "type"]
        if not self.per_type:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            [asdict(m) for m in self.per_type.values()], index=list(self.per_type)
        ).drop(columns="type")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe nested dict (``NaN`` becomes ``None``)."""
        data = asdict(self)
        data["per_type"] = {k: asdict(v) for k, v in self.per_type.items()}
        cleaned: dict[str, Any] = _clean(data)
        return cleaned


@dataclass(frozen=True)
class ThresholdSweep:
    """Metrics over a grid of thresholds and the best operating point.

    Attributes:
        thresholds: Ascending threshold grid.
        precision: Precision at each threshold.
        recall: Recall at each threshold.
        f1: F1 at each threshold.
        false_positive_rate: FPR at each threshold.
        pa_precision: Point-adjusted precision.
        pa_recall: Point-adjusted recall.
        pa_f1: Point-adjusted F1.
        optimize: The curve the optimum was picked from (``f1`` or ``pa_f1``).
        optimal_threshold: Threshold with the highest value of that curve; among exact ties the
            highest threshold wins (``NaN`` when no threshold detects anything).
        optimal_f1: The curve's value there.
    """

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    false_positive_rate: np.ndarray
    pa_precision: np.ndarray
    pa_recall: np.ndarray
    pa_f1: np.ndarray
    optimize: str
    optimal_threshold: float
    optimal_f1: float

    def to_frame(self) -> pd.DataFrame:
        """Return the sweep as a frame with one row per threshold."""
        return pd.DataFrame(
            {
                "threshold": self.thresholds,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "false_positive_rate": self.false_positive_rate,
                "pa_precision": self.pa_precision,
                "pa_recall": self.pa_recall,
                "pa_f1": self.pa_f1,
            }
        )

    def at(self, threshold: float) -> dict[str, float]:
        """Return the row of the grid point nearest to ``threshold``."""
        row = self.to_frame().iloc[int(np.argmin(np.abs(self.thresholds - threshold)))]
        return {str(k): float(v) for k, v in row.items()}


@dataclass(frozen=True)
class DetectionResults:
    """One scorer's output on a labelled evaluation set, row-aligned.

    Attributes:
        y_true: Ground truth, ``True`` for an anomalous row.
        y_scores: Anomaly scores (``NaN`` = no opinion).
        entity_ids: Sensor / vehicle / SKU id per row.
        timestamps: Event time per row (UTC epoch seconds); orders the rows of an entity.
        anomaly_types: Ground-truth type per row (``None`` for normal rows).
        pred_types: Predicted type per row, when the scorer predicts one.
    """

    y_true: np.ndarray
    y_scores: np.ndarray
    entity_ids: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    anomaly_types: np.ndarray | None = None
    pred_types: np.ndarray | None = None

    def __len__(self) -> int:
        return int(np.asarray(self.y_true).shape[0])

    @classmethod
    def from_member_table(
        cls, table: Any, scores: np.ndarray, pred_types: np.ndarray | None = None
    ) -> DetectionResults:
        """Join a ``MemberScoreTable`` (Day 12) with one score vector.

        Args:
            table: Provides ``y``, ``entity_ids``, ``timestamps`` and ``anomaly_type``.
            scores: One score per row of the table (a member's column or the ensemble score).
            pred_types: Optional predicted type per row.
        """
        return cls(
            y_true=np.asarray(table.y, dtype=bool),
            y_scores=np.asarray(scores, dtype=np.float64),
            entity_ids=np.asarray(table.entity_ids),
            timestamps=np.asarray(table.timestamps, dtype=np.float64),
            anomaly_types=np.asarray(table.anomaly_type, dtype=object),
            pred_types=pred_types,
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        y_true: str = "y_true",
        y_scores: str = "score",
        entity_id: str | None = "entity_id",
        timestamp: str | None = "timestamp",
        anomaly_type: str | None = "anomaly_type",
        pred_type: str | None = None,
    ) -> DetectionResults:
        """Build results from a frame; optional columns that are absent are ignored."""

        def column(name: str | None) -> np.ndarray | None:
            return None if name is None or name not in frame else frame[name].to_numpy()

        timestamps = column(timestamp)
        return cls(
            y_true=frame[y_true].to_numpy(dtype=bool),
            y_scores=frame[y_scores].to_numpy(dtype=np.float64),
            entity_ids=column(entity_id),
            timestamps=None if timestamps is None else timestamps.astype(np.float64),
            anomaly_types=column(anomaly_type),
            pred_types=column(pred_type),
        )


# --------------------------------------------------------------------------- #
# Input preparation: ordering, events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Prepared:
    """Validated, ordered inputs plus the anomaly events found in them (internal)."""

    y: np.ndarray
    score: np.ndarray
    entity: np.ndarray
    timestamps: np.ndarray | None
    types: np.ndarray
    pred: np.ndarray | None
    event_id: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    event_type: np.ndarray
    n_missing: int


def _type_labels(values: Any, flags: np.ndarray, *, default: str) -> np.ndarray:
    """Normalise type labels: valid string where ``flags``, else the normal label."""
    given = np.asarray(values, dtype=object)
    out = np.empty(len(flags), dtype=object)
    for i, (label, flag) in enumerate(zip(given, flags, strict=True)):
        usable = (
            label is not None
            and not (isinstance(label, float) and math.isnan(label))
            and str(label) not in {"", "None", "nan", NORMAL_LABEL}
        )
        out[i] = (str(label) if usable else default) if flag else NORMAL_LABEL
    return out


def _prepare(
    y_true: Any,
    y_scores: Any,
    *,
    entity_ids: Any = None,
    timestamps: Any = None,
    anomaly_types: Any = None,
    pred_types: Any = None,
    max_gap_s: float | None = None,
) -> _Prepared:
    y = np.asarray(y_true)
    scores = np.asarray(y_scores)
    if y.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_scores must be one-dimensional")
    n = y.shape[0]
    if n == 0:
        raise ValueError("y_true is empty")
    if scores.shape[0] != n:
        raise ValueError(f"y_true has {n} rows but y_scores has {scores.shape[0]}")
    y = y.astype(bool)
    raw = scores.astype(np.float64)
    if np.isinf(raw).any():
        raise ValueError("y_scores contains infinite values")
    missing = np.isnan(raw)
    score = np.where(missing, 0.0, raw)

    def check(name: str, values: Any) -> np.ndarray:
        arr = np.asarray(values)
        if arr.shape[0] != n:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
        return arr

    codes = (
        np.zeros(n, dtype=np.int64)
        if entity_ids is None
        else pd.factorize(check("entity_ids", entity_ids).astype(str), sort=True)[0].astype(
            np.int64
        )
    )
    ts = None if timestamps is None else check("timestamps", timestamps).astype(np.float64)
    types = (
        np.where(y, ANOMALY_LABEL, NORMAL_LABEL).astype(object)
        if anomaly_types is None
        else _type_labels(check("anomaly_types", anomaly_types), y, default=ANOMALY_LABEL)
    )
    pred = (
        None
        if pred_types is None
        else _type_labels(
            check("pred_types", pred_types), np.ones(n, dtype=bool), default=ANOMALY_LABEL
        )
    )

    order = np.lexsort((ts, codes)) if ts is not None else np.argsort(codes, kind="stable")
    y, score, codes, types = y[order], score[order], codes[order], types[order]
    ts = None if ts is None else ts[order]
    pred = None if pred is None else pred[order]

    previous_y = np.concatenate(([False], y[:-1]))
    previous_code = np.concatenate(([-1], codes[:-1]))
    previous_type = np.concatenate((np.array([""], dtype=object), types[:-1]))
    new = y & (~previous_y | (codes != previous_code) | (types != previous_type))
    if ts is not None and max_gap_s is not None:
        new |= y & np.concatenate(([False], np.diff(ts) > max_gap_s))
    starts = np.flatnonzero(new)
    event_id = np.where(y, np.cumsum(new) - 1, -1)
    lengths = np.bincount(event_id[y], minlength=len(starts)) if y.any() else np.zeros(0, int)
    return _Prepared(
        y=y,
        score=score,
        entity=codes,
        timestamps=ts,
        types=types,
        pred=pred,
        event_id=event_id,
        starts=starts,
        ends=starts + lengths,
        event_type=types[starts],
        n_missing=int(missing.sum()),
    )


# --------------------------------------------------------------------------- #
# Metrics at one threshold
# --------------------------------------------------------------------------- #
def _evaluate(p: _Prepared, threshold: float) -> MetricsReport:
    y, s = p.y, p.score
    flagged = s >= threshold
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    root = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    both = bool(y.any() and (~y).any())

    # events: first flagged row inside each, point-adjusted prediction
    n_events = len(p.starts)
    detected = np.zeros(n_events, dtype=bool)
    delay_steps = np.zeros(n_events)
    delay_seconds = np.zeros(n_events)
    hits = np.flatnonzero(flagged & y)
    if n_events and hits.size:
        first = np.searchsorted(hits, p.starts)
        candidate = hits[np.minimum(first, hits.size - 1)]
        detected = (first < hits.size) & (candidate < p.ends)
        delay_steps = (candidate - p.starts).astype(np.float64)
        if p.timestamps is not None:
            delay_seconds = p.timestamps[candidate] - p.timestamps[p.starts]
    pa_pred = flagged.copy()
    if detected.any():
        in_detected = np.zeros(len(y), dtype=bool)
        in_detected[y] = detected[p.event_id[y]]
        pa_pred |= in_detected
    pa_tp = int(np.sum(pa_pred & y))
    pa_precision = _ratio(pa_tp, pa_tp + fp)
    pa_recall = _ratio(pa_tp, int(y.sum()))

    # per anomaly type
    per_type: dict[str, TypeMetrics] = {}
    for label in sorted({str(t) for t in p.types[y]}):
        rows = p.types == label
        if p.pred is None:
            tp_t, fp_t = int(np.sum(flagged & rows)), fp
        else:
            typed_hit = flagged & (p.pred == label)
            tp_t, fp_t = int(np.sum(typed_hit & rows)), int(np.sum(typed_hit & ~rows))
        precision_t = _ratio(tp_t, tp_t + fp_t)
        recall_t = _ratio(tp_t, int(rows.sum()))
        events = p.event_type == label
        found = events & detected
        per_type[label] = TypeMetrics(
            type=label,
            support=int(rows.sum()),
            tp=tp_t,
            precision=precision_t,
            recall=recall_t,
            f1=_f1(precision_t, recall_t),
            events=int(events.sum()),
            events_detected=int(found.sum()),
            mttd=float(delay_steps[found].mean()) if found.any() else math.nan,
        )

    def average(attribute: str, *, weighted: bool) -> float:
        if not per_type:
            return math.nan
        values = np.array([getattr(m, attribute) for m in per_type.values()], dtype=np.float64)
        weights = np.array([m.support for m in per_type.values()], dtype=np.float64)
        return float(np.average(values, weights=weights) if weighted else values.mean())

    return MetricsReport(
        threshold=float(threshold),
        n_samples=len(y),
        n_anomalies=int(y.sum()),
        n_flagged=int(flagged.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        specificity=specificity,
        false_positive_rate=_ratio(fp, fp + tn),
        false_negative_rate=_ratio(fn, fn + tp),
        accuracy=_ratio(tp + tn, len(y)),
        balanced_accuracy=(recall + specificity) / 2.0,
        mcc=float((tp * tn - fp * fn) / root) if root > 0 else 0.0,
        roc_auc=float(roc_auc_score(y, s)) if both else math.nan,
        average_precision=float(average_precision_score(y, s)) if y.any() else math.nan,
        macro_precision=average("precision", weighted=False),
        macro_recall=average("recall", weighted=False),
        macro_f1=average("f1", weighted=False),
        weighted_precision=average("precision", weighted=True),
        weighted_recall=average("recall", weighted=True),
        weighted_f1=average("f1", weighted=True),
        n_events=n_events,
        n_events_detected=int(detected.sum()),
        event_recall=_ratio(int(detected.sum()), n_events, undefined=math.nan),
        mttd=float(delay_steps[detected].mean()) if detected.any() else math.nan,
        mttd_seconds=(
            float(delay_seconds[detected].mean())
            if detected.any() and p.timestamps is not None
            else math.nan
        ),
        pa_precision=pa_precision,
        pa_recall=pa_recall,
        pa_f1=_f1(pa_precision, pa_recall),
        per_type=per_type,
        n_missing_scores=p.n_missing,
    )


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class AnomalyDetectionEvaluator:
    """Computes the metrics of a scorer, sweeps its threshold and splits results per entity.

    Args:
        threshold: Default decision threshold (``score >= threshold`` is flagged).
        max_gap_s: Split an anomaly event where consecutive rows of an entity are further apart
            than this many seconds (needs timestamps); ``None`` never splits on time.

    Example::

        evaluator = AnomalyDetectionEvaluator()
        report = evaluator.compute_metrics(y, scores, entity_ids=sensors, timestamps=ts,
                                           anomaly_types=kinds)
        sweep = evaluator.sweep_thresholds(y, scores, n_thresholds=100)
    """

    def __init__(self, threshold: float = BOUNDARY, *, max_gap_s: float | None = None) -> None:
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if max_gap_s is not None and max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive")
        self.threshold = float(threshold)
        self.max_gap_s = max_gap_s

    def _prepare(self, y_true: Any, y_scores: Any, **kwargs: Any) -> _Prepared:
        return _prepare(y_true, y_scores, max_gap_s=self.max_gap_s, **kwargs)

    def compute_metrics(
        self,
        y_true: Any,
        y_scores: Any,
        threshold: float | None = None,
        *,
        entity_ids: Any = None,
        timestamps: Any = None,
        anomaly_types: Any = None,
        pred_types: Any = None,
    ) -> MetricsReport:
        """Compute every metric of one scorer at one threshold.

        Args:
            y_true: Ground truth per row (``True`` = anomalous).
            y_scores: Anomaly score per row (``NaN`` counts as 0).
            threshold: Decision threshold; defaults to the evaluator's.
            entity_ids: Entity per row. Events never span two entities.
            timestamps: Event time per row (UTC epoch seconds); rows are ordered by it.
            anomaly_types: Ground-truth type per row (``None`` for normal rows).
            pred_types: Predicted type per row; enables strictly typed per-type metrics.

        Returns:
            The :class:`MetricsReport`.

        Raises:
            ValueError: Empty or mismatched inputs, infinite scores or a non-finite threshold.
        """
        limit = self.threshold if threshold is None else float(threshold)
        if not math.isfinite(limit):
            raise ValueError("threshold must be finite")
        prepared = self._prepare(
            y_true,
            y_scores,
            entity_ids=entity_ids,
            timestamps=timestamps,
            anomaly_types=anomaly_types,
            pred_types=pred_types,
        )
        return _evaluate(prepared, limit)

    def sweep_thresholds(
        self,
        y_true: Any,
        y_scores: Any,
        n_thresholds: int = 100,
        *,
        entity_ids: Any = None,
        timestamps: Any = None,
        anomaly_types: Any = None,
        optimize: str = "f1",
    ) -> ThresholdSweep:
        """Evaluate ``n_thresholds`` evenly spaced thresholds and pick the best.

        The grid spans ``0..1`` when every score lies in that range, else ``min..max``. All
        thresholds are evaluated in one sorted pass, so 100 thresholds cost about one metrics
        computation.

        Args:
            y_true: Ground truth per row.
            y_scores: Anomaly score per row.
            n_thresholds: Grid size (at least 2).
            entity_ids: Entity per row (needed for point-adjusted F1 across entities).
            timestamps: Event time per row.
            anomaly_types: Ground-truth type per row.
            optimize: ``"f1"`` or ``"pa_f1"`` -- the curve whose maximum is the optimum.

        Raises:
            ValueError: ``n_thresholds < 2``, unknown ``optimize`` or invalid inputs.
        """
        if n_thresholds < 2:
            raise ValueError("n_thresholds must be at least 2")
        if optimize not in {"f1", "pa_f1"}:
            raise ValueError("optimize must be 'f1' or 'pa_f1'")
        p = self._prepare(
            y_true,
            y_scores,
            entity_ids=entity_ids,
            timestamps=timestamps,
            anomaly_types=anomaly_types,
        )
        s, y = p.score, p.y
        low, high = float(s.min()), float(s.max())
        if low >= 0.0 and high <= 1.0:
            low, high = 0.0, 1.0
        thresholds = np.linspace(low, high, n_thresholds)
        positives, negatives = np.sort(s[y]), np.sort(s[~y])
        tp = (len(positives) - np.searchsorted(positives, thresholds, side="left")).astype(float)
        fp = (len(negatives) - np.searchsorted(negatives, thresholds, side="left")).astype(float)

        def safe(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)

        def f1_of(pr: np.ndarray, rc: np.ndarray) -> np.ndarray:
            return safe(2 * pr * rc, pr + rc)

        precision = safe(tp, tp + fp)
        recall = safe(tp, np.full_like(tp, float(len(positives))))
        # point adjustment: an event is detected at t iff its best score is >= t
        pa_tp = np.zeros_like(tp)
        if len(p.starts):
            best = np.full(len(p.starts), -np.inf)
            np.maximum.at(best, p.event_id[y], s[y])
            by_best = np.argsort(best)
            suffix = np.concatenate(
                (np.cumsum((p.ends - p.starts)[by_best][::-1])[::-1], [0])
            ).astype(float)
            pa_tp = suffix[np.searchsorted(best[by_best], thresholds, side="left")]
        pa_precision = safe(pa_tp, pa_tp + fp)
        pa_recall = safe(pa_tp, np.full_like(tp, float(len(positives))))
        curves = {"f1": f1_of(precision, recall), "pa_f1": f1_of(pa_precision, pa_recall)}
        target = curves[optimize]
        peak = float(target.max())
        if peak > 0:
            tied = np.flatnonzero(target >= peak - 1e-12)
            optimal_threshold = float(thresholds[tied[-1]])
        else:
            optimal_threshold = math.nan
        return ThresholdSweep(
            thresholds=thresholds,
            precision=precision,
            recall=recall,
            f1=curves["f1"],
            false_positive_rate=safe(fp, np.full_like(fp, float(len(negatives)))),
            pa_precision=pa_precision,
            pa_recall=pa_recall,
            pa_f1=curves["pa_f1"],
            optimize=optimize,
            optimal_threshold=optimal_threshold,
            optimal_f1=peak,
        )

    def compute_per_entity_metrics(
        self, results: DetectionResults, threshold: float | None = None
    ) -> pd.DataFrame:
        """Break one scorer's results down per sensor / vehicle / SKU.

        Metrics that are undefined for an entity are ``NaN`` rather than 0: recall, F1, FNR and
        point-adjusted values when it had no anomalies, precision when nothing was flagged.

        Args:
            results: The scorer's :class:`DetectionResults` (``entity_ids`` required).
            threshold: Decision threshold; defaults to the evaluator's.

        Returns:
            One row per entity (sorted by id) with ``n``, ``n_anomalies``, ``anomaly_rate``,
            ``n_flagged``, ``precision``, ``recall``, ``f1``, ``fpr``, ``roc_auc``, ``auc_pr``,
            ``pa_f1``, ``mttd``, ``events`` and ``events_detected``.

        Raises:
            ValueError: ``results.entity_ids`` is missing.
        """
        if results.entity_ids is None:
            raise ValueError("per-entity metrics need entity_ids")
        limit = self.threshold if threshold is None else float(threshold)
        entities = np.asarray(results.entity_ids).astype(str)
        codes, names = pd.factorize(entities, sort=True)
        order = np.argsort(codes, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        rows: list[dict[str, Any]] = []
        for index in groups:

            def take(values: Any, rows_: np.ndarray = index) -> Any:
                return None if values is None else np.asarray(values)[rows_]

            prepared = self._prepare(
                np.asarray(results.y_true)[index],
                np.asarray(results.y_scores)[index],
                timestamps=take(results.timestamps),
                anomaly_types=take(results.anomaly_types),
            )
            m = _evaluate(prepared, limit)
            no_positives, nothing_flagged = m.n_anomalies == 0, m.n_flagged == 0
            nan = math.nan
            rows.append(
                {
                    "entity_id": str(names[codes[index[0]]]),
                    "n": m.n_samples,
                    "n_anomalies": m.n_anomalies,
                    "anomaly_rate": m.n_anomalies / m.n_samples,
                    "n_flagged": m.n_flagged,
                    "precision": nan if nothing_flagged else m.precision,
                    "recall": nan if no_positives else m.recall,
                    "f1": nan if no_positives else m.f1,
                    "fpr": m.false_positive_rate if m.n_samples > m.n_anomalies else nan,
                    "roc_auc": m.roc_auc,
                    "auc_pr": m.average_precision,
                    "pa_f1": nan if no_positives else m.pa_f1,
                    "mttd": m.mttd,
                    "events": m.n_events,
                    "events_detected": m.n_events_detected,
                }
            )
        return pd.DataFrame(rows).set_index("entity_id")

    def evaluate(
        self,
        results: DetectionResults,
        *,
        threshold: float | None = None,
        n_thresholds: int = 100,
        optimize: str = "f1",
    ) -> tuple[MetricsReport, ThresholdSweep]:
        """Return the report at ``threshold`` and the threshold sweep of one scorer."""
        common = {
            "entity_ids": results.entity_ids,
            "timestamps": results.timestamps,
            "anomaly_types": results.anomaly_types,
        }
        report = self.compute_metrics(
            results.y_true,
            results.y_scores,
            threshold,
            pred_types=results.pred_types,
            **common,
        )
        sweep = self.sweep_thresholds(
            results.y_true, results.y_scores, n_thresholds, optimize=optimize, **common
        )
        return report, sweep

    def evaluate_scorers(
        self,
        table: Any,
        scores: Mapping[str, np.ndarray],
        *,
        thresholds: Mapping[str, float] | None = None,
        latency_ms: Mapping[str, float] | None = None,
        n_thresholds: int = 100,
        leaderboard: ModelLeaderboard | None = None,
    ) -> ModelLeaderboard:
        """Evaluate several scorers on one labelled table and rank them.

        Args:
            table: A ``MemberScoreTable`` (or anything with ``y``, ``entity_ids``, ``timestamps``
                and ``anomaly_type``) holding the ground truth.
            scores: Model name -> one score per table row.
            thresholds: Per-model decision threshold (default: the evaluator's).
            latency_ms: Per-model inference latency, shown in the leaderboard.
            n_thresholds: Grid size of each model's threshold sweep.
            leaderboard: Add to this leaderboard instead of creating a new one.

        Returns:
            The leaderboard with one registered entry per model.
        """
        board = leaderboard if leaderboard is not None else ModelLeaderboard()
        for name, vector in scores.items():
            results = DetectionResults.from_member_table(table, vector)
            limit = None if thresholds is None else thresholds.get(name)
            report, sweep = self.evaluate(results, threshold=limit, n_thresholds=n_thresholds)
            board.register_model(
                name,
                report,
                sweep=sweep,
                latency_ms=None if latency_ms is None else latency_ms.get(name),
            )
        return board


# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
_COLUMN_LABELS: Final[dict[str, str]] = {
    "model": "Model",
    "weighted_f1": "Weighted F1",
    "macro_f1": "Macro F1",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "pa_f1": "PA F1",
    "auc_pr": "AUC-PR",
    "roc_auc": "AUC-ROC",
    "fpr": "FPR",
    "fnr": "FNR",
    "mttd": "MTTD (steps)",
    "mttd_s": "MTTD (s)",
    "event_recall": "Event recall",
    "threshold": "Threshold",
    "oracle_f1": "Oracle F1",
    "oracle_threshold": "Oracle thr.",
    "latency_ms": "Latency (ms)",
}

_LEADERBOARD_TEMPLATE: Final = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
 body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1200px;
      padding:0 1rem;color:#111827}
 h1{font-size:1.5rem;margin-bottom:.25rem} p.sub{color:#6b7280;margin-top:0}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{padding:.45rem .7rem;text-align:right;border-bottom:1px solid #e5e7eb}
 th{background:#f3f4f6;font-weight:600;position:sticky;top:0}
 th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
 td.best{background:#d9f0e6;font-weight:700} tr:first-child td:nth-child(2){font-weight:700}
 .note{color:#6b7280;font-size:.85rem;margin-top:1rem}
</style></head><body>
<h1>{{ title }}</h1>
<p class="sub">Ranked by {{ rank_label }}. Generated {{ generated }}.</p>
<table><thead><tr><th>#</th>{% for c in columns %}<th>{{ labels[c] }}</th>{% endfor %}</tr></thead>
<tbody>{% for row in rows %}<tr><td>{{ loop.index }}</td>
{% for cell in row %}<td{% if cell.best %} class="best"{% endif %}>{{ cell.text }}</td>{% endfor %}
</tr>{% endfor %}</tbody></table>
<p class="note">Green = best value in the column (lowest for FPR, FNR, MTTD and latency).
PA F1 is point-adjusted: a hit anywhere in an anomaly event counts the whole event, so read it with
Event recall. Oracle F1 is the best F1 any threshold reaches on this very set -- an upper bound, not
a deployable number.</p></body></html>
"""


@dataclass
class _Entry:
    report: MetricsReport
    sweep: ThresholdSweep | None
    latency_ms: float | None
    notes: str


class ModelLeaderboard:
    """Collects :class:`MetricsReport` objects and ranks the models.

    Args:
        rank_by: Column to rank by, descending (default ``weighted_f1``). Ties are broken by AUC-PR
            and then F1; models without a value for it rank last.

    Example::

        board = ModelLeaderboard()
        board.register_model("ensemble", report, sweep=sweep, latency_ms=4.2)
        print(board.generate_comparison_table())
        board.export_leaderboard("reports/leaderboard")
    """

    def __init__(self, rank_by: str = "weighted_f1") -> None:
        self.rank_by = rank_by
        self._entries: dict[str, _Entry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def models(self) -> list[str]:
        """Registered model names in registration order."""
        return list(self._entries)

    def register_model(
        self,
        model_name: str,
        metrics_report: MetricsReport,
        *,
        sweep: ThresholdSweep | None = None,
        latency_ms: float | None = None,
        notes: str = "",
    ) -> None:
        """Add (or replace) a model.

        Args:
            model_name: Unique name shown in the table.
            metrics_report: The model's report at its operating threshold.
            sweep: Its threshold sweep; adds the oracle F1 and threshold columns.
            latency_ms: Mean inference latency per event, if measured.
            notes: Free text (kept with the entry, not in the table).

        Raises:
            ValueError: Empty name.
        """
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self._entries[model_name] = _Entry(metrics_report, sweep, latency_ms, notes)

    def report(self, model_name: str) -> MetricsReport:
        """Return the registered report of a model (``KeyError`` if unknown)."""
        return self._entries[model_name].report

    def sweep(self, model_name: str) -> ThresholdSweep | None:
        """Return the registered threshold sweep of a model, if any."""
        return self._entries[model_name].sweep

    def generate_comparison_table(self) -> pd.DataFrame:
        """Return all models ranked by ``rank_by`` (rank 1 = best), one row per model."""
        if not self._entries:
            return pd.DataFrame(columns=[c for c in _COLUMN_LABELS if c != "model"])
        rows = []
        for name, entry in self._entries.items():
            r = entry.report
            row: dict[str, Any] = {
                "model": name,
                "weighted_f1": r.weighted_f1,
                "macro_f1": r.macro_f1,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "pa_f1": r.pa_f1,
                "auc_pr": r.average_precision,
                "roc_auc": r.roc_auc,
                "fpr": r.false_positive_rate,
                "fnr": r.false_negative_rate,
                "mttd": r.mttd,
                "mttd_s": r.mttd_seconds,
                "event_recall": r.event_recall,
                "threshold": r.threshold,
            }
            if entry.sweep is not None:
                row["oracle_f1"] = entry.sweep.optimal_f1
                row["oracle_threshold"] = entry.sweep.optimal_threshold
            if entry.latency_ms is not None:
                row["latency_ms"] = entry.latency_ms
            rows.append(row)
        frame = pd.DataFrame(rows)
        keys = [k for k in (self.rank_by, "auc_pr", "f1") if k in frame.columns]
        frame = frame.sort_values(keys, ascending=False, na_position="last", kind="stable")
        frame = frame.reset_index(drop=True)
        frame.index = pd.RangeIndex(1, len(frame) + 1, name="rank")
        return frame

    def per_type_table(self, metric: str = "recall") -> pd.DataFrame:
        """Return ``metric`` (``precision`` / ``recall`` / ``f1``) per model x anomaly type.

        Rows follow the ranking of :meth:`generate_comparison_table`; this is the table that shows
        which detector family wins on which anomaly type.
        """
        if metric not in {"precision", "recall", "f1"}:
            raise ValueError("metric must be 'precision', 'recall' or 'f1'")
        order = self.generate_comparison_table()["model"].tolist()
        data = {
            name: {t: getattr(m, metric) for t, m in self._entries[name].report.per_type.items()}
            for name in order
        }
        return pd.DataFrame.from_dict(data, orient="index").reindex(order)

    def best_model(self) -> str:
        """Return the name of the top-ranked model (``ValueError`` if none is registered)."""
        if not self._entries:
            raise ValueError("no models registered")
        return str(self.generate_comparison_table()["model"].iloc[0])

    def recommend(
        self,
        *,
        max_false_positive_rate: float | None = None,
        min_recall: float | None = None,
        min_event_recall: float | None = None,
        max_latency_ms: float | None = None,
    ) -> str | None:
        """Return the top-ranked model that satisfies every given constraint, or ``None``.

        Models without a latency measurement fail a ``max_latency_ms`` constraint.
        """
        table = self.generate_comparison_table()
        for _, row in table.iterrows():
            if max_false_positive_rate is not None and not row["fpr"] <= max_false_positive_rate:
                continue
            if min_recall is not None and not row["recall"] >= min_recall:
                continue
            if min_event_recall is not None and not row["event_recall"] >= min_event_recall:
                continue
            if max_latency_ms is not None and not (
                row.get("latency_ms", math.nan) <= max_latency_ms
            ):
                continue
            return str(row["model"])
        return None

    def to_html(self, title: str = "Model leaderboard") -> str:
        """Render the ranked table as a standalone HTML page (best value per column in green)."""
        table = self.generate_comparison_table()
        columns = [c for c in table.columns if c != "model"]
        columns = ["model", *columns]
        best: dict[str, float] = {}
        for column in columns[1:]:
            values = table[column].dropna()
            if values.empty or column == "threshold" or column == "oracle_threshold":
                continue
            best[column] = float(values.min() if column in LOWER_IS_BETTER else values.max())
        rows = []
        for _, record in table.iterrows():
            cells = []
            for column in columns:
                value = record[column]
                text = (
                    str(value)
                    if column == "model"
                    else _fmt(value, 1 if column in {"mttd", "mttd_s", "latency_ms"} else 3)
                )
                is_best = (
                    column in best
                    and not _is_nan(value)
                    and math.isclose(float(value), best[column])
                )
                cells.append({"text": text, "best": is_best})
            rows.append(cells)
        env = Environment(autoescape=select_autoescape(default=True))
        return env.from_string(_LEADERBOARD_TEMPLATE).render(
            title=title,
            columns=columns,
            labels=_COLUMN_LABELS,
            rows=rows,
            rank_label=_COLUMN_LABELS.get(self.rank_by, self.rank_by),
            generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        )

    def export_leaderboard(
        self, path: str | Path, *, title: str = "Model leaderboard"
    ) -> dict[str, Path]:
        """Write ``<path>.csv`` and ``<path>.html`` (a ``.csv`` / ``.html`` suffix is ignored).

        Returns:
            ``{"csv": Path, "html": Path}``.

        Raises:
            ValueError: No model is registered.
        """
        if not self._entries:
            raise ValueError("no models registered")
        target = Path(path)
        stem = (
            str(target.with_suffix(""))
            if target.suffix.lower() in {".csv", ".html"}
            else str(target)
        )
        csv_path, html_path = Path(f"{stem}.csv"), Path(f"{stem}.html")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.generate_comparison_table().to_csv(csv_path)
        html_path.write_text(self.to_html(title), encoding="utf-8")
        _log.info("leaderboard_exported", csv=str(csv_path), html=str(html_path), models=len(self))
        return {"csv": csv_path, "html": html_path}


# --------------------------------------------------------------------------- #
# Figures (plotly)
# --------------------------------------------------------------------------- #
def _figure(title: str, x_title: str, y_title: str, height: int = 420) -> go.Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "font": {"size": 15}},
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin={"l": 55, "r": 20, "t": 50, "b": 50},
        legend={"orientation": "h", "y": -0.22},
        font={"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    )
    return fig


def roc_figure(y_true: Any, y_scores: Any, name: str = "model") -> go.Figure:
    """ROC curve with the chance diagonal (``NaN`` scores count as 0)."""
    import plotly.graph_objects as go

    y = np.asarray(y_true, dtype=bool)
    s = np.nan_to_num(np.asarray(y_scores, dtype=np.float64), nan=0.0)
    fig = _figure("ROC curve", "False positive rate", "True positive rate")
    if y.any() and (~y).any():
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                       line={"color": PALETTE["blue"], "width": 2.5})
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                   line={"color": PALETTE["grey"], "dash": "dot"})
    )  # fmt: skip
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def pr_figure(y_true: Any, y_scores: Any, name: str = "model") -> go.Figure:
    """Precision-recall curve with the no-skill line at the anomaly prevalence."""
    import plotly.graph_objects as go

    y = np.asarray(y_true, dtype=bool)
    s = np.nan_to_num(np.asarray(y_scores, dtype=np.float64), nan=0.0)
    fig = _figure("Precision-recall curve", "Recall", "Precision")
    if y.any():
        precision, recall, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AP {ap:.3f})",
                       line={"color": PALETTE["vermillion"], "width": 2.5})
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[y.mean()] * 2, mode="lines", name="no skill",
                       line={"color": PALETTE["grey"], "dash": "dot"})
        )  # fmt: skip
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def sweep_figure(sweep: ThresholdSweep, operating_threshold: float | None = None) -> go.Figure:
    """Precision / recall / F1 / PA-F1 against the threshold, optimum and operating point marked."""
    import plotly.graph_objects as go

    fig = _figure("Threshold sweep", "Threshold", "Metric value")
    lines = (
        ("Precision", sweep.precision, PALETTE["blue"], "solid"),
        ("Recall", sweep.recall, PALETTE["orange"], "solid"),
        ("F1", sweep.f1, PALETTE["green"], "solid"),
        ("PA F1", sweep.pa_f1, PALETTE["purple"], "dash"),
    )
    for label, values, color, dash in lines:
        fig.add_trace(
            go.Scatter(x=sweep.thresholds, y=values, mode="lines", name=label,
                       line={"color": color, "dash": dash, "width": 2.2})
        )  # fmt: skip
    if math.isfinite(sweep.optimal_threshold):
        fig.add_vline(
            x=sweep.optimal_threshold,
            line={"color": PALETTE["vermillion"], "dash": "dot"},
            annotation_text=f"best {sweep.optimize.upper().replace('_', ' ')} "
            f"{sweep.optimal_f1:.3f} @ {sweep.optimal_threshold:.2f}",
            annotation_position="top left",
        )
    if operating_threshold is not None:
        fig.add_vline(
            x=operating_threshold,
            line={"color": PALETTE["grey"], "dash": "dash"},
            annotation_text=f"operating {operating_threshold:.2f}",
            annotation_position="bottom right",
        )
    fig.update_yaxes(range=[0, 1.02])
    return fig


def entity_heatmap_figure(
    per_entity: pd.DataFrame,
    metrics: Sequence[str] = ("precision", "recall", "f1", "pa_f1"),
    *,
    max_entities: int = 40,
) -> go.Figure:
    """Entity x metric heatmap, the weakest ``max_entities`` entities first (by mean of ``metrics``).

    Cells that are undefined for an entity (for example recall without anomalies) are left blank.
    """
    import plotly.graph_objects as go

    frame = per_entity.loc[:, list(metrics)]
    ranking = frame.mean(axis=1, skipna=True).fillna(1.0).sort_values()
    frame = frame.loc[ranking.index[:max_entities]]
    fig = _figure(
        f"Per-entity quality (weakest {len(frame)} of {len(per_entity)})",
        "",
        "",
        height=max(320, 22 * len(frame) + 130),
    )
    fig.add_trace(
        go.Heatmap(
            z=frame.to_numpy(dtype=float),
            x=[_COLUMN_LABELS.get(m, m) for m in frame.columns],
            y=list(frame.index),
            zmin=0.0,
            zmax=1.0,
            colorscale="Blues",
            texttemplate="%{z:.2f}",
            hoverongaps=False,
            colorbar={"title": "value"},
        )
    )
    fig.update_yaxes(autorange="reversed", type="category")
    return fig


def leaderboard_figure(
    table: pd.DataFrame, metrics: Sequence[str] = ("f1", "pa_f1", "auc_pr")
) -> go.Figure:
    """Draw grouped bars of the chosen leaderboard columns, one group per model."""
    import plotly.graph_objects as go

    colors = (PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"])
    fig = _figure("Model comparison", "", "Score")
    for color, metric in zip(colors, metrics, strict=False):
        fig.add_trace(
            go.Bar(x=table["model"], y=table[metric], name=_COLUMN_LABELS.get(metric, metric),
                   marker_color=color, text=[_fmt(v, 2) for v in table[metric]], textposition="outside")
        )  # fmt: skip
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.1])
    return fig


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
_METRIC_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("precision", "Precision", "Share of flagged rows that are real anomalies."),
    ("recall", "Recall", "Share of anomalous rows that were flagged."),
    ("f1", "F1", "Harmonic mean of precision and recall."),
    ("macro_f1", "Macro F1", "Unweighted mean of the per-type F1."),
    ("weighted_f1", "Weighted F1", "Per-type F1 weighted by rows per type (leaderboard rank)."),
    ("roc_auc", "AUC-ROC", "Ranking quality over all thresholds."),
    ("auc_pr", "AUC-PR", "Average precision; the honest curve at low prevalence."),
    ("fpr", "False positive rate", "False alarms per normal row."),
    ("fnr", "False negative rate", "Missed anomalies per anomalous row."),
    ("specificity", "Specificity", "Share of normal rows left alone."),
    ("accuracy", "Accuracy", "Misleading at 1-3% prevalence; shown for completeness."),
    ("balanced_accuracy", "Balanced accuracy", "Mean of recall and specificity."),
    ("mcc", "MCC", "Matthews correlation; robust to imbalance."),
    ("mttd", "MTTD (timesteps)", "Mean rows from event start to first flag, detected events only."),
    ("event_recall", "Event recall", "Share of anomaly events with at least one flag."),
    ("pa_precision", "PA precision", "Point-adjusted precision."),
    ("pa_recall", "PA recall", "Point-adjusted recall (a hit anywhere counts the whole event)."),
    ("pa_f1", "PA F1", "Point-adjusted F1; generous on long events, read with event recall."),
)

_REPORT_TEMPLATE: Final = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ model_name }} - evaluation report</title>
{{ plotly_script|safe }}
<style>
 :root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb;--accent:#0072B2}
 *{box-sizing:border-box}
 body{font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:var(--ink);background:#fff}
 main{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 4rem}
 h1{font-size:1.65rem;margin:0 0 .25rem} h2{font-size:1.15rem;margin:2.2rem 0 .7rem;
   border-bottom:2px solid var(--line);padding-bottom:.3rem}
 .sub{color:var(--muted);margin:0}
 .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.75rem;margin-top:1.25rem}
 .kpi{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
   border-radius:8px;padding:.7rem .9rem}
 .kpi b{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums}
 .kpi span{color:var(--muted);font-size:.8rem}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{padding:.4rem .65rem;border-bottom:1px solid var(--line);text-align:right}
 th{background:var(--card);font-weight:600}
 th:first-child,td:first-child{text-align:left}
 td.desc{text-align:left;color:var(--muted);font-size:.85rem}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:1rem}
 .plot{border:1px solid var(--line);border-radius:8px;padding:.25rem;min-width:0}
 .note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:.7rem .9rem;font-size:.9rem}
 .cm td,.cm th{text-align:center}
 @media print{ main{padding:0} .plot{break-inside:avoid} h2{break-after:avoid} .grid{display:block}
   .kpi{break-inside:avoid} }
</style></head><body><main>
<h1>{{ model_name }}</h1>
<p class="sub">Evaluation report &middot; generated {{ generated }} &middot; threshold
 {{ r.threshold|fmt }} &middot; {{ r.n_samples|fmt }} rows, {{ r.n_anomalies|fmt }} anomalous
 ({{ prevalence }}) in {{ r.n_events|fmt }} events{% if r.n_missing_scores %} &middot;
 {{ r.n_missing_scores|fmt }} missing scores counted as 0{% endif %}</p>

<div class="kpis">
{% for label, value in kpis %}<div class="kpi"><b>{{ value }}</b><span>{{ label }}</span></div>
{% endfor %}</div>

<h2>Metrics</h2>
<table><thead><tr><th>Metric</th><th>Value</th><th style="text-align:left">Meaning</th></tr></thead><tbody>
{% for label, value, desc in metric_rows %}<tr><td>{{ label }}</td><td>{{ value }}</td><td class="desc">{{ desc }}</td></tr>
{% endfor %}</tbody></table>
<p class="note" style="margin-top:1rem"><b>Reading the point-adjusted numbers.</b> A hit anywhere in an
anomaly event counts the whole event as detected, so long events with one lucky flag give a perfect PA
recall. Compare PA F1 with F1 and event recall before drawing conclusions.</p>

<h2>Confusion matrix</h2>
<table class="cm" style="max-width:460px"><thead><tr><th></th><th>Flagged</th><th>Not flagged</th></tr></thead><tbody>
<tr><th style="text-align:left">Anomalous</th><td>{{ r.tp|fmt }} (TP)</td><td>{{ r.fn|fmt }} (FN)</td></tr>
<tr><th style="text-align:left">Normal</th><td>{{ r.fp|fmt }} (FP)</td><td>{{ r.tn|fmt }} (TN)</td></tr>
</tbody></table>

{% if per_type %}<h2>By anomaly type</h2>
<table><thead><tr><th>Type</th><th>Rows</th><th>Precision</th><th>Recall</th><th>F1</th>
<th>Events</th><th>Detected</th><th>MTTD</th></tr></thead><tbody>
{% for t in per_type %}<tr><td>{{ t.type }}</td><td>{{ t.support|fmt }}</td><td>{{ t.precision|fmt }}</td>
<td>{{ t.recall|fmt }}</td><td>{{ t.f1|fmt }}</td><td>{{ t.events|fmt }}</td>
<td>{{ t.events_detected|fmt }}</td><td>{{ t.mttd|fmt(1) }}</td></tr>
{% endfor %}</tbody></table>{% endif %}

{% if plots %}<h2>Curves and breakdowns</h2><div class="grid">
{% for title, html in plots %}<div class="plot">{{ html|safe }}</div>
{% endfor %}</div>{% endif %}
</main></body></html>
"""


class EvaluationReporter:
    """Renders one self-contained HTML evaluation report per model.

    Args:
        output_dir: Folder for reports (default ``reports/``); created on demand.
        plotlyjs: ``"cdn"`` loads plotly.js from the CDN (small file, needs internet),
            ``"inline"`` embeds it (about 4 MB, works offline).

    Example::

        reporter = EvaluationReporter()
        path = reporter.generate_report("ensemble", report, y_true=y, y_scores=s, sweep=sweep,
                                        per_entity=evaluator.compute_per_entity_metrics(results))
    """

    def __init__(
        self, output_dir: str | Path = DEFAULT_REPORT_DIR, *, plotlyjs: str = "cdn"
    ) -> None:
        if plotlyjs not in {"cdn", "inline"}:
            raise ValueError("plotlyjs must be 'cdn' or 'inline'")
        self.output_dir = Path(output_dir)
        self.plotlyjs = plotlyjs
        self._env = Environment(autoescape=select_autoescape(default=True))
        self._env.filters["fmt"] = _fmt

    def _plotly_script(self) -> str:
        import plotly.offline

        if self.plotlyjs == "inline":
            return f"<script>{plotly.offline.get_plotlyjs()}</script>"
        version = plotly.offline.get_plotlyjs_version()
        return (
            f'<script src="https://cdn.plot.ly/plotly-{version}.min.js" charset="utf-8"></script>'
        )

    def build_plots(
        self,
        model_name: str,
        metrics_report: MetricsReport,
        *,
        y_true: Any = None,
        y_scores: Any = None,
        sweep: ThresholdSweep | None = None,
        per_entity: pd.DataFrame | None = None,
    ) -> dict[str, go.Figure]:
        """Return the default figures for whatever inputs are available."""
        figures: dict[str, go.Figure] = {}
        if y_true is not None and y_scores is not None:
            figures["ROC curve"] = roc_figure(y_true, y_scores, model_name)
            figures["Precision-recall curve"] = pr_figure(y_true, y_scores, model_name)
        if sweep is not None:
            figures["Threshold sweep"] = sweep_figure(sweep, metrics_report.threshold)
        if per_entity is not None and len(per_entity):
            figures["Per-entity heatmap"] = entity_heatmap_figure(per_entity)
        return figures

    def generate_report(
        self,
        model_name: str,
        metrics_report: MetricsReport,
        plots: Mapping[str, go.Figure] | None = None,
        *,
        y_true: Any = None,
        y_scores: Any = None,
        sweep: ThresholdSweep | None = None,
        per_entity: pd.DataFrame | None = None,
        output_dir: str | Path | None = None,
        timestamp: datetime | None = None,
    ) -> Path:
        """Write ``{model_name}_{timestamp}.html`` and return its path.

        The default ROC / PR curves need ``y_true`` and ``y_scores``, the threshold sweep needs
        ``sweep`` and the heatmap needs ``per_entity``; whatever is missing is skipped.

        Args:
            model_name: Name in the title and the file name (unsafe characters become ``_``).
            metrics_report: The model's :class:`MetricsReport`.
            plots: Extra (or overriding, by title) plotly figures.
            y_true: Ground truth for the curves.
            y_scores: Scores for the curves.
            sweep: Threshold sweep for the sweep plot.
            per_entity: Output of ``compute_per_entity_metrics`` for the heatmap.
            output_dir: Overrides the reporter's folder.
            timestamp: Report time (UTC); defaults to now.

        Returns:
            The written file. An existing file is never overwritten (a counter is appended).

        Raises:
            ValueError: ``model_name`` is empty.
        """
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        moment = timestamp if timestamp is not None else datetime.now(UTC)
        figures = self.build_plots(
            model_name,
            metrics_report,
            y_true=y_true,
            y_scores=y_scores,
            sweep=sweep,
            per_entity=per_entity,
        )
        figures.update(plots or {})
        rendered = [
            (
                title,
                fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={"displaylogo": False, "responsive": True},
                ),
            )
            for title, fig in figures.items()
        ]
        r = metrics_report
        scalars = r.scalars()
        html = self._env.from_string(_REPORT_TEMPLATE).render(
            model_name=model_name,
            r=r,
            generated=moment.strftime("%Y-%m-%d %H:%M UTC"),
            prevalence=f"{r.n_anomalies / r.n_samples:.2%}" if r.n_samples else "n/a",
            kpis=[
                ("F1", _fmt(r.f1)),
                ("Weighted F1", _fmt(r.weighted_f1)),
                ("PA F1", _fmt(r.pa_f1)),
                ("AUC-PR", _fmt(r.average_precision)),
                ("AUC-ROC", _fmt(r.roc_auc)),
                ("False positive rate", _fmt(r.false_positive_rate, 4)),
                ("Event recall", _fmt(r.event_recall)),
                ("MTTD (steps)", _fmt(r.mttd, 1)),
            ],
            metric_rows=[
                (label, _fmt(scalars[key], 1 if key == "mttd" else 3), desc)
                for key, label, desc in _METRIC_ROWS
            ],
            per_type=list(r.per_type.values()),
            plots=rendered,
            plotly_script=self._plotly_script() if rendered else "",
        )
        folder = Path(output_dir) if output_dir is not None else self.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_") or "model"
        stem = f"{safe}_{moment.strftime('%Y%m%dT%H%M%SZ')}"
        path, counter = folder / f"{stem}.html", 1
        while path.exists():
            path = folder / f"{stem}-{counter}.html"
            counter += 1
        path.write_text(html, encoding="utf-8")
        _log.info("evaluation_report_written", model=model_name, path=str(path), f1=r.f1)
        return path


__all__ = [
    "ANOMALY_LABEL",
    "NORMAL_LABEL",
    "AnomalyDetectionEvaluator",
    "DetectionResults",
    "EvaluationReporter",
    "MetricsReport",
    "ModelLeaderboard",
    "ThresholdSweep",
    "TypeMetrics",
    "entity_heatmap_figure",
    "leaderboard_figure",
    "pr_figure",
    "roc_figure",
    "sweep_figure",
]
