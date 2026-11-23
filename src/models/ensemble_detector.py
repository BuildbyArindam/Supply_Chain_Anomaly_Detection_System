"""Ensemble detector, severity scoring and alert de-duplication (Day 12).

This module turns four independent detectors into one decision and one alert:

* :class:`EnsembleDetector` -- stacks the Day 8-11 detectors. Level 1 is the four
  :class:`~src.models.statistical_detector.AnomalyScore` objects, level 2 is a
  :class:`MetaLearner` (LightGBM) that learns how far to trust each of them. Until the
  meta-learner is fitted (or when too few detectors are available) it falls back to a weighted
  average.
* :class:`SeverityScorer` -- maps an ensemble score to INFO / WARNING / CRITICAL / EMERGENCY with
  business-impact, time-of-day and asset-value weights, and proposes an action.
* :class:`AnomalyEvent` -- the canonical, JSON-friendly alert record, built by
  :func:`build_anomaly_event`.
* :class:`AlertDeduplicator` -- suppresses repeats of the same alert and escalates persistent ones.

Score convention
----------------
Every detector reports a ``score`` in ``0..1`` where ``0.5`` is its own decision boundary. The
ensemble keeps that convention: ``ensemble_score >= 0.5`` iff ``is_anomaly``. The meta-learner
outputs a probability and the operating threshold that maximises out-of-fold F1 is rarely 0.5, so
the probability is mapped piecewise-linearly (:func:`to_boundary_scale`) so that the threshold lands
on 0.5. The mapping is monotone, so rankings (AUC) are unchanged, and the severity thresholds from
``config.alerts`` (0.3 / 0.6 / 0.85) keep a meaning that does not depend on how the stacker was
trained.

Stacking without leakage
------------------------
The four base detectors are fitted on training data, so their scores on that data are optimistic.
The meta-learner must therefore be fitted on **held-out labelled data the base detectors never
saw** (``config.models.ensemble.meta_learner_holdout``, 20% by default); fitting it on the base
training rows teaches it to over-trust whichever detector memorised them.
:meth:`MetaLearner.fit` reports a 5-fold cross-validation of the stacker on that hold-out slice,
grouped by entity so correlated neighbouring readings never sit in both train and validation folds.

Missing members
---------------
A detector can be absent (not configured), warming up (``details["warming_up"]``) or fail at
runtime. Its score is then ``NaN``. LightGBM handles ``NaN`` natively, the meta-learner is trained
with random member dropout so it has seen missing members, and the weighted-average fallback
re-normalises the weights over the members that did report. Fewer than ``min_members_for_meta``
usable members means the fallback is used for that event.
"""

from __future__ import annotations

import math
import threading
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.models.statistical_detector import AnomalyResult, AnomalyScore, Timestamp, parse_timestamp
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.models.feature_engineering import FeatureVector, OnlineScaler
    from src.models.isolation_forest_detector import (
        IsolationForestDetector,
        IsolationForestExplainer,
    )
    from src.models.lstm_autoencoder import LSTMAutoencoderDetector
    from src.models.lstm_classifier import LSTMClassifierDetector
    from src.models.model_registry import ModelRegistry, ModelVersion
    from src.models.offline_dataset import LabeledFeatureSet
    from src.models.statistical_detector import UnifiedStatisticalDetector
    from src.utils.config import AlertsConfig, EnsembleConfig

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STATISTICAL: Final = "statistical"
ISOLATION_FOREST: Final = "isolation_forest"
LSTM_AUTOENCODER: Final = "lstm_autoencoder"
LSTM_CLASSIFIER: Final = "lstm_classifier"
#: The four level-1 detectors, in the column order of every score matrix.
MEMBERS: Final[tuple[str, ...]] = (STATISTICAL, ISOLATION_FOREST, LSTM_AUTOENCODER, LSTM_CLASSIFIER)
N_MEMBERS: Final = len(MEMBERS)
#: Weighted-average fallback (the Day 12 specification; mirrored in ``config.yaml``).
DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    STATISTICAL: 0.20,
    ISOLATION_FOREST: 0.30,
    LSTM_AUTOENCODER: 0.25,
    LSTM_CLASSIFIER: 0.25,
}
#: Columns of the meta-learner input: 4 scores, 4 confidences, entity anomaly rate, hour of day.
META_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"{m}_score" for m in MEMBERS),
    *(f"{m}_confidence" for m in MEMBERS),
    "historical_anomaly_rate",
    "hour_of_day",
)
#: Where every detector's decision boundary sits on the 0..1 score scale.
BOUNDARY: Final = 0.5
_RATE_FEATURE: Final = "anomaly_rate_7d"
MODEL_TYPE_META: Final = "ensemble_meta_learner"


# --------------------------------------------------------------------------- #
# Score helpers
# --------------------------------------------------------------------------- #
def _aware(moment: datetime) -> datetime:
    """Return ``moment`` as a timezone-aware datetime (naive values are taken as UTC)."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def to_boundary_scale(probability: Any, threshold: float) -> Any:
    """Map a probability so that ``threshold`` lands on 0.5 (monotone, piecewise linear).

    ``[0, threshold)`` is stretched onto ``[0, 0.5)`` and ``[threshold, 1]`` onto ``[0.5, 1]``.

    Args:
        probability: A probability or an array of them.
        threshold: The operating threshold, clipped to ``(0, 1)``.

    Returns:
        The mapped value(s), a ``float`` for a scalar input and an ``ndarray`` otherwise.
    """
    t = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    mapped = np.where(p < t, BOUNDARY * p / t, BOUNDARY + (1.0 - BOUNDARY) * (p - t) / (1.0 - t))
    return float(mapped) if mapped.ndim == 0 else mapped


def is_usable(score: AnomalyScore | None) -> bool:
    """Return whether a member result carries information (present, finite, not warming up)."""
    return (
        score is not None
        and not score.details.get("warming_up", False)
        and math.isfinite(score.score)
    )


def statistical_to_score(result: AnomalyResult | AnomalyScore) -> AnomalyScore:
    """Express a :class:`UnifiedStatisticalDetector` result as an :class:`AnomalyScore`."""
    if isinstance(result, AnomalyScore):
        return result
    details = dict(result.details)
    details["contributing_detectors"] = list(result.contributing_detectors)
    return AnomalyScore(
        entity_id=result.entity_id,
        timestamp=result.timestamp,
        score=result.severity_score,
        is_anomaly=result.is_anomaly,
        anomaly_type=result.anomaly_type,
        confidence=result.confidence,
        detector="statistical_unified",
        raw_value=result.raw_value,
        details=details,
    )


def meta_feature_row(
    scores: Mapping[str, AnomalyScore | None], anomaly_rate: float | None, hour: int | float
) -> np.ndarray:
    """Build the meta-learner input of one event (``NaN`` for a missing member).

    Args:
        scores: Member name to result (see :data:`MEMBERS`); absent / warming members become NaN.
        anomaly_rate: The entity's historical anomaly rate (``anomaly_rate_7d``), or ``None``.
        hour: Hour of day of the event.

    Returns:
        A float vector in the order of :data:`META_FEATURE_NAMES`.
    """
    row = np.full(len(META_FEATURE_NAMES), np.nan)
    for i, name in enumerate(MEMBERS):
        score = scores.get(name)
        if score is not None and is_usable(score):
            row[i] = score.score
            row[N_MEMBERS + i] = score.confidence
    row[2 * N_MEMBERS] = np.nan if anomaly_rate is None else float(anomaly_rate)
    row[2 * N_MEMBERS + 1] = float(hour)
    return row


# --------------------------------------------------------------------------- #
# Meta-learner
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Return the probability threshold with the highest F1 (clipped to ``[0.01, 0.99]``)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    if f1.size == 0:
        return BOUNDARY
    return float(np.clip(thresholds[int(np.argmax(f1))], 0.01, 0.99))


def binary_metrics(
    y_true: np.ndarray, score: np.ndarray, threshold: float = BOUNDARY
) -> dict[str, float]:
    """Precision, recall, F1, false-positive rate and the threshold-free AUCs of one scorer.

    ``NaN`` scores (a detector that had no opinion) count as "not anomalous". The AUCs are ``NaN``
    when only one class is present.
    """
    y = np.asarray(y_true, dtype=bool)
    s = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0)
    flagged = s >= threshold
    tp = float(np.sum(flagged & y))
    fp = float(np.sum(flagged & ~y))
    fn = float(np.sum(~flagged & y))
    tn = float(np.sum(~flagged & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    both = y.any() and (~y).any()
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "average_precision": float(average_precision_score(y, s)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "n_flagged": float(flagged.sum()),
    }


@dataclass(frozen=True)
class CrossValidationReport:
    """Out-of-fold quality of the stacker on its hold-out slice.

    Attributes:
        n_folds: Folds used (may be below the request when anomalies are scarce).
        grouped: Whether folds were grouped by entity.
        threshold: F1-optimal probability threshold on the out-of-fold predictions.
        precision: Out-of-fold precision at ``threshold``.
        recall: Out-of-fold recall at ``threshold``.
        f1: Out-of-fold F1 at ``threshold``.
        average_precision: Out-of-fold average precision (AUC-PR).
        roc_auc: Out-of-fold ROC-AUC.
        fold_f1: F1 of each fold at the pooled ``threshold`` (the spread shows stability).
        fold_average_precision: Average precision of each fold.
        oof_probability: The out-of-fold probabilities, aligned with the training rows.
    """

    n_folds: int
    grouped: bool
    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float
    fold_f1: tuple[float, ...]
    fold_average_precision: tuple[float, ...]
    oof_probability: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (without the per-row probabilities)."""
        return {
            "n_folds": self.n_folds,
            "grouped": self.grouped,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "fold_f1": list(self.fold_f1),
            "fold_average_precision": list(self.fold_average_precision),
        }


class MetaLearner:
    """LightGBM stacker over the level-1 scores (Level 2 of the ensemble).

    Inputs are the :data:`META_FEATURE_NAMES` columns. The four score columns carry a monotone
    ``+1`` constraint by default (a higher detector score can never lower the anomaly
    probability), which regularises hard on the small hold-out slices the stacker is fitted on and
    keeps the model explainable. Rows are optionally duplicated with random members blanked out
    (``augment_missing``) so it behaves sensibly when a detector is warming up or unavailable.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        monotone_scores: bool = True,
        seed: int = 42,
        n_jobs: int = 1,
    ) -> None:
        """Create an unfitted stacker (see the class docs for the meaning of the options)."""
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "monotone_scores": monotone_scores,
            "seed": seed,
            "n_jobs": n_jobs,
        }
        self.feature_names: tuple[str, ...] = META_FEATURE_NAMES
        self.threshold: float = BOUNDARY
        self.report: CrossValidationReport | None = None
        self._model: Any = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._model is not None

    def _new_model(self) -> Any:
        from lightgbm import LGBMClassifier

        p = self.params
        constraints = [1] * N_MEMBERS + [0] * (len(META_FEATURE_NAMES) - N_MEMBERS)
        return LGBMClassifier(
            objective="binary",
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"],
            subsample=p["subsample"],
            subsample_freq=1,
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            monotone_constraints=constraints if p["monotone_scores"] else None,
            random_state=p["seed"],
            n_jobs=p["n_jobs"],
            verbose=-1,
        )

    @staticmethod
    def _check(features: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(META_FEATURE_NAMES):
            raise ValueError(
                f"expected a (n, {len(META_FEATURE_NAMES)}) matrix in the order of "
                f"META_FEATURE_NAMES, got shape {matrix.shape}"
            )
        if y is not None and len(y) != len(matrix):
            raise ValueError("features and labels have different lengths")
        return matrix

    @staticmethod
    def _augment(
        matrix: np.ndarray, y: np.ndarray, fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append copies of random rows with random members blanked (at least one stays)."""
        n_extra = int(round(fraction * len(matrix)))
        if n_extra <= 0:
            return matrix, y
        rng = np.random.default_rng(seed)
        picked = rng.integers(0, len(matrix), n_extra)
        extra = matrix[picked].copy()
        drop = rng.random((n_extra, N_MEMBERS)) < 0.35
        drop[drop.all(axis=1), int(rng.integers(0, N_MEMBERS))] = False
        for member in range(N_MEMBERS):
            extra[drop[:, member], member] = np.nan
            extra[drop[:, member], N_MEMBERS + member] = np.nan
        return np.vstack([matrix, extra]), np.concatenate([y, y[picked]])

    def cross_validate(
        self,
        features: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        n_folds: int = 5,
        augment_missing: float = 0.0,
    ) -> CrossValidationReport:
        """Out-of-fold evaluation of the stacker.

        Args:
            features: ``(n, 10)`` meta-feature matrix.
            y: Boolean ground truth.
            groups: Entity id per row; folds then never split an entity (otherwise the strongly
                autocorrelated neighbouring readings of one sensor leak between folds).
            n_folds: Requested folds; lowered when fewer anomalies (or normals) exist.
            augment_missing: Share of extra training rows with blanked members per fold.

        Returns:
            The :class:`CrossValidationReport`.

        Raises:
            ValueError: Bad shapes, or fewer than two anomalies / normals to split.
        """
        matrix = self._check(features, y)
        labels = np.asarray(y, dtype=bool)
        n_pos, n_neg = int(labels.sum()), int((~labels).sum())
        folds = min(n_folds, n_pos, n_neg)
        if folds < 2:
            raise ValueError(
                f"need at least 2 anomalies and 2 normals to cross-validate ({n_pos}/{n_neg})"
            )
        use_groups = groups is not None and len(np.unique(groups)) >= folds
        if use_groups:
            splitter: Any = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=self.params["seed"]
            )
            splits = splitter.split(matrix, labels, np.asarray(groups))
        else:
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=self.params["seed"]
            )
            splits = splitter.split(matrix, labels)
        oof = np.full(len(matrix), np.nan)
        fold_index = np.zeros(len(matrix), dtype=int)
        for k, (train_idx, valid_idx) in enumerate(splits):
            train_x, train_y = self._augment(
                matrix[train_idx],
                labels[train_idx].astype(int),
                augment_missing,
                self.params["seed"] + k,
            )
            model = self._new_model()
            model.fit(train_x, train_y)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                oof[valid_idx] = model.predict_proba(matrix[valid_idx])[:, 1]
            fold_index[valid_idx] = k
        valid = ~np.isnan(oof)
        threshold = best_f1_threshold(labels[valid], oof[valid])
        pooled = binary_metrics(labels[valid], oof[valid], threshold)
        fold_f1, fold_ap = [], []
        for k in range(folds):
            mask = valid & (fold_index == k)
            if labels[mask].any():
                fold_f1.append(binary_metrics(labels[mask], oof[mask], threshold)["f1"])
                fold_ap.append(
                    binary_metrics(labels[mask], oof[mask], threshold)["average_precision"]
                )
        return CrossValidationReport(
            n_folds=folds,
            grouped=use_groups,
            threshold=threshold,
            precision=pooled["precision"],
            recall=pooled["recall"],
            f1=pooled["f1"],
            average_precision=pooled["average_precision"],
            roc_auc=pooled["roc_auc"],
            fold_f1=tuple(fold_f1),
            fold_average_precision=tuple(fold_ap),
            oof_probability=oof,
        )

    def fit(
        self,
        features: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        cv_folds: int = 5,
        augment_missing: float = 0.2,
    ) -> CrossValidationReport:
        """Cross-validate, then fit on every row and store the F1-optimal threshold.

        The data must be **held out from the base detectors' training** (see the module docs).

        Args:
            features: ``(n, 10)`` meta-feature matrix (:func:`meta_feature_row` per event).
            y: Boolean ground truth (any anomaly).
            groups: Entity id per row for grouped cross-validation.
            cv_folds: Folds of the validation run.
            augment_missing: Share of extra rows with randomly blanked members (``0`` disables).

        Returns:
            The cross-validation report; ``self.threshold`` is its out-of-fold threshold.

        Raises:
            ValueError: Bad shapes, or too few anomalies / normals.
        """
        matrix = self._check(features, y)
        labels = np.asarray(y, dtype=bool)
        report = self.cross_validate(
            matrix, labels, groups=groups, n_folds=cv_folds, augment_missing=augment_missing
        )
        train_x, train_y = self._augment(
            matrix, labels.astype(int), augment_missing, self.params["seed"]
        )
        model = self._new_model()
        model.fit(train_x, train_y)
        self._model = model
        self.threshold = report.threshold
        self.report = report
        _log.info(
            "meta_learner_fitted",
            n_rows=len(matrix),
            n_anomalies=int(labels.sum()),
            cv_f1=report.f1,
            cv_average_precision=report.average_precision,
            threshold=report.threshold,
            n_folds=report.n_folds,
        )
        return report

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the anomaly probability of each row.

        Raises:
            RuntimeError: The learner is not fitted.
            ValueError: Wrong feature shape.
        """
        if self._model is None:
            raise RuntimeError("MetaLearner is not fitted")
        matrix = self._check(features)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            result: np.ndarray = self._model.predict_proba(matrix)[:, 1]
        return result

    def feature_importance(self) -> dict[str, float]:
        """Return the share of total split gain of each meta-feature (sums to 1)."""
        if self._model is None:
            raise RuntimeError("MetaLearner is not fitted")
        gain = np.asarray(
            self._model.booster_.feature_importance(importance_type="gain"), dtype=float
        )
        total = gain.sum()
        share = gain / total if total > 0 else gain
        return dict(zip(self.feature_names, (float(g) for g in share), strict=True))

    def save(self, path: str | Path) -> None:
        """Pickle the fitted learner with ``joblib``."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a learner saved with :meth:`save` (only load files you trust)."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return loaded


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleAnomalyScore:
    """The ensemble's verdict on one event.

    Attributes:
        entity_id: Sensor, vehicle or SKU.
        timestamp: Event time.
        ensemble_score: ``0..1`` with the decision boundary at 0.5.
        individual_scores: Score of each member that reported (warming / missing ones are absent).
        predicted_anomaly_type: ``point`` / ``contextual`` / ``collective`` (``None`` if normal).
        confidence: ``|2 * ensemble_score - 1|``: 0 on the boundary, 1 when certain either way.
        is_anomaly: ``ensemble_score >= 0.5``.
        raw_value: The raw reading, if a member knows it.
        members: The full member results, for explanations and the dashboard.
        details: ``method`` (``meta_learner`` / ``weighted_average`` / ``none``), the raw
            probability, the threshold and which members were used / missing / warming up.
    """

    entity_id: str
    timestamp: datetime
    ensemble_score: float
    individual_scores: dict[str, float]
    predicted_anomaly_type: str | None
    confidence: float
    is_anomaly: bool
    raw_value: float | None = None
    members: dict[str, AnomalyScore] = field(default_factory=dict, repr=False)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "entity_id": self.entity_id,
            "timestamp": self.timestamp.isoformat(),
            "ensemble_score": self.ensemble_score,
            "individual_scores": dict(self.individual_scores),
            "predicted_anomaly_type": self.predicted_anomaly_type,
            "confidence": self.confidence,
            "is_anomaly": self.is_anomaly,
            "raw_value": self.raw_value,
            "details": dict(self.details),
        }


def _vote_type(
    types: Sequence[str | None],
    scores: Sequence[float],
    flagged: Sequence[bool],
    weights: Sequence[float],
) -> str | None:
    """Pick the anomaly type with the largest ``weight * score`` among the typed members.

    Flagged members vote; if none of the typed members is flagged (the ensemble fired on the
    combined evidence), every typed member votes.
    """
    typed = [i for i, t in enumerate(types) if t is not None and math.isfinite(scores[i])]
    voters = [i for i in typed if flagged[i]] or typed
    if not voters:
        return None
    totals: dict[str, float] = {}
    for i in voters:
        totals[str(types[i])] = totals.get(str(types[i]), 0.0) + weights[i] * scores[i]
    return max(totals, key=lambda name: totals[name])


class EnsembleDetector:
    """Stacking ensemble over the four detectors (Level 1 -> Level 2).

    Streaming use (Day 15)::

        ensemble = EnsembleDetector.from_config(cfg.models.ensemble, statistical=stat, ...)
        score = ensemble.detect("wh1-A-temp-01", event=raw_event, vector=feature_vector)
        if score.is_anomaly:
            ...  # SeverityScorer -> build_anomaly_event -> AlertDeduplicator

    ``detect`` runs each available member, isolates their failures (a crashing or unfitted member
    is logged, counted in :attr:`member_failures` and treated as missing) and combines the rest.
    Use one ensemble per source type: the Isolation Forest and the LSTM detectors are bound to one
    feature schema.

    Offline use: build a :class:`MemberScoreTable` with :func:`collect_member_scores`, fit the
    stacker on held-out data with :meth:`fit_meta_learner` and score with :meth:`score_table`.
    """

    def __init__(
        self,
        statistical: UnifiedStatisticalDetector | None = None,
        isolation_forest: IsolationForestDetector | None = None,
        lstm_autoencoder: LSTMAutoencoderDetector | None = None,
        lstm_classifier: LSTMClassifierDetector | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        meta_learner: MetaLearner | None = None,
        fallback_threshold: float = BOUNDARY,
        min_members_for_meta: int = 2,
        cv_folds: int = 5,
    ) -> None:
        """Create the ensemble; every member is optional.

        Args:
            statistical: The Day 8 detector (needs ``event`` in :meth:`detect`).
            isolation_forest: The Day 9 detector (needs ``vector``).
            lstm_autoencoder: The Day 10 detector (reads sequences from the feature store).
            lstm_classifier: The Day 11 detector (reads sequences from the feature store).
            weights: Fallback weights per member (default :data:`DEFAULT_WEIGHTS`); normalised.
            meta_learner: A fitted :class:`MetaLearner`, or ``None`` until :meth:`fit_meta_learner`.
            fallback_threshold: Decision threshold of the weighted average (mapped onto 0.5).
            min_members_for_meta: Usable members needed to trust the stacker for an event.
            cv_folds: Cross-validation folds used by :meth:`fit_meta_learner`.

        Raises:
            ValueError: Unknown / negative weights, an empty weight sum or a bad threshold.
        """
        self.members: dict[str, Any] = {
            STATISTICAL: statistical,
            ISOLATION_FOREST: isolation_forest,
            LSTM_AUTOENCODER: lstm_autoencoder,
            LSTM_CLASSIFIER: lstm_classifier,
        }
        raw = dict(DEFAULT_WEIGHTS if weights is None else weights)
        unknown = set(raw) - set(MEMBERS)
        if unknown:
            raise ValueError(f"unknown ensemble members in weights: {sorted(unknown)}")
        vector = np.array([float(raw.get(name, 0.0)) for name in MEMBERS])
        if (vector < 0).any() or vector.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        self._weights = vector / vector.sum()
        if not 0.0 < fallback_threshold < 1.0:
            raise ValueError("fallback_threshold must be in (0, 1)")
        if not 1 <= min_members_for_meta <= N_MEMBERS:
            raise ValueError(f"min_members_for_meta must be in [1, {N_MEMBERS}]")
        self.fallback_threshold = float(fallback_threshold)
        self.min_members_for_meta = min_members_for_meta
        self.cv_folds = cv_folds
        self.meta_learner = meta_learner
        self.member_failures: dict[str, int] = dict.fromkeys(MEMBERS, 0)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: EnsembleConfig, **members: Any) -> EnsembleDetector:
        """Create the ensemble with weights and folds from ``config.models.ensemble``."""
        weights = {
            STATISTICAL: cfg.statistical_weight,
            ISOLATION_FOREST: cfg.isolation_forest_weight,
            LSTM_AUTOENCODER: cfg.lstm_autoencoder_weight,
            LSTM_CLASSIFIER: cfg.lstm_classifier_weight,
        }
        return cls(weights=weights, cv_folds=cfg.cv_folds, **members)

    # ------------------------------------------------------------------ #
    @property
    def weights(self) -> dict[str, float]:
        """The normalised fallback weights."""
        return dict(zip(MEMBERS, (float(w) for w in self._weights), strict=True))

    @property
    def uses_meta_learner(self) -> bool:
        """Whether a fitted stacker is attached."""
        return self.meta_learner is not None and self.meta_learner.is_fitted

    @property
    def available_members(self) -> list[str]:
        """Names of the members that were configured."""
        return [name for name in MEMBERS if self.members[name] is not None]

    # ------------------------------------------------------------------ #
    # Core scoring (vectorised, shared by the streaming and the offline path)
    # ------------------------------------------------------------------ #
    def score_arrays(
        self,
        scores: np.ndarray,
        confidence: np.ndarray,
        anomaly_rate: np.ndarray,
        hour: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine member scores of ``n`` events.

        Args:
            scores: ``(n, 4)`` member scores in :data:`MEMBERS` order, ``NaN`` where missing.
            confidence: ``(n, 4)`` member confidences.
            anomaly_rate: ``(n,)`` historical anomaly rate per event.
            hour: ``(n,)`` hour of day.

        Returns:
            ``(ensemble_score, probability, used_meta)``: scores on the 0.5-boundary scale, the
            probability before the mapping (the weighted average itself for fallback rows) and a
            boolean mask of the rows the stacker decided.
        """
        s = np.asarray(scores, dtype=np.float64)
        usable = ~np.isnan(s)
        weighted_sum = np.where(usable, s * self._weights, 0.0).sum(axis=1)
        weight_sum = np.where(usable, self._weights, 0.0).sum(axis=1)
        average = np.divide(weighted_sum, weight_sum, out=np.zeros(len(s)), where=weight_sum > 0)
        fallback = to_boundary_scale(average, self.fallback_threshold)
        result = np.asarray(fallback, dtype=np.float64)
        probability = average.copy()
        used_meta = np.zeros(len(s), dtype=bool)
        if self.uses_meta_learner:
            assert self.meta_learner is not None
            ok = usable.sum(axis=1) >= self.min_members_for_meta
            if ok.any():
                matrix = np.column_stack([s, confidence, anomaly_rate, hour])[ok]
                p = self.meta_learner.predict_proba(matrix)
                result[ok] = to_boundary_scale(p, self.meta_learner.threshold)
                probability[ok] = p
                used_meta = ok
        return result, probability, used_meta

    def combine(
        self,
        entity_id: str,
        timestamp: Timestamp,
        scores: Mapping[str, AnomalyScore | None],
        *,
        anomaly_rate: float | None = None,
        raw_value: float | None = None,
    ) -> EnsembleAnomalyScore:
        """Combine already computed member results into an :class:`EnsembleAnomalyScore`.

        Args:
            entity_id: The entity the results belong to.
            timestamp: Event time.
            scores: Member name to :class:`AnomalyScore` (``None`` / warming-up members are
                ignored).
            anomaly_rate: The entity's historical anomaly rate, a meta-learner input.
            raw_value: The raw reading to report (default: the first member that knows it).
        """
        stamp = _aware(parse_timestamp(timestamp))
        row = meta_feature_row(scores, anomaly_rate, stamp.hour)
        member_scores = row[:N_MEMBERS][None, :]
        final, probability, used_meta = self.score_arrays(
            member_scores,
            row[N_MEMBERS : 2 * N_MEMBERS][None, :],
            row[2 * N_MEMBERS][None],
            row[-1][None],
        )
        score = float(final[0])
        usable_scores = {n: s for n in MEMBERS if (s := scores.get(n)) is not None and is_usable(s)}
        usable = list(usable_scores)
        warming = [
            n
            for n in MEMBERS
            if (s := scores.get(n)) is not None and s.details.get("warming_up", False)
        ]
        is_anomaly = score >= BOUNDARY
        anomaly_type: str | None = None
        if is_anomaly:
            anomaly_type = _vote_type(
                [usable_scores[n].anomaly_type if n in usable_scores else None for n in MEMBERS],
                [float(v) for v in member_scores[0]],
                [usable_scores[n].is_anomaly if n in usable_scores else False for n in MEMBERS],
                list(self._weights),
            )
        raw = raw_value
        if raw is None:
            raw = next(
                (s.raw_value for s in usable_scores.values() if s.raw_value is not None), None
            )
        method = "none" if not usable else ("meta_learner" if used_meta[0] else "weighted_average")
        return EnsembleAnomalyScore(
            entity_id=entity_id,
            timestamp=stamp,
            ensemble_score=score,
            individual_scores={n: float(s.score) for n, s in usable_scores.items()},
            predicted_anomaly_type=anomaly_type,
            confidence=abs(2.0 * score - 1.0),
            is_anomaly=is_anomaly,
            raw_value=raw,
            members=dict(usable_scores),
            details={
                "method": method,
                "probability": float(probability[0]),
                "threshold": (
                    self.meta_learner.threshold
                    if method == "meta_learner" and self.meta_learner is not None
                    else self.fallback_threshold
                ),
                "members_used": usable,
                "members_missing": [n for n in MEMBERS if n not in usable and n not in warming],
                "members_warming_up": warming,
                "warming_up": not usable and bool(warming),
            },
        )

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def _run(self, name: str, call: Callable[[], Any]) -> AnomalyScore | None:
        """Run one member, turning any failure into a missing result."""
        try:
            result = call()
            scored: AnomalyScore = statistical_to_score(result) if name == STATISTICAL else result
            return scored
        except Exception as error:  # noqa: BLE001 - one broken member must not stop the others
            with self._lock:
                self.member_failures[name] += 1
                count = self.member_failures[name]
            _log.warning("ensemble_member_failed", member=name, error=repr(error), failures=count)
            return None

    @staticmethod
    def _rate_of(vector: FeatureVector | None, override: float | None) -> float | None:
        if override is not None:
            return override
        if vector is None:
            return None
        try:
            return float(vector.raw[vector.names.index(_RATE_FEATURE)])
        except (ValueError, IndexError):
            return None

    def detect(
        self,
        entity_id: str,
        *,
        event: Mapping[str, Any] | None = None,
        vector: FeatureVector | None = None,
        timestamp: Timestamp | None = None,
        anomaly_rate: float | None = None,
        learn: bool = True,
    ) -> EnsembleAnomalyScore:
        """Run every available member on one event and combine the results.

        Args:
            entity_id: The pipeline's entity id (sensor / vehicle / SKU).
            event: The raw simulator event; feeds the statistical detector.
            vector: ``FeaturePipeline.process(entity_id)``; feeds the Isolation Forest and
                supplies the historical anomaly rate.
            timestamp: Event time; defaults to the vector's, the event's or the members'.
            anomaly_rate: Overrides the rate taken from ``vector``.
            learn: Let the Isolation Forest learn from the vector (``False`` for replays).

        Returns:
            The ensemble verdict. If every member is missing or warming up the score is 0.
        """
        results: dict[str, AnomalyScore | None] = dict.fromkeys(MEMBERS)
        if self.members[STATISTICAL] is not None and event is not None:
            results[STATISTICAL] = self._run(
                STATISTICAL, lambda: self.members[STATISTICAL].detect_event(event)
            )
        if self.members[ISOLATION_FOREST] is not None and vector is not None:
            results[ISOLATION_FOREST] = self._run(
                ISOLATION_FOREST,
                lambda: self.members[ISOLATION_FOREST].process(vector, learn=learn),
            )
        if self.members[LSTM_AUTOENCODER] is not None:
            results[LSTM_AUTOENCODER] = self._run(
                LSTM_AUTOENCODER, lambda: self.members[LSTM_AUTOENCODER].detect(entity_id)
            )
        if self.members[LSTM_CLASSIFIER] is not None:
            results[LSTM_CLASSIFIER] = self._run(
                LSTM_CLASSIFIER, lambda: self.members[LSTM_CLASSIFIER].detect(entity_id)
            )
        stamp: Timestamp | None = timestamp
        if stamp is None and vector is not None:
            stamp = vector.timestamp
        if stamp is None and event is not None:
            stamp = event.get("timestamp")
        if stamp is None:
            stamp = next(
                (r.timestamp for r in results.values() if r is not None), datetime.now(UTC)
            )
        return self.combine(
            entity_id, stamp, results, anomaly_rate=self._rate_of(vector, anomaly_rate)
        )

    def detect_many(
        self,
        entity_ids: Sequence[str],
        *,
        events: Sequence[Mapping[str, Any] | None] | None = None,
        vectors: Sequence[FeatureVector | None] | None = None,
        learn: bool = True,
    ) -> list[EnsembleAnomalyScore]:
        """Micro-batch version of :meth:`detect` (one batched call per neural / forest member).

        All entities must be of one source type. ``events`` and ``vectors`` are aligned with
        ``entity_ids``; a ``None`` entry means that entity has no event / vector yet.
        """
        n = len(entity_ids)
        events = events if events is not None else [None] * n
        vectors = vectors if vectors is not None else [None] * n
        if len(events) != n or len(vectors) != n:
            raise ValueError("events and vectors must be aligned with entity_ids")
        per_member: dict[str, list[AnomalyScore | None]] = {name: [None] * n for name in MEMBERS}

        stat = self.members[STATISTICAL]
        if stat is not None:
            for i, event in enumerate(events):
                if event is not None:
                    per_member[STATISTICAL][i] = self._run(
                        STATISTICAL, partial(stat.detect_event, event)
                    )
        forest = self.members[ISOLATION_FOREST]
        have_vector = [i for i, v in enumerate(vectors) if v is not None]
        if forest is not None and have_vector:
            batch = self._run_batch(
                ISOLATION_FOREST,
                partial(forest.process_many, [vectors[i] for i in have_vector], learn=learn),
            )
            for slot, i in enumerate(have_vector):
                per_member[ISOLATION_FOREST][i] = batch[slot] if batch is not None else None
        for name in (LSTM_AUTOENCODER, LSTM_CLASSIFIER):
            member = self.members[name]
            if member is not None and n:
                batch = self._run_batch(name, partial(member.detect_many, list(entity_ids)))
                if batch is not None:
                    per_member[name] = list(batch)
        out = []
        for i, entity_id in enumerate(entity_ids):
            row = {name: per_member[name][i] for name in MEMBERS}
            vector_i, event_i = vectors[i], events[i]
            stamp: Timestamp | None = vector_i.timestamp if vector_i is not None else None
            if stamp is None and event_i is not None:
                stamp = event_i.get("timestamp")
            if stamp is None:
                stamp = next(
                    (r.timestamp for r in row.values() if r is not None), datetime.now(UTC)
                )
            out.append(
                self.combine(entity_id, stamp, row, anomaly_rate=self._rate_of(vectors[i], None))
            )
        return out

    def _run_batch(
        self, name: str, call: Callable[[], Sequence[AnomalyScore]]
    ) -> Sequence[AnomalyScore] | None:
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - see _run
            with self._lock:
                self.member_failures[name] += 1
                count = self.member_failures[name]
            _log.warning("ensemble_member_failed", member=name, error=repr(error), failures=count)
            return None

    # ------------------------------------------------------------------ #
    # Offline: stacking and evaluation
    # ------------------------------------------------------------------ #
    def fit_meta_learner(
        self,
        table: MemberScoreTable,
        *,
        meta_learner: MetaLearner | None = None,
        cv_folds: int | None = None,
        augment_missing: float = 0.2,
    ) -> CrossValidationReport:
        """Fit the stacker on a held-out :class:`MemberScoreTable` and attach it.

        The table must come from labelled data the four base detectors were **not** trained on.
        Cross-validation is grouped by entity.

        Returns:
            The out-of-fold report (its threshold becomes the stacker's operating threshold).
        """
        learner = meta_learner or self.meta_learner or MetaLearner()
        report = learner.fit(
            table.meta_features(),
            table.y,
            groups=table.entity_ids,
            cv_folds=cv_folds or self.cv_folds,
            augment_missing=augment_missing,
        )
        self.meta_learner = learner
        return report

    def score_table(self, table: MemberScoreTable) -> np.ndarray:
        """Ensemble scores (0.5 boundary) of every row of a table, with the current setup."""
        final, _, _ = self.score_arrays(
            table.scores, table.confidence, table.anomaly_rate, table.hour
        )
        return final

    def predict_types(self, table: MemberScoreTable) -> np.ndarray:
        """Predicted anomaly type of every row (``None`` where the ensemble does not fire)."""
        final = self.score_table(table)
        out = np.full(len(table), None, dtype=object)
        weights = list(self._weights)
        for i in np.flatnonzero(final >= BOUNDARY):
            out[i] = _vote_type(
                list(table.types[i]),
                [float(v) for v in table.scores[i]],
                list(table.flagged[i]),
                weights,
            )
        return out

    def save_meta_learner(
        self,
        registry: ModelRegistry,
        *,
        metrics: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ModelVersion:
        """Store the fitted stacker as a new ``ensemble_meta_learner`` registry version."""
        if self.meta_learner is None or not self.meta_learner.is_fitted:
            raise RuntimeError("no fitted meta-learner to save")
        merged = dict(self.meta_learner.report.to_dict() if self.meta_learner.report else {})
        merged.update(metrics or {})
        return registry.save(
            MODEL_TYPE_META,
            self.meta_learner,
            metrics=merged,
            hyperparams=self.meta_learner.params,
            extra={
                "feature_names": list(META_FEATURE_NAMES),
                "threshold": self.meta_learner.threshold,
                **(extra or {}),
            },
            serializer="joblib",
        )

    def load_meta_learner(self, registry: ModelRegistry, version: str | None = None) -> MetaLearner:
        """Load (the latest or a given version of) the stacker from the registry and attach it."""
        loaded = registry.load(MODEL_TYPE_META, version)
        if not isinstance(loaded.model, MetaLearner):
            raise TypeError("registry entry is not a MetaLearner")
        self.meta_learner = loaded.model
        return loaded.model


# --------------------------------------------------------------------------- #
# Offline score tables
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemberScoreTable:
    """The four detectors' scores for a set of labelled events (the stacker's training data).

    Attributes:
        entity_ids: Entity of each row.
        timestamps: Event time, UTC epoch seconds.
        scores: ``(n, 4)`` member scores in :data:`MEMBERS` order; ``NaN`` = no opinion.
        confidence: ``(n, 4)`` member confidences (``NaN`` where the score is).
        flagged: ``(n, 4)`` each member's own decision.
        types: ``(n, 4)`` each member's native anomaly type (``None`` where it has none).
        anomaly_rate: Historical anomaly rate feature of the entity (``NaN`` if unknown).
        hour: Hour of day (UTC) of the event.
        y: Ground truth: the event is anomalous.
        anomaly_type: Ground-truth type (``None`` for normal events).
    """

    entity_ids: np.ndarray
    timestamps: np.ndarray
    scores: np.ndarray
    confidence: np.ndarray
    flagged: np.ndarray
    types: np.ndarray
    anomaly_rate: np.ndarray
    hour: np.ndarray
    y: np.ndarray
    anomaly_type: np.ndarray

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def meta_features(self) -> np.ndarray:
        """Return the ``(n, 10)`` meta-learner input (order of :data:`META_FEATURE_NAMES`)."""
        return np.column_stack([self.scores, self.confidence, self.anomaly_rate, self.hour])

    def member_score(self, name: str) -> np.ndarray:
        """Return the score column of one member."""
        return np.asarray(self.scores[:, MEMBERS.index(name)])

    def subset(self, mask: np.ndarray) -> MemberScoreTable:
        """Rows selected by a boolean mask or an index array."""
        return MemberScoreTable(
            *(
                getattr(self, f)[mask]
                for f in (
                    "entity_ids",
                    "timestamps",
                    "scores",
                    "confidence",
                    "flagged",
                    "types",
                    "anomaly_rate",
                    "hour",
                    "y",
                    "anomaly_type",
                )
            )
        )

    @classmethod
    def concat(cls, tables: Sequence[MemberScoreTable]) -> MemberScoreTable:
        """Stack several tables (for example one per sensor type or per simulation)."""
        fields = ("entity_ids", "timestamps", "scores", "confidence", "flagged", "types",
                  "anomaly_rate", "hour", "y", "anomaly_type")  # fmt: skip
        return cls(*(np.concatenate([getattr(t, f) for t in tables]) for f in fields))


def holdout_mask(
    timestamps: np.ndarray,
    entity_ids: np.ndarray,
    fraction: float = 0.2,
    *,
    block_s: float = 900.0,
    seed: int = 0,
) -> np.ndarray:
    """Pick about ``fraction`` of the rows as a leakage-aware hold-out for the stacker.

    Whole ``(entity, block_s)`` blocks are chosen at random, so two neighbouring readings of one
    sensor (which are almost identical) never land on both sides of the split. Train the base
    detectors on ``~mask`` and the stacker on ``mask``.

    Raises:
        ValueError: ``fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    blocks = np.floor(np.asarray(timestamps, dtype=np.float64) / block_s).astype(np.int64)
    codes = np.unique(np.asarray(entity_ids).astype(str), return_inverse=True)[1].astype(np.int64)
    unique, inverse = np.unique(codes * (1 << 32) + blocks, return_inverse=True)
    chosen = np.random.default_rng(seed).random(len(unique)) < fraction
    mask: np.ndarray = chosen[inverse]
    return mask


def collect_member_scores(
    dataset: LabeledFeatureSet,
    scaler: OnlineScaler,
    *,
    statistical: UnifiedStatisticalDetector | None = None,
    isolation_forest: IsolationForestDetector | None = None,
    lstm_autoencoder: LSTMAutoencoderDetector | None = None,
    lstm_classifier: LSTMClassifierDetector | None = None,
    stride: int = 10,
    statistical_feature: str | None = None,
    reset_statistical: bool = True,
) -> MemberScoreTable:
    """Score labelled feature rows with every member and join the results per event.

    The events scored are the window ends of the longest-window member (one every ``stride`` rows
    per entity, newest first); the other members are looked up at the same ``(entity, time)``.
    The statistical detector is stateful, so it is streamed over **all** rows of the dataset in
    order and only its results at the selected events are kept. The neural detectors are evaluated
    on the standardised windows (``scaler``, the same one they were trained with); the Isolation
    Forest scores the raw feature rows with ``score_batch`` (no learning).

    Args:
        dataset: Output of ``simulate_labeled_features`` for one source type, in stream order.
        scaler: The pipeline's scaler of that source (fitted on the base-training data).
        statistical: Day 8 detector (``None`` leaves its column ``NaN``); reset first by default.
        isolation_forest: A *fitted* Day 9 detector for the same feature schema.
        lstm_autoencoder: A calibrated Day 10 detector.
        lstm_classifier: A Day 11 detector.
        stride: Rows between selected events of one entity.
        statistical_feature: The feature column holding the raw reading (default: the first
            ``*__last`` column).
        reset_statistical: Forget the statistical detector's state before streaming.

    Returns:
        The :class:`MemberScoreTable` (chronological).

    Raises:
        ValueError: No window could be formed, or the feature schemas disagree.
    """
    import pandas as pd

    from src.models.lstm_autoencoder import make_sequences

    names = tuple(dataset.feature_names)
    lstm_members = [m for m in (lstm_autoencoder, lstm_classifier) if m is not None]
    master_len = max((m.seq_len for m in lstm_members), default=2)
    master_columns = None
    if lstm_members:
        master_columns = max(lstm_members, key=lambda m: m.seq_len).columns
    master = make_sequences(
        dataset, scaler, seq_len=master_len, stride=stride, columns=master_columns
    )
    n = len(master)
    if n == 0:
        raise ValueError(
            "no events to score: each entity needs more than the longest window of warm rows"
        )

    # row of the dataset behind each selected event
    keys = pd.MultiIndex.from_arrays([dataset.entity_ids.astype(str), dataset.timestamps])
    unique_rows = np.flatnonzero(~keys.duplicated(keep="last"))
    targets = pd.MultiIndex.from_arrays([master.entity_ids.astype(str), master.end_timestamps])
    position = keys[unique_rows].get_indexer(targets)
    if (position < 0).any():
        raise ValueError("could not align the windows with the dataset rows")
    rows = unique_rows[position]

    scores = np.full((n, N_MEMBERS), np.nan)
    confidence = np.full((n, N_MEMBERS), np.nan)
    flagged = np.zeros((n, N_MEMBERS), dtype=bool)
    types = np.full((n, N_MEMBERS), None, dtype=object)

    # Statistical: stream every row, keep the selected ones
    if statistical is not None:
        column = statistical_feature or next((c for c in names if c.endswith("__last")), None)
        if column is None:
            raise ValueError("no '*__last' feature to feed the statistical detector")
        values = dataset.X[:, names.index(column)]
        if reset_statistical:
            statistical.reset()
        slot_of_row = {int(r): k for k, r in enumerate(rows)}
        zones = dataset.zones
        for i in range(len(dataset)):
            value = float(values[i])
            if not math.isfinite(value):
                continue
            zone = zones[i]
            result = statistical.detect(
                str(dataset.entity_ids[i]),
                value,
                float(dataset.timestamps[i]),
                None if zone is None else str(zone),
            )
            k = slot_of_row.get(i)
            if k is not None and not result.details.get("warming_up", False):
                col = MEMBERS.index(STATISTICAL)
                scores[k, col] = result.severity_score
                confidence[k, col] = result.confidence
                flagged[k, col] = result.is_anomaly
                types[k, col] = result.anomaly_type

    # Isolation Forest: batch scoring of the raw rows
    if isolation_forest is not None:
        if tuple(isolation_forest.feature_names) != names:
            raise ValueError("the Isolation Forest was built for a different feature schema")
        forest_scores = np.asarray(isolation_forest.score_batch(dataset.X[rows]), dtype=np.float64)
        col = MEMBERS.index(ISOLATION_FOREST)
        scores[:, col] = forest_scores
        confidence[:, col] = np.abs(2.0 * forest_scores - 1.0)
        flagged[:, col] = forest_scores >= BOUNDARY

    # Neural detectors on standardised windows, joined by (entity, time)
    def _windows(member: Any) -> tuple[Any, np.ndarray]:
        windows = (
            master
            if member.seq_len == master_len and member.columns == master_columns
            else make_sequences(
                dataset, scaler, seq_len=member.seq_len, stride=stride, columns=member.columns
            )
        )
        member_keys = pd.MultiIndex.from_arrays(
            [windows.entity_ids.astype(str), windows.end_timestamps]
        )
        return windows, member_keys.get_indexer(targets)

    if lstm_autoencoder is not None:
        windows, index = _windows(lstm_autoencoder)
        ok = index >= 0
        if ok.any():
            out = lstm_autoencoder.score_sequences(windows.x[index[ok]])
            col = MEMBERS.index(LSTM_AUTOENCODER)
            scores[ok, col] = out.score
            confidence[ok, col] = out.confidence
            flagged[ok, col] = out.is_anomaly
            types[ok, col] = "collective"
    if lstm_classifier is not None:
        windows, index = _windows(lstm_classifier)
        ok = index >= 0
        if ok.any():
            out_c = lstm_classifier.classify_sequences(windows.x[index[ok]])
            col = MEMBERS.index(LSTM_CLASSIFIER)
            cls_scores = np.asarray(out_c.score, dtype=np.float64)
            scores[ok, col] = cls_scores
            confidence[ok, col] = np.maximum(cls_scores, 1.0 - cls_scores)
            flagged[ok, col] = cls_scores >= lstm_classifier.decision_threshold
            anomaly_names = np.asarray(lstm_classifier.class_names[1:], dtype=object)
            types[ok, col] = anomaly_names[out_c.probabilities[:, 1:].argmax(axis=1)]

    rate = np.full(n, np.nan)
    if "anomaly_rate_7d" in names:
        rate = dataset.X[rows, names.index("anomaly_rate_7d")].astype(np.float64)
    return MemberScoreTable(
        entity_ids=master.entity_ids,
        timestamps=master.end_timestamps,
        scores=scores,
        confidence=confidence,
        flagged=flagged,
        types=types,
        anomaly_rate=rate,
        hour=((master.end_timestamps // 3600) % 24).astype(np.float64),
        y=master.y_last.astype(bool),
        anomaly_type=master.type_last,
    )


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
class Severity(IntEnum):
    """Alert severity, ordered so that levels can be compared and escalated."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

    def escalated(self, steps: int = 1) -> Severity:
        """Return the level ``steps`` above this one (capped at EMERGENCY)."""
        return Severity(min(int(Severity.EMERGENCY), int(self) + steps))


IMPACT_COLD_STORAGE: Final = "cold_storage_temp"
IMPACT_VEHICLE: Final = "vehicle_deviation"
IMPACT_INVENTORY: Final = "inventory_drop"
IMPACT_DEFAULT: Final = "default"

_ACTIONS: Final[dict[tuple[str, str | None], str]] = {
    (IMPACT_COLD_STORAGE, "point"): (
        "Check the cold-storage door, compressor and the neighbouring sensor in the zone; "
        "move perishables if the reading stays out of range."
    ),
    (IMPACT_COLD_STORAGE, "contextual"): (
        "Reading is unusual for this time of day: check whether the door was left open or the "
        "unit was serviced, and review the off-hours access log."
    ),
    (IMPACT_COLD_STORAGE, "collective"): (
        "Temperature is drifting: schedule a refrigeration inspection and recalibrate the sensor; "
        "prepare to relocate stock if the drift continues."
    ),
    (IMPACT_COLD_STORAGE, None): "Inspect the cold-storage unit and verify the sensor.",
    (IMPACT_VEHICLE, "point"): (
        "Contact the driver to confirm the stop, detour or speed; verify route and cargo status."
    ),
    (IMPACT_VEHICLE, "contextual"): (
        "Vehicle activity is unusual for this time: confirm the dispatch plan with the driver."
    ),
    (IMPACT_VEHICLE, "collective"): (
        "Progress is falling behind plan: check for traffic or a breakdown, re-estimate the ETA "
        "and notify the receiving site."
    ),
    (IMPACT_VEHICLE, None): "Contact the driver and verify the route.",
    (IMPACT_INVENTORY, "point"): (
        "Run a cycle count of the SKU; check recent shipments and system errors for loss or theft."
    ),
    (IMPACT_INVENTORY, "contextual"): (
        "Stock movement is unusual for this time: check who booked the transaction and whether "
        "it matches a shipment."
    ),
    (IMPACT_INVENTORY, "collective"): (
        "Stock drifts away from the expected level: reconcile receipts and shipments "
        "(possible ghost stock or systematic shrinkage)."
    ),
    (IMPACT_INVENTORY, None): "Verify the stock level with a cycle count.",
    (
        IMPACT_DEFAULT,
        "point",
    ): "Inspect the entity and verify the reading; check the sensor itself.",
    (
        IMPACT_DEFAULT,
        "contextual",
    ): "Reading is unusual for this time of day: review recent activity.",
    (
        IMPACT_DEFAULT,
        "collective",
    ): "Reading is drifting: schedule an inspection and recalibration.",
    (IMPACT_DEFAULT, None): "Inspect the entity and verify the reading.",
}
_URGENCY: Final[dict[Severity, str]] = {
    Severity.INFO: "Low priority - ",
    Severity.WARNING: "",
    Severity.CRITICAL: "Urgent: ",
    Severity.EMERGENCY: "IMMEDIATE ACTION: ",
}


@dataclass(frozen=True)
class EntityContext:
    """What the severity scorer needs to know about the entity behind an alert.

    Attributes:
        entity_type: ``warehouse_sensor``, ``vehicle`` or ``inventory_sku``.
        impact_key: The business-impact key (``cold_storage_temp``, ``vehicle_deviation``,
            ``inventory_drop`` or ``default``).
        sensor_type: ``temperature`` / ``humidity`` for warehouse sensors.
        zone: Warehouse zone.
        storage_type: ``cold_storage`` / ``dry_goods`` / ``ambient``.
        expected_range: Healthy ``(low, high)`` of the reported value.
        asset_value: Value of the goods at risk (currency units), if known.
        unit: Unit of the raw value.
        raw_value: The raw reading.
    """

    entity_type: str = "unknown"
    impact_key: str = IMPACT_DEFAULT
    sensor_type: str | None = None
    zone: str | None = None
    storage_type: str | None = None
    expected_range: tuple[float, float] | None = None
    asset_value: float | None = None
    unit: str | None = None
    raw_value: float | None = None

    @classmethod
    def from_event(
        cls, event: Mapping[str, Any], *, asset_value: float | None = None
    ) -> EntityContext:
        """Derive the context from a raw simulator event (Day 2/3 schema).

        Ground-truth fields (``is_anomaly``, ``anomaly_type``, ``anomaly_subtype``) are never read.

        Raises:
            ValueError: The event has none of ``sensor_id`` / ``vehicle_id`` / ``sku_id``.
        """
        from src.models.feature_engineering import acceptable_range

        meta = dict(event.get("metadata") or {})
        if "sensor_id" in event:
            kind = str(event.get("reading_type", ""))
            storage = meta.get("storage_type")
            cold = kind == "temperature" and storage == "cold_storage"
            rng = acceptable_range("warehouse", "value", meta)
            return cls(
                "warehouse_sensor", IMPACT_COLD_STORAGE if cold else IMPACT_DEFAULT, kind or None,
                meta.get("zone"), storage, rng, asset_value, event.get("unit"),
                _num(event.get("value")),
            )  # fmt: skip
        if "vehicle_id" in event:
            return cls(
                "vehicle", IMPACT_VEHICLE, None, None, None,
                acceptable_range("vehicle", "speed_kmh", meta), asset_value, "km/h",
                _num(event.get("speed_kmh")),
            )  # fmt: skip
        if "sku_id" in event:
            return cls(
                "inventory_sku", IMPACT_INVENTORY, None, meta.get("zone"), None,
                acceptable_range("inventory", "new_level", meta), asset_value, "units",
                _num(event.get("new_level")),
            )  # fmt: skip
        raise ValueError("event has none of sensor_id / vehicle_id / sku_id")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class SeverityResult:
    """Outcome of :meth:`SeverityScorer.score`.

    Attributes:
        level: The :class:`Severity` level.
        score: The final severity score in ``0..1``.
        business_impact: The impact weight that applied (``config.alerts.business_impact_weights``).
        recommended_action: Text for the operator.
        components: The factors behind ``score``: ``ensemble_score``, ``business_impact``,
            ``time_factor`` and ``asset_factor``.
    """

    level: Severity
    score: float
    business_impact: float
    recommended_action: str
    components: dict[str, float] = field(default_factory=dict)


class SeverityScorer:
    """Multi-dimensional severity: ``clip(ensemble * impact * time_factor * asset_factor)``.

    * ``impact`` is the business-impact weight of the entity class (cold-storage temperature 1.0,
      vehicle deviation 0.8, inventory drop 0.7, anything else 0.5 by default). It caps severity:
      a default-impact entity can reach CRITICAL but only just EMERGENCY.
    * ``time_factor`` amplifies the score by ``business_hours_boost`` during business hours for
      the classes whose damage depends on operations running (vehicles, inventory, ambient
      sensors). Cold-chain temperature is time-invariant (food spoils at 3 a.m. too), so it is
      not amplified.
    * ``asset_factor`` is ``1 + asset_gain * min(asset_value / asset_reference_value, 1)`` and
      neutral (1.0) when the value at risk is unknown.

    Levels: ``< warning`` INFO, ``< critical`` WARNING, ``< emergency`` CRITICAL, else EMERGENCY.
    """

    def __init__(
        self,
        thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85),
        impact_weights: Mapping[str, float] | None = None,
        *,
        time_sensitive: Mapping[str, bool] | None = None,
        business_hours: tuple[int, int] = (8, 18),
        business_hours_boost: float = 1.25,
        asset_reference_value: float = 10_000.0,
        asset_gain: float = 0.2,
    ) -> None:
        """Create the scorer.

        Args:
            thresholds: ``(warning, critical, emergency)`` boundaries, strictly increasing.
            impact_weights: Business-impact weight per key; needs a ``default`` entry.
            time_sensitive: Which impact keys get the business-hours boost.
            business_hours: ``[start, end)`` hours of the event timestamp (as written).
            business_hours_boost: Multiplier applied during business hours (``>= 1``).
            asset_reference_value: Asset value at which the asset factor saturates.
            asset_gain: Largest extra weight of the asset factor.

        Raises:
            ValueError: Inconsistent thresholds, weights outside ``[0, 1]`` or bad parameters.
        """
        warning, critical, emergency = thresholds
        if not 0 < warning < critical < emergency < 1:
            raise ValueError("thresholds must satisfy 0 < warning < critical < emergency < 1")
        weights = dict(
            impact_weights
            or {
                IMPACT_COLD_STORAGE: 1.0,
                IMPACT_VEHICLE: 0.8,
                IMPACT_INVENTORY: 0.7,
                IMPACT_DEFAULT: 0.5,
            }
        )
        if IMPACT_DEFAULT not in weights:
            raise ValueError("impact_weights needs a 'default' entry")
        if any(not 0.0 <= w <= 1.0 for w in weights.values()):
            raise ValueError("impact weights must be within [0, 1]")
        if business_hours_boost < 1.0 or asset_gain < 0 or asset_reference_value <= 0:
            raise ValueError(
                "business_hours_boost >= 1, asset_gain >= 0 and asset_reference_value > 0"
            )
        self.thresholds = (float(warning), float(critical), float(emergency))
        self.impact_weights = weights
        self.time_sensitive = dict(
            time_sensitive
            if time_sensitive is not None
            else {
                IMPACT_COLD_STORAGE: False,
                IMPACT_VEHICLE: True,
                IMPACT_INVENTORY: True,
                IMPACT_DEFAULT: True,
            }
        )
        self.business_hours = business_hours
        self.business_hours_boost = float(business_hours_boost)
        self.asset_reference_value = float(asset_reference_value)
        self.asset_gain = float(asset_gain)

    @classmethod
    def from_config(cls, cfg: AlertsConfig, **overrides: Any) -> SeverityScorer:
        """Create the scorer with thresholds and impact weights from ``config.alerts``."""
        t = cfg.severity_thresholds
        return cls((t.warning, t.critical, t.emergency), cfg.business_impact_weights, **overrides)

    def level_for(self, value: float) -> Severity:
        """Map a severity score to its level."""
        warning, critical, emergency = self.thresholds
        if value >= emergency:
            return Severity.EMERGENCY
        if value >= critical:
            return Severity.CRITICAL
        if value >= warning:
            return Severity.WARNING
        return Severity.INFO

    def is_business_hours(self, moment: datetime) -> bool:
        """Whether ``moment`` (hour as written) falls into the business hours."""
        start, end = self.business_hours
        return start <= moment.hour < end

    def recommended_action(self, impact_key: str, anomaly_type: str | None, level: Severity) -> str:
        """Return the action text for an entity class, anomaly type and level."""
        text = (
            _ACTIONS.get((impact_key, anomaly_type))
            or _ACTIONS.get((impact_key, None))
            or _ACTIONS[(IMPACT_DEFAULT, None)]
        )
        return _URGENCY[level] + (text if _URGENCY[level] == "" else text[0].lower() + text[1:])

    def score(
        self,
        ensemble: EnsembleAnomalyScore | float,
        context: EntityContext | None = None,
        *,
        timestamp: datetime | None = None,
        anomaly_type: str | None = None,
    ) -> SeverityResult:
        """Score one detection.

        Args:
            ensemble: The :class:`EnsembleAnomalyScore`, or a bare ensemble score.
            context: What is known about the entity (default: an unknown entity, impact 0.5).
            timestamp: Event time (taken from ``ensemble`` when it is an object).
            anomaly_type: Anomaly type (taken from ``ensemble`` when it is an object).

        Returns:
            The :class:`SeverityResult`.
        """
        if isinstance(ensemble, EnsembleAnomalyScore):
            value = ensemble.ensemble_score
            timestamp = timestamp or ensemble.timestamp
            anomaly_type = anomaly_type or ensemble.predicted_anomaly_type
        else:
            value = float(ensemble)
        ctx = context or EntityContext()
        key = ctx.impact_key if ctx.impact_key in self.impact_weights else IMPACT_DEFAULT
        impact = self.impact_weights[key]
        time_factor = 1.0
        if (
            timestamp is not None
            and self.time_sensitive.get(key, True)
            and self.is_business_hours(timestamp)
        ):
            time_factor = self.business_hours_boost
        asset_factor = 1.0
        if ctx.asset_value is not None and ctx.asset_value > 0:
            asset_factor += self.asset_gain * min(ctx.asset_value / self.asset_reference_value, 1.0)
        final = float(np.clip(value * impact * time_factor * asset_factor, 0.0, 1.0))
        level = self.level_for(final)
        return SeverityResult(
            level=level,
            score=final,
            business_impact=impact,
            recommended_action=self.recommended_action(key, anomaly_type, level),
            components={
                "ensemble_score": float(value),
                "business_impact": impact,
                "time_factor": time_factor,
                "asset_factor": asset_factor,
            },
        )


# --------------------------------------------------------------------------- #
# Canonical alert record
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyEvent:
    """The canonical alert record (stored, published to ``anomaly-alerts``, shown on the dashboard).

    Attributes:
        event_id: UUID of this alert.
        entity_id: Sensor, vehicle or SKU.
        entity_type: ``warehouse_sensor``, ``vehicle`` or ``inventory_sku``.
        timestamp: Event time.
        anomaly_type: ``point`` / ``contextual`` / ``collective``.
        severity: The :class:`Severity` level.
        raw_value: The reading that triggered the alert.
        expected_range: Healthy ``(low, high)`` of that reading.
        top_contributing_features: SHAP explanation, ``[{feature, shap_value, feature_value}]``.
        recommended_action: Operator guidance.
        is_resolved: Set by an analyst.
        false_positive_marked: Set by the feedback loop (Day 16).
        severity_score: The final score behind ``severity``.
        ensemble_score: The ensemble score before business weighting.
        confidence: Ensemble confidence.
        business_impact: Impact weight that applied.
        individual_scores: Member scores, for the dashboard.
        first_seen: First time this alert (episode) was observed.
        last_seen: Last time it re-fired.
        occurrence_count: How many detections this alert stands for (updated by the deduplicator).
        escalated_from: The level before an escalation, if this alert is an escalation.
    """

    event_id: str
    entity_id: str
    entity_type: str
    timestamp: datetime
    anomaly_type: str | None
    severity: Severity
    raw_value: float | None
    expected_range: tuple[float, float] | None
    top_contributing_features: list[dict[str, Any]]
    recommended_action: str
    is_resolved: bool = False
    false_positive_marked: bool = False
    severity_score: float = 0.0
    ensemble_score: float = 0.0
    confidence: float = 0.0
    business_impact: float = 0.0
    individual_scores: dict[str, float] = field(default_factory=dict)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    occurrence_count: int = 1
    escalated_from: Severity | None = None

    def __post_init__(self) -> None:
        self.timestamp = _aware(self.timestamp)
        self.first_seen = _aware(self.first_seen) if self.first_seen else self.timestamp
        self.last_seen = _aware(self.last_seen) if self.last_seen else self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (levels by name, times in ISO 8601)."""
        assert self.first_seen is not None and self.last_seen is not None
        return {
            "event_id": self.event_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type,
            "severity": self.severity.name,
            "raw_value": self.raw_value,
            "expected_range": list(self.expected_range) if self.expected_range else None,
            "top_contributing_features": [dict(f) for f in self.top_contributing_features],
            "recommended_action": self.recommended_action,
            "is_resolved": self.is_resolved,
            "false_positive_marked": self.false_positive_marked,
            "severity_score": self.severity_score,
            "ensemble_score": self.ensemble_score,
            "confidence": self.confidence,
            "business_impact": self.business_impact,
            "individual_scores": dict(self.individual_scores),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count,
            "escalated_from": self.escalated_from.name if self.escalated_from is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnomalyEvent:
        """Rebuild an event from :meth:`to_dict` output (JSON snapshot or Kafka message)."""
        rng = data.get("expected_range")
        escalated = data.get("escalated_from")
        return cls(
            event_id=str(data["event_id"]),
            entity_id=str(data["entity_id"]),
            entity_type=str(data["entity_type"]),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            anomaly_type=data.get("anomaly_type"),
            severity=Severity[str(data["severity"])],
            raw_value=data.get("raw_value"),
            expected_range=(float(rng[0]), float(rng[1])) if rng else None,
            top_contributing_features=[dict(f) for f in data.get("top_contributing_features", [])],
            recommended_action=str(data.get("recommended_action", "")),
            is_resolved=bool(data.get("is_resolved", False)),
            false_positive_marked=bool(data.get("false_positive_marked", False)),
            severity_score=float(data.get("severity_score", 0.0)),
            ensemble_score=float(data.get("ensemble_score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            business_impact=float(data.get("business_impact", 0.0)),
            individual_scores=dict(data.get("individual_scores", {})),
            first_seen=(
                datetime.fromisoformat(str(data["first_seen"])) if data.get("first_seen") else None
            ),
            last_seen=(
                datetime.fromisoformat(str(data["last_seen"])) if data.get("last_seen") else None
            ),
            occurrence_count=int(data.get("occurrence_count", 1)),
            escalated_from=Severity[str(escalated)] if escalated else None,
        )


def build_anomaly_event(
    ensemble: EnsembleAnomalyScore,
    severity: SeverityResult,
    context: EntityContext | None = None,
    *,
    explainer: IsolationForestExplainer | None = None,
    vector: FeatureVector | None = None,
    top_n: int = 5,
) -> AnomalyEvent:
    """Enrich an ensemble detection into the canonical :class:`AnomalyEvent`.

    Args:
        ensemble: The detection.
        severity: Its :meth:`SeverityScorer.score` result.
        context: Entity context (type, expected range); ``EntityContext()`` if omitted.
        explainer: A Day 9 :class:`IsolationForestExplainer`; with ``vector`` it supplies the
            SHAP top features. Failures are logged and leave the list empty (an explanation must
            never block an alert).
        vector: The feature vector the detection was made from.
        top_n: Number of SHAP features to keep.
    """
    ctx = context or EntityContext()
    top: list[dict[str, Any]] = []
    if explainer is not None and vector is not None:
        try:
            top = [
                {"feature": c.feature, "shap_value": c.shap_value, "feature_value": c.feature_value}
                for c in explainer.get_top_features(vector.raw, n=top_n)
            ]
        except Exception as error:  # noqa: BLE001 - explanations are best-effort
            _log.warning("shap_explanation_failed", entity_id=ensemble.entity_id, error=repr(error))
    raw = ensemble.raw_value if ensemble.raw_value is not None else ctx.raw_value
    return AnomalyEvent(
        event_id=str(uuid.uuid4()),
        entity_id=ensemble.entity_id,
        entity_type=ctx.entity_type,
        timestamp=ensemble.timestamp,
        anomaly_type=ensemble.predicted_anomaly_type,
        severity=severity.level,
        raw_value=raw,
        expected_range=ctx.expected_range,
        top_contributing_features=top,
        recommended_action=severity.recommended_action,
        severity_score=severity.score,
        ensemble_score=ensemble.ensemble_score,
        confidence=ensemble.confidence,
        business_impact=severity.business_impact,
        individual_scores=dict(ensemble.individual_scores),
    )


# --------------------------------------------------------------------------- #
# De-duplication and escalation
# --------------------------------------------------------------------------- #
@dataclass
class _Episode:
    """One uninterrupted run of anomalies of an entity."""

    first_seen: datetime
    last_seen: datetime
    last_emit: datetime
    last_level: Severity
    emitted: AnomalyEvent
    count: int = 1
    suppressed: int = 0
    escalation_steps: int = 0


class AlertDeduplicator:
    """Suppress repeated alerts of one entity and escalate persistent anomalies.

    An *episode* starts with the first anomalous event of an entity and continues while further
    anomalous events arrive less than ``episode_gap_s`` apart. Within an episode:

    * an event within ``window_s`` of the last emitted alert is a **duplicate** and is suppressed
      (it only bumps ``occurrence_count`` / ``last_seen`` of the alert already emitted, so a stored
      alert shows how often it fired);
    * an event **more severe** than the last emitted alert is never suppressed
      (``allow_worsening``): a WARNING at 10:00 must not hide an EMERGENCY at 10:01;
    * once the episode is older than ``escalation_after_s`` (10 minutes by default) the next event
      is emitted as an **escalation**: a copy one severity level higher (capped at EMERGENCY, then
      no further escalation alert), and again after every further ``escalation_after_s``;
    * after ``window_s`` the next event is emitted again as a reminder.

    Everything runs on event time, not wall-clock time, so it behaves identically when the
    simulator runs at 10x speed or a topic is replayed. Late events (the aggregator tolerates 30 s)
    never reopen a window. Thread-safe.
    """

    def __init__(
        self,
        window_s: float = 300.0,
        escalation_after_s: float = 600.0,
        *,
        episode_gap_s: float | None = None,
        allow_worsening: bool = True,
        max_entities: int = 100_000,
    ) -> None:
        """Create the deduplicator.

        Args:
            window_s: Duplicate-suppression window; ``0`` disables suppression.
            escalation_after_s: Episode age that triggers an escalation; ``0`` disables it.
            episode_gap_s: Silence that ends an episode (default: ``window_s``).
            allow_worsening: Let a more severe alert through inside the window.
            max_entities: Cap on tracked episodes (the least recently active are evicted).

        Raises:
            ValueError: A negative duration or a non-positive cap.
        """
        if (
            window_s < 0
            or escalation_after_s < 0
            or (episode_gap_s is not None and episode_gap_s < 0)
        ):
            raise ValueError("durations must be non-negative")
        if max_entities < 1:
            raise ValueError("max_entities must be positive")
        self.window_s = float(window_s)
        self.escalation_after_s = float(escalation_after_s)
        self.episode_gap_s = float(window_s if episode_gap_s is None else episode_gap_s)
        self.allow_worsening = allow_worsening
        self.max_entities = max_entities
        self._episodes: OrderedDict[str, _Episode] = OrderedDict()
        self._lock = threading.Lock()
        self._counts = {"emitted": 0, "suppressed": 0, "escalated": 0, "episodes": 0}
        self._latest: datetime | None = None

    @classmethod
    def from_config(cls, cfg: AlertsConfig, **overrides: Any) -> AlertDeduplicator:
        """Create the deduplicator with the windows of ``config.alerts``."""
        return cls(cfg.dedup_window_s, cfg.escalation_after_s, **overrides)

    # ------------------------------------------------------------------ #
    def process(self, event: AnomalyEvent) -> AnomalyEvent | None:
        """Decide what to do with a new alert.

        Returns:
            The alert to publish (the event itself, or an escalated copy with a new ``event_id``),
            or ``None`` when it is a duplicate.
        """
        now = event.timestamp
        with self._lock:
            if self._latest is None or now > self._latest:
                self._latest = now
            episode = self._episodes.get(event.entity_id)
            if episode is None or (now - episode.last_seen).total_seconds() > self.episode_gap_s:
                return self._open(event)
            now = max(now, episode.last_seen)
            episode.last_seen = now
            episode.count += 1
            self._episodes.move_to_end(event.entity_id)
            age = (now - episode.first_seen).total_seconds()
            if self.escalation_after_s > 0:
                due = int(age // self.escalation_after_s)
                if due > episode.escalation_steps:
                    episode.escalation_steps = due
                    base = max(event.severity, episode.last_level)
                    if base < Severity.EMERGENCY:
                        return self._escalate(event, episode, base, now)
            if self.allow_worsening and event.severity > episode.last_level:
                episode.last_level = event.severity
                return self._emit(event, episode, now)
            if (now - episode.last_emit).total_seconds() >= self.window_s:
                return self._emit(event, episode, now)
            episode.suppressed += 1
            self._counts["suppressed"] += 1
            episode.emitted.occurrence_count = episode.count
            episode.emitted.last_seen = now
            return None

    def _open(self, event: AnomalyEvent) -> AnomalyEvent:
        self._episodes[event.entity_id] = _Episode(
            event.timestamp, event.timestamp, event.timestamp, event.severity, event
        )
        self._episodes.move_to_end(event.entity_id)
        self._counts["episodes"] += 1
        self._counts["emitted"] += 1
        while len(self._episodes) > self.max_entities:
            self._episodes.popitem(last=False)
        return event

    def _emit(self, event: AnomalyEvent, episode: _Episode, now: datetime) -> AnomalyEvent:
        episode.last_emit = now
        episode.last_level = max(episode.last_level, event.severity)
        episode.emitted = event
        event.occurrence_count = episode.count
        event.first_seen = episode.first_seen
        event.last_seen = now
        self._counts["emitted"] += 1
        return event

    def _escalate(
        self, event: AnomalyEvent, episode: _Episode, base: Severity, now: datetime
    ) -> AnomalyEvent:
        minutes = (now - episode.first_seen).total_seconds() / 60
        upgraded = replace(
            event,
            event_id=str(uuid.uuid4()),
            severity=base.escalated(),
            escalated_from=base,
            first_seen=episode.first_seen,
            last_seen=now,
            occurrence_count=episode.count,
            recommended_action=f"ESCALATED after {minutes:.0f} min "
            f"without recovery. {event.recommended_action}",
        )
        episode.last_emit = now
        episode.last_level = upgraded.severity
        episode.emitted = upgraded
        self._counts["emitted"] += 1
        self._counts["escalated"] += 1
        return upgraded

    # ------------------------------------------------------------------ #
    def resolve(self, entity_id: str) -> bool:
        """Close an entity's episode (an analyst resolved it); the next anomaly alerts afresh.

        Returns:
            Whether an open episode existed.
        """
        with self._lock:
            episode = self._episodes.pop(entity_id, None)
        if episode is None:
            return False
        episode.emitted.is_resolved = True
        return True

    def purge(self, now: datetime | None = None) -> int:
        """Forget episodes silent for longer than the episode gap (dropped-out sensors).

        Args:
            now: Reference event time; defaults to the latest event time seen.

        Returns:
            How many episodes were removed.
        """
        with self._lock:
            reference = _aware(now) if now is not None else self._latest
            if reference is None:
                return 0
            stale = [
                key
                for key, ep in self._episodes.items()
                if (reference - ep.last_seen).total_seconds()
                > max(self.episode_gap_s, self.escalation_after_s)
            ]
            for key in stale:
                del self._episodes[key]
            return len(stale)

    def active_entities(self) -> list[str]:
        """Entities with an open episode, least recently active first."""
        with self._lock:
            return list(self._episodes)

    def stats(self) -> dict[str, int]:
        """Counters: ``emitted``, ``suppressed``, ``escalated``, ``episodes`` and ``active``."""
        with self._lock:
            return {**self._counts, "active": len(self._episodes)}


__all__ = [
    "BOUNDARY",
    "DEFAULT_WEIGHTS",
    "IMPACT_COLD_STORAGE",
    "IMPACT_DEFAULT",
    "IMPACT_INVENTORY",
    "IMPACT_VEHICLE",
    "ISOLATION_FOREST",
    "LSTM_AUTOENCODER",
    "LSTM_CLASSIFIER",
    "MEMBERS",
    "META_FEATURE_NAMES",
    "MODEL_TYPE_META",
    "STATISTICAL",
    "AlertDeduplicator",
    "AnomalyEvent",
    "CrossValidationReport",
    "EnsembleAnomalyScore",
    "EnsembleDetector",
    "EntityContext",
    "MemberScoreTable",
    "MetaLearner",
    "Severity",
    "SeverityResult",
    "SeverityScorer",
    "best_f1_threshold",
    "binary_metrics",
    "build_anomaly_event",
    "collect_member_scores",
    "holdout_mask",
    "is_usable",
    "meta_feature_row",
    "statistical_to_score",
    "to_boundary_scale",
]
