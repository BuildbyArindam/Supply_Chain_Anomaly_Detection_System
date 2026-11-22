"""Isolation Forest anomaly detector for the engineered feature vectors (Day 9).

Isolation Forest isolates a sample with random axis-parallel splits; anomalies are few and
different, so they need fewer splits (a shorter path) than normal points. It is unsupervised,
needs no scaling, and scores a whole feature vector at once, which is what makes it the
"tabular / mixed features" member of the ensemble next to the per-signal statistical
detectors (Day 8) and the sequence models (Days 10-11).

Contents
--------
* :class:`IsolationForestDetector` -- wraps :class:`sklearn.ensemble.IsolationForest` with a
  warm-up fit, a sample reservoir and periodic refits for streaming use.
* :class:`HyperparameterTuner` -- Optuna search (TPE sampler, median pruner) that maximises
  F1 on held-out labeled data and writes ``configs/isolation_forest_best_params.json``.
* :class:`IsolationForestExplainer` -- SHAP values and the top features behind one alert.
* :class:`Reservoir` -- the fixed-size sample buffer used for refits.

Score convention
----------------
``details["raw_score"]`` is sklearn's ``decision_function``: **negative means more anomalous**
and ``0`` is the decision boundary implied by ``contamination``. The reported ``score`` maps it
to ``0..1`` with a logistic curve scaled by the spread of the training scores, so ``0.5`` is the
boundary and ``score >= 0.5`` is an anomaly. That matches the Day 8 convention, so the two
detectors' scores can be compared and stacked directly.

Streaming behaviour
-------------------
1. **Warm-up.** Until ``warmup_samples`` vectors have been seen the detector buffers them in a
   uniform reservoir and reports ``score=0`` with ``details["warming_up"] = True``. Then it fits.
   (Or call :meth:`IsolationForestDetector.fit` with an hour of data yourself.)
2. **Scoring.** Every vector is scored with the current model.
3. **Learning.** Scored vectors enter the reservoir (``reservoir_size`` rows) with a
   recency bias, so it always reflects the recent past. Flagged vectors are kept *in* by
   default: the contamination quantile that sets the decision boundary assumes the reservoir
   holds the stream's natural share of anomalies. ``exclude_flagged=True`` keeps freshly flagged
   vectors out (like the Day 8 baselines freeze while an entity is flagged), but with a
   quantile-based boundary this feeds back on itself: every refit sees a truncated
   distribution, cuts deeper into normal data, and flags more (measured on the simulated
   warehouse stream: flag rate 10% -> 19% over four hours, versus a stable 2.2% without
   exclusion). Only use it together with a much lower ``contamination``.
4. **Refit.** Every ``refit_every`` scored vectors the forest is refit on the reservoir and
   swapped in atomically.

Performance note: scikit-learn pays roughly 10-15 ms of fixed overhead per
``decision_function`` call (Python loops over the trees) plus a few tens of microseconds per
row. Measured: about 70 events/s one at a time, 9 000/s in batches of 200. Feed the detector
micro-batches (:meth:`IsolationForestDetector.update_batch`, a few dozen to a few hundred
events) in the streaming engine instead of one event at a time.

Use one detector per source type: warehouse, vehicle and inventory have different feature
schemas (:func:`src.models.feature_engineering.feature_names`).
"""

from __future__ import annotations

import json
import math
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

from src.models.statistical_detector import AnomalyScore, Timestamp, parse_timestamp
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.models.feature_engineering import FeatureVector
    from src.utils.config import IsolationForestConfig

DETECTOR_NAME: Final[str] = "isolation_forest"
TUNABLE_PARAMS: Final[tuple[str, ...]] = (
    "n_estimators",
    "contamination",
    "max_features",
    "max_samples",
)
#: A forest needs at least this many rows to be meaningful.
MIN_FIT_SAMPLES: Final[int] = 32
#: Lower bound of the score scale, so a near-constant training set cannot make scores binary.
SCALE_FLOOR: Final[float] = 1e-3
#: ``IQR / 1.349`` is the standard deviation of a normal distribution.
_IQR_TO_STD: Final[float] = 1.349
_LOGIT_CLIP: Final[float] = 50.0

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def fill_values(X: np.ndarray) -> np.ndarray:
    """Return the per-feature median of the finite values (``0.0`` for an all-NaN column)."""
    finite = np.where(np.isfinite(X), X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(finite, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def impute(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Replace NaN and +-inf in ``X`` by the matching entry of ``fill`` (new array)."""
    return np.where(np.isfinite(X), X, fill)


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 of the positive class; ``0.0`` when there are no positives and no predictions."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


class Reservoir:
    """Fixed-capacity buffer of feature rows with uniform or recency-biased replacement.

    * ``recent=False`` -- *Algorithm R*: after ``n`` arrivals every one of them is in the
      buffer with probability ``capacity / n``. Used for the warm-up so the first model sees a
      fair sample of the whole warm-up period, not just its last few minutes.
    * ``recent=True`` -- *biased reservoir sampling* (Aggarwal, 2006, with
      ``lambda = 1 / capacity``): a new row always enters and replaces a random slot, so a row
      stays for ``capacity`` arrivals on average (geometric lifetime). The buffer holds mostly
      recent rows plus a thin tail of older ones.
    """

    def __init__(self, capacity: int, n_features: int, *, seed: int | None = 0) -> None:
        """Create an empty reservoir.

        Raises:
            ValueError: ``capacity`` or ``n_features`` is not positive.
        """
        if capacity < 1 or n_features < 1:
            raise ValueError("capacity and n_features must be positive")
        self.capacity = capacity
        self.n_features = n_features
        self._rows = np.empty((capacity, n_features), dtype=np.float64)
        self._size = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def seen(self) -> int:
        """Rows offered so far (not only those kept)."""
        return self._seen

    def add(self, row: np.ndarray, *, recent: bool = False) -> None:
        """Offer one row (shape ``(n_features,)``)."""
        self._seen += 1
        if self._size < self.capacity:
            if recent and self._size and self._rng.random() < self._size / self.capacity:
                self._rows[int(self._rng.integers(self._size))] = row
            else:
                self._rows[self._size] = row
                self._size += 1
            return
        if recent:
            self._rows[int(self._rng.integers(self.capacity))] = row
        else:
            slot = int(self._rng.integers(self._seen))
            if slot < self.capacity:
                self._rows[slot] = row

    def load(self, X: np.ndarray) -> None:
        """Replace the content by a uniform sample of ``X`` (all of it if it fits)."""
        n = len(X)
        if n <= self.capacity:
            self._rows[:n] = X
            self._size = n
        else:
            keep = np.sort(self._rng.choice(n, size=self.capacity, replace=False))
            self._rows[:] = X[keep]
            self._size = self.capacity
        self._seen = n

    def data(self) -> np.ndarray:
        """Return a copy of the stored rows, shape ``(len(self), n_features)``."""
        return self._rows[: self._size].copy()

    def clear(self) -> None:
        """Remove all rows."""
        self._size = 0
        self._seen = 0


@dataclass(frozen=True, slots=True)
class _FittedModel:
    """One immutable fit; swapped in as a whole so readers never see a half-updated model."""

    estimator: IsolationForest
    fill: np.ndarray
    scale: float
    version: int
    n_train: int
    fitted_at: datetime


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class IsolationForestDetector:
    """Streaming Isolation Forest over :class:`FeaturePipeline` vectors of one source type.

    Offline use::

        detector = IsolationForestDetector(names, source_type="warehouse").fit(X_warmup)
        scores = detector.score_batch(X_test)           # 0..1, >= 0.5 is an anomaly
        flags = detector.predict(X_test)

    Streaming use (one detector per source type)::

        results = detector.update_batch(entity_ids, timestamps, X)      # list[AnomalyScore]
        result = detector.process(feature_pipeline.process("wh1-A-temp-01"))   # one event

    Thread-safe: updates are serialised by a lock, pure scoring is lock-free on an immutable
    snapshot. A refit (a fraction of a second for 5 000 rows) blocks other updates meanwhile.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        source_type: str = "warehouse",
        n_estimators: int = 200,
        contamination: float = 0.01,
        max_features: float = 1.0,
        max_samples: float = 1.0,
        warmup_samples: int | None = None,
        reservoir_size: int = 5000,
        refit_every: int = 1000,
        exclude_flagged: bool = False,
        max_flag_run: int = 120,
        random_state: int = 42,
        n_jobs: int | None = 1,
    ) -> None:
        """Create an unfitted detector.

        Args:
            feature_names: Column names of the vectors (defines the schema and its order).
            source_type: ``warehouse``, ``vehicle`` or ``inventory`` (reported in details).
            n_estimators: Trees in the forest.
            contamination: Expected share of anomalies in the training data; sets the decision
                boundary (this share of the training scores falls below it).
            max_features: Share of features each tree may split on, ``(0, 1]``.
            max_samples: Share of the training rows drawn for each tree, ``(0, 1]``.
            warmup_samples: Vectors to collect before the first fit; defaults to
                ``reservoir_size``. Use :meth:`from_config` to derive it from
                ``warmup_seconds``.
            reservoir_size: Rows kept for (re)fitting.
            refit_every: Scored vectors between refits.
            exclude_flagged: Keep freshly flagged vectors out of the reservoir. Off by default:
                with the contamination quantile it makes the flag rate drift upwards (see the
                module docs).
            max_flag_run: Consecutive flags of one entity after which its vectors are admitted
                to the reservoir again.
            random_state: Seed of the forest and the reservoir.
            n_jobs: Threads for fitting and scoring (``1`` keeps results reproducible and the
                service single-threaded; ``-1`` uses all cores).

        Raises:
            ValueError: An argument is out of range or ``feature_names`` is empty / duplicated.
        """
        names = tuple(feature_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique")
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if not 0 < max_features <= 1 or not 0 < max_samples <= 1:
            raise ValueError("max_features and max_samples must be in (0, 1]")
        warmup = reservoir_size if warmup_samples is None else warmup_samples
        if n_estimators < 1 or refit_every < 1 or max_flag_run < 1:
            raise ValueError("n_estimators, refit_every and max_flag_run must be positive")
        if reservoir_size < MIN_FIT_SAMPLES or warmup < MIN_FIT_SAMPLES:
            raise ValueError(f"reservoir_size and warmup_samples must be >= {MIN_FIT_SAMPLES}")
        self.feature_names = names
        self.source_type = source_type
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.max_samples = max_samples
        self.warmup_samples = warmup
        self.reservoir_size = reservoir_size
        self.refit_every = refit_every
        self.exclude_flagged = exclude_flagged
        self.max_flag_run = max_flag_run
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._raw_idx = next((i for i, n in enumerate(names) if n.endswith("__last")), None)
        self._reservoir = Reservoir(reservoir_size, len(names), seed=random_state)
        self._model: _FittedModel | None = None
        self._n_warm = 0
        self._n_since_fit = 0
        self._n_scored = 0
        self._flag_runs: dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: IsolationForestConfig,
        feature_names: Sequence[str],
        *,
        source_type: str = "warehouse",
        events_per_second: float | None = None,
        use_best_params: bool = True,
        best_params_path: str | Path | None = None,
    ) -> IsolationForestDetector:
        """Create a detector from ``config.models.isolation_forest``.

        If ``use_best_params`` and the tuning file (``cfg.best_params_path``) exists, its
        ``best_params`` override ``n_estimators``, ``contamination``, ``max_features`` and
        ``max_samples`` of the config.

        Args:
            cfg: The ``models.isolation_forest`` config section.
            feature_names: Column names of the vectors.
            source_type: Source type this detector serves.
            events_per_second: Event rate of the source; with it the warm-up length is
                ``cfg.warmup_seconds * events_per_second`` vectors (capped to what is useful),
                otherwise ``reservoir_size``.
            use_best_params: Apply the tuned hyperparameters when the file exists.
            best_params_path: Overrides ``cfg.best_params_path``.

        Raises:
            ValueError: The tuning file exists but is not valid.
        """
        params: dict[str, Any] = {
            "n_estimators": cfg.n_estimators,
            "contamination": cfg.contamination,
            "max_features": cfg.max_features,
            "max_samples": cfg.max_samples,
        }
        if use_best_params:
            params.update(load_best_params(best_params_path or cfg.best_params_path, source_type))
        warmup = None
        if events_per_second is not None and events_per_second > 0:
            warmup = max(int(cfg.warmup_seconds * events_per_second), MIN_FIT_SAMPLES)
        return cls(
            feature_names,
            source_type=source_type,
            warmup_samples=warmup,
            reservoir_size=cfg.reservoir_size,
            refit_every=cfg.refit_every,
            random_state=cfg.random_state,
            **params,
        )

    def get_params(self) -> dict[str, Any]:
        """Return the hyperparameters (for the model registry and for logging)."""
        return {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "warmup_samples": self.warmup_samples,
            "reservoir_size": self.reservoir_size,
            "refit_every": self.refit_every,
            "exclude_flagged": self.exclude_flagged,
            "max_flag_run": self.max_flag_run,
            "random_state": self.random_state,
            "source_type": self.source_type,
        }

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        """Whether a model exists (after the warm-up or an explicit :meth:`fit`)."""
        return self._model is not None

    @property
    def model_version(self) -> int:
        """Number of fits so far (``0`` before the first)."""
        model = self._model
        return model.version if model else 0

    @property
    def n_scored(self) -> int:
        """Vectors scored by :meth:`update_batch` since the first fit."""
        return self._n_scored

    @property
    def warmup_progress(self) -> float:
        """Share of the warm-up completed, ``0..1`` (``1.0`` once fitted)."""
        return 1.0 if self.is_fitted else min(self._n_warm / self.warmup_samples, 1.0)

    @property
    def estimator(self) -> IsolationForest:
        """The fitted scikit-learn forest (for the explainer and diagnostics).

        Raises:
            NotFittedError: No model yet.
        """
        return self._require().estimator

    def impute(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` as the forest sees it: NaN / inf replaced by the training medians.

        Raises:
            NotFittedError: No model yet.
            ValueError: ``X`` does not have one column per feature.
        """
        model = self._require()
        return impute(self._as_matrix(X), model.fill)

    def _require(self) -> _FittedModel:
        model = self._model
        if model is None:
            raise NotFittedError("IsolationForestDetector is not fitted yet (still warming up)")
        return model

    def _as_matrix(self, X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} feature columns, got shape {arr.shape}"
            )
        return arr

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> Self:
        """Fit on warm-up data (typically one hour of feature vectors) and reset the stream state.

        The forest is trained on at most ``reservoir_size`` rows, drawn uniformly from ``X``,
        so offline fits and the periodic refits (and therefore the tuned hyperparameters) see
        the same amount of data.

        Raises:
            ValueError: Wrong number of columns or fewer than :data:`MIN_FIT_SAMPLES` rows.
        """
        matrix = self._as_matrix(X)
        if len(matrix) < MIN_FIT_SAMPLES:
            raise ValueError(f"need at least {MIN_FIT_SAMPLES} rows to fit, got {len(matrix)}")
        with self._lock:
            self._reservoir.load(matrix)
            self._flag_runs.clear()
            self._n_warm = len(matrix)
            self._refit()
        return self

    def _refit(self) -> None:
        """Fit a new forest on the reservoir and swap it in. Caller holds the lock."""
        data = self._reservoir.data()
        fill = fill_values(data)
        filled = impute(data, fill)
        estimator = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(filled)
        decision = estimator.decision_function(filled)
        q25, q75 = np.percentile(decision, [25, 75])
        scale = max(float(q75 - q25) / _IQR_TO_STD, SCALE_FLOOR)
        version = (self._model.version if self._model else 0) + 1
        self._model = _FittedModel(estimator, fill, scale, version, len(data), datetime.now(UTC))
        self._n_since_fit = 0
        _log.info(
            "isolation_forest_fitted",
            source_type=self.source_type,
            version=version,
            n_train=len(data),
            score_scale=round(scale, 5),
        )

    # ------------------------------------------------------------------ #
    # Pure scoring
    # ------------------------------------------------------------------ #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores, **negative = more anomalous**, ``0`` = decision boundary.

        Raises:
            NotFittedError: No model yet.
            ValueError: Wrong number of columns.
        """
        model = self._require()
        return self._decision(model, self._as_matrix(X))

    @staticmethod
    def _decision(model: _FittedModel, X: np.ndarray) -> np.ndarray:
        return np.asarray(model.estimator.decision_function(impute(X, model.fill)))

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Return normalised anomaly scores in ``[0, 1]``; ``0.5`` is the boundary.

        Raises:
            NotFittedError: No model yet.
            ValueError: Wrong number of columns.
        """
        model = self._require()
        return self._normalise(model, self._decision(model, self._as_matrix(X)))

    @staticmethod
    def _normalise(model: _FittedModel, decision: np.ndarray) -> np.ndarray:
        return np.asarray(expit(-np.clip(decision / model.scale, -_LOGIT_CLIP, _LOGIT_CLIP)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a boolean array, ``True`` where the vector is anomalous (raw score < 0)."""
        return self.decision_function(X) < 0.0

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def update(
        self,
        entity_id: str,
        timestamp: Timestamp,
        x: np.ndarray,
        *,
        learn: bool = True,
    ) -> AnomalyScore:
        """Score one vector and (unless ``learn`` is false) learn from it.

        Convenient but slow (about 14 ms per call, see the module docs): prefer
        :meth:`update_batch` on a hot path.
        """
        return self.update_batch(
            [entity_id], [timestamp], np.asarray(x).reshape(1, -1), learn=learn
        )[0]

    def process(self, vector: FeatureVector, *, learn: bool = True) -> AnomalyScore:
        """Score a :class:`FeatureVector` produced by ``FeaturePipeline.process``.

        Raises:
            ValueError: The vector's feature names differ from this detector's schema.
        """
        return self.process_many([vector], learn=learn)[0]

    def process_many(
        self, vectors: Sequence[FeatureVector], *, learn: bool = True
    ) -> list[AnomalyScore]:
        """Micro-batch version of :meth:`process`.

        Raises:
            ValueError: A vector's feature names differ from this detector's schema.
        """
        for vector in vectors:
            if tuple(vector.names) != self.feature_names:
                raise ValueError(
                    f"feature schema mismatch for {vector.entity_id!r}: expected "
                    f"{len(self.feature_names)} features of source {self.source_type!r}, got "
                    f"{len(vector.names)} of source {vector.source_type!r}"
                )
        if not vectors:
            return []
        return self.update_batch(
            [v.entity_id for v in vectors],
            [v.timestamp for v in vectors],
            np.vstack([v.raw for v in vectors]),
            learn=learn,
        )

    def update_batch(
        self,
        entity_ids: Sequence[str],
        timestamps: Sequence[Timestamp],
        X: np.ndarray,
        *,
        learn: bool = True,
    ) -> list[AnomalyScore]:
        """Score a micro-batch in order and learn from it (warm-up, reservoir, refits).

        The batch is cut wherever the model changes (end of the warm-up, a refit), so the
        result for each row is the one a one-by-one stream would give.

        Args:
            entity_ids: Entity of each row.
            timestamps: Event time of each row (datetime, ISO string or epoch seconds).
            X: Raw feature vectors, shape ``(n, n_features)``.
            learn: ``False`` only scores: the reservoir, counters and model stay untouched
                (before the first fit it returns warm-up results).

        Returns:
            One :class:`AnomalyScore` per row. While warming up: ``score=0``,
            ``is_anomaly=False``, ``details["warming_up"] = True``.

        Raises:
            ValueError: Mismatched lengths or wrong number of columns.
        """
        matrix = self._as_matrix(X)
        n = len(matrix)
        if len(entity_ids) != n or len(timestamps) != n:
            raise ValueError("entity_ids, timestamps and X must have the same length")
        stamps = [parse_timestamp(t) for t in timestamps]
        results: list[AnomalyScore] = []
        with self._lock:
            start = 0
            while start < n:
                model = self._model
                if model is None:
                    take = (
                        min(n - start, self.warmup_samples - self._n_warm) if learn else n - start
                    )
                    rows = slice(start, start + take)
                    if learn:
                        for row in matrix[rows]:
                            self._reservoir.add(row, recent=False)
                        self._n_warm += take
                    results += [
                        self._warming(entity_ids[i], stamps[i], matrix[i])
                        for i in range(rows.start, rows.stop)
                    ]
                    start += take
                    if learn and self._n_warm >= self.warmup_samples:
                        self._refit()
                    continue
                take = min(n - start, self.refit_every - self._n_since_fit) if learn else n - start
                rows = slice(start, start + take)
                decision = self._decision(model, matrix[rows])
                scores = self._normalise(model, decision)
                for k, i in enumerate(range(rows.start, rows.stop)):
                    flagged = bool(decision[k] < 0.0)
                    results.append(
                        self._result(
                            entity_ids[i],
                            stamps[i],
                            matrix[i],
                            float(scores[k]),
                            float(decision[k]),
                            model,
                        )
                    )
                    if learn:
                        self._learn(entity_ids[i], matrix[i], flagged)
                if learn:
                    self._n_since_fit += take
                    self._n_scored += take
                    if self._n_since_fit >= self.refit_every:
                        self._refit()
                start += take
        return results

    def _learn(self, entity_id: str, row: np.ndarray, flagged: bool) -> None:
        """Offer a scored row to the reservoir unless it is a (still short) flagged run."""
        run = self._flag_runs.get(entity_id, 0) + 1 if flagged else 0
        self._flag_runs[entity_id] = run
        if flagged and self.exclude_flagged and run < self.max_flag_run:
            return
        self._reservoir.add(row, recent=True)

    def _raw_value(self, row: np.ndarray) -> float | None:
        if self._raw_idx is None:
            return None
        value = float(row[self._raw_idx])
        return value if math.isfinite(value) else None

    def _warming(self, entity_id: str, timestamp: datetime, row: np.ndarray) -> AnomalyScore:
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=timestamp,
            score=0.0,
            is_anomaly=False,
            anomaly_type=None,
            confidence=0.0,
            detector=DETECTOR_NAME,
            raw_value=self._raw_value(row),
            details={
                "warming_up": True,
                "source_type": self.source_type,
                "n_samples": self._n_warm,
                "warmup_samples": self.warmup_samples,
            },
        )

    def _result(
        self,
        entity_id: str,
        timestamp: datetime,
        row: np.ndarray,
        score: float,
        decision: float,
        model: _FittedModel,
    ) -> AnomalyScore:
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=timestamp,
            score=score,
            is_anomaly=decision < 0.0,
            anomaly_type=None,  # an Isolation Forest says "unusual", not which kind
            confidence=abs(2.0 * score - 1.0),
            detector=DETECTOR_NAME,
            raw_value=self._raw_value(row),
            details={
                "raw_score": decision,
                "warming_up": False,
                "source_type": self.source_type,
                "model_version": model.version,
                "score_scale": model.scale,
            },
        )

    # ------------------------------------------------------------------ #
    # Persistence (the lock cannot be pickled)
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()


# --------------------------------------------------------------------------- #
# Tuned parameters on disk
# --------------------------------------------------------------------------- #
def load_best_params(path: str | Path, source_type: str | None = None) -> dict[str, Any]:
    """Read ``best_params`` from a tuning file; ``{}`` when the file does not exist.

    Only the four tunable keys are returned. A file tuned for another ``source_type`` is still
    applied (the search space is the same) but logged as a warning.

    Raises:
        ValueError: The file is not valid JSON or has no ``best_params`` object.
    """
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        best = payload["best_params"]
        if not isinstance(best, dict):
            raise TypeError("best_params must be an object")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid Isolation Forest tuning file {target}: {exc}") from exc
    tuned_for = payload.get("source_type")
    if source_type and tuned_for and tuned_for != source_type:
        _log.warning(
            "isolation_forest_params_other_source",
            file=str(target),
            tuned_for=tuned_for,
            requested=source_type,
        )
    return {k: best[k] for k in TUNABLE_PARAMS if k in best}


# --------------------------------------------------------------------------- #
# Hyperparameter tuning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TuningResult:
    """Outcome of :meth:`HyperparameterTuner.tune`."""

    best_params: dict[str, Any]
    best_f1: float
    default_f1: float | None
    n_trials: int
    n_complete: int
    n_pruned: int
    study: Any = None  # optuna.Study

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (without the study object)."""
        return {
            "best_params": dict(self.best_params),
            "best_f1": self.best_f1,
            "default_f1": self.default_f1,
            "n_trials": self.n_trials,
            "n_complete": self.n_complete,
            "n_pruned": self.n_pruned,
        }


class HyperparameterTuner:
    """Optuna search for the Isolation Forest hyperparameters.

    * **Search space:** ``n_estimators`` 50-500, ``contamination`` 0.005-0.05 (log scale),
      ``max_features`` 0.3-1.0, ``max_samples`` 0.3-1.0.
    * **Objective:** binary F1 (any anomaly vs normal) on held-out labeled data, pooled over
      the whole validation set. The forest is fit on ``X_train`` only, unsupervised; labels are
      used for validation scoring, nothing else.
    * **Pruning:** the validation set is scored in ``n_chunks`` consecutive time blocks; after
      each block the pooled F1 so far is reported and a :class:`optuna.pruners.MedianPruner`
      can stop a trial that is behind the median of its peers.
    * The default parameters (clamped into the search space) are enqueued as the first trial,
      so the best result is never worse than not tuning.

    Keep ``X_train`` about ``reservoir_size`` rows (a uniform sample of the warm-up period):
    that is what the streaming refits use, so the tuned values transfer. Tune on validation
    data from *after* the training period and report the final numbers on a *third*, untouched
    test period.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        source_type: str = "warehouse",
        n_trials: int = 50,
        n_chunks: int = 5,
        random_state: int = 42,
        n_jobs: int | None = 1,
        default_params: Mapping[str, Any] | None = None,
        search_space: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        """Create the tuner.

        Args:
            X_train: Warm-up feature rows the forests are fit on.
            X_val: Held-out feature rows in time order.
            y_val: Boolean ground truth of ``X_val`` (``True`` = anomaly).
            source_type: Recorded in the output file.
            n_trials: Optuna trials (50 by default).
            n_chunks: Time blocks of the validation set used for pruning.
            random_state: Seed of the sampler and the forests.
            n_jobs: Threads of each forest.
            default_params: Hyperparameters of the first (enqueued) trial; ``None`` for the
                class defaults (200 trees, contamination 0.01, max_features 1.0, max_samples 1.0).
            search_space: Overrides of the ``(low, high)`` bounds by parameter name.

        Raises:
            ValueError: Inconsistent shapes, too little data, or no positive validation label.
        """
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.X_val = np.asarray(X_val, dtype=np.float64)
        self.y_val = np.asarray(y_val, dtype=bool)
        if self.X_train.ndim != 2 or self.X_val.ndim != 2:
            raise ValueError("X_train and X_val must be 2-D")
        if self.X_train.shape[1] != self.X_val.shape[1]:
            raise ValueError("X_train and X_val must have the same number of columns")
        if len(self.X_val) != len(self.y_val):
            raise ValueError("X_val and y_val must have the same length")
        if len(self.X_train) < MIN_FIT_SAMPLES:
            raise ValueError(f"need at least {MIN_FIT_SAMPLES} training rows")
        if not self.y_val.any():
            raise ValueError("y_val has no anomalies: F1 is undefined")
        if n_trials < 1 or n_chunks < 1 or n_chunks > len(self.X_val):
            raise ValueError("n_trials and n_chunks must be positive and n_chunks <= len(X_val)")
        self.source_type = source_type
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.default_params: dict[str, Any] = {
            "n_estimators": 200,
            "contamination": 0.01,
            "max_features": 1.0,
            "max_samples": 1.0,
            **(dict(default_params) if default_params else {}),
        }
        self.space: dict[str, tuple[float, float]] = {
            "n_estimators": (50, 500),
            "contamination": (0.005, 0.05),
            "max_features": (0.3, 1.0),
            "max_samples": (0.3, 1.0),
            **(dict(search_space) if search_space else {}),
        }
        edges = np.linspace(0, len(self.X_val), n_chunks + 1).astype(int)
        self._chunks = [
            (int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a
        ]
        self._fill = fill_values(self.X_train)
        self._train = impute(self.X_train, self._fill)
        self._val = impute(self.X_val, self._fill)
        self._result: TuningResult | None = None

    # ------------------------------------------------------------------ #
    def _fit(self, params: Mapping[str, Any]) -> IsolationForest:
        return IsolationForest(
            n_estimators=int(params["n_estimators"]),
            contamination=float(params["contamination"]),
            max_features=float(params["max_features"]),
            max_samples=float(params["max_samples"]),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(self._train)

    def evaluate(self, params: Mapping[str, Any]) -> float:
        """Return the validation F1 of one parameter set (full pass, no pruning)."""
        model = self._fit(params)
        return binary_f1(self.y_val, model.decision_function(self._val) < 0.0)

    def objective(self, trial: Any) -> float:
        """Optuna objective: pooled validation F1, reported after every time block."""
        import optuna

        space = self.space
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", int(space["n_estimators"][0]), int(space["n_estimators"][1])
            ),
            "contamination": trial.suggest_float(
                "contamination", *space["contamination"], log=True
            ),
            "max_features": trial.suggest_float("max_features", *space["max_features"]),
            "max_samples": trial.suggest_float("max_samples", *space["max_samples"]),
        }
        model = self._fit(params)
        tp = fp = fn = 0
        f1 = 0.0
        for step, (lo, hi) in enumerate(self._chunks):
            pred = model.decision_function(self._val[lo:hi]) < 0.0
            truth = self.y_val[lo:hi]
            tp += int(np.sum(truth & pred))
            fp += int(np.sum(~truth & pred))
            fn += int(np.sum(truth & ~pred))
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom else 0.0
            trial.report(f1, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return f1

    def tune(self, *, verbosity: int | None = None) -> TuningResult:
        """Run the study and return the best parameters.

        Args:
            verbosity: Optuna log level during the run (``optuna.logging.WARNING`` by default);
                the previous level is restored afterwards.
        """
        import optuna

        previous = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING if verbosity is None else verbosity)
        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            )
            # Defaults go in first, clamped into the (possibly narrowed) search space.
            first_trial = {
                k: min(max(self.default_params[k], self.space[k][0]), self.space[k][1])
                for k in TUNABLE_PARAMS
            }
            first_trial["n_estimators"] = int(first_trial["n_estimators"])
            study.enqueue_trial(first_trial)
            study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
        finally:
            optuna.logging.set_verbosity(previous)
        states = [t.state for t in study.trials]
        complete = sum(s == optuna.trial.TrialState.COMPLETE for s in states)
        pruned = sum(s == optuna.trial.TrialState.PRUNED for s in states)
        first = study.trials[0]
        default_f1 = first.value if first.state == optuna.trial.TrialState.COMPLETE else None
        best = {k: study.best_params[k] for k in TUNABLE_PARAMS}
        self._result = TuningResult(
            best_params=best,
            best_f1=float(study.best_value),
            default_f1=None if default_f1 is None else float(default_f1),
            n_trials=len(study.trials),
            n_complete=int(complete),
            n_pruned=int(pruned),
            study=study,
        )
        _log.info(
            "isolation_forest_tuned",
            **{k: v for k, v in self._result.to_dict().items() if k != "best_params"},
            **best,
        )
        return self._result

    def save_best_params(
        self,
        path: str | Path = "configs/isolation_forest_best_params.json",
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write the tuning result as JSON (atomically) and return its path.

        Raises:
            RuntimeError: :meth:`tune` has not been run.
        """
        if self._result is None:
            raise RuntimeError("run tune() before save_best_params()")
        payload = {
            "source_type": self.source_type,
            "created_at": datetime.now(UTC).isoformat(),
            "objective": "binary F1 (any anomaly vs normal) on held-out validation data",
            "n_train_rows": len(self.X_train),
            "n_val_rows": len(self.X_val),
            "val_anomaly_rate": round(float(self.y_val.mean()), 6),
            **self._result.to_dict(),
            **(dict(extra) if extra else {}),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's share in an anomaly score.

    Attributes:
        feature: Feature name.
        shap_value: SHAP value in units of expected isolation path length; **negative values
            shorten the path, i.e. make the sample look more anomalous**.
        feature_value: The sample's raw value of the feature (``None`` if unknown).
    """

    feature: str
    shap_value: float
    feature_value: float | None


class IsolationForestExplainer:
    """SHAP explanations for an :class:`IsolationForestDetector`.

    Uses :class:`shap.TreeExplainer` (exact tree SHAP, no background data needed). The values
    explain the forest's expected isolation path length: a **negative** SHAP value pushes a
    sample towards a shorter path, which is the anomalous direction. The explainer follows the
    detector: after a refit it rebuilds itself on the new forest.

    Imputed values: the SHAP values are computed on the detector's view of the sample (NaN
    replaced by the training median), exactly what was scored.
    """

    def __init__(self, detector: IsolationForestDetector) -> None:
        """Create the explainer for a fitted (or later-fitted) detector."""
        self.detector = detector
        self._explainer: Any = None
        self._built_for: int | None = None
        self._lock = threading.Lock()

    def _tree_explainer(self) -> Any:
        import shap

        version = self.detector.model_version
        with self._lock:
            if self._explainer is None or self._built_for != version:
                self._explainer = shap.TreeExplainer(self.detector.estimator)
                self._built_for = version
            return self._explainer

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values for a batch, shape ``(n, n_features)``.

        Raises:
            NotFittedError: The detector has no model yet.
            ValueError: ``X`` does not have one column per feature.
        """
        explainer = self._tree_explainer()
        values = explainer.shap_values(self.detector.impute(X), check_additivity=False)
        return np.asarray(values, dtype=np.float64)

    def expected_value(self) -> float:
        """Return the base value (the forest's average expected path length)."""
        return float(np.ravel(self._tree_explainer().expected_value)[0])

    def get_top_features(
        self,
        sample: np.ndarray,
        n: int = 5,
        *,
        only_anomalous: bool = True,
    ) -> list[FeatureContribution]:
        """Return the features that make ``sample`` look most anomalous.

        Args:
            sample: One feature vector, shape ``(n_features,)``.
            n: How many features to return.
            only_anomalous: Keep only features with a negative SHAP value (those that push
                towards "anomaly"). ``False`` returns the ``n`` most negative regardless of sign.

        Returns:
            Contributions sorted from most to least anomalous (most negative SHAP first); fewer
            than ``n`` when ``only_anomalous`` and fewer features push towards "anomaly".

        Raises:
            ValueError: ``n`` is not positive or ``sample`` is not a single vector.
        """
        if n < 1:
            raise ValueError("n must be positive")
        row = np.asarray(sample, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError("sample must be a single 1-D feature vector")
        values = self.compute_shap_values(row.reshape(1, -1))[0]
        order = np.argsort(values)  # most negative first
        picked: list[FeatureContribution] = []
        for idx in order[:n]:
            if only_anomalous and values[idx] >= 0.0:
                break
            raw = float(row[idx])
            picked.append(
                FeatureContribution(
                    self.detector.feature_names[idx],
                    float(values[idx]),
                    raw if math.isfinite(raw) else None,
                )
            )
        return picked


__all__ = [
    "DETECTOR_NAME",
    "FeatureContribution",
    "HyperparameterTuner",
    "IsolationForestDetector",
    "IsolationForestExplainer",
    "Reservoir",
    "TuningResult",
    "binary_f1",
    "fill_values",
    "impute",
    "load_best_params",
]
