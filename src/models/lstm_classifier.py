"""Supervised LSTM classifier for point and contextual anomalies (Day 11).

Day 8's statistical detectors and Day 9's Isolation Forest judge one observation at a time and
Day 10's autoencoder only learns what *normal* looks like. This model is the supervised
counterpart: it uses the simulators' ground truth to learn what a **point** spike and a
**contextual** (right value, wrong time) event look like inside a short window of engineered
features, and it says which of the two it is.

Contents
--------
* :class:`LSTMClassifier` -- 2-layer bidirectional LSTM -> additive attention over all time steps
  -> ``Linear(2H, 64) -> ReLU -> Dropout(0.3) -> Linear(64, n_classes)``; returns
  ``(logits, attention_weights)``.
* :class:`DatasetBuilder` / :class:`ClassifierData` -- labelled windows from the simulators with a
  70/15/15 split stratified by class, SMOTE on the training split only, and class weights.
* :class:`LSTMClassifierTrainer` -- class-weighted ``CrossEntropyLoss``, AdamW (weight decay
  ``1e-4``), cosine annealing, early stopping (patience 8), per-class validation F1.
* :class:`TemperatureScaler` -- post-hoc calibration: one temperature ``T`` minimising the NLL of
  the validation logits.
* :class:`LSTMClassifierDetector` -- inference wrapper with a sliding window over incoming events;
  returns the Day 8 :class:`~src.models.statistical_detector.AnomalyScore`.
* :func:`evaluate_predictions` -- per-class precision / recall / F1, confusion matrix, one-vs-rest
  ROC-AUC and average precision, NLL and expected calibration error.

What a label means
------------------
A window is labelled with the class of its **newest** row: the question the model answers is "is
the reading that just arrived normal, a point anomaly or a contextual anomaly?", given the last
``seq_len`` readings as context. Collective anomalies (slow drifts) belong to the Day 10
autoencoder and are not a class here; windows that *end* in one are dropped
(``collective="drop"``, the default) rather than labelled "normal", because teaching the model
that a drifting sensor is normal would be wrong. ``collective="normal"`` keeps them as normal.

Contextual anomalies are defined by the *hour*, so unlike the autoencoder this model keeps the
cyclical clock features. Do not pass ``columns`` that exclude them.

Leakage control
---------------
Consecutive windows of one sensor overlap in 29 of 30 rows, so a plain random split would put
near-copies of a window into train and test. The split is therefore made on **blocks** of time
per entity (default ten window-lengths), stratified on the rarest class a block contains, and
windows that straddle a block boundary are dropped (an embargo). The scaler's mean / std are fit on
all warm rows of the simulated stream, exactly as the online pipeline's running scaler would see
them; labels never enter it.

Imbalance: SMOTE *and* class weights
------------------------------------
Anomalies are a few percent of the windows. Two corrections are applied, and they are meant to
complement each other: SMOTE raises each rare class on the **training** split to
``smote_ratio`` times the majority count (default ``0.5``, not fully balanced), and the loss
weights are computed from the class frequencies the model *actually trains on* (after SMOTE), so
the same imbalance is never corrected twice. Validation and test keep the natural class
distribution, so their metrics describe deployment. SMOTE interpolates between flattened windows
of the same class; it is not applied to categorical one-hot columns specially, which is a known
approximation.

Score convention
----------------
``score = 1 - P(normal)`` from the temperature-scaled probabilities, ``is_anomaly = score >=
decision_threshold`` (0.5, the boundary the Day 8-10 detectors also use) and ``anomaly_type`` is
the more probable of the anomaly classes. ``details`` carries all class probabilities, the
argmax ``predicted_class`` and the attention weights (which time steps drove the decision).
Because the training prior is rebalanced, probabilities are *not* the deployment frequencies:
temperature scaling fixes over-confidence, not a shifted base rate, so check
:func:`expected_calibration_error` on the natural-prevalence validation set.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.feature_engineering import OnlineScaler
from src.models.lstm_autoencoder import (
    SequenceSource,
    _inference_mode,
    _make_writer,
    make_sequences,
)
from src.models.statistical_detector import AnomalyScore
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.models.model_registry import ModelRegistry, ModelVersion
    from src.models.offline_dataset import LabeledFeatureSet
    from src.utils.config import LSTMClassifierConfig, SimulationConfig

DETECTOR_NAME: Final[str] = "lstm_classifier"
MODEL_TYPE: Final[str] = "lstm_classifier"
DEFAULT_LOG_DIR: Final[str] = "runs/lstm_classifier"
DEFAULT_CLASS_NAMES: Final[tuple[str, ...]] = ("normal", "point", "contextual")
#: Index of the "normal" class. ``class_names[0]`` must be ``"normal"``.
NORMAL: Final[int] = 0
#: Share of ``seq_len`` that must be real (not padding) before a window is scored.
MIN_REAL_FRACTION: Final[float] = 0.8
_MIN_TEMPERATURE: Final[float] = 0.05
_MAX_TEMPERATURE: Final[float] = 20.0
_EPS: Final[float] = 1e-12

CollectivePolicy = Literal["drop", "normal"]
Monitor = Literal["val_macro_f1", "val_loss"]

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with attention pooling; ``(batch, seq_len, F)`` in, logits out.

    The top layer's final hidden state (forward and backward, concatenated: it has read the whole
    window) is the query of an additive attention over the outputs of **all** time steps::

        e_t = v . tanh(W h_t + U h_final),   alpha = softmax(e),   context = sum_t alpha_t h_t

    ``forward`` returns ``(logits, alpha)`` with ``logits`` of shape ``(batch, n_classes)`` and
    ``alpha`` of shape ``(batch, seq_len)`` (rows sum to one).
    """

    def __init__(
        self,
        n_features: int,
        *,
        n_classes: int = 3,
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_size: int = 64,
    ) -> None:
        """Create the model.

        Args:
            n_features: Features per time step.
            n_classes: Number of output classes.
            seq_len: Default window length (``forward`` accepts any length).
            hidden_size: Hidden size per direction of the LSTM.
            num_layers: Stacked LSTM layers.
            dropout: Dropout probability inside the classification head.
            head_size: Width of the hidden layer of the head.

        Raises:
            ValueError: An argument is out of range.
        """
        super().__init__()
        if min(n_features, seq_len, hidden_size, num_layers, head_size) < 1 or n_classes < 2:
            raise ValueError("sizes must be positive and n_classes at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.head_size = head_size

        width = 2 * hidden_size
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.attn_key = nn.Linear(width, width, bias=False)
        self.attn_query = nn.Linear(width, width)
        self.attn_score = nn.Linear(width, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(width, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, n_classes),
        )

    @classmethod
    def from_config(
        cls, cfg: LSTMClassifierConfig, n_features: int, n_classes: int | None = None
    ) -> LSTMClassifier:
        """Create the model from ``config.models.lstm_classifier``."""
        return cls(
            n_features,
            n_classes=len(cfg.class_names) if n_classes is None else n_classes,
            seq_len=cfg.seq_len,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the constructor arguments (stored in checkpoints)."""
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "head_size": self.head_size,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, attention_weights)`` for ``x`` of shape ``(batch, seq_len, F)``."""
        if x.ndim != 3 or x.shape[-1] != self.n_features:
            raise ValueError(
                f"expected input of shape (batch, seq_len, {self.n_features}), got {tuple(x.shape)}"
            )
        outputs, (h_n, _) = self.lstm(x)  # outputs (B, T, 2H); h_n (2 * layers, B, H)
        final = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # top layer: forward, backward
        energy = self.attn_score(
            torch.tanh(self.attn_key(outputs) + self.attn_query(final).unsqueeze(1))
        ).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        logits: torch.Tensor = self.head(context)
        return logits, weights


@torch.no_grad()
def predict_logits(
    model: LSTMClassifier, x: np.ndarray | torch.Tensor, *, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model in eval mode (dropout off) and return ``(logits, attention)`` as arrays.

    Args:
        model: The classifier (its train / eval mode is restored afterwards).
        x: Windows, shape ``(n, seq_len, n_features)`` (or one window ``(seq_len, n_features)``).
        batch_size: Windows per forward call.

    Raises:
        ValueError: The shape is wrong or ``x`` is not finite.
    """
    tensor = torch.as_tensor(x, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected (n, seq_len, n_features), got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("windows contain NaN or infinite values (fill gaps first)")
    device = next(model.parameters()).device
    logits: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    with _inference_mode(model, mc_dropout=False):
        for start in range(0, len(tensor), batch_size):
            out, weights = model(tensor[start : start + batch_size].to(device))
            logits.append(out.cpu().numpy())
            attention.append(weights.cpu().numpy())
    if not logits:
        return np.empty((0, model.n_classes), np.float32), np.empty(
            (0, tensor.shape[1]), np.float32
        )
    return np.concatenate(logits), np.concatenate(attention)


# --------------------------------------------------------------------------- #
# Labels, SMOTE, splitting
# --------------------------------------------------------------------------- #
def _check_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(class_names)
    if len(names) < 2 or names[NORMAL] != "normal" or len(set(names)) != len(names):
        raise ValueError(f"class_names must be unique and start with 'normal', got {names}")
    return names


def encode_labels(
    y_last: np.ndarray,
    type_last: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    *,
    collective: CollectivePolicy = "drop",
) -> np.ndarray:
    """Map the ground truth of each window's newest row to a class index.

    Args:
        y_last: ``True`` where the newest row is an anomaly.
        type_last: Anomaly type of the newest row (``None`` for a normal row).
        class_names: Classes; the first must be ``"normal"``.
        collective: What to do with a ``collective`` row when it is not a class: ``"drop"``
            marks it ``-1`` (the caller removes it), ``"normal"`` labels it normal.

    Returns:
        ``int64`` labels; ``-1`` marks windows to drop.

    Raises:
        ValueError: ``class_names`` is malformed or an anomaly type is unknown.
    """
    names = _check_class_names(class_names)
    index = {name: i for i, name in enumerate(names)}
    kinds = np.asarray(
        [str(kind) if flag else "normal" for flag, kind in zip(y_last, type_last, strict=True)],
        dtype=object,
    )
    labels = np.zeros(len(kinds), dtype=np.int64)
    for kind in np.unique(kinds.astype(str)):
        mask = kinds == kind
        if kind in index:
            labels[mask] = index[kind]
        elif kind == "collective":
            labels[mask] = -1 if collective == "drop" else NORMAL
        else:
            raise ValueError(f"unknown anomaly type {kind!r}; classes are {names}")
    return labels


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Loss weights inversely proportional to class frequency: ``n / (n_classes * count_c)``.

    Raises:
        ValueError: A class has no samples (its weight would be infinite).
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(
            f"classes without training samples: {np.flatnonzero(counts == 0).tolist()}"
        )
    return counts.sum() / (n_classes * counts)


class Oversampled(NamedTuple):
    """Result of :func:`smote_oversample`: originals first, synthetic samples after."""

    x: np.ndarray
    y: np.ndarray
    synthetic: np.ndarray


def smote_oversample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ratio: float = 0.5,
    k_neighbors: int = 5,
    seed: int = 0,
) -> Oversampled:
    """SMOTE: synthesise minority samples by interpolating towards a nearest same-class neighbour.

    Each class with fewer than ``ratio * (majority count)`` samples is raised to that count with
    ``x_new = x_i + u * (x_j - x_i)``, ``u ~ U(0, 1)``, ``x_j`` one of the ``k`` nearest
    neighbours of ``x_i`` inside the class. (Implemented here, not imported from imbalanced-learn,
    to avoid a dependency that has to track every scikit-learn release.)

    Args:
        x: Samples, shape ``(n, d)`` -- flatten windows first.
        y: Integer class labels, shape ``(n,)``.
        ratio: Target share of the majority count, in ``[0, 1]`` (``0`` disables, ``1`` balances).
        k_neighbors: Neighbourhood size (reduced for tiny classes).
        seed: Seed of the random choices.

    Returns:
        The original samples followed by the synthetic ones, and a mask of the synthetic ones.
        A class with a single sample is duplicated, since there is nobody to interpolate with.

    Raises:
        ValueError: Wrong shapes, ``ratio`` outside ``[0, 1]`` or ``k_neighbors < 1``.
    """
    data = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y)
    if data.ndim != 2 or labels.shape != (len(data),):
        raise ValueError("x must be (n, d) and y must be (n,)")
    if not 0.0 <= ratio <= 1.0 or k_neighbors < 1:
        raise ValueError("ratio must be in [0, 1] and k_neighbors at least 1")
    classes, counts = np.unique(labels, return_counts=True)
    target = int(round(ratio * counts.max())) if len(counts) else 0
    rng = np.random.default_rng(seed)
    parts_x, parts_y = [data], [labels]
    n_synthetic = 0
    for cls, count in zip(classes, counts, strict=True):
        need = target - int(count)
        if need <= 0:
            continue
        members = data[labels == cls]
        if count < 2:
            synth = np.repeat(members, need, axis=0)
        else:
            k = min(k_neighbors, int(count) - 1)
            neighbours = (
                NearestNeighbors(n_neighbors=k + 1)
                .fit(members)
                .kneighbors(members, return_distance=False)[:, 1:]
            )
            base = rng.integers(0, count, need)
            pick = neighbours[base, rng.integers(0, k, need)]
            gap = rng.random((need, 1), dtype=np.float32)
            synth = members[base] + gap * (members[pick] - members[base])
        parts_x.append(synth.astype(np.float32, copy=False))
        parts_y.append(np.full(need, cls, dtype=labels.dtype))
        n_synthetic += need
    synthetic = np.zeros(len(data) + n_synthetic, dtype=bool)
    synthetic[len(data) :] = True
    return Oversampled(np.concatenate(parts_x), np.concatenate(parts_y), synthetic)


def group_stratified_split(
    groups: np.ndarray,
    labels: np.ndarray,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    *,
    seed: int = 0,
) -> np.ndarray:
    """Assign every sample to train (0), validation (1) or test (2), keeping groups together.

    Whole groups are assigned, stratified on the **rarest class present in the group**, so the
    class proportions of the three parts stay close to the overall ones even though overlapping
    windows never straddle a split.

    Args:
        groups: Group id per sample.
        labels: Class index per sample (``>= 0``).
        fractions: Train / validation / test shares (normalised to sum to one).
        seed: Seed of the shuffling.

    Returns:
        ``int`` array with the split of each sample.

    Raises:
        ValueError: Bad fractions, or too few groups of some class to stratify.
    """
    if len(fractions) != 3 or min(fractions) <= 0:
        raise ValueError("fractions must be three positive numbers")
    train_f, val_f, test_f = (f / sum(fractions) for f in fractions)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = int(labels.max()) + 1
    present = np.zeros((len(unique), n_classes), dtype=bool)
    present[inverse, labels] = True
    rarity = np.argsort(np.bincount(labels, minlength=n_classes), kind="stable")
    group_label = rarity[present[:, rarity].argmax(axis=1)]
    index = np.arange(len(unique))
    try:
        train_g, rest_g = train_test_split(
            index, train_size=train_f, stratify=group_label, random_state=seed
        )
        val_g, test_g = train_test_split(
            rest_g,
            train_size=val_f / (val_f + test_f),
            stratify=group_label[rest_g],
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(
            "too few groups of a class to stratify the split; simulate a longer period, "
            f"use a smaller `group_blocks`, or drop the rarest class ({exc})"
        ) from exc
    group_split = np.empty(len(unique), dtype=np.int64)
    group_split[train_g], group_split[val_g], group_split[test_g] = 0, 1, 2
    result: np.ndarray = group_split[inverse]
    return result


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory labelled windows: ``dataset[i] -> (x_i, y_i)`` with ``x_i`` ``(seq_len, F)``."""

    def __init__(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        """Wrap windows ``(n, seq_len, F)`` and integer labels ``(n,)``.

        Raises:
            ValueError: Shapes do not match.
        """
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        if self.x.ndim != 3 or self.y.shape != (len(self.x),):
            raise ValueError("x must be (n, seq_len, n_features) and y must be (n,)")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def make_loader(
    dataset: SequenceDataset,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a :class:`~torch.utils.data.DataLoader` (workers 4 by default, as specified).

    ``pin_memory=None`` pins only when CUDA is available (pinning without a GPU only warns).
    The shuffle order is seeded, so training is reproducible. Use ``num_workers=0`` in notebooks
    on Windows and in tests: with data already in memory, workers cost more than they save.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


@dataclass(frozen=True)
class LabeledSequences:
    """Windows with integer labels and where they came from.

    Attributes:
        x: ``float32`` windows ``(n, seq_len, n_features)``.
        y: Class index per window.
        entity_ids: Sensor / vehicle / SKU of each window (``"synthetic"`` for SMOTE samples).
        end_timestamps: Event time of the newest row (UTC epoch seconds, ``NaN`` for synthetic).
        synthetic: ``True`` for SMOTE-generated windows (only ever in the training split).
    """

    x: np.ndarray
    y: np.ndarray
    entity_ids: np.ndarray
    end_timestamps: np.ndarray
    synthetic: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, mask: np.ndarray) -> LabeledSequences:
        """Return the windows selected by a boolean mask or an index array."""
        return LabeledSequences(
            self.x[mask],
            self.y[mask],
            self.entity_ids[mask],
            self.end_timestamps[mask],
            self.synthetic[mask],
        )

    def class_counts(self, n_classes: int) -> np.ndarray:
        """Return the number of windows per class."""
        return np.bincount(self.y, minlength=n_classes)

    def to_dataset(self) -> SequenceDataset:
        """Wrap as a PyTorch dataset."""
        return SequenceDataset(self.x, self.y)


@dataclass(frozen=True)
class ClassifierData:
    """Everything training and evaluation need, from :meth:`DatasetBuilder.build`.

    Attributes:
        class_names: Class names; index 0 is ``"normal"``.
        feature_names: Column names of the windows.
        columns: Feature columns kept out of the pipeline vector (``None`` = all).
        scaler: The scaler the windows were standardised with (keep it next to the model).
        train: Training windows **after** SMOTE (real windows first, synthetic after).
        val: Validation windows (natural class distribution).
        test: Test windows (natural class distribution).
        class_weights: Loss weights from the class frequencies of ``train``.
        info: Bookkeeping: dropped windows, group counts, window step, parameters.
    """

    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    columns: tuple[int, ...] | None
    scaler: OnlineScaler
    train: LabeledSequences
    val: LabeledSequences
    test: LabeledSequences
    class_weights: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Features per time step."""
        return int(self.train.x.shape[2])

    @property
    def seq_len(self) -> int:
        """Window length."""
        return int(self.train.x.shape[1])

    @property
    def train_real(self) -> LabeledSequences:
        """The training windows without the SMOTE samples."""
        return self.train.subset(~self.train.synthetic)

    def summary(self) -> dict[str, dict[str, int]]:
        """Windows per class for each split (``train`` is the real part, plus ``train_smote``)."""
        k = len(self.class_names)

        def row(part: LabeledSequences) -> dict[str, int]:
            return {n: int(c) for n, c in zip(self.class_names, part.class_counts(k), strict=True)}

        return {
            "train_real": row(self.train_real),
            "train_smote": row(self.train),
            "val": row(self.val),
            "test": row(self.test),
        }


class DatasetBuilder:
    """Builds :class:`ClassifierData` from the simulators' labelled feature rows.

    Pipeline: simulate -> features (the production path, via ``simulate_labeled_features``) ->
    standardise -> sliding windows (:func:`~src.models.lstm_autoencoder.make_sequences`) -> class
    labels -> block-wise stratified 70/15/15 split -> SMOTE on train -> class weights.
    """

    def __init__(
        self,
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        source_type: str = "warehouse",
        seq_len: int = 30,
        stride: int = 5,
        split: tuple[float, float, float] = (0.70, 0.15, 0.15),
        smote_ratio: float | None = 0.5,
        smote_k: int = 5,
        collective: CollectivePolicy = "drop",
        columns: Sequence[int] | None = None,
        group_blocks: int = 10,
        seed: int = 42,
    ) -> None:
        """Create the builder.

        Args:
            class_names: Classes; the first must be ``"normal"``.
            source_type: ``warehouse``, ``vehicle`` or ``inventory``.
            seq_len: Window length (30 events: short, fast, enough for a spike).
            stride: Rows between window ends of one entity.
            split: Train / validation / test shares.
            smote_ratio: Raise rare training classes to this share of the majority count;
                ``None`` or ``0`` disables SMOTE.
            smote_k: SMOTE neighbourhood size.
            collective: How to treat windows that end in a collective anomaly (see module doc).
            columns: Feature columns to keep (``None`` = all; keep the clock features).
            group_blocks: Split-block length in window-lengths (see module doc, "Leakage").
            seed: Seed of the split and of SMOTE.

        Raises:
            ValueError: An argument is out of range.
        """
        self.class_names = _check_class_names(class_names)
        if seq_len < 2 or stride < 1 or group_blocks < 2:
            raise ValueError("seq_len must be >= 2, stride >= 1 and group_blocks >= 2")
        if smote_ratio is not None and not 0.0 <= smote_ratio <= 1.0:
            raise ValueError("smote_ratio must be in [0, 1]")
        self.source_type = source_type
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.smote_ratio = smote_ratio
        self.smote_k = smote_k
        self.collective = collective
        self.columns = None if columns is None else tuple(int(c) for c in columns)
        self.group_blocks = group_blocks
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: LSTMClassifierConfig, **overrides: Any) -> DatasetBuilder:
        """Create the builder with ``seq_len`` and ``class_names`` from the project config."""
        params: dict[str, Any] = {"class_names": cfg.class_names, "seq_len": cfg.seq_len}
        params.update(overrides)
        return cls(**params)

    def build_from_simulation(
        self,
        duration_s: float,
        *,
        start_time: datetime | None = None,
        seed: int | None = None,
        sim_config: SimulationConfig | None = None,
    ) -> ClassifierData:
        """Run the simulators and build the data (see ``simulate_labeled_features``).

        Contextual anomalies are only injected during the off-hours of the simulator, so start
        the clock a few hours before them (for example 22:00) and run long enough to cover both
        business hours and the night; otherwise "contextual" and "off-hours" are the same thing.
        """
        from src.models.offline_dataset import simulate_labeled_features

        rows = simulate_labeled_features(
            duration_s,
            start_time=start_time,
            seed=seed,
            sources=(self.source_type,),
            sim_config=sim_config,
        )[self.source_type]
        return self.build(rows)

    def build(
        self, dataset: LabeledFeatureSet, scaler: OnlineScaler | None = None
    ) -> ClassifierData:
        """Turn labelled feature rows into split, oversampled, weighted training data.

        Args:
            dataset: Output of ``simulate_labeled_features`` for ``source_type``.
            scaler: A fitted scaler; by default one is fit on all warm rows of ``dataset``.

        Raises:
            ValueError: No windows could be built, or a class is missing from a split.
        """
        names = self.class_names
        if scaler is None:
            scaler = OnlineScaler(dataset.X.shape[1])
            scaler.fit_update(dataset.X[dataset.warm])
        windows = make_sequences(
            dataset,
            scaler,
            seq_len=self.seq_len,
            stride=self.stride,
            warm_only=True,
            columns=self.columns,
        )
        if len(windows) == 0:
            raise ValueError(
                f"no windows: each entity needs more than seq_len={self.seq_len} warm rows "
                "(the aggregator needs ~15 simulated minutes to warm up first)"
            )
        labels = encode_labels(windows.y_last, windows.type_last, names, collective=self.collective)
        keep = labels >= 0
        n_collective = int((~keep).sum())

        step = _row_step_seconds(dataset)
        span = (self.seq_len - 1) * step
        block = self.group_blocks * max(span, step)
        end_block = np.floor(windows.end_timestamps / block).astype(np.int64)
        start_block = np.floor((windows.end_timestamps - span) / block).astype(np.int64)
        straddles = (end_block != start_block) & keep
        keep &= end_block == start_block  # embargo: a window must lie inside one block
        n_straddling = int(straddles.sum())

        codes = np.unique(windows.entity_ids.astype(str), return_inverse=True)[1]
        pairs = np.stack([codes.astype(np.int64), end_block], axis=1)[keep]
        groups = np.unique(pairs, axis=0, return_inverse=True)[1].ravel()
        kept = windows.subset(keep)
        y = labels[keep]
        if len(kept) == 0:
            raise ValueError("every window was dropped (collective / block boundary)")

        which = group_stratified_split(groups, y, self.split, seed=self.seed)
        parts: list[LabeledSequences] = []
        for part_id in range(3):
            mask = which == part_id
            parts.append(
                LabeledSequences(
                    kept.x[mask],
                    y[mask],
                    kept.entity_ids[mask],
                    kept.end_timestamps[mask],
                    np.zeros(int(mask.sum()), dtype=bool),
                )
            )
        train, val, test = parts
        n_real = len(train)
        for label, part in (("train", train), ("validation", val), ("test", test)):
            missing = np.flatnonzero(part.class_counts(len(names)) == 0)
            if missing.size:
                raise ValueError(
                    f"{label} split has no window of class {[names[i] for i in missing]}; "
                    "simulate a longer period"
                )

        if self.smote_ratio:
            flat = train.x.reshape(len(train), -1)
            over = smote_oversample(
                flat, train.y, ratio=self.smote_ratio, k_neighbors=self.smote_k, seed=self.seed
            )
            n_new = len(over.y) - n_real
            train = LabeledSequences(
                over.x.reshape(len(over.y), *train.x.shape[1:]),
                over.y,
                np.concatenate([train.entity_ids, np.full(n_new, "synthetic", dtype=object)]),
                np.concatenate([train.end_timestamps, np.full(n_new, np.nan)]),
                over.synthetic,
            )
        weights = balanced_class_weights(train.y, len(names))
        info = {
            "n_windows": len(windows),
            "n_collective_dropped": n_collective,
            "n_straddling_dropped": n_straddling,
            "n_groups": int(groups.max()) + 1,
            "row_step_seconds": step,
            "block_seconds": block,
            "n_synthetic": len(train) - n_real,
            "smote_ratio": self.smote_ratio,
            "collective_policy": self.collective,
            "split": self.split,
            "stride": self.stride,
            "seed": self.seed,
        }
        data = ClassifierData(
            names,
            tuple(windows.feature_names),
            self.columns,
            scaler,
            train,
            val,
            test,
            weights,
            info,
        )
        _log.info("lstm_classifier_data_built", summary=data.summary(), **info)
        return data


def _row_step_seconds(dataset: LabeledFeatureSet) -> float:
    """Median time between consecutive rows of the entity with the most rows."""
    if len(dataset) == 0:
        return 1.0
    entities = dataset.entity_ids.astype(str)
    names, counts = np.unique(entities, return_counts=True)
    ts = np.sort(dataset.timestamps[entities == names[counts.argmax()]])
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else 1.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class ClassifierHistory:
    """Per-epoch curves of one training run.

    ``train_loss`` and ``val_loss`` are the same class-weighted cross-entropy, so the curves are
    comparable; ``val_macro_f1`` / ``val_f1`` (per class, in ``class_names`` order) are measured
    on the natural-prevalence validation set. ``gradient_norm`` is the mean pre-clipping norm.
    """

    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    monitor: str = "val_macro_f1"
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    val_f1: list[list[float]] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = -math.inf
    stopped_early: bool = False

    @property
    def epochs_run(self) -> int:
        """Epochs trained."""
        return len(self.train_loss)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        return {
            "class_names": list(self.class_names),
            "monitor": self.monitor,
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "val_macro_f1": list(self.val_macro_f1),
            "val_f1": [list(row) for row in self.val_f1],
            "learning_rate": list(self.learning_rate),
            "gradient_norm": list(self.gradient_norm),
            "epoch_seconds": list(self.epoch_seconds),
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "stopped_early": self.stopped_early,
            "epochs_run": self.epochs_run,
        }


class LSTMClassifierTrainer:
    """Trains an :class:`LSTMClassifier` with class-weighted cross-entropy.

    AdamW (``weight_decay=1e-4``), ``CosineAnnealingLR`` over ``max_epochs``, gradient-norm
    clipping, and early stopping (patience 8) on the validation **macro F1** -- the number that
    matters under this imbalance; plain accuracy or unweighted loss would reward predicting
    "normal" for everything. The best-epoch weights are restored at the end. Each epoch logs
    ``train_loss``, ``val_loss``, ``val_macro_f1``, per-class ``val_f1/<class>``,
    ``learning_rate`` and ``gradient_norm`` to TensorBoard under ``log_dir/<run_name>``.
    """

    def __init__(
        self,
        model: LSTMClassifier,
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 8,
        grad_clip_norm: float = 1.0,
        monitor: Monitor = "val_macro_f1",
        min_delta: float = 1e-4,
        min_lr: float = 1e-6,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        device: str | torch.device | None = None,
        log_dir: str | Path | None = DEFAULT_LOG_DIR,
        run_name: str | None = None,
        seed: int = 42,
    ) -> None:
        """Create the trainer.

        Args:
            model: The model to train (moved to ``device``).
            class_names: Names of the classes, in label order.
            learning_rate: Initial AdamW learning rate.
            weight_decay: AdamW weight decay.
            batch_size: Training batch size.
            max_epochs: Upper bound on epochs (also the cosine period).
            patience: Epochs without improvement before stopping.
            grad_clip_norm: Maximum global gradient norm.
            monitor: ``val_macro_f1`` (maximise) or ``val_loss`` (minimise) drives early stopping.
            min_delta: Smallest improvement that counts.
            min_lr: Floor of the cosine schedule.
            num_workers: DataLoader workers (4 as specified; ``0`` for notebooks / tests).
            pin_memory: ``None`` pins when CUDA is available.
            device: ``cpu`` / ``cuda``; automatic when ``None``.
            log_dir: TensorBoard root; ``None`` disables TensorBoard.
            run_name: Sub-directory of ``log_dir``; a timestamp when ``None``.
            seed: Seed of batch order and dropout (set at the start of ``fit``).

        Raises:
            ValueError: An argument is out of range or ``class_names`` does not fit the model.
        """
        if learning_rate <= 0 or batch_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("learning_rate, batch_size, max_epochs and patience must be positive")
        if weight_decay < 0 or grad_clip_norm <= 0 or num_workers < 0:
            raise ValueError("weight_decay >= 0, grad_clip_norm > 0 and num_workers >= 0 required")
        if monitor not in ("val_macro_f1", "val_loss"):
            raise ValueError("monitor must be 'val_macro_f1' or 'val_loss'")
        self.class_names = _check_class_names(class_names)
        if len(self.class_names) != model.n_classes:
            raise ValueError(f"{len(self.class_names)} class names for {model.n_classes} outputs")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.monitor: Monitor = monitor
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.run_name = run_name
        self.seed = seed

    @classmethod
    def from_config(
        cls, cfg: LSTMClassifierConfig, model: LSTMClassifier, **overrides: Any
    ) -> LSTMClassifierTrainer:
        """Create the trainer from ``config.models.lstm_classifier``."""
        params: dict[str, Any] = {
            "class_names": cfg.class_names,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.early_stopping_patience,
        }
        params.update(overrides)
        return cls(model, **params)

    def _evaluate(
        self, val: SequenceDataset, weight: torch.Tensor | None
    ) -> tuple[float, float, list[float]]:
        """Return ``(weighted loss, macro F1, per-class F1)`` of the validation set."""
        logits, _ = predict_logits(self.model, val.x, batch_size=self.batch_size * 4)
        labels = val.y.numpy()
        loss = torch_f.cross_entropy(
            torch.from_numpy(logits),
            val.y,
            weight=None if weight is None else weight.cpu(),
        ).item()
        per_class = f1_score(
            labels,
            logits.argmax(axis=1),
            labels=list(range(len(self.class_names))),
            average=None,
            zero_division=0,
        )
        return loss, float(np.mean(per_class)), [float(v) for v in per_class]

    def fit(
        self,
        train: SequenceDataset,
        val: SequenceDataset,
        *,
        class_weights: np.ndarray | Sequence[float] | None = None,
    ) -> ClassifierHistory:
        """Train on ``train``, early-stopping on ``val``; the best weights are restored.

        Args:
            train: Training windows (after SMOTE).
            val: Validation windows at the natural class distribution.
            class_weights: Per-class loss weights (``ClassifierData.class_weights``).

        Raises:
            ValueError: Empty data, a feature-count mismatch or bad weights.
            FloatingPointError: The loss became NaN or infinite.
        """
        if len(train) == 0 or len(val) == 0:
            raise ValueError("training and validation sets must not be empty")
        for name, part in (("train", train), ("validation", val)):
            if part.x.shape[-1] != self.model.n_features:
                raise ValueError(
                    f"{name} windows have {part.x.shape[-1]} features, model expects "
                    f"{self.model.n_features}"
                )
        weight: torch.Tensor | None = None
        if class_weights is not None:
            weight = torch.as_tensor(np.asarray(class_weights), dtype=torch.float32)
            if weight.shape != (len(self.class_names),) or bool((weight <= 0).any()):
                raise ValueError("class_weights must be positive, one per class")
        device_weight = None if weight is None else weight.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=device_weight)

        torch.manual_seed(self.seed)
        loader = make_loader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=self.seed,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.min_lr
        )
        run = self.run_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        writer = _make_writer(self.log_dir / run) if self.log_dir is not None else None
        history = ClassifierHistory(self.class_names, self.monitor)
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        try:
            for epoch in range(1, self.max_epochs + 1):
                started = time.perf_counter()
                lr = float(optimizer.param_groups[0]["lr"])
                train_loss, grad_norm = self._train_epoch(loader, optimizer, criterion)
                scheduler.step()
                val_loss, macro_f1, per_class = self._evaluate(val, weight)
                history.train_loss.append(train_loss)
                history.val_loss.append(val_loss)
                history.val_macro_f1.append(macro_f1)
                history.val_f1.append(per_class)
                history.learning_rate.append(lr)
                history.gradient_norm.append(grad_norm)
                history.epoch_seconds.append(time.perf_counter() - started)
                if writer is not None:
                    scalars = {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_macro_f1": macro_f1,
                        "learning_rate": lr,
                        "gradient_norm": grad_norm,
                    }
                    scalars.update(
                        {
                            f"val_f1/{n}": v
                            for n, v in zip(self.class_names, per_class, strict=False)
                        }
                    )
                    for tag, value in scalars.items():
                        writer.add_scalar(tag, value, epoch)
                _log.debug(
                    "lstm_classifier_epoch", epoch=epoch, train_loss=train_loss,
                    val_loss=val_loss, val_macro_f1=macro_f1, lr=lr,
                )  # fmt: skip
                score = macro_f1 if self.monitor == "val_macro_f1" else -val_loss
                if score > history.best_score + self.min_delta:
                    history.best_score, history.best_epoch, stale = score, epoch, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    stale += 1
                    if stale >= self.patience:
                        history.stopped_early = True
                        break
        finally:
            if writer is not None:
                writer.close()
        self.model.load_state_dict(best_state)
        _log.info(
            "lstm_classifier_trained",
            epochs=history.epochs_run,
            best_epoch=history.best_epoch,
            monitor=self.monitor,
            best_score=round(history.best_score, 4),
            stopped_early=history.stopped_early,
        )
        return history

    def _train_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run one epoch; return ``(mean loss, mean pre-clipping gradient norm)``."""
        self.model.train()
        n = batches = 0
        loss_sum = grad_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = self.model(xb)
            loss = criterion(logits, yb)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "loss is not finite; lower the learning rate or check the inputs"
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            optimizer.step()
            n += len(xb)
            loss_sum += loss.item() * len(xb)
            grad_sum += float(norm)
            batches += 1
        return loss_sum / n, grad_sum / batches


# --------------------------------------------------------------------------- #
# Calibration and evaluation
# --------------------------------------------------------------------------- #
class TemperatureScaler:
    """Post-hoc calibration with one scalar: ``softmax(logits / T)``.

    ``T`` minimises the negative log-likelihood of held-out (validation) logits; ``T > 1`` softens
    an over-confident network, ``T < 1`` sharpens an under-confident one. Dividing all logits by
    one number never changes the argmax, so accuracy and F1 are unchanged -- only the
    probabilities are.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Create a scaler (``T = 1`` until :meth:`calibrate` runs)."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.is_fitted = False

    @staticmethod
    def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
        log_p = log_softmax(logits / temperature, axis=1)
        return float(-log_p[np.arange(len(labels)), labels].mean())

    def calibrate(
        self, val_logits: np.ndarray | torch.Tensor, val_labels: np.ndarray | torch.Tensor
    ) -> float:
        """Find and store the NLL-optimal temperature on a validation set; return it.

        Raises:
            ValueError: Empty input, shapes that do not match or labels out of range.
        """
        logits = np.asarray(val_logits, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.int64)
        if logits.ndim != 2 or labels.shape != (len(logits),) or len(logits) == 0:
            raise ValueError("val_logits must be (n, n_classes) and val_labels (n,), n > 0")
        if labels.min() < 0 or labels.max() >= logits.shape[1]:
            raise ValueError("val_labels contain values outside the class range")
        result = minimize_scalar(
            lambda log_t: self._nll(logits, labels, math.exp(log_t)),
            bounds=(math.log(_MIN_TEMPERATURE), math.log(_MAX_TEMPERATURE)),
            method="bounded",
            options={"xatol": 1e-5},
        )
        fitted = math.exp(float(result.x))
        # T = 1 is inside the search interval: never return something worse than "no scaling".
        self.temperature = (
            fitted if self._nll(logits, labels, fitted) <= self._nll(logits, labels, 1.0) else 1.0
        )
        self.is_fitted = True
        return self.temperature

    def scale(
        self, logits: np.ndarray | torch.Tensor, temperature: float | None = None
    ) -> np.ndarray:
        """Return calibrated probabilities ``softmax(logits / T)`` (``T`` defaults to the fit)."""
        t = self.temperature if temperature is None else float(temperature)
        if t <= 0:
            raise ValueError("temperature must be positive")
        result: np.ndarray = softmax(np.asarray(logits, dtype=np.float64) / t, axis=-1)
        return result

    def nll(self, logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
        """Mean negative log-likelihood of ``labels`` under the scaled probabilities."""
        return self._nll(
            np.asarray(logits, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            self.temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly state."""
        return {"temperature": self.temperature, "is_fitted": self.is_fitted}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemperatureScaler:
        """Rebuild from :meth:`to_dict` output."""
        scaler = cls(float(data["temperature"]))
        scaler.is_fitted = bool(data.get("is_fitted", True))
        return scaler


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predictions by confidence; return ``(mean_confidence, accuracy, count)`` per bin.

    Empty bins have ``count == 0`` and ``NaN`` for the other two entries.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    slot = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(slot, minlength=n_bins)
    mean_conf = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    for b in np.flatnonzero(count):
        mean_conf[b] = conf[slot == b].mean()
        accuracy[b] = hit[slot == b].mean()
    return mean_conf, accuracy, count


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE: the count-weighted gap between confidence and accuracy over the bins."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    mean_conf, accuracy, count = reliability_bins(p.max(axis=1), p.argmax(axis=1) == y, n_bins)
    filled = count > 0
    return float((count[filled] / count.sum() * np.abs(accuracy[filled] - mean_conf[filled])).sum())


@dataclass(frozen=True)
class ClassificationReport:
    """Metrics of one set of predictions (see :func:`evaluate_predictions`).

    ``confusion`` has true classes in rows and predicted classes in columns. ``roc_auc`` and
    ``average_precision`` are one-vs-rest per class (``NaN`` when the class is absent or the only
    one). ``anomaly_roc_auc`` / ``anomaly_average_precision`` score "any anomaly vs normal" with
    ``1 - P(normal)``, which is what the detector's ``score`` is.
    """

    class_names: tuple[str, ...]
    confusion: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    accuracy: float
    macro_f1: float
    weighted_f1: float
    roc_auc: dict[str, float]
    average_precision: dict[str, float]
    anomaly_roc_auc: float
    anomaly_average_precision: float
    nll: float
    ece: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        names = self.class_names
        return {
            "class_names": list(names),
            "confusion": self.confusion.tolist(),
            "per_class": {
                n: {
                    "precision": float(self.precision[i]),
                    "recall": float(self.recall[i]),
                    "f1": float(self.f1[i]),
                    "support": int(self.support[i]),
                    "roc_auc": self.roc_auc[n],
                    "average_precision": self.average_precision[n],
                }
                for i, n in enumerate(names)
            },
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "anomaly_roc_auc": self.anomaly_roc_auc,
            "anomaly_average_precision": self.anomaly_average_precision,
            "nll": self.nll,
            "ece": self.ece,
        }


def _safe_binary(metric: Any, positives: np.ndarray, score: np.ndarray) -> float:
    if not positives.any() or positives.all():
        return math.nan
    return float(metric(positives, score))


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    *,
    n_bins: int = 15,
) -> ClassificationReport:
    """Per-class metrics, confusion matrix, ROC / PR and calibration of probabilistic predictions.

    Args:
        y_true: True class index per window.
        probabilities: Class probabilities, shape ``(n, n_classes)`` (argmax gives the prediction).
        class_names: Class names in index order.
        n_bins: Bins of the expected calibration error.

    Raises:
        ValueError: Shapes do not match.
    """
    names = tuple(class_names)
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape != (len(y), len(names)) or len(y) == 0:
        raise ValueError("probabilities must be (n, n_classes) with one row per label, n > 0")
    pred = p.argmax(axis=1)
    labels = np.arange(len(names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    roc = {n: _safe_binary(roc_auc_score, y == k, p[:, k]) for k, n in enumerate(names)}
    ap = {n: _safe_binary(average_precision_score, y == k, p[:, k]) for k, n in enumerate(names)}
    anomalous = y != NORMAL
    anomaly_score = 1.0 - p[:, NORMAL]
    return ClassificationReport(
        class_names=names,
        confusion=confusion_matrix(y, pred, labels=labels),
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        accuracy=float((pred == y).mean()),
        macro_f1=float(f1.mean()),
        weighted_f1=float(np.average(f1, weights=support)) if support.sum() else 0.0,
        roc_auc=roc,
        average_precision=ap,
        anomaly_roc_auc=_safe_binary(roc_auc_score, anomalous, anomaly_score),
        anomaly_average_precision=_safe_binary(average_precision_score, anomalous, anomaly_score),
        nll=float(-np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0)).mean()),
        ece=expected_calibration_error(p, y, n_bins),
    )


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassifierOutput:
    """Vectorised detector output for a batch of windows.

    Attributes:
        logits: Raw network outputs ``(n, n_classes)``.
        probabilities: Temperature-scaled class probabilities ``(n, n_classes)``.
        attention: Attention weights over the time steps ``(n, seq_len)``.
        predicted: Argmax class index per window.
        score: ``1 - P(normal)``; ``>= decision_threshold`` is an anomaly.
    """

    logits: np.ndarray
    probabilities: np.ndarray
    attention: np.ndarray
    predicted: np.ndarray
    score: np.ndarray


class LSTMClassifierDetector:
    """Inference wrapper: window in, :class:`AnomalyScore` (``point`` / ``contextual``) out.

    Offline::

        detector = LSTMClassifierDetector(model, names, scaler=temperature_scaler)
        out = detector.classify_sequences(x_test)          # arrays, for evaluation

    Streaming, from events pushed one at a time (sliding window kept per entity)::

        for vector in stream:                              # FeatureVector.scaled
            result = detector.push(entity_id, vector, timestamp)

    Streaming, from the feature store (Day 15; the same window the autoencoder reads)::

        detector = LSTMClassifierDetector(..., sequences=pipeline.sequences)
        result = detector.detect("wh1-A-temp-01")
        results = detector.detect_many(entity_ids)         # one batched forward pass

    Until an entity has ``min_real_fraction * seq_len`` real events the result is a warm-up
    result (``score=0``, ``details["warming_up"]``): padded windows are never scored. Thread-safe.
    """

    def __init__(
        self,
        model: LSTMClassifier,
        feature_names: Sequence[str],
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        temperature: TemperatureScaler | None = None,
        source_type: str = "warehouse",
        sequences: SequenceSource | None = None,
        decision_threshold: float = 0.5,
        min_real_fraction: float = MIN_REAL_FRACTION,
        require_warm: bool = True,
        batch_size: int = 256,
        columns: Sequence[int] | None = None,
    ) -> None:
        """Create the detector.

        Args:
            model: A trained classifier.
            feature_names: Column names of the windows (``ClassifierData.feature_names``).
            class_names: Class names in label order (first is ``"normal"``).
            temperature: Fitted :class:`TemperatureScaler`; ``None`` means ``T = 1``.
            source_type: ``warehouse``, ``vehicle`` or ``inventory``.
            sequences: Where :meth:`detect` reads windows (``FeaturePipeline.sequences``).
            decision_threshold: ``score`` at or above which an event is an anomaly.
            min_real_fraction: Share of the window that must be real events.
            require_warm: Wait until the entity's longest aggregation window is warm.
            batch_size: Windows per forward call.
            columns: Columns of the pipeline's feature vector the model was trained on.

        Raises:
            ValueError: Feature or class counts do not match the model, or a bad threshold.
        """
        names = tuple(feature_names)
        if len(names) != model.n_features:
            raise ValueError(f"{len(names)} feature names for a model with {model.n_features}")
        self.class_names = _check_class_names(class_names)
        if len(self.class_names) != model.n_classes:
            raise ValueError(f"{len(self.class_names)} class names for {model.n_classes} outputs")
        if not 0.0 < decision_threshold < 1.0 or not 0.0 < min_real_fraction <= 1.0:
            raise ValueError("decision_threshold in (0, 1) and min_real_fraction in (0, 1]")
        self.model = model
        self.feature_names = names
        self.temperature = temperature if temperature is not None else TemperatureScaler()
        self.source_type = source_type
        self.sequences = sequences
        self.decision_threshold = decision_threshold
        self.min_real_fraction = min_real_fraction
        self.require_warm = require_warm
        self.batch_size = batch_size
        self.columns = None if columns is None else tuple(int(c) for c in columns)
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self._lock = threading.RLock()

    @property
    def seq_len(self) -> int:
        """Window length of the model."""
        return self.model.seq_len

    @property
    def min_real_steps(self) -> int:
        """Real (non-padding) steps a window needs before it is scored."""
        return math.ceil(self.min_real_fraction * self.seq_len)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def classify_sequences(self, x: np.ndarray | torch.Tensor) -> ClassifierOutput:
        """Classify a batch of windows ``(n, seq_len, n_features)``.

        Raises:
            ValueError: Bad shape or non-finite values.
        """
        with self._lock:
            logits, attention = predict_logits(self.model, x, batch_size=self.batch_size)
        probabilities = self.temperature.scale(logits)
        return ClassifierOutput(
            logits,
            probabilities,
            attention,
            probabilities.argmax(axis=1),
            1.0 - probabilities[:, NORMAL],
        )

    def _to_anomaly_score(
        self,
        entity_id: str,
        timestamp: datetime,
        out: ClassifierOutput,
        i: int,
        real_steps: int,
        raw_value: float | None,
    ) -> AnomalyScore:
        probs = out.probabilities[i]
        score = float(out.score[i])
        kind = self.class_names[1 + int(probs[1:].argmax())]
        attention = out.attention[i]
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=timestamp,
            score=score,
            is_anomaly=score >= self.decision_threshold,
            anomaly_type=kind,
            confidence=max(score, 1.0 - score),
            detector=DETECTOR_NAME,
            raw_value=raw_value,
            details={
                "class_probabilities": {
                    n: float(p) for n, p in zip(self.class_names, probs, strict=False)
                },
                "predicted_class": self.class_names[int(out.predicted[i])],
                "attention_weights": [float(a) for a in attention],
                "top_attention_steps": [int(s) for s in np.argsort(-attention)[:3]],
                "temperature": self.temperature.temperature,
                "decision_threshold": self.decision_threshold,
                "real_steps": real_steps,
                "warming_up": False,
                "source_type": self.source_type,
            },
        )

    def detect_sequence(
        self,
        entity_id: str,
        timestamp: datetime,
        sequence: np.ndarray | torch.Tensor,
        *,
        real_steps: int | None = None,
        raw_value: float | None = None,
    ) -> AnomalyScore:
        """Classify one ready-made window ``(seq_len, n_features)``."""
        out = self.classify_sequences(sequence)
        steps = self.seq_len if real_steps is None else real_steps
        return self._to_anomaly_score(entity_id, timestamp, out, 0, steps, raw_value)

    def push(
        self,
        entity_id: str,
        vector: np.ndarray | Sequence[float],
        timestamp: datetime,
        *,
        raw_value: float | None = None,
    ) -> AnomalyScore:
        """Append one standardised event to the entity's sliding window and classify it.

        Args:
            entity_id: Sensor / vehicle / SKU the event belongs to.
            vector: The pipeline's standardised feature vector (``FeatureVector.scaled``); the
                detector's ``columns`` are applied to it. NaNs are forward-filled per feature
                from the entity's previous event (``0`` if there is none).
            timestamp: Event time.
            raw_value: The raw reading, echoed in the result.

        Returns:
            The classification of the newest event, or a warm-up result while the entity has
            fewer than :attr:`min_real_steps` events.

        Raises:
            ValueError: The vector does not have the model's feature count.
        """
        row = np.asarray(vector, dtype=np.float32).ravel()
        if self.columns is not None:
            row = row[list(self.columns)]
        if row.shape[0] != self.model.n_features:
            raise ValueError(f"expected {self.model.n_features} features, got {row.shape[0]}")
        with self._lock:
            buffer = self._buffers.setdefault(entity_id, deque(maxlen=self.seq_len))
            bad = ~np.isfinite(row)
            if bad.any():
                row = row.copy()
                row[bad] = buffer[-1][bad] if buffer else 0.0
            buffer.append(row)
            real = len(buffer)
            if real < self.min_real_steps:
                return self._warming_up(entity_id, timestamp, real)
            window = np.zeros((self.seq_len, self.model.n_features), dtype=np.float32)
            window[self.seq_len - real :] = np.stack(buffer)
        return self.detect_sequence(
            entity_id, timestamp, window, real_steps=real, raw_value=raw_value
        )

    def reset(self, entity_id: str | None = None) -> None:
        """Forget the pushed history of one entity (or of all of them)."""
        with self._lock:
            if entity_id is None:
                self._buffers.clear()
            else:
                self._buffers.pop(entity_id, None)

    def detect(self, entity_id: str) -> AnomalyScore:
        """Read the entity's window from the feature store and classify it.

        Returns a warm-up result (``score=0``) while the entity has too little warm history;
        never raises for unknown entities.

        Raises:
            RuntimeError: The detector was created without ``sequences``.
        """
        return self.detect_many([entity_id])[0]

    def detect_many(self, entity_ids: Sequence[str]) -> list[AnomalyScore]:
        """Micro-batch version of :meth:`detect` (one forward pass for all ready entities).

        Raises:
            RuntimeError: The detector was created without ``sequences``.
        """
        if self.sequences is None:
            raise RuntimeError("detect() needs `sequences` (pass FeaturePipeline.sequences)")
        source = self.sequences
        shortest, longest = source.window_seconds[0], source.window_seconds[-1]
        results: list[AnomalyScore | None] = [None] * len(entity_ids)
        ready: list[tuple[int, datetime, int, float | None]] = []
        windows: list[torch.Tensor] = []
        for slot, entity_id in enumerate(entity_ids):
            sequence, mask = source.build_sequence_with_mask(
                entity_id, self.seq_len, self.source_type
            )
            latest = source.store.get_features(entity_id, shortest)
            slowest = source.store.get_features(entity_id, longest)
            real = int(mask.sum())
            warm = slowest is not None and (slowest.is_warm or not self.require_warm)
            if latest is None or not warm or real < self.min_real_steps:
                stamp = latest.window_end if latest is not None else datetime.now(UTC)
                results[slot] = self._warming_up(entity_id, stamp, real)
                continue
            primary = next(iter(latest.signals.values()), None)
            ready.append((slot, latest.window_end, real, primary.last if primary else None))
            windows.append(sequence if self.columns is None else sequence[:, list(self.columns)])
        if windows:
            out = self.classify_sequences(torch.stack(windows))
            for k, (slot, stamp, real, raw) in enumerate(ready):
                results[slot] = self._to_anomaly_score(entity_ids[slot], stamp, out, k, real, raw)
        return [r for r in results if r is not None]

    def _warming_up(self, entity_id: str, timestamp: datetime, real_steps: int) -> AnomalyScore:
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=timestamp,
            score=0.0,
            is_anomaly=False,
            anomaly_type=None,
            confidence=0.0,
            detector=DETECTOR_NAME,
            raw_value=None,
            details={
                "warming_up": True,
                "real_steps": real_steps,
                "required_steps": self.min_real_steps,
                "source_type": self.source_type,
            },
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_checkpoint(self) -> dict[str, Any]:
        """Return a plain dict (weights, architecture, temperature, schema) for ``torch.save``."""
        return {
            "model_config": self.model.get_config(),
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "temperature": self.temperature.to_dict(),
            "class_names": list(self.class_names),
            "feature_names": list(self.feature_names),
            "source_type": self.source_type,
            "decision_threshold": self.decision_threshold,
            "min_real_fraction": self.min_real_fraction,
            "require_warm": self.require_warm,
            "columns": None if self.columns is None else list(self.columns),
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any], *, sequences: SequenceSource | None = None
    ) -> LSTMClassifierDetector:
        """Rebuild a detector from :meth:`to_checkpoint` output."""
        config = dict(checkpoint["model_config"])
        n_features = int(config.pop("n_features"))
        model = LSTMClassifier(n_features, **config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return cls(
            model,
            checkpoint["feature_names"],
            class_names=checkpoint["class_names"],
            temperature=TemperatureScaler.from_dict(checkpoint["temperature"]),
            source_type=checkpoint["source_type"],
            sequences=sequences,
            decision_threshold=float(checkpoint["decision_threshold"]),
            min_real_fraction=float(checkpoint["min_real_fraction"]),
            require_warm=bool(checkpoint["require_warm"]),
            columns=checkpoint.get("columns"),
        )

    def save(
        self,
        registry: ModelRegistry,
        *,
        metrics: Mapping[str, Any] | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ModelVersion:
        """Store this detector as a new version of ``lstm_classifier`` in the registry."""
        return registry.save(
            MODEL_TYPE,
            self.to_checkpoint(),
            metrics=metrics,
            hyperparams={**self.model.get_config(), **(hyperparams or {})},
            extra={
                "feature_names": list(self.feature_names),
                "class_names": list(self.class_names),
                **(extra or {}),
            },
            serializer="torch",
        )

    @classmethod
    def load(
        cls,
        registry: ModelRegistry,
        version: str | None = None,
        *,
        sequences: SequenceSource | None = None,
    ) -> LSTMClassifierDetector:
        """Load a version (default: the latest) saved with :meth:`save`."""
        loaded = registry.load(MODEL_TYPE, version)
        return cls.from_checkpoint(loaded.model, sequences=sequences)


# --------------------------------------------------------------------------- #
# One-call training
# --------------------------------------------------------------------------- #
def train_classifier(
    data: ClassifierData,
    cfg: LSTMClassifierConfig,
    *,
    source_type: str = "warehouse",
    sequences: SequenceSource | None = None,
    log_dir: str | Path | None = DEFAULT_LOG_DIR,
    run_name: str | None = None,
    device: str | torch.device | None = None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    monitor: Monitor = "val_macro_f1",
    seed: int = 42,
) -> tuple[LSTMClassifierDetector, ClassifierHistory]:
    """Train on ``data.train``, early-stop on ``data.val``, then temperature-scale on ``data.val``.

    The test split is not touched; report final numbers on it with
    :func:`evaluate_predictions`.

    Args:
        data: Output of :meth:`DatasetBuilder.build`.
        cfg: ``config.models.lstm_classifier``.
        source_type: Source type the detector serves.
        sequences: Sequence source for ``detect()`` (optional).
        log_dir: TensorBoard root (``None`` disables it).
        run_name: TensorBoard run name.
        device: Torch device; automatic when ``None``.
        num_workers: DataLoader workers.
        pin_memory: Pin batches (``None`` = when CUDA is available).
        monitor: Early-stopping metric.
        seed: Seed of batch order and dropout.

    Returns:
        ``(detector, history)``; the detector holds the best weights and a fitted temperature.

    Raises:
        ValueError: The config's classes or window length do not match ``data``.
    """
    if tuple(cfg.class_names) != data.class_names:
        raise ValueError(f"config classes {cfg.class_names} differ from data {data.class_names}")
    if cfg.seq_len != data.seq_len:
        raise ValueError(f"config seq_len {cfg.seq_len} differs from data seq_len {data.seq_len}")
    torch.manual_seed(seed)
    model = LSTMClassifier.from_config(cfg, data.n_features, len(data.class_names))
    trainer = LSTMClassifierTrainer.from_config(
        cfg,
        model,
        log_dir=log_dir,
        run_name=run_name,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        monitor=monitor,
        seed=seed,
    )
    history = trainer.fit(
        data.train.to_dataset(), data.val.to_dataset(), class_weights=data.class_weights
    )
    detector = LSTMClassifierDetector(
        trainer.model,
        data.feature_names,
        class_names=data.class_names,
        source_type=source_type,
        sequences=sequences,
        columns=data.columns,
    )
    logits, _ = predict_logits(trainer.model, data.val.x)
    detector.temperature.calibrate(logits, data.val.y)
    _log.info("lstm_classifier_calibrated", temperature=detector.temperature.temperature)
    return detector, history


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "DETECTOR_NAME",
    "ClassificationReport",
    "ClassifierData",
    "ClassifierHistory",
    "ClassifierOutput",
    "DatasetBuilder",
    "LSTMClassifier",
    "LSTMClassifierDetector",
    "LSTMClassifierTrainer",
    "LabeledSequences",
    "Oversampled",
    "SequenceDataset",
    "TemperatureScaler",
    "balanced_class_weights",
    "encode_labels",
    "evaluate_predictions",
    "expected_calibration_error",
    "group_stratified_split",
    "make_loader",
    "predict_logits",
    "reliability_bins",
    "smote_oversample",
    "train_classifier",
]
