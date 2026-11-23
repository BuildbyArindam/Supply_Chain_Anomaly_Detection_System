"""LSTM autoencoder for collective anomalies (Day 10).

A *collective* anomaly is a stretch of readings that is abnormal as a **shape** while every single
reading may still look plausible: the gradual drift of a failing cooling unit is the classic
example. The Day 8 detectors judge one value at a time (Z-score, IQR) or accumulate a drift
(CUSUM); this model learns what a *normal minute of a sensor looks like* -- the whole
``(seq_len, n_features)`` window of engineered features -- and flags windows it cannot
reconstruct.

It is unsupervised: it is trained on **normal windows only**, so the simulators' labels are used
to select clean training windows and to evaluate, never to fit the weights.

Contents
--------
* :class:`LSTMAutoencoder` -- encoder (LSTM 64 -> 128) -> variational bottleneck -> decoder
  (LSTM 128 -> 64) in PyTorch, with LayerNorm after each LSTM layer and dropout between layers.
* :class:`LSTMAutoencoderTrainer` -- Adam, ``ReduceLROnPlateau``, early stopping, gradient
  clipping, TensorBoard logging; loss = MSE + ``kl_weight`` * KL.
* :class:`AnomalyThresholdCalibrator` -- ``mean + sigma * std`` of the reconstruction errors of
  normal validation windows, overall and per feature.
* :class:`LSTMAutoencoderDetector` -- inference wrapper: ``detect(entity_id)`` reads the entity's
  sequence from the feature store and returns an ``AnomalyScore`` (Day 8 result type).
* :func:`make_sequences` / :class:`SequenceSet` -- offline windows (with ground truth kept next
  to, never inside, the model input) built exactly like the online ``SequenceBuilder`` does.
* :func:`train_detector` -- one call from windows to a calibrated detector.

Architecture
------------
::

    x (B, T, F) -> LSTM(F,64) -> LN -> Dropout -> LSTM(64,128) -> LN -> last step (B, 128)
                -> mu, logvar (B, latent_dim) -> z  (sampled in training, z = mu otherwise)
    z -> repeat T times -> LSTM(latent,128) -> LN -> Dropout -> LSTM(128,64) -> LN -> Dropout
                -> Linear(64, F)                                           -> x_hat (B, T, F)

The bottleneck is a small *variational* one: the KL term of the loss (weight ``0.1``) keeps the
context vector compact, which is what stops an LSTM autoencoder from learning the identity map
and reconstructing anomalies as well as normal data. ``forward`` returns
``(reconstruction, LatentState)``; ``LatentState.z`` is the context vector.

Score convention
----------------
The reconstruction error of a window is the mean squared error over time and features. The
detector reports ``score = error / (error + threshold)``: monotone in the error, ``0.5`` exactly
at the calibrated threshold and never saturating, so ``score >= 0.5`` is an anomaly -- the same
boundary convention as the Day 8 and Day 9 detectors, and the scores can be compared and stacked.
The raw error is in ``details["reconstruction_error"]`` (use it for AUC-ROC / AUC-PR).

Uncertainty (MC Dropout)
------------------------
Dropout stays active at inference and each window goes through ``mc_dropout_passes`` stochastic
forward passes. The reported error is the **mean** over the passes and ``error_std`` their
spread. The threshold is calibrated on the *same* MC-mean quantity, not on a deterministic pass:
dropout noise raises the error of every window, so mixing the two would make the detector flag
normal data. ``confidence`` is the probability that the verdict survives that noise
(``erf(|error - threshold| / (sqrt(2) * error_std))``).

Train / serve consistency
-------------------------
Windows are standardised with the pipeline's :class:`~src.models.feature_engineering.OnlineScaler`
and gap-filled like :class:`~src.models.feature_engineering.SequenceBuilder` does, and only
*warm* rows (longest aggregation window at least 90% filled) are used, so the detector sees at
serving time what it was trained on. Until an entity has enough warm history, ``detect`` returns a
warm-up result (``score=0``) instead of scoring padded data. Keep the scaler that was used for
training alongside the model (``FeaturePipeline.save_scalers``): the online pipeline keeps
updating it, so very long-running services drift slowly away from the training standardisation
and should be retrained (Day 17).
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol

import numpy as np
import torch
import torch.nn.functional as torch_f
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import erf
from sklearn.exceptions import NotFittedError
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.statistical_detector import AnomalyScore
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.kafka.feature_cache import FeatureStore
    from src.models.feature_engineering import OnlineScaler
    from src.models.model_registry import ModelRegistry, ModelVersion
    from src.models.offline_dataset import LabeledFeatureSet
    from src.utils.config import LSTMAutoencoderConfig

DETECTOR_NAME: Final[str] = "lstm_autoencoder"
MODEL_TYPE: Final[str] = "lstm_autoencoder"
DEFAULT_LOG_DIR: Final[str] = "runs/lstm_autoencoder"
#: Fewest normal windows a threshold can be calibrated on.
MIN_CALIBRATION_SAMPLES: Final[int] = 10
#: Per-feature errors vary at least this share of their mean (keeps near-constant features
#: such as the zone one-hots from getting a threshold that any rounding noise exceeds).
MIN_REL_SPREAD: Final[float] = 0.05
#: Share of ``seq_len`` that must be real (not padding) before a window is scored.
MIN_REAL_FRACTION: Final[float] = 0.8
_LOGVAR_BOUND: Final[float] = 10.0
_EPS: Final[float] = 1e-12

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LatentState(NamedTuple):
    """Bottleneck of one forward pass: the context vector and its Gaussian parameters."""

    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder: ``(batch, seq_len, n_features)`` in, same shape out.

    ``forward`` returns ``(reconstruction, latent)`` where ``latent`` is a :class:`LatentState`.
    In training mode ``z`` is sampled (``mu + eps * sigma``); in eval mode, and during MC
    Dropout, ``z = mu``, so the only randomness at inference is the dropout masks.
    """

    def __init__(
        self,
        n_features: int,
        *,
        seq_len: int = 60,
        hidden_sizes: tuple[int, int] = (64, 128),
        latent_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        """Create the model.

        Args:
            n_features: Features per time step.
            seq_len: Default window length (``forward`` accepts any length).
            hidden_sizes: Hidden sizes of the two encoder LSTM layers; the decoder mirrors them.
            latent_dim: Width of the bottleneck.
            dropout: Dropout probability between layers.

        Raises:
            ValueError: An argument is out of range.
        """
        super().__init__()
        h1, h2 = hidden_sizes
        if min(n_features, seq_len, h1, h2, latent_dim) < 1:
            raise ValueError("n_features, seq_len, hidden_sizes and latent_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_sizes = (h1, h2)
        self.latent_dim = latent_dim
        self.dropout_p = dropout

        self.enc_lstm1 = nn.LSTM(n_features, h1, batch_first=True)
        self.enc_norm1 = nn.LayerNorm(h1)
        self.enc_lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.enc_norm2 = nn.LayerNorm(h2)
        self.to_mu = nn.Linear(h2, latent_dim)
        self.to_logvar = nn.Linear(h2, latent_dim)

        self.dec_lstm1 = nn.LSTM(latent_dim, h2, batch_first=True)
        self.dec_norm1 = nn.LayerNorm(h2)
        self.dec_lstm2 = nn.LSTM(h2, h1, batch_first=True)
        self.dec_norm2 = nn.LayerNorm(h1)
        self.output = nn.Linear(h1, n_features)

        self.drop_enc = nn.Dropout(dropout)
        self.drop_dec1 = nn.Dropout(dropout)
        self.drop_dec2 = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, cfg: LSTMAutoencoderConfig, n_features: int) -> LSTMAutoencoder:
        """Create the model from ``config.models.lstm_autoencoder``."""
        return cls(
            n_features,
            seq_len=cfg.seq_len,
            hidden_sizes=cfg.hidden_sizes,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the constructor arguments (stored in checkpoints)."""
        return {
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden_sizes": list(self.hidden_sizes),
            "latent_dim": self.latent_dim,
            "dropout": self.dropout_p,
        }

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.n_features:
            raise ValueError(
                f"expected input of shape (batch, seq_len, {self.n_features}), got {tuple(x.shape)}"
            )

    def encode(self, x: torch.Tensor) -> LatentState:
        """Return the context vector of a batch of sequences (see :class:`LatentState`)."""
        self._check(x)
        out, _ = self.enc_lstm1(x)
        out = self.drop_enc(self.enc_norm1(out))
        out, _ = self.enc_lstm2(out)
        context = self.enc_norm2(out)[:, -1]
        mu = self.to_mu(context)
        logvar = self.to_logvar(context).clamp(-_LOGVAR_BOUND, _LOGVAR_BOUND)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return LatentState(z, mu, logvar)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Reconstruct ``seq_len`` steps from a context vector."""
        steps = z.unsqueeze(1).expand(-1, seq_len, -1)
        out, _ = self.dec_lstm1(steps)
        out = self.drop_dec1(self.dec_norm1(out))
        out, _ = self.dec_lstm2(out)
        out = self.drop_dec2(self.dec_norm2(out))
        result: torch.Tensor = self.output(out)
        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, LatentState]:
        """Return ``(reconstruction, latent)`` for ``x`` of shape ``(batch, seq_len, F)``."""
        latent = self.encode(x)
        return self.decode(latent.z, x.shape[1]), latent


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, 1) ), averaged over batch *and* latent dimensions.

    Averaging (not summing) keeps the term on the scale of the MSE of standardised features, so
    ``kl_weight = 0.1`` is a mild regulariser rather than the dominant loss.
    """
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()


@contextmanager
def _inference_mode(model: nn.Module, *, mc_dropout: bool) -> Iterator[None]:
    """Put ``model`` in eval mode, optionally with only its Dropout layers active; then restore."""
    was_training = model.training
    model.eval()
    if mc_dropout:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    try:
        yield
    finally:
        model.train(was_training)


def _as_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            f"expected sequences of shape (n, seq_len, n_features), got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("sequences contain NaN or infinite values (fill gaps first)")
    return tensor


@dataclass(frozen=True)
class ReconstructionErrors:
    """Reconstruction errors of a batch of windows.

    Attributes:
        error: Mean squared error per window (mean over the MC passes), shape ``(n,)``.
        error_std: Standard deviation of the error across the MC passes (``0`` for one pass).
        per_feature: Time-averaged squared error per feature (mean over the passes), ``(n, F)``.
    """

    error: np.ndarray
    error_std: np.ndarray
    per_feature: np.ndarray


@torch.no_grad()
def reconstruction_errors(
    model: LSTMAutoencoder,
    x: np.ndarray | torch.Tensor,
    *,
    mc_passes: int = 1,
    batch_size: int = 256,
) -> ReconstructionErrors:
    """Reconstruct ``x`` and return the errors per window and per feature.

    Args:
        model: The autoencoder (its train / eval mode is restored afterwards).
        x: Windows, shape ``(n, seq_len, n_features)``.
        mc_passes: ``1`` is one deterministic pass; more run that many stochastic passes with
            dropout active (MC Dropout) and average them.
        batch_size: Windows (times passes) per forward call.

    Raises:
        ValueError: ``mc_passes`` is below 1, the shape is wrong or ``x`` is not finite.
    """
    if mc_passes < 1:
        raise ValueError("mc_passes must be at least 1")
    tensor = _as_tensor(x)
    n, _, n_features = tensor.shape
    device = next(model.parameters()).device
    chunk = max(1, batch_size // mc_passes)
    errors = np.empty(n)
    spread = np.zeros(n)
    per_feature = np.empty((n, n_features))
    with _inference_mode(model, mc_dropout=mc_passes > 1):
        for start in range(0, n, chunk):
            batch = tensor[start : start + chunk].to(device)
            b = batch.shape[0]
            # Passes share one forward call: each repeated copy draws its own dropout masks.
            stacked = batch.repeat(mc_passes, 1, 1) if mc_passes > 1 else batch
            recon, _ = model(stacked)
            squared = (recon - stacked).pow(2).mean(dim=1)  # (passes * b, F)
            squared = squared.reshape(mc_passes, b, n_features)
            per_pass = squared.mean(dim=2)  # (passes, b)
            errors[start : start + b] = per_pass.mean(dim=0).cpu().numpy()
            if mc_passes > 1:
                spread[start : start + b] = per_pass.std(dim=0, unbiased=True).cpu().numpy()
            per_feature[start : start + b] = squared.mean(dim=0).cpu().numpy()
    return ReconstructionErrors(errors, spread, per_feature)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
#: Cyclical clock encodings. They are deterministic functions of the timestamp, so they cannot
#: carry information about a drifting sensor, but a model trained on a few hours of data cannot
#: reconstruct hours or weekdays it has not seen, which adds nuisance error (section 2 of
#: notebook 04 measures it).
TEMPORAL_FEATURES: Final[tuple[str, ...]] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "minute_sin",
    "minute_cos",
)


def select_features(
    feature_names: Sequence[str], exclude: Sequence[str] | None = None
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Choose the model's input columns out of the pipeline's feature vector.

    Args:
        feature_names: All feature names of the source (``feature_engineering.feature_names``).
        exclude: Names to drop. ``None`` drops whichever of :data:`TEMPORAL_FEATURES` the source
            has; names given explicitly must all exist.

    Returns:
        ``(names, columns)``: kept names and their column indices. Pass ``columns`` to both
        :func:`make_sequences` and :class:`LSTMAutoencoderDetector` so training and serving agree.

    Raises:
        ValueError: ``exclude`` names a feature that does not exist, or nothing would be left.
    """
    drop = set(TEMPORAL_FEATURES if exclude is None else exclude)
    unknown = drop - set(feature_names)
    if unknown and exclude is not None:
        raise ValueError(f"unknown features to exclude: {sorted(unknown)}")
    columns = tuple(i for i, name in enumerate(feature_names) if name not in drop)
    if not columns:
        raise ValueError("every feature was excluded")
    return tuple(feature_names[i] for i in columns), columns


@dataclass(frozen=True)
class SequenceSet:
    """Standardised windows with their ground truth, in chronological order of their end.

    The ground truth is for training-window selection and evaluation only; it is never an input
    of the model.

    Attributes:
        x: Windows, ``float32`` shape ``(n, seq_len, n_features)``.
        entity_ids: Sensor / vehicle / SKU of each window.
        end_timestamps: Event time of the newest row of each window (UTC epoch seconds).
        y_last: ``True`` where the newest row of the window is an anomaly.
        type_last: Anomaly type of the newest row (``None`` for a normal row).
        anomaly_fraction: Share of anomalous rows inside the window.
        feature_names: Column names of ``x``.
    """

    x: np.ndarray
    entity_ids: np.ndarray
    end_timestamps: np.ndarray
    y_last: np.ndarray
    type_last: np.ndarray
    anomaly_fraction: np.ndarray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def seq_len(self) -> int:
        """Window length."""
        return int(self.x.shape[1])

    @property
    def normal(self) -> np.ndarray:
        """Boolean mask of windows without a single anomalous row (what the model trains on)."""
        result: np.ndarray = self.anomaly_fraction == 0.0
        return result

    def subset(self, mask: np.ndarray) -> SequenceSet:
        """Return the windows selected by a boolean mask or an index array."""
        return SequenceSet(
            self.x[mask],
            self.entity_ids[mask],
            self.end_timestamps[mask],
            self.y_last[mask],
            self.type_last[mask],
            self.anomaly_fraction[mask],
            self.feature_names,
        )

    def between(self, start: datetime, end: datetime) -> np.ndarray:
        """Boolean mask of the windows that end in ``[start, end)``."""
        lo = (start if start.tzinfo else start.replace(tzinfo=UTC)).timestamp()
        hi = (end if end.tzinfo else end.replace(tzinfo=UTC)).timestamp()
        mask: np.ndarray = (self.end_timestamps >= lo) & (self.end_timestamps < hi)
        return mask


def _forward_fill_windows(windows: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along time in every window, ``0.0`` where nothing precedes them.

    Vectorised equivalent of :func:`~src.models.feature_engineering.fill_missing` applied to each
    window separately (which is what the online ``SequenceBuilder`` does), for ``(k, T, F)``.
    """
    valid = np.isfinite(windows)
    if valid.all():
        return windows
    steps = np.arange(windows.shape[1])[None, :, None]
    last_valid = np.maximum.accumulate(np.where(valid, steps, 0), axis=1)
    filled = np.take_along_axis(windows, last_valid, axis=1)
    filled[~np.maximum.accumulate(valid, axis=1)] = 0.0
    return filled


def make_sequences(
    dataset: LabeledFeatureSet,
    scaler: OnlineScaler,
    *,
    seq_len: int = 60,
    stride: int = 10,
    warm_only: bool = True,
    columns: Sequence[int] | None = None,
    max_sequences: int | None = None,
    seed: int = 0,
) -> SequenceSet:
    """Cut the labeled feature rows of a source into per-entity sliding windows.

    Rows of one entity are consecutive events; windows end every ``stride`` rows, counted back
    from the entity's newest row (so the most recent state is always covered).

    Args:
        dataset: Output of ``simulate_labeled_features`` for one source type.
        scaler: Standardises the raw features; fit it on the *training* rows only.
        seq_len: Window length.
        stride: Rows between window ends of one entity.
        warm_only: Drop aggregator cold-start rows (what training and serving both do).
        columns: Keep only these feature columns (see :func:`select_features`); the scaler is
            still applied to the full vector, so it stays the pipeline's scaler.
        max_sequences: Keep a random subset of at most this many windows (order preserved).
        seed: Seed of the subsampling.

    Returns:
        A :class:`SequenceSet` sorted by window end time (empty if no entity has ``seq_len``
        usable rows).

    Raises:
        ValueError: ``seq_len < 2`` or ``stride < 1``.
    """
    if seq_len < 2 or stride < 1:
        raise ValueError("seq_len must be at least 2 and stride at least 1")
    cols = None if columns is None else np.asarray(columns, dtype=int)
    names = (
        tuple(dataset.feature_names)
        if cols is None
        else tuple(dataset.feature_names[i] for i in cols)
    )
    n_features = len(names)
    keep = np.flatnonzero(dataset.warm) if warm_only else np.arange(len(dataset))
    parts: list[tuple[np.ndarray, ...]] = []
    if keep.size:
        entity = dataset.entity_ids[keep]
        order = np.argsort(entity, kind="stable")  # stable: time order survives within an entity
        grouped = keep[order]
        cuts = np.flatnonzero(entity[order][1:] != entity[order][:-1]) + 1
        for rows in np.split(grouped, cuts):
            n = rows.size
            if n < seq_len:
                continue
            standardised = scaler.transform(dataset.X[rows])
            if cols is not None:
                standardised = standardised[:, cols]
            ends = np.arange(n - 1, seq_len - 2, -stride)[::-1]
            picked = sliding_window_view(standardised, seq_len, axis=0)[ends - (seq_len - 1)]
            windows = _forward_fill_windows(picked.transpose(0, 2, 1))
            y = dataset.y[rows]
            cumulative = np.concatenate(([0], np.cumsum(y)))
            parts.append(
                (
                    np.ascontiguousarray(windows, dtype=np.float32),
                    dataset.entity_ids[rows][ends],
                    dataset.timestamps[rows][ends],
                    y[ends],
                    dataset.anomaly_type[rows][ends],
                    (cumulative[ends + 1] - cumulative[ends + 1 - seq_len]) / seq_len,
                )
            )
    if not parts:
        return SequenceSet(
            np.empty((0, seq_len, n_features), dtype=np.float32),
            np.empty(0, dtype=object),
            np.empty(0),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=object),
            np.empty(0),
            names,
        )
    x, entities, ends_ts, y_last, kinds, fraction = (
        np.concatenate(c) for c in zip(*parts, strict=True)
    )
    chronological = np.argsort(ends_ts, kind="stable")
    if max_sequences is not None and len(chronological) > max_sequences:
        rng = np.random.default_rng(seed)
        chronological = np.sort(rng.choice(chronological, size=max_sequences, replace=False))
        chronological = chronological[np.argsort(ends_ts[chronological], kind="stable")]
    return SequenceSet(
        x[chronological],
        entities[chronological],
        ends_ts[chronological],
        y_last[chronological].astype(bool),
        kinds[chronological],
        fraction[chronological],
        names,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingHistory:
    """Per-epoch curves of one training run (for plots and the notebook).

    ``train_loss`` / ``val_loss`` are the full objective (MSE + ``kl_weight`` * KL); the ``*_mse``
    and ``*_kl`` lists are its parts. ``learning_rate`` is the rate *used* in that epoch and
    ``gradient_norm`` the mean pre-clipping gradient norm.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_mse: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf
    stopped_early: bool = False

    @property
    def epochs_run(self) -> int:
        """Epochs trained."""
        return len(self.train_loss)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        return {
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "train_mse": list(self.train_mse),
            "val_mse": list(self.val_mse),
            "train_kl": list(self.train_kl),
            "val_kl": list(self.val_kl),
            "learning_rate": list(self.learning_rate),
            "gradient_norm": list(self.gradient_norm),
            "epoch_seconds": list(self.epoch_seconds),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
            "epochs_run": self.epochs_run,
        }


def _make_writer(log_dir: Path) -> Any | None:
    """Open a TensorBoard writer, or ``None`` (with a warning) if TensorBoard is missing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        _log.warning(
            "tensorboard_unavailable", hint="pip install tensorboard", log_dir=str(log_dir)
        )
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


class LSTMAutoencoderTrainer:
    """Trains an :class:`LSTMAutoencoder` on normal windows.

    Objective ``MSE + kl_weight * KL`` with Adam, ``ReduceLROnPlateau`` on the validation loss,
    early stopping (the best weights are restored at the end) and gradient-norm clipping. Each
    epoch logs ``train_loss``, ``val_loss``, ``learning_rate`` and ``gradient_norm`` (plus the
    MSE / KL parts) to TensorBoard under ``log_dir/<run_name>``::

        tensorboard --logdir runs/lstm_autoencoder
    """

    def __init__(
        self,
        model: LSTMAutoencoder,
        *,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip_norm: float = 1.0,
        kl_weight: float = 0.1,
        lr_factor: float = 0.5,
        lr_patience: int = 4,
        min_lr: float = 1e-6,
        min_delta: float = 1e-6,
        device: str | torch.device | None = None,
        log_dir: str | Path | None = DEFAULT_LOG_DIR,
        run_name: str | None = None,
        seed: int = 42,
    ) -> None:
        """Create the trainer.

        Args:
            model: The model to train (moved to ``device``).
            learning_rate: Initial Adam learning rate.
            batch_size: Training batch size.
            max_epochs: Upper bound on epochs.
            patience: Epochs without validation improvement before stopping.
            grad_clip_norm: Maximum global gradient norm.
            kl_weight: Weight of the KL term.
            lr_factor: Learning-rate reduction factor of the plateau scheduler.
            lr_patience: Epochs without improvement before the learning rate is reduced (keep
                it below ``patience`` so the rate drops before training stops).
            min_lr: Lower bound of the learning rate.
            min_delta: Smallest validation-loss decrease that counts as an improvement.
            device: ``cpu`` / ``cuda``; automatic when ``None``.
            log_dir: TensorBoard root; ``None`` disables TensorBoard.
            run_name: Sub-directory of ``log_dir``; a timestamp when ``None``.
            seed: Seed of batch order, latent sampling and dropout (set at the start of ``fit``).

        Raises:
            ValueError: An argument is out of range.
        """
        if learning_rate <= 0 or batch_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("learning_rate, batch_size, max_epochs and patience must be positive")
        if grad_clip_norm <= 0 or kl_weight < 0 or not 0 < lr_factor < 1:
            raise ValueError("grad_clip_norm > 0, kl_weight >= 0 and 0 < lr_factor < 1 required")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.kl_weight = kl_weight
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.run_name = run_name
        self.seed = seed

    @classmethod
    def from_config(
        cls, cfg: LSTMAutoencoderConfig, model: LSTMAutoencoder, **overrides: Any
    ) -> LSTMAutoencoderTrainer:
        """Create the trainer from ``config.models.lstm_autoencoder``."""
        params: dict[str, Any] = {
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.early_stopping_patience,
            "grad_clip_norm": cfg.grad_clip_norm,
            "kl_weight": cfg.kl_weight,
        }
        params.update(overrides)
        return cls(model, **params)

    def evaluate(self, x: np.ndarray | torch.Tensor) -> tuple[float, float, float]:
        """Return ``(loss, mse, kl)`` of ``x`` in eval mode (``z = mu``, dropout off)."""
        tensor = _as_tensor(x)
        mse_sum = kl_sum = 0.0
        with torch.no_grad(), _inference_mode(self.model, mc_dropout=False):
            for (batch,) in DataLoader(TensorDataset(tensor), batch_size=self.batch_size * 4):
                batch = batch.to(self.device)
                recon, latent = self.model(batch)
                mse = torch_f.mse_loss(recon, batch).item()
                kl = kl_divergence(latent.mu, latent.logvar).item()
                mse_sum += mse * len(batch)
                kl_sum += kl * len(batch)
        n = len(tensor)
        mse_mean, kl_mean = mse_sum / n, kl_sum / n
        return mse_mean + self.kl_weight * kl_mean, mse_mean, kl_mean

    def fit(
        self, train_x: np.ndarray | torch.Tensor, val_x: np.ndarray | torch.Tensor
    ) -> TrainingHistory:
        """Train on ``train_x`` (normal windows), early-stopping on ``val_x`` (normal windows).

        The best-epoch weights are loaded back into the model before returning.

        Raises:
            ValueError: Inputs are empty, not finite or do not match the model's feature count.
            FloatingPointError: The loss became NaN or infinite.
        """
        train, val = _as_tensor(train_x), _as_tensor(val_x)
        if len(train) == 0 or len(val) == 0:
            raise ValueError("training and validation sets must not be empty")
        for name, tensor in (("train", train), ("validation", val)):
            if tensor.shape[-1] != self.model.n_features:
                raise ValueError(
                    f"{name} windows have {tensor.shape[-1]} features, model expects "
                    f"{self.model.n_features}"
                )
        torch.manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
        )
        run = self.run_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        writer = _make_writer(self.log_dir / run) if self.log_dir is not None else None
        history = TrainingHistory()
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        try:
            for epoch in range(1, self.max_epochs + 1):
                started = time.perf_counter()
                lr = float(optimizer.param_groups[0]["lr"])
                train_loss, train_mse, train_kl, grad_norm = self._train_epoch(loader, optimizer)
                val_loss, val_mse, val_kl = self.evaluate(val)
                scheduler.step(val_loss)
                history.train_loss.append(train_loss)
                history.val_loss.append(val_loss)
                history.train_mse.append(train_mse)
                history.val_mse.append(val_mse)
                history.train_kl.append(train_kl)
                history.val_kl.append(val_kl)
                history.learning_rate.append(lr)
                history.gradient_norm.append(grad_norm)
                history.epoch_seconds.append(time.perf_counter() - started)
                if writer is not None:
                    for tag, value in (
                        ("train_loss", train_loss),
                        ("val_loss", val_loss),
                        ("learning_rate", lr),
                        ("gradient_norm", grad_norm),
                        ("train_mse", train_mse),
                        ("val_mse", val_mse),
                        ("train_kl", train_kl),
                        ("val_kl", val_kl),
                    ):
                        writer.add_scalar(tag, value, epoch)
                _log.debug(
                    "lstm_autoencoder_epoch", epoch=epoch, train_loss=train_loss,
                    val_loss=val_loss, lr=lr, grad_norm=grad_norm,
                )  # fmt: skip
                if val_loss < history.best_val_loss - self.min_delta:
                    history.best_val_loss, history.best_epoch, stale = val_loss, epoch, 0
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
        best_kl = history.val_kl[history.best_epoch - 1] if history.best_epoch else 0.0
        if self.kl_weight > 0 and best_kl < 1e-3:
            _log.warning(
                "lstm_autoencoder_posterior_collapse",
                val_kl=best_kl,
                hint="the decoder ignores the input; lower kl_weight, widen the model or "
                "train longer",
            )
        _log.info(
            "lstm_autoencoder_trained",
            epochs=history.epochs_run,
            best_epoch=history.best_epoch,
            best_val_loss=round(history.best_val_loss, 6),
            stopped_early=history.stopped_early,
        )
        return history

    def _train_epoch(
        self, loader: DataLoader[Any], optimizer: torch.optim.Optimizer
    ) -> tuple[float, float, float, float]:
        """Run one epoch; return ``(loss, mse, kl, mean_gradient_norm)`` over its batches."""
        self.model.train()
        n = 0
        loss_sum = mse_sum = kl_sum = grad_sum = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            recon, latent = self.model(batch)
            mse = torch_f.mse_loss(recon, batch)
            kl = kl_divergence(latent.mu, latent.logvar)
            loss = mse + self.kl_weight * kl
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "loss is not finite; lower the learning rate or check the inputs"
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            optimizer.step()
            size = len(batch)
            n += size
            loss_sum += loss.item() * size
            mse_sum += mse.item() * size
            kl_sum += kl.item() * size
            grad_sum += float(norm)
            batches += 1
        return loss_sum / n, mse_sum / n, kl_sum / n, grad_sum / batches


# --------------------------------------------------------------------------- #
# Threshold calibration
# --------------------------------------------------------------------------- #
class CalibrationResult(NamedTuple):
    """What :meth:`AnomalyThresholdCalibrator.fit` returns."""

    calibrated_threshold: float
    threshold_per_feature: dict[str, float]


class AnomalyThresholdCalibrator:
    """Turns the reconstruction errors of **normal** windows into alarm thresholds.

    ``threshold = mean + sigma * std`` of the errors (``sigma = 3`` by default), overall and --
    from the time-averaged squared error of each feature -- per feature, which tells *which*
    features of a flagged window could not be reconstructed.

    Reconstruction errors are right-skewed, so this threshold is not "the 99.87th percentile":
    check the false-positive rate on held-out normal windows (the notebook does).
    """

    def __init__(self, sigma: float = 3.0, *, min_rel_spread: float = MIN_REL_SPREAD) -> None:
        """Create an unfitted calibrator.

        Args:
            sigma: Standard deviations above the mean.
            min_rel_spread: Floor of the standard deviation, as a share of the mean; keeps
                near-constant features from getting a threshold that rounding noise exceeds.

        Raises:
            ValueError: ``sigma`` is not positive or ``min_rel_spread`` is negative.
        """
        if sigma <= 0 or min_rel_spread < 0:
            raise ValueError("sigma must be positive and min_rel_spread non-negative")
        self.sigma = sigma
        self.min_rel_spread = min_rel_spread
        self._threshold: float | None = None
        self._per_feature: dict[str, float] = {}
        self._feature_names: tuple[str, ...] = ()
        self._per_feature_array: np.ndarray | None = None
        self.n_samples = 0

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run."""
        return self._threshold is not None

    @property
    def threshold(self) -> float:
        """The calibrated window-level threshold.

        Raises:
            NotFittedError: Not fitted yet.
        """
        if self._threshold is None:
            raise NotFittedError("AnomalyThresholdCalibrator is not fitted yet")
        return self._threshold

    @property
    def threshold_per_feature(self) -> dict[str, float]:
        """Per-feature thresholds (empty if fitted without per-feature errors)."""
        return dict(self._per_feature)

    def _bound(self, values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
        mean = values.mean(axis=axis)
        std = values.std(axis=axis, ddof=1)
        return mean + self.sigma * np.maximum(np.maximum(std, self.min_rel_spread * mean), _EPS)

    def fit(
        self,
        errors: np.ndarray,
        per_feature_errors: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> CalibrationResult:
        """Fit on the errors of normal validation windows.

        Args:
            errors: Window errors, shape ``(n,)``.
            per_feature_errors: Time-averaged squared error per feature, shape ``(n, F)``.
            feature_names: Names of the ``F`` features (``f0, f1, ...`` if omitted).

        Returns:
            ``(calibrated_threshold, threshold_per_feature)``.

        Raises:
            ValueError: Too few or non-finite errors, or shapes that do not match.
        """
        errs = np.asarray(errors, dtype=np.float64).ravel()
        if errs.size < MIN_CALIBRATION_SAMPLES:
            raise ValueError(f"need at least {MIN_CALIBRATION_SAMPLES} windows, got {errs.size}")
        if not np.isfinite(errs).all():
            raise ValueError("errors must be finite")
        per_feature: dict[str, float] = {}
        array: np.ndarray | None = None
        names: tuple[str, ...] = ()
        if per_feature_errors is not None:
            matrix = np.asarray(per_feature_errors, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != errs.size or not np.isfinite(matrix).all():
                raise ValueError("per_feature_errors must be finite with shape (n, n_features)")
            names = (
                tuple(feature_names)
                if feature_names is not None
                else tuple(f"f{i}" for i in range(matrix.shape[1]))
            )
            if len(names) != matrix.shape[1]:
                raise ValueError("feature_names must have one name per feature column")
            array = np.asarray(self._bound(matrix, axis=0), dtype=np.float64)
            per_feature = {name: float(v) for name, v in zip(names, array, strict=True)}
        self._threshold = float(self._bound(errs))
        self._per_feature, self._feature_names, self._per_feature_array = per_feature, names, array
        self.n_samples = int(errs.size)
        return CalibrationResult(self._threshold, dict(per_feature))

    def anomalous_features(self, per_feature_errors: np.ndarray) -> list[str]:
        """Names of the features whose error exceeds their threshold, worst first.

        Args:
            per_feature_errors: Errors of one window, shape ``(F,)``.

        Raises:
            NotFittedError: Not fitted, or fitted without per-feature errors.
        """
        if self._per_feature_array is None:
            raise NotFittedError("calibrator has no per-feature thresholds")
        ratio = np.asarray(per_feature_errors, dtype=np.float64) / self._per_feature_array
        flagged = np.flatnonzero(ratio > 1.0)
        return [self._feature_names[i] for i in flagged[np.argsort(-ratio[flagged])]]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly state."""
        return {
            "sigma": self.sigma,
            "min_rel_spread": self.min_rel_spread,
            "threshold": self._threshold,
            "feature_names": list(self._feature_names),
            "threshold_per_feature": dict(self._per_feature),
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnomalyThresholdCalibrator:
        """Rebuild from :meth:`to_dict` output."""
        calibrator = cls(float(data["sigma"]), min_rel_spread=float(data["min_rel_spread"]))
        if data.get("threshold") is not None:
            calibrator._threshold = float(data["threshold"])
            calibrator._feature_names = tuple(data.get("feature_names", ()))
            calibrator._per_feature = {
                k: float(v) for k, v in dict(data.get("threshold_per_feature", {})).items()
            }
            if calibrator._per_feature:
                calibrator._per_feature_array = np.array(
                    [calibrator._per_feature[n] for n in calibrator._feature_names]
                )
            calibrator.n_samples = int(data.get("n_samples", 0))
        return calibrator


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class SequenceSource(Protocol):
    """What the detector needs to read sequences: a ``SequenceBuilder`` (``pipeline.sequences``)."""

    store: FeatureStore
    window_seconds: tuple[int, ...]

    def build_sequence_with_mask(
        self, entity_id: str, lookback: int = 60, source_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the standardised ``(lookback, F)`` sequence and its real-step mask."""


@dataclass(frozen=True)
class SequenceScores:
    """Vectorised detector output for a batch of windows.

    Attributes:
        error: MC-mean reconstruction error per window.
        error_std: Spread of the error across the MC passes.
        score: ``error / (error + threshold)``; ``>= 0.5`` is an anomaly.
        is_anomaly: ``error > threshold``.
        confidence: Probability the verdict survives the dropout noise.
        per_feature: Time-averaged squared error per feature, ``(n, F)``.
        feature_exceeds: ``(n, F)`` bool, where a feature's error exceeds its threshold.
    """

    error: np.ndarray
    error_std: np.ndarray
    score: np.ndarray
    is_anomaly: np.ndarray
    confidence: np.ndarray
    per_feature: np.ndarray
    feature_exceeds: np.ndarray


class LSTMAutoencoderDetector:
    """Inference wrapper: window in, :class:`AnomalyScore` (``anomaly_type='collective'``) out.

    Offline::

        detector = LSTMAutoencoderDetector(model, AnomalyThresholdCalibrator(), names)
        detector.calibrate(x_val_normal)
        scores = detector.score_sequences(x_test)      # arrays, for evaluation

    Streaming (Day 15)::

        detector = LSTMAutoencoderDetector(model, calibrator, names, sequences=pipeline.sequences)
        result = detector.detect("wh1-A-temp-01")
        results = detector.detect_many(entity_ids)     # one batched forward pass

    Thread-safe: inference is serialised by a lock (MC Dropout briefly switches the dropout
    layers of the shared model into training mode).
    """

    def __init__(
        self,
        model: LSTMAutoencoder,
        calibrator: AnomalyThresholdCalibrator,
        feature_names: Sequence[str],
        *,
        source_type: str = "warehouse",
        sequences: SequenceSource | None = None,
        mc_passes: int = 10,
        min_real_fraction: float = MIN_REAL_FRACTION,
        require_warm: bool = True,
        batch_size: int = 256,
        columns: Sequence[int] | None = None,
    ) -> None:
        """Create the detector.

        Args:
            model: A trained autoencoder.
            calibrator: Fitted, or fitted later with :meth:`calibrate`.
            feature_names: Column names of the windows (the schema of ``source_type``).
            source_type: ``warehouse``, ``vehicle`` or ``inventory``.
            sequences: Where :meth:`detect` reads sequences (``FeaturePipeline.sequences``).
            mc_passes: MC Dropout passes (``1`` = deterministic, no uncertainty).
            min_real_fraction: Share of the window that must be real events (not padding).
            require_warm: Wait until the longest aggregation window of the entity is warm, as
                training only used warm rows.
            batch_size: Windows (times passes) per forward call.
            columns: Columns of the pipeline's feature vector the model was trained on
                (``select_features``); ``None`` means all. ``feature_names`` names these columns.

        Raises:
            ValueError: Bad ``mc_passes`` / ``min_real_fraction`` or a feature-count mismatch.
        """
        names = tuple(feature_names)
        if len(names) != model.n_features:
            raise ValueError(f"{len(names)} feature names for a model with {model.n_features}")
        if mc_passes < 1 or not 0.0 < min_real_fraction <= 1.0:
            raise ValueError("mc_passes must be >= 1 and min_real_fraction in (0, 1]")
        self.model = model
        self.calibrator = calibrator
        self.feature_names = names
        self.source_type = source_type
        self.sequences = sequences
        self.mc_passes = mc_passes
        self.min_real_fraction = min_real_fraction
        self.require_warm = require_warm
        self.columns = None if columns is None else tuple(int(c) for c in columns)
        self.batch_size = batch_size
        self._lock = threading.RLock()

    @property
    def seq_len(self) -> int:
        """Window length of the model."""
        return self.model.seq_len

    @property
    def min_real_steps(self) -> int:
        """Real (non-padding) steps a window needs before it is scored."""
        return math.ceil(self.min_real_fraction * self.seq_len)

    @property
    def is_calibrated(self) -> bool:
        """Whether a threshold exists."""
        return self.calibrator.is_fitted

    def calibrate(self, x_normal: np.ndarray | torch.Tensor) -> CalibrationResult:
        """Fit the threshold on normal validation windows, with this detector's MC procedure."""
        with self._lock:
            errors = reconstruction_errors(
                self.model, x_normal, mc_passes=self.mc_passes, batch_size=self.batch_size
            )
        return self.calibrator.fit(errors.error, errors.per_feature, self.feature_names)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_sequences(self, x: np.ndarray | torch.Tensor) -> SequenceScores:
        """Score a batch of windows ``(n, seq_len, n_features)``.

        Raises:
            NotFittedError: No threshold yet (call :meth:`calibrate`).
            ValueError: Bad shape or non-finite values.
        """
        threshold = self.calibrator.threshold
        with self._lock:
            errors = reconstruction_errors(
                self.model, x, mc_passes=self.mc_passes, batch_size=self.batch_size
            )
        error, spread = errors.error, errors.error_std
        score = error / (error + threshold)
        if self.mc_passes > 1:
            z = np.abs(error - threshold) / np.maximum(spread, 1e-3 * threshold)
            confidence = np.asarray(erf(z / math.sqrt(2.0)))
        else:
            confidence = np.minimum(1.0, np.abs(error / threshold - 1.0))
        per_feature_thr = self.calibrator.threshold_per_feature
        exceeds = (
            errors.per_feature > np.array([per_feature_thr[n] for n in self.feature_names])
            if per_feature_thr
            else np.zeros_like(errors.per_feature, dtype=bool)
        )
        return SequenceScores(
            error, spread, score, error > threshold, confidence, errors.per_feature, exceeds
        )

    def _to_anomaly_score(
        self,
        entity_id: str,
        timestamp: datetime,
        scores: SequenceScores,
        i: int,
        real_steps: int,
        raw_value: float | None,
    ) -> AnomalyScore:
        per_feature = scores.per_feature[i]
        names = (
            self.calibrator.anomalous_features(per_feature)
            if scores.feature_exceeds[i].any()
            else []
        )
        top = np.argsort(-per_feature)[:5]
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=timestamp,
            score=float(scores.score[i]),
            is_anomaly=bool(scores.is_anomaly[i]),
            anomaly_type="collective",
            confidence=float(scores.confidence[i]),
            detector=DETECTOR_NAME,
            raw_value=raw_value,
            details={
                "reconstruction_error": float(scores.error[i]),
                "error_std": float(scores.error_std[i]),
                "threshold": self.calibrator.threshold,
                "which_features_anomalous": names,
                "top_feature_errors": {self.feature_names[j]: float(per_feature[j]) for j in top},
                "mc_passes": self.mc_passes,
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
    ) -> AnomalyScore:
        """Score one ready-made window ``(seq_len, n_features)``."""
        scores = self.score_sequences(_as_tensor(sequence))
        return self._to_anomaly_score(
            entity_id,
            timestamp,
            scores,
            0,
            self.seq_len if real_steps is None else real_steps,
            None,
        )

    def detect(self, entity_id: str) -> AnomalyScore:
        """Read the entity's window from the feature store and score it.

        Returns a warm-up result (``score=0``, ``details["warming_up"] = True``) while the entity
        has too little warm history; never raises for unknown entities.

        Raises:
            RuntimeError: The detector was created without ``sequences``.
            NotFittedError: No threshold yet.
        """
        return self.detect_many([entity_id])[0]

    def detect_many(self, entity_ids: Sequence[str]) -> list[AnomalyScore]:
        """Micro-batch version of :meth:`detect` (one forward pass for all ready entities).

        Raises:
            RuntimeError: The detector was created without ``sequences``.
            NotFittedError: No threshold yet.
        """
        if self.sequences is None:
            raise RuntimeError("detect() needs `sequences` (pass FeaturePipeline.sequences)")
        _ = self.calibrator.threshold  # fail early (NotFittedError) before touching the store
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
                results[slot] = self._warming_up(entity_id, latest, real)
                continue
            primary = next(iter(latest.signals.values()), None)
            ready.append((slot, latest.window_end, real, primary.last if primary else None))
            windows.append(sequence if self.columns is None else sequence[:, list(self.columns)])
        if windows:
            scores = self.score_sequences(torch.stack(windows))
            for k, (slot, stamp, real, raw) in enumerate(ready):
                results[slot] = self._to_anomaly_score(
                    entity_ids[slot], stamp, scores, k, real, raw
                )
        return [r for r in results if r is not None]

    def _warming_up(self, entity_id: str, latest: Any, real_steps: int) -> AnomalyScore:
        return AnomalyScore(
            entity_id=entity_id,
            timestamp=latest.window_end if latest is not None else datetime.now(UTC),
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
        """Return a plain dict (weights, architecture, threshold, schema) for ``torch.save``."""
        return {
            "model_config": self.model.get_config(),
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "calibrator": self.calibrator.to_dict(),
            "feature_names": list(self.feature_names),
            "source_type": self.source_type,
            "mc_passes": self.mc_passes,
            "min_real_fraction": self.min_real_fraction,
            "require_warm": self.require_warm,
            "columns": None if self.columns is None else list(self.columns),
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any], *, sequences: SequenceSource | None = None
    ) -> LSTMAutoencoderDetector:
        """Rebuild a detector from :meth:`to_checkpoint` output."""
        config = dict(checkpoint["model_config"])
        n_features = int(config.pop("n_features"))
        config["hidden_sizes"] = tuple(config["hidden_sizes"])
        model = LSTMAutoencoder(n_features, **config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return cls(
            model,
            AnomalyThresholdCalibrator.from_dict(checkpoint["calibrator"]),
            checkpoint["feature_names"],
            source_type=checkpoint["source_type"],
            sequences=sequences,
            mc_passes=int(checkpoint["mc_passes"]),
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
        """Store this detector as a new version of ``lstm_autoencoder`` in the registry."""
        return registry.save(
            MODEL_TYPE,
            self.to_checkpoint(),
            metrics=metrics,
            hyperparams={**self.model.get_config(), **(hyperparams or {})},
            extra={"feature_names": list(self.feature_names), **(extra or {})},
            serializer="torch",
        )

    @classmethod
    def load(
        cls,
        registry: ModelRegistry,
        version: str | None = None,
        *,
        sequences: SequenceSource | None = None,
    ) -> LSTMAutoencoderDetector:
        """Load a version (default: the latest) saved with :meth:`save`."""
        loaded = registry.load(MODEL_TYPE, version)
        return cls.from_checkpoint(loaded.model, sequences=sequences)


# --------------------------------------------------------------------------- #
# One-call training
# --------------------------------------------------------------------------- #
def train_detector(
    train_x: np.ndarray,
    val_x: np.ndarray,
    feature_names: Sequence[str],
    cfg: LSTMAutoencoderConfig,
    *,
    source_type: str = "warehouse",
    sequences: SequenceSource | None = None,
    log_dir: str | Path | None = DEFAULT_LOG_DIR,
    run_name: str | None = None,
    device: str | torch.device | None = None,
    columns: Sequence[int] | None = None,
) -> tuple[LSTMAutoencoderDetector, TrainingHistory]:
    """Train on normal windows and calibrate on normal validation windows.

    Both inputs must contain **only normal** windows (``SequenceSet.normal``); the validation
    set drives early stopping and the threshold. Report final numbers on a third, later period.

    Args:
        train_x: Normal training windows ``(n, seq_len, n_features)``.
        val_x: Normal validation windows.
        feature_names: Column names of the windows.
        cfg: ``config.models.lstm_autoencoder``.
        source_type: Source type the detector serves.
        sequences: Sequence source for ``detect()`` (optional).
        log_dir: TensorBoard root (``None`` disables it).
        run_name: TensorBoard run name.
        device: Torch device; automatic when ``None``.
        columns: Columns of the pipeline's feature vector the windows were cut with
            (see :func:`select_features`); needed for ``detect()`` to read the same columns.

    Returns:
        ``(detector, history)``; the detector is calibrated and holds the best weights.

    Raises:
        ValueError: The windows are not ``(n, seq_len, n_features)`` arrays.
    """
    windows = np.asarray(train_x)
    if windows.ndim != 3:
        raise ValueError("train_x must have shape (n, seq_len, n_features)")
    model = LSTMAutoencoder(
        windows.shape[2],
        seq_len=windows.shape[1],
        hidden_sizes=cfg.hidden_sizes,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    )
    trainer = LSTMAutoencoderTrainer.from_config(
        cfg, model, log_dir=log_dir, run_name=run_name, device=device
    )
    history = trainer.fit(train_x, val_x)
    detector = LSTMAutoencoderDetector(
        trainer.model,
        AnomalyThresholdCalibrator(cfg.threshold_sigma),
        feature_names,
        source_type=source_type,
        sequences=sequences,
        mc_passes=cfg.mc_dropout_passes,
        columns=columns,
    )
    detector.calibrate(val_x)
    return detector, history


__all__ = [
    "DETECTOR_NAME",
    "AnomalyThresholdCalibrator",
    "CalibrationResult",
    "LSTMAutoencoder",
    "LSTMAutoencoderDetector",
    "LSTMAutoencoderTrainer",
    "LatentState",
    "ReconstructionErrors",
    "SequenceScores",
    "SequenceSet",
    "SequenceSource",
    "TEMPORAL_FEATURES",
    "TrainingHistory",
    "kl_divergence",
    "make_sequences",
    "reconstruction_errors",
    "select_features",
    "train_detector",
]
