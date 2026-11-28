"""Every detector must reach F1 > 0.7 on a labelled stream (Day 23).

Each detector gets a stream with *known* anomalies, runs it through the same entry points
production uses, and is scored against the labels:

* statistical detectors  - a mock sensor stream with point spikes and gradual drifts, scored
  point-wise (Z-score, IQR) and per *incident* (CUSUM and the unified detector, whose alarm
  deliberately lingers for a few readings after an anomaly, see notebook 02);
* Isolation Forest       - a feature stream fed through ``update_batch`` (warm-up, fit, refits),
  plus the real simulator -> aggregator -> feature pipeline;
* LSTM autoencoder       - sinusoidal windows, half of them with a drifting feature;
* LSTM classifier        - noise windows with a spike (point) or a level shift (contextual).

The streams are synthetic and seeded, so the F1 floor checks that the detector logic works, not that
it solves the simulator's hardest cases; those numbers live in notebooks 02-07.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from src.models.feature_engineering import OnlineScaler
from src.models.isolation_forest_detector import IsolationForestDetector
from src.models.lstm_autoencoder import train_detector as train_autoencoder
from src.models.lstm_classifier import (
    ClassifierData,
    LabeledSequences,
    balanced_class_weights,
    train_classifier,
)
from src.models.offline_dataset import simulate_labeled_features
from src.models.statistical_detector import (
    CUSUMDetector,
    IQRDetector,
    UnifiedStatisticalDetector,
    ZScoreDetector,
)
from src.utils.config import LSTMAutoencoderConfig, LSTMClassifierConfig

F1_FLOOR = 0.7
START = datetime(2026, 3, 10, 6, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the ``(first, last)`` index of every contiguous run of ``True``."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


@dataclass(frozen=True)
class IncidentScore:
    """Incident-level precision, recall and F1."""

    precision: float
    recall: float
    f1: float


def incident_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 10) -> IncidentScore:
    """Score what an operator sees (the view used in ``notebooks/02_statistical_baseline_eval``).

    A true event is *detected* if a flag run overlaps it or starts within ``tolerance`` readings
    after it ends; a flag run is a *false alarm* if it is not near any true event.
    """
    events, flags = runs(y_true), runs(y_pred)
    detected = sum(any(f0 <= e1 + tolerance and f1 >= e0 for f0, f1 in flags) for e0, e1 in events)
    matched = sum(any(f0 <= e1 + tolerance and f1 >= e0 for e0, e1 in events) for f0, f1 in flags)
    recall = detected / len(events) if events else 0.0
    precision = matched / len(flags) if flags else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return IncidentScore(precision, recall, f1)


def point_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Reading-by-reading F1 of the positive (anomalous) class."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def test_incident_f1_scores_what_the_operator_sees():
    truth = np.zeros(100, dtype=bool)
    truth[20:23] = truth[60:62] = True
    perfect = incident_f1(truth, truth)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)

    late = np.zeros(100, dtype=bool)
    late[28:30] = True  # fires 6 readings after the first event ended: within tolerance
    late[90:92] = True  # nowhere near an event: false alarm
    score = incident_f1(truth, late)
    assert score.recall == 0.5 and score.precision == 0.5

    assert incident_f1(truth, np.zeros(100, dtype=bool)).f1 == 0.0
    assert incident_f1(np.zeros(100, dtype=bool), truth).f1 == 0.0


# --------------------------------------------------------------------------- #
# Statistical detectors
# --------------------------------------------------------------------------- #
WARMUP = 600


@dataclass(frozen=True)
class SensorStream:
    """A labelled one-sensor stream (1 reading / second)."""

    values: np.ndarray
    is_anomaly: np.ndarray
    kind: np.ndarray

    def timestamp(self, i: int) -> datetime:
        return START + timedelta(seconds=i)


@pytest.fixture(scope="module")
def sensor_stream() -> SensorStream:
    """Cold-storage-like readings (3 degC, sd 0.15) with 40 spikes and several 25-reading drifts."""
    rng = np.random.default_rng(0)
    n = 6 * 3600
    values = rng.normal(3.0, 0.15, n)
    anomalous = np.zeros(n, dtype=bool)
    kind = np.full(n, "", dtype=object)
    for s in rng.choice(np.arange(WARMUP, n - 10), 40, replace=False):
        length = int(rng.integers(1, 4))
        values[s : s + length] += rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
        anomalous[s : s + length] = True
        kind[s : s + length] = "point"
    for s in range(1500, n - 600, 2500):
        values[s : s + 25] += np.linspace(0, 2.0, 25)
        anomalous[s : s + 25] = True
        kind[s : s + 25] = "collective"
    return SensorStream(values, anomalous, kind)


def flags_from(detector, stream: SensorStream) -> np.ndarray:
    """Feed the stream through ``detector.update`` and collect the ``is_anomaly`` decisions."""
    return np.array(
        [
            detector.update("sensor-1", float(v), stream.timestamp(i), "A").is_anomaly
            for i, v in enumerate(stream.values)
        ]
    )


class TestStatisticalDetectors:
    def test_zscore_f1(self, sensor_stream):
        pred = flags_from(ZScoreDetector(threshold=3.5), sensor_stream)
        assert point_f1(sensor_stream.is_anomaly[WARMUP:], pred[WARMUP:]) > F1_FLOOR

    def test_iqr_f1(self, sensor_stream):
        pred = flags_from(IQRDetector(multiplier=2.5), sensor_stream)
        assert point_f1(sensor_stream.is_anomaly[WARMUP:], pred[WARMUP:]) > F1_FLOOR

    def test_cusum_detects_every_drift_and_scores_above_the_floor(self, sensor_stream):
        pred = flags_from(CUSUMDetector(k=0.75, h=10.0), sensor_stream)
        truth = sensor_stream.is_anomaly
        score = incident_f1(truth[WARMUP:], pred[WARMUP:])
        assert score.f1 > F1_FLOOR
        drift = sensor_stream.kind[WARMUP:] == "collective"
        drift_events = runs(drift)
        assert all(pred[WARMUP:][s : e + 11].any() for s, e in drift_events)

    def test_unified_detector_incident_f1(self, sensor_stream):
        detector = UnifiedStatisticalDetector.production()
        pred = np.array(
            [
                detector.detect("sensor-1", float(v), sensor_stream.timestamp(i), "A").is_anomaly
                for i, v in enumerate(sensor_stream.values)
            ]
        )
        score = incident_f1(sensor_stream.is_anomaly[WARMUP:], pred[WARMUP:])
        assert score.f1 > F1_FLOOR and score.recall > 0.9

    def test_unified_detector_stays_quiet_on_a_clean_stream(self):
        rng = np.random.default_rng(1)
        detector = UnifiedStatisticalDetector.production()
        flags = [
            detector.detect("s", float(v), START + timedelta(seconds=i), "A").is_anomaly
            for i, v in enumerate(rng.normal(3.0, 0.15, 3000))
        ]
        assert np.mean(flags[WARMUP:]) < 0.01


# --------------------------------------------------------------------------- #
# Isolation Forest
# --------------------------------------------------------------------------- #
FEATURES = ["value__last", "value__z_baseline", "f2", "f3", "f4", "f5"]


def feature_stream(n: int, clean_head: int, rate: float, seed: int):
    """Gaussian feature vectors; after ``clean_head`` rows a share is shifted by 5-8 sigma."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, len(FEATURES)))
    y = np.zeros(n, dtype=bool)
    idx = rng.choice(np.arange(clean_head, n), int((n - clean_head) * rate), replace=False)
    sign = rng.choice([-1, 1], (len(idx), 1))
    features[np.ix_(idx, [1, 2, 3])] += sign * rng.uniform(5, 8, (len(idx), 3))
    y[idx] = True
    return features, y


class TestIsolationForest:
    def test_streaming_f1_through_warmup_fit_and_refits(self):
        features, y = feature_stream(4000, clean_head=1000, rate=0.03, seed=0)
        detector = IsolationForestDetector(
            FEATURES, contamination=0.03, warmup_samples=1000, refit_every=1000
        )
        stamps = [START + timedelta(seconds=i) for i in range(len(features))]
        scores = detector.update_batch(["s1"] * len(features), stamps, features)

        assert all(s.details["warming_up"] for s in scores[:1000])
        assert detector.model_version >= 3  # first fit plus refits while streaming
        pred = np.array([s.is_anomaly for s in scores])
        assert point_f1(y[1000:], pred[1000:]) > F1_FLOOR

    def test_real_pipeline_features_catch_point_anomalies(self):
        """Simulator -> aggregator -> features -> IF: the spikes the simulator injects are found."""
        start = datetime(2026, 3, 10, 8, 0, tzinfo=UTC)
        data = simulate_labeled_features(2 * 3600, start_time=start, seed=3)["warehouse"]
        cut = data.timestamps.min() + 3600
        train = data.warm & (data.timestamps < cut) & ~data.y
        test = data.warm & (data.timestamps >= cut)
        detector = IsolationForestDetector(list(data.feature_names), contamination=0.02)
        detector.fit(data.X[train])
        flagged = detector.predict(data.X[test])
        points = data.anomaly_type[test] == "point"

        assert points.sum() > 100
        assert flagged[points].mean() > 0.9  # recall on point anomalies
        assert flagged.mean() < 0.15  # and it does not flag everything


# --------------------------------------------------------------------------- #
# LSTM autoencoder (collective anomalies)
# --------------------------------------------------------------------------- #
SEQ_LEN, N_FEATURES = 20, 4
AE_NAMES = tuple(f"feat{i}" for i in range(N_FEATURES))


def sine_windows(n: int, seed: int) -> np.ndarray:
    """Normal behaviour: noisy sinusoids with a random phase per window."""
    rng = np.random.default_rng(seed)
    t = np.arange(SEQ_LEN)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 1))
    freq = np.array([1.0, 2.0, 3.0, 4.0])[None, None, :] * 2 * np.pi / SEQ_LEN
    return (np.sin(t * freq + phase) + rng.normal(0, 0.05, (n, SEQ_LEN, N_FEATURES))).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def autoencoder():
    torch.manual_seed(0)
    cfg = LSTMAutoencoderConfig(
        seq_len=SEQ_LEN, hidden_sizes=(16, 32), latent_dim=8, max_epochs=40,
        early_stopping_patience=10, batch_size=32, learning_rate=5e-3, kl_weight=0.01,
        mc_dropout_passes=10,
    )  # fmt: skip
    detector, _ = train_autoencoder(
        sine_windows(400, 1), sine_windows(120, 2), AE_NAMES, cfg, log_dir=None
    )
    return detector


def test_lstm_autoencoder_f1_on_labelled_windows(autoencoder):
    normal = sine_windows(200, 3)
    drift = sine_windows(100, 4)
    drift[:, :, 0] += 0.25 * np.arange(SEQ_LEN)[None, :]  # gradual degradation of one feature
    x = np.concatenate([normal, drift])
    y = np.concatenate([np.zeros(len(normal), dtype=bool), np.ones(len(drift), dtype=bool)])

    predicted = autoencoder.score_sequences(x).is_anomaly
    assert point_f1(y, predicted) > F1_FLOOR


# --------------------------------------------------------------------------- #
# LSTM classifier (point and contextual anomalies)
# --------------------------------------------------------------------------- #
CLS_LEN, CLS_FEATURES = 12, 4
CLASSES = ("normal", "point", "contextual")


def class_windows(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Noise windows; 'point' ends in a spike on feature 0, 'contextual' in a shift of feature 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, (3 * n_per_class, CLS_LEN, CLS_FEATURES)).astype(np.float32)
    y = np.repeat(np.arange(3), n_per_class)
    x[y == 1, -1, 0] += 4.0
    x[y == 2, -4:, 1] += 2.0
    order = rng.permutation(len(y))
    return x[order], y[order]


@pytest.fixture(scope="module")
def classifier():
    torch.manual_seed(0)
    x, y = class_windows(120, seed=1)
    n = len(y)
    a, b = int(0.7 * n), int(0.85 * n)

    def part(sl: slice) -> LabeledSequences:
        k = len(y[sl])
        return LabeledSequences(
            x[sl], y[sl], np.array(["e"] * k, dtype=object), np.zeros(k), np.zeros(k, dtype=bool)
        )

    data = ClassifierData(
        CLASSES, tuple(f"feat{i}" for i in range(CLS_FEATURES)), None, OnlineScaler(CLS_FEATURES),
        part(slice(0, a)), part(slice(a, b)), part(slice(b, n)), balanced_class_weights(y[:a], 3),
    )  # fmt: skip
    cfg = LSTMClassifierConfig(
        seq_len=CLS_LEN, hidden_size=12, max_epochs=25, early_stopping_patience=8,
        batch_size=32, learning_rate=5e-3, dropout=0.1,
    )  # fmt: skip
    detector, _ = train_classifier(data, cfg, log_dir=None, num_workers=0, run_name="detectors")
    return detector


def test_lstm_classifier_f1_on_labelled_windows(classifier):
    x, y = class_windows(100, seed=9)  # unseen windows, balanced classes
    out = classifier.classify_sequences(x)

    assert point_f1(y > 0, out.predicted > 0) > F1_FLOOR  # anomaly vs normal
    per_class = f1_score(y, out.predicted, average=None, labels=[0, 1, 2], zero_division=0)
    assert per_class.min() > F1_FLOOR  # normal, point and contextual each separately
