"""Unit tests for the Day 11 LSTM classifier (model, data, SMOTE, trainer, calibration)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.feature_engineering import OnlineScaler
from src.models.lstm_classifier import (
    DatasetBuilder,
    LSTMClassifier,
    LSTMClassifierDetector,
    LSTMClassifierTrainer,
    SequenceDataset,
    TemperatureScaler,
    balanced_class_weights,
    encode_labels,
    evaluate_predictions,
    expected_calibration_error,
    group_stratified_split,
    make_loader,
    predict_logits,
    reliability_bins,
    smote_oversample,
    train_classifier,
)
from src.models.model_registry import ModelRegistry
from src.models.offline_dataset import LabeledFeatureSet
from src.utils.config import LSTMClassifierConfig

T, F, K = 12, 4, 3
NAMES = tuple(f"feat{i}" for i in range(F))
CLASSES = ("normal", "point", "contextual")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def toy_windows(n_per_class: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noise windows; a 'point' window ends in a spike on feature 0, a 'contextual' one in a
    sustained shift of feature 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, (3 * n_per_class, T, F)).astype(np.float32)
    y = np.repeat(np.arange(3), n_per_class)
    x[y == 1, -1, 0] += 4.0
    x[y == 2, -4:, 1] += 2.0
    order = rng.permutation(len(y))
    return x[order], y[order]


def fast_cfg(**over) -> LSTMClassifierConfig:
    base = {"seq_len": T, "hidden_size": 12, "max_epochs": 25, "early_stopping_patience": 8,
            "batch_size": 32, "learning_rate": 5e-3, "dropout": 0.1}  # fmt: skip
    base.update(over)
    return LSTMClassifierConfig(**base)


def make_data(**over):
    x, y = toy_windows(80)
    n = len(y)
    cut1, cut2 = int(0.7 * n), int(0.85 * n)
    return x, y, (slice(0, cut1), slice(cut1, cut2), slice(cut2, n)), over


@pytest.fixture(scope="module")
def trained():
    """A small detector trained once on the toy windows."""
    from src.models.lstm_classifier import ClassifierData, LabeledSequences

    x, y = toy_windows(120, seed=1)
    n = len(y)
    a, b = int(0.7 * n), int(0.85 * n)

    def part(sl: slice) -> LabeledSequences:
        k = len(y[sl])
        return LabeledSequences(
            x[sl], y[sl], np.array(["e"] * k, dtype=object), np.zeros(k), np.zeros(k, dtype=bool)
        )

    data = ClassifierData(
        CLASSES, NAMES, None, OnlineScaler(F), part(slice(0, a)), part(slice(a, b)),
        part(slice(b, n)), balanced_class_weights(y[:a], 3),
    )  # fmt: skip
    detector, history = train_classifier(
        data, fast_cfg(), log_dir=None, num_workers=0, run_name="test"
    )
    return detector, history, data


# --------------------------------------------------------------------------- model
class TestModel:
    def test_shapes_and_attention(self):
        model = LSTMClassifier(F, seq_len=T, hidden_size=8).eval()
        logits, attn = model(torch.randn(5, T, F))
        assert logits.shape == (5, 3) and attn.shape == (5, T)
        assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-5)
        assert bool((attn >= 0).all())

    def test_spec_architecture(self):
        model = LSTMClassifier(F)
        assert model.lstm.bidirectional and model.lstm.num_layers == 2
        assert model.lstm.hidden_size == 64
        linear = [m for m in model.head if isinstance(m, torch.nn.Linear)]
        assert (linear[0].in_features, linear[0].out_features) == (128, 64)
        assert (linear[1].in_features, linear[1].out_features) == (64, 3)
        drop = next(m for m in model.head if isinstance(m, torch.nn.Dropout))
        assert drop.p == 0.3

    def test_accepts_other_sequence_lengths(self):
        logits, attn = LSTMClassifier(F, hidden_size=8).eval()(torch.randn(2, T + 5, F))
        assert logits.shape == (2, 3) and attn.shape == (2, T + 5)

    def test_eval_is_deterministic_train_uses_dropout(self):
        model, x = LSTMClassifier(F, hidden_size=8, dropout=0.5), torch.randn(4, T, F)
        model.eval()
        assert torch.equal(model(x)[0], model(x)[0])
        model.train()
        assert not torch.equal(model(x)[0], model(x)[0])

    def test_rejects_bad_input_and_args(self):
        with pytest.raises(ValueError, match="expected input"):
            LSTMClassifier(F)(torch.randn(2, T, F + 1))
        with pytest.raises(ValueError):
            LSTMClassifier(F, dropout=1.0)
        with pytest.raises(ValueError):
            LSTMClassifier(F, n_classes=1)
        with pytest.raises(ValueError):
            LSTMClassifier(0)

    def test_from_config_and_get_config_roundtrip(self):
        model = LSTMClassifier.from_config(fast_cfg(), F)
        rebuilt = LSTMClassifier(**model.get_config())
        assert rebuilt.get_config() == model.get_config()
        assert model.seq_len == T and model.hidden_size == 12

    def test_predict_logits_restores_mode_and_validates(self):
        model = LSTMClassifier(F, hidden_size=8)
        model.train()
        logits, attn = predict_logits(model, np.random.randn(7, T, F).astype(np.float32))
        assert model.training and logits.shape == (7, 3) and attn.shape == (7, T)
        with pytest.raises(ValueError, match="NaN"):
            predict_logits(model, np.full((1, T, F), np.nan, dtype=np.float32))
        with pytest.raises(ValueError):
            predict_logits(model, np.zeros((T,), dtype=np.float32))
        assert predict_logits(model, np.empty((0, T, F), np.float32))[0].shape == (0, 3)

    def test_gradients_flow_to_attention_and_lstm(self):
        model = LSTMClassifier(F, hidden_size=8)
        logits, _ = model(torch.randn(3, T, F))
        logits.sum().backward()
        assert all(
            p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.attn_key.parameters()
        )
        assert model.lstm.weight_ih_l0.grad is not None


# --------------------------------------------------------------------------- labels / weights
class TestLabels:
    def test_encode_and_drop_collective(self):
        y = np.array([False, True, True, True, False])
        kinds = np.array([None, "point", "contextual", "collective", None], dtype=object)
        assert encode_labels(y, kinds, CLASSES).tolist() == [0, 1, 2, -1, 0]
        assert encode_labels(y, kinds, CLASSES, collective="normal").tolist() == [0, 1, 2, 0, 0]

    def test_collective_class_when_named(self):
        y, kinds = np.array([True]), np.array(["collective"], dtype=object)
        assert encode_labels(y, kinds, (*CLASSES, "collective")).tolist() == [3]

    def test_unknown_type_and_bad_names(self):
        with pytest.raises(ValueError, match="unknown anomaly type"):
            encode_labels(np.array([True]), np.array(["bogus"], dtype=object), CLASSES)
        with pytest.raises(ValueError, match="start with 'normal'"):
            encode_labels(np.array([False]), np.array([None], dtype=object), ("point", "normal"))

    def test_balanced_weights_are_inverse_frequency(self):
        w = balanced_class_weights(np.array([0] * 90 + [1] * 9 + [2] * 1), 3)
        assert w[0] < w[1] < w[2]
        assert w * np.array([90, 9, 1]) == pytest.approx(np.full(3, 100 / 3))
        with pytest.raises(ValueError, match="without training samples"):
            balanced_class_weights(np.array([0, 0, 1]), 3)


# --------------------------------------------------------------------------- SMOTE
class TestSmote:
    def test_raises_minority_to_ratio_of_majority(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(110, 6)).astype(np.float32)
        y = np.array([0] * 100 + [1] * 10)
        out = smote_oversample(x, y, ratio=0.5, seed=1)
        assert np.bincount(out.y).tolist() == [100, 50]
        assert out.synthetic.sum() == 40 and not out.synthetic[:110].any()
        assert np.array_equal(out.x[:110], x) and out.x.dtype == np.float32

    def test_synthetic_points_lie_in_the_class_neighbourhood(self):
        rng = np.random.default_rng(0)
        majority = rng.normal(0, 1, (200, 3))
        minority = rng.normal(10, 0.5, (12, 3))
        out = smote_oversample(
            np.vstack([majority, minority]), np.array([0] * 200 + [1] * 12), ratio=1
        )
        synthetic = out.x[out.synthetic]
        assert len(synthetic) == 188
        # interpolations of minority points stay inside their bounding box
        assert (synthetic >= minority.min(axis=0) - 1e-5).all()
        assert (synthetic <= minority.max(axis=0) + 1e-5).all()

    def test_deterministic_for_a_seed(self):
        x = np.random.default_rng(0).normal(size=(60, 4))
        y = np.array([0] * 50 + [1] * 10)
        a, b = smote_oversample(x, y, seed=3), smote_oversample(x, y, seed=3)
        assert np.array_equal(a.x, b.x)

    def test_single_sample_class_is_duplicated_and_ratio_zero_is_noop(self):
        x = np.arange(24, dtype=np.float32).reshape(12, 2)
        y = np.array([0] * 11 + [1])
        out = smote_oversample(x, y, ratio=0.5)
        assert np.bincount(out.y).tolist() == [11, 6]
        assert (out.x[out.synthetic] == x[-1]).all()
        assert len(smote_oversample(x, y, ratio=0.0).y) == 12

    def test_validation(self):
        with pytest.raises(ValueError):
            smote_oversample(np.zeros((4, 2, 2)), np.zeros(4))
        with pytest.raises(ValueError):
            smote_oversample(np.zeros((4, 2)), np.zeros(4), ratio=1.5)


# --------------------------------------------------------------------------- split
class TestSplit:
    def test_groups_stay_together_and_classes_are_stratified(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(200), 5)
        labels = np.zeros(1000, dtype=np.int64)
        for g in range(0, 200, 10):  # 10% of groups hold a class-1 window
            labels[g * 5] = 1
        which = group_stratified_split(groups, labels, seed=1)
        for g in range(200):
            assert len(set(which[groups == g])) == 1
        shares = np.bincount(which) / len(which)
        assert shares == pytest.approx([0.7, 0.15, 0.15], abs=0.05)
        for part in range(3):
            assert (labels[which == part] == 1).any()
        assert rng is not None

    def test_too_few_groups_gives_a_helpful_error(self):
        groups = np.arange(10)
        labels = np.array([0] * 9 + [1])
        with pytest.raises(ValueError, match="too few groups"):
            group_stratified_split(groups, labels)

    def test_bad_fractions(self):
        with pytest.raises(ValueError):
            group_stratified_split(np.arange(4), np.zeros(4, dtype=int), (0.7, 0.3, 0.0))


# --------------------------------------------------------------------------- datasets
class TestDatasets:
    def test_sequence_dataset_and_loader(self):
        x, y = toy_windows(10)
        ds = SequenceDataset(x, y)
        assert len(ds) == 30 and ds[0][0].shape == (T, F) and ds[0][1].dtype == torch.long
        batch_x, batch_y = next(iter(make_loader(ds, batch_size=8, num_workers=0)))
        assert batch_x.shape == (8, T, F) and batch_y.shape == (8,)
        with pytest.raises(ValueError):
            SequenceDataset(x, y[:-1])

    def test_loader_shuffle_is_seeded(self):
        ds = SequenceDataset(*toy_windows(10))
        a = next(iter(make_loader(ds, batch_size=30, shuffle=True, num_workers=0, seed=5)))[1]
        b = next(iter(make_loader(ds, batch_size=30, shuffle=True, num_workers=0, seed=5)))[1]
        assert torch.equal(a, b)

    def test_loader_with_workers_matches_spec(self):
        loader = make_loader(SequenceDataset(*toy_windows(10)), num_workers=2, pin_memory=True)
        assert loader.num_workers == 2 and loader.pin_memory is True


def synthetic_rows(n_entities: int = 4, n_rows: int = 3000, seed: int = 0) -> LabeledFeatureSet:
    """Labelled feature rows: per-entity series with point spikes and contextual shifts."""
    rng = np.random.default_rng(seed)
    xs, ys, kinds, ents, ts = [], [], [], [], []
    for e in range(n_entities):
        x = rng.normal(0, 1, (n_rows, F))
        y = np.zeros(n_rows, dtype=bool)
        kind = np.full(n_rows, None, dtype=object)
        for start in rng.choice(np.arange(60, n_rows - 20), size=60, replace=False):
            label = "point" if rng.random() < 0.5 else "contextual"
            if label == "point":
                x[start, 0] += 6
                y[start], kind[start] = True, "point"
            else:
                x[start : start + 4, 1] += 3
                y[start : start + 4], kind[start : start + 4] = True, "contextual"
        for start in rng.choice(np.arange(60, n_rows - 40), size=3, replace=False):
            y[start : start + 20], kind[start : start + 20] = True, "collective"
        xs.append(x), ys.append(y), kinds.append(kind)
        ents.append(np.full(n_rows, f"s{e}", dtype=object))
        ts.append(1.7e9 + np.arange(n_rows, dtype=np.float64))
    order = np.argsort(np.concatenate(ts), kind="stable")
    cat = lambda parts: np.concatenate(parts)[order]  # noqa: E731
    n = n_entities * n_rows
    return LabeledFeatureSet(
        "warehouse", NAMES, cat(xs), cat(ys), cat(kinds), cat(ents), cat(ts),
        np.full(n, None, dtype=object), np.ones(n, dtype=bool),
    )  # fmt: skip


class TestDatasetBuilder:
    @pytest.fixture(scope="class")
    def data(self):
        return DatasetBuilder(seq_len=T, stride=3, smote_ratio=0.5, group_blocks=10, seed=1).build(
            synthetic_rows()
        )

    def test_shapes_names_and_scaler(self, data):
        assert data.train.x.shape[1:] == (T, F) and data.n_features == F and data.seq_len == T
        assert data.class_names == CLASSES and data.feature_names == NAMES
        assert data.scaler.n_features == F and data.scaler.count.min() > 0

    def test_no_collective_windows_and_split_proportions(self, data):
        assert data.info["n_collective_dropped"] > 0
        real = {"train": data.train_real, "val": data.val, "test": data.test}
        total = sum(len(p) for p in real.values())
        assert len(real["train"]) / total == pytest.approx(0.7, abs=0.06)
        for part in real.values():
            assert (part.class_counts(3) > 0).all()

    def test_val_and_test_keep_natural_distribution_and_have_no_synthetic(self, data):
        for part in (data.val, data.test):
            assert not part.synthetic.any()
            share = part.class_counts(3) / len(part)
            assert share[0] > 0.8  # still dominated by normal windows

    def test_smote_only_on_train_and_to_the_requested_ratio(self, data):
        counts = data.train.class_counts(3)
        assert data.train.synthetic.any() and counts[1] == pytest.approx(0.5 * counts[0], abs=1)
        assert (data.train.entity_ids[data.train.synthetic] == "synthetic").all()
        assert data.info["n_synthetic"] == int(data.train.synthetic.sum())

    def test_class_weights_come_from_the_resampled_training_labels(self, data):
        expected = balanced_class_weights(data.train.y, 3)
        assert data.class_weights == pytest.approx(expected)
        assert data.class_weights[0] < data.class_weights[1]

    def test_no_window_overlaps_across_splits(self, data):
        """Windows of one entity in different splits must not share a single row (embargo)."""
        spans = {}
        for name, part in (("train", data.train_real), ("val", data.val), ("test", data.test)):
            for ent, end in zip(part.entity_ids, part.end_timestamps, strict=True):
                spans.setdefault(ent, []).append((end - (T - 1), end, name))
        for ent, items in spans.items():
            items.sort()
            for (_s1, e1, n1), (s2, _e2, n2) in zip(items, items[1:], strict=False):
                if n1 != n2:
                    assert s2 > e1, f"{ent}: windows of {n1} and {n2} overlap"

    def test_labels_match_newest_row_ground_truth(self):
        rows = synthetic_rows(n_entities=1, n_rows=6000)
        data = DatasetBuilder(seq_len=T, stride=1, smote_ratio=None, group_blocks=4, seed=0).build(
            rows
        )
        point = data.train_real.subset(data.train_real.y == 1)
        assert len(point) > 0
        assert (point.x[:, -1, 0] > 1.0).all()  # newest row carries the spike

    def test_collective_normal_policy_keeps_those_windows(self):
        rows = synthetic_rows(n_entities=2)
        dropped = DatasetBuilder(seq_len=T, stride=3, smote_ratio=None).build(rows)
        kept = DatasetBuilder(seq_len=T, stride=3, smote_ratio=None, collective="normal").build(
            rows
        )
        assert kept.info["n_collective_dropped"] == 0
        n = lambda d: len(d.train) + len(d.val) + len(d.test)  # noqa: E731
        assert n(kept) > n(dropped)

    def test_smote_disabled_gives_natural_train_and_stronger_weights(self):
        rows = synthetic_rows(n_entities=2)
        off = DatasetBuilder(seq_len=T, stride=3, smote_ratio=None).build(rows)
        on = DatasetBuilder(seq_len=T, stride=3, smote_ratio=0.5).build(rows)
        assert not off.train.synthetic.any()
        assert off.class_weights[1] > on.class_weights[1]  # weights carry the whole correction

    def test_columns_subset_and_errors(self):
        rows = synthetic_rows(n_entities=2)
        data = DatasetBuilder(seq_len=T, stride=3, smote_ratio=None, columns=[0, 1]).build(rows)
        assert data.n_features == 2 and data.feature_names == NAMES[:2] and data.columns == (0, 1)
        with pytest.raises(ValueError, match="no windows"):
            DatasetBuilder(seq_len=T).build(
                synthetic_rows(n_entities=1, n_rows=300).subset(np.arange(T - 2))
            )
        with pytest.raises(ValueError):
            DatasetBuilder(class_names=("point", "normal"))
        with pytest.raises(ValueError):
            DatasetBuilder(smote_ratio=2.0)

    def test_summary_keys(self, data):
        assert set(data.summary()) == {"train_real", "train_smote", "val", "test"}
        assert set(data.summary()["val"]) == set(CLASSES)

    def test_build_from_simulation_wires_the_simulator(self, monkeypatch):
        rows = synthetic_rows(n_entities=2)
        seen = {}

        def fake(duration_s, **kw):
            seen.update(duration_s=duration_s, **kw)
            return {"warehouse": rows}

        monkeypatch.setattr("src.models.offline_dataset.simulate_labeled_features", fake)
        data = DatasetBuilder(seq_len=T, stride=3, smote_ratio=None).build_from_simulation(
            600, seed=3
        )
        assert seen["duration_s"] == 600 and seen["seed"] == 3 and seen["sources"] == ("warehouse",)
        assert len(data.val) > 0


# --------------------------------------------------------------------------- trainer
class TestTrainer:
    def test_learns_the_toy_task_and_restores_best_weights(self, trained):
        detector, history, data = trained
        out = detector.classify_sequences(data.test.x)
        report = evaluate_predictions(data.test.y, out.probabilities, CLASSES)
        assert report.macro_f1 > 0.9
        assert history.best_epoch >= 1 and history.epochs_run <= 25
        assert max(history.val_macro_f1) == pytest.approx(history.best_score, abs=1e-9)

    def test_history_is_complete(self, trained):
        _, history, _ = trained
        n = history.epochs_run
        for name in ("train_loss", "val_loss", "val_macro_f1", "learning_rate", "gradient_norm"):
            assert len(getattr(history, name)) == n
        assert len(history.val_f1) == n and len(history.val_f1[0]) == 3
        assert history.train_loss[-1] < history.train_loss[0]
        assert history.to_dict()["epochs_run"] == n

    def test_cosine_schedule_decays_the_learning_rate(self, trained):
        lrs = trained[1].learning_rate
        assert lrs[0] == pytest.approx(5e-3) and lrs[-1] < lrs[0]

    def test_early_stopping_triggers(self):
        x, y = toy_windows(40)
        rng = np.random.default_rng(0)
        trainer = LSTMClassifierTrainer(
            LSTMClassifier(F, seq_len=T, hidden_size=8), class_names=CLASSES, max_epochs=40,
            patience=2, learning_rate=1e-4, num_workers=0, log_dir=None,
        )  # fmt: skip
        # validation labels are random, so macro F1 stops improving almost at once
        history = trainer.fit(
            SequenceDataset(x, y), SequenceDataset(x[:60], rng.integers(0, 3, 60))
        )
        assert history.stopped_early and history.epochs_run < 40

    def test_class_weights_change_the_loss(self):
        x, y = toy_windows(30)
        ds = SequenceDataset(x, y)

        def first_loss(weights):
            torch.manual_seed(0)
            trainer = LSTMClassifierTrainer(
                LSTMClassifier(F, seq_len=T, hidden_size=8), class_names=CLASSES, max_epochs=1,
                num_workers=0, log_dir=None, batch_size=90,
            )  # fmt: skip
            return trainer.fit(ds, ds, class_weights=weights).train_loss[0]

        assert first_loss([1, 1, 1]) != pytest.approx(first_loss([0.1, 5, 5]))

    def test_validation_errors(self):
        model = LSTMClassifier(F, hidden_size=8)
        ds = SequenceDataset(*toy_windows(5))
        kwargs = {"num_workers": 0, "log_dir": None}
        with pytest.raises(ValueError):
            LSTMClassifierTrainer(model, learning_rate=0, **kwargs)
        with pytest.raises(ValueError):
            LSTMClassifierTrainer(model, monitor="accuracy", **kwargs)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            LSTMClassifierTrainer(model, class_names=("normal", "point"), **kwargs)
        trainer = LSTMClassifierTrainer(model, max_epochs=1, **kwargs)
        with pytest.raises(ValueError, match="positive"):
            trainer.fit(ds, ds, class_weights=[1, 0, 1])
        with pytest.raises(ValueError, match="features"):
            trainer.fit(SequenceDataset(np.zeros((4, T, F + 1)), np.zeros(4)), ds)
        with pytest.raises(ValueError, match="empty"):
            trainer.fit(SequenceDataset(np.zeros((0, T, F)), np.zeros(0)), ds)

    def test_val_loss_monitor_and_tensorboard_logging(self, tmp_path):
        x, y = toy_windows(20)
        ds = SequenceDataset(x, y)
        trainer = LSTMClassifierTrainer(
            LSTMClassifier(F, seq_len=T, hidden_size=8), class_names=CLASSES, max_epochs=2,
            monitor="val_loss", num_workers=0, log_dir=tmp_path, run_name="r",
        )  # fmt: skip
        history = trainer.fit(ds, ds)
        assert history.monitor == "val_loss" and history.best_score <= 0
        assert any((tmp_path / "r").iterdir())

    def test_train_classifier_checks_config(self, trained):
        _, _, data = trained
        with pytest.raises(ValueError, match="seq_len"):
            train_classifier(data, fast_cfg(seq_len=T + 1), log_dir=None, num_workers=0)
        with pytest.raises(ValueError, match="classes"):
            train_classifier(
                data, fast_cfg(class_names=["normal", "point"]), log_dir=None, num_workers=0
            )


# --------------------------------------------------------------------------- calibration
class TestTemperatureScaler:
    def test_overconfident_logits_get_t_above_one(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 3, 2000)
        logits = rng.normal(0, 1, (2000, 3))
        logits[np.arange(2000), labels] += 1.0
        logits *= 5.0  # same ranking, wildly over-confident
        scaler = TemperatureScaler()
        t = scaler.calibrate(logits, labels)
        assert t > 2.0 and scaler.is_fitted
        assert scaler.nll(logits, labels) < TemperatureScaler._nll(logits, labels, 1.0)

    def test_underconfident_logits_get_t_below_one(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 3, 2000)
        logits = rng.normal(0, 1, (2000, 3))
        logits[np.arange(2000), labels] += 3.0
        assert TemperatureScaler().calibrate(logits * 0.2, labels) < 1.0

    def test_well_calibrated_logits_stay_near_one(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(0, 2, (20000, 3))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        labels = np.array([rng.choice(3, p=p) for p in probs])
        assert TemperatureScaler().calibrate(logits, labels) == pytest.approx(1.0, abs=0.1)

    def test_scale_sums_to_one_keeps_argmax_and_uses_given_t(self):
        logits = np.array([[2.0, 1.0, 0.1], [0.0, 3.0, 1.0]])
        scaler = TemperatureScaler(2.0)
        p = scaler.scale(logits)
        assert p.sum(axis=1) == pytest.approx([1, 1])
        assert (p.argmax(axis=1) == logits.argmax(axis=1)).all()
        assert scaler.scale(logits, 1.0) == pytest.approx(
            np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        )
        assert (scaler.scale(torch.tensor(logits)) == p).all()

    def test_roundtrip_and_validation(self):
        restored = TemperatureScaler.from_dict(TemperatureScaler(1.7).to_dict())
        assert restored.temperature == pytest.approx(1.7)
        with pytest.raises(ValueError):
            TemperatureScaler(0.0)
        with pytest.raises(ValueError):
            TemperatureScaler().scale(np.zeros((1, 3)), -1.0)
        with pytest.raises(ValueError):
            TemperatureScaler().calibrate(np.zeros((0, 3)), np.zeros(0))
        with pytest.raises(ValueError, match="outside"):
            TemperatureScaler().calibrate(np.zeros((2, 3)), np.array([0, 5]))

    def test_calibration_never_worsens_nll(self, trained):
        detector, _, data = trained
        logits, _ = predict_logits(detector.model, data.val.x)
        scaler = TemperatureScaler()
        scaler.calibrate(logits, data.val.y)
        assert (
            scaler.nll(logits, data.val.y) <= TemperatureScaler._nll(logits, data.val.y, 1.0) + 1e-9
        )


class TestCalibrationMetrics:
    def test_ece_zero_for_perfect_calibration_and_large_for_overconfidence(self):
        n = 10000
        rng = np.random.default_rng(0)
        conf = rng.uniform(0.5, 1.0, n)
        correct = rng.random(n) < conf
        p = np.stack([conf, 1 - conf], axis=1)
        y = np.where(correct, 0, 1)
        assert expected_calibration_error(p, y) < 0.03
        assert (
            expected_calibration_error(
                np.tile([0.99, 0.01], (n, 1)), (rng.random(n) < 0.5).astype(int)
            )
            > 0.4
        )

    def test_reliability_bins(self):
        mean_conf, acc, count = reliability_bins(
            np.array([0.15, 0.17, 0.95]), np.array([0, 1, 1]), 10
        )
        assert count.sum() == 3 and count[1] == 2
        assert acc[1] == pytest.approx(0.5) and mean_conf[9] == pytest.approx(0.95)
        assert math.isnan(acc[5])


class TestEvaluate:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 2, 2, 0])
        p = np.eye(3)[y] * 0.9 + 0.05
        report = evaluate_predictions(y, p, CLASSES)
        assert report.macro_f1 == 1.0 and report.accuracy == 1.0
        assert np.diag(report.confusion).tolist() == [3, 1, 2]
        assert report.roc_auc["point"] == 1.0 and report.anomaly_roc_auc == 1.0

    def test_confusion_orientation_and_per_class_metrics(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        pred = np.array([0, 0, 1, 1, 0, 2])
        p = np.eye(3)[pred]
        r = evaluate_predictions(y, p, CLASSES)
        assert r.confusion.tolist() == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]  # rows = true
        assert r.precision[1] == pytest.approx(0.5) and r.recall[1] == pytest.approx(0.5)
        assert r.support.tolist() == [3, 2, 1]
        d = r.to_dict()
        assert d["per_class"]["point"]["support"] == 2 and d["macro_f1"] == pytest.approx(
            r.macro_f1
        )

    def test_absent_class_gives_nan_auc_and_validation(self):
        y = np.array([0, 0, 1, 1])
        r = evaluate_predictions(y, np.eye(3)[y], CLASSES)
        assert math.isnan(r.roc_auc["contextual"]) and r.f1[2] == 0.0
        with pytest.raises(ValueError):
            evaluate_predictions(y, np.zeros((3, 3)), CLASSES)


# --------------------------------------------------------------------------- detector
def fake_source(window: np.ndarray, *, warm: bool = True, real: int | None = None):
    """A minimal SequenceSource over one entity."""
    n = len(window)
    mask = torch.ones(n, dtype=torch.bool)
    if real is not None:
        mask[: n - real] = False
    store = SimpleNamespace(
        get_features=lambda entity, w: (
            SimpleNamespace(
                window_end=NOW, is_warm=warm, signals={"temp": SimpleNamespace(last=21.5)}
            )
            if entity != "ghost"
            else None
        )
    )
    return SimpleNamespace(
        store=store,
        window_seconds=(60, 300, 900),
        build_sequence_with_mask=lambda entity, lookback, source_type=None: (
            torch.from_numpy(window),
            mask,
        ),
    )


class TestDetector:
    def test_classify_sequences_and_anomaly_score_shape(self, trained):
        detector, _, data = trained
        spikes = data.test.x[data.test.y == 1]
        result = detector.detect_sequence("s1", NOW, spikes[0])
        assert result.detector == "lstm_classifier" and result.is_anomaly
        assert result.anomaly_type == "point" and result.score >= 0.5
        probs = result.details["class_probabilities"]
        assert set(probs) == set(CLASSES) and sum(probs.values()) == pytest.approx(1.0)
        assert result.details["predicted_class"] == "point"
        assert len(result.details["attention_weights"]) == T
        assert sum(result.details["attention_weights"]) == pytest.approx(1.0, abs=1e-4)
        assert result.score == pytest.approx(1 - probs["normal"])
        assert result.confidence == pytest.approx(max(result.score, 1 - result.score))

    def test_attention_focuses_on_the_spike(self, trained):
        detector, _, data = trained
        spikes = data.test.x[data.test.y == 1][:20]
        out = detector.classify_sequences(spikes)
        assert out.attention[:, -3:].sum(axis=1).mean() > 3 / T  # more than a uniform share

    def test_normal_windows_are_not_anomalies(self, trained):
        detector, _, data = trained
        out = detector.classify_sequences(data.test.x[data.test.y == 0])
        assert (out.score >= 0.5).mean() < 0.1
        assert out.probabilities.sum(axis=1) == pytest.approx(np.ones(len(out.score)))

    def test_calibrated_probabilities_are_softer_than_raw(self, trained):
        detector, _, data = trained
        out = detector.classify_sequences(data.test.x)
        raw = torch.softmax(torch.from_numpy(out.logits), dim=1).numpy()
        if detector.temperature.temperature > 1:
            assert out.probabilities.max(axis=1).mean() < raw.max(axis=1).mean()
        assert (out.predicted == raw.argmax(axis=1)).all()

    def test_decision_threshold_is_respected(self, trained):
        detector, _, data = trained
        window = data.test.x[data.test.y == 1][0]
        soft = TemperatureScaler(200.0)  # keeps the score away from 0 / 1 saturation

        def at(threshold: float):
            return LSTMClassifierDetector(
                detector.model, NAMES, temperature=soft, decision_threshold=threshold
            ).detect_sequence("e", NOW, window)

        score = at(0.5).score
        assert 0.0 < score < 1.0
        assert at(score / 2).is_anomaly and not at(min(0.999999, (1 + score) / 2)).is_anomaly

    def test_push_warms_up_then_classifies_a_sliding_window(self, trained):
        detector, _, data = trained
        detector.reset()
        stream = data.test.x[data.test.y == 1][0]  # (T, F), ends in a spike
        results = [detector.push("s9", row, NOW) for row in stream]
        warm = detector.min_real_steps
        assert all(r.details["warming_up"] for r in results[: warm - 1])
        assert results[warm - 1].details["warming_up"] is False
        assert results[-1].is_anomaly and results[-1].anomaly_type == "point"
        assert results[-1].details["real_steps"] == T

    def test_push_buffers_are_per_entity_bounded_and_resettable(self, trained):
        detector, _, _ = trained
        detector.reset()
        row = np.zeros(F)
        for _ in range(3 * T):
            detector.push("a", row, NOW)
        detector.push("b", row, NOW)
        assert len(detector._buffers["a"]) == T and len(detector._buffers["b"]) == 1
        detector.reset("a")
        assert "a" not in detector._buffers and "b" in detector._buffers
        detector.reset()
        assert not detector._buffers

    def test_push_fills_nan_and_validates_width(self, trained):
        detector, _, _ = trained
        detector.reset()
        detector.push("n", [1.0, 2.0, 3.0, 4.0], NOW)
        detector.push("n", [np.nan, np.inf, 0.0, 0.0], NOW)
        assert detector._buffers["n"][-1].tolist() == [1.0, 2.0, 0.0, 0.0]
        detector.push("fresh", [np.nan] * F, NOW)
        assert detector._buffers["fresh"][-1].tolist() == [0.0] * F
        with pytest.raises(ValueError, match="features"):
            detector.push("n", [1.0, 2.0], NOW)

    def test_push_applies_column_selection(self, trained):
        detector, _, data = trained
        narrow = LSTMClassifierDetector(
            LSTMClassifier(2, seq_len=T, hidden_size=8), NAMES[:2], columns=[3, 0]
        )
        narrow.push("e", [10.0, 11.0, 12.0, 13.0], NOW)
        assert narrow._buffers["e"][-1].tolist() == [13.0, 10.0]

    def test_detect_reads_the_feature_store(self, trained):
        detector, _, data = trained
        window = data.test.x[data.test.y == 1][0].astype(np.float32)
        wired = LSTMClassifierDetector(
            detector.model, NAMES, temperature=detector.temperature,
            sequences=fake_source(window),
        )  # fmt: skip
        result = wired.detect("s1")
        assert result.is_anomaly and result.raw_value == 21.5 and result.timestamp == NOW
        many = wired.detect_many(["s1", "s2", "ghost"])
        assert len(many) == 3 and many[2].details["warming_up"] is True

    def test_detect_warming_up_cases(self, trained):
        detector, _, data = trained
        window = data.test.x[0].astype(np.float32)
        cold = LSTMClassifierDetector(
            detector.model, NAMES, sequences=fake_source(window, warm=False)
        )
        assert cold.detect("s1").details["warming_up"] is True
        short = LSTMClassifierDetector(detector.model, NAMES, sequences=fake_source(window, real=3))
        r = short.detect("s1")
        assert r.score == 0.0 and not r.is_anomaly and r.anomaly_type is None
        assert (
            r.details["real_steps"] == 3 and r.details["required_steps"] == detector.min_real_steps
        )
        with pytest.raises(RuntimeError, match="sequences"):
            detector.detect("s1")

    def test_constructor_validation(self, trained):
        detector, _, _ = trained
        with pytest.raises(ValueError, match="feature names"):
            LSTMClassifierDetector(detector.model, NAMES[:2])
        with pytest.raises(ValueError, match="class names"):
            LSTMClassifierDetector(detector.model, NAMES, class_names=("normal", "point"))
        with pytest.raises(ValueError):
            LSTMClassifierDetector(detector.model, NAMES, decision_threshold=1.0)

    def test_checkpoint_roundtrip_reproduces_scores(self, trained):
        detector, _, data = trained
        clone = LSTMClassifierDetector.from_checkpoint(detector.to_checkpoint())
        a = detector.classify_sequences(data.test.x)
        b = clone.classify_sequences(data.test.x)
        assert np.allclose(a.probabilities, b.probabilities, atol=1e-6)
        assert clone.temperature.temperature == pytest.approx(detector.temperature.temperature)
        assert clone.class_names == detector.class_names and clone.seq_len == T

    def test_registry_roundtrip(self, trained, tmp_path):
        detector, _, data = trained
        registry = ModelRegistry(tmp_path)
        version = detector.save(registry, metrics={"macro_f1": 0.9})
        loaded = LSTMClassifierDetector.load(registry)
        assert version.model_type == "lstm_classifier"
        assert np.allclose(
            detector.classify_sequences(data.test.x[:10]).probabilities,
            loaded.classify_sequences(data.test.x[:10]).probabilities,
            atol=1e-6,
        )

    def test_threadsafe_concurrent_pushes(self, trained):
        import threading

        detector, _, _ = trained
        detector.reset()
        errors: list[Exception] = []

        def worker(name: str) -> None:
            try:
                for _ in range(2 * T):
                    detector.push(name, np.zeros(F), NOW)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors and all(len(detector._buffers[f"t{i}"]) == T for i in range(4))
