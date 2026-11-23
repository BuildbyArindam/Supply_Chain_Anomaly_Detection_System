"""Unit tests for the Day 10 LSTM autoencoder (model, trainer, calibrator, detector, windows)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError

from src.models.feature_engineering import OnlineScaler, fill_missing
from src.models.lstm_autoencoder import (
    AnomalyThresholdCalibrator,
    LSTMAutoencoder,
    LSTMAutoencoderDetector,
    LSTMAutoencoderTrainer,
    SequenceSet,
    _forward_fill_windows,
    kl_divergence,
    make_sequences,
    reconstruction_errors,
    select_features,
    train_detector,
)
from src.models.model_registry import ModelRegistry
from src.models.offline_dataset import LabeledFeatureSet
from src.utils.config import LSTMAutoencoderConfig

T, F = 20, 4
NAMES = tuple(f"feat{i}" for i in range(F))
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def tiny_model(**kw) -> LSTMAutoencoder:
    params = {"seq_len": T, "hidden_sizes": (8, 12), "latent_dim": 4, "dropout": 0.2}
    params.update(kw)
    return LSTMAutoencoder(F, **params)


def sine_windows(n: int, seed: int = 0) -> np.ndarray:
    """Normal behaviour: noisy sinusoids with a random phase per window."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 1))
    freq = np.array([1.0, 2.0, 3.0, 4.0])[None, None, :] * 2 * np.pi / T
    return (np.sin(t * freq + phase) + rng.normal(0, 0.05, (n, T, F))).astype(np.float32)


def drifted(x: np.ndarray, feature: int = 0, slope: float = 0.25) -> np.ndarray:
    out = x.copy()
    out[:, :, feature] += slope * np.arange(T)[None, :]
    return out


def fast_cfg(**over) -> LSTMAutoencoderConfig:
    # kl_weight 0.01: with the production 0.1 a model this small collapses onto the mean (see
    # TestTrainer.test_posterior_collapse_is_flagged); the full-size model does not.
    base = {"seq_len": T, "hidden_sizes": (16, 32), "latent_dim": 8, "max_epochs": 40,
            "early_stopping_patience": 10, "batch_size": 32, "learning_rate": 5e-3,
            "kl_weight": 0.01, "mc_dropout_passes": 10}  # fmt: skip
    base.update(over)
    return LSTMAutoencoderConfig(**base)


@pytest.fixture(scope="module")
def trained():
    """A small detector trained once on sinusoids and calibrated on normal validation data."""
    torch.manual_seed(0)
    detector, history = train_detector(
        sine_windows(400, 1), sine_windows(120, 2), NAMES, fast_cfg(), log_dir=None
    )
    return detector, history


# --------------------------------------------------------------------------- model
class TestModel:
    def test_shapes_and_latent(self):
        model = tiny_model().eval()
        x = torch.randn(3, T, F)
        recon, latent = model(x)
        assert recon.shape == x.shape
        assert latent.z.shape == latent.mu.shape == latent.logvar.shape == (3, 4)

    def test_accepts_other_sequence_lengths(self):
        recon, _ = tiny_model().eval()(torch.randn(2, T + 7, F))
        assert recon.shape == (2, T + 7, F)

    def test_spec_architecture(self):
        model = tiny_model(hidden_sizes=(64, 128), latent_dim=32)
        assert (model.enc_lstm1.hidden_size, model.enc_lstm2.hidden_size) == (64, 128)
        assert (model.dec_lstm1.hidden_size, model.dec_lstm2.hidden_size) == (128, 64)
        norms = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
        drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(norms) == 4 and all(d.p == 0.2 for d in drops)

    def test_eval_is_deterministic_train_samples_latent(self):
        model, x = tiny_model(), torch.randn(2, T, F)
        model.eval()
        assert torch.equal(model(x)[0], model(x)[0])
        model.train()
        assert not torch.equal(model(x)[1].z, model(x)[1].z)

    def test_rejects_bad_input_and_args(self):
        with pytest.raises(ValueError, match="expected input"):
            tiny_model()(torch.randn(2, T, F + 1))
        with pytest.raises(ValueError):
            tiny_model(dropout=1.0)
        with pytest.raises(ValueError):
            LSTMAutoencoder(0)

    def test_kl_zero_for_standard_normal(self):
        zeros = torch.zeros(5, 4)
        assert kl_divergence(zeros, zeros).item() == pytest.approx(0.0)
        assert kl_divergence(zeros + 1.0, zeros).item() > 0.0

    def test_from_config(self):
        model = LSTMAutoencoder.from_config(fast_cfg(), F)
        assert model.get_config()["latent_dim"] == 8 and model.seq_len == T


# --------------------------------------------------------------------------- errors
class TestReconstructionErrors:
    def test_shapes_and_deterministic_std_zero(self):
        res = reconstruction_errors(tiny_model(), sine_windows(10), mc_passes=1)
        assert res.error.shape == res.error_std.shape == (10,)
        assert res.per_feature.shape == (10, F)
        assert np.all(res.error_std == 0)
        np.testing.assert_allclose(res.per_feature.mean(axis=1), res.error, rtol=1e-5)

    def test_batching_does_not_change_result(self):
        model, x = tiny_model(), sine_windows(9)
        a = reconstruction_errors(model, x, batch_size=2)
        b = reconstruction_errors(model, x, batch_size=64)
        np.testing.assert_allclose(a.error, b.error, rtol=1e-4, atol=1e-6)

    def test_mc_dropout_has_spread_and_restores_mode(self):
        model = tiny_model().train()
        res = reconstruction_errors(model, sine_windows(6), mc_passes=10)
        assert np.all(res.error_std > 0)
        assert model.training  # mode restored
        reconstruction_errors(model.eval(), sine_windows(2), mc_passes=3)
        assert not model.training and not any(
            m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
        )

    def test_validation(self):
        with pytest.raises(ValueError, match="mc_passes"):
            reconstruction_errors(tiny_model(), sine_windows(2), mc_passes=0)
        bad = sine_windows(2)
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            reconstruction_errors(tiny_model(), bad)


# --------------------------------------------------------------------------- calibrator
class TestCalibrator:
    def test_threshold_is_mean_plus_three_std(self):
        errs = np.random.default_rng(0).gamma(4.0, 0.1, 500)
        result = AnomalyThresholdCalibrator().fit(errs)
        assert result.calibrated_threshold == pytest.approx(errs.mean() + 3 * errs.std(ddof=1))
        assert result.threshold_per_feature == {}

    def test_per_feature_thresholds_and_localisation(self):
        rng = np.random.default_rng(0)
        pf = rng.gamma(4.0, 0.1, (300, 3)) * np.array([1.0, 5.0, 1.0])
        cal = AnomalyThresholdCalibrator(sigma=2.0)
        thr, per = cal.fit(pf.mean(axis=1), pf, ["a", "b", "c"])
        assert set(per) == {"a", "b", "c"} and per["b"] > per["a"]
        assert cal.anomalous_features(np.array([0.1, 0.1, 0.1])) == []
        assert cal.anomalous_features(np.array([per["a"] * 1.5, 0.1, per["c"] * 3])) == ["c", "a"]
        assert thr > 0

    def test_spread_floor_for_constant_feature(self):
        pf = np.full((50, 2), 1e-6)
        pf[:, 1] += np.random.default_rng(0).normal(0, 1e-7, 50)
        _, per = AnomalyThresholdCalibrator().fit(pf.mean(axis=1), pf, ["flat", "noisy"])
        assert per["flat"] >= 1e-6 * (1 + 3 * 0.05) - 1e-12

    def test_validation_and_unfitted(self):
        cal = AnomalyThresholdCalibrator()
        with pytest.raises(NotFittedError):
            _ = cal.threshold
        with pytest.raises(NotFittedError):
            cal.anomalous_features(np.zeros(2))
        with pytest.raises(ValueError, match="at least"):
            cal.fit(np.ones(3))
        with pytest.raises(ValueError, match="finite"):
            cal.fit(np.append(np.ones(20), np.nan))
        with pytest.raises(ValueError, match="one name per feature"):
            cal.fit(np.ones(20), np.ones((20, 2)), ["only"])
        with pytest.raises(ValueError):
            AnomalyThresholdCalibrator(sigma=0)

    def test_round_trip(self):
        pf = np.random.default_rng(1).gamma(3.0, 0.1, (60, 2))
        cal = AnomalyThresholdCalibrator(2.5)
        cal.fit(pf.mean(axis=1), pf, ["x", "y"])
        clone = AnomalyThresholdCalibrator.from_dict(cal.to_dict())
        assert (
            clone.threshold == cal.threshold
            and clone.threshold_per_feature == cal.threshold_per_feature
        )
        assert clone.anomalous_features(np.array([9.0, 0.0])) == ["x"]
        assert not AnomalyThresholdCalibrator.from_dict(
            AnomalyThresholdCalibrator().to_dict()
        ).is_fitted


# --------------------------------------------------------------------------- trainer
class TestTrainer:
    def test_learns_and_records_curves(self):
        torch.manual_seed(0)
        model = tiny_model()
        trainer = LSTMAutoencoderTrainer(
            model, max_epochs=8, batch_size=32, learning_rate=5e-3, log_dir=None
        )
        hist = trainer.fit(sine_windows(200, 1), sine_windows(60, 2))
        assert hist.epochs_run == 8 and not hist.stopped_early
        assert hist.val_loss[-1] < hist.val_loss[0]
        assert all(g > 0 and math.isfinite(g) for g in hist.gradient_norm)
        assert (
            hist.best_val_loss == min(hist.val_loss)
            and hist.best_epoch == int(np.argmin(hist.val_loss)) + 1
        )
        for key in ("train_loss", "val_loss", "learning_rate", "gradient_norm"):
            assert len(hist.to_dict()[key]) == 8

    def test_early_stopping_restores_best_weights_and_lr_drops(self, monkeypatch):
        model = tiny_model()
        trainer = LSTMAutoencoderTrainer(
            model, max_epochs=50, patience=3, lr_patience=1, log_dir=None
        )
        losses = iter([1.0, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        snapshots = {}

        def fake_eval(_x):
            value = next(losses)
            snapshots[value, len(snapshots)] = None
            return value, value, 0.0

        monkeypatch.setattr(trainer, "evaluate", fake_eval)
        hist = trainer.fit(sine_windows(40), sine_windows(10))
        assert hist.stopped_early and hist.best_epoch == 2 and hist.epochs_run == 5
        assert hist.learning_rate[-1] < hist.learning_rate[0]

    def test_best_weights_restored(self, monkeypatch):
        model = tiny_model()
        trainer = LSTMAutoencoderTrainer(
            model, max_epochs=3, patience=5, learning_rate=5e-2, log_dir=None
        )
        seen: list[dict] = []
        losses = iter([0.5, 1.0, 1.0])

        def fake_eval(_x):
            seen.append({k: v.clone() for k, v in model.state_dict().items()})
            return next(losses), 0.0, 0.0

        monkeypatch.setattr(trainer, "evaluate", fake_eval)
        trainer.fit(sine_windows(40), sine_windows(10))
        for key, value in model.state_dict().items():
            assert torch.equal(value, seen[0][key])  # weights of epoch 1 (best)

    def test_gradient_clipping_bounds_update_norm(self):
        model = tiny_model()
        trainer = LSTMAutoencoderTrainer(model, max_epochs=1, grad_clip_norm=1e-3, log_dir=None)
        trainer.fit(sine_windows(32), sine_windows(8))
        assert trainer.grad_clip_norm == 1e-3  # recorded norm is pre-clip, must stay reported
        assert max(p.grad.norm().item() for p in model.parameters() if p.grad is not None) < 1e-2

    def test_tensorboard_logs_written(self, tmp_path):
        pytest.importorskip("tensorboard")
        trainer = LSTMAutoencoderTrainer(
            tiny_model(), max_epochs=2, log_dir=tmp_path, run_name="r1"
        )
        trainer.fit(sine_windows(32), sine_windows(8))
        assert list((tmp_path / "r1").glob("events.out.tfevents.*"))

    def test_validation_errors(self):
        trainer = LSTMAutoencoderTrainer(tiny_model(), log_dir=None)
        with pytest.raises(ValueError, match="features"):
            trainer.fit(np.zeros((4, T, F + 1), np.float32), np.zeros((4, T, F + 1), np.float32))
        with pytest.raises(ValueError, match="empty"):
            trainer.fit(np.zeros((0, T, F), np.float32), sine_windows(4))
        with pytest.raises(ValueError):
            LSTMAutoencoderTrainer(tiny_model(), learning_rate=0)

    def test_posterior_collapse_is_flagged(self, monkeypatch):
        import src.models.lstm_autoencoder as module

        calls = []
        monkeypatch.setattr(module._log, "warning", lambda event, **kw: calls.append(event))
        torch.manual_seed(0)
        model = tiny_model(hidden_sizes=(8, 12), latent_dim=4)
        LSTMAutoencoderTrainer(
            model, max_epochs=10, batch_size=32, learning_rate=5e-3, kl_weight=0.1, log_dir=None
        ).fit(sine_windows(200, 1), sine_windows(60, 2))
        assert "lstm_autoencoder_posterior_collapse" in calls

    def test_from_config(self):
        trainer = LSTMAutoencoderTrainer.from_config(fast_cfg(), tiny_model(), log_dir=None)
        assert (
            trainer.patience == 10 and trainer.kl_weight == 0.01 and trainer.grad_clip_norm == 1.0
        )


# --------------------------------------------------------------------------- detector
class TestDetector:
    def test_training_converged(self, trained):
        _, history = trained
        assert history.best_val_loss < 0.2 * history.val_loss[0]

    def test_separates_drift_from_normal(self, trained):
        detector, _ = trained
        normal = detector.score_sequences(sine_windows(150, 3))
        drift = detector.score_sequences(drifted(sine_windows(150, 4)))
        assert normal.is_anomaly.mean() < 0.1
        assert drift.is_anomaly.mean() > 0.9
        assert drift.error.mean() > 3 * normal.error.mean()

    def test_score_convention(self, trained):
        detector, _ = trained
        thr = detector.calibrator.threshold
        res = detector.score_sequences(drifted(sine_windows(20, 5)))
        np.testing.assert_allclose(res.score, res.error / (res.error + thr))
        assert np.array_equal(res.score >= 0.5, res.is_anomaly)
        assert np.all((res.confidence >= 0) & (res.confidence <= 1))
        assert np.all(res.error_std > 0)

    def test_localises_drifted_feature(self, trained):
        detector, _ = trained
        res = detector.detect_sequence("s1", NOW, drifted(sine_windows(1, 6), feature=2)[0])
        assert (
            res.is_anomaly
            and res.anomaly_type == "collective"
            and res.detector == "lstm_autoencoder"
        )
        assert res.details["which_features_anomalous"][0] == "feat2"
        assert (
            res.details["mc_passes"] == 10
            and res.details["reconstruction_error"] > res.details["threshold"]
        )

    def test_requires_calibration(self):
        detector = LSTMAutoencoderDetector(tiny_model(), AnomalyThresholdCalibrator(), NAMES)
        with pytest.raises(NotFittedError):
            detector.score_sequences(sine_windows(2))
        with pytest.raises(RuntimeError, match="sequences"):
            detector.detect("x")

    def test_single_pass_mode_has_no_uncertainty(self, trained):
        detector, _ = trained
        det1 = LSTMAutoencoderDetector(
            detector.model, AnomalyThresholdCalibrator(), NAMES, mc_passes=1
        )
        det1.calibrate(sine_windows(60, 7))
        res = det1.score_sequences(sine_windows(5, 8))
        assert np.all(res.error_std == 0)
        np.testing.assert_array_equal(res.error, det1.score_sequences(sine_windows(5, 8)).error)

    def test_feature_name_mismatch(self):
        with pytest.raises(ValueError, match="feature names"):
            LSTMAutoencoderDetector(tiny_model(), AnomalyThresholdCalibrator(), NAMES[:2])

    def test_checkpoint_round_trip_via_registry(self, trained, tmp_path):
        detector, _ = trained
        registry = ModelRegistry(tmp_path)
        version = detector.save(registry, metrics={"f1": 0.5})
        assert version.model_type == "lstm_autoencoder"
        clone = LSTMAutoencoderDetector.load(registry)
        x = sine_windows(8, 9)
        np.testing.assert_allclose(
            clone.model.eval()(torch.as_tensor(x))[0].detach().numpy(),
            detector.model.eval()(torch.as_tensor(x))[0].detach().numpy(),
            rtol=1e-5,
        )
        assert clone.calibrator.threshold == detector.calibrator.threshold
        assert clone.feature_names == detector.feature_names and clone.mc_passes == 10


class FakeStore:
    def __init__(self, items):
        self.items = items

    def get_features(self, entity_id, window_seconds):
        return self.items.get((entity_id, window_seconds))


class FakeSequences:
    """Stands in for ``pipeline.sequences``: fixed windows, masks and latest features."""

    window_seconds = (60, 300, 900)

    def __init__(self, windows, masks, warm):
        self.windows, self.masks = windows, masks
        feats = {}
        for eid, is_warm in warm.items():
            for w in self.window_seconds:
                feats[eid, w] = SimpleNamespace(
                    window_end=NOW,
                    is_warm=is_warm or w != 900,
                    signals={"value": SimpleNamespace(last=3.5)},
                )
        self.store = FakeStore(feats)

    def build_sequence_with_mask(self, entity_id, lookback=60, source_type=None):
        return torch.as_tensor(self.windows[entity_id]), torch.as_tensor(self.masks[entity_id])


class TestStreamingDetect:
    def make(self, trained, **kw):
        detector, _ = trained
        normal, bad = sine_windows(1, 11)[0], drifted(sine_windows(1, 12))[0]
        full = np.ones(T, dtype=bool)
        sequences = FakeSequences(
            {
                "ok": normal,
                "bad": bad,
                "cold": normal,
                "padded": normal,
                "ghost": np.zeros((T, F), np.float32),
            },
            {
                "ok": full,
                "bad": full,
                "cold": full,
                "padded": np.r_[np.zeros(15, bool), np.ones(5, bool)],
                "ghost": ~full,
            },
            {"ok": True, "bad": True, "cold": False, "padded": True},
        )
        return LSTMAutoencoderDetector(
            detector.model,
            detector.calibrator,
            NAMES,
            sequences=sequences,
            source_type="warehouse",
            **kw,
        )

    def test_detect_scores_ready_entities_and_warms_up_others(self, trained):
        det = self.make(trained)
        ok, bad, cold, padded, ghost = det.detect_many(["ok", "bad", "cold", "padded", "ghost"])
        assert not ok.is_anomaly and not ok.details["warming_up"] and ok.raw_value == 3.5
        assert bad.is_anomaly and bad.timestamp == NOW
        for result in (cold, padded, ghost):
            assert result.score == 0 and not result.is_anomaly and result.details["warming_up"]
            assert result.anomaly_type is None
        assert padded.details["required_steps"] == math.ceil(0.8 * T)

    def test_detect_matches_detect_many_and_order_is_kept(self, trained):
        det = self.make(trained, mc_passes=1)
        single = det.detect("bad")
        batch = det.detect_many(["ok", "bad"])
        assert [r.entity_id for r in batch] == ["ok", "bad"]
        assert single.details["reconstruction_error"] == pytest.approx(
            batch[1].details["reconstruction_error"], rel=1e-4
        )

    def test_require_warm_can_be_disabled(self, trained):
        assert not self.make(trained, require_warm=False).detect("cold").details["warming_up"]

    def test_columns_select_model_inputs_from_wider_feature_vector(self, trained):
        detector, _ = trained
        extra = np.concatenate([sine_windows(1, 11)[0], np.full((T, 2), 99.0, np.float32)], axis=1)
        wide = np.concatenate(
            [extra[:, :1], extra[:, 4:5], extra[:, 1:4], extra[:, 5:]], axis=1
        )  # 6 cols
        full = np.ones(T, dtype=bool)
        det = LSTMAutoencoderDetector(
            detector.model,
            detector.calibrator,
            NAMES,
            columns=(0, 2, 3, 4),
            sequences=FakeSequences({"w": wide}, {"w": full}, {"w": True}),
        )
        res = det.detect("w")
        assert not res.details["warming_up"]
        expected = detector.detect_sequence("w", NOW, wide[:, [0, 2, 3, 4]])
        assert res.details["reconstruction_error"] == pytest.approx(
            expected.details["reconstruction_error"], rel=0.5
        )  # MC passes differ; same input columns => same order of magnitude
        clone = LSTMAutoencoderDetector.from_checkpoint(det.to_checkpoint())
        assert clone.columns == (0, 2, 3, 4)

    def test_unknown_entity_never_raises(self, trained):
        det = self.make(trained)
        det.sequences.store = FakeStore({})
        det.sequences.windows["nobody"] = np.zeros((T, F), np.float32)
        det.sequences.masks["nobody"] = np.zeros(T, bool)
        assert det.detect("nobody").details["warming_up"]


# --------------------------------------------------------------------------- windows
def make_dataset(n_per_entity=100, entities=("a", "b"), nan_at=None, anomalous=()):
    rows = []
    for step in range(n_per_entity):
        for e in entities:
            rows.append((e, step))
    ent = np.array([r[0] for r in rows], dtype=object)
    step = np.array([r[1] for r in rows], dtype=float)
    feats = np.stack([step + (ent == "b") * 1000.0 + k for k in range(3)], axis=1)
    if nan_at is not None:
        feats[nan_at, 1] = np.nan
    y = np.zeros(len(rows), dtype=bool)
    kinds = np.full(len(rows), None, dtype=object)
    for idx in anomalous:
        y[idx], kinds[idx] = True, "collective"
    return LabeledFeatureSet(
        "warehouse",
        ("f0", "f1", "f2"),
        feats,
        y,
        kinds,
        ent,
        1.7e9 + step,
        np.full(len(rows), None, dtype=object),
        np.ones(len(rows), dtype=bool),
    )


class TestMakeSequences:
    scaler = OnlineScaler(3, clip=1e9)

    def test_windows_are_per_entity_and_end_at_newest_row(self):
        ds = make_dataset()
        seqs = make_sequences(ds, self.scaler, seq_len=10, stride=15)
        assert seqs.x.shape[1:] == (10, 3) and seqs.x.dtype == np.float32
        for eid in ("a", "b"):
            sel = seqs.entity_ids == eid
            assert seqs.end_timestamps[sel].max() == ds.timestamps[-1]
            # each window holds ten consecutive steps of one entity (raw f0 increases by 1)
            diffs = np.diff(seqs.x[sel][:, :, 0], axis=1)
            assert np.allclose(diffs, diffs[0, 0])
        assert np.all(np.diff(seqs.end_timestamps) >= 0)

    def test_stride_counts_back_from_newest(self):
        seqs = make_sequences(make_dataset(100, ("a",)), self.scaler, seq_len=10, stride=30)
        assert sorted(seqs.end_timestamps - 1.7e9) == [9.0, 39.0, 69.0, 99.0]

    def test_labels_and_fractions(self):
        ds = make_dataset(100, ("a",), anomalous=[95, 96, 97])
        seqs = make_sequences(ds, self.scaler, seq_len=10, stride=100)  # only the last window
        assert (
            len(seqs) == 1 and not seqs.y_last[0] and seqs.anomaly_fraction[0] == pytest.approx(0.3)
        )
        assert not seqs.normal[0]
        ds2 = make_dataset(100, ("a",), anomalous=[99])
        s2 = make_sequences(ds2, self.scaler, seq_len=10, stride=100)
        assert s2.y_last[0] and s2.type_last[0] == "collective"

    def test_warm_only_and_short_entities(self):
        ds = make_dataset(100, ("a",))
        ds.warm[:50] = False
        assert len(make_sequences(ds, self.scaler, seq_len=10, stride=1)) == 41
        assert len(make_sequences(ds, self.scaler, seq_len=10, stride=1, warm_only=False)) == 91
        assert len(make_sequences(make_dataset(5, ("a",)), self.scaler, seq_len=10)) == 0

    def test_subsample_keeps_chronological_order(self):
        seqs = make_sequences(
            make_dataset(), self.scaler, seq_len=10, stride=1, max_sequences=20, seed=1
        )
        assert len(seqs) == 20 and np.all(np.diff(seqs.end_timestamps) >= 0)

    def test_nan_filled_like_sequence_builder(self):
        ds = make_dataset(30, ("a",), nan_at=15)
        seqs = make_sequences(ds, self.scaler, seq_len=10, stride=1)
        assert np.isfinite(seqs.x).all()
        window = seqs.x[seqs.end_timestamps == 1.7e9 + 17][0]  # step 15 is the 7th of 8..17
        assert window[7, 1] == window[6, 1]  # forward-filled from the previous step

    def test_forward_fill_matches_fill_missing(self):
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(6, 12, 5))
        windows[rng.random(windows.shape) < 0.3] = np.nan
        windows[0, :, 2] = np.nan
        expected = np.stack([fill_missing(w.copy()) for w in windows])
        np.testing.assert_allclose(_forward_fill_windows(windows.copy()), expected)

    def test_columns_subset_keeps_scaler_on_full_vector(self):
        ds = make_dataset(40, ("a",))
        scaler = OnlineScaler(3, clip=1e9)
        scaler.fit_update(ds.X)
        full = make_sequences(ds, scaler, seq_len=10, stride=10)
        part = make_sequences(ds, scaler, seq_len=10, stride=10, columns=(2, 0))
        assert part.feature_names == ("f2", "f0") and part.x.shape[2] == 2
        np.testing.assert_allclose(part.x, full.x[:, :, [2, 0]])

    def test_select_features(self):
        names = ("value__last", "hour_sin", "dow_cos", "zone_A")
        kept, cols = select_features(names)
        assert kept == ("value__last", "zone_A") and cols == (0, 3)
        assert select_features(names, exclude=())[1] == (0, 1, 2, 3)
        with pytest.raises(ValueError, match="unknown"):
            select_features(names, exclude=("nope",))
        with pytest.raises(ValueError, match="every feature"):
            select_features(("hour_sin",), exclude=("hour_sin",))

    def test_between_subset_and_validation(self):
        seqs = make_sequences(make_dataset(), self.scaler, seq_len=10, stride=5)
        lo = datetime.fromtimestamp(1.7e9 + 40, UTC)
        hi = datetime.fromtimestamp(1.7e9 + 60, UTC)
        part = seqs.subset(seqs.between(lo, hi))
        assert isinstance(part, SequenceSet) and 0 < len(part) < len(seqs)
        assert part.end_timestamps.min() >= 1.7e9 + 40 and part.end_timestamps.max() < 1.7e9 + 60
        with pytest.raises(ValueError):
            make_sequences(make_dataset(), self.scaler, seq_len=1)


# --------------------------------------------------------------------------- real pipeline
class TestTrainServeConsistency:
    """Offline windows must equal what the online SequenceBuilder serves (real aggregator/cache)."""

    START = datetime(2026, 11, 21, 2, 0, tzinfo=UTC)
    SEED = 7

    @pytest.fixture(scope="class")
    def world(self):
        from src.data_simulation.orchestrator import SimulationOrchestrator
        from src.kafka.feature_cache import FeatureCache
        from src.kafka.schemas import KafkaMessage
        from src.kafka.stream_processor import WindowedAggregator
        from src.models.feature_engineering import FeaturePipeline
        from src.models.lstm_autoencoder import select_features
        from src.models.offline_dataset import simulate_labeled_features
        from src.utils.config import get_config

        ds = simulate_labeled_features(1500, start_time=self.START, seed=self.SEED)["warehouse"]
        scaler = OnlineScaler(len(ds.feature_names), clip=10.0)
        scaler.fit_update(ds.X[ds.warm])
        cache = FeatureCache()
        pipe = FeaturePipeline(cache)
        pipe.scalers["warehouse"] = scaler
        agg = WindowedAggregator()
        cfg = get_config().simulation.model_copy(update={"seed": self.SEED})
        orchestrator = SimulationOrchestrator.from_config(
            cfg, start_time=self.START, include_orders=False
        )

        def replay_until(seconds: float) -> None:
            for event in orchestrator.stream(seconds, throttle=False):
                if event["source_type"] != "warehouse":
                    continue
                message = KafkaMessage.from_payload("warehouse", dict(event["payload"]))
                for f in agg.process_message(message).features:
                    cache.store_features(message.entity_id, f)

        names, cols = select_features(ds.feature_names)
        return SimpleNamespace(
            ds=ds, scaler=scaler, pipe=pipe, replay=replay_until, names=names, cols=cols
        )

    def test_offline_window_equals_online_window(self, world):
        world.replay(1500)
        off = make_sequences(
            world.ds, world.scaler, seq_len=60, stride=1, warm_only=False, columns=world.cols
        )
        for entity in sorted(set(world.ds.entity_ids)):
            seq, mask = world.pipe.sequences.build_sequence_with_mask(entity, 60, "warehouse")
            idx = np.flatnonzero(off.entity_ids == entity)
            newest = idx[np.argmax(off.end_timestamps[idx])]
            assert mask.all()
            np.testing.assert_array_equal(seq.numpy()[:, list(world.cols)], off.x[newest])

    def test_detect_uses_real_store_and_columns(self, world):
        model = LSTMAutoencoder(len(world.names), seq_len=60, hidden_sizes=(8, 12), latent_dim=4)
        windows = make_sequences(world.ds, world.scaler, seq_len=60, stride=20, columns=world.cols)
        det = LSTMAutoencoderDetector(
            model, AnomalyThresholdCalibrator(), world.names, sequences=world.pipe.sequences,
            columns=world.cols, mc_passes=3,
        )  # fmt: skip
        det.calibrate(windows.x)
        entities = sorted(set(world.ds.entity_ids))
        results = det.detect_many(entities)
        assert [r.entity_id for r in results] == entities
        assert all(not r.details["warming_up"] for r in results)
        assert all(
            0.0 < r.score < 1.0 and np.isfinite(r.details["reconstruction_error"]) for r in results
        )
        assert all(r.raw_value is not None for r in results)

    def test_unknown_sensor_is_warming_up_not_an_error(self, world):
        det = LSTMAutoencoderDetector(
            tiny_model(),
            AnomalyThresholdCalibrator(),
            NAMES,
            sequences=world.pipe.sequences,
            source_type="warehouse",
            mc_passes=1,
        )
        det.calibrator.fit(np.ones(20), np.ones((20, F)), NAMES)
        result = det.detect("no-such-sensor")
        assert result.details["warming_up"] and result.score == 0.0
