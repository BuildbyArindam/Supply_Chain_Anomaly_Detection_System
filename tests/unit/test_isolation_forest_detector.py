"""Tests for src/models/isolation_forest_detector.py (Day 9)."""

import json
import pickle
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.models.isolation_forest_detector import (
    HyperparameterTuner,
    IsolationForestDetector,
    IsolationForestExplainer,
    Reservoir,
    binary_f1,
    fill_values,
    load_best_params,
)
from src.models.model_registry import ModelRegistry
from src.utils.config import IsolationForestConfig

NAMES = ["value__last", "value__z_baseline", "f2", "f3", "f4", "f5"]
T0 = datetime(2026, 11, 20, 8, tzinfo=UTC)


def make_data(n=2000, anomaly_rate=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES)))
    y = np.zeros(n, dtype=bool)
    k = int(n * anomaly_rate)
    if k:
        idx = rng.choice(n, k, replace=False)
        sign = rng.choice([-1, 1], (k, 1))
        X[np.ix_(idx, [1, 2, 3])] += sign * rng.uniform(5, 8, (k, 3))  # several features at once
        y[idx] = True
    return X, y


def stamps(n, start=0):
    return [T0 + timedelta(seconds=start + i) for i in range(n)]


# ----------------------------------------------------------------- reservoir
class TestReservoir:
    def test_uniform_keeps_capacity_and_is_uniform(self):
        res = Reservoir(100, 1, seed=1)
        for i in range(10_000):
            res.add(np.array([float(i)]), recent=False)
        data = res.data().ravel()
        assert len(res) == 100 and res.seen == 10_000
        assert 3500 < data.mean() < 6500  # uniform over history, not only the tail

    def test_recent_mode_favours_recent_rows(self):
        res = Reservoir(100, 1, seed=1)
        for i in range(10_000):
            res.add(np.array([float(i)]), recent=True)
        data = res.data().ravel()
        assert len(res) == 100 and data.mean() > 9000
        assert data.min() < 9900  # but a thin tail of older rows survives

    def test_load_samples_uniformly_when_larger_than_capacity(self):
        res = Reservoir(50, 2, seed=0)
        res.load(np.arange(1000).reshape(500, 2))
        assert len(res) == 50 and res.seen == 500
        res.load(np.ones((10, 2)))
        assert len(res) == 10

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            Reservoir(0, 3)


# ----------------------------------------------------------------- helpers
def test_fill_values_and_f1():
    X = np.array([[1.0, np.nan], [3.0, np.nan], [np.inf, np.nan]])
    assert fill_values(X).tolist() == [2.0, 0.0]
    y = np.array([True, True, False, False])
    assert binary_f1(y, np.array([True, False, True, False])) == pytest.approx(0.5)
    assert binary_f1(y, np.zeros(4, dtype=bool)) == 0.0
    assert binary_f1(~y & False, np.zeros(4, dtype=bool)) == 0.0


# ----------------------------------------------------------------- detector
class TestDetector:
    def test_fit_scores_anomalies_higher(self):
        X, _ = make_data(3000)
        det = IsolationForestDetector(NAMES, n_estimators=100).fit(X)
        normal, _ = make_data(500, seed=1)
        weird, _ = make_data(500, anomaly_rate=1.0, seed=2)
        s_n, s_a = det.score_batch(normal), det.score_batch(weird)
        assert s_a.mean() > 0.8 > 0.5 > np.median(s_n)
        assert np.all((s_n >= 0) & (s_n <= 1))
        assert det.predict(weird).mean() > 0.9
        assert det.predict(normal).mean() < 0.05

    def test_raw_score_negative_means_anomalous_and_matches_normalised(self):
        X, _ = make_data(2000)
        det = IsolationForestDetector(NAMES).fit(X)
        weird, _ = make_data(50, anomaly_rate=1.0, seed=3)
        raw, score = det.decision_function(weird), det.score_batch(weird)
        assert np.all((raw < 0) == (score > 0.5))
        assert np.corrcoef(raw, score)[0, 1] < -0.9  # monotone decreasing

    def test_contamination_controls_training_flag_rate(self):
        X, _ = make_data(4000)
        for c in (0.01, 0.05):
            det = IsolationForestDetector(NAMES, contamination=c).fit(X)
            assert det.predict(X).mean() == pytest.approx(c, abs=0.01)

    def test_not_fitted_errors(self):
        det = IsolationForestDetector(NAMES)
        for call in (det.score_batch, det.decision_function, det.predict, det.impute):
            with pytest.raises(NotFittedError):
                call(np.zeros((1, 6)))
        with pytest.raises(NotFittedError):
            _ = det.estimator

    def test_nan_and_inf_are_imputed(self):
        X, _ = make_data(1000)
        det = IsolationForestDetector(NAMES).fit(X)
        row = X[:1].copy()
        row[0, 2], row[0, 3] = np.nan, np.inf
        assert np.isfinite(det.score_batch(row)).all()
        assert np.isfinite(det.impute(row)).all()

    def test_shape_validation(self):
        X, _ = make_data(500)
        det = IsolationForestDetector(NAMES).fit(X)
        with pytest.raises(ValueError, match="columns"):
            det.score_batch(np.zeros((2, 5)))
        with pytest.raises(ValueError, match="at least"):
            IsolationForestDetector(NAMES).fit(X[:10])
        with pytest.raises(ValueError, match="same length"):
            det.update_batch(["a"], stamps(2), X[:2])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"contamination": 0.0},
            {"contamination": 0.7},
            {"max_features": 0.0},
            {"max_samples": 1.5},
            {"n_estimators": 0},
            {"refit_every": 0},
            {"reservoir_size": 5},
        ],
    )
    def test_invalid_constructor_arguments(self, kwargs):
        with pytest.raises(ValueError):
            IsolationForestDetector(NAMES, **kwargs)

    def test_invalid_feature_names(self):
        with pytest.raises(ValueError):
            IsolationForestDetector([])
        with pytest.raises(ValueError):
            IsolationForestDetector(["a", "a"])

    def test_deterministic_for_same_seed(self):
        X, _ = make_data(1500)
        a = IsolationForestDetector(NAMES, random_state=3).fit(X).score_batch(X[:100])
        b = IsolationForestDetector(NAMES, random_state=3).fit(X).score_batch(X[:100])
        assert np.array_equal(a, b)

    def test_fit_caps_training_rows_at_reservoir_size(self):
        X, _ = make_data(3000)
        det = IsolationForestDetector(NAMES, reservoir_size=500).fit(X)
        assert det._model.n_train == 500


class TestStreaming:
    def test_warmup_then_scoring_and_result_fields(self):
        X, _ = make_data(600)
        det = IsolationForestDetector(NAMES, warmup_samples=200, refit_every=10_000)
        res = det.update_batch(["s1"] * 600, stamps(600), X)
        assert all(r.details["warming_up"] for r in res[:200])
        assert not any(r.is_anomaly or r.score != 0.0 for r in res[:200])
        assert det.is_fitted and det.model_version == 1 and det.warmup_progress == 1.0
        late = res[300]
        assert not late.details["warming_up"] and late.detector == "isolation_forest"
        assert late.anomaly_type is None and 0 <= late.confidence <= 1
        assert late.raw_value == pytest.approx(X[300, 0])
        assert late.details["model_version"] == 1 and "raw_score" in late.details
        assert late.timestamp == stamps(1, 300)[0] and late.entity_id == "s1"

    def test_streaming_flags_injected_anomalies(self):
        X, _ = make_data(1500)
        det = IsolationForestDetector(NAMES, warmup_samples=1000).fit(X[:1000])
        weird, _ = make_data(100, anomaly_rate=1.0, seed=5)
        res = det.update_batch(["s"] * 100, stamps(100), weird)
        assert np.mean([r.is_anomaly for r in res]) > 0.9

    def test_refit_every_n_scored_and_batch_split_matches_one_by_one(self):
        X, _ = make_data(1200)
        ids = [f"e{i % 3}" for i in range(1200)]
        a = IsolationForestDetector(NAMES, warmup_samples=200, refit_every=300)
        b = IsolationForestDetector(NAMES, warmup_samples=200, refit_every=300)
        batch = a.update_batch(ids, stamps(1200), X)  # one big batch, cut at refits
        single = [b.update(ids[i], stamps(1, i)[0], X[i]) for i in range(0, 1200, 1)]
        assert a.model_version == b.model_version == 1 + 1000 // 300
        assert [r.details.get("model_version") for r in batch] == [
            r.details.get("model_version") for r in single
        ]
        assert np.allclose([r.score for r in batch], [r.score for r in single])

    def test_learn_false_does_not_change_state(self):
        X, _ = make_data(600)
        det = IsolationForestDetector(NAMES, warmup_samples=200).fit(X[:200])
        before = (det.model_version, det._n_since_fit, det._reservoir.seen, det.n_scored)
        det.update_batch(["s"] * 100, stamps(100), X[:100], learn=False)
        assert (det.model_version, det._n_since_fit, det._reservoir.seen, det.n_scored) == before
        cold = IsolationForestDetector(NAMES, warmup_samples=200)
        out = cold.update("s", T0, X[0], learn=False)
        assert out.details["warming_up"] and not cold.is_fitted and cold._n_warm == 0

    def test_flagged_rows_stay_out_of_reservoir_until_max_run(self):
        X, _ = make_data(600)
        det = IsolationForestDetector(
            NAMES, warmup_samples=200, max_flag_run=50, exclude_flagged=True
        ).fit(X[:200])
        weird, _ = make_data(120, anomaly_rate=1.0, seed=9)
        seen0 = det._reservoir.seen
        det.update_batch(["bad"] * 40, stamps(40), weird[:40])
        assert det._reservoir.seen - seen0 < 10  # almost nothing admitted
        det.update_batch(["bad"] * 80, stamps(80), weird[40:])
        assert det._reservoir.seen - seen0 > 30  # run >= 50 flags: admitted again

    def test_flagged_rows_are_kept_by_default(self):
        X, _ = make_data(400)
        det = IsolationForestDetector(NAMES, warmup_samples=200).fit(X[:200])
        assert det.exclude_flagged is False
        weird, _ = make_data(40, anomaly_rate=1.0, seed=9)
        seen0 = det._reservoir.seen
        det.update_batch(["x"] * 40, stamps(40), weird)
        assert det._reservoir.seen - seen0 == 40

    def test_process_checks_schema(self):
        from types import SimpleNamespace

        X, _ = make_data(300)
        det = IsolationForestDetector(NAMES, warmup_samples=100).fit(X)
        good = SimpleNamespace(
            entity_id="s", timestamp=T0, names=tuple(NAMES), raw=X[0], source_type="warehouse"
        )
        assert det.process(good).entity_id == "s"
        assert det.process_many([]) == []
        bad = SimpleNamespace(
            entity_id="v", timestamp=T0, names=("x",), raw=X[0, :1], source_type="vehicle"
        )
        with pytest.raises(ValueError, match="schema mismatch"):
            det.process(bad)

    def test_pickle_roundtrip_keeps_scores(self):
        X, _ = make_data(800)
        det = IsolationForestDetector(NAMES).fit(X)
        clone = pickle.loads(pickle.dumps(det))
        assert np.array_equal(clone.score_batch(X[:50]), det.score_batch(X[:50]))
        clone.update("s", T0, X[0])  # lock was recreated

    def test_registry_roundtrip(self, tmp_path):
        X, _ = make_data(800)
        det = IsolationForestDetector(NAMES).fit(X)
        reg = ModelRegistry(tmp_path)
        reg.save("isolation_forest_warehouse", det, hyperparams=det.get_params())
        loaded = reg.get_latest_model("isolation_forest_warehouse")
        assert np.array_equal(loaded.model.score_batch(X[:20]), det.score_batch(X[:20]))
        assert loaded.metadata["hyperparams"]["n_estimators"] == 200


# ----------------------------------------------------------------- config
class TestConfig:
    def test_from_config_with_and_without_best_params(self, tmp_path):
        cfg = IsolationForestConfig(best_params_path=str(tmp_path / "missing.json"))
        det = IsolationForestDetector.from_config(cfg, NAMES, events_per_second=12)
        assert det.n_estimators == 200 and det.warmup_samples == 3600 * 12
        path = tmp_path / "best.json"
        path.write_text(
            json.dumps(
                {
                    "source_type": "warehouse",
                    "best_params": {"n_estimators": 77, "contamination": 0.04, "junk": 1},
                }
            )
        )
        tuned = IsolationForestDetector.from_config(cfg, NAMES, best_params_path=path)
        assert (tuned.n_estimators, tuned.contamination) == (77, 0.04)
        assert tuned.max_features == cfg.max_features and tuned.warmup_samples == cfg.reservoir_size
        plain = IsolationForestDetector.from_config(
            cfg, NAMES, best_params_path=path, use_best_params=False
        )
        assert plain.n_estimators == 200

    def test_invalid_tuning_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="invalid"):
            load_best_params(path)
        path.write_text('{"x": 1}')
        with pytest.raises(ValueError, match="invalid"):
            load_best_params(path)
        assert load_best_params(tmp_path / "nope.json") == {}


# ----------------------------------------------------------------- tuner
class TestTuner:
    @pytest.fixture
    def data(self):
        Xtr, _ = make_data(1500, seed=0)
        Xv, yv = make_data(1500, anomaly_rate=0.05, seed=1)
        return Xtr, Xv, yv

    def test_tune_improves_or_matches_defaults_and_saves(self, data, tmp_path):
        Xtr, Xv, yv = data
        tuner = HyperparameterTuner(
            Xtr, Xv, yv, n_trials=8, n_chunks=4, search_space={"n_estimators": (30, 80)}
        )
        res = tuner.tune()
        assert res.n_trials == 8 and res.n_complete + res.n_pruned == 8
        assert res.default_f1 is not None and res.best_f1 >= res.default_f1
        assert set(res.best_params) == {
            "n_estimators",
            "contamination",
            "max_features",
            "max_samples",
        }
        assert 0.005 <= res.best_params["contamination"] <= 0.05
        assert 0.3 <= res.best_params["max_features"] <= 1.0
        assert res.best_f1 == pytest.approx(tuner.evaluate(res.best_params))
        out = tuner.save_best_params(tmp_path / "cfg" / "best.json", extra={"note": "t"})
        payload = json.loads(out.read_text())
        assert payload["best_params"] == res.best_params and payload["note"] == "t"
        assert load_best_params(out, "warehouse") == res.best_params

    def test_pruning_happens_with_enough_trials(self, data):
        Xtr, Xv, yv = data
        res = HyperparameterTuner(
            Xtr, Xv, yv, n_trials=25, n_chunks=5, search_space={"n_estimators": (30, 60)}
        ).tune()
        assert res.n_pruned >= 1

    def test_save_before_tune_and_bad_inputs(self, data):
        Xtr, Xv, yv = data
        with pytest.raises(RuntimeError):
            HyperparameterTuner(Xtr, Xv, yv).save_best_params("x.json")
        with pytest.raises(ValueError, match="no anomalies"):
            HyperparameterTuner(Xtr, Xv, np.zeros(len(Xv), bool))
        with pytest.raises(ValueError, match="columns"):
            HyperparameterTuner(Xtr, Xv[:, :3], yv)
        with pytest.raises(ValueError):
            HyperparameterTuner(Xtr, Xv, yv[:5])
        with pytest.raises(ValueError):
            HyperparameterTuner(Xtr, Xv, yv, n_chunks=0)


# ----------------------------------------------------------------- explainer
class TestExplainer:
    @pytest.fixture
    def fitted(self):
        X, _ = make_data(2000)
        return IsolationForestDetector(NAMES, n_estimators=100).fit(X)

    def test_top_feature_is_the_one_that_was_perturbed(self, fitted):
        ex = IsolationForestExplainer(fitted)
        sample = np.zeros(len(NAMES))
        sample[3] = 9.0
        top = ex.get_top_features(sample, n=3)
        assert top[0].feature == "f3" and top[0].shap_value < 0
        assert top[0].feature_value == 9.0
        assert [c.shap_value for c in top] == sorted(c.shap_value for c in top)

    def test_shap_values_shape_and_direction(self, fitted):
        ex = IsolationForestExplainer(fitted)
        normal, _ = make_data(30, seed=4)
        weird = normal.copy()
        weird[:, 1] += 8
        sv_n, sv_a = ex.compute_shap_values(normal), ex.compute_shap_values(weird)
        assert sv_n.shape == (30, len(NAMES))
        assert sv_a[:, 1].mean() < sv_n[:, 1].mean() - 0.5
        assert np.isfinite(ex.expected_value())

    def test_normal_sample_has_few_anomalous_features_and_n_validation(self, fitted):
        ex = IsolationForestExplainer(fitted)
        assert len(ex.get_top_features(np.zeros(len(NAMES)), n=6)) < 6
        assert len(ex.get_top_features(np.zeros(len(NAMES)), n=2, only_anomalous=False)) == 2
        with pytest.raises(ValueError):
            ex.get_top_features(np.zeros(len(NAMES)), n=0)
        with pytest.raises(ValueError):
            ex.get_top_features(np.zeros((2, len(NAMES))))

    def test_explainer_follows_refits_and_handles_nan(self):
        X, _ = make_data(600)
        det = IsolationForestDetector(NAMES, n_estimators=30).fit(X)
        ex = IsolationForestExplainer(det)
        row = X[0].copy()
        row[4] = np.nan
        contribs = ex.get_top_features(row, n=6, only_anomalous=False)
        assert len(contribs) == 6 and np.isfinite([c.shap_value for c in contribs]).all()
        assert [c.feature_value for c in contribs if c.feature == "f4"] == [None]
        first = ex._tree_explainer()
        det.fit(X)
        assert ex._tree_explainer() is not first

    def test_unfitted_detector_raises(self):
        with pytest.raises(NotFittedError):
            IsolationForestExplainer(IsolationForestDetector(NAMES)).compute_shap_values(
                np.zeros((1, 6))
            )
