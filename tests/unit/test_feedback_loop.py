"""Feedback loop (Day 16): events, store, calibrator, recalibrator, drift, processor."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.models.ensemble_detector import LSTM_CLASSIFIER, MEMBERS, EnsembleDetector, MetaLearner
from src.models.feedback_loop import (
    EnsembleRecalibrator,
    FeedbackDriftDetector,
    FeedbackEvent,
    FeedbackProcessor,
    FeedbackStore,
    FeedbackValidationError,
    ThresholdCalibrator,
    validate_feedback,
)
from src.models.lstm_classifier import TemperatureScaler
from src.models.model_registry import ModelRegistry

T0 = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)
N_FEATURES = 10


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def event(
    n: int = 0,
    kind: str = "false_positive",
    *,
    entity: str = "wh1-A-temp-01",
    analyst: str = "a1",
    at: datetime | None = None,
    **extra: Any,
) -> FeedbackEvent:
    return FeedbackEvent.create(
        f"alert-{n}",
        analyst,
        kind,
        entity_id=entity,
        timestamp=at if at is not None else T0 + timedelta(seconds=n),
        **extra,
    )


def make_rows(
    rng: np.random.Generator, n: int, positive_share: float, *, noisy_statistical: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Meta-feature rows: informative but noisy member scores (``noisy_statistical`` makes the
    statistical detector fire on normal events: the typical false-positive source)."""
    y = rng.random(n) < positive_share
    x = np.full((n, N_FEATURES), np.nan)
    for i in range(len(MEMBERS)):
        score = np.where(y, rng.normal(0.7, 0.2, n), rng.normal(0.3, 0.2, n))
        x[:, i] = np.clip(score, 0, 1)
        x[:, 4 + i] = np.abs(2 * x[:, i] - 1)
    if noisy_statistical:
        x[~y, 0] = np.clip(rng.normal(0.9, 0.05, int((~y).sum())), 0, 1)
    x[:, 8] = rng.random(n) * 0.1
    x[:, 9] = rng.integers(0, 24, n)
    return x, y


def as_features(row: np.ndarray) -> tuple[float | None, ...]:
    return tuple(None if np.isnan(v) else float(v) for v in row)


@pytest.fixture(scope="module")
def fitted_meta_learner() -> MetaLearner:
    x, y = make_rows(np.random.default_rng(0), 600, 0.3)
    learner = MetaLearner(n_estimators=60, min_child_samples=5)
    learner.fit(x, y, cv_folds=3)
    return learner


def fill_store(
    store: FeedbackStore, learner: MetaLearner, n_alerts: int = 120, seed: int = 1
) -> dict[str, int]:
    """Verdicts on the alerts the (old) learner raised in a drifted stream."""
    rng = np.random.default_rng(seed)
    x, y = make_rows(rng, 400, 0.5, noisy_statistical=True)
    flagged = np.flatnonzero(learner.predict_proba(x) >= learner.threshold)[:n_alerts]
    for i in flagged:
        store.add(
            event(
                int(i),
                "true_positive" if y[i] else "false_positive",
                entity=f"e{int(i) % 5}",
                at=datetime.now(UTC) - timedelta(minutes=int(i)),
                meta_features=as_features(x[i]),
            )
        )
    return {"tp": int(y[flagged].sum()), "fp": int((~y[flagged]).sum())}


class FakeClassifier:
    """Looks like an LSTM classifier detector for the temperature step."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = TemperatureScaler(temperature)


def ensembles_with(learner: MetaLearner | None, *, classifier: FakeClassifier | None = None):
    sources = ("warehouse", "vehicle", "inventory")
    result = {}
    for source in sources:
        ensemble = EnsembleDetector(meta_learner=learner)
        if classifier is not None:
            ensemble.members[LSTM_CLASSIFIER] = classifier
        result[source] = ensemble
    return result


# --------------------------------------------------------------------------- #
# FeedbackEvent and validation
# --------------------------------------------------------------------------- #
def test_event_round_trips_through_a_dict():
    original = event(
        3,
        "severity_adjustment",
        corrected_severity="CRITICAL",
        original_severity="INFO",
        notes="more serious",
        meta_features=(0.5, None) + (0.1,) * 8,
    )
    restored = FeedbackEvent.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.timestamp.tzinfo is not None


def test_event_reads_the_day_14_record_layout():
    legacy = {
        "feedback_id": "f-1",
        "anomaly_event_id": "alert-1",
        "entity_id": "s1",
        "feedback_type": "false_positive",
        "analyst_id": "a1",
        "severity": "WARNING",
        "flagged_by": ["statistical"],
        "notes": None,
        "timestamp": "2026-11-24T08:00:00+00:00",
    }
    restored = FeedbackEvent.from_dict(legacy)
    assert restored.event_id == "f-1" and restored.original_severity == "WARNING"
    assert restored.is_false_positive and restored.meta_features is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"feedback_type": "maybe"}, "feedback_type"),
        ({"entity_id": " "}, "entity_id"),
        ({"analyst_id": ""}, "analyst_id"),
        ({"anomaly_event_id": ""}, "anomaly_event_id"),
        ({"notes": "n" * 1001}, "notes"),
        ({"feedback_type": "severity_adjustment"}, "corrected_severity"),
        (
            {"feedback_type": "severity_adjustment", "corrected_severity": "SEVERE"},
            "corrected_severity",
        ),
        ({"corrected_severity": "INFO"}, "only applies"),
        ({"missed": True}, "missed"),
        ({"meta_features": (0.1,) * 3}, "meta_features"),
        ({"meta_features": (float("nan"),) + (0.1,) * 9}, "finite"),
    ],
)
def test_validation_rejects_malformed_events(changes, message):
    from dataclasses import replace

    with pytest.raises(FeedbackValidationError, match=message):
        validate_feedback(replace(event(), **changes))


def test_validation_accepts_the_three_kinds():
    validate_feedback(event(kind="false_positive"))
    validate_feedback(event(kind="true_positive", missed=True))
    validate_feedback(event(kind="severity_adjustment", corrected_severity="EMERGENCY"))


# --------------------------------------------------------------------------- #
# FeedbackStore
# --------------------------------------------------------------------------- #
def test_store_replaces_same_analyst_and_keeps_other_analysts():
    store = FeedbackStore()
    assert store.add(event(1, "false_positive", analyst="a1")) is None
    replaced = store.add(event(1, "true_positive", analyst="a1"))
    assert replaced is not None and replaced.feedback_type == "false_positive"
    assert store.add(event(1, "true_positive", analyst="a2")) is None
    assert len(store) == 2 and store.total_added == 3
    assert {e.analyst_id for e in store.for_alert("alert-1")} == {"a1", "a2"}


def test_severity_correction_never_replaces_a_label():
    store = FeedbackStore()
    store.add(event(1, "false_positive"))
    store.add(event(1, "severity_adjustment", corrected_severity="INFO"))
    assert len(store) == 2
    assert [e.feedback_type for _, e in store.labelled_since(0)] == ["false_positive"]


def test_labelled_since_uses_sequence_numbers():
    store = FeedbackStore()
    for i in range(5):
        store.add(event(i))
    assert [s for s, _ in store.labelled_since(2)] == [3, 4, 5]
    assert store.labelled_since(5) == []


def test_save_and_load_round_trip_with_sequence(tmp_path: Path):
    path = tmp_path / "nested" / "feedback_store.json"
    store = FeedbackStore(path)
    for i in range(4):
        store.add(event(i, "true_positive" if i % 2 else "false_positive"))
    assert store.save() == 4
    assert json.loads(path.read_text())["version"] == 2
    assert not list(path.parent.glob("*.tmp"))

    restored = FeedbackStore(path)
    assert restored.load() == 4
    assert restored.all() == store.all()
    assert restored.total_added == 4
    restored.add(event(99))
    assert restored.total_added == 5  # sequence numbers continue, they are never reused


def test_save_is_atomic_when_the_write_fails(tmp_path: Path, monkeypatch):
    path = tmp_path / "feedback_store.json"
    store = FeedbackStore(path)
    store.add(event(1))
    store.save()
    before = path.read_text()
    store.add(event(2))

    def boom(*_: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.save()
    assert path.read_text() == before  # the old file is intact
    assert not list(tmp_path.glob("*.tmp"))  # and no temp file is left behind


def test_autosave_persists_every_add(tmp_path: Path):
    path = tmp_path / "feedback_store.json"
    store = FeedbackStore()
    store.attach(path)
    store.add(event(1))
    store.add(event(2))
    assert len(json.loads(path.read_text())["events"]) == 2


def test_autosave_failure_does_not_break_add(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    store = FeedbackStore(blocker / "feedback_store.json", autosave=True)
    assert store.add(event(1)) is None  # the save fails (logged); the verdict is still kept
    assert len(store) == 1


def test_load_reads_the_day_14_file_and_skips_bad_rows(tmp_path: Path):
    path = tmp_path / "feedback_store.json"
    good = {
        "feedback_id": "f-1",
        "anomaly_event_id": "alert-1",
        "entity_id": "s1",
        "feedback_type": "true_positive",
        "analyst_id": "a1",
        "severity": "CRITICAL",
        "timestamp": "2026-11-24T08:00:00+00:00",
    }
    broken = {"feedback_id": "f-2"}
    invalid_type = {**good, "feedback_id": "f-3", "feedback_type": "maybe"}
    path.write_text(json.dumps({"version": 1, "feedback": [good, broken, invalid_type]}))
    store = FeedbackStore(path)
    assert store.load() == 1
    assert store.all()[0].original_severity == "CRITICAL"


def test_load_tolerates_missing_and_corrupt_files(tmp_path: Path):
    assert FeedbackStore(tmp_path / "absent.json").load() == 0
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert FeedbackStore(corrupt).load() == 0


def test_store_is_thread_safe():
    store = FeedbackStore()

    def work(worker: int) -> None:
        for i in range(100):
            store.add(event(worker * 1000 + i, analyst=f"a{worker}"))

    threads = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(store) == 800 and store.total_added == 800


# --------------------------------------------------------------------------- #
# ThresholdCalibrator
# --------------------------------------------------------------------------- #
def test_ema_follows_the_specification_with_alpha_point_one():
    calibrator = ThresholdCalibrator(fp_burst_count=100)
    state = calibrator.update("s1", True)
    assert state.fp_rate == pytest.approx(0.1) and state.tp_rate == pytest.approx(0.0)
    state = calibrator.update("s1", True)
    assert state.fp_rate == pytest.approx(0.19)
    state = calibrator.update("s1", False)
    assert state.fp_rate == pytest.approx(0.171) and state.tp_rate == pytest.approx(0.1)
    assert (state.n_fp, state.n_tp) == (2, 1)


def test_more_than_five_false_positives_in_an_hour_raise_the_threshold_by_a_tenth():
    calibrator = ThresholdCalibrator()
    for i in range(5):
        calibrator.update("s1", True, timestamp=T0 + timedelta(minutes=i))
    assert calibrator.get_adjusted_threshold("s1", 0.5) == pytest.approx(0.5)  # 5 is not "> 5"
    calibrator.update("s1", True, timestamp=T0 + timedelta(minutes=5))
    assert calibrator.get_adjusted_threshold("s1", 0.5) == pytest.approx(0.6)
    # the burst counter restarted: it takes another six false positives to raise it again
    for i in range(5):
        calibrator.update("s1", True, timestamp=T0 + timedelta(minutes=6 + i))
    assert calibrator.get_adjusted_threshold("s1", 0.5) == pytest.approx(0.6)
    calibrator.update("s1", True, timestamp=T0 + timedelta(minutes=11))
    assert calibrator.get_adjusted_threshold("s1", 0.5) == pytest.approx(0.7)


def test_false_positives_spread_over_more_than_an_hour_do_not_raise_it():
    calibrator = ThresholdCalibrator()
    for i in range(12):
        calibrator.update("s1", True, timestamp=T0 + timedelta(minutes=15 * i))
    assert calibrator.get_adjustment("s1") == 0.0


def test_burst_of_one_entity_does_not_affect_another():
    calibrator = ThresholdCalibrator()
    for i in range(6):
        calibrator.update("noisy", True, timestamp=T0 + timedelta(seconds=i))
    assert calibrator.get_adjustment("noisy") == pytest.approx(0.1)
    assert calibrator.get_adjustment("quiet") == 0.0
    assert calibrator.get_adjusted_threshold("quiet", 0.5) == 0.5
    assert calibrator.state("quiet") is None


def test_more_than_three_missed_true_positives_lower_the_threshold_by_five_hundredths():
    calibrator = ThresholdCalibrator()
    for _ in range(3):
        calibrator.update("s1", False, missed=True)
    assert calibrator.get_adjustment("s1") == 0.0
    state = calibrator.update("s1", False, missed=True)
    assert state.adjustment == pytest.approx(-0.05) and state.missed_tps == 0
    assert calibrator.get_adjusted_threshold("s1", 0.5) == pytest.approx(0.45)


def test_a_true_positive_that_was_alerted_is_not_a_miss():
    calibrator = ThresholdCalibrator()
    for _ in range(10):
        calibrator.update("s1", False)
    assert calibrator.get_adjustment("s1") == 0.0


def test_adjustment_is_capped_in_both_directions():
    calibrator = ThresholdCalibrator(max_raise=0.25, max_lower=0.1)
    for i in range(60):
        calibrator.update("up", True, timestamp=T0 + timedelta(seconds=i))
    assert calibrator.get_adjustment("up") == pytest.approx(0.25)
    for _ in range(40):
        calibrator.update("down", False, missed=True)
    assert calibrator.get_adjustment("down") == pytest.approx(-0.1)


def test_adjusted_threshold_stays_inside_bounds():
    calibrator = ThresholdCalibrator(max_raise=0.45, max_lower=0.45)
    for i in range(60):
        calibrator.update("up", True, timestamp=T0 + timedelta(seconds=i))
    assert calibrator.get_adjusted_threshold("up", 0.9) == 0.95
    for _ in range(80):
        calibrator.update("down", False, missed=True)
    assert calibrator.get_adjusted_threshold("down", 0.1) == 0.05


def test_missed_false_positive_is_rejected_and_bad_settings_too():
    with pytest.raises(ValueError, match="missed"):
        ThresholdCalibrator().update("s1", True, missed=True)
    with pytest.raises(ValueError, match="alpha"):
        ThresholdCalibrator(alpha=0.0)
    with pytest.raises(ValueError, match="burst"):
        ThresholdCalibrator(fp_burst_count=0)
    with pytest.raises(ValueError, match="non-negative"):
        ThresholdCalibrator(fp_threshold_raise=-0.1)


def test_calibrator_from_config_uses_the_configured_values():
    from src.utils.config import FeedbackConfig

    cfg = FeedbackConfig(ema_alpha=0.5, fp_burst_count=2, fp_threshold_raise=0.2)
    calibrator = ThresholdCalibrator.from_config(cfg)
    for i in range(3):
        calibrator.update("s1", True, timestamp=T0 + timedelta(seconds=i))
    assert calibrator.get_adjustment("s1") == pytest.approx(0.2)
    assert calibrator.state("s1")["fp_rate"] == pytest.approx(0.875)


def test_rebuild_replays_stored_verdicts_like_the_live_calibrator():
    verdicts = [event(i, "false_positive", at=T0 + timedelta(minutes=i)) for i in range(8)]
    verdicts += [
        event(20 + i, "true_positive", missed=True, at=T0 + timedelta(hours=2, minutes=i))
        for i in range(4)
    ]
    verdicts.append(event(50, "severity_adjustment", corrected_severity="INFO"))
    live = ThresholdCalibrator()
    for v in verdicts[:-1]:
        live.update(v.entity_id, v.is_false_positive, missed=v.missed, timestamp=v.timestamp)
    replayed = ThresholdCalibrator()
    assert replayed.rebuild(reversed(verdicts)) == 12  # order-independent, labels only
    assert replayed.snapshot() == live.snapshot()


# --------------------------------------------------------------------------- #
# MetaLearner.continue_fit
# --------------------------------------------------------------------------- #
def test_continue_fit_adds_trees_and_leaves_the_original_untouched(fitted_meta_learner):
    x, y = make_rows(np.random.default_rng(5), 80, 0.5)
    before = fitted_meta_learner.predict_proba(x)
    trees = fitted_meta_learner._model.booster_.num_trees()
    successor = fitted_meta_learner.continue_fit(x, y, n_estimators=7)
    assert successor is not fitted_meta_learner
    assert successor._model.booster_.num_trees() == trees + 7
    assert successor.threshold == fitted_meta_learner.threshold
    np.testing.assert_array_equal(fitted_meta_learner.predict_proba(x), before)
    assert not np.allclose(successor.predict_proba(x), before)
    assert fitted_meta_learner.params["n_estimators"] == successor.params["n_estimators"]


def test_continue_fit_validates_its_input(fitted_meta_learner):
    x, y = make_rows(np.random.default_rng(5), 30, 0.5)
    with pytest.raises(ValueError, match="both classes"):
        fitted_meta_learner.continue_fit(x, np.ones(30, dtype=bool))
    with pytest.raises(ValueError, match="n_estimators"):
        fitted_meta_learner.continue_fit(x, y, n_estimators=0)
    with pytest.raises(ValueError, match="shape"):
        fitted_meta_learner.continue_fit(x[:, :4], y)
    with pytest.raises(RuntimeError, match="not fitted"):
        MetaLearner().continue_fit(x, y)


# --------------------------------------------------------------------------- #
# EnsembleRecalibrator
# --------------------------------------------------------------------------- #
def test_recalibration_reduces_false_positives_and_swaps_the_learner(fitted_meta_learner):
    store = FeedbackStore()
    counts = fill_store(store, fitted_meta_learner)
    assert counts["fp"] > 10 and counts["tp"] > 10  # the old model really over-alerts
    ensembles = ensembles_with(fitted_meta_learner)
    recalibrator = EnsembleRecalibrator(lambda: ensembles, store)
    old_trees = fitted_meta_learner._model.booster_.num_trees()

    report = recalibrator.run_once(force=True)

    assert report.status == "applied", report
    assert report.n_feedback_events_used == counts["tp"] + counts["fp"]
    assert report.n_train + report.n_validation == report.n_feedback_events_used
    assert report.n_validation >= 2
    assert report.new_log_loss <= report.old_log_loss
    assert report.estimated_fp_reduction is not None and report.estimated_fp_reduction > 0
    assert report.tp_retention is not None and report.tp_retention > 0.5
    assert set(report.old_weights) == set(report.new_weights)
    assert report.new_weights != report.old_weights
    assert sum(report.new_weights.values()) == pytest.approx(1.0, abs=1e-3)
    # every source-type ensemble now serves the same new learner; the old one is untouched
    served = {id(e.meta_learner) for e in ensembles.values()}
    assert len(served) == 1 and served != {id(fitted_meta_learner)}
    assert next(iter(ensembles.values())).meta_learner._model.booster_.num_trees() > old_trees
    assert fitted_meta_learner._model.booster_.num_trees() == old_trees
    assert recalibrator.history[-1] is report


def test_threshold_moves_by_at_most_the_configured_shift(fitted_meta_learner):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    report = EnsembleRecalibrator(lambda: ensembles, store, max_threshold_shift=0.02).run_once(
        force=True
    )
    assert report.status == "applied"
    assert abs(report.new_threshold - report.old_threshold) <= 0.02 + 1e-6


def test_each_verdict_is_used_once(fitted_meta_learner):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    recalibrator = EnsembleRecalibrator(lambda: ensembles, store)
    assert recalibrator.run_once(force=True).status == "applied"
    again = recalibrator.run_once(force=True)
    assert again.status == "skipped" and "0 usable" in again.reason
    fill_store(store, ensembles["warehouse"].meta_learner, seed=2)  # new verdicts arrive
    assert recalibrator.run_once(force=True).n_feedback_events_used > 0


def test_skips_with_too_few_verdicts_and_keeps_them_for_later(fitted_meta_learner):
    store = FeedbackStore()
    ensembles = ensembles_with(fitted_meta_learner)
    recalibrator = EnsembleRecalibrator(lambda: ensembles, store, min_events=20)
    x, y = make_rows(np.random.default_rng(3), 10, 0.5)
    for i in range(10):
        store.add(
            event(i, "true_positive" if y[i] else "false_positive", meta_features=as_features(x[i]))
        )
    report = recalibrator.run_once(force=True)
    assert report.status == "skipped" and "need 20" in report.reason
    assert ensembles["warehouse"].meta_learner is fitted_meta_learner
    # the skipped verdicts are not consumed: they count when more arrive
    x2, y2 = make_rows(np.random.default_rng(4), 40, 0.5)
    for i in range(40):
        store.add(
            event(
                100 + i,
                "true_positive" if y2[i] else "false_positive",
                meta_features=as_features(x2[i]),
            )
        )
    assert recalibrator.run_once(force=True).n_feedback_events_used == 50


def test_skips_when_one_class_is_missing(fitted_meta_learner):
    store = FeedbackStore()
    x, _ = make_rows(np.random.default_rng(3), 40, 0.5)
    for i in range(40):
        store.add(event(i, "false_positive", meta_features=as_features(x[i])))
    ensembles = ensembles_with(fitted_meta_learner)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.status == "skipped" and "of each class" in report.reason


def test_verdicts_without_meta_features_are_ignored(fitted_meta_learner):
    store = FeedbackStore()
    for i in range(40):
        store.add(event(i, "true_positive" if i % 2 else "false_positive"))  # no meta_features
    ensembles = ensembles_with(fitted_meta_learner)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.status == "skipped" and "0 usable" in report.reason


def test_skips_the_stacker_when_only_the_weighted_average_is_serving():
    store = FeedbackStore()
    fill_store(store, _fresh_learner())
    ensembles = ensembles_with(None)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.status == "skipped" and "weighted-average" in report.reason
    assert all(e.meta_learner is None for e in ensembles.values())


def _fresh_learner() -> MetaLearner:
    x, y = make_rows(np.random.default_rng(0), 600, 0.3)
    learner = MetaLearner(n_estimators=60, min_child_samples=5)
    learner.fit(x, y, cv_folds=3)
    return learner


def test_a_worse_challenger_is_rejected_and_nothing_changes(fitted_meta_learner, monkeypatch):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    original = MetaLearner.continue_fit

    def sabotaged(self, features, y, **kwargs):
        # a challenger trained on the opposite labels is certainly worse on the held-out part
        return original(self, features, ~np.asarray(y, dtype=bool), n_estimators=60)

    monkeypatch.setattr(MetaLearner, "continue_fit", sabotaged)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.status == "rejected" and "worse" in report.reason
    assert report.new_log_loss > report.old_log_loss
    assert all(e.meta_learner is fitted_meta_learner for e in ensembles.values())


def test_refit_failure_is_reported_not_raised(fitted_meta_learner, monkeypatch):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)

    def boom(self, *args, **kwargs):
        raise ValueError("lightgbm exploded")

    monkeypatch.setattr(MetaLearner, "continue_fit", boom)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.status == "skipped" and "lightgbm exploded" in report.reason


def test_the_new_learner_is_saved_as_a_registry_version(fitted_meta_learner, tmp_path: Path):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    registry = ModelRegistry(tmp_path / "registry")
    report = EnsembleRecalibrator(lambda: ensembles, store, registry=registry).run_once(force=True)
    assert report.status == "applied" and report.registry_version
    loaded = registry.load("ensemble_meta_learner", report.registry_version)
    assert isinstance(loaded.model, MetaLearner)
    assert loaded.metadata["extra"]["source"] == "feedback_recalibration"
    np.testing.assert_allclose(
        loaded.model.predict_proba(make_rows(np.random.default_rng(9), 20, 0.5)[0]),
        ensembles["warehouse"].meta_learner.predict_proba(
            make_rows(np.random.default_rng(9), 20, 0.5)[0]
        ),
    )


def test_a_failing_registry_does_not_undo_the_swap(fitted_meta_learner):
    class BrokenRegistry:
        def save(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("read-only file system")

    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    report = EnsembleRecalibrator(lambda: ensembles, store, registry=BrokenRegistry()).run_once(
        force=True
    )
    assert report.status == "applied" and report.registry_version is None
    assert ensembles["warehouse"].meta_learner is not fitted_meta_learner


def test_temperature_is_softened_when_the_classifier_is_overconfident(fitted_meta_learner):
    rng = np.random.default_rng(11)
    store = FeedbackStore()
    x, y = make_rows(rng, 200, 0.5)
    # the classifier claims 0.97 / 0.03, but is right only ~65% of the time
    right = rng.random(200) < 0.65
    verdict = np.where(right, y, ~y)
    x[:, 3] = np.where(y, 0.97, 0.03)
    for i in range(200):
        store.add(
            event(
                i,
                "true_positive" if verdict[i] else "false_positive",
                meta_features=as_features(x[i]),
            )
        )
    classifier = FakeClassifier(1.0)
    ensembles = ensembles_with(fitted_meta_learner, classifier=classifier)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.temperature_old == 1.0
    assert report.temperature_new > 1.0
    assert classifier.temperature.temperature == pytest.approx(report.temperature_new)
    assert classifier.temperature.temperature <= 2.0  # one run moves it by at most a factor of 2
    assert classifier.temperature.is_fitted


def test_temperature_is_unchanged_without_a_classifier(fitted_meta_learner):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    report = EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    assert report.temperature_old is None and report.temperature_new is None


def test_triggers_count_and_time(fitted_meta_learner):
    clock = {"t": 0.0}
    store = FeedbackStore()
    ensembles = ensembles_with(fitted_meta_learner)
    recalibrator = EnsembleRecalibrator(
        lambda: ensembles,
        store,
        every_minutes=30,
        after_events=50,
        clock=lambda: clock["t"],
    )
    assert not recalibrator.is_due() and recalibrator.maybe_run() is None  # nothing new
    for i in range(10):
        store.add(event(i))
    assert not recalibrator.is_due()  # 10 new, 0 minutes
    clock["t"] = 29 * 60
    assert not recalibrator.is_due()
    clock["t"] = 30 * 60
    assert recalibrator.is_due()  # time trigger
    assert recalibrator.maybe_run() is not None  # (skipped for lack of data, but it ran)
    assert not recalibrator.is_due() and recalibrator.new_events == 0
    for i in range(50):
        store.add(event(100 + i))
    assert recalibrator.is_due()  # count trigger, no time has passed


def test_background_thread_runs_when_the_count_trigger_fires(fitted_meta_learner):
    store = FeedbackStore()
    ensembles = ensembles_with(fitted_meta_learner)
    recalibrator = EnsembleRecalibrator(
        lambda: ensembles, store, after_events=40, poll_interval_s=60.0
    )
    recalibrator.start()
    recalibrator.start()  # idempotent
    try:
        fill_store(store, fitted_meta_learner)
        recalibrator.notify()  # what the processor does after each verdict
        deadline = time.monotonic() + 30
        while not recalibrator.history and time.monotonic() < deadline:
            time.sleep(0.05)
        assert recalibrator.history and recalibrator.history[-1].status == "applied"
    finally:
        recalibrator.stop()
    assert recalibrator._thread is None


def test_background_thread_survives_a_crashing_run(fitted_meta_learner, monkeypatch):
    store = FeedbackStore()
    recalibrator = EnsembleRecalibrator(
        lambda: ensembles_with(fitted_meta_learner), store, after_events=1, poll_interval_s=0.02
    )
    calls = []

    def crash() -> None:
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(recalibrator, "_recalibrate", crash)
    store.add(event(1))
    recalibrator.start()
    try:
        deadline = time.monotonic() + 10
        while not calls and time.monotonic() < deadline:
            time.sleep(0.02)
        assert calls
        assert recalibrator._thread is not None and recalibrator._thread.is_alive()
    finally:
        recalibrator.stop()


def test_recalibrator_logs_the_required_fields(fitted_meta_learner, caplog):
    store = FeedbackStore()
    fill_store(store, fitted_meta_learner)
    ensembles = ensembles_with(fitted_meta_learner)
    with caplog.at_level(logging.INFO, logger="src.models.feedback_loop"):
        EnsembleRecalibrator(lambda: ensembles, store).run_once(force=True)
    for key in ("n_feedback_events_used", "new_vs_old_weights", "estimated_fp_reduction"):
        assert key in caplog.text


def test_recalibrator_from_config_uses_the_configured_values(fitted_meta_learner):
    from src.utils.config import FeedbackConfig

    cfg = FeedbackConfig(
        recalibrate_every_minutes=5,
        recalibrate_after_events=7,
        recalibrate_min_events=9,
        recalibrate_new_trees=11,
    )
    recalibrator = EnsembleRecalibrator.from_config(cfg, lambda: {}, FeedbackStore())
    assert (recalibrator.every_s, recalibrator.after_events) == (300.0, 7)
    assert (recalibrator.min_events, recalibrator.n_new_trees) == (9, 11)


# --------------------------------------------------------------------------- #
# FeedbackDriftDetector (Page-Hinkley)
# --------------------------------------------------------------------------- #
def bernoulli(rng: np.random.Generator, p: float, n: int) -> list[bool]:
    return [bool(v) for v in rng.random(n) < p]


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_no_alert_while_the_false_positive_rate_is_stable(rate):
    false_alarms = 0
    for seed in range(20):
        detector = FeedbackDriftDetector()
        stream = bernoulli(np.random.default_rng(seed), rate, 1000)
        false_alarms += sum(detector.update(v) is not None for v in stream)
    # 20 stable runs x 1000 verdicts: a few stray alarms at most (the old defaults gave ~15)
    assert false_alarms <= 4


def test_alert_when_the_false_positive_rate_jumps():
    rng = np.random.default_rng(0)
    published: list[dict[str, Any]] = []
    detector = FeedbackDriftDetector(publisher=published.append)
    for v in bernoulli(rng, 0.2, 200):
        assert detector.update(v) is None
    alert = None
    for n, v in enumerate(bernoulli(rng, 0.9, 100), start=1):  # noqa: B007
        alert = detector.update(v)
        if alert:
            break
    assert alert is not None and n < 40  # caught within a few dozen verdicts of the shift
    assert alert.direction == "increase" and alert.recommended_action == "full_retrain"
    assert alert.recent_fp_rate > alert.baseline_fp_rate
    assert alert.baseline_fp_rate < 0.3 and alert.recent_fp_rate > 0.5  # ~0.2 before, ~0.9 after
    assert "rose to" in alert.message and f"{alert.recent_fp_rate:.0%}" in alert.message
    assert len(published) == 1
    message = published[0]
    assert message["event_type"] == "ModelDriftAlert" and message["detector"] == "page_hinkley"
    assert message["alert_id"] == alert.alert_id
    json.dumps(message)  # JSON-serialisable for the Kafka topic


def test_detector_resets_after_an_alarm_and_needs_fresh_evidence():
    detector = FeedbackDriftDetector(min_samples=10, threshold=3.0)
    for _ in range(20):
        assert detector.update(False) is None
    fired_at = None
    for i in range(1, 60):
        if detector.update(True):
            fired_at = i
            break
    assert fired_at is not None and len(detector.alerts) == 1
    # the test started over: no statistic, no memory of the old baseline
    assert detector.statistic == 0.0 and detector._n == 0
    # ... so the next alarm needs at least min_samples fresh verdicts
    again = [detector.update(True) for _ in range(9)]
    assert again == [None] * 9 and len(detector.alerts) == 1


def test_no_alert_before_min_samples():
    detector = FeedbackDriftDetector(min_samples=30, threshold=0.5)
    assert all(detector.update(i >= 5) is None for i in range(29))


def test_decrease_direction_catches_a_sudden_drop_in_false_positives():
    rng = np.random.default_rng(1)
    increase_only = FeedbackDriftDetector()
    decrease = FeedbackDriftDetector(direction="decrease")
    both = FeedbackDriftDetector(direction="both")
    stream = bernoulli(rng, 0.8, 200) + bernoulli(rng, 0.02, 100)
    results = {
        name: [d.update(v) for v in stream]
        for name, d in (("inc", increase_only), ("dec", decrease), ("both", both))
    }
    assert not any(results["inc"])
    assert any(results["dec"]) and any(results["both"])
    assert decrease.alerts[0].direction == "decrease" and both.alerts[0].direction == "decrease"


def test_a_failing_publisher_does_not_break_feedback_processing():
    def broken(_: dict[str, Any]) -> None:
        raise ConnectionError("kafka is down")

    detector = FeedbackDriftDetector(min_samples=5, threshold=1.0, publisher=broken)
    alerts = [detector.update(i >= 5) for i in range(40)]
    assert any(alerts) and detector.alerts


def test_detector_rejects_bad_settings_and_reads_the_config():
    from src.utils.config import FeedbackConfig

    with pytest.raises(ValueError, match="threshold"):
        FeedbackDriftDetector(threshold=0)
    with pytest.raises(ValueError, match="min_samples"):
        FeedbackDriftDetector(min_samples=1)
    cfg = FeedbackConfig(drift_delta=0.1, drift_threshold=2.0, drift_min_samples=5)
    detector = FeedbackDriftDetector.from_config(cfg)
    assert (detector.delta, detector.threshold, detector.min_samples) == (0.1, 2.0, 5)


# --------------------------------------------------------------------------- #
# FeedbackProcessor
# --------------------------------------------------------------------------- #
def make_processor(**kwargs: Any) -> FeedbackProcessor:
    return FeedbackProcessor(FeedbackStore(), ThresholdCalibrator(), **kwargs)


def test_process_stores_and_calibrates():
    processor = make_processor()
    results = [processor.process(event(i, at=T0 + timedelta(seconds=i))) for i in range(6)]
    assert len(processor.store) == 6
    assert results[4].threshold_adjustment == 0.0
    assert results[5].threshold_adjustment == pytest.approx(0.1)
    assert processor.calibrator.get_adjusted_threshold("wh1-A-temp-01", 0.5) == pytest.approx(0.6)


def test_invalid_event_is_rejected_before_anything_changes():
    processor = make_processor()
    with pytest.raises(FeedbackValidationError):
        processor.process(event(1, kind="maybe"))
    assert len(processor.store) == 0 and processor.calibrator.snapshot() == {}


def test_a_changed_verdict_replaces_but_is_not_counted_twice():
    processor = make_processor()
    first = processor.process(event(1, "false_positive"))
    assert first.replaced is None
    second = processor.process(event(1, "true_positive"))
    assert second.replaced is not None and second.replaced.is_false_positive
    state = processor.calibrator.state("wh1-A-temp-01")
    assert (state["n_fp"], state["n_tp"]) == (1, 0)
    assert processor.store.all()[0].feedback_type == "true_positive"  # the store keeps the latest


def test_severity_adjustment_is_stored_but_does_not_calibrate():
    drift = FeedbackDriftDetector()
    processor = make_processor(drift_detector=drift)
    result = processor.process(
        event(1, "severity_adjustment", corrected_severity="EMERGENCY", original_severity="WARNING")
    )
    assert result.threshold_adjustment is None and result.drift_alert is None
    assert len(processor.store) == 1 and processor.calibrator.snapshot() == {}
    assert drift.statistic == 0.0


def test_missed_detection_reaches_the_calibrator():
    processor = make_processor()
    for i in range(4):
        processor.process(event(i, "true_positive", missed=True, analyst=f"a{i}"))
    assert processor.calibrator.get_adjustment("wh1-A-temp-01") == pytest.approx(-0.05)


def test_drift_alert_is_returned_and_published():
    published: list[dict[str, Any]] = []
    detector = FeedbackDriftDetector(min_samples=10, threshold=3.0, publisher=published.append)
    processor = make_processor(drift_detector=detector)
    alerts = []
    for i in range(80):
        result = processor.process(
            event(i, "false_positive" if i >= 30 else "true_positive", entity=f"e{i % 7}")
        )
        if result.drift_alert:
            alerts.append(result.drift_alert)
    assert alerts and published and published[0]["event_type"] == "ModelDriftAlert"


def test_processor_nudges_the_recalibrator(fitted_meta_learner):
    store = FeedbackStore()
    ensembles = ensembles_with(fitted_meta_learner)
    recalibrator = EnsembleRecalibrator(lambda: ensembles, store, after_events=3)
    processor = FeedbackProcessor(store, ThresholdCalibrator(), recalibrator=recalibrator)
    processor.process(event(1))
    assert not recalibrator._wake.is_set()
    processor.process(event(2, entity="e2"))
    processor.process(event(3, entity="e3"))
    assert recalibrator._wake.is_set()


def test_restore_rebuilds_the_calibrator_from_a_loaded_store(tmp_path: Path):
    path = tmp_path / "feedback_store.json"
    first = FeedbackProcessor(FeedbackStore(path, autosave=True), ThresholdCalibrator())
    for i in range(7):
        first.process(event(i, at=T0 + timedelta(seconds=i)))
    assert first.calibrator.get_adjustment("wh1-A-temp-01") == pytest.approx(0.1)

    second = FeedbackProcessor(FeedbackStore(path), ThresholdCalibrator())
    assert second.store.load() == 7
    assert second.restore() == 7
    assert second.calibrator.snapshot() == first.calibrator.snapshot()


def test_analytics_summarises_the_feedback():
    detector = FeedbackDriftDetector()
    processor = make_processor(drift_detector=detector)
    now = datetime.now(UTC)
    for i in range(8):
        processor.process(
            event(
                i,
                "false_positive",
                entity="noisy",
                analyst=f"a{i % 2}",
                at=now - timedelta(minutes=i),
            )
        )
    for i in range(3):
        processor.process(
            event(20 + i, "true_positive", entity="good", at=now - timedelta(hours=5))
        )
    processor.process(event(30, "true_positive", entity="gap", missed=True, at=now))
    processor.process(
        event(
            40,
            "severity_adjustment",
            corrected_severity="EMERGENCY",
            original_severity="INFO",
            at=now,
        )
    )
    processor.process(
        event(
            41,
            "severity_adjustment",
            corrected_severity="INFO",
            original_severity="CRITICAL",
            at=now,
            analyst="a9",
        )
    )

    report = processor.analytics(now)

    assert report["overall"] == {
        "total": 12,
        "false_positive": 8,
        "true_positive": 4,
        "false_positive_rate": 8 / 12,
    }
    assert report["windows"]["last_hour"]["total"] == 9  # 8 FPs + the missed TP
    assert report["windows"]["last_24h"]["total"] == 12
    assert report["missed_detections"] == 1
    assert report["severity_adjustments"] == {"total": 2, "upgrades": 1, "downgrades": 1}
    assert report["noisiest_entities"][0]["entity_id"] == "noisy"
    assert report["noisiest_entities"][0]["false_positive"] == 8
    assert "noisy" in report["adjusted_entities"]
    assert report["analysts"] == 3
    assert report["drift"]["alerts"] == 0 and report["last_recalibration"] is None
    json.dumps(report)


def test_analytics_of_an_empty_store_is_well_formed():
    report = make_processor().analytics()
    assert report["overall"]["total"] == 0 and report["overall"]["false_positive_rate"] == 0.0
    assert report["noisiest_entities"] == [] and report["drift"] is None


def test_automatic_threshold_changes_are_logged_as_warnings(caplog):
    calibrator = ThresholdCalibrator()
    with caplog.at_level(logging.INFO, logger="src.models.feedback_loop"):
        for i in range(6):
            calibrator.update("s1", True, timestamp=T0 + timedelta(seconds=i))
        for _ in range(4):
            calibrator.update("s2", False, missed=True)
    levels = {r.levelname for r in caplog.records if "entity_threshold" in r.getMessage()}
    assert levels == {"WARNING"}
    assert sum("entity_threshold_raised" in r.getMessage() for r in caplog.records) == 1
    assert sum("entity_threshold_lowered" in r.getMessage() for r in caplog.records) == 1
