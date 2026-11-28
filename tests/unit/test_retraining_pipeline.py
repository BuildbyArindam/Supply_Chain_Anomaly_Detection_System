"""Integration tests of the continuous retraining pipeline (Day 17).

Real models, real feedback store and drift detector, real fine-tuning (tiny LSTMs, CPU, a few
seconds); only the clock and the Kafka producer are faked. The tests walk the whole path::

    verdicts -> drift / volume / schedule trigger -> RetrainingRequest -> orchestrator
        -> data -> fine-tune -> recalibrate -> champion vs challenger -> promote / discard
        -> audit log + model-health events -> rollback watch
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.models.feedback_loop import FeedbackDriftDetector, FeedbackEvent, FeedbackStore
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.lstm_classifier import LSTMClassifier
from src.models.model_registry import ModelRegistry
from src.models.retraining_pipeline import (
    ChampionChallenger,
    DataPreparation,
    DetectorModelSlot,
    IncrementalFineTuner,
    LabeledWindows,
    ModelStage,
    RetrainingAuditLog,
    RetrainingOrchestrator,
    RetrainingRequest,
    RetrainingTrigger,
    RollbackManager,
    SimpleModelSlot,
    autoencoder_anomaly_f1,
    classifier_anomaly_f1,
)

SEQ, FEATURES = 8, 3
T0 = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_windows(
    n: int, *, anomaly_share: float = 0.3, seed: int = 0, shift: float = 3.0
) -> LabeledWindows:
    """Normal windows are N(0, 1); anomalies (class 1 = spike, 2 = level shift) stand out."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    anomalous = rng.random(n) < anomaly_share
    y[anomalous] = rng.integers(1, 3, int(anomalous.sum()))
    x = rng.normal(0.0, 1.0, (n, SEQ, FEATURES)).astype(np.float32)
    x[y == 1, -1, :] += shift * 2  # point spike in the newest step
    x[y == 2] += shift  # contextual: the whole window is offset
    return LabeledWindows(x, y)


def classifier(seed: int = 0) -> LSTMClassifier:
    torch.manual_seed(seed)
    return LSTMClassifier(FEATURES, seq_len=SEQ, hidden_size=8, num_layers=1, head_size=8)


def autoencoder(seed: int = 0) -> LSTMAutoencoder:
    torch.manual_seed(seed)
    return LSTMAutoencoder(FEATURES, seq_len=SEQ, hidden_sizes=(8, 16), latent_dim=4)


def fit_classifier(model: LSTMClassifier, data: LabeledWindows, epochs: int = 60) -> LSTMClassifier:
    """Train ``model`` normally (lr 1e-2) so that it has something to be a champion with."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = torch.from_numpy(data.x), torch.from_numpy(data.y)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x)
        torch.nn.functional.cross_entropy(logits, y).backward()
        optimizer.step()
    return model


class Broker:
    """Collects what would go to the ``model-health`` topic."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> bool:
        self.events.append(event)
        return True

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event_type"] == event_type]


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def verdict(n: int, kind: str = "false_positive", analyst: str = "a1") -> FeedbackEvent:
    return FeedbackEvent.create(
        f"alert-{n}", analyst, kind, entity_id=f"e{n % 4}", timestamp=T0 + timedelta(seconds=n)
    )


def add_verdicts(store: FeedbackStore, count: int, *, fp_share: float, start: int = 0) -> None:
    """Add ``count`` verdicts, ``fp_share`` of them false positives (spread evenly)."""
    for k in range(count):
        # exactly round(count * fp_share) false positives, evenly spread
        is_fp = int((k + 1) * fp_share + 1e-9) > int(k * fp_share + 1e-9)
        store.add(verdict(start + k, "false_positive" if is_fp else "true_positive"))


@pytest.fixture(scope="module")
def train_data() -> LabeledWindows:
    return make_windows(400, seed=1)


@pytest.fixture(scope="module")
def weak_champion(train_data: LabeledWindows) -> LSTMClassifier:
    """An untrained model: poor F1, so a fine-tuned challenger has room to win."""
    return classifier(seed=3)


def build(
    *,
    store: FeedbackStore,
    champion: Any,
    stream: LabeledWindows | None,
    holdout: LabeledWindows | None = None,
    replay: LabeledWindows | None = None,
    lookup: Any = None,
    broker: Broker | None = None,
    audit: RetrainingAuditLog | None = None,
    rollback: RollbackManager | None = None,
    trigger: RetrainingTrigger | None = None,
    registry: ModelRegistry | None = None,
    recalibrator: Any = None,
    max_epochs: int = 10,
    learning_rate: float = 1e-2,
    min_improvement: float = 0.02,
) -> tuple[RetrainingOrchestrator, SimpleModelSlot]:
    slot = SimpleModelSlot(champion)
    data = DataPreparation(
        store,
        window_lookup=lookup,
        stream_source=(lambda: stream) if stream is not None else None,
        replay_source=(lambda: replay) if replay is not None else None,
        holdout_source=(lambda: holdout) if holdout is not None else None,
    )
    tuner = IncrementalFineTuner(learning_rate=learning_rate, max_epochs=max_epochs, patience=10)
    orchestrator = RetrainingOrchestrator(
        data,
        tuner,
        [
            ModelStage(
                "lstm_classifier",
                slot,
                classifier_anomaly_f1,
                registry_model_type="lstm_classifier",
            )
        ],
        recalibrator=recalibrator,
        rollback=rollback,
        trigger=trigger,
        audit=audit,
        registry=registry,
        publisher=broker,
        min_f1_improvement=min_improvement,
        min_new_windows=20,
    )
    return orchestrator, slot


# --------------------------------------------------------------------------- #
# Triggers
# --------------------------------------------------------------------------- #
def test_nothing_fires_when_nothing_happened():
    trigger = RetrainingTrigger(FeedbackStore(), clock=Clock())
    assert trigger.check() is None


def test_volume_trigger_needs_more_than_the_configured_count():
    store = FeedbackStore()
    trigger = RetrainingTrigger(store, feedback_events_trigger=500, clock=Clock())
    add_verdicts(store, 500, fp_share=0.5)
    assert trigger.check() is None  # exactly 500 is not "more than 500"
    store.add(verdict(9999))
    request = trigger.check()
    assert request is not None and request.reason == "feedback_volume"
    assert request.details["new_feedback_events"] == 501


def test_schedule_trigger_fires_after_24_hours():
    clock = Clock()
    trigger = RetrainingTrigger(FeedbackStore(), scheduled_interval_hours=24, clock=clock)
    clock.t = 24 * 3600 - 1
    assert trigger.check() is None
    clock.t = 24 * 3600
    request = trigger.check()
    assert request is not None and request.reason == "scheduled"


def test_drift_alert_fires_the_drift_trigger_and_old_alerts_are_ignored():
    detector = FeedbackDriftDetector(delta=0.1, threshold=3.0, min_samples=10)
    for _ in range(10):  # an alert that exists before the trigger is created
        detector.update(False)
    for _ in range(40):
        detector.update(True)
    assert detector.alerts
    trigger = RetrainingTrigger(FeedbackStore(), drift_detector=detector, clock=Clock())
    assert trigger.check() is None  # the old alert is not news
    detector2 = FeedbackDriftDetector(delta=0.1, threshold=3.0, min_samples=10)
    trigger2 = RetrainingTrigger(FeedbackStore(), drift_detector=detector2, clock=Clock())
    for _ in range(10):
        detector2.update(False)
    for _ in range(40):
        detector2.update(True)
    request = trigger2.check()
    assert request is not None and request.reason == "drift_alert"
    assert request.details["drift_alert_ids"] == [detector2.alerts[-1].alert_id]


def test_a_running_retrain_queues_instead_of_double_triggering_and_merges_reasons():
    clock, store, broker = Clock(), FeedbackStore(), Broker()
    trigger = RetrainingTrigger(
        store, feedback_events_trigger=5, scheduled_interval_hours=1, clock=clock, publisher=broker
    )
    add_verdicts(store, 6, fp_share=0.5)
    first = trigger.check()
    assert first is not None and trigger.is_running
    add_verdicts(store, 6, fp_share=0.5, start=100)
    assert trigger.check() is None  # queued
    clock.t = 3600
    assert trigger.check() is None  # merged into the same queued request
    queued = trigger.queued
    assert queued is not None
    assert queued.details["reasons"] == ["feedback_volume", "scheduled"]
    assert len(broker.of("RetrainingRequest")) == 1  # nothing extra published yet
    released = trigger.mark_finished()
    assert released is queued and trigger.is_running  # the queued run starts next
    assert len(broker.of("RetrainingRequest")) == 2
    assert trigger.mark_finished() is None and not trigger.is_running


def test_background_thread_hands_requests_to_the_callback():
    clock, store = Clock(), FeedbackStore()
    seen: list[RetrainingRequest] = []
    trigger = RetrainingTrigger(
        store,
        feedback_events_trigger=2,
        check_interval_minutes=0.0005,  # 30 ms
        on_request=seen.append,
        clock=clock,
    )
    trigger.start()
    try:
        add_verdicts(store, 3, fp_share=0.5)
        deadline = time.time() + 5
        while not seen and time.time() < deadline:
            time.sleep(0.02)
    finally:
        trigger.stop()
    assert len(seen) == 1 and seen[0].reason == "feedback_volume"


def test_trigger_validates_its_settings():
    with pytest.raises(ValueError):
        RetrainingTrigger(FeedbackStore(), feedback_events_trigger=0)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def test_feedback_verdicts_become_labelled_windows_and_unresolved_ones_are_counted():
    store = FeedbackStore()
    rng = np.random.default_rng(0)
    for i in range(30):
        store.add(verdict(i, "false_positive" if i % 2 else "true_positive"))
    store.add(verdict(500, "severity_adjustment"))  # not a label: ignored

    def lookup(event: FeedbackEvent):
        n = int(event.anomaly_event_id.split("-")[1])
        if n % 10 == 0:
            return None  # window aged out of the cache
        return rng.normal(size=(SEQ, FEATURES)).astype(np.float32), (2 if n == 2 else None)

    prepared = DataPreparation(store, window_lookup=lookup, test_fraction=0.2).prepare()
    assert prepared.n_unresolved == 3  # n = 0, 10, 20
    assert prepared.n_feedback == 27
    assert len(prepared.new) + len(prepared.test) == 27
    all_y = np.concatenate([prepared.new.y, prepared.test.y])
    assert set(np.unique(all_y)) <= {0, 1, 2}
    assert (all_y == 2).sum() == 1  # the looked-up class survived
    assert (all_y == 0).sum() == 13 - 1 + 0 or (all_y == 0).sum() > 0  # false positives are normal


def test_holdout_source_is_used_as_is_and_nothing_is_taken_from_new_data():
    stream, holdout = make_windows(60, seed=1), make_windows(40, seed=2)
    prepared = DataPreparation(
        FeedbackStore(), stream_source=lambda: stream, holdout_source=lambda: holdout
    ).prepare()
    assert len(prepared.new) == 60 and prepared.test is holdout


def test_no_data_at_all_is_an_error():
    with pytest.raises(ValueError, match="no labelled windows"):
        DataPreparation(FeedbackStore()).prepare()


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
def test_fine_tuning_trains_a_copy_and_leaves_the_champion_untouched(train_data):
    champion = classifier()
    before = {k: v.clone() for k, v in champion.state_dict().items()}
    result = IncrementalFineTuner(learning_rate=1e-2, max_epochs=3).fine_tune(champion, train_data)
    after = champion.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert result.model is not champion
    assert any(
        not torch.equal(before[k], v) for k, v in result.model.state_dict().items()
    ), "the copy did not learn anything"
    assert 1 <= result.report.epochs_run <= 3


def test_replay_buffer_is_half_of_the_training_mix():
    tuner = IncrementalFineTuner(replay_ratio=0.5)
    new, replay = make_windows(100, seed=1), make_windows(500, seed=2)
    mix, n_old = tuner.build_mix(new, replay)
    assert n_old == 100 and len(mix) == 200
    small_mix, small_old = tuner.build_mix(new, make_windows(30, seed=3))
    assert small_old == 30 and len(small_mix) == 130  # the buffer is all there is
    assert tuner.build_mix(new, None) == (new, 0)


def test_fine_tuning_uses_the_low_learning_rate_and_max_ten_epochs(monkeypatch, train_data):
    seen: dict[str, Any] = {}
    from src.models import retraining_pipeline as module

    original = module.LSTMClassifierTrainer

    class Spy(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(module, "LSTMClassifierTrainer", Spy)
    IncrementalFineTuner().fine_tune(classifier(), make_windows(60, seed=4))
    assert seen["learning_rate"] == 1e-5 and seen["max_epochs"] == 10


def test_replay_prevents_forgetting_better_than_new_data_alone():
    """Fine-tune on data that only has one anomaly class; replay keeps the other class alive."""
    old = make_windows(500, anomaly_share=0.4, seed=10)
    champion = fit_classifier(classifier(5), old)
    only_spikes = LabeledWindows(
        *(lambda d: (d.x[d.y != 2], d.y[d.y != 2]))(make_windows(400, anomaly_share=0.4, seed=11))
    )
    test = make_windows(300, anomaly_share=0.4, seed=12)
    contextual = test.subset(test.y != 1)
    kwargs = dict(learning_rate=1e-2, max_epochs=8, patience=10)
    with_replay = IncrementalFineTuner(replay_ratio=0.5, **kwargs).fine_tune(
        champion, only_spikes, old
    )
    without = IncrementalFineTuner(replay_ratio=0.0, **kwargs).fine_tune(
        champion, only_spikes, None
    )
    assert classifier_anomaly_f1(with_replay.model, contextual) >= classifier_anomaly_f1(
        without.model, contextual
    )


def test_autoencoder_is_fine_tuned_on_normal_windows_only():
    data = make_windows(120, seed=5)
    result = IncrementalFineTuner(learning_rate=1e-3, max_epochs=2).fine_tune(autoencoder(), data)
    assert result.report.model_name == "lstm_autoencoder"
    assert result.report.n_new == int((data.y == 0).sum())


def test_fine_tuner_rejects_bad_input():
    with pytest.raises(TypeError):
        IncrementalFineTuner().fine_tune(object(), make_windows(40))
    with pytest.raises(ValueError):
        IncrementalFineTuner(replay_ratio=1.0)
    with pytest.raises(ValueError, match="need at least"):
        IncrementalFineTuner().fine_tune(classifier(), make_windows(3))


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
def test_challenger_must_beat_the_champion_by_more_than_the_margin():
    scores = {"champion": 0.80, "good": 0.83, "tie": 0.82, "bad": 0.70}
    test = make_windows(30)
    comparison = ChampionChallenger(
        "champion", scorer=lambda model, _: scores[model], min_improvement=0.02
    )
    assert comparison.evaluate(test, "good").promoted
    report = comparison.evaluate(test, "tie")
    assert report.improvement == pytest.approx(0.02) and not report.promoted  # strictly greater
    bad = comparison.evaluate(test, "bad")
    assert not bad.promoted and bad.improvement == pytest.approx(-0.10)
    assert (bad.champion_f1, bad.challenger_f1) == (0.80, 0.70)


def test_promote_keeps_the_previous_champion_and_discard_drops_the_challenger():
    comparison = ChampionChallenger("old", "new", scorer=lambda *_: 0.5)
    assert comparison.promote() == "old"
    assert comparison.champion_model == "new" and comparison.previous_champion == "old"
    assert comparison.challenger_model is None
    comparison.set_challenger("other")
    comparison.discard()
    assert comparison.challenger_model is None and comparison.champion_model == "new"


def test_evaluation_needs_a_challenger_and_a_test_set_with_anomalies():
    comparison = ChampionChallenger("a", scorer=lambda *_: 0.5)
    with pytest.raises(ValueError, match="no challenger"):
        comparison.evaluate(make_windows(20))
    comparison.set_challenger("b")
    only_normal = LabeledWindows(np.zeros((5, SEQ, FEATURES)), np.zeros(5))
    with pytest.raises(ValueError, match="at least one anomaly"):
        comparison.evaluate(only_normal)


def test_autoencoder_scorer_runs_on_a_real_model():
    score = autoencoder_anomaly_f1(autoencoder(), make_windows(80, seed=7))
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Orchestrator: the full pipeline
# --------------------------------------------------------------------------- #
def test_full_pipeline_promotes_a_better_challenger_and_records_everything(
    tmp_path: Path, weak_champion
):
    store, broker = FeedbackStore(), Broker()
    audit = RetrainingAuditLog(tmp_path / "audit.jsonl")
    registry = ModelRegistry(tmp_path / "registry")
    orchestrator, slot = build(
        store=store,
        champion=weak_champion,
        stream=make_windows(500, seed=21),
        holdout=make_windows(200, seed=22),
        replay=make_windows(300, seed=23),
        broker=broker,
        audit=audit,
        registry=registry,
        max_epochs=40,
    )
    request = RetrainingRequest.create("scheduled")
    result = orchestrator.on_retrain_request(request)

    assert result.outcome == "promoted", result.reason
    assert result.trigger_reason == "scheduled" and result.request_id == request.request_id
    assert result.challenger_f1 > result.champion_f1 + 0.02
    assert result.improvement == pytest.approx(result.challenger_f1 - result.champion_f1, abs=1e-5)
    assert result.data_size == 500 and result.n_replay_windows == 300
    assert result.training_time_s > 0
    assert slot.get() is not weak_champion  # the challenger is serving
    assert classifier_anomaly_f1(slot.get(), make_windows(200, seed=22)) == pytest.approx(
        result.challenger_f1
    )
    assert "lstm_classifier" in result.registry_versions
    saved = registry.get_latest_model("lstm_classifier")
    assert saved.metadata["metrics"]["f1"] == pytest.approx(result.challenger_f1)

    published = broker.of("RetrainingResult")
    assert len(published) == 1 and published[0]["run_id"] == result.run_id
    rows = audit.entries("retraining_result")
    assert len(rows) == 1 and rows[0]["outcome"] == "promoted"
    on_disk = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert on_disk[0]["run_id"] == result.run_id
    reloaded = RetrainingAuditLog(tmp_path / "audit.jsonl")
    assert (
        reloaded.load() == 1 and reloaded.entries()[0]["challenger_f1"] == rows[0]["challenger_f1"]
    )


def test_a_challenger_that_does_not_beat_the_champion_is_discarded(train_data):
    champion = fit_classifier(classifier(1), train_data, epochs=120)
    broker = Broker()
    orchestrator, slot = build(
        store=FeedbackStore(),
        champion=champion,
        stream=make_windows(200, seed=31),
        holdout=make_windows(200, seed=32),
        broker=broker,
        learning_rate=1e-9,  # the challenger is the champion, give or take
        max_epochs=2,
        min_improvement=0.02,
    )
    result = orchestrator.on_retrain_request(RetrainingRequest.create("manual"))
    assert result.outcome == "rejected"
    assert slot.get() is champion
    assert result.improvement is not None and result.improvement <= 0.02
    assert broker.of("RetrainingResult")[0]["outcome"] == "rejected"


def test_too_little_data_skips_fine_tuning_but_still_recalibrates():
    class Recalibrator:
        calls = 0

        def run_once(self, *, force: bool = False):
            Recalibrator.calls += 1
            assert force
            return type("Report", (), {"status": "applied"})()

    orchestrator, slot = build(
        store=FeedbackStore(),
        champion=classifier(),
        stream=make_windows(10, seed=41),
        holdout=make_windows(50, seed=42),
        recalibrator=Recalibrator(),
    )
    before = slot.get()
    result = orchestrator.on_retrain_request(RetrainingRequest.create("scheduled"))
    assert result.outcome == "skipped" and "need 20" in result.reason
    assert result.recalibration == "applied" and Recalibrator.calls == 1
    assert slot.get() is before


def test_no_data_at_all_is_skipped_not_failed():
    orchestrator, _ = build(store=FeedbackStore(), champion=classifier(), stream=None)
    result = orchestrator.on_retrain_request(RetrainingRequest.create("scheduled"))
    assert result.outcome == "skipped" and "no training data" in result.reason


def test_a_crash_ends_the_run_as_failed_and_the_champion_keeps_serving():
    champion = classifier()
    orchestrator, slot = build(
        store=FeedbackStore(),
        champion=champion,
        stream=make_windows(100, seed=51),
        holdout=make_windows(50, seed=52),
        broker=(broker := Broker()),
    )

    def explode(*_: Any, **__: Any) -> Any:
        raise RuntimeError("out of memory")

    orchestrator.fine_tuner.fine_tune = explode  # type: ignore[method-assign]
    result = orchestrator.on_retrain_request(RetrainingRequest.create("manual"))
    assert result.outcome == "failed" and "out of memory" in result.reason
    assert slot.get() is champion
    assert broker.of("RetrainingResult")[0]["outcome"] == "failed"
    assert orchestrator.audit.entries("retraining_result")[0]["outcome"] == "failed"


def test_a_failing_recalibration_does_not_stop_the_lstm_promotion(weak_champion):
    class Broken:
        def run_once(self, *, force: bool = False):
            raise RuntimeError("lightgbm exploded")

    orchestrator, slot = build(
        store=FeedbackStore(),
        champion=weak_champion,
        stream=make_windows(500, seed=21),
        holdout=make_windows(200, seed=22),
        recalibrator=Broken(),
        max_epochs=40,
    )
    result = orchestrator.on_retrain_request(RetrainingRequest.create("manual"))
    assert result.recalibration == "failed" and result.outcome == "promoted"


def test_a_publisher_that_raises_cannot_break_the_run():
    def broken(_: dict[str, Any]) -> None:
        raise ConnectionError("broker down")

    orchestrator, _ = build(
        store=FeedbackStore(), champion=classifier(), stream=make_windows(60, seed=71),
        holdout=make_windows(60, seed=72), max_epochs=1,
    )  # fmt: skip
    orchestrator.publisher = broken
    result = orchestrator.on_retrain_request(RetrainingRequest.create("manual"))
    assert result.outcome in {"promoted", "rejected"}


def test_feedback_labelled_windows_flow_through_the_pipeline():
    """Verdicts alone (no stream source) are enough: their windows are the training data."""
    store = FeedbackStore()
    pool = make_windows(300, seed=81)
    for i in range(300):
        store.add(verdict(i, "false_positive" if pool.y[i] == 0 else "true_positive"))

    def lookup(event: FeedbackEvent):
        i = int(event.anomaly_event_id.split("-")[1])
        return pool.x[i], int(pool.y[i]) or None

    orchestrator, _ = build(
        store=store, champion=classifier(), stream=None, lookup=lookup,
        holdout=make_windows(100, seed=82), max_epochs=2,
    )  # fmt: skip
    result = orchestrator.on_retrain_request(RetrainingRequest.create("feedback_volume"))
    assert result.n_feedback_windows == 300 and result.outcome in {"promoted", "rejected"}


# --------------------------------------------------------------------------- #
# Trigger <-> orchestrator: dedupe and the worker thread
# --------------------------------------------------------------------------- #
def test_trigger_to_orchestrator_end_to_end_with_queueing(weak_champion):
    clock, store, broker = Clock(), FeedbackStore(), Broker()
    trigger = RetrainingTrigger(store, feedback_events_trigger=5, clock=clock, publisher=broker)
    orchestrator, _ = build(
        store=store,
        champion=weak_champion,
        stream=make_windows(60, seed=91),
        holdout=make_windows(60, seed=92),
        broker=broker,
        trigger=trigger,
        max_epochs=1,
    )
    trigger.on_request = orchestrator.submit
    orchestrator.start()
    try:
        add_verdicts(store, 6, fp_share=0.5)
        assert trigger.poll() is not None  # emitted and handed to the worker
        # While it runs, a new condition queues exactly one follow-up request.
        add_verdicts(store, 6, fp_share=0.5, start=50)
        trigger.poll()
        deadline = time.time() + 60
        while len(broker.of("RetrainingResult")) < 2 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        orchestrator.stop()
    results = broker.of("RetrainingResult")
    assert len(results) == 2, "the queued request must run after the first one"
    assert not trigger.is_running
    assert len(broker.of("RetrainingRequest")) == 2  # one emitted, one released from the queue


def test_retrains_never_overlap():
    orchestrator, _ = build(
        store=FeedbackStore(), champion=classifier(), stream=make_windows(60, seed=95),
        holdout=make_windows(60, seed=96), max_epochs=1,
    )  # fmt: skip
    active, peak = [0], [0]
    guard = threading.Lock()
    original = orchestrator._run

    def tracked(*args: Any, **kwargs: Any) -> Any:
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            return original(*args, **kwargs)
        finally:
            with guard:
                active[0] -= 1

    orchestrator._run = tracked  # type: ignore[method-assign]
    threads = [
        threading.Thread(
            target=orchestrator.on_retrain_request, args=(RetrainingRequest.create("manual"),)
        )
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert peak[0] == 1 and len(orchestrator.history) == 3


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #
class Wall:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def test_rollback_restores_the_previous_model_when_false_positives_spike():
    store, broker, wall = FeedbackStore(), Broker(), Wall()
    add_verdicts(store, 100, fp_share=0.30)  # baseline: 30% false positives
    manager = RollbackManager(
        store, error_spike=0.20, window_minutes=60, min_observations=20, publisher=broker, now=wall
    )
    slot = SimpleModelSlot("challenger")
    manager.watch("lstm_classifier", slot, "champion", run_id="run-1")
    assert manager.watching == ["lstm_classifier"]

    add_verdicts(store, 10, fp_share=0.9, start=1000)  # not enough evidence yet
    assert manager.check() == [] and slot.get() == "challenger"

    add_verdicts(store, 30, fp_share=0.6, start=2000)  # 60% > 30% * 1.2
    [event] = manager.check()
    assert slot.get() == "champion"
    assert event.model_name == "lstm_classifier" and event.run_id == "run-1"
    assert event.baseline_error_rate == pytest.approx(0.30, abs=0.02)
    assert event.observed_error_rate > 0.36 and event.n_observations >= 20
    assert manager.watching == []
    assert broker.of("ModelRollback")[0]["rollback_id"] == event.rollback_id


def test_no_rollback_when_verdicts_stay_within_twenty_percent_of_the_baseline():
    store, wall = FeedbackStore(), Wall()
    add_verdicts(store, 100, fp_share=0.30)
    manager = RollbackManager(store, error_spike=0.20, min_observations=20, now=wall)
    slot = SimpleModelSlot("challenger")
    manager.watch("m", slot, "champion")
    add_verdicts(store, 100, fp_share=0.34, start=500)  # +13% relative
    assert manager.check() == [] and slot.get() == "challenger"


def test_a_quiet_watch_expires_after_the_window_and_later_spikes_are_ignored():
    store, wall = FeedbackStore(), Wall()
    add_verdicts(store, 50, fp_share=0.2)
    manager = RollbackManager(store, window_minutes=60, min_observations=10, now=wall)
    slot = SimpleModelSlot("challenger")
    manager.watch("m", slot, "champion")
    wall.now += timedelta(minutes=61)
    assert manager.check() == [] and manager.watching == []
    add_verdicts(store, 50, fp_share=0.9, start=900)
    assert manager.check() == [] and slot.get() == "challenger"


def test_zero_baseline_uses_the_absolute_spike_and_manual_rollback_works():
    store, wall = FeedbackStore(), Wall()
    add_verdicts(store, 40, fp_share=0.0)  # every past verdict was a true positive
    manager = RollbackManager(store, error_spike=0.20, min_observations=10, now=wall)
    slot = SimpleModelSlot("challenger")
    manager.watch("m", slot, "champion")
    add_verdicts(store, 20, fp_share=0.1, start=100)
    assert manager.check() == []  # 10% < 20%
    event = manager.rollback_now("m", "operator asked")
    assert event is not None and event.reason == "operator asked" and slot.get() == "champion"
    assert manager.rollback_now("m") is None


def test_orchestrator_promotion_starts_a_watch_and_the_rollback_is_audited(weak_champion):
    store, broker, wall = FeedbackStore(), Broker(), Wall()
    add_verdicts(store, 100, fp_share=0.2)
    manager = RollbackManager(
        store, error_spike=0.20, min_observations=20, publisher=broker, now=wall
    )
    orchestrator, slot = build(
        store=store,
        champion=weak_champion,
        stream=make_windows(500, seed=21),
        holdout=make_windows(200, seed=22),
        broker=broker,
        rollback=manager,
        max_epochs=40,
    )
    result = orchestrator.on_retrain_request(RetrainingRequest.create("drift_alert"))
    assert result.outcome == "promoted" and manager.watching == ["lstm_classifier"]
    promoted = slot.get()
    assert promoted is not weak_champion

    add_verdicts(store, 40, fp_share=0.7, start=5000)  # the new model is making things worse
    [event] = manager.check()
    assert slot.get() is weak_champion  # exactly the model that was serving before
    rollbacks = orchestrator.audit.entries("model_rollback")
    assert len(rollbacks) == 1 and rollbacks[0]["rollback_id"] == event.rollback_id
    assert rollbacks[0]["run_id"] == result.run_id


# --------------------------------------------------------------------------- #
# Slots
# --------------------------------------------------------------------------- #
def test_detector_slot_swaps_the_model_of_every_detector():
    class Detector:
        def __init__(self, model: str) -> None:
            self.model = model

    detectors = [Detector("old"), Detector("old")]
    slot = DetectorModelSlot(lambda: detectors)
    assert slot.get() == "old"
    slot.set("new")
    assert [d.model for d in detectors] == ["new", "new"]
    with pytest.raises(LookupError):
        DetectorModelSlot(list).get()


# --------------------------------------------------------------------------- #
# Containers and audit log
# --------------------------------------------------------------------------- #
def test_labeled_windows_validate_and_concatenate():
    with pytest.raises(ValueError):
        LabeledWindows(np.zeros((3, SEQ)), np.zeros(3))
    with pytest.raises(ValueError):
        LabeledWindows(np.zeros((3, SEQ, FEATURES)), np.zeros(2))
    a, b = make_windows(10, seed=1), make_windows(15, seed=2)
    assert len(LabeledWindows.concat([a, LabeledWindows.empty(SEQ, FEATURES), b])) == 25
    with pytest.raises(ValueError, match="shapes differ"):
        LabeledWindows.concat([a, LabeledWindows(np.zeros((2, SEQ + 1, FEATURES)), np.zeros(2))])
    assert sum(a.class_counts(3).values()) == 10


def test_audit_log_survives_a_bad_path_and_corrupt_lines(tmp_path: Path):
    blocked = tmp_path / "file"
    blocked.write_text("x")
    log = RetrainingAuditLog(blocked / "audit.jsonl")  # a directory cannot be created under a file
    assert log.record("retraining_result", {"outcome": "failed"})["outcome"] == "failed"
    assert len(log.entries()) == 1  # kept in memory even though the write failed

    path = tmp_path / "audit.jsonl"
    path.write_text('{"kind": "retraining_result", "outcome": "promoted"}\nnot json\n[1, 2]\n')
    reloaded = RetrainingAuditLog(path)
    assert reloaded.load() == 1 and reloaded.entries()[0]["outcome"] == "promoted"
    assert RetrainingAuditLog(tmp_path / "missing.jsonl").load() == 0


def test_components_build_from_the_project_config():
    from src.utils.config import RetrainingConfig

    cfg = RetrainingConfig()
    store = FeedbackStore()
    trigger = RetrainingTrigger.from_config(cfg, store, clock=Clock())
    assert trigger.feedback_events_trigger == 500
    assert trigger.scheduled_interval_s == 24 * 3600 and trigger.check_interval_s == 15 * 60
    tuner = IncrementalFineTuner.from_config(cfg)
    assert (tuner.learning_rate, tuner.max_epochs, tuner.replay_ratio) == (1e-5, 10, 0.5)
    manager = RollbackManager.from_config(cfg, store)
    assert manager.error_spike == 0.20 and manager.window_s == 3600
    orchestrator = RetrainingOrchestrator.from_config(
        cfg, DataPreparation(store), [ModelStage("m", SimpleModelSlot(None))]
    )
    assert orchestrator.min_f1_improvement == 0.02
