"""Unit tests for the Day 12 ensemble, severity scorer, alert event and deduplicator."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.ensemble_detector import (
    BOUNDARY,
    IMPACT_COLD_STORAGE,
    IMPACT_DEFAULT,
    IMPACT_INVENTORY,
    IMPACT_VEHICLE,
    ISOLATION_FOREST,
    LSTM_AUTOENCODER,
    LSTM_CLASSIFIER,
    MEMBERS,
    META_FEATURE_NAMES,
    STATISTICAL,
    AlertDeduplicator,
    AnomalyEvent,
    EnsembleDetector,
    EntityContext,
    MemberScoreTable,
    MetaLearner,
    Severity,
    SeverityScorer,
    binary_metrics,
    build_anomaly_event,
    holdout_mask,
    meta_feature_row,
    statistical_to_score,
    to_boundary_scale,
)
from src.models.model_registry import ModelRegistry
from src.models.statistical_detector import AnomalyResult, AnomalyScore
from src.utils.config import AlertsConfig, EnsembleConfig

T0 = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)


def make_score(score, *, kind="point", flagged=None, warming=False, raw=3.0, conf=0.8, at=T0):
    return AnomalyScore(
        entity_id="s1",
        timestamp=at,
        score=score,
        is_anomaly=score >= 0.5 if flagged is None else flagged,
        anomaly_type=kind,
        confidence=conf,
        detector="x",
        raw_value=raw,
        details={"warming_up": warming},
    )


def all_members(values, **kw):
    return {name: make_score(v, **kw) for name, v in zip(MEMBERS, values, strict=True)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_boundary_scale_is_monotone_and_pins_threshold():
    p = np.linspace(0, 1, 101)
    out = to_boundary_scale(p, 0.2)
    assert np.all(np.diff(out) > 0)
    assert out[0] == 0 and out[-1] == pytest.approx(1)
    assert to_boundary_scale(0.2, 0.2) == pytest.approx(0.5)
    assert isinstance(to_boundary_scale(0.1, 0.2), float)


def test_meta_features_nan_for_missing_and_warming():
    scores = {
        STATISTICAL: make_score(0.9),
        ISOLATION_FOREST: make_score(0.0, warming=True),
        LSTM_AUTOENCODER: None,
    }
    row = meta_feature_row(scores, 0.04, 13)
    assert row.shape == (len(META_FEATURE_NAMES),) == (10,)
    assert row[0] == 0.9 and row[4] == 0.8
    assert np.isnan(row[1]) and np.isnan(row[2]) and np.isnan(row[3])
    assert row[8] == 0.04 and row[9] == 13


def test_statistical_result_conversion():
    res = AnomalyResult("s1", T0, 0.7, True, "point", 0.6, ["zscore"], {}, raw_value=4.0)
    score = statistical_to_score(res)
    assert score.score == 0.7 and score.is_anomaly and score.anomaly_type == "point"
    assert score.details["contributing_detectors"] == ["zscore"]
    assert statistical_to_score(score) is score


# --------------------------------------------------------------------------- #
# Weighted-average fallback
# --------------------------------------------------------------------------- #
def test_fallback_weighted_average_matches_spec_weights():
    ens = EnsembleDetector()
    assert ens.weights == pytest.approx(
        {STATISTICAL: 0.2, ISOLATION_FOREST: 0.3, LSTM_AUTOENCODER: 0.25, LSTM_CLASSIFIER: 0.25}
    )
    result = ens.combine("s1", T0, all_members([1.0, 0.0, 0.0, 1.0]))
    assert result.details["method"] == "weighted_average"
    assert result.details["probability"] == pytest.approx(0.2 + 0.25)
    assert not result.is_anomaly  # 0.45 < 0.5


def test_fallback_renormalises_over_missing_and_warming_members():
    ens = EnsembleDetector()
    scores = {
        STATISTICAL: make_score(1.0),
        ISOLATION_FOREST: make_score(0.0, warming=True),
        LSTM_AUTOENCODER: None,
        LSTM_CLASSIFIER: make_score(1.0),
    }
    res = ens.combine("s1", T0, scores)
    assert res.details["probability"] == pytest.approx(1.0)
    assert res.is_anomaly and res.ensemble_score == pytest.approx(1.0)
    assert res.details["members_warming_up"] == [ISOLATION_FOREST]
    assert res.details["members_missing"] == [LSTM_AUTOENCODER]
    assert set(res.individual_scores) == {STATISTICAL, LSTM_CLASSIFIER}


def test_everything_missing_is_not_an_anomaly():
    res = EnsembleDetector().combine("s1", T0, {})
    assert res.ensemble_score == 0 and not res.is_anomaly and res.details["method"] == "none"
    warm = EnsembleDetector().combine("s1", T0, {STATISTICAL: make_score(0, warming=True)})
    assert warm.details["warming_up"] is True


def test_type_vote_prefers_flagged_weighted_members():
    ens = EnsembleDetector()
    scores = {
        STATISTICAL: make_score(0.9, kind="point"),
        ISOLATION_FOREST: make_score(0.8, kind=None),
        LSTM_AUTOENCODER: make_score(0.95, kind="collective"),
        LSTM_CLASSIFIER: make_score(0.9, kind="point"),
    }
    res = ens.combine("s1", T0, scores)
    assert res.is_anomaly and res.predicted_anomaly_type == "point"  # 0.2*.9+0.25*.9 > 0.25*.95
    scores[LSTM_CLASSIFIER] = make_score(0.1, kind="contextual", flagged=False)
    scores[STATISTICAL] = make_score(0.95, kind="collective")
    assert ens.combine("s1", T0, scores).predicted_anomaly_type == "collective"


def test_normal_event_has_no_type_and_confidence_is_symmetric():
    res = EnsembleDetector().combine("s1", T0, all_members([0.0] * 4))
    assert res.predicted_anomaly_type is None
    assert res.confidence == pytest.approx(1.0)
    assert res.raw_value == 3.0


def test_validation_of_constructor_arguments():
    with pytest.raises(ValueError):
        EnsembleDetector(weights={"nope": 1.0})
    with pytest.raises(ValueError):
        EnsembleDetector(weights=dict.fromkeys(MEMBERS, 0.0))
    with pytest.raises(ValueError):
        EnsembleDetector(fallback_threshold=1.5)
    with pytest.raises(ValueError):
        EnsembleDetector(min_members_for_meta=9)


def test_from_config_uses_config_weights():
    cfg = EnsembleConfig()
    ens = EnsembleDetector.from_config(cfg)
    assert ens.weights[ISOLATION_FOREST] == pytest.approx(cfg.isolation_forest_weight)
    assert ens.cv_folds == cfg.cv_folds


def test_config_defaults_match_the_day_12_specification():
    cfg = EnsembleConfig()
    assert (
        cfg.statistical_weight,
        cfg.isolation_forest_weight,
        cfg.lstm_autoencoder_weight,
        cfg.lstm_classifier_weight,
    ) == pytest.approx((0.2, 0.3, 0.25, 0.25))


# --------------------------------------------------------------------------- #
# Meta-learner
# --------------------------------------------------------------------------- #
def synthetic_table(n=900, seed=0, rate=0.1):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < rate
    entities = np.array([f"e{i % 12}" for i in range(n)], dtype=object)
    scores = np.clip(rng.normal(0.15, 0.1, (n, 4)), 0, 1)
    # member 2 is the only informative one, member 0 is noise, member 3 informative but weaker
    scores[y, 2] = np.clip(rng.normal(0.85, 0.1, y.sum()), 0, 1)
    scores[y, 3] = np.clip(rng.normal(0.6, 0.2, y.sum()), 0, 1)
    scores[:, 0] = np.clip(rng.normal(0.3, 0.2, n), 0, 1)
    confidence = np.abs(2 * scores - 1)
    return MemberScoreTable(
        entity_ids=entities,
        timestamps=np.arange(n, dtype=float) * 10,
        scores=scores,
        confidence=confidence,
        flagged=scores >= 0.5,
        types=np.full((n, 4), "point", dtype=object),
        anomaly_rate=np.full(n, 0.03),
        hour=np.full(n, 3.0),
        y=y,
        anomaly_type=np.where(y, "point", None).astype(object),
    )


def test_meta_learner_cv_fit_and_threshold():
    table = synthetic_table()
    learner = MetaLearner(n_estimators=60)
    report = learner.fit(table.meta_features(), table.y, groups=table.entity_ids, cv_folds=5)
    assert report.n_folds == 5 and report.grouped
    assert len(report.fold_f1) == 5
    assert report.f1 > 0.8 and report.roc_auc > 0.95
    assert 0.01 <= learner.threshold <= 0.99
    p = learner.predict_proba(table.meta_features())
    assert p.shape == (len(table),) and p[table.y].mean() > p[~table.y].mean() + 0.4
    importance = learner.feature_importance()
    assert importance["lstm_autoencoder_score"] > importance["statistical_score"]
    assert sum(importance.values()) == pytest.approx(1.0)


def test_meta_learner_errors_and_monotone_constraint():
    learner = MetaLearner()
    with pytest.raises(RuntimeError):
        learner.predict_proba(np.zeros((1, 10)))
    with pytest.raises(ValueError):
        learner.fit(np.zeros((10, 3)), np.zeros(10, dtype=bool))
    with pytest.raises(ValueError):
        learner.fit(np.zeros((10, 10)), np.zeros(10, dtype=bool))  # no anomalies
    table = synthetic_table(600, seed=1)
    learner = MetaLearner(n_estimators=40)
    learner.fit(table.meta_features(), table.y, groups=table.entity_ids, cv_folds=3)
    grid = np.tile(table.meta_features()[0], (21, 1))
    grid[:, 2] = np.linspace(0, 1, 21)
    assert np.all(np.diff(learner.predict_proba(grid)) >= -1e-9)


def test_meta_learner_handles_missing_members_and_save_load(tmp_path):
    table = synthetic_table()
    ens = EnsembleDetector(cv_folds=3)
    report = ens.fit_meta_learner(table)
    assert ens.uses_meta_learner and report.n_folds == 3
    # drop the informative detector: still finite, bounded scores
    blanked = table.subset(np.arange(50))
    scores = blanked.scores.copy()
    scores[:, 2] = np.nan
    final, prob, used = ens.score_arrays(
        scores, blanked.confidence, blanked.anomaly_rate, blanked.hour
    )
    assert np.all(np.isfinite(final)) and used.all() and prob.shape == (50,)
    path = tmp_path / "meta.joblib"
    ens.meta_learner.save(path)
    loaded = MetaLearner.load(path)
    np.testing.assert_allclose(
        loaded.predict_proba(table.meta_features()),
        ens.meta_learner.predict_proba(table.meta_features()),
    )
    registry = ModelRegistry(tmp_path / "registry")
    version = ens.save_meta_learner(registry, metrics={"note": 1})
    other = EnsembleDetector()
    other.load_meta_learner(registry, version.version)
    assert other.uses_meta_learner and other.meta_learner.threshold == ens.meta_learner.threshold


def test_stacked_score_uses_boundary_scale_and_beats_fallback():
    table = synthetic_table(1500, seed=3)
    ens = EnsembleDetector(cv_folds=4)
    before = binary_metrics(table.y, ens.score_table(table))
    ens.fit_meta_learner(table)
    scores = ens.score_table(table)
    after = binary_metrics(table.y, scores)
    assert after["f1"] > before["f1"] and after["roc_auc"] >= before["roc_auc"]
    assert (
        (scores >= BOUNDARY)
        == (ens.meta_learner.predict_proba(table.meta_features()) >= ens.meta_learner.threshold)
    ).all()
    types = ens.predict_types(table)
    assert (
        set(types[scores >= BOUNDARY]) <= {"point"}
        and (types[scores < BOUNDARY] == None).all()  # noqa: E711
    )


def test_too_few_members_falls_back_to_weighted_average():
    table = synthetic_table()
    ens = EnsembleDetector(cv_folds=3, min_members_for_meta=2)
    ens.fit_meta_learner(table)
    res = ens.combine("s1", T0, {LSTM_AUTOENCODER: make_score(0.9)})
    assert res.details["method"] == "weighted_average"
    res2 = ens.combine("s1", T0, all_members([0.9] * 4), anomaly_rate=0.03)
    assert res2.details["method"] == "meta_learner"


def test_holdout_mask_keeps_blocks_together():
    ts = np.arange(0, 7200, 1.0).repeat(2)
    ent = np.tile(np.array(["a", "b"], dtype=object), 7200)
    mask = holdout_mask(ts, ent, 0.3, block_s=600, seed=1)
    for entity in ("a", "b"):
        for block in range(12):
            sel = (ent == entity) & (ts // 600 == block)
            assert mask[sel].all() or not mask[sel].any()
    assert 0.05 < mask.mean() < 0.6
    with pytest.raises(ValueError):
        holdout_mask(ts, ent, 1.5)


# --------------------------------------------------------------------------- #
# detect() with stub members
# --------------------------------------------------------------------------- #
class Stat:
    def detect_event(self, event):
        return AnomalyResult(event["sensor_id"], T0, 0.9, True, "point", 0.7, ["zscore"], {}, 5.0)


class Forest:
    def __init__(self, fail=False):
        self.fail = fail
        self.learn_flags = []

    def process(self, vector, *, learn=True):
        if self.fail:
            raise RuntimeError("boom")
        self.learn_flags.append(learn)
        return make_score(0.8, kind=None, at=vector.timestamp)

    def process_many(self, vectors, *, learn=True):
        return [self.process(v, learn=learn) for v in vectors]


class Seq:
    def __init__(self, score, kind):
        self.score, self.kind = score, kind

    def detect(self, entity_id):
        return make_score(self.score, kind=self.kind)

    def detect_many(self, ids):
        return [self.detect(i) for i in ids]


def vector(rate=0.07):
    names = ("a", "anomaly_rate_7d")
    return SimpleNamespace(
        entity_id="s1",
        timestamp=T0,
        names=names,
        raw=np.array([1.0, rate]),
        source_type="warehouse",
    )


def test_detect_runs_members_and_reads_rate_from_vector():
    forest = Forest()
    ens = EnsembleDetector(Stat(), forest, Seq(0.9, "collective"), Seq(0.85, "point"))
    res = ens.detect("s1", event={"sensor_id": "s1", "timestamp": T0}, vector=vector(), learn=False)
    assert res.is_anomaly and res.predicted_anomaly_type == "point"
    assert set(res.individual_scores) == set(MEMBERS)
    assert forest.learn_flags == [False]
    assert res.timestamp == T0 and res.raw_value == 5.0


def test_failing_member_is_isolated_and_counted():
    ens = EnsembleDetector(Stat(), Forest(fail=True), Seq(0.9, "collective"), None)
    res = ens.detect("s1", event={"sensor_id": "s1"}, vector=vector())
    assert ISOLATION_FOREST in res.details["members_missing"]
    assert ens.member_failures[ISOLATION_FOREST] == 1
    assert set(res.individual_scores) == {STATISTICAL, LSTM_AUTOENCODER}


def test_detect_many_matches_detect():
    ens = EnsembleDetector(Stat(), Forest(), Seq(0.9, "collective"), Seq(0.85, "point"))
    many = ens.detect_many(
        ["s1", "s2"],
        events=[{"sensor_id": "s1"}, None],
        vectors=[vector(), vector(0.01)],
    )
    one = ens.detect("s1", event={"sensor_id": "s1"}, vector=vector())
    assert many[0].ensemble_score == pytest.approx(one.ensemble_score)
    assert STATISTICAL not in many[1].individual_scores
    with pytest.raises(ValueError):
        ens.detect_many(["s1"], events=[None, None])


def test_detect_with_nothing_available_is_zero():
    res = EnsembleDetector().detect("s1", timestamp=T0)
    assert res.ensemble_score == 0 and not res.is_anomaly


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
def ensemble_result(value, kind="point", at=T0):
    return EnsembleDetector().combine("s1", at, all_members([value] * 4, kind=kind, at=at))


@pytest.mark.parametrize(
    ("value", "level"),
    [
        (0.29, Severity.INFO),
        (0.3, Severity.WARNING),
        (0.59, Severity.WARNING),
        (0.6, Severity.CRITICAL),
        (0.84, Severity.CRITICAL),
        (0.85, Severity.EMERGENCY),
        (1.0, Severity.EMERGENCY),
    ],
)
def test_all_four_levels_at_the_boundaries(value, level):
    night = T0.replace(hour=2)
    ctx = EntityContext(impact_key=IMPACT_COLD_STORAGE)  # impact 1.0, never amplified
    assert SeverityScorer().score(value, ctx, timestamp=night).level == level


def test_impact_weights_scale_severity():
    night = T0.replace(hour=2)
    scorer = SeverityScorer()
    got = {
        key: scorer.score(0.9, EntityContext(impact_key=key), timestamp=night).score
        for key in (IMPACT_COLD_STORAGE, IMPACT_VEHICLE, IMPACT_INVENTORY, IMPACT_DEFAULT)
    }
    assert got == pytest.approx(
        {
            IMPACT_COLD_STORAGE: 0.9,
            IMPACT_VEHICLE: 0.72,
            IMPACT_INVENTORY: 0.63,
            IMPACT_DEFAULT: 0.45,
        }
    )
    result = scorer.score(0.9, EntityContext(impact_key=IMPACT_VEHICLE), timestamp=night)
    assert result.business_impact == 0.8 and result.level == Severity.CRITICAL


def test_business_hours_amplify_only_time_sensitive_classes():
    scorer = SeverityScorer()
    day, night = T0.replace(hour=11), T0.replace(hour=2)
    veh = EntityContext(impact_key=IMPACT_VEHICLE)
    assert scorer.score(0.6, veh, timestamp=day).score == pytest.approx(0.6 * 0.8 * 1.25)
    assert scorer.score(0.6, veh, timestamp=night).score == pytest.approx(0.6 * 0.8)
    cold = EntityContext(impact_key=IMPACT_COLD_STORAGE)
    assert (
        scorer.score(0.6, cold, timestamp=day).score
        == scorer.score(0.6, cold, timestamp=night).score
    )


def test_asset_value_raises_score_and_is_capped():
    scorer = SeverityScorer()
    night = T0.replace(hour=2)
    base = scorer.score(0.5, EntityContext(impact_key=IMPACT_INVENTORY), timestamp=night).score
    rich = scorer.score(
        0.5, EntityContext(impact_key=IMPACT_INVENTORY, asset_value=5_000), timestamp=night
    ).score
    huge = scorer.score(
        0.5, EntityContext(impact_key=IMPACT_INVENTORY, asset_value=1e9), timestamp=night
    ).score
    assert base < rich < huge == pytest.approx(base * 1.2)
    assert (
        scorer.score(1.0, EntityContext(impact_key=IMPACT_COLD_STORAGE, asset_value=1e9)).score
        == 1.0
    )


def test_actions_depend_on_class_type_and_level():
    scorer = SeverityScorer()
    emergency = scorer.score(
        ensemble_result(1.0, "point"), EntityContext(impact_key=IMPACT_COLD_STORAGE)
    )
    assert emergency.level == Severity.EMERGENCY
    assert emergency.recommended_action.startswith("IMMEDIATE ACTION: ")
    assert "compressor" in emergency.recommended_action
    drift = scorer.score(
        0.6, EntityContext(impact_key=IMPACT_COLD_STORAGE), anomaly_type="collective"
    )
    assert "drifting" in drift.recommended_action
    unknown = scorer.score(0.4, None, anomaly_type="weird")
    assert unknown.level == Severity.INFO and unknown.recommended_action.startswith("Low priority")


def test_from_config_and_validation():
    scorer = SeverityScorer.from_config(AlertsConfig())
    assert scorer.thresholds == (0.3, 0.6, 0.85)
    assert scorer.impact_weights["cold_storage_temp"] == 1.0
    with pytest.raises(ValueError):
        SeverityScorer(thresholds=(0.6, 0.3, 0.85))
    with pytest.raises(ValueError):
        SeverityScorer(impact_weights={"vehicle_deviation": 0.8})
    with pytest.raises(ValueError):
        SeverityScorer(impact_weights={"default": 1.5})


def test_severity_escalation_is_capped():
    assert Severity.INFO.escalated() == Severity.WARNING
    assert Severity.CRITICAL.escalated() == Severity.EMERGENCY
    assert Severity.EMERGENCY.escalated(3) == Severity.EMERGENCY


def test_entity_context_from_events():
    wh = {
        "sensor_id": "wh1-A-temp-01",
        "reading_type": "temperature",
        "value": 9.0,
        "unit": "C",
        "metadata": {
            "zone": "A",
            "storage_type": "cold_storage",
            "acceptable_range": [2.0, 4.0],
            "is_anomaly": True,
        },
    }
    ctx = EntityContext.from_event(wh)
    assert (ctx.entity_type, ctx.impact_key, ctx.expected_range, ctx.raw_value) == (
        "warehouse_sensor",
        IMPACT_COLD_STORAGE,
        (2.0, 4.0),
        9.0,
    )
    wh["metadata"]["storage_type"] = "dry_goods"
    assert EntityContext.from_event(wh).impact_key == IMPACT_DEFAULT
    veh = EntityContext.from_event(
        {"vehicle_id": "v1", "speed_kmh": 50.0, "metadata": {"cruise_speed_kmh": 60.0}}
    )
    assert veh.impact_key == IMPACT_VEHICLE and veh.expected_range == pytest.approx((36.0, 72.0))
    inv = EntityContext.from_event(
        {"sku_id": "k", "new_level": 40, "metadata": {"min_stock": 50, "max_stock": 1000}}
    )
    assert inv.impact_key == IMPACT_INVENTORY and inv.expected_range == (50.0, 1000.0)
    with pytest.raises(ValueError):
        EntityContext.from_event({"x": 1})


# --------------------------------------------------------------------------- #
# AnomalyEvent
# --------------------------------------------------------------------------- #
def make_event(at=T0, level=Severity.WARNING, entity="s1"):
    return AnomalyEvent(
        event_id=f"id-{at.timestamp()}-{level.name}",
        entity_id=entity,
        entity_type="warehouse_sensor",
        timestamp=at,
        anomaly_type="point",
        severity=level,
        raw_value=9.0,
        expected_range=(2.0, 4.0),
        top_contributing_features=[],
        recommended_action="check it",
    )


def test_build_event_and_json_roundtrip():
    class Explainer:
        def get_top_features(self, raw, n=5):
            return [SimpleNamespace(feature="value__last", shap_value=-0.4, feature_value=9.0)]

    res = ensemble_result(0.95)
    ctx = EntityContext(
        entity_type="warehouse_sensor", impact_key=IMPACT_COLD_STORAGE, expected_range=(2.0, 4.0)
    )
    sev = SeverityScorer().score(res, ctx)
    event = build_anomaly_event(res, sev, ctx, explainer=Explainer(), vector=vector())
    assert event.severity == Severity.EMERGENCY and event.expected_range == (2.0, 4.0)
    assert event.top_contributing_features[0]["feature"] == "value__last"
    assert not event.is_resolved and not event.false_positive_marked
    assert len(event.event_id) == 36
    data = json.loads(json.dumps(event.to_dict()))
    assert data["severity"] == "EMERGENCY"
    back = AnomalyEvent.from_dict(data)
    assert back == event


def test_explainer_failure_does_not_block_the_event():
    class Broken:
        def get_top_features(self, raw, n=5):
            raise ValueError("no model")

    res = ensemble_result(0.95)
    event = build_anomaly_event(
        res, SeverityScorer().score(res), explainer=Broken(), vector=vector()
    )
    assert event.top_contributing_features == []


# --------------------------------------------------------------------------- #
# Deduplication and escalation
# --------------------------------------------------------------------------- #
def minutes(m):
    return T0 + timedelta(minutes=m)


def test_duplicates_within_window_are_suppressed_and_counted():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=0)
    first = make_event(T0)
    assert dedup.process(first) is first
    for sec in (1, 30, 120, 299):
        assert dedup.process(make_event(T0 + timedelta(seconds=sec))) is None
    assert first.occurrence_count == 5 and first.last_seen == T0 + timedelta(seconds=299)
    assert dedup.stats()["suppressed"] == 4


def test_after_the_window_a_reminder_is_emitted():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=0)
    dedup.process(make_event(T0))
    dedup.process(make_event(minutes(4)))
    again = make_event(minutes(5))
    assert dedup.process(again) is again
    assert dedup.process(make_event(minutes(6))) is None


def test_other_entities_are_independent():
    dedup = AlertDeduplicator()
    assert dedup.process(make_event(T0, entity="a")) is not None
    assert dedup.process(make_event(T0, entity="b")) is not None
    assert dedup.process(make_event(T0, entity="a")) is None


def test_worsening_inside_the_window_is_not_hidden():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=0)
    dedup.process(make_event(T0, Severity.WARNING))
    worse = make_event(minutes(1), Severity.EMERGENCY)
    assert dedup.process(worse) is worse
    assert dedup.process(make_event(minutes(2), Severity.CRITICAL)) is None
    hidden = AlertDeduplicator(window_s=300, escalation_after_s=0, allow_worsening=False)
    hidden.process(make_event(T0, Severity.WARNING))
    assert hidden.process(make_event(minutes(1), Severity.EMERGENCY)) is None


def test_persistent_anomaly_escalates_after_ten_minutes():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=600)
    emitted = []
    for sec in range(0, 25 * 60 + 1, 30):  # an anomaly every 30 s for 25 minutes
        out = dedup.process(make_event(T0 + timedelta(seconds=sec), Severity.WARNING))
        if out is not None:
            emitted.append((sec // 60, out.severity, out.escalated_from))
    kinds = [(m, s.name, f.name if f else None) for m, s, f in emitted]
    assert kinds[0] == (0, "WARNING", None)
    assert (5, "WARNING", None) in kinds  # reminder after the window
    assert (10, "CRITICAL", "WARNING") in kinds  # first escalation at 10 min
    assert (20, "EMERGENCY", "CRITICAL") in kinds  # second at 20 min
    assert dedup.stats()["escalated"] == 2
    assert len({e[0] for e in emitted}) == len(emitted)  # at most one alert per minute


def test_escalation_copy_has_new_id_and_stops_at_emergency():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=600)
    base = make_event(T0, Severity.CRITICAL)
    dedup.process(base)
    dedup.process(make_event(minutes(5), Severity.CRITICAL))
    up = dedup.process(make_event(minutes(10), Severity.CRITICAL))
    assert up is not None and up.event_id != base.event_id and up.severity == Severity.EMERGENCY
    assert up.recommended_action.startswith("ESCALATED after 10 min")
    assert (
        dedup.process(make_event(minutes(20), Severity.CRITICAL)) is not None
    )  # reminder, not a 2nd escalation
    assert dedup.stats()["escalated"] == 1


def test_episode_ends_after_silence_and_restarts_cleanly():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=600)
    dedup.process(make_event(T0))
    fresh = make_event(minutes(30))
    assert dedup.process(fresh) is fresh and fresh.occurrence_count == 1
    assert dedup.stats()["episodes"] == 2


def test_late_events_never_reopen_the_window():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=0)
    dedup.process(make_event(T0))
    assert dedup.process(make_event(T0 + timedelta(seconds=200))) is None
    assert dedup.process(make_event(T0 + timedelta(seconds=180))) is None  # 20 s late
    assert dedup.process(make_event(T0 + timedelta(seconds=301))) is not None


def test_resolve_purge_and_eviction():
    dedup = AlertDeduplicator(window_s=300, escalation_after_s=600, max_entities=3)
    ev = make_event(T0, entity="a")
    dedup.process(ev)
    assert dedup.resolve("a") and ev.is_resolved and not dedup.resolve("a")
    assert dedup.process(make_event(T0, entity="a")) is not None  # fresh episode
    for name in ("b", "c", "d"):
        dedup.process(make_event(T0, entity=name))
    assert "a" not in dedup.active_entities() and len(dedup.active_entities()) == 3
    dedup.process(make_event(minutes(60), entity="b"))
    assert dedup.purge() == 2  # c, d dropped out silently
    assert dedup.active_entities() == ["b"]


def test_zero_window_disables_suppression_and_from_config():
    dedup = AlertDeduplicator(window_s=0, escalation_after_s=0)
    assert all(dedup.process(make_event(T0 + timedelta(seconds=i))) is not None for i in range(3))
    cfg = AlertsConfig()
    d = AlertDeduplicator.from_config(cfg)
    assert (d.window_s, d.escalation_after_s) == (300, 600)
    with pytest.raises(ValueError):
        AlertDeduplicator(window_s=-1)
