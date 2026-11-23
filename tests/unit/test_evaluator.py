"""Unit tests for the Day 13 evaluation framework."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.models.evaluator import (
    AnomalyDetectionEvaluator,
    DetectionResults,
    EvaluationReporter,
    ModelLeaderboard,
)

EV = AnomalyDetectionEvaluator()


def toy():
    """One entity, 12 rows, two events: rows 2-4 (point) and 8-9 (collective)."""
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    types = np.array([None, None, "point", "point", "point", None, None, None,
                      "collective", "collective", None, None], dtype=object)  # fmt: skip
    s = np.array([0.1, 0.2, 0.1, 0.9, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
    ts = np.arange(12, dtype=float) * 60
    return y, s, types, ts


def test_confusion_and_basic_metrics():
    y, s, types, ts = toy()
    r = EV.compute_metrics(y, s, 0.5, anomaly_types=types, timestamps=ts)
    assert (r.tp, r.fp, r.fn, r.tn) == (1, 1, 4, 6)
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(0.2)
    assert r.f1 == pytest.approx(f1_score(y, s >= 0.5))
    assert r.false_positive_rate == pytest.approx(1 / 7)
    assert r.false_negative_rate == pytest.approx(0.8)
    assert r.roc_auc == pytest.approx(roc_auc_score(y, s))
    assert r.average_precision == pytest.approx(average_precision_score(y, s))
    assert r.auc_pr == r.average_precision


def test_events_mttd_and_point_adjustment():
    y, s, types, ts = toy()
    r = EV.compute_metrics(y, s, 0.5, anomaly_types=types, timestamps=ts)
    assert r.n_events == 2 and r.n_events_detected == 1
    assert r.event_recall == pytest.approx(0.5)
    assert r.mttd == pytest.approx(1.0)  # event 1 starts at row 2, first flag at row 3
    assert r.mttd_seconds == pytest.approx(60.0)
    # point adjustment: event 1 (3 rows) counts fully, event 2 stays missed
    assert r.pa_recall == pytest.approx(3 / 5)
    assert r.pa_precision == pytest.approx(3 / 4)
    assert r.pa_f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert r.pa_f1 > r.f1


def test_per_type_macro_and_weighted():
    y, s, types, ts = toy()
    r = EV.compute_metrics(y, s, 0.5, anomaly_types=types, timestamps=ts)
    assert set(r.per_type) == {"point", "collective"}
    assert r.per_type["point"].recall == pytest.approx(1 / 3)
    assert r.per_type["collective"].recall == 0.0
    assert r.per_type["collective"].events_detected == 0
    assert math.isnan(r.per_type["collective"].mttd)
    assert r.macro_recall == pytest.approx((1 / 3 + 0) / 2)
    assert r.weighted_recall == pytest.approx((3 * (1 / 3) + 0) / 5)
    assert r.weighted_f1 == pytest.approx(
        (3 * r.per_type["point"].f1 + 2 * r.per_type["collective"].f1) / 5
    )


def test_typed_predictions_must_match():
    y, s, types, ts = toy()
    pred = np.array(["point"] * 12, dtype=object)
    loose = EV.compute_metrics(y, s, 0.0, anomaly_types=types)
    strict = EV.compute_metrics(y, s, 0.0, anomaly_types=types, pred_types=pred)
    assert loose.per_type["collective"].recall == 1.0
    assert strict.per_type["collective"].recall == 0.0
    assert strict.per_type["point"].recall == 1.0


def test_without_types_macro_equals_binary():
    y, s, _, _ = toy()
    r = EV.compute_metrics(y, s, 0.5)
    assert set(r.per_type) == {"anomaly"}
    assert r.macro_f1 == pytest.approx(r.f1) and r.weighted_f1 == pytest.approx(r.f1)


def test_events_do_not_span_entities_and_rows_are_ordered():
    # two adjacent anomalous rows belong to different sensors -> two events
    y = np.array([1, 1], dtype=bool)
    r = EV.compute_metrics(y, [0.9, 0.1], entity_ids=["a", "b"])
    assert r.n_events == 2 and r.n_events_detected == 1
    # shuffled input is re-ordered by timestamp
    y, s, types, ts = toy()
    perm = np.random.default_rng(0).permutation(12)
    r2 = EV.compute_metrics(y[perm], s[perm], 0.5, anomaly_types=types[perm], timestamps=ts[perm])
    assert r2.n_events == 2 and r2.mttd == pytest.approx(1.0)


def test_max_gap_splits_events():
    y = np.array([1, 1, 1, 1], dtype=bool)
    ts = np.array([0, 60, 10_000, 10_060.0])
    whole = AnomalyDetectionEvaluator().compute_metrics(y, [0.9] * 4, timestamps=ts)
    split = AnomalyDetectionEvaluator(max_gap_s=600).compute_metrics(y, [0.9] * 4, timestamps=ts)
    assert whole.n_events == 1 and split.n_events == 2


def test_nan_scores_count_as_zero():
    y, s, _, _ = toy()
    s2 = s.copy()
    s2[3] = np.nan
    r = EV.compute_metrics(y, s2, 0.5)
    assert r.n_missing_scores == 1 and r.tp == 0


def test_degenerate_inputs():
    none = EV.compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(none.roc_auc) and math.isnan(none.event_recall) and none.f1 == 0.0
    assert none.n_events == 0
    allpos = EV.compute_metrics([1, 1], [0.9, 0.8])
    assert allpos.recall == 1.0 and math.isnan(allpos.roc_auc)
    for bad in ([],):
        with pytest.raises(ValueError, match="empty"):
            EV.compute_metrics(bad, bad)
    with pytest.raises(ValueError, match="rows"):
        EV.compute_metrics([0, 1], [0.1])
    with pytest.raises(ValueError, match="infinite"):
        EV.compute_metrics([0, 1], [0.1, np.inf])
    with pytest.raises(ValueError):
        EV.compute_metrics([0, 1], [0.1, 0.2], entity_ids=["a"])


def test_to_dict_is_json_safe():
    import json

    r = EV.compute_metrics([0, 0, 1], [0.1, 0.2, 0.3])
    text = json.dumps(r.to_dict())  # NaN would raise with allow_nan=False
    json.loads(text)
    json.dumps(r.to_dict(), allow_nan=False)


def test_sweep_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 400
    ent = rng.integers(0, 4, n).astype(str)
    ts = np.tile(np.arange(100.0), 4)[:n]
    y = np.zeros(n, dtype=bool)
    for start in rng.integers(0, n - 8, 6):
        y[start : start + rng.integers(2, 7)] = True
    s = np.clip(0.55 * y + rng.normal(0.25, 0.2, n), 0, 1)
    sweep = EV.sweep_thresholds(y, s, 50, entity_ids=ent, timestamps=ts)
    assert len(sweep.thresholds) == 50 and sweep.thresholds[0] == 0 and sweep.thresholds[-1] == 1
    for i in (0, 7, 25, 40, 49):
        t = sweep.thresholds[i]
        m = EV.compute_metrics(y, s, t, entity_ids=ent, timestamps=ts)
        assert sweep.f1[i] == pytest.approx(m.f1)
        assert sweep.precision[i] == pytest.approx(m.precision)
        assert sweep.recall[i] == pytest.approx(m.recall)
        assert sweep.false_positive_rate[i] == pytest.approx(m.false_positive_rate)
        assert sweep.pa_f1[i] == pytest.approx(m.pa_f1)
    best = int(np.argmax(sweep.f1))
    assert sweep.optimal_f1 == pytest.approx(sweep.f1[best])
    assert EV.compute_metrics(y, s, sweep.optimal_threshold).f1 == pytest.approx(sweep.optimal_f1)
    assert sweep.at(0.5)["threshold"] == pytest.approx(
        sweep.thresholds[np.argmin(abs(sweep.thresholds - 0.5))]
    )
    assert len(sweep.to_frame()) == 50


def test_sweep_prefers_highest_tied_threshold_and_pa_objective():
    y, s, _, _ = toy()
    sweep = EV.sweep_thresholds(y, np.where(y, 0.8, 0.2), 11)
    assert sweep.optimal_f1 == pytest.approx(1.0)
    assert sweep.optimal_threshold == pytest.approx(0.8)
    pa = EV.sweep_thresholds(*toy()[:2], 21, optimize="pa_f1")
    assert pa.optimize == "pa_f1" and pa.optimal_f1 >= pa.f1.max() - 1e-9


def test_sweep_edge_cases():
    nothing = EV.sweep_thresholds([0, 0, 0], [0.1, 0.2, 0.3], 5)
    assert math.isnan(nothing.optimal_threshold) and nothing.optimal_f1 == 0.0
    with pytest.raises(ValueError):
        EV.sweep_thresholds([0, 1], [0.1, 0.2], 1)
    with pytest.raises(ValueError):
        EV.sweep_thresholds([0, 1], [0.1, 0.2], 5, optimize="bogus")
    wide = EV.sweep_thresholds([0, 1, 0, 1], [-3.0, 4.0, 0.0, 2.0], 5)
    assert wide.thresholds[0] == -3.0 and wide.thresholds[-1] == 4.0


def make_results():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    ent = np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c", "c"])
    s = np.array([0.9, 0.2, 0.1, 0.7, 0.1, 0.2, 0.1, 0.1, 0.8, 0.1])
    ts = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1.0])
    types = np.array(
        ["point", "point", None, None, None, None, None, None, "point", None], dtype=object
    )
    return DetectionResults(y, s, ent, ts, types)


def test_per_entity_metrics():
    frame = EV.compute_per_entity_metrics(make_results())
    assert list(frame.index) == ["a", "b", "c"]
    a, b, c = frame.loc["a"], frame.loc["b"], frame.loc["c"]
    assert (a.n, a.n_anomalies, a.n_flagged) == (4, 2, 2)
    assert a.precision == pytest.approx(0.5) and a.recall == pytest.approx(0.5)
    assert a.fpr == pytest.approx(0.5)
    assert (
        b.n_anomalies == 0 and math.isnan(b.recall) and math.isnan(b.f1) and math.isnan(b.precision)
    )
    assert b.fpr == 0.0
    assert c.f1 == pytest.approx(1.0) and c.events_detected == 1
    with pytest.raises(ValueError, match="entity_ids"):
        EV.compute_per_entity_metrics(
            DetectionResults(np.array([0, 1], bool), np.array([0.1, 0.2]))
        )


def make_board():
    y, s, types, ts = toy()
    board = ModelLeaderboard()
    good = np.where(y, 0.9, 0.1)
    for name, scores, latency in (
        ("weak", s, 1.0),
        ("strong", good, 9.0),
        ("mid", np.where(y, 0.6, 0.45) * (s + 0.5), None),
    ):
        res = DetectionResults(y, scores, np.zeros(12, int), ts, types)
        report, sweep = EV.evaluate(res, n_thresholds=21)
        board.register_model(name, report, sweep=sweep, latency_ms=latency)
    return board


def test_leaderboard_ranking_and_recommend():
    board = make_board()
    table = board.generate_comparison_table()
    assert table["model"].iloc[0] == "strong" == board.best_model()
    assert table.index.name == "rank" and table.index[0] == 1
    assert table["weighted_f1"].is_monotonic_decreasing
    assert {"oracle_f1", "latency_ms", "pa_f1", "mttd", "auc_pr"} <= set(table.columns)
    assert board.recommend(min_recall=0.9) == "strong"
    assert board.recommend(max_latency_ms=5) == "weak" or board.recommend(max_latency_ms=5) is None
    assert board.recommend(min_recall=1.1) is None
    per_type = board.per_type_table("recall")
    assert (
        list(per_type.index) == table["model"].tolist() and per_type.loc["strong", "point"] == 1.0
    )
    with pytest.raises(ValueError):
        board.per_type_table("bogus")


def test_leaderboard_export(tmp_path):
    board = make_board()
    paths = board.export_leaderboard(tmp_path / "out" / "lb.csv")
    assert paths["csv"].name == "lb.csv" and paths["html"].name == "lb.html"
    csv = pd.read_csv(paths["csv"], index_col=0)
    assert csv["model"].iloc[0] == "strong"
    html = paths["html"].read_text()
    assert "<table>" in html and "strong" in html and 'class="best"' in html
    with pytest.raises(ValueError):
        ModelLeaderboard().export_leaderboard(tmp_path / "x")
    with pytest.raises(ValueError):
        ModelLeaderboard().best_model()
    assert ModelLeaderboard().generate_comparison_table().empty


def test_leaderboard_escapes_model_names(tmp_path):
    board = ModelLeaderboard()
    board.register_model("<script>x</script>", EV.compute_metrics([0, 1], [0.1, 0.9]))
    assert "<script>x</script>" not in board.to_html()


def test_evaluate_scorers_with_member_table():
    y, s, types, ts = toy()
    table = SimpleNamespace(y=y, entity_ids=np.array(["e"] * 12), timestamps=ts, anomaly_type=types)
    board = EV.evaluate_scorers(table, {"a": s, "b": np.where(y, 0.9, 0.1)}, latency_ms={"a": 2.0})
    assert board.best_model() == "b" and len(board) == 2
    assert board.generate_comparison_table().set_index("model").loc["a", "latency_ms"] == 2.0


def test_report_generation(tmp_path):
    y, s, types, ts = toy()
    res = DetectionResults(y, s, np.array(["x"] * 6 + ["y"] * 6), ts, types)
    report, sweep = EV.evaluate(res, n_thresholds=21)
    reporter = EvaluationReporter(tmp_path)
    when = datetime(2026, 11, 23, 12, 0, 5, tzinfo=UTC)
    kwargs = dict(
        y_true=y,
        y_scores=s,
        sweep=sweep,
        per_entity=EV.compute_per_entity_metrics(res),
        timestamp=when,
    )
    path = reporter.generate_report("my model/v1", report, **kwargs)
    assert path.parent == tmp_path and path.name == "my_model_v1_20261123T120005Z.html"
    html = path.read_text()
    for needle in (
        "Confusion matrix",
        "ROC curve",
        "Precision-recall curve",
        "Threshold sweep",
        "Per-entity",
        "By anomaly type",
        "Point-adjusted",
        '<script src="https://cdn.plot.ly/plotly-',
    ):
        assert needle in html
    assert "{{" not in html and "{%" not in html
    again = reporter.generate_report("my model/v1", report, **kwargs)
    assert again != path and again.exists()  # never overwrites


def test_report_inline_js_and_custom_plots(tmp_path):
    import plotly.graph_objects as go

    report = EV.compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.3])
    reporter = EvaluationReporter(tmp_path, plotlyjs="inline")
    path = reporter.generate_report("m", report, plots={"Custom": go.Figure(go.Bar(x=[1], y=[2]))})
    html = path.read_text()
    assert '<script src="https://cdn.plot.ly' not in html and "&#34;use strict&#34;" not in html
    assert len(html) > 1_000_000 and html.count("<script>") >= 1
    with pytest.raises(ValueError):
        EvaluationReporter(tmp_path, plotlyjs="bogus")
    with pytest.raises(ValueError):
        reporter.generate_report(" ", report)
