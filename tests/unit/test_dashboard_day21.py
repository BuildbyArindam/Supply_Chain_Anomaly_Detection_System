"""Tests of the Day 21 Model Health page: feedback timeline, data service, widgets, page, API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.dashboard.app import create_dash_app
from src.dashboard.components import model_health_widgets as w
from src.dashboard.data_service import DashboardDataService
from src.models.feedback_loop import FeedbackProcessor
from tests.unit.test_dashboard_app import walk
from tests.unit.test_dashboard_data_service import NOW, FakeSource, alert

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
RUN_1 = {
    "run_id": "aaaaaaaa-1",
    "trigger_reason": "scheduled",
    "started_at": "2026-11-26T08:00:00+00:00",
    "finished_at": "2026-11-26T08:05:00+00:00",
    "data_size": 1200,
    "training_time_s": 300.0,
    "champion_f1": 0.80,
    "challenger_f1": 0.84,
    "improvement": 0.04,
    "outcome": "promoted",
}
RUN_2 = {
    **RUN_1,
    "run_id": "bbbbbbbb-2",
    "trigger_reason": "drift",
    "started_at": "2026-11-26T10:00:00+00:00",
    "finished_at": "2026-11-26T10:01:00+00:00",
    "challenger_f1": 0.78,
    "improvement": -0.02,
    "outcome": "rejected",
}
CHALLENGER = {
    "model_name": "lstm_classifier",
    "min_improvement": 0.02,
    "champion": {"version": "v1", "metrics": {"f1": 0.80, "precision": 0.70, "recall": 0.90}},
    "challenger": {"version": "v2", "metrics": {"f1": 0.88, "precision": 0.85, "recall": 0.91}},
    "pr_curve": {
        "champion": {"recall": [0, 0.5, 1], "precision": [1, 0.8, 0.5]},
        "challenger": {"recall": [0, 0.5, 1], "precision": [1, 0.9, 0.6]},
    },
}


@pytest.fixture
def source() -> FakeSource:
    src = FakeSource()
    src.alert_rows = [alert(1, "CRITICAL", 5), alert(2, "WARNING", 10)]
    return src


@pytest.fixture
def service(source: FakeSource) -> DashboardDataService:
    svc = DashboardDataService(source, clock=lambda: NOW)
    svc.refresh()
    return svc


# --------------------------------------------------------------------------- #
# FeedbackProcessor.timeline
# --------------------------------------------------------------------------- #
def fb(kind: str, entity: str, when: datetime, analyst: str = "a") -> Any:
    return SimpleNamespace(
        timestamp=when,
        entity_id=entity,
        is_label=kind in ("false_positive", "true_positive"),
        is_false_positive=kind == "false_positive",
    )


def timeline_of(events: list[Any], days: int = 7) -> dict[str, Any]:
    processor = SimpleNamespace(store=SimpleNamespace(all=lambda: events))
    return FeedbackProcessor.timeline(processor, days, now=NOW)  # type: ignore[arg-type]


def test_timeline_rates_gaps_heatmap_and_entities():
    day = NOW.replace(hour=9, minute=0)
    events = [
        fb("false_positive", "s1", day),
        fb("false_positive", "s1", day + timedelta(minutes=5)),
        fb("true_positive", "s1", day + timedelta(minutes=10)),
        fb("severity_adjustment", "s2", day + timedelta(hours=1)),  # counts as activity only
        fb("true_positive", "s2", day - timedelta(days=2)),
        fb("false_positive", "old", NOW - timedelta(days=30)),  # outside the window
    ]
    out = timeline_of(events)
    assert [r["date"] for r in out["daily"]][-1] == "2026-11-26"
    assert len(out["daily"]) == 7
    today = out["daily"][-1]
    assert (today["total"], today["false_positive"]) == (3, 2)
    assert today["false_positive_rate"] == pytest.approx(2 / 3)
    assert out["daily"][0]["false_positive_rate"] is None  # no verdicts: a gap, not 0 %
    cells = {(c["date"], c["hour"]): c["count"] for c in out["heatmap"]}
    assert cells[("2026-11-26", 9)] == 3 and cells[("2026-11-26", 10)] == 1
    assert [e["entity_id"] for e in out["entities"]] == ["s1", "s2"]
    assert out["entities"][1]["false_positive_rate"] == 0.0


def test_timeline_clamps_the_window():
    assert len(timeline_of([], days=0)["daily"]) == 1
    assert len(timeline_of([], days=500)["daily"]) == 90


# --------------------------------------------------------------------------- #
# Data service
# --------------------------------------------------------------------------- #
def test_model_scores_average_families_and_leave_missing_ones_none(source, service):
    def entry(model_type: str, **metrics: float) -> dict[str, Any]:
        return {
            "model_type": model_type,
            "versions": [{"version": "1", "active": True, "metrics": metrics}],
        }

    source.models_row = {
        "registry": [
            entry("isolation_forest_warehouse", f1=0.8),
            entry("isolation_forest_vehicle", f1=0.6),
            entry("lstm_classifier", weighted_f1=0.9),
            entry("ensemble_meta_learner", f1=0.88, precision=0.9, recall=0.86),
        ],
        "current": {},
    }
    service.refresh()
    scores = service.get_model_scores()
    assert scores["f1"]["isolation_forest"] == pytest.approx(0.7)
    assert scores["f1"]["lstm_classifier"] == 0.9
    assert scores["f1"]["statistical"] is None and scores["f1"]["lstm_autoencoder"] is None
    assert scores["ensemble"] == {"f1": 0.88, "precision": 0.9, "recall": 0.86}


def test_anomaly_type_breakdown_counts_only_today(source, service):
    source.alert_rows = [
        alert(1, "CRITICAL", 5),
        alert(2, "WARNING", 10),
        alert(3, "WARNING", 60 * 30),  # yesterday
    ]
    service.refresh()
    out = service.get_anomaly_type_breakdown()
    assert out["total"] == 2
    assert out["counts"]["point"]["CRITICAL"] == 1 and out["counts"]["point"]["WARNING"] == 1
    assert out["counts"]["collective"]["WARNING"] == 0  # always listed, zero-filled


def test_feedback_sections_read_the_timeline(source, service):
    source.timeline_row = {
        "daily": [
            {"date": "2026-11-25", "total": 4, "false_positive": 1, "false_positive_rate": 0.25},
            {"date": "2026-11-26", "total": 0, "false_positive": 0, "false_positive_rate": None},
        ],
        "heatmap": [{"date": "2026-11-25", "hour": 14, "count": 4}],
        "entities": [
            {"entity_id": "s1", "total": 5, "false_positive": 4, "false_positive_rate": 0.8},
            {"entity_id": "s2", "total": 1, "false_positive": 1, "false_positive_rate": 1.0},
            {"entity_id": "s3", "total": 6, "false_positive": 0, "false_positive_rate": 0.0},
        ],
    }
    service.refresh()
    assert [r["date"] for r in service.get_feedback_trend()] == ["2026-11-25", "2026-11-26"]
    grid = service.get_feedback_heatmap()
    assert grid["z"][14] == [4, 0] and len(grid["z"]) == 24
    top = service.get_top_fp_entities()
    assert [r["entity_id"] for r in top] == ["s1"]  # s2 too few verdicts, s3 no false positives


def test_retraining_runs_sorted_and_delta_flags_rollback(source, service):
    source.retraining_row = {
        "enabled": True,
        "runs": [RUN_2, RUN_1],
        "rollbacks": [{"run_id": "aaaaaaaa-1"}],
        "challenger": CHALLENGER,
    }
    service.refresh()
    runs = service.get_retraining()["runs"]
    assert [r["run_id"] for r in runs] == ["aaaaaaaa-1", "bbbbbbbb-2"]
    delta = service.get_retraining_delta()
    assert delta is not None and delta["run_id"] == "aaaaaaaa-1" and delta["rolled_back"] is True
    source.retraining_row = {"enabled": True, "runs": [RUN_2]}
    service.refresh()
    assert service.get_retraining_delta() is None


def test_review_challenger_success_failure_and_unknown(source, service):
    source.retraining_row = {"enabled": True, "runs": [], "challenger": CHALLENGER}
    service.refresh()
    assert service.get_retraining()["challenger"] is not None
    out = service.review_challenger("promote")
    assert out == {"ok": True, "action": "promote", "message": "Challenger promoted."}
    assert source.reviewed == ["promote"]
    assert service.get_retraining()["challenger"] is None  # snapshot refreshed

    source.review_error = httpx.HTTPStatusError(
        "x", request=httpx.Request("POST", "http://x"), response=httpx.Response(401)
    )
    assert "API_ADMIN_KEY" in service.review_challenger("reject")["message"]
    source.review_error = ConnectionError("down")
    assert service.review_challenger("reject")["ok"] is False
    assert service.review_challenger("explode")["ok"] is False


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "color"),
    [(0.9, "#2ecc71"), (0.8, "#f39c12"), (0.5, "#e74c3c")],
)
def test_gauge_bar_colour_follows_the_thresholds(value, color):
    from src.dashboard.components.overview_widgets import health_color

    fig = w.gauge("F1", value)
    assert fig.data[0].value == pytest.approx(value * 100)
    assert fig.data[0].gauge.bar.color == health_color(value)
    assert color  # documents the intended green / amber / red


def test_gauge_without_a_value_is_a_placeholder_not_zero():
    fig = w.gauge("F1", None)
    assert not fig.data and "no evaluation metrics" in fig.layout.annotations[0].text


def test_f1_chart_marks_models_without_metrics():
    fig = w.model_f1_chart({"ensemble": 0.9, "statistical": None})
    assert list(fig.data[0].x) == ["Statistical", "IForest", "LSTM-AE", "LSTM-Cls", "Ensemble"]
    assert fig.data[0].y[0] is None and fig.data[0].y[4] == 0.9
    assert not w.model_f1_chart({}).data


def test_fp_trend_has_a_dashed_10_percent_line_and_keeps_gaps():
    daily = [
        {"date": "2026-11-25", "total": 4, "false_positive": 1, "false_positive_rate": 0.25},
        {"date": "2026-11-26", "total": 0, "false_positive": 0, "false_positive_rate": None},
    ]
    fig = w.fp_trend_chart(daily)
    line = next(s for s in fig.layout.shapes if s.type == "line")
    assert line.y0 == pytest.approx(0.10) and line.line.dash == "dash"
    assert fig.data[0].y[1] is None and fig.data[0].connectgaps is False
    assert not w.fp_trend_chart([]).data


def test_heatmap_and_anomaly_chart_empty_states():
    assert not w.feedback_heatmap({"days": [], "z": []}).data
    assert not w.anomaly_type_chart({"total": 0}).data
    fig = w.feedback_heatmap({"days": ["d1"], "hours": list(range(24)), "z": [[1]] * 24})
    assert fig.data[0].type == "heatmap"


def test_timeline_colours_by_outcome_and_widens_short_runs():
    fig = w.retraining_timeline([RUN_1, RUN_2])
    colours = {t.name: t.marker.color for t in fig.data}
    assert colours["Promoted"] == w.OUTCOME_STYLE["promoted"][0]
    assert colours["Rejected"] == w.OUTCOME_STYLE["rejected"][0]
    assert colours["Promoted"] != colours["Rejected"]
    assert all(min(t.x) > 0 for t in fig.data)
    assert not w.retraining_timeline([]).data


def test_retraining_notice_distinguishes_disabled_from_empty():
    assert "not running" in (w.retraining_notice(False, []) or "")
    assert "No retraining run yet" in (w.retraining_notice(True, []) or "")
    assert w.retraining_notice(True, [RUN_1]) is None


def test_delta_arrow_direction():
    assert w.delta_arrow(0.021).children.startswith("▲")
    assert w.delta_arrow(-0.02).children.startswith("▼")
    assert w.delta_arrow(0.0).children.startswith("►")
    assert w.delta_arrow(None).children == "—"


def test_comparison_rows_percent_change_vs_champion():
    rows = {r["metric"]: r for r in w.comparison_rows(CHALLENGER)}
    assert list(rows)[:3] == ["f1", "precision", "recall"]
    assert rows["f1"]["change_pct"] == pytest.approx(10.0)
    assert rows["precision"]["change_pct"] == pytest.approx(21.428, abs=1e-2)
    zero = {"champion": {"metrics": {"f1": 0.0}}, "challenger": {"metrics": {"f1": 0.5}}}
    assert w.comparison_rows(zero)[0]["change_pct"] is None


def test_pr_overlay_champion_blue_challenger_orange():
    fig = w.pr_overlay(CHALLENGER)
    colours = {t.name: t.line.color for t in fig.data}
    assert colours == {"Champion": w.CHAMPION_COLOR, "Challenger": w.CHALLENGER_COLOR}
    assert not w.pr_overlay({}).data


def test_drift_banner_only_when_active():
    assert not any(
        getattr(c, "color", None) == "danger" for c in walk(w.drift_banner(False, {"message": "x"}))
    )
    banner = w.drift_banner(True, {"message": "FP rate rose", "recent_fp_rate": 0.4})
    assert any(getattr(c, "color", None) == "danger" for c in walk(banner))


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_and_source(source):
    app = create_dash_app(source, public_api_url="", start_poller=False)
    return app, source


@pytest.fixture
def model_health(app_and_source):
    """The page module as Dash loaded it (importing it directly would register the page twice)."""
    import sys

    import dash

    assert app_and_source  # the app must exist first: it loads the pages folder
    return sys.modules[dash.page_registry["pages.model_health"]["module"]]


def ids(component: Any) -> set[str]:
    return {getattr(c, "id", None) for c in walk(component) if getattr(c, "id", None)}


def test_page_layout_has_every_section(app_and_source, model_health):
    app, _ = app_and_source
    with app.server.test_request_context():
        page = model_health.layout()
    found = ids(page)
    for ident in (
        *model_health.GAUGE_IDS,
        "mh-f1-chart",
        "mh-type-chart",
        "mh-fp-trend",
        "mh-heatmap",
        "mh-top-fp",
        "mh-timeline",
        "mh-log",
        "mh-delta-wrap",
        "mh-cc-section",
        "mh-promote",
        "mh-reject",
    ):
        assert ident in found, ident


def test_page_renders_on_an_empty_system(app_and_source, model_health):
    app, _ = app_and_source
    client = app.server.test_client()
    assert client.get("/model-health").status_code == 200
    with app.server.test_request_context():
        content = model_health.build_content(app.scad_service)
    assert content["has_challenger"] is False
    assert "not running" in content["notice"]


def test_challenger_section_shows_only_with_a_challenger(app_and_source, model_health):
    app, source = app_and_source
    with app.server.test_request_context():
        assert model_health.layout().children  # builds
        hidden = next(
            c for c in walk(model_health.layout()) if getattr(c, "id", None) == "mh-cc-section"
        )
        assert hidden.style == {"display": "none"}
        source.retraining_row = {"enabled": True, "runs": [RUN_1], "challenger": CHALLENGER}
        app.scad_service.refresh()
        shown = next(
            c for c in walk(model_health.layout()) if getattr(c, "id", None) == "mh-cc-section"
        )
        assert shown.style == {}


def test_drift_banner_shows_when_the_detector_fired_recently(app_and_source, model_health):
    app, source = app_and_source
    fresh = datetime.now(UTC).isoformat()
    source.feedback_row = {"drift": {"last_alert": {"detected_at": fresh, "message": "FP up"}}}
    app.scad_service.refresh()
    with app.server.test_request_context():
        content = model_health.build_content(app.scad_service)
    assert any(getattr(c, "id", None) == "mh-drift-alert" for c in walk(content["drift"]))


def test_refresh_skips_work_when_the_snapshot_is_unchanged(app_and_source, model_health):
    app, _ = app_and_source
    snap = app.scad_service.refresh()
    with app.server.test_request_context():
        out = model_health.refresh_model_health(1, snap.version)
        assert len(out) == 20 and all(o is model_health.no_update for o in out)
        changed = model_health.refresh_model_health(1, snap.version - 1)
    assert len(changed) == 20 and changed[-1] == snap.version


def test_confirm_then_submit_calls_the_api(app_and_source, model_health):
    app, source = app_and_source
    source.retraining_row = {"enabled": True, "runs": [], "challenger": CHALLENGER}
    app.scad_service.refresh()
    with app.server.test_request_context():
        assert model_health.review_challenger(1, None) is model_health.no_update
        assert source.reviewed == []
        notice = model_health.review_challenger(1, "promote")
    assert source.reviewed == ["promote"]
    assert notice.color == "success"
