"""Tests of the Overview and Warehouse pages and the service methods behind them (Day 19)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import dash
import pytest

from src.dashboard.app import create_dash_app
from src.dashboard.components.overview_widgets import (
    FEED_SIZE,
    NEUTRAL_GREY,
    alert_feed,
    health_color,
    hex_to_rgba,
    kpi_row,
    severity_donut,
    status_row,
    trend_chart,
)
from src.dashboard.components.theme import CRITICAL_RED, NORMAL_GREEN, WARNING_AMBER
from src.dashboard.components.warehouse_widgets import (
    EXTEND_JS,
    MAX_POINTS,
    alert_points,
    anomaly_card,
    grid_layout,
    make_cursor,
    match_alert,
    new_points,
    resolve_selection,
    series_note,
    timeseries_figure,
    zone_heatmap,
)
from tests.unit.test_dashboard_app import text_of, walk
from tests.unit.test_dashboard_data_service import NOW, FakeSource, alert


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
def sensor(eid: str, wh: str, zone: str, kind: str, value: float, rng=(2.0, 4.0)) -> dict[str, Any]:
    return {
        "entity_id": eid,
        "source_type": "warehouse",
        "signal": "value",
        "value": value,
        "timestamp": NOW.isoformat(),
        "unit": "celsius" if kind == "temperature" else "percent",
        "zone": zone,
        "warehouse_id": wh,
        "sensor_type": kind,
        "expected_range": list(rng),
    }


def fleet() -> list[dict[str, Any]]:
    rows = []
    for wh in ("wh1", "wh2"):
        for zone in "ABC":
            rows.append(sensor(f"{wh}-{zone}-temp-01", wh, zone, "temperature", 3.0))
            rows.append(sensor(f"{wh}-{zone}-hum-01", wh, zone, "humidity", 55.0, (40, 70)))
    return rows


def series(entity: str = "wh1-A-temp-01", n: int = 5, start: float = 3.0) -> dict[str, Any]:
    points = [
        {"t": (NOW + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"), "v": start + i / 10}
        for i in range(n)
    ]
    return {
        "entity_id": entity,
        "unit": "celsius",
        "sensor_type": "temperature",
        "expected_range": [2.0, 4.0],
        "points": points,
    }


@pytest.fixture
def app_and_source():
    source = FakeSource()
    source.alert_rows = [
        alert(1, "CRITICAL", 2),
        alert(2, "WARNING", 3),
        alert(3, "INFO", 4),
        alert(4, "EMERGENCY", 5, fp=True),
    ]
    source.entity_rows = fleet()
    source.health_row = {
        "status": "ok",
        "models_loaded": True,
        "checks": {"models": "ok", "kafka": "ok"},
    }
    source.pipeline_row = {
        "latency": {"p99_ms": 120.0},
        "latency_target_ms": 500,
        "consumer_lag": 7,
    }
    app = create_dash_app(source, public_api_url="", start_poller=False)
    app.scad_service._clock = lambda: NOW
    app.scad_service.refresh()
    return app, source


def page_layout(path: str):
    page = next(p for p in dash.page_registry.values() if p["path"] == path)
    return page["layout"]()


def ids_in(component: Any) -> set[str]:
    return {
        getattr(c, "id", None) for c in walk(component) if isinstance(getattr(c, "id", None), str)
    }


# --------------------------------------------------------------------------- #
# Data service additions
# --------------------------------------------------------------------------- #
@pytest.fixture
def service(app_and_source):
    return app_and_source[0].scad_service


def test_summary_has_active_and_critical_24h_counts(service):
    metrics = service.get_summary_metrics()
    assert metrics["active_alerts"] == 3  # the false positive is not active
    assert metrics["active_critical"] == 1
    assert metrics["critical_24h"] == 2  # CRITICAL plus the EMERGENCY, FP or not


def test_recent_alerts_are_newest_first_and_capped(service):
    recent = service.get_recent_alerts(2)
    assert [a["event_id"] for a in recent] == ["e1", "e2"]
    assert "top_contributing_features" not in recent[0]


def test_entity_alerts_skip_false_positives_and_old_ones(app_and_source):
    app, source = app_and_source
    source.alert_rows = [
        alert(1, minutes_ago=1, entity="s1"),
        alert(2, minutes_ago=30, entity="s1"),
        alert(3, minutes_ago=2, entity="s1", fp=True),
        alert(4, minutes_ago=3, entity="other"),
        alert(5, minutes_ago=4, entity="s1", resolved=True),
    ]
    svc = app.scad_service
    svc.refresh()
    found = svc.get_entity_alerts("s1")
    assert [a["event_id"] for a in found] == ["e2", "e5", "e1"]  # oldest first
    recent = svc.get_entity_alerts("s1", since=NOW - timedelta(minutes=10))
    assert [a["event_id"] for a in recent] == ["e5", "e1"]
    assert svc.get_entity_alerts("s1", limit=1)[0]["event_id"] == "e1"


def test_status_badges_when_everything_is_healthy(service):
    badges = {b["name"]: b for b in service.get_status_badges()}
    assert [b["state"] for b in badges.values()] == ["ok"] * 4
    assert badges["Kafka lag"]["detail"] == "7 msgs"


@pytest.mark.parametrize(
    ("lag", "state"), [(0, "ok"), (999, "ok"), (1_000, "warn"), (9_999, "warn"), (10_000, "error")]
)
def test_kafka_lag_badge_thresholds(app_and_source, lag, state):
    app, source = app_and_source
    source.pipeline_row = {"consumer_lag": lag}
    app.scad_service.refresh()
    assert app.scad_service.get_status_badges()[3]["state"] == state


def test_status_badges_degraded_and_unreachable(app_and_source):
    app, source = app_and_source
    source.health_row = {
        "status": "degraded",
        "models_loaded": True,
        "checks": {"models": "degraded", "kafka": "unreachable"},
    }
    source.pipeline_row = {}
    source.fail = {"feedback"}
    app.scad_service.refresh()
    badges = {b["name"]: b for b in app.scad_service.get_status_badges()}
    assert badges["Models"]["state"] == "warn"
    assert badges["Kafka"]["state"] == "error"
    assert badges["Dashboard"]["state"] == "warn" and "feedback" in badges["Dashboard"]["detail"]
    assert badges["Kafka lag"]["state"] == "unknown"

    source.fail = {"alerts", "health", "entities", "pipeline", "feedback", "models"}
    app.scad_service._values.clear()
    app.scad_service.refresh()
    down = {b["name"]: b["state"] for b in app.scad_service.get_status_badges()}
    assert down == {
        "Kafka": "unknown",
        "Models": "error",
        "Dashboard": "error",
        "Kafka lag": "unknown",
    }


# --------------------------------------------------------------------------- #
# Overview widgets
# --------------------------------------------------------------------------- #
def kpi_values(row) -> dict[str, str]:
    return {
        c.id: c.children
        for c in walk(row)
        if isinstance(getattr(c, "id", None), str)
        and c.id.startswith("ov-")
        and isinstance(getattr(c, "children", None), str)
    }


def border_colors(row) -> dict[str, str]:
    cards = [c for c in walk(row) if str(getattr(c, "id", "")).endswith("-card")]
    return {c.id: c.style["border"].split()[-1] for c in cards}


def test_kpi_row_shows_values_and_status_colours():
    metrics = {
        "active_alerts": 12,
        "active_critical": 2,
        "critical_24h": 5,
        "model_health_score": 0.92,
        "avg_latency_p99": 120.0,
    }
    row = kpi_row(metrics, {"latency_target_ms": 500})
    assert kpi_values(row) == {
        "ov-active-alerts": "12",
        "ov-critical-24h": "5",
        "ov-model-health": "92%",
        "ov-latency": "120 ms",
    }
    colors = border_colors(row)
    assert colors["ov-active-alerts-card"] == CRITICAL_RED  # a critical alert is open
    assert colors["ov-model-health-card"] == NORMAL_GREEN
    assert colors["ov-latency-card"] == NORMAL_GREEN


def test_kpi_row_without_data_uses_placeholders_and_grey():
    row = kpi_row({})
    assert set(kpi_values(row).values()) == {"—"}
    assert set(border_colors(row).values()) == {NEUTRAL_GREY}


def test_kpi_latency_over_target_is_red_and_quiet_system_is_green():
    metrics = {
        "active_alerts": 0,
        "critical_24h": 0,
        "model_health_score": 0.5,
        "avg_latency_p99": 900,
    }
    colors = border_colors(kpi_row(metrics, {"latency_target_ms": 500}))
    assert colors["ov-latency-card"] == CRITICAL_RED
    assert colors["ov-active-alerts-card"] == NORMAL_GREEN
    assert colors["ov-model-health-card"] == CRITICAL_RED


@pytest.mark.parametrize(
    ("score", "color"),
    [(0.9, NORMAL_GREEN), (0.85, WARNING_AMBER), (0.7, WARNING_AMBER), (0.69, CRITICAL_RED)],
)
def test_health_colour_bands(score, color):
    assert health_color(score) == color
    assert health_color(None) == NEUTRAL_GREY


def test_alert_feed_shows_at_most_twenty_rows_edged_by_severity():
    alerts = [alert(i, "CRITICAL" if i % 2 else "INFO", i) for i in range(30)]
    feed = alert_feed(alerts)
    rows = [c for c in walk(feed) if getattr(c, "className", None) == "alert-row"]
    assert len(rows) == FEED_SIZE == 20
    assert CRITICAL_RED in rows[1].style["borderLeft"]
    assert "CRITICAL" in text_of(rows[1])  # the word, not just the colour


def test_alert_feed_dims_closed_alerts_and_handles_empty():
    closed = alert_feed([alert(1, resolved=True), alert(2)])
    rows = [c for c in walk(closed) if getattr(c, "className", None) == "alert-row"]
    assert rows[0].style["opacity"] == 0.6 and "opacity" not in rows[1].style
    assert "No alerts yet" in text_of(alert_feed([]))


def test_severity_donut_counts_and_empty_state():
    fig = severity_donut([alert(1, "CRITICAL"), alert(2, "CRITICAL"), alert(3, "INFO")])
    pie = fig.data[0]
    assert pie.hole == 0.5
    assert list(pie.labels) == ["INFO", "CRITICAL"] and list(pie.values) == [1, 2]
    assert "3" in fig.layout.annotations[0].text
    assert "No active alerts" in severity_donut([]).layout.annotations[0].text


def test_trend_chart_stacks_four_severities_and_handles_no_alerts(service):
    fig = trend_chart(service.get_alert_trend(24))
    assert fig.layout.barmode == "stack" and len(fig.data) == 4
    assert all(len(trace.x) == 24 for trace in fig.data)
    assert sum(sum(trace.y) for trace in fig.data) == 4
    assert "No alerts" in trend_chart([]).layout.annotations[0].text


def test_status_row_names_the_state_in_words():
    row = status_row(
        [
            {"name": "Kafka", "state": "ok", "detail": "reachable"},
            {"name": "Kafka lag", "state": "error", "detail": "12,000 msgs"},
        ]
    )
    text = text_of(row)
    assert "OK" in text and "Down" in text and "12,000 msgs" in text
    assert "ov-status-kafka-lag" in ids_in(row)


def test_hex_to_rgba():
    assert hex_to_rgba("#e74c3c", 0.5) == "rgba(231, 76, 60, 0.5)"


# --------------------------------------------------------------------------- #
# Warehouse widgets
# --------------------------------------------------------------------------- #
def test_selection_defaults_to_first_choices_and_cascades():
    sel = resolve_selection(fleet())
    assert (sel["warehouse"], sel["zone"], sel["sensor"]) == ("wh1", "A", "wh1-A-hum-01")
    assert [o["value"] for o in sel["warehouse_options"]] == ["wh1", "wh2"]
    assert [o["label"] for o in sel["zone_options"]] == ["Zone A", "Zone B", "Zone C"]
    assert [o["value"] for o in sel["sensor_options"]] == ["wh1-A-hum-01", "wh1-A-temp-01"]
    assert "humidity" in sel["sensor_options"][0]["label"]


def test_selection_keeps_valid_choices_and_resets_invalid_ones():
    kept = resolve_selection(fleet(), "wh2", "B", "wh2-B-temp-01")
    assert (kept["warehouse"], kept["zone"], kept["sensor"]) == ("wh2", "B", "wh2-B-temp-01")
    moved = resolve_selection(fleet(), "wh2", "B", "wh1-A-temp-01")  # sensor of another warehouse
    assert moved["sensor"] == "wh2-B-hum-01"
    changed_zone = resolve_selection(fleet(), "wh1", "Z", "wh1-A-temp-01")
    assert changed_zone["zone"] == "A"


def test_selection_with_no_sensors_is_all_none():
    sel = resolve_selection([])
    assert sel["warehouse"] is sel["zone"] is sel["sensor"] is None
    assert sel["sensor_options"] == []
    vehicles = [{"entity_id": "v1", "source_type": "vehicle"}]
    assert resolve_selection(vehicles)["warehouse"] is None


def test_timeseries_figure_has_three_traces_range_lines_and_band():
    fig = timeseries_figure(series(), [alert(1)], entity_id="wh1-A-temp-01", now=NOW)
    assert [t.name for t in fig.data][:2] == ["Temperature", "Anomaly"] and len(fig.data) == 3
    assert fig.data[1].marker.color == CRITICAL_RED
    dashed = [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert sorted(s.y0 for s in dashed) == [2.0, 4.0]
    assert any(s.type == "rect" for s in fig.layout.shapes)  # healthy band
    assert list(fig.data[2].y) == [2.0, 4.0] and fig.data[2].marker.opacity == 0
    assert fig.layout.uirevision == "wh1-A-temp-01"


def test_timeseries_figure_without_data_keeps_all_three_traces():
    for empty in (None, {"points": []}):
        fig = timeseries_figure(empty, [], now=NOW)
        assert len(fig.data) == 3 and len(fig.data[0].x) == 0  # extendData needs the traces


def test_alert_points_fall_back_to_last_reading():
    marks = alert_points(
        [{"timestamp": "2026-11-26T12:30:00Z", "raw_value": None, "severity": "info"}], series()
    )
    assert marks[0]["v"] == pytest.approx(3.4) and marks[0]["severity"] == "INFO"
    assert alert_points([{"timestamp": None, "raw_value": 1}], None) == []


def test_series_note():
    assert "No data" in series_note(None)
    assert "Waiting" in series_note({"points": []})
    assert series_note(series()) == ""


def test_new_points_returns_only_what_is_newer_than_the_cursor():
    s = series(n=6)
    cursor = make_cursor("e", s, [])
    assert new_points(s, [], cursor, "e") is None  # nothing new
    longer = series(n=9)
    delta = new_points(longer, [], cursor, "e")
    assert delta is not None and len(delta["points"]) == 3 and delta["alerts"] == []
    assert delta["points"][0]["v"] == pytest.approx(3.6)


def test_new_points_reports_new_alerts():
    s = series()
    cursor = make_cursor("e", s, [])
    late = alert(7, minutes_ago=0)
    late["timestamp"] = (NOW + timedelta(seconds=30)).isoformat()
    delta = new_points(s, [late], cursor, "e")
    assert delta is not None and delta["points"] == [] and len(delta["alerts"]) == 1


def test_new_points_ignores_a_cursor_of_another_sensor_and_cold_start():
    s = series()
    assert new_points(s, [], make_cursor("other", s, []), "e") is None
    assert new_points(s, [], None, "e") is None
    assert new_points(None, [], make_cursor("e", None, []), "e") is None
    cold = make_cursor("e", {"points": []}, [])
    assert cold["t"] == 0
    assert new_points(s, [], cold, "e") is not None  # first data after an empty chart


def test_extend_js_uses_the_configured_point_limit_and_trace_order():
    assert f"[0, 1], {MAX_POINTS}]" in EXTEND_JS
    assert "__MAX_POINTS__" not in EXTEND_JS


def test_grid_is_at_least_4_by_5_and_one_row_per_zone():
    rows, labels = grid_layout(
        [e for e in fleet() if e["warehouse_id"] == "wh1" and "temp" in e["entity_id"]]
    )
    assert len(rows) == 4 and all(len(r) == 5 for r in rows)
    assert labels == ["Zone A", "Zone B", "Zone C", ""]
    assert [r[0]["entity_id"] for r in rows[:3]] == [
        "wh1-A-temp-01",
        "wh1-B-temp-01",
        "wh1-C-temp-01",
    ]
    assert all(cell is None for cell in rows[3])


def test_grid_wraps_and_grows_rather_than_dropping_sensors():
    many = [sensor(f"wh1-A-temp-{i:02d}", "wh1", "A", "temperature", 3.0) for i in range(12)]
    many += [sensor(f"wh1-{z}-temp-01", "wh1", z, "temperature", 3.0) for z in "BCDE"]
    rows, labels = grid_layout(many)
    placed = [c["entity_id"] for r in rows for c in r if c]
    assert len(placed) == 16 and len(set(placed)) == 16
    assert len(rows) == 7  # zone A needs 3 rows, B-E one each, then padding is not needed
    assert labels[:4] == ["Zone A", "", "", "Zone B"]


def test_heatmap_marks_out_of_range_cells_and_filters_by_kind_and_warehouse():
    entities = fleet()
    hot = next(e for e in entities if e["entity_id"] == "wh1-B-temp-01")
    hot["value"] = 9.5  # cold-storage limit is 4 degrees
    fig = zone_heatmap(entities, "wh1", "temperature")
    z = fig.data[0].z
    assert sum(v is not None for row in z for v in row) == 3  # wh1's three temperature sensors
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].line.color == CRITICAL_RED
    assert any("OUTSIDE" in cell for row in fig.data[0].customdata for cell in row)
    humidity = zone_heatmap(entities, "wh1", "humidity")
    assert len(humidity.layout.shapes) == 0 and "%" in humidity.data[0].text[0][0]


def test_heatmap_empty_and_flat_colour_range():
    assert "No temperature readings" in zone_heatmap([], "wh1").layout.annotations[0].text
    flat = zone_heatmap([sensor("wh1-A-temp-01", "wh1", "A", "temperature", 3.0)], "wh1")
    assert flat.data[0].zmin < 3.0 < flat.data[0].zmax
    bad = [sensor("wh1-A-temp-01", "wh1", "A", "temperature", float("nan"))]
    assert zone_heatmap(bad, "wh1").data == ()


def test_match_alert_finds_the_nearest_within_tolerance():
    a1, a2 = alert(1, minutes_ago=1), alert(2, minutes_ago=2)
    clicked = (NOW - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S.%f")  # plotly's format
    assert match_alert([a1, a2], clicked)["event_id"] == "e2"
    assert match_alert([a1, a2], (NOW - timedelta(minutes=10)).isoformat()) is None
    assert match_alert([a1], None) is None and match_alert([a1], "junk") is None


def test_anomaly_card_explains_the_alert():
    full = alert(1, "CRITICAL", 2)
    full["recommended_action"] = "Check the compressor"
    full["top_contributing_features"] = [
        {"feature": "z_score", "shap_value": 0.4, "feature_value": 4.0},
        {"feature": "range_margin", "shap_value": -0.1},
        {"name": "legacy", "contribution": 0.05},
        {"feature": "bad", "shap_value": "x"},  # unusable entries are skipped
    ]
    card = anomaly_card(full, "°C")
    text = text_of(card)
    assert "CRITICAL" in text and "Check the compressor" in text and "9.00°C" in text
    assert (
        "z_score" in text and "legacy" in text and "bad" not in text.replace("Top contributing", "")
    )
    assert "2 to 4" in text and "Open" in text


def test_anomaly_card_placeholders_and_status():
    assert "Click a red anomaly marker" in text_of(anomaly_card(None))
    bare = anomaly_card({"event_id": "x", "severity": "INFO", "false_positive_marked": True})
    text = text_of(bare)
    assert "False positive" in text and "No feature explanation" in text
    assert "No recommendation" in text


# --------------------------------------------------------------------------- #
# Pages and callbacks through the Dash server
# --------------------------------------------------------------------------- #
def test_overview_page_renders_data_on_first_paint(app_and_source):
    app, _ = app_and_source
    with app.server.test_request_context("/"):
        layout = page_layout("/")
    ids = ids_in(layout)
    assert {"ov-active-alerts", "ov-critical-24h", "ov-model-health", "ov-latency"} <= ids
    assert {"ov-feed", "ov-donut", "ov-trend", "ov-status", "ov-version"} <= ids
    assert kpi_values(layout)["ov-active-alerts"] == "3"
    assert len([c for c in walk(layout) if getattr(c, "className", None) == "alert-row"]) == 4


def test_warehouse_page_renders_chart_heatmap_and_selectors(app_and_source):
    app, _ = app_and_source
    with app.server.test_request_context("/warehouse"):
        layout = page_layout("/warehouse")
    ids = ids_in(layout)
    assert {"wh-warehouse", "wh-zone", "wh-sensor", "wh-graph", "wh-heatmap", "wh-detail"} <= ids
    assert {"wh-cursor", "wh-delta", "wh-kind", "wh-note"} <= ids
    heatmap = next(c for c in walk(layout) if getattr(c, "id", None) == "wh-heatmap")
    assert sum(v is not None for row in heatmap.figure.data[0].z for v in row) == 3


def test_warehouse_page_without_sensors_still_renders(app_and_source):
    app, source = app_and_source
    source.entity_rows = []
    app.scad_service.refresh()
    with app.server.test_request_context("/warehouse"):
        layout = page_layout("/warehouse")
    note = next(c for c in walk(layout) if getattr(c, "id", None) == "wh-note")
    assert "No warehouse sensors" in note.children


def dependency(client, output_fragment: str) -> dict[str, Any]:
    client.get("/_dash-layout")
    deps = json.loads(client.get("/_dash-dependencies").data)
    return next(d for d in deps if output_fragment in d["output"])


def call(client, dep: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """POST a callback the way the browser does; ``values`` maps ``id.prop`` to a value."""

    def entry(spec):
        return {**spec, "value": values.get(f"{spec['id']}.{spec['property']}")}

    outputs = dep["output"].strip(".").split("...")
    body = {
        "output": dep["output"],
        "outputs": [{"id": o.rsplit(".", 1)[0], "property": o.rsplit(".", 1)[1]} for o in outputs],
        "inputs": [entry(i) for i in dep["inputs"]],
        "state": [entry(s) for s in dep.get("state", [])],
        "changedPropIds": [f"{dep['inputs'][0]['id']}.{dep['inputs'][0]['property']}"],
    }
    if len(outputs) == 1:
        body["outputs"] = body["outputs"][0]
    response = client.post("/_dash-update-component", json=body)
    assert response.status_code == 200, response.data
    return json.loads(response.data).get("response", {})


def test_overview_callback_rerenders_only_when_the_snapshot_changed(app_and_source):
    app, source = app_and_source
    client = app.server.test_client()
    dep = dependency(client, "ov-kpis.children")
    version = app.scad_service.snapshot().version
    same = call(client, dep, {"live-interval.n_intervals": 3, "ov-version.data": version})
    assert same == {}  # nothing changed: every output was no_update
    source.alert_rows = [*source.alert_rows, alert(9, "EMERGENCY", 1)]
    app.scad_service.refresh()
    fresh = call(client, dep, {"live-interval.n_intervals": 4, "ov-version.data": version})
    assert set(fresh) == {"ov-status", "ov-kpis", "ov-feed", "ov-donut", "ov-trend", "ov-version"}
    assert fresh["ov-version"]["data"] == app.scad_service.snapshot().version
    assert "ov-version" in fresh and "ov-active-alerts" in json.dumps(fresh["ov-kpis"])


def test_warehouse_selector_callback_follows_the_choice(app_and_source):
    app, _ = app_and_source
    client = app.server.test_client()
    dep = dependency(client, "wh-warehouse.options")
    out = call(
        client,
        dep,
        {
            "wh-warehouse.value": "wh2",
            "wh-zone.value": "B",
            "wh-sensor.value": "wh1-A-temp-01",
            "wh-warehouse.options": [
                {"label": "wh1", "value": "wh1"},
                {"label": "wh2", "value": "wh2"},
            ],
        },
    )
    assert out["wh-sensor"]["value"] == "wh2-B-hum-01"  # the old sensor is in another warehouse
    assert "wh-warehouse" not in out  # unchanged options and value are not resent


def test_rebuild_chart_callback_returns_figure_cursor_and_note(app_and_source):
    app, source = app_and_source
    source.timeseries = lambda entity_id, lookback: series(entity_id)  # type: ignore[method-assign]
    client = app.server.test_client()
    dep = dependency(client, "wh-graph.figure")
    out = call(client, dep, {"wh-sensor.value": "wh1-A-temp-01"})
    assert len(out["wh-graph"]["figure"]["data"]) == 3
    assert out["wh-cursor"]["data"]["entity_id"] == "wh1-A-temp-01"
    assert out["wh-cursor"]["data"]["t"] > 0 and out["wh-note"]["children"] == ""


def test_poll_callback_sends_only_new_points(app_and_source):
    app, source = app_and_source
    current = series(n=5)
    source.timeseries = lambda entity_id, lookback: current  # type: ignore[method-assign]
    app.scad_service._series_cache.clear()
    client = app.server.test_client()
    dep = dependency(client, "wh-delta.data")
    cursor = make_cursor("wh1-A-temp-01", current, [])
    values = {"wh-sensor.value": "wh1-A-temp-01", "wh-cursor.data": cursor}
    assert call(client, dep, values) == {}  # nothing new: nothing sent

    current = series(n=8)
    app.scad_service._series_cache.clear()
    out = call(client, dep, values)
    assert len(out["wh-delta"]["data"]["points"]) == 3
    assert "wh-note" not in out  # the chart already had data, so the note is left alone

    cold = {
        "wh-sensor.value": "wh1-A-temp-01",
        "wh-cursor.data": make_cursor("wh1-A-temp-01", None, []),
    }
    first = call(client, dep, cold)
    assert len(first["wh-delta"]["data"]["points"]) == 8
    assert first["wh-note"]["children"] == ""  # first data arrives: the "waiting" note is cleared


def test_heatmap_callback_follows_warehouse_and_kind(app_and_source):
    app, _ = app_and_source
    client = app.server.test_client()
    dep = dependency(client, "wh-heatmap.figure")
    out = call(
        client,
        dep,
        {"wh-warehouse.value": "wh2", "wh-kind.value": "humidity", "live-interval.n_intervals": 2},
    )
    z = out["wh-heatmap"]["figure"]["data"][0]["z"]
    assert sorted(v for row in z for v in row if v is not None) == [55.0, 55.0, 55.0]


def test_clicking_a_marker_shows_the_alert_even_with_the_unified_hover(app_and_source):
    app, source = app_and_source
    source.alert_rows = [
        {**alert(1, "CRITICAL", 1, entity="wh1-A-temp-01"), "recommended_action": "Act now"}
    ]
    app.scad_service.refresh()
    client = app.server.test_client()
    dep = dependency(client, "wh-detail.children")
    x = (NOW - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    # a unified-hover click lists the line (curve 0) before the marker (curve 1)
    click = {"points": [{"curveNumber": 0, "x": x, "y": 3.1}, {"curveNumber": 1, "x": x, "y": 9.0}]}
    out = call(client, dep, {"wh-graph.clickData": click, "wh-sensor.value": "wh1-A-temp-01"})
    assert "Act now" in json.dumps(out["wh-detail"]["children"])
    on_line = {"points": [{"curveNumber": 0, "x": x, "y": 3.1}]}
    assert (
        call(client, dep, {"wh-graph.clickData": on_line, "wh-sensor.value": "wh1-A-temp-01"}) == {}
    )


def test_pages_use_prevent_initial_call_and_the_live_interval(app_and_source):
    app, _ = app_and_source
    client = app.server.test_client()
    client.get("/_dash-layout")
    deps = json.loads(client.get("/_dash-dependencies").data)
    page_deps = [d for d in deps if d["output"].lstrip(".").startswith(("ov-", "wh-"))]
    assert len(page_deps) >= 7
    assert all(d.get("prevent_initial_call") for d in page_deps)
    live = [d for d in page_deps if any(i["id"] == "live-interval" for i in d["inputs"])]
    assert len(live) >= 4  # overview, selectors, increment and heatmap
    clientside = [d for d in page_deps if d.get("clientside_function")]
    assert len(clientside) == 1 and "extendData" in clientside[0]["output"]


def test_both_pages_are_served_with_data(app_and_source):
    app, _ = app_and_source
    client = app.server.test_client()
    for path in ("/", "/warehouse"):
        assert client.get(path).status_code == 200
