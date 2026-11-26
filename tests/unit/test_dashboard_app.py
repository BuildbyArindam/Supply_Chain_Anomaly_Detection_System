"""Tests of the Dash app shell: layout, pages, the live-refresh callback and helpers (Day 18)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import dash
import pytest
from dash import html

from src.dashboard.app import _status_strip, create_dash_app
from src.dashboard.components.theme import (
    BACKGROUND_DARK,
    CRITICAL_RED,
    INFO_BLUE,
    SEVERITY_COLORS,
    WARNING_AMBER,
    apply_theme,
    empty_figure,
)
from src.dashboard.utils import (
    format_age,
    format_timestamp,
    format_value_with_unit,
    hour_range,
    parse_timestamp,
    severity_badge,
)
from tests.unit.test_dashboard_data_service import FakeSource, alert

PAGES = ["/", "/warehouse", "/vehicles", "/inventory", "/alerts", "/model-health"]


def walk(component: Any) -> Any:
    """Yield every component of a layout tree."""
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, list | tuple):
        for child in children:
            yield from walk(child)
    elif children is not None:
        yield from walk(children)


@pytest.fixture
def app_and_source():
    source = FakeSource()
    source.alert_rows = [alert(1, "CRITICAL", 2), alert(2, "WARNING", 3), alert(3, "INFO", 4)]
    app = create_dash_app(source, public_api_url="http://api.example:8000", start_poller=False)
    return app, source


def ids_of(app: dash.Dash) -> set[str]:
    return {getattr(c, "id", None) for c in walk(app.layout) if getattr(c, "id", None)}


def test_layout_has_the_stores_interval_and_navigation(app_and_source):
    app, _ = app_and_source
    ids = ids_of(app)
    assert {
        "anomaly-store",
        "metrics-store",
        "sse-latest",
        "live-interval",
        "sidebar",
        "status-strip",
        "sse-status",
        "page-content",
        "alert-toast",
    } <= ids
    interval = next(c for c in walk(app.layout) if getattr(c, "id", None) == "live-interval")
    assert interval.interval == 2000  # dashboard.refresh_interval_ms


def test_sse_url_is_taken_from_the_public_api_url(app_and_source):
    app, _ = app_and_source
    cfg = next(c for c in walk(app.layout) if getattr(c, "id", None) == "sse-config")
    assert cfg.to_plotly_json()["props"]["data-sse-url"] == (
        "http://api.example:8000/api/v1/stream/alerts"
    )


def test_sse_can_be_disabled():
    app = create_dash_app(FakeSource(), public_api_url="", start_poller=False)
    cfg = next(c for c in walk(app.layout) if getattr(c, "id", None) == "sse-config")
    assert "data-sse-url" not in cfg.to_plotly_json()["props"]


def test_all_six_pages_are_registered_in_order_and_linked(app_and_source):
    app, _ = app_and_source
    pages = sorted(dash.page_registry.values(), key=lambda p: p["order"])
    assert [p["path"] for p in pages] == PAGES
    links = [c for c in walk(app.layout) if getattr(c, "id", "") and str(c.id).startswith("nav-")]
    assert len(links) == 6


@pytest.mark.parametrize("path", PAGES)
def test_every_page_and_the_assets_are_served(app_and_source, path):
    app, _ = app_and_source
    client = app.server.test_client()
    assert client.get(path).status_code == 200
    assert client.get("/assets/sse_client.js").status_code == 200
    assert client.get("/assets/style.css").status_code == 200


def refresh_request(version: int | None, n: int = 1) -> dict[str, Any]:
    state_value = None if version is None else {"version": version}
    return {
        "output": "..anomaly-store.data...metrics-store.data..",
        "outputs": [
            {"id": "anomaly-store", "property": "data"},
            {"id": "metrics-store", "property": "data"},
        ],
        "inputs": [{"id": "live-interval", "property": "n_intervals", "value": n}],
        "changedPropIds": ["live-interval.n_intervals"],
        "state": [{"id": "metrics-store", "property": "data", "value": state_value}],
    }


def post_refresh(client, body: dict[str, Any]):
    client.get("/_dash-layout")
    client.get("/_dash-dependencies")
    return client.post("/_dash-update-component", json=body)


def test_refresh_callback_fills_both_stores_from_the_snapshot(app_and_source):
    app, _ = app_and_source
    snapshot = app.scad_service.refresh()
    response = post_refresh(app.server.test_client(), refresh_request(None))
    assert response.status_code == 200
    payload = json.loads(response.data)["response"]
    anomalies, metrics = payload["anomaly-store"]["data"], payload["metrics-store"]["data"]
    assert anomalies["version"] == metrics["version"] == snapshot.version
    assert [a["event_id"] for a in anomalies["alerts"]] == ["e1", "e2", "e3"]  # by severity
    assert "top_contributing_features" not in anomalies["alerts"][0]  # slimmed for the browser
    assert metrics["connected"] is True and metrics["stored_alerts"] == 3


def test_refresh_callback_sends_nothing_while_the_snapshot_is_unchanged(app_and_source):
    app, _ = app_and_source
    snapshot = app.scad_service.refresh()
    response = post_refresh(app.server.test_client(), refresh_request(snapshot.version))
    assert response.status_code == 200
    assert json.loads(response.data)["response"] == {}  # every output was no_update


def test_refresh_callback_updates_again_when_the_data_changed(app_and_source):
    app, source = app_and_source
    old = app.scad_service.refresh()
    source.alert_rows = [*source.alert_rows, alert(9, "EMERGENCY", 1)]
    app.scad_service.refresh()
    response = post_refresh(app.server.test_client(), refresh_request(old.version))
    payload = json.loads(response.data)["response"]
    assert payload["anomaly-store"]["data"]["alerts"][0]["event_id"] == "e9"


def test_refresh_callback_works_when_the_source_is_down():
    source = FakeSource()
    source.fail = {"alerts", "health", "entities", "pipeline", "feedback", "models"}
    app = create_dash_app(source, start_poller=False)
    app.scad_service.refresh()
    response = post_refresh(app.server.test_client(), refresh_request(None))
    metrics = json.loads(response.data)["response"]["metrics-store"]["data"]
    assert metrics["connected"] is False and "alerts" in metrics["errors"]


def test_poller_starts_with_the_first_request_and_not_at_import():
    source = FakeSource()
    app = create_dash_app(source, start_poller=True)
    service = app.scad_service
    try:
        assert not service.running
        app.server.test_client().get("/")
        assert service.running
    finally:
        service.stop()


# --------------------------------------------------------------------------- #
# Status strip, theme, utils
# --------------------------------------------------------------------------- #
def text_of(component: Any) -> str:
    parts = []
    for c in walk(component):
        children = getattr(c, "children", None)
        if isinstance(children, str):
            parts.append(children)
    return " ".join(parts)


def test_status_strip_states():
    assert "Connecting" in text_of(_status_strip(None))
    live = _status_strip(
        {
            "connected": True,
            "api_status": "ok",
            "updated_at": 1_700_000_000.0,
            "total_alerts_today": 4,
            "avg_latency_p99": 120.0,
            "errors": {},
        }
    )
    assert (
        "Live" in text_of(live) and "4 alerts today" in text_of(live) and "120 ms" in text_of(live)
    )
    degraded = _status_strip(
        {"connected": True, "api_status": "degraded", "errors": {"models": "x"}}
    )
    assert "Degraded" in text_of(degraded) and "Unavailable: models" in text_of(degraded)
    assert "Offline" in text_of(_status_strip({"connected": False, "errors": {}}))


def test_theme_constants_and_apply_theme():
    assert SEVERITY_COLORS["CRITICAL"] == CRITICAL_RED
    assert SEVERITY_COLORS["WARNING"] == WARNING_AMBER and SEVERITY_COLORS["INFO"] == INFO_BLUE
    assert len(set(SEVERITY_COLORS.values())) == 4  # four levels, four distinct colours
    import plotly.graph_objects as go

    fig = apply_theme(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])), height=300)
    assert fig.layout.height == 300
    assert fig.layout.template.layout.paper_bgcolor == BACKGROUND_DARK
    placeholder = empty_figure("No data available")
    assert placeholder.layout.annotations[0].text == "No data available"


def test_format_timestamp_and_age():
    when = datetime(2026, 11, 26, 9, 5, 7, tzinfo=UTC)
    assert format_timestamp(when) == "09:05:07"
    assert format_timestamp(when.isoformat().replace("+00:00", "Z"), with_date=True) == (
        "2026-11-26 09:05:07"
    )
    assert format_timestamp(when.timestamp(), with_seconds=False) == "09:05"
    assert format_timestamp(None) == "—" and format_timestamp("not a date") == "—"
    assert format_age(when, now=when + timedelta(seconds=12)) == "12s ago"
    assert format_age(when, now=when + timedelta(minutes=3)) == "3 min ago"
    assert format_age(when, now=when + timedelta(hours=5)) == "5.0 h ago"
    assert format_age("bad") == "—"


def test_parse_timestamp_treats_naive_as_utc():
    assert parse_timestamp("2026-11-26T09:00:00") == datetime(2026, 11, 26, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (3.14159, "celsius", "3.1°C"),
        (54, "%", "54.0%"),
        (88.46, "km/h", "88.5 km/h"),
        (1234.5, "units", "1,234.5 units"),
        (7, None, "7.0"),
        (None, "°C", "—"),
        ("abc", "°C", "—"),
        (float("nan"), "°C", "—"),
        (float("inf"), "km/h", "—"),
    ],
)
def test_format_value_with_unit(value, unit, expected):
    assert format_value_with_unit(value, unit) == expected


def test_format_value_with_unit_precision():
    assert format_value_with_unit(120.4, "ms", 0) == "120 ms"


@pytest.mark.parametrize("level", ["INFO", "WARNING", "CRITICAL", "EMERGENCY", "critical"])
def test_severity_badge_shows_the_level_as_text(level):
    badge = severity_badge(level)
    assert badge.children == level.upper()  # colour is never the only signal
    assert f"severity-{level.lower()}" in badge.className


def test_severity_badge_tolerates_unknown_levels():
    assert severity_badge("???").children == "???"
    assert severity_badge(None).children == "—"


def test_hour_range_ends_with_the_reference_hour():
    end = datetime(2026, 11, 26, 12, 30, tzinfo=UTC)
    hours = hour_range(end, 3)
    assert hours == [end.replace(hour=h, minute=0) for h in (10, 11, 12)]


def test_html_import_is_used():  # keeps the dash.html import honest for linters
    assert html.Div().to_plotly_json()["type"] == "Div"
