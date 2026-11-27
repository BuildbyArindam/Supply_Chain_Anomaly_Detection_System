"""Overview page (Day 19): KPI scorecards, live alert feed, severity donut, trend and status row.

Everything is rendered by :func:`build_content` from the :class:`DashboardDataService` snapshot.
The page layout is a *function*, so the first paint already holds data (the callback below has
``prevent_initial_call=True``); after that the ``live-interval`` tick re-renders the content, and
only when the snapshot version changed, so an idle system costs the browser nothing.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, State, dcc, html, no_update

from src.dashboard.components.layout import page_header
from src.dashboard.components.overview_widgets import (
    alert_feed,
    card_with_title,
    graph,
    kpi_row,
    severity_donut,
    status_row,
    trend_chart,
)
from src.dashboard.data_service import DashboardDataService

dash.register_page(
    __name__, path="/", name="Overview", title="Overview · Supply Chain Anomaly Monitor", order=0
)

TREND_HOURS = 24


def _service() -> DashboardDataService:
    """Return the dashboard's data service (``create_dash_app`` attaches it to the app)."""
    return flask.current_app.extensions["scad_service"]


def build_content(service: DashboardDataService) -> dict[str, Any]:
    """Render every dynamic part of the page from the current snapshot.

    Returns:
        ``{"version", "kpis", "feed", "donut", "trend", "status"}``.
    """
    snap = service.snapshot()
    metrics = service.get_summary_metrics()
    active = service.get_active_anomalies(max(len(snap.alerts), 1))
    return {
        "version": snap.version,
        "kpis": kpi_row(metrics, snap.pipeline),
        "feed": alert_feed(service.get_recent_alerts()),
        "donut": severity_donut(active),
        "trend": trend_chart(service.get_alert_trend(TREND_HOURS)),
        "status": status_row(service.get_status_badges()),
    }


def layout() -> html.Div:
    """Page layout, filled from the latest snapshot at the time of navigation."""
    content = build_content(_service())
    return html.Div(
        [
            dcc.Store(id="ov-version", data=content["version"]),
            page_header("Overview", "System-wide health and the latest alerts · times in UTC"),
            html.Div(content["status"], id="ov-status", className="mb-2"),
            html.Div(content["kpis"], id="ov-kpis"),
            dbc.Row(
                [
                    dbc.Col(
                        card_with_title("Live alert feed", html.Div(content["feed"], id="ov-feed")),
                        xs=12,
                        xl=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card_with_title(
                            "Active alerts by severity", graph("ov-donut", content["donut"])
                        ),
                        xs=12,
                        md=6,
                        xl=5,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    card_with_title(
                        f"Alerts per hour, last {TREND_HOURS} h",
                        graph("ov-trend", content["trend"]),
                    ),
                    xs=12,
                    className="mb-3",
                )
            ),
        ],
        id="ov-page",
    )


@dash.callback(
    Output("ov-status", "children"),
    Output("ov-kpis", "children"),
    Output("ov-feed", "children"),
    Output("ov-donut", "figure"),
    Output("ov-trend", "figure"),
    Output("ov-version", "data"),
    Input("live-interval", "n_intervals"),
    State("ov-version", "data"),
    prevent_initial_call=True,
)
def refresh_overview(_n: int | None, shown_version: int | None):
    """Re-render the page on each tick, unless the snapshot is the one already shown."""
    service = _service()
    if shown_version is not None and service.snapshot().version == shown_version:
        return (no_update,) * 6
    content = build_content(service)
    return (
        content["status"],
        content["kpis"],
        content["feed"],
        content["donut"],
        content["trend"],
        content["version"],
    )
