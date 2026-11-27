"""Warehouse page (Day 19): live sensor chart, zone heatmap and anomaly detail panel.

How the live chart stays cheap
------------------------------
The chart is *built* once per sensor choice (``rebuild_chart``). After that each ``live-interval``
tick runs ``poll_increment``, which sends only the readings newer than the browser's cursor into
``wh-delta``; a clientside callback turns them into ``extendData`` so Plotly appends the points
in place (``Plotly.extendTraces``) instead of redrawing the figure. A tick with nothing new sends
nothing at all.

The page layout is a function (first paint already has data), so every callback uses
``prevent_initial_call=True``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import dash
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, State, clientside_callback, dcc, html, no_update

from src.dashboard.components.layout import page_header
from src.dashboard.components.overview_widgets import card_with_title
from src.dashboard.components.theme import MUTED_TEXT
from src.dashboard.components.warehouse_widgets import (
    EXTEND_JS,
    LOOKBACK_SECONDS,
    SENSOR_KINDS,
    TRACE_ANOMALIES,
    alert_points,
    anomaly_card,
    chart_window_start,
    make_cursor,
    match_alert,
    new_points,
    resolve_selection,
    series_note,
    timeseries_figure,
    unit_symbol,
    zone_heatmap,
)
from src.dashboard.data_service import DashboardDataService

dash.register_page(
    __name__,
    path="/warehouse",
    name="Warehouse",
    title="Warehouse · Supply Chain Anomaly Monitor",
    order=1,
)

MAX_ALERTS_ON_CHART = 200
DEFAULT_KIND = "temperature"


def _service() -> DashboardDataService:
    """Return the dashboard's data service (``create_dash_app`` attaches it to the app)."""
    return flask.current_app.extensions["scad_service"]


def _chart_inputs(
    service: DashboardDataService, entity_id: str | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Series and alerts of one sensor (``(None, [])`` when no sensor is selected)."""
    if not entity_id:
        return None, []
    series = service.get_sensor_timeseries(entity_id, LOOKBACK_SECONDS)
    since = chart_window_start(series, service.snapshot().reference_now)
    return series, service.get_entity_alerts(entity_id, since=since, limit=MAX_ALERTS_ON_CHART)


def _chart_state(service: DashboardDataService, entity_id: str | None):
    """Figure, cursor and note of a freshly built chart."""
    series, alerts = _chart_inputs(service, entity_id)
    figure = timeseries_figure(series, alerts, entity_id=entity_id, now=datetime.now(UTC))
    cursor_alerts = alert_points(alerts, series)
    note = series_note(series) if entity_id else "No warehouse sensors are reporting yet."
    return figure, make_cursor(entity_id, series, cursor_alerts), note


def layout() -> html.Div:
    """Page layout, filled from the latest snapshot at the time of navigation."""
    service = _service()
    entities = service.get_entities("warehouse")
    sel = resolve_selection(entities)
    figure, cursor, note = _chart_state(service, sel["sensor"])
    return html.Div(
        [
            dcc.Store(id="wh-cursor", data=cursor),
            dcc.Store(id="wh-delta"),
            page_header("Warehouse", "Live temperature and humidity per sensor · times in UTC"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Warehouse", html_for="wh-warehouse", className="small"),
                            dcc.Dropdown(
                                id="wh-warehouse",
                                options=sel["warehouse_options"],
                                value=sel["warehouse"],
                                clearable=False,
                                className="dash-dark-dropdown",
                            ),
                        ],
                        xs=12,
                        md=4,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Zone", html_for="wh-zone", className="small"),
                            dcc.Dropdown(
                                id="wh-zone",
                                options=sel["zone_options"],
                                value=sel["zone"],
                                clearable=False,
                                className="dash-dark-dropdown",
                            ),
                        ],
                        xs=12,
                        md=4,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Sensor", html_for="wh-sensor", className="small"),
                            dcc.Dropdown(
                                id="wh-sensor",
                                options=sel["sensor_options"],
                                value=sel["sensor"],
                                clearable=False,
                                className="dash-dark-dropdown",
                            ),
                        ],
                        xs=12,
                        md=4,
                        className="mb-2",
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card_with_title(
                            "Live readings · last 5 minutes",
                            [
                                html.Div(
                                    note,
                                    id="wh-note",
                                    className="small mb-1",
                                    style={"color": MUTED_TEXT, "minHeight": "1.2em"},
                                ),
                                dcc.Graph(
                                    id="wh-graph",
                                    figure=figure,
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "380px"},
                                ),
                            ],
                            ident="wh-chart-card",
                        ),
                        xs=12,
                        xl=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card_with_title(
                            "Zone heatmap · current readings",
                            [
                                dbc.RadioItems(
                                    id="wh-kind",
                                    options=[
                                        {"label": k.title(), "value": k} for k in SENSOR_KINDS
                                    ],
                                    value=DEFAULT_KIND,
                                    inline=True,
                                    className="mb-1 small",
                                ),
                                dcc.Graph(
                                    id="wh-heatmap",
                                    figure=zone_heatmap(entities, sel["warehouse"], DEFAULT_KIND),
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "340px"},
                                ),
                            ],
                            ident="wh-heatmap-card",
                        ),
                        xs=12,
                        xl=5,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(dbc.Col(html.Div(anomaly_card(None), id="wh-detail"), xs=12, className="mb-3")),
        ],
        id="wh-page",
    )


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("wh-warehouse", "options"),
    Output("wh-warehouse", "value"),
    Output("wh-zone", "options"),
    Output("wh-zone", "value"),
    Output("wh-sensor", "options"),
    Output("wh-sensor", "value"),
    Input("wh-warehouse", "value"),
    Input("wh-zone", "value"),
    Input("live-interval", "n_intervals"),
    State("wh-sensor", "value"),
    State("wh-warehouse", "options"),
    State("wh-zone", "options"),
    State("wh-sensor", "options"),
    prevent_initial_call=True,
)
def update_selectors(
    warehouse: str | None,
    zone: str | None,
    _n: int | None,
    sensor: str | None,
    warehouse_options: list[dict[str, str]] | None,
    zone_options: list[dict[str, str]] | None,
    sensor_options: list[dict[str, str]] | None,
):
    """Keep the three dropdowns consistent and follow sensors that appear or disappear."""
    sel = resolve_selection(_service().get_entities("warehouse"), warehouse, zone, sensor)

    def same(new: Any, old: Any) -> Any:
        return no_update if new == old else new

    return (
        same(sel["warehouse_options"], warehouse_options),
        same(sel["warehouse"], warehouse),
        same(sel["zone_options"], zone_options),
        same(sel["zone"], zone),
        same(sel["sensor_options"], sensor_options),
        same(sel["sensor"], sensor),
    )


# --------------------------------------------------------------------------- #
# Live chart
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("wh-graph", "figure"),
    Output("wh-cursor", "data"),
    Output("wh-note", "children"),
    Input("wh-sensor", "value"),
    prevent_initial_call=True,
)
def rebuild_chart(entity_id: str | None):
    """Build the full chart when another sensor is chosen."""
    return _chart_state(_service(), entity_id)


@dash.callback(
    Output("wh-delta", "data"),
    Output("wh-note", "children", allow_duplicate=True),
    Input("live-interval", "n_intervals"),
    State("wh-sensor", "value"),
    State("wh-cursor", "data"),
    prevent_initial_call=True,
)
def poll_increment(_n: int | None, entity_id: str | None, cursor: dict[str, Any] | None):
    """Send the readings and alerts newer than the browser's cursor (nothing when none)."""
    if not entity_id:
        return no_update, no_update
    series, alerts = _chart_inputs(_service(), entity_id)
    delta = new_points(series, alerts, cursor, entity_id)
    if delta is None:
        return no_update, no_update
    first_data = not (cursor or {}).get("t")
    return delta, ("" if first_data else no_update)


clientside_callback(
    EXTEND_JS,
    Output("wh-graph", "extendData"),
    Output("wh-cursor", "data", allow_duplicate=True),
    Input("wh-delta", "data"),
    State("wh-cursor", "data"),
    prevent_initial_call=True,
)


# --------------------------------------------------------------------------- #
# Heatmap and detail panel
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("wh-heatmap", "figure"),
    Input("live-interval", "n_intervals"),
    Input("wh-warehouse", "value"),
    Input("wh-kind", "value"),
    prevent_initial_call=True,
)
def update_heatmap(_n: int | None, warehouse: str | None, kind: str | None):
    """Redraw the floor grid from the latest reading of every sensor."""
    entities = _service().get_entities("warehouse")
    return zone_heatmap(entities, warehouse, kind or DEFAULT_KIND)


@dash.callback(
    Output("wh-detail", "children"),
    Input("wh-graph", "clickData"),
    State("wh-sensor", "value"),
    prevent_initial_call=True,
)
def show_anomaly(click: dict[str, Any] | None, entity_id: str | None):
    """Show the alert behind a clicked red marker (clicks on the line are ignored)."""
    # With the unified hover label a click reports every trace at that x, so pick the marker.
    marker = next(
        (p for p in (click or {}).get("points") or [] if p.get("curveNumber") == TRACE_ANOMALIES),
        None,
    )
    if marker is None or not entity_id:
        return no_update
    service = _service()
    alerts = service.get_entity_alerts(entity_id, limit=1000)
    found = match_alert(alerts, marker.get("x"))
    if found is None:
        return anomaly_card(None)
    full = service.get_alert(str(found["event_id"])) or found
    series = service.get_sensor_timeseries(entity_id, LOOKBACK_SECONDS)
    return anomaly_card(full, unit_symbol((series or {}).get("unit")))
