"""Vehicles page (Day 20): live fleet map, vehicle detail panel and fleet summary table.

Every ``live-interval`` tick rebuilds the map from the latest snapshot. The map is only sent to
the browser when it changed, and its ``uirevision`` is the focused vehicle, so panning and zooming
survive the updates. The page layout is a function (the first paint already has data), so every
callback uses ``prevent_initial_call=True``.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, State, dcc, html, no_update

from src.dashboard.components.layout import page_header
from src.dashboard.components.overview_widgets import card_with_title
from src.dashboard.components.theme import MUTED_TEXT
from src.dashboard.components.vehicle_widgets import (
    ALL_VEHICLES,
    MAX_TRAILS,
    TRAIL_SECONDS,
    build_fleet,
    deviation_markers,
    fleet_map,
    fleet_table,
    vehicle_info_card,
)
from src.dashboard.data_service import DashboardDataService
from src.dashboard.utils import figure_signature

dash.register_page(
    __name__,
    path="/vehicles",
    name="Vehicles",
    title="Vehicles · Supply Chain Anomaly Monitor",
    order=2,
)


def _service() -> DashboardDataService:
    """Return the dashboard's data service (``create_dash_app`` attaches it to the app)."""
    return flask.current_app.extensions["scad_service"]


def _state(service: DashboardDataService) -> dict[str, Any]:
    """Everything the page shows, read from the latest snapshot."""
    snap = service.snapshot()
    entities = service.get_entities("vehicle")
    alerts = service.query_alerts(entity_types=["vehicle"], window_hours=24)
    fleet = build_fleet(entities, alerts, snap.reference_now)
    routes = {}
    for vehicle in fleet:
        route = service.get_route(vehicle.get("route_id"))
        if route is not None:
            routes[str(vehicle["route_id"])] = route
    trails = {}
    for vehicle in fleet[:MAX_TRAILS]:
        series = service.get_sensor_timeseries(vehicle["vehicle_id"], TRAIL_SECONDS)
        if series and series.get("points"):
            trails[vehicle["vehicle_id"]] = series["points"]
    return {
        "fleet": fleet,
        "routes": routes,
        "trails": trails,
        "marks": deviation_markers(alerts, trails),
        "now": snap.reference_now,
    }


def _focus_options(fleet: list[dict[str, Any]]) -> list[dict[str, str]]:
    ordered = sorted(fleet, key=lambda v: v["vehicle_id"])
    return [{"label": "Whole fleet", "value": ALL_VEHICLES}] + [
        {"label": v["vehicle_id"], "value": v["vehicle_id"]} for v in ordered
    ]


def _valid_focus(focus: str | None, fleet: list[dict[str, Any]]) -> str:
    return focus if focus and any(v["vehicle_id"] == focus for v in fleet) else ALL_VEHICLES


def _selected(fleet: list[dict[str, Any]], focus: str) -> dict[str, Any] | None:
    return next((v for v in fleet if v["vehicle_id"] == focus), None)


def layout() -> html.Div:
    """Page layout, filled from the latest snapshot at the time of navigation."""
    service = _service()
    state = _state(service)
    fleet = state["fleet"]
    focus = ALL_VEHICLES
    figure = fleet_map(fleet, state["routes"], state["marks"], state["trails"], focus)
    return html.Div(
        [
            dcc.Store(id="veh-map-sig"),
            page_header(
                "Vehicles",
                "Live fleet positions against their planned routes · times in UTC · "
                "✕ marks where an alert was raised",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Focus on", html_for="veh-focus", className="small"),
                        dcc.Dropdown(
                            id="veh-focus",
                            options=_focus_options(fleet),
                            value=focus,
                            clearable=False,
                            className="dash-dark-dropdown",
                        ),
                    ],
                    xs=12,
                    md=4,
                    className="mb-2",
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card_with_title(
                            "Live map",
                            dcc.Graph(
                                id="veh-map",
                                figure=figure,
                                config={"displayModeBar": False, "responsive": True, "scrollZoom": True},
                                style={"height": "520px"},
                            ),
                            ident="veh-map-card",
                        ),
                        xs=12,
                        xl=8,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card_with_title(
                            "Vehicle details",
                            html.Div(
                                vehicle_info_card(
                                    _selected(fleet, focus), None, state["now"]
                                ),
                                id="veh-info",
                            ),
                            ident="veh-info-panel",
                        ),
                        xs=12,
                        xl=4,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    card_with_title(
                        "Fleet summary",
                        html.Div(fleet_table(fleet, state["now"]), id="veh-fleet"),
                        ident="veh-fleet-card",
                    ),
                    xs=12,
                    className="mb-3",
                )
            ),
            html.Div(
                "Colours are paired with text labels: green On route, red Deviation, amber Delayed.",
                className="small",
                style={"color": MUTED_TEXT},
            ),
        ],
        id="veh-page",
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("veh-focus", "options"),
    Output("veh-focus", "value"),
    Input("live-interval", "n_intervals"),
    Input("veh-map", "clickData"),
    State("veh-focus", "value"),
    State("veh-focus", "options"),
    prevent_initial_call=True,
)
def update_focus(
    _n: int | None,
    click: dict[str, Any] | None,
    focus: str | None,
    current_options: list[dict[str, str]] | None,
):
    """Keep the vehicle list current and move the focus to a clicked marker."""
    fleet = build_fleet(
        _service().get_entities("vehicle"), [], _service().snapshot().reference_now
    )
    options = _focus_options(fleet)
    clicked = None
    if dash.ctx.triggered_id == "veh-map":
        point = ((click or {}).get("points") or [{}])[0]
        clicked = point.get("customdata")
        if isinstance(clicked, list):
            clicked = clicked[0] if clicked else None
    chosen = _valid_focus(clicked or focus, fleet)
    return (
        no_update if options == current_options else options,
        no_update if chosen == focus else chosen,
    )


@dash.callback(
    Output("veh-map", "figure"),
    Output("veh-map-sig", "data"),
    Output("veh-info", "children"),
    Output("veh-fleet", "children"),
    Input("live-interval", "n_intervals"),
    Input("veh-focus", "value"),
    State("veh-map-sig", "data"),
    prevent_initial_call=True,
)
def refresh_page(_n: int | None, focus: str | None, previous_sig: str | None):
    """Redraw the map, the detail panel and the fleet table from the latest snapshot."""
    service = _service()
    state = _state(service)
    fleet = state["fleet"]
    chosen = _valid_focus(focus, fleet)
    figure = fleet_map(fleet, state["routes"], state["marks"], state["trails"], chosen)
    signature = figure_signature(figure)
    vehicle = _selected(fleet, chosen)
    route = state["routes"].get(str(vehicle["route_id"])) if vehicle else None
    return (
        no_update if signature == previous_sig else figure,
        signature,
        vehicle_info_card(vehicle, route, state["now"]),
        fleet_table(fleet, state["now"]),
    )
