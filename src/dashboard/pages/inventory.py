"""Inventory page (Day 20): stock gauges, a transaction waterfall and the anomaly table.

The gauges and the table come from one shared read of every SKU's last hour of transactions
(``DashboardDataService.get_inventory_series``, cached for a few seconds). Charts are only sent
to the browser when they changed. The layout is a function, so every callback uses
``prevent_initial_call=True``.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, State, dcc, html, no_update

from src.dashboard.components.inventory_widgets import (
    WINDOW_SECONDS,
    anomaly_table,
    inventory_anomaly_rows,
    resolve_sku_selection,
    stock_gauges,
    top_active_skus,
    transaction_buckets,
    waterfall_figure,
)
from src.dashboard.components.layout import page_header
from src.dashboard.components.overview_widgets import card_with_title
from src.dashboard.components.theme import MUTED_TEXT
from src.dashboard.data_service import DashboardDataService
from src.dashboard.utils import figure_signature

dash.register_page(
    __name__,
    path="/inventory",
    name="Inventory",
    title="Inventory · Supply Chain Anomaly Monitor",
    order=3,
)


def _service() -> DashboardDataService:
    """Return the dashboard's data service (``create_dash_app`` attaches it to the app)."""
    return flask.current_app.extensions["scad_service"]


def _gauge_figure(service: DashboardDataService):
    entities = service.get_entities("inventory")
    alerts = service.query_alerts(entity_types=["inventory_sku"], window_hours=1)
    rows = top_active_skus(entities, service.get_inventory_series(WINDOW_SECONDS), alerts)
    return stock_gauges(rows)


def _waterfall(service: DashboardDataService, sku: str | None):
    if not sku:
        return waterfall_figure([])
    entity = next((e for e in service.get_entities("inventory") if e["entity_id"] == sku), {})
    details = entity.get("details") or {}
    points = service.get_inventory_series(WINDOW_SECONDS).get(sku, [])
    alerts = service.query_alerts(entity_id=sku, window_hours=1)
    return waterfall_figure(
        transaction_buckets(points, alerts),
        sku_id=sku,
        reorder_point=details.get("reorder_point"),
        min_stock=details.get("min_stock"),
    )


def _table(service: DashboardDataService):
    alerts = service.query_alerts(entity_types=["inventory_sku"], window_hours=24)
    return anomaly_table(
        inventory_anomaly_rows(alerts, service.get_inventory_series(WINDOW_SECONDS))
    )


def _dropdown(ident: str, label: str, options: Any, value: Any) -> dbc.Col:
    return dbc.Col(
        [
            dbc.Label(label, html_for=ident, className="small"),
            dcc.Dropdown(
                id=ident,
                options=options,
                value=value,
                clearable=False,
                className="dash-dark-dropdown",
            ),
        ],
        xs=12,
        md=4,
        className="mb-2",
    )


def layout() -> html.Div:
    """Page layout, filled from the latest snapshot at the time of navigation."""
    service = _service()
    sel = resolve_sku_selection(service.get_entities("inventory"))
    gauges = _gauge_figure(service)
    waterfall = _waterfall(service, sel["sku"])
    return html.Div(
        [
            dcc.Store(id="inv-gauge-sig"),
            dcc.Store(id="inv-wf-sig"),
            page_header(
                "Inventory",
                "Stock levels and transactions of the busiest SKUs · times in UTC",
            ),
            dbc.Row(
                dbc.Col(
                    card_with_title(
                        "Stock levels · 10 most active SKUs (last hour)",
                        [
                            html.Div(
                                "Bands: red below the minimum, amber down to the reorder point "
                                "(the line), green above it.",
                                className="small",
                                style={"color": MUTED_TEXT},
                            ),
                            dcc.Graph(
                                id="inv-gauges",
                                figure=gauges,
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                        ident="inv-gauge-card",
                    ),
                    xs=12,
                    className="mb-3",
                )
            ),
            dbc.Row(
                [
                    _dropdown(
                        "inv-warehouse", "Warehouse", sel["warehouse_options"], sel["warehouse"]
                    ),
                    _dropdown("inv-zone", "Zone", sel["zone_options"], sel["zone"]),
                    _dropdown("inv-sku", "SKU", sel["sku_options"], sel["sku"]),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card_with_title(
                            "Transaction waterfall · last hour",
                            dcc.Graph(
                                id="inv-waterfall",
                                figure=waterfall,
                                config={"displayModeBar": False, "responsive": True},
                            ),
                            ident="inv-waterfall-card",
                        ),
                        xs=12,
                        xl=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card_with_title(
                            "Inventory anomalies · last 24 hours",
                            html.Div(_table(service), id="inv-anomalies"),
                            ident="inv-anomaly-card",
                        ),
                        xs=12,
                        xl=5,
                        className="mb-3",
                    ),
                ]
            ),
        ],
        id="inv-page",
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("inv-warehouse", "options"),
    Output("inv-warehouse", "value"),
    Output("inv-zone", "options"),
    Output("inv-zone", "value"),
    Output("inv-sku", "options"),
    Output("inv-sku", "value"),
    Input("inv-warehouse", "value"),
    Input("inv-zone", "value"),
    Input("live-interval", "n_intervals"),
    State("inv-sku", "value"),
    State("inv-warehouse", "options"),
    State("inv-zone", "options"),
    State("inv-sku", "options"),
    prevent_initial_call=True,
)
def update_selectors(
    warehouse: str | None,
    zone: str | None,
    _n: int | None,
    sku: str | None,
    warehouse_options: list[dict[str, str]] | None,
    zone_options: list[dict[str, str]] | None,
    sku_options: list[dict[str, str]] | None,
):
    """Keep the three dropdowns consistent and follow SKUs that appear or disappear."""
    sel = resolve_sku_selection(_service().get_entities("inventory"), warehouse, zone, sku)

    def same(new: Any, old: Any) -> Any:
        return no_update if new == old else new

    return (
        same(sel["warehouse_options"], warehouse_options),
        same(sel["warehouse"], warehouse),
        same(sel["zone_options"], zone_options),
        same(sel["zone"], zone),
        same(sel["sku_options"], sku_options),
        same(sel["sku"], sku),
    )


@dash.callback(
    Output("inv-gauges", "figure"),
    Output("inv-gauge-sig", "data"),
    Output("inv-anomalies", "children"),
    Input("live-interval", "n_intervals"),
    State("inv-gauge-sig", "data"),
    prevent_initial_call=True,
)
def refresh_gauges(_n: int | None, previous: str | None):
    """Redraw the gauges (only when they changed) and the anomaly table."""
    service = _service()
    figure = _gauge_figure(service)
    signature = figure_signature(figure)
    return (no_update if signature == previous else figure), signature, _table(service)


@dash.callback(
    Output("inv-waterfall", "figure"),
    Output("inv-wf-sig", "data"),
    Input("live-interval", "n_intervals"),
    Input("inv-sku", "value"),
    State("inv-wf-sig", "data"),
    prevent_initial_call=True,
)
def refresh_waterfall(_n: int | None, sku: str | None, previous: str | None):
    """Redraw the waterfall of the chosen SKU (only when it changed)."""
    figure = _waterfall(_service(), sku)
    signature = f"{sku}:{figure_signature(figure)}"
    return (no_update if signature == previous else figure), signature
