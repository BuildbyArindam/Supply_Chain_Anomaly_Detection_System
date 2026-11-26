"""Overview page (Day 19 builds the KPI cards, alert feed and charts)."""

from __future__ import annotations

import dash
from dash import Input, Output, dcc, html  # noqa: F401

from src.dashboard.components.layout import coming_soon, page_header
from src.dashboard.utils import format_value_with_unit

dash.register_page(
    __name__, path="/", name="Overview", title="Overview · Supply Chain Anomaly Monitor", order=0
)


def _stat(label: str, value: str, ident: str) -> html.Div:
    return html.Div(
        [html.Div(label, className="small text-muted"), html.Div(value, id=ident, className="h4")],
        className="me-4",
    )


layout = html.Div(
    [
        page_header("Overview", "System-wide health and the latest alerts"),
        html.Div(
            [
                _stat("Alerts today", "—", "ov-alerts-today"),
                _stat("Active critical", "—", "ov-active-critical"),
                _stat("False positives (24 h)", "—", "ov-fp-rate"),
                _stat("Detection p99", "—", "ov-latency"),
            ],
            className="d-flex flex-wrap mb-3",
            id="ov-data-layer",
        ),
        coming_soon(19, "KPI cards, the alert feed and the severity charts"),
    ]
)


@dash.callback(
    Output("ov-alerts-today", "children"),
    Output("ov-active-critical", "children"),
    Output("ov-fp-rate", "children"),
    Output("ov-latency", "children"),
    Input("metrics-store", "data"),
)
def show_data_layer(metrics: dict | None):
    """Render the headline numbers straight from the shared metrics store."""
    if not metrics:
        return "—", "—", "—", "—"
    fp = metrics.get("fp_rate_24h")
    return (
        str(metrics.get("total_alerts_today", "—")),
        str(metrics.get("active_critical", "—")),
        "—" if fp is None else f"{fp:.0%}",
        format_value_with_unit(metrics.get("avg_latency_p99"), "ms", 0),
    )
