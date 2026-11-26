"""Alerts page (Day 20 builds it)."""

from __future__ import annotations

import dash
from dash import html

from src.dashboard.components.layout import coming_soon, page_header

dash.register_page(
    __name__, path="/alerts", name="Alerts", title="Alerts · Supply Chain Anomaly Monitor", order=4
)

layout = html.Div(
    [
        page_header("Alerts", "Filter, review and give feedback on alerts"),
        coming_soon(20, "The alerts table, detail modal and CSV export"),
    ]
)
