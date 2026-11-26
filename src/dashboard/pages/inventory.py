"""Inventory page (Day 20 builds it)."""

from __future__ import annotations

import dash
from dash import html

from src.dashboard.components.layout import coming_soon, page_header

dash.register_page(
    __name__,
    path="/inventory",
    name="Inventory",
    title="Inventory · Supply Chain Anomaly Monitor",
    order=3,
)

layout = html.Div(
    [
        page_header("Inventory", "Stock levels and transactions"),
        coming_soon(20, "Stock gauges and the transaction waterfall"),
    ]
)
