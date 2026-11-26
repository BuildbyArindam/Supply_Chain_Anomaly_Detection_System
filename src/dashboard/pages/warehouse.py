"""Warehouse page (Day 19 builds it)."""

from __future__ import annotations

import dash
from dash import html

from src.dashboard.components.layout import coming_soon, page_header

dash.register_page(
    __name__,
    path="/warehouse",
    name="Warehouse",
    title="Warehouse · Supply Chain Anomaly Monitor",
    order=1,
)

layout = html.Div(
    [
        page_header("Warehouse", "Live temperature and humidity per sensor"),
        coming_soon(19, "Sensor charts and the zone heatmap"),
    ]
)
