"""Vehicles page (Day 20 builds it)."""

from __future__ import annotations

import dash
from dash import html

from src.dashboard.components.layout import coming_soon, page_header

dash.register_page(
    __name__,
    path="/vehicles",
    name="Vehicles",
    title="Vehicles · Supply Chain Anomaly Monitor",
    order=2,
)

layout = html.Div(
    [
        page_header("Vehicles", "Fleet positions, routes and deviations"),
        coming_soon(20, "The live map and the fleet table"),
    ]
)
