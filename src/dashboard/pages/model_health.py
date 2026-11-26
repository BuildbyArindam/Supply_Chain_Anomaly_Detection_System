"""Model Health page (Day 21 builds it)."""

from __future__ import annotations

import dash
from dash import html

from src.dashboard.components.layout import coming_soon, page_header

dash.register_page(
    __name__,
    path="/model-health",
    name="Model Health",
    title="Model Health · Supply Chain Anomaly Monitor",
    order=5,
)

layout = html.Div(
    [
        page_header("Model Health", "Model quality, feedback and retraining"),
        coming_soon(21, "Model gauges, feedback analytics and retraining history"),
    ]
)
