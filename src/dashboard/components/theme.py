"""Colour palette and the shared plotly template of the dashboard (Day 18).

Every chart goes through :func:`apply_theme`, so the whole dashboard looks alike and a palette
change is a one-file edit. The page itself is styled by the Bootstrap ``DARKLY`` theme; the
constants here are chosen to sit on its background (``#222``).

Severity colours are *status* colours: they always travel with a text label (see
:func:`src.dashboard.utils.severity_badge`), never colour alone.
"""

from __future__ import annotations

from typing import Any, Final

import plotly.graph_objects as go
import plotly.io as pio

CRITICAL_RED: Final = "#e74c3c"
WARNING_AMBER: Final = "#f39c12"
INFO_BLUE: Final = "#3498db"
NORMAL_GREEN: Final = "#00bc8c"
EMERGENCY_PURPLE: Final = "#9b59b6"
BACKGROUND_DARK: Final = "#222222"
PANEL_DARK: Final = "#2b2b2b"
GRID_COLOR: Final = "#3d3d3d"
TEXT_COLOR: Final = "#ebebeb"
MUTED_TEXT: Final = "#9a9a9a"
FONT_FAMILY: Final = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

#: Severity level name -> colour (names match ``Severity`` in the ensemble detector).
SEVERITY_COLORS: Final[dict[str, str]] = {
    "INFO": INFO_BLUE,
    "WARNING": WARNING_AMBER,
    "CRITICAL": CRITICAL_RED,
    "EMERGENCY": EMERGENCY_PURPLE,
}
#: Severity names from least to most severe.
SEVERITY_ORDER: Final[tuple[str, ...]] = ("INFO", "WARNING", "CRITICAL", "EMERGENCY")
#: Bootstrap contextual colour of each severity (for badges and borders).
SEVERITY_BOOTSTRAP: Final[dict[str, str]] = {
    "INFO": "info",
    "WARNING": "warning",
    "CRITICAL": "danger",
    "EMERGENCY": "danger",
}

TEMPLATE_NAME: Final = "scad_dark"


def build_template() -> go.layout.Template:
    """Return the plotly template: dark background, quiet grid, brand font."""
    axis: dict[str, Any] = {
        "gridcolor": GRID_COLOR,
        "zerolinecolor": GRID_COLOR,
        "linecolor": GRID_COLOR,
        "tickfont": {"color": MUTED_TEXT},
        "title": {"font": {"color": MUTED_TEXT}},
    }
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=BACKGROUND_DARK,
            plot_bgcolor=BACKGROUND_DARK,
            font={"family": FONT_FAMILY, "color": TEXT_COLOR, "size": 13},
            colorway=[INFO_BLUE, NORMAL_GREEN, WARNING_AMBER, CRITICAL_RED, EMERGENCY_PURPLE],
            xaxis=axis,
            yaxis=axis,
            legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT_COLOR}},
            hoverlabel={"bgcolor": PANEL_DARK, "font": {"family": FONT_FAMILY}},
            margin={"l": 48, "r": 16, "t": 40, "b": 36},
        )
    )


pio.templates[TEMPLATE_NAME] = build_template()


def apply_theme(fig: go.Figure, **layout: Any) -> go.Figure:
    """Style ``fig`` with the dashboard template and return it.

    Args:
        fig: Any plotly figure.
        **layout: Extra ``update_layout`` arguments (title, height, ...) applied after the theme.
    """
    fig.update_layout(template=TEMPLATE_NAME, **layout)
    return fig


def empty_figure(message: str = "No data available", height: int = 260) -> go.Figure:
    """Return a themed placeholder figure that shows ``message`` (graceful empty state)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 15, "color": MUTED_TEXT},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, height=height)
