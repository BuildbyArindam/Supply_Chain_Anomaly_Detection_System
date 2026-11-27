"""Building blocks of the Overview page (Day 19).

Pure functions that turn :class:`~src.dashboard.data_service.DashboardDataService` output into
Dash components and plotly figures. They live outside ``pages/`` so that they can be unit-tested
without registering a page, and so the page module only holds layout and callbacks.

Colour is never the only signal: severities carry a text badge, status pills carry a word, and
KPI cards print their value.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from src.dashboard.components.theme import (
    BACKGROUND_DARK,
    CRITICAL_RED,
    MUTED_TEXT,
    NORMAL_GREEN,
    PANEL_DARK,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    TEXT_COLOR,
    WARNING_AMBER,
    apply_theme,
    empty_figure,
)
from src.dashboard.utils import (
    PLACEHOLDER,
    format_timestamp,
    format_value_with_unit,
    parse_timestamp,
    severity_badge,
)

FEED_SIZE: Final = 20
NEUTRAL_GREY: Final = "#6c757d"
#: Model-health thresholds shared with the Day 21 gauges: green above 85 %, amber 70-85 %.
HEALTH_GREEN: Final = 0.85
HEALTH_AMBER: Final = 0.70
DEFAULT_LATENCY_TARGET_MS: Final = 500.0

#: Status-row state -> (Bootstrap colour, word shown next to the component name).
STATE_STYLE: Final[dict[str, tuple[str, str]]] = {
    "ok": ("success", "OK"),
    "warn": ("warning", "Degraded"),
    "error": ("danger", "Down"),
    "unknown": ("secondary", "Unknown"),
}
STATE_COLORS: Final[dict[str, str]] = {
    "ok": NORMAL_GREEN,
    "warn": WARNING_AMBER,
    "error": CRITICAL_RED,
    "unknown": NEUTRAL_GREY,
}


def hex_to_rgba(color: str, alpha: float) -> str:
    """Return ``rgba(r, g, b, alpha)`` for a ``#rrggbb`` colour."""
    value = color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def kpi_card(ident: str, label: str, value: str, color: str, caption: str = "") -> dbc.Card:
    """One KPI scorecard: a coloured border, the label, a large value and a caption.

    Args:
        ident: Element id of the value (tests and the smoke test read it).
        label: What is measured.
        value: Formatted value (``—`` when unknown).
        color: Border colour (a ``#rrggbb`` status colour).
        caption: Small muted line under the value.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="small text-uppercase", style={"color": MUTED_TEXT}),
                html.Div(value, id=ident, className="display-6 fw-semibold"),
                html.Div(caption or " ", className="small", style={"color": MUTED_TEXT}),
            ]
        ),
        id=f"{ident}-card",
        className="kpi-card h-100",
        style={"border": f"1px solid {color}", "borderLeft": f"6px solid {color}"},
    )


def health_color(score: float | None) -> str:
    """Status colour of a model-health score in ``[0, 1]``."""
    if score is None:
        return NEUTRAL_GREY
    return (
        NORMAL_GREEN
        if score > HEALTH_GREEN
        else WARNING_AMBER if score >= HEALTH_AMBER else CRITICAL_RED
    )


def kpi_row(metrics: Mapping[str, Any], pipeline: Mapping[str, Any] | None = None) -> dbc.Row:
    """Build the four KPI cards: Active Alerts, Critical Events (24 h), Model Health, Avg Latency.

    Args:
        metrics: ``DashboardDataService.get_summary_metrics()``.
        pipeline: The snapshot's pipeline stats, only for the latency target.
    """
    active = metrics.get("active_alerts")
    critical_active = metrics.get("active_critical") or 0
    critical_24h = metrics.get("critical_24h")
    score = metrics.get("model_health_score")
    p99 = metrics.get("avg_latency_p99")
    target = float((pipeline or {}).get("latency_target_ms") or DEFAULT_LATENCY_TARGET_MS)

    if active is None:
        active_color = NEUTRAL_GREY
    elif critical_active:
        active_color = CRITICAL_RED
    else:
        active_color = WARNING_AMBER if active else NORMAL_GREEN
    critical_color = (
        NEUTRAL_GREY if critical_24h is None else CRITICAL_RED if critical_24h else NORMAL_GREEN
    )
    latency_color = NEUTRAL_GREY if p99 is None else NORMAL_GREEN if p99 <= target else CRITICAL_RED

    cards = [
        kpi_card(
            "ov-active-alerts",
            "Active alerts",
            PLACEHOLDER if active is None else f"{active:,}",
            active_color,
            f"{critical_active} critical or worse" if active is not None else "",
        ),
        kpi_card(
            "ov-critical-24h",
            "Critical events (24 h)",
            PLACEHOLDER if critical_24h is None else f"{critical_24h:,}",
            critical_color,
            "critical and emergency",
        ),
        kpi_card(
            "ov-model-health",
            "Model health",
            PLACEHOLDER if score is None else f"{score:.0%}",
            health_color(score),
            "availability, feedback, latency, drift",
        ),
        kpi_card(
            "ov-latency",
            "Avg detection latency",
            format_value_with_unit(p99, "ms", 0),
            latency_color,
            f"p99, target {target:.0f} ms" if p99 is not None else "no streaming engine data",
        ),
    ]
    return dbc.Row(
        [dbc.Col(card, xs=12, sm=6, xl=3, className="mb-3") for card in cards],
        id="ov-kpi-row",
    )


# --------------------------------------------------------------------------- #
# Alert feed
# --------------------------------------------------------------------------- #
def _format_alert_value(alert: Mapping[str, Any]) -> str:
    value = alert.get("raw_value")
    if value is None:
        return PLACEHOLDER
    return format_value_with_unit(value, None, 2)


def alert_feed(alerts: Sequence[Mapping[str, Any]]) -> html.Div:
    """Build the live feed: a table of the newest alerts, rows tinted and edged by severity.

    The wrapper has a fixed height and scrolls; the newest alert is the first row, so the
    visible part always shows the latest alerts.
    """
    if not alerts:
        return html.Div(
            "No alerts yet.", id="ov-feed-empty", className="p-3", style={"color": MUTED_TEXT}
        )
    rows = []
    for alert in alerts[:FEED_SIZE]:
        name = str(alert.get("severity") or "").upper()
        color = SEVERITY_COLORS.get(name, NEUTRAL_GREY)
        closed = alert.get("false_positive_marked") or alert.get("is_resolved")
        style: dict[str, Any] = {
            "borderLeft": f"4px solid {color}",
            "backgroundColor": hex_to_rgba(color, 0.12),
        }
        if closed:
            style["opacity"] = 0.6
        rows.append(
            html.Tr(
                [
                    html.Td(format_timestamp(alert.get("timestamp"))),
                    html.Td(str(alert.get("entity_id") or PLACEHOLDER)),
                    html.Td(str(alert.get("anomaly_type") or PLACEHOLDER)),
                    html.Td(severity_badge(name)),
                    html.Td(_format_alert_value(alert), className="text-end"),
                ],
                style=style,
                className="alert-row",
            )
        )
    header = html.Thead(
        html.Tr(
            [
                html.Th("Time (UTC)"),
                html.Th("Entity"),
                html.Th("Type"),
                html.Th("Severity"),
                html.Th("Value", className="text-end"),
            ]
        )
    )
    return html.Div(
        dbc.Table(
            [header, html.Tbody(rows)],
            id="ov-feed-table",
            size="sm",
            borderless=True,
            className="mb-0 align-middle",
        ),
        id="ov-feed-scroll",
        style={"maxHeight": "340px", "overflowY": "auto"},
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def severity_donut(active_alerts: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Donut (``hole=0.5``) of the active alerts by severity, total in the centre."""
    counts = Counter(str(a.get("severity") or "").upper() for a in active_alerts)
    levels = [name for name in SEVERITY_ORDER if counts.get(name)]
    if not levels:
        return empty_figure("No active alerts", height=300)
    fig = go.Figure(
        go.Pie(
            labels=levels,
            values=[counts[name] for name in levels],
            hole=0.5,
            sort=False,
            direction="clockwise",
            marker={
                "colors": [SEVERITY_COLORS[name] for name in levels],
                "line": {"color": BACKGROUND_DARK, "width": 2},
            },
            textinfo="label+value",
            textfont={"color": TEXT_COLOR},
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{sum(counts[name] for name in levels)}</b><br>active",
        showarrow=False,
        font={"size": 20, "color": TEXT_COLOR},
        x=0.5,
        y=0.5,
    )
    return apply_theme(fig, height=300, showlegend=False, margin={"l": 8, "r": 8, "t": 8, "b": 8})


def trend_chart(trend: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Hourly alert counts of the last 24 h, stacked by severity (oldest hour left)."""
    if not trend or not any(row.get("total") for row in trend):
        return empty_figure("No alerts in the last 24 hours", height=300)
    hours = [parse_timestamp(row["hour"]) for row in trend]
    fig = go.Figure()
    for name in SEVERITY_ORDER:
        fig.add_bar(
            x=hours,
            y=[row.get(name, 0) for row in trend],
            name=name.title(),
            marker_color=SEVERITY_COLORS[name],
            hovertemplate=f"{name.title()}: %{{y}}<extra></extra>",
        )
    fig.update_xaxes(tickformat="%H:%M", dtick=3 * 3600 * 1000)
    fig.update_yaxes(title_text="Alerts per hour", rangemode="tozero")
    return apply_theme(
        fig,
        height=300,
        barmode="stack",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 48, "r": 8, "t": 28, "b": 36},
    )


# --------------------------------------------------------------------------- #
# System status row
# --------------------------------------------------------------------------- #
def status_row(badges: Sequence[Mapping[str, str]]) -> dbc.Row:
    """One pill per component (Kafka, Models, Dashboard, Kafka lag): name, state word, detail."""
    columns = []
    for badge in badges:
        state = badge.get("state", "unknown")
        color, word = STATE_STYLE.get(state, STATE_STYLE["unknown"])
        name = badge.get("name", "")
        columns.append(
            dbc.Col(
                html.Div(
                    [
                        dbc.Badge(word, color=color, pill=True, className="me-2"),
                        html.Span(name, className="fw-semibold me-2"),
                        html.Span(
                            badge.get("detail", ""), className="small", style={"color": MUTED_TEXT}
                        ),
                    ],
                    id=f"ov-status-{name.lower().replace(' ', '-')}",
                    className="status-pill",
                    style={
                        "border": f"1px solid {STATE_COLORS.get(state, NEUTRAL_GREY)}",
                        "backgroundColor": PANEL_DARK,
                    },
                ),
                xs=12,
                sm=6,
                xl=3,
                className="mb-2",
            )
        )
    return dbc.Row(columns, id="ov-status-row")


def card_with_title(title: str, body: Any, ident: str | None = None) -> dbc.Card:
    """Wrap ``body`` in a titled panel used by the chart areas of the page."""
    props: dict[str, Any] = {"id": ident} if ident else {}
    return dbc.Card(
        [
            dbc.CardHeader(title, className="small text-uppercase"),
            dbc.CardBody(body, className="p-2"),
        ],
        className="h-100",
        **props,
    )


def graph(ident: str, figure: go.Figure) -> dcc.Graph:
    """Return a :class:`dcc.Graph` without a mode bar, so it renders and resizes quickly."""
    return dcc.Graph(
        id=ident,
        figure=figure,
        config={"displayModeBar": False, "responsive": True},
        style={"height": "300px"},
    )


__all__ = [
    "FEED_SIZE",
    "alert_feed",
    "card_with_title",
    "graph",
    "health_color",
    "hex_to_rgba",
    "kpi_card",
    "kpi_row",
    "severity_donut",
    "status_row",
    "trend_chart",
]
