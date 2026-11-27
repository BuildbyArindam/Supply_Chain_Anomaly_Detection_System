"""Building blocks of the Model Health page (Day 21).

Pure functions that turn :class:`~src.dashboard.data_service.DashboardDataService` output into
Dash components and plotly figures, kept out of ``pages/`` so they can be tested without
registering a page. Every builder accepts empty input and returns a readable placeholder, so the
page renders on a fresh system with no models, no feedback and no retraining history.

Colour is never the only signal: gauges print their value, run outcomes carry a word, and changes
carry an arrow.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from src.dashboard.components.overview_widgets import (
    HEALTH_AMBER,
    HEALTH_GREEN,
    NEUTRAL_GREY,
    health_color,
    hex_to_rgba,
)
from src.dashboard.components.theme import (
    CRITICAL_RED,
    GRID_COLOR,
    INFO_BLUE,
    MUTED_TEXT,
    NORMAL_GREEN,
    PANEL_DARK,
    SEVERITY_COLORS,
    TEXT_COLOR,
    WARNING_AMBER,
    apply_theme,
    empty_figure,
)
from src.dashboard.utils import PLACEHOLDER, format_timestamp, parse_timestamp

#: Target ceiling of the false-positive rate (the dashed line of the trend chart).
FP_RATE_TARGET: Final = 0.10
#: Display names of the model families, in chart order.
MODEL_LABELS: Final[dict[str, str]] = {
    "statistical": "Statistical",
    "isolation_forest": "IForest",
    "lstm_autoencoder": "LSTM-AE",
    "lstm_classifier": "LSTM-Cls",
    "ensemble": "Ensemble",
}
#: Outcome of a retraining run -> (colour, word). Promoted green and rejected red as specified;
#: runs that did not reach a verdict get neutral colours so they are not mistaken for either.
OUTCOME_STYLE: Final[dict[str, tuple[str, str]]] = {
    "promoted": (NORMAL_GREEN, "Promoted"),
    "rejected": (CRITICAL_RED, "Rejected"),
    "skipped": (NEUTRAL_GREY, "Skipped"),
    "failed": (WARNING_AMBER, "Failed"),
}
CHAMPION_COLOR: Final = INFO_BLUE
CHALLENGER_COLOR: Final = "#fd7e14"
#: Most runs drawn on the retraining timeline and listed in the log.
MAX_RUNS_SHOWN: Final = 15
#: Smallest drawn width of a run bar, as a share of the time axis (runs last seconds, the axis
#: spans days).
MIN_BAR_SHARE: Final = 0.012
#: A change smaller than this (in absolute score points) shows a flat arrow.
FLAT_DELTA: Final = 0.0005


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def percent(value: Any, digits: int = 1) -> str:
    """Format a ``0..1`` score as a percentage; ``—`` if it is missing."""
    return f"{float(value):.{digits}%}" if _is_number(value) else PLACEHOLDER


def graph_height(fig: go.Figure, height: int) -> go.Figure:
    """Set the figure height (the page lays charts out in fixed-height cards)."""
    fig.update_layout(height=height)
    return fig


# --------------------------------------------------------------------------- #
# Drift banner
# --------------------------------------------------------------------------- #
def drift_banner(active: bool, last_alert: Mapping[str, Any] | None) -> html.Div:
    """Banner shown while the feedback drift detector has fired within the last hour.

    Returns an empty div (not ``None``) when inactive, so the callback keeps a stable child.
    """
    if not active or not last_alert:
        return html.Div(id="mh-drift-banner-body")
    action = str(last_alert.get("recommended_action") or "full_retrain").replace("_", " ")
    detail = [
        f"Detected {format_timestamp(last_alert.get('detected_at'), with_date=True)} UTC · "
        f"false-positive rate {percent(last_alert.get('baseline_fp_rate'), 0)} → "
        f"{percent(last_alert.get('recent_fp_rate'), 0)} · "
        f"recommended action: {action}"
    ]
    return html.Div(
        dbc.Alert(
            [
                html.Strong("Model drift alert. "),
                str(last_alert.get("message") or "Analyst feedback has shifted."),
                html.Div(detail, className="small mt-1"),
            ],
            color="danger",
            className="mb-3",
            id="mh-drift-alert",
        ),
        id="mh-drift-banner-body",
    )


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
def gauge(title: str, value: float | None, *, height: int = 210) -> go.Figure:
    """Build a 0-100 % gauge: green above 85 %, amber from 70 to 85 %, red below 70 %.

    The number is always printed. ``None`` gives a placeholder figure instead of a misleading 0.
    """
    if value is None or not _is_number(value):
        return empty_figure(f"{title}: no evaluation metrics yet", height=height)
    color = health_color(float(value))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value) * 100.0,
            number={"suffix": " %", "valueformat": ".1f", "font": {"color": TEXT_COLOR}},
            title={"text": title, "font": {"color": MUTED_TEXT, "size": 14}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": MUTED_TEXT,
                    "tickfont": {"color": MUTED_TEXT},
                },
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": PANEL_DARK,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, HEALTH_AMBER * 100], "color": hex_to_rgba(CRITICAL_RED, 0.28)},
                    {
                        "range": [HEALTH_AMBER * 100, HEALTH_GREEN * 100],
                        "color": hex_to_rgba(WARNING_AMBER, 0.28),
                    },
                    {"range": [HEALTH_GREEN * 100, 100], "color": hex_to_rgba(NORMAL_GREEN, 0.28)},
                ],
            },
        )
    )
    return apply_theme(fig, height=height, margin={"l": 24, "r": 24, "t": 48, "b": 8})


def live_precision_caption(live: Mapping[str, Any] | None) -> str:
    """One line under the gauges: the analysts' verdict-based precision of the last 24 h."""
    precision = (live or {}).get("precision")
    verdicts = int((live or {}).get("verdicts") or 0)
    if not _is_number(precision) or not verdicts:
        return "Live analyst precision (24 h): no verdicts yet"
    return f"Live analyst precision (24 h): {percent(precision)} from {verdicts:,} verdicts"


def model_f1_chart(
    f1_by_family: Mapping[str, float | None], versions: Mapping[str, int] | None = None
) -> go.Figure:
    """Side-by-side F1 bars of Statistical, IForest, LSTM-AE, LSTM-Cls and the Ensemble.

    A model without a saved F1 gets no bar and an ``n/a`` label rather than a zero bar.
    """
    families = list(MODEL_LABELS)
    values = [f1_by_family.get(family) for family in families]
    if not any(_is_number(v) for v in values):
        return empty_figure("No model metrics in the registry yet", height=300)
    labels = [MODEL_LABELS[f] for f in families]
    drawn = [float(v) if _is_number(v) else None for v in values]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=drawn,
            marker={"color": [health_color(v) if v is not None else NEUTRAL_GREY for v in drawn]},
            text=[percent(v) if v is not None else "" for v in drawn],
            textposition="outside",
            textfont={"color": TEXT_COLOR},
            customdata=[(versions or {}).get(f, 0) for f in families],
            hovertemplate="%{x}: %{y:.3f} (%{customdata} registry entries)<extra></extra>",
            name="F1",
        )
    )
    for label, value in zip(labels, drawn, strict=True):
        if value is None:
            fig.add_annotation(
                x=label, y=0.04, text="n/a", showarrow=False, font={"color": MUTED_TEXT}, yref="y"
            )
    for edge in (HEALTH_AMBER, HEALTH_GREEN):
        fig.add_hline(y=edge, line={"color": GRID_COLOR, "dash": "dot", "width": 1})
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="F1")
    return apply_theme(fig, height=300, showlegend=False)


def anomaly_type_chart(breakdown: Mapping[str, Any]) -> go.Figure:
    """Stacked bars of today's alerts: one bar per anomaly type, stacked by severity."""
    types: Sequence[str] = breakdown.get("types") or ()
    counts: Mapping[str, Mapping[str, int]] = breakdown.get("counts") or {}
    if not breakdown.get("total"):
        return empty_figure("No anomalies detected today", height=300)
    fig = go.Figure()
    for severity in breakdown.get("severities") or ():
        ys = [int(counts.get(kind, {}).get(severity, 0)) for kind in types]
        if not any(ys):
            continue
        fig.add_bar(
            x=list(types),
            y=ys,
            name=severity,
            marker={"color": SEVERITY_COLORS.get(severity, NEUTRAL_GREY)},
            hovertemplate=f"{severity}: %{{y}}<extra>%{{x}}</extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Alerts today (UTC)", rangemode="tozero", dtick=None)
    return apply_theme(fig, height=300, legend={"orientation": "h", "y": -0.2})


# --------------------------------------------------------------------------- #
# Feedback analytics
# --------------------------------------------------------------------------- #
def fp_trend_chart(daily: Sequence[Mapping[str, Any]], target: float = FP_RATE_TARGET) -> go.Figure:
    """False-positive rate per day with a dashed line at ``target`` (10 %).

    Days without verdicts stay gaps. The rate is the share of false positives among the verdicts
    of that day; the hover also shows how many verdicts that is.
    """
    if not any(row.get("total") for row in daily):
        return empty_figure("No analyst verdicts in the last 7 days", height=300)
    rates = [row.get("false_positive_rate") for row in daily]
    fig = go.Figure(
        go.Scatter(
            x=[str(row.get("date")) for row in daily],
            y=[float(r) if _is_number(r) else None for r in rates],
            mode="lines+markers",
            line={"color": INFO_BLUE, "width": 2},
            marker={"size": 8},
            connectgaps=False,
            customdata=[(row.get("false_positive", 0), row.get("total", 0)) for row in daily],
            hovertemplate=(
                "%{x}: %{y:.1%} (%{customdata[0]} of %{customdata[1]} verdicts)<extra></extra>"
            ),
            name="False-positive rate",
        )
    )
    fig.add_hline(
        y=target,
        line={"color": WARNING_AMBER, "dash": "dash", "width": 1.5},
        annotation_text=f"target {target:.0%}",
        annotation_font={"color": WARNING_AMBER},
        annotation_position="top left",
    )
    top = max([target, *[float(r) for r in rates if _is_number(r)]])
    fig.update_yaxes(range=[0, min(1.0, top * 1.25)], tickformat=".0%", title="False positives")
    fig.update_xaxes(type="category")
    return apply_theme(fig, height=300, showlegend=False)


def feedback_heatmap(grid: Mapping[str, Any]) -> go.Figure:
    """Days (x) by UTC hour (y); colour is the number of feedback events given in that hour."""
    days: Sequence[str] = grid.get("days") or ()
    matrix: Sequence[Sequence[int]] = grid.get("z") or ()
    if not days or not any(any(row) for row in matrix):
        return empty_figure("No feedback in the last 7 days", height=300)
    fig = go.Figure(
        go.Heatmap(
            x=list(days),
            y=list(grid.get("hours") or range(24)),
            z=[list(row) for row in matrix],
            colorscale=[[0.0, PANEL_DARK], [1.0, INFO_BLUE]],
            zmin=0,
            colorbar={"title": {"text": "Events"}, "tickfont": {"color": MUTED_TEXT}},
            xgap=2,
            ygap=1,
            hovertemplate="%{x} %{y:02d}:00 UTC: %{z} events<extra></extra>",
        )
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="Hour (UTC)", dtick=4, autorange="reversed")
    return apply_theme(fig, height=300)


def top_fp_table(rows: Sequence[Mapping[str, Any]]) -> html.Div:
    """Table of the entities with the highest false-positive rate (threshold candidates)."""
    if not rows:
        return html.Div(
            "No entity has enough verdicts yet.",
            id="mh-fp-empty",
            className="p-3",
            style={"color": MUTED_TEXT},
        )
    header = html.Thead(
        html.Tr(
            [
                html.Th("Entity"),
                html.Th("FP rate", className="text-end"),
                html.Th("FP / verdicts", className="text-end"),
                html.Th("Threshold shift", className="text-end"),
            ]
        )
    )
    body = []
    for row in rows:
        rate = float(row.get("false_positive_rate") or 0.0)
        shift = float(row.get("threshold_adjustment") or 0.0)
        color = (
            CRITICAL_RED if rate > 0.5 else WARNING_AMBER if rate > FP_RATE_TARGET else TEXT_COLOR
        )
        body.append(
            html.Tr(
                [
                    html.Td(str(row.get("entity_id"))),
                    html.Td(percent(rate, 0), className="text-end", style={"color": color}),
                    html.Td(
                        f"{row.get('false_positive', 0)} / {row.get('total', 0)}",
                        className="text-end",
                    ),
                    html.Td(
                        f"{shift:+.2f}" if shift else "none yet",
                        className="text-end",
                        style={} if shift else {"color": MUTED_TEXT},
                    ),
                ]
            )
        )
    return html.Div(
        dbc.Table(
            [header, html.Tbody(body)], size="sm", borderless=True, className="mb-0 align-middle"
        ),
        id="mh-fp-table",
        style={"maxHeight": "300px", "overflowY": "auto"},
    )


# --------------------------------------------------------------------------- #
# Retraining history
# --------------------------------------------------------------------------- #
def retraining_notice(enabled: bool, runs: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the retraining section is empty (``None`` when there is something to show)."""
    if runs:
        return None
    if not enabled:
        return (
            "Continuous retraining is not running in this deployment, so there is no history. "
            "Runs appear here once a retraining orchestrator is attached to the API."
        )
    return (
        "No retraining run yet. Retraining starts on drift, after enough feedback or on schedule."
    )


def _duration_ms(run: Mapping[str, Any]) -> float | None:
    start, end = parse_timestamp(run.get("started_at")), parse_timestamp(run.get("finished_at"))
    return (end - start).total_seconds() * 1000.0 if start and end else None


def retraining_timeline(runs: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Gantt-style chart of the newest runs: one bar per run from start to end.

    Promoted runs are green and rejected ones red; skipped and failed runs use neutral and amber
    so they are not read as either verdict. Runs last seconds on an axis of hours or days, so a
    bar is drawn at least ``MIN_BAR_SHARE`` of the axis wide; the hover shows the true duration.
    """
    shown = [r for r in runs if parse_timestamp(r.get("started_at")) is not None][-MAX_RUNS_SHOWN:]
    if not shown:
        return empty_figure("No retraining runs yet", height=300)
    starts = [parse_timestamp(r["started_at"]) for r in shown]
    ends = [parse_timestamp(r.get("finished_at")) or s for r, s in zip(shown, starts, strict=True)]
    span_ms = max((max(ends) - min(starts)).total_seconds() * 1000.0, 1.0)
    floor = span_ms * MIN_BAR_SHARE
    fig = go.Figure()
    for outcome, (color, word) in OUTCOME_STYLE.items():
        picked = [(i, r) for i, r in enumerate(shown) if str(r.get("outcome")) == outcome]
        if not picked:
            continue
        real = [_duration_ms(r) or 0.0 for _, r in picked]
        fig.add_bar(
            y=[f"{i + 1}. {str(r.get('run_id') or '')[:8]}" for i, r in picked],
            x=[max(d, floor) for d in real],
            base=[starts[i] for i, _ in picked],
            orientation="h",
            name=word,
            marker={"color": color},
            customdata=[
                (
                    str(r.get("trigger_reason") or PLACEHOLDER).replace("_", " "),
                    f"{d / 1000.0:.1f} s",
                    str(r.get("reason") or ""),
                )
                for (_, r), d in zip(picked, real, strict=True)
            ],
            hovertemplate=(
                f"{word} · %{{customdata[0]}}<br>duration %{{customdata[1]}}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        )
    fig.update_xaxes(type="date", title="Time (UTC)")
    fig.update_yaxes(autorange="reversed", title="Run")
    return apply_theme(
        fig,
        height=max(260, 60 + 28 * len(shown)),
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 110, "r": 24, "t": 24, "b": 56},
    )


def retraining_log(runs: Sequence[Mapping[str, Any]]) -> html.Div:
    """Table of the newest runs (newest first) with the columns of the audit log."""
    if not runs:
        return html.Div(
            "No retraining runs to list.",
            id="mh-log-empty",
            className="p-3",
            style={"color": MUTED_TEXT},
        )
    header = html.Thead(
        html.Tr(
            [
                html.Th(name, className=cls)
                for name, cls in (
                    ("Run", ""),
                    ("Trigger", ""),
                    ("Data", "text-end"),
                    ("Training", "text-end"),
                    ("Champion F1", "text-end"),
                    ("Challenger F1", "text-end"),
                    ("Outcome", ""),
                )
            ]
        )
    )
    body = []
    for run in reversed(list(runs)[-MAX_RUNS_SHOWN:]):
        color, word = OUTCOME_STYLE.get(
            str(run.get("outcome")), (NEUTRAL_GREY, str(run.get("outcome")))
        )
        body.append(
            html.Tr(
                [
                    html.Td(str(run.get("run_id") or "")[:8], title=str(run.get("run_id") or "")),
                    html.Td(str(run.get("trigger_reason") or PLACEHOLDER).replace("_", " ")),
                    html.Td(f"{int(run.get('data_size') or 0):,}", className="text-end"),
                    html.Td(
                        f"{float(run.get('training_time_s') or 0.0):.1f} s", className="text-end"
                    ),
                    html.Td(percent(run.get("champion_f1")), className="text-end"),
                    html.Td(percent(run.get("challenger_f1")), className="text-end"),
                    html.Td(
                        dbc.Badge(
                            word, style={"backgroundColor": color, "color": "#111"}, pill=True
                        ),
                        title=str(run.get("reason") or ""),
                    ),
                ]
            )
        )
    return html.Div(
        dbc.Table(
            [header, html.Tbody(body)], size="sm", borderless=True, className="mb-0 align-middle"
        ),
        id="mh-log-table",
        style={"maxHeight": "320px", "overflowY": "auto"},
    )


def delta_arrow(delta: Any) -> html.Span:
    """Return an up, down or flat arrow with the change in percentage points (``▲ +2.1 pp``)."""
    if not _is_number(delta):
        return html.Span(PLACEHOLDER, style={"color": MUTED_TEXT})
    points = float(delta) * 100.0
    if abs(float(delta)) < FLAT_DELTA:
        return html.Span("► 0.0 pp", style={"color": MUTED_TEXT})
    up = float(delta) > 0
    return html.Span(
        f"{'▲' if up else '▼'} {points:+.1f} pp",
        style={"color": NORMAL_GREEN if up else CRITICAL_RED},
    )


def retrain_delta_card(delta: Mapping[str, Any] | None) -> html.Div:
    """Improvement of the latest promoted run: baseline (the champion it replaced) vs current."""
    if not delta:
        return html.Div(
            "No retraining run has been promoted yet, so there is no before / after to compare.",
            id="mh-delta-empty",
            className="p-3",
            style={"color": MUTED_TEXT},
        )
    rows = []
    models = dict(delta.get("models") or {}) or {
        "primary": {
            "champion_f1": delta.get("champion_f1"),
            "challenger_f1": delta.get("challenger_f1"),
            "improvement": delta.get("improvement"),
        }
    }
    for name, report in models.items():
        rows.append(
            html.Tr(
                [
                    html.Td(name.replace("_", " ")),
                    html.Td(percent(report.get("champion_f1")), className="text-end"),
                    html.Td(percent(report.get("challenger_f1")), className="text-end"),
                    html.Td(delta_arrow(report.get("improvement")), className="text-end"),
                ]
            )
        )
    header = html.Thead(
        html.Tr(
            [
                html.Th("Model"),
                html.Th("Baseline F1", className="text-end"),
                html.Th("Current F1", className="text-end"),
                html.Th("Change", className="text-end"),
            ]
        )
    )
    promoted_at = format_timestamp(delta.get("finished_at"), with_date=True)
    parts: list[Any] = [
        html.Div(
            f"Since the last promotion ({promoted_at} UTC)",
            className="small mb-2",
            style={"color": MUTED_TEXT},
        ),
        dbc.Table(
            [header, html.Tbody(rows)], size="sm", borderless=True, className="mb-0 align-middle"
        ),
    ]
    if delta.get("rolled_back"):
        parts.append(
            dbc.Alert(
                "This promotion was rolled back.", color="warning", className="mt-2 mb-0 py-1"
            )
        )
    return html.Div(parts, id="mh-delta")


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
def _metric_values(side: Mapping[str, Any] | None) -> dict[str, float]:
    metrics = (side or {}).get("metrics") or {}
    return {k: float(v) for k, v in metrics.items() if _is_number(v)}


def comparison_rows(challenger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Metric rows of the champion/challenger table: ``metric``, both values and the change in %.

    The change is relative to the champion (``None`` when the champion value is zero or missing).
    F1, precision and recall come first, then any other metric both models report.
    """
    champion = _metric_values(challenger.get("champion"))
    contender = _metric_values(challenger.get("challenger"))
    shared = [k for k in champion if k in contender]
    order = [k for k in ("f1", "precision", "recall") if k in shared]
    order += sorted(k for k in shared if k not in order)
    rows = []
    for name in order:
        before, after = champion[name], contender[name]
        rows.append(
            {
                "metric": name,
                "champion": before,
                "challenger": after,
                "change_pct": (after - before) / abs(before) * 100.0 if before else None,
            }
        )
    return rows


def comparison_table(challenger: Mapping[str, Any]) -> html.Div:
    """Side-by-side metrics of the serving champion and the waiting challenger."""
    rows = comparison_rows(challenger)
    if not rows:
        return html.Div(
            "The challenger reports no metrics that the champion also has.",
            id="mh-cc-empty",
            className="p-3",
            style={"color": MUTED_TEXT},
        )
    body = []
    for row in rows:
        change = row["change_pct"]
        color = (
            MUTED_TEXT
            if change is None or abs(change) < 0.05
            else NORMAL_GREEN if change > 0 else CRITICAL_RED
        )
        arrow = (
            "" if change is None else "▲ " if change > 0.05 else "▼ " if change < -0.05 else "► "
        )
        body.append(
            html.Tr(
                [
                    html.Td(str(row["metric"]).replace("_", " ")),
                    html.Td(f"{row['champion']:.4f}", className="text-end"),
                    html.Td(f"{row['challenger']:.4f}", className="text-end"),
                    html.Td(
                        PLACEHOLDER if change is None else f"{arrow}{change:+.1f} %",
                        className="text-end",
                        style={"color": color},
                    ),
                ]
            )
        )
    champion_version = (challenger.get("champion") or {}).get("version") or ""
    challenger_version = (challenger.get("challenger") or {}).get("version") or ""
    header = html.Thead(
        html.Tr(
            [
                html.Th("Metric"),
                html.Th(
                    f"Champion {champion_version}".strip(),
                    className="text-end",
                ),
                html.Th(
                    f"Challenger {challenger_version}".strip(),
                    className="text-end",
                ),
                html.Th("Change", className="text-end"),
            ]
        )
    )
    return html.Div(
        dbc.Table(
            [header, html.Tbody(body)], size="sm", borderless=True, className="mb-0 align-middle"
        ),
        id="mh-cc-table",
    )


def pr_overlay(challenger: Mapping[str, Any]) -> go.Figure:
    """Precision-recall curves of champion (blue) and challenger (orange) on one chart."""
    curves = challenger.get("pr_curve") or {}
    fig = go.Figure()
    for side, color, label in (
        ("champion", CHAMPION_COLOR, "Champion"),
        ("challenger", CHALLENGER_COLOR, "Challenger"),
    ):
        curve = curves.get(side) or {}
        recall, precision = curve.get("recall") or (), curve.get("precision") or ()
        if len(recall) != len(precision) or len(recall) < 2:
            continue
        fig.add_scatter(
            x=list(recall),
            y=list(precision),
            mode="lines",
            name=label,
            line={"color": color, "width": 2.5, "dash": "solid" if side == "champion" else "dash"},
            hovertemplate=f"{label}: precision %{{y:.3f}} at recall %{{x:.3f}}<extra></extra>",
        )
    if not fig.data:
        return empty_figure("The challenger carries no precision-recall curve", height=300)
    fig.update_xaxes(range=[0, 1.0], title="Recall")
    fig.update_yaxes(range=[0, 1.05], title="Precision")
    return apply_theme(fig, height=300, legend={"orientation": "h", "y": -0.25})


def challenger_summary(challenger: Mapping[str, Any]) -> str:
    """Heading line of the champion/challenger card."""
    name = str(challenger.get("model_name") or "model").replace("_", " ")
    margin = challenger.get("min_improvement")
    extra = (
        f" · needs +{float(margin):.0%} F1 to be promoted automatically"
        if _is_number(margin)
        else ""
    )
    created = challenger.get("created_at")
    when = f" · trained {format_timestamp(created, with_date=True)} UTC" if created else ""
    return f"Challenger for {name}{when}{extra}"


__all__ = [
    "CHALLENGER_COLOR",
    "CHAMPION_COLOR",
    "FP_RATE_TARGET",
    "MODEL_LABELS",
    "OUTCOME_STYLE",
    "anomaly_type_chart",
    "challenger_summary",
    "comparison_rows",
    "comparison_table",
    "delta_arrow",
    "drift_banner",
    "feedback_heatmap",
    "fp_trend_chart",
    "gauge",
    "live_precision_caption",
    "model_f1_chart",
    "percent",
    "pr_overlay",
    "retrain_delta_card",
    "retraining_log",
    "retraining_notice",
    "retraining_timeline",
    "top_fp_table",
]
