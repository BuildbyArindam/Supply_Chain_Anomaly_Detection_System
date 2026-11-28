"""Building blocks of the Alerts page (Day 20).

Pure functions (no Dash callbacks) so they can be unit-tested directly:

* :func:`grid_rows` / :func:`column_defs` / :func:`grid_options` - the AG Grid table.
* :func:`alert_detail` / :func:`shap_figure` - the detail modal: full ``AnomalyEvent`` plus the
  top SHAP features as a bar chart.
* :func:`alerts_to_csv` - the export of the filtered alerts with every field.
* :func:`feedback_notice` - the message shown after feedback was sent.

Times are UTC.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from src.dashboard.components.theme import (
    CRITICAL_RED,
    INFO_BLUE,
    MUTED_TEXT,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    apply_theme,
    empty_figure,
)
from src.dashboard.data_service import alert_status
from src.dashboard.utils import (
    PLACEHOLDER,
    format_timestamp,
    format_value_with_unit,
    severity_badge,
)

DEFAULT_ANALYST: Final = "dashboard-analyst"
ROW_LIMIT: Final = 1000
TOP_FEATURES: Final = 5
STATUS_LABELS: Final[dict[str, str]] = {
    "open": "Open",
    "false_positive": "False positive",
    "resolved": "Resolved",
}
ENTITY_LABELS: Final[dict[str, str]] = {
    "warehouse_sensor": "Warehouse sensor",
    "vehicle": "Vehicle",
    "inventory_sku": "Inventory SKU",
}
SEVERITY_OPTIONS: Final = [{"label": s.title(), "value": s} for s in SEVERITY_ORDER]
TYPE_OPTIONS: Final = [
    {"label": t.title(), "value": t} for t in ("point", "contextual", "collective")
]
ENTITY_OPTIONS: Final = [{"label": label, "value": key} for key, label in ENTITY_LABELS.items()]
RANGE_OPTIONS: Final = [
    {"label": "Last hour", "value": 1},
    {"label": "Last 6 hours", "value": 6},
    {"label": "Last 24 hours", "value": 24},
    {"label": "All stored", "value": 0},
]
DEFAULT_RANGE_HOURS: Final = 24
STATUS_OPTIONS: Final = [{"label": "Any status", "value": "all"}] + [
    {"label": label, "value": key} for key, label in STATUS_LABELS.items()
]
#: Columns of the CSV export, in order; everything else of an alert is added after them.
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "entity_id",
    "entity_type",
    "timestamp",
    "first_seen",
    "last_seen",
    "anomaly_type",
    "severity",
    "severity_score",
    "ensemble_score",
    "confidence",
    "business_impact",
    "raw_value",
    "expected_min",
    "expected_max",
    "status",
    "is_resolved",
    "false_positive_marked",
    "occurrence_count",
    "escalated_from",
    "recommended_action",
    "individual_scores",
    "top_contributing_features",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def grid_rows(alerts: Sequence[Mapping[str, Any]], limit: int = ROW_LIMIT) -> list[dict[str, Any]]:
    """Rows of the alerts table (newest first), keyed by ``event_id``."""
    rows = []
    for alert in alerts[: max(0, limit)]:
        rows.append(
            {
                "event_id": str(alert.get("event_id")),
                "timestamp": format_timestamp(alert.get("timestamp"), with_date=True),
                "entity_id": str(alert.get("entity_id") or PLACEHOLDER),
                "entity_type": ENTITY_LABELS.get(
                    str(alert.get("entity_type")), str(alert.get("entity_type") or PLACEHOLDER)
                ),
                "severity": str(alert.get("severity") or PLACEHOLDER).upper(),
                "anomaly_type": str(alert.get("anomaly_type") or PLACEHOLDER),
                "value": _number(alert.get("raw_value")),
                "status": STATUS_LABELS[alert_status(alert)],
                "actions": "View details ▸",
            }
        )
    return rows


def _style_rules(colors: Mapping[str, str]) -> dict[str, Any]:
    return {
        "styleConditions": [
            {
                "condition": f"params.value == '{value}'",
                "style": {"color": color, "fontWeight": "600"},
            }
            for value, color in colors.items()
        ]
    }


def column_defs() -> list[dict[str, Any]]:
    """Return AG Grid columns: timestamp, entity, type, severity, anomaly type, value, status."""
    severity_colors = {name: SEVERITY_COLORS[name] for name in SEVERITY_ORDER}
    status_colors = {"Open": CRITICAL_RED, "False positive": MUTED_TEXT, "Resolved": INFO_BLUE}
    return [
        {"field": "timestamp", "headerName": "Time (UTC)", "sort": "desc", "minWidth": 170},
        {"field": "entity_id", "headerName": "Entity", "minWidth": 130},
        {"field": "entity_type", "headerName": "Type", "minWidth": 140},
        {
            "field": "severity",
            "headerName": "Severity",
            "minWidth": 120,
            "cellStyle": _style_rules(severity_colors),
        },
        {"field": "anomaly_type", "headerName": "Anomaly type", "minWidth": 130},
        {
            "field": "value",
            "headerName": "Value",
            "type": "rightAligned",
            "minWidth": 110,
            "valueFormatter": {
                "function": "params.value == null ? '—' : d3.format(',.2f')(params.value)"
            },
        },
        {
            "field": "status",
            "headerName": "Status",
            "minWidth": 130,
            "cellStyle": _style_rules(status_colors),
        },
        {
            "field": "actions",
            "headerName": "Actions",
            "minWidth": 130,
            "sortable": False,
            "filter": False,
            "cellStyle": {"cursor": "pointer", "textDecoration": "underline"},
        },
    ]


def grid_options() -> dict[str, Any]:
    """Grid options: paging, checkbox selection (a click on a row opens its details)."""
    return {
        "pagination": True,
        "paginationPageSize": 25,
        "paginationPageSizeSelector": [25, 50, 100],
        "animateRows": False,
        "rowSelection": {
            "mode": "multiRow",
            "checkboxes": True,
            "headerCheckbox": True,
            "enableClickSelection": False,
        },
        "suppressCellFocus": True,
        "theme": {"function": "agGrid.themeQuartz.withPart(agGrid.colorSchemeDark)"},
    }


# --------------------------------------------------------------------------- #
# Detail modal
# --------------------------------------------------------------------------- #
def shap_figure(features: Sequence[Mapping[str, Any]] | None, n: int = TOP_FEATURES) -> go.Figure:
    """Horizontal bars of the ``n`` strongest SHAP contributions.

    Red bars push the score towards "anomalous", blue bars away from it; the sign is also in the
    label. Accepts ``{feature, shap_value, feature_value}`` (also ``name`` / ``contribution``).
    """
    parsed = []
    for item in features or []:
        name = item.get("feature") or item.get("name")
        weight = _number(item.get("shap_value", item.get("contribution", item.get("importance"))))
        if name is None or weight is None:
            continue
        parsed.append((str(name), weight, item.get("feature_value")))
    parsed.sort(key=lambda row: abs(row[1]), reverse=True)
    parsed = parsed[: max(0, n)]
    if not parsed:
        return empty_figure("No feature explanation was stored", height=200)
    parsed.reverse()  # plotly draws the first bar at the bottom
    fig = go.Figure(
        go.Bar(
            x=[w for _, w, _ in parsed],
            y=[name for name, _, _ in parsed],
            orientation="h",
            marker={"color": [CRITICAL_RED if w >= 0 else INFO_BLUE for _, w, _ in parsed]},
            text=[f"{w:+.3f}" for _, w, _ in parsed],
            textposition="outside",
            cliponaxis=False,
            customdata=[["—" if v is None else v] for _, _, v in parsed],
            hovertemplate="%{y}<br>SHAP %{x:+.4f}<br>feature value %{customdata[0]}<extra></extra>",
        )
    )
    return apply_theme(
        fig,
        height=max(200, 52 * len(parsed) + 70),
        title={"text": "Top contributing features (SHAP)", "font": {"size": 13}},
        xaxis={"title": "← lowers anomaly score · raises anomaly score →", "zeroline": True},
        margin={"l": 150, "r": 56, "t": 40, "b": 44},
    )


def _facts(alert: Mapping[str, Any]) -> list[tuple[str, str]]:
    expected = alert.get("expected_range")
    range_text = (
        f"{expected[0]:g} to {expected[1]:g}" if expected and len(expected) == 2 else PLACEHOLDER
    )

    def score(key: str) -> str:
        value = _number(alert.get(key))
        return PLACEHOLDER if value is None else f"{value:.2f}"

    return [
        ("Event ID", str(alert.get("event_id") or PLACEHOLDER)),
        ("Entity", f"{alert.get('entity_id') or PLACEHOLDER} ({alert.get('entity_type') or '—'})"),
        ("Detected", f"{format_timestamp(alert.get('timestamp'), with_date=True)} UTC"),
        ("First seen", f"{format_timestamp(alert.get('first_seen'), with_date=True)} UTC"),
        ("Last seen", f"{format_timestamp(alert.get('last_seen'), with_date=True)} UTC"),
        ("Detections", str(alert.get("occurrence_count") or 1)),
        ("Reading", format_value_with_unit(alert.get("raw_value"), None, 2)),
        ("Expected range", range_text),
        ("Severity score", score("severity_score")),
        ("Ensemble score", score("ensemble_score")),
        ("Confidence", score("confidence")),
        ("Business impact", score("business_impact")),
        ("Escalated from", str(alert.get("escalated_from") or "—")),
        ("Status", STATUS_LABELS[alert_status(alert)]),
    ]


def alert_detail(alert: Mapping[str, Any] | None) -> html.Div:
    """Body of the detail modal: every field of the alert and the SHAP bar chart."""
    if not alert:
        return html.Div(
            "This alert is no longer stored, so its details cannot be shown.",
            id="alert-detail-missing",
            style={"color": MUTED_TEXT},
        )
    scores = alert.get("individual_scores") or {}
    score_rows = [
        html.Div(
            [html.Span(f"{name}: ", style={"color": MUTED_TEXT}), html.Span(f"{value:.2f}")],
            className="small",
        )
        for name, value in sorted(scores.items())
        if _number(value) is not None
    ]
    return html.Div(
        [
            html.Div(
                [
                    severity_badge(alert.get("severity")),
                    html.Span(
                        f"{alert.get('anomaly_type') or 'anomaly'} anomaly",
                        className="ms-2 fw-semibold",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(f"{label}: ", style={"color": MUTED_TEXT}),
                                    html.Span(value),
                                ],
                                className="small mb-1",
                            )
                            for label, value in _facts(alert)
                        ],
                        xs=12,
                        lg=5,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="alert-shap-graph",
                                figure=shap_figure(alert.get("top_contributing_features")),
                                config={"displayModeBar": False, "responsive": True},
                            ),
                            html.Div(
                                "Detector scores",
                                className="small text-uppercase mt-2 mb-1",
                                style={"color": MUTED_TEXT},
                            ),
                            *(score_rows or [html.Div(PLACEHOLDER, className="small")]),
                        ],
                        xs=12,
                        lg=7,
                    ),
                ]
            ),
            html.Div(
                "Recommended action",
                className="small text-uppercase mt-3 mb-1",
                style={"color": MUTED_TEXT},
            ),
            html.Div(
                alert.get("recommended_action") or "No recommendation was stored.",
                className="p-2 rounded",
                style={"border": f"1px solid {INFO_BLUE}"},
            ),
        ],
        id="alert-detail",
    )


# --------------------------------------------------------------------------- #
# Export and feedback
# --------------------------------------------------------------------------- #
def _csv_cell(value: Any) -> Any:
    """Make a value safe for a spreadsheet: text that starts like a formula gets a ``'`` prefix."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def alerts_to_csv(alerts: Sequence[Mapping[str, Any]]) -> str:
    """CSV text of the alerts with every field (``CSV_COLUMNS`` first, then any extra field).

    The expected range is split into ``expected_min`` / ``expected_max``; nested values
    (``individual_scores``, ``top_contributing_features``) are written as JSON; ``status`` is
    ``open`` / ``false_positive`` / ``resolved``.
    """
    flat = []
    for alert in alerts:
        row = dict(alert)
        expected = row.pop("expected_range", None)
        row["expected_min"], row["expected_max"] = (
            (expected[0], expected[1]) if expected and len(expected) == 2 else (None, None)
        )
        row["status"] = alert_status(alert)
        for key in ("individual_scores", "top_contributing_features"):
            if key in row:
                row[key] = json.dumps(row[key], sort_keys=True, default=str)
        flat.append(row)
    extra = sorted({key for row in flat for key in row} - set(CSV_COLUMNS))
    columns = [*CSV_COLUMNS, *extra]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in flat:
        writer.writerow([_csv_cell("" if row.get(c) is None else row.get(c)) for c in columns])
    return buffer.getvalue()


def feedback_notice(result: Mapping[str, Any], verdict_label: str, skipped: int = 0) -> dbc.Alert:
    """Message after sending feedback: how many succeeded and which alerts failed."""
    ok, failed = int(result.get("ok", 0)), list(result.get("failed", []))
    requested = int(result.get("requested", ok + len(failed)))
    if requested == 0:
        return dbc.Alert(
            "Nothing to send: no alert was selected, or all were already marked.",
            color="secondary",
            id="alerts-feedback-alert",
        )
    extra = f" ({skipped} already {verdict_label}, skipped)" if skipped else ""
    if not failed:
        text = f"Marked {ok} alert{'s' if ok != 1 else ''} as {verdict_label}.{extra}"
        return dbc.Alert(text, color="success", dismissable=True, id="alerts-feedback-alert")
    detail = "; ".join(f"{f['event_id'][:8]}…: {f['error']}" for f in failed[:5])
    more = f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""
    return dbc.Alert(
        (
            f"{ok} of {requested} marked as {verdict_label}; "
            f"{len(failed)} failed — {detail}{more}{extra}"
        ),
        color="warning" if ok else "danger",
        dismissable=True,
        id="alerts-feedback-alert",
    )
