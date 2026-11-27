"""Building blocks of the Warehouse page (Day 19).

Pure functions (no Dash callbacks) so they can be unit-tested directly:

* :func:`resolve_selection` - the cascading warehouse -> zone -> sensor dropdowns.
* :func:`timeseries_figure` / :func:`new_points` - the live chart and the increment that the
  browser appends with ``Plotly.extendTraces`` (see the page's clientside callback).
* :func:`zone_heatmap` - current readings laid out as a warehouse floor grid.
* :func:`anomaly_card` - the detail panel of a clicked anomaly marker.

Times are UTC. A chart "cursor" is ``{"entity_id", "t", "alert_t"}`` with epoch *milliseconds*: the
newest point and the newest alert already drawn in the browser.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from src.dashboard.components.overview_widgets import NEUTRAL_GREY, hex_to_rgba
from src.dashboard.components.theme import (
    BACKGROUND_DARK,
    CRITICAL_RED,
    INFO_BLUE,
    MUTED_TEXT,
    NORMAL_GREEN,
    PANEL_DARK,
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

LOOKBACK_SECONDS: Final = 300.0
#: ``maxPoints`` given to ``extendTraces``: the browser keeps at most this many points per trace.
MAX_POINTS: Final = 600
HEATMAP_COLS: Final = 5
HEATMAP_MIN_ROWS: Final = 4
SENSOR_KINDS: Final[tuple[str, ...]] = ("temperature", "humidity")
#: Trace order of the time-series figure; the clientside callback relies on it.
TRACE_READINGS: Final = 0
TRACE_ANOMALIES: Final = 1
TRACE_RANGE_ANCHOR: Final = 2
CLICK_TOLERANCE_S: Final = 1.0
TOP_FEATURES: Final = 5


def sensor_kind(entity: Mapping[str, Any]) -> str:
    """Normalise an entity's sensor type to ``temperature`` / ``humidity`` (or its own name)."""
    raw = str(entity.get("sensor_type") or "").lower()
    if "humid" in raw:
        return "humidity"
    if "temp" in raw:
        return "temperature"
    return raw


_UNIT_SYMBOLS: Final[dict[str, str]] = {"celsius": "°C", "c": "°C", "percent": "%", "pct": "%"}


def unit_symbol(unit: str | None) -> str:
    """Display symbol of a unit name (``celsius`` -> ``°C``); unknown units pass through."""
    return _UNIT_SYMBOLS.get((unit or "").lower(), unit or "")


def epoch_ms(value: Any) -> int | None:
    """Epoch milliseconds of a timestamp, ``None`` if it cannot be parsed."""
    moment = parse_timestamp(value)
    return None if moment is None else int(round(moment.timestamp() * 1000))


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def _options(
    values: Sequence[str], labels: Mapping[str, str] | None = None
) -> list[dict[str, str]]:
    return [{"label": (labels or {}).get(v, v), "value": v} for v in values]


def _keep_or_first(current: str | None, values: Sequence[str]) -> str | None:
    if current in values:
        return current
    return values[0] if values else None


def resolve_selection(
    entities: Sequence[Mapping[str, Any]],
    warehouse: str | None = None,
    zone: str | None = None,
    sensor: str | None = None,
) -> dict[str, Any]:
    """Compute the three cascading dropdowns from the live warehouse sensors.

    A current choice that is still valid is kept; otherwise the first option is taken (``None``
    when there is nothing to choose).

    Returns:
        ``{"warehouse_options", "zone_options", "sensor_options", "warehouse", "zone", "sensor"}``.
    """
    sensors = [e for e in entities if e.get("source_type") == "warehouse"]
    warehouses = sorted({str(e["warehouse_id"]) for e in sensors if e.get("warehouse_id")})
    chosen_warehouse = _keep_or_first(warehouse, warehouses)
    in_warehouse = [e for e in sensors if str(e.get("warehouse_id")) == chosen_warehouse]
    zones = sorted({str(e["zone"]) for e in in_warehouse if e.get("zone")})
    chosen_zone = _keep_or_first(zone, zones)
    in_zone = sorted(
        (e for e in in_warehouse if str(e.get("zone")) == chosen_zone),
        key=lambda e: str(e["entity_id"]),
    )
    ids = [str(e["entity_id"]) for e in in_zone]
    labels = {
        str(e["entity_id"]): f"{e['entity_id']} · {sensor_kind(e) or 'sensor'}" for e in in_zone
    }
    return {
        "warehouse_options": _options(warehouses),
        "zone_options": _options(zones, {z: f"Zone {z}" for z in zones}),
        "sensor_options": _options(ids, labels),
        "warehouse": chosen_warehouse,
        "zone": chosen_zone,
        "sensor": _keep_or_first(sensor, ids),
    }


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def alert_points(
    alerts: Sequence[Mapping[str, Any]], series: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Return alerts as chart points ``{"t", "v", "severity"}``.

    An alert without a reading is drawn at the series' last value.
    """
    fallback = None
    points = (series or {}).get("points") or []
    if points:
        fallback = points[-1].get("v")
    found = []
    for alert in alerts:
        value = alert.get("raw_value")
        value = fallback if value is None else value
        if alert.get("timestamp") is None or value is None:
            continue
        found.append(
            {
                "t": alert["timestamp"],
                "v": value,
                "severity": str(alert.get("severity") or "").upper(),
            }
        )
    return found


def series_note(series: Mapping[str, Any] | None) -> str:
    """Message shown over the chart when there is nothing to draw, else an empty string."""
    if series is None:
        return "No data available for this sensor."
    if not series.get("points"):
        return "Waiting for readings…"
    return ""


def make_cursor(
    entity_id: str | None, series: Mapping[str, Any] | None, alerts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the cursor of what a freshly built chart already shows."""
    point_times = [epoch_ms(p.get("t")) for p in (series or {}).get("points") or []]
    alert_times = [epoch_ms(a.get("t")) for a in alerts]
    return {
        "entity_id": entity_id,
        "t": max((t for t in point_times if t is not None), default=0),
        "alert_t": max((t for t in alert_times if t is not None), default=0),
    }


def chart_window_start(series: Mapping[str, Any] | None, now: datetime) -> datetime:
    """Earliest time an alert can have to be drawn: the lookback before the newest reading."""
    points = (series or {}).get("points") or []
    newest = parse_timestamp(points[-1].get("t")) if points else None
    return (newest or now) - timedelta(seconds=LOOKBACK_SECONDS)


def timeseries_figure(
    series: Mapping[str, Any] | None,
    alerts: Sequence[Mapping[str, Any]] = (),
    *,
    entity_id: str | None = None,
    now: datetime | None = None,
) -> go.Figure:
    """Build the live line chart of one sensor.

    Three traces, always present (the browser extends them in place, so they must exist even
    when there is no data yet): ``0`` readings, ``1`` anomaly markers, ``2`` an invisible anchor
    that keeps the acceptable range inside the autoscaled axis. The acceptable range itself is
    drawn as two dashed lines over a faint band.

    Args:
        series: ``DashboardDataService.get_sensor_timeseries`` output, or ``None``.
        alerts: This sensor's alerts (``get_entity_alerts``), oldest first.
        entity_id: Used as the ``uirevision`` so that a zoom survives re-renders.
        now: Wall clock, for the anchor of an empty chart (injectable for tests).
    """
    series = series or {}
    points = series.get("points") or []
    unit = series.get("unit")
    unit_text = unit_symbol(unit)
    kind = str(series.get("sensor_type") or "reading").title()
    marks = alert_points(alerts, series)

    fig = go.Figure()
    fig.add_scatter(
        x=[p["t"] for p in points],
        y=[p["v"] for p in points],
        mode="lines",
        name=kind,
        line={"color": INFO_BLUE, "width": 2},
        hovertemplate=f"%{{y:.2f}} {unit_text}<br>%{{x|%H:%M:%S}}<extra></extra>",
    )
    fig.add_scatter(
        x=[m["t"] for m in marks],
        y=[m["v"] for m in marks],
        mode="markers",
        name="Anomaly",
        marker={
            "color": CRITICAL_RED,
            "size": 12,
            "symbol": "x",
            "line": {"color": CRITICAL_RED, "width": 2},
        },
        hovertemplate="Anomaly<br>%{y:.2f} " + unit_text + "<br>%{x|%H:%M:%S}<extra></extra>",
    )
    expected = series.get("expected_range")
    if expected and len(expected) == 2:
        low, high = float(expected[0]), float(expected[1])
        anchor_x = points[0]["t"] if points else (now or datetime.now(UTC)).isoformat()
        fig.add_scatter(
            x=[anchor_x, anchor_x],
            y=[low, high],
            mode="markers",
            marker={"opacity": 0},
            hoverinfo="skip",
            showlegend=False,
            name="range",
        )
        fig.add_hrect(
            y0=low, y1=high, fillcolor=hex_to_rgba(NORMAL_GREEN, 0.07), line_width=0, layer="below"
        )
        for level, label in ((low, "min"), (high, "max")):
            fig.add_hline(
                y=level,
                line={"color": WARNING_AMBER, "width": 1.5, "dash": "dash"},
                annotation_text=f"{label} {level:g}{unit_text}",
                annotation_position="top left",
                annotation_font={"color": WARNING_AMBER, "size": 11},
            )
    else:
        fig.add_scatter(
            x=[], y=[], mode="markers", marker={"opacity": 0}, hoverinfo="skip", showlegend=False
        )
    fig.update_xaxes(type="date", tickformat="%H:%M:%S", title_text="Time (UTC)")
    fig.update_yaxes(title_text=f"{kind} ({unit_text})" if unit_text else kind)
    return apply_theme(
        fig,
        height=380,
        hovermode="x unified",
        uirevision=entity_id or "none",
        legend={"orientation": "h", "y": 1.1, "x": 0},
        margin={"l": 56, "r": 16, "t": 36, "b": 44},
    )


def new_points(
    series: Mapping[str, Any] | None,
    alerts: Sequence[Mapping[str, Any]],
    cursor: Mapping[str, Any] | None,
    entity_id: str,
) -> dict[str, Any] | None:
    """Return what the browser has not drawn yet, or ``None`` when there is nothing new.

    ``None`` is also returned when the cursor belongs to another sensor: the chart is then being
    rebuilt by the selection callback and must not be extended with this sensor's points.
    """
    if not cursor or cursor.get("entity_id") != entity_id or series is None:
        return None
    last_point = cursor.get("t") or 0
    last_alert = cursor.get("alert_t") or 0
    points = [p for p in series.get("points") or [] if (epoch_ms(p.get("t")) or 0) > last_point]
    marks = [m for m in alert_points(alerts, series) if (epoch_ms(m["t"]) or 0) > last_alert]
    if not points and not marks:
        return None
    return {"entity_id": entity_id, "points": points, "alerts": marks}


# The clientside callback: turns the server's increment into Plotly.extendTraces arguments and
# moves the cursor. It is idempotent (it re-filters by the cursor), so a repeated or late
# increment can never draw a point twice.
EXTEND_JS: Final = """
function(delta, cursor) {
  var NU = window.dash_clientside.no_update;
  if (!delta || !cursor || delta.entity_id !== cursor.entity_id) { return [NU, NU]; }
  var ms = function (item) { return Date.parse(item.t); };
  var pts = (delta.points || []).filter(function (p) { return ms(p) > cursor.t; });
  var als = (delta.alerts || []).filter(function (a) { return ms(a) > cursor.alert_t; });
  if (!pts.length && !als.length) { return [NU, NU]; }
  var update = {
    x: [pts.map(function (p) { return p.t; }), als.map(function (a) { return a.t; })],
    y: [pts.map(function (p) { return p.v; }), als.map(function (a) { return a.v; })]
  };
  var next = {
    entity_id: cursor.entity_id,
    t: pts.length ? ms(pts[pts.length - 1]) : cursor.t,
    alert_t: als.length ? Math.max.apply(null, als.map(ms)) : cursor.alert_t
  };
  return [[update, [0, 1], __MAX_POINTS__], next];
}
""".replace("__MAX_POINTS__", str(MAX_POINTS))


# --------------------------------------------------------------------------- #
# Zone heatmap
# --------------------------------------------------------------------------- #
def grid_layout(
    sensors: Sequence[Mapping[str, Any]], cols: int = HEATMAP_COLS, min_rows: int = HEATMAP_MIN_ROWS
) -> tuple[list[list[Mapping[str, Any] | None]], list[str]]:
    """Place sensors on the warehouse floor: one row per zone, wrapping after ``cols`` sensors.

    The grid is at least ``min_rows`` x ``cols`` (4 x 5 = 20 positions) and grows if a warehouse
    has more sensors, so none is ever dropped.

    Returns:
        ``(rows, row_labels)``; empty positions are ``None``.
    """
    by_zone: dict[str, list[Mapping[str, Any]]] = {}
    for sensor in sorted(sensors, key=lambda s: (str(s.get("zone")), str(s["entity_id"]))):
        by_zone.setdefault(str(sensor.get("zone") or "?"), []).append(sensor)
    rows: list[list[Mapping[str, Any] | None]] = []
    labels: list[str] = []
    for zone, members in by_zone.items():
        for start in range(0, len(members), cols):
            chunk = list(members[start : start + cols])
            rows.append(chunk + [None] * (cols - len(chunk)))
            labels.append(f"Zone {zone}" if start == 0 else "")
    while len(rows) < min_rows:
        rows.append([None] * cols)
        labels.append("")
    return rows, labels


def _outside(value: float, expected: Sequence[float] | None) -> bool:
    """Whether ``value`` lies outside the ``[low, high]`` range (``False`` without a range)."""
    if expected is None or len(expected) != 2:
        return False
    return not (expected[0] <= value <= expected[1])


def zone_heatmap(
    entities: Sequence[Mapping[str, Any]], warehouse_id: str | None, kind: str = "temperature"
) -> go.Figure:
    """Heatmap of the current reading of every ``kind`` sensor of a warehouse, as a floor grid.

    Cells show the sensor id and value; cells outside the sensor's acceptable range get a red
    outline and say so on hover.
    """
    sensors = [
        e
        for e in entities
        if e.get("source_type") == "warehouse"
        and str(e.get("warehouse_id")) == str(warehouse_id)
        and sensor_kind(e) == kind
        and isinstance(e.get("value"), int | float)
        and math.isfinite(e["value"])
    ]
    if not sensors:
        return empty_figure(f"No {kind} readings for this warehouse", height=340)
    rows, row_labels = grid_layout(sensors)
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    hover: list[list[str]] = []
    outlines: list[tuple[int, int]] = []
    for r, row in enumerate(rows):
        z_row: list[float | None] = []
        text_row: list[str] = []
        hover_row: list[str] = []
        for c, sensor in enumerate(row):
            if sensor is None:
                z_row.append(None)
                text_row.append("")
                hover_row.append("")
                continue
            value = float(sensor["value"])
            unit = sensor.get("unit")
            short = str(sensor["entity_id"]).removeprefix(f"{warehouse_id}-")
            shown = format_value_with_unit(value, unit, 1)
            expected = sensor.get("expected_range")
            outside = _outside(value, expected)
            if outside:
                outlines.append((r, c))
            z_row.append(value)
            text_row.append(f"<span style='font-size:11px'>{short}</span><br><b>{shown}</b>")
            range_text = (
                f"{expected[0]:g} to {expected[1]:g}"
                if expected and len(expected) == 2
                else PLACEHOLDER
            )
            hover_row.append(
                f"<b>{sensor['entity_id']}</b><br>{shown}<br>acceptable: {range_text}"
                + ("<br><b>OUTSIDE acceptable range</b>" if outside else "")
                + f"<br>{format_timestamp(sensor.get('timestamp'))} UTC"
            )
        z.append(z_row)
        text.append(text_row)
        hover.append(hover_row)
    values = [v for row in z for v in row if v is not None]
    low, high = min(values), max(values)
    if high - low < 1.0:  # a flat grid still needs a sensible colour range
        low, high = low - 0.5, high + 0.5
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(range(len(rows[0]))),
            y=list(range(len(rows))),
            text=text,
            texttemplate="%{text}",
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="RdYlBu",
            reversescale=True,
            zmin=low,
            zmax=high,
            xgap=3,
            ygap=3,
            hoverongaps=False,
            colorbar={"thickness": 10, "tickfont": {"color": MUTED_TEXT}},
        )
    )
    for r, c in outlines:
        fig.add_shape(
            type="rect",
            x0=c - 0.5,
            x1=c + 0.5,
            y0=r - 0.5,
            y1=r + 0.5,
            line={"color": CRITICAL_RED, "width": 3},
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(rows))),
        ticktext=row_labels,
        autorange="reversed",
        showgrid=False,
    )
    return apply_theme(
        fig,
        height=340,
        plot_bgcolor=PANEL_DARK,
        margin={"l": 64, "r": 8, "t": 8, "b": 8},
    )


# --------------------------------------------------------------------------- #
# Anomaly detail panel
# --------------------------------------------------------------------------- #
def match_alert(
    alerts: Sequence[Mapping[str, Any]], clicked_x: Any, tolerance_s: float = CLICK_TOLERANCE_S
) -> Mapping[str, Any] | None:
    """Return the alert closest in time to a clicked marker (within ``tolerance_s``), if any."""
    clicked = parse_timestamp(str(clicked_x).replace(" ", "T")) if clicked_x is not None else None
    if clicked is None:
        return None
    best, best_gap = None, tolerance_s
    for alert in alerts:
        moment = parse_timestamp(alert.get("timestamp"))
        if moment is None:
            continue
        gap = abs((moment - clicked).total_seconds())
        if gap <= best_gap:
            best, best_gap = alert, gap
    return best


def _feature_rows(features: Sequence[Mapping[str, Any]] | None) -> list[html.Div]:
    """Bars of the top contributing features (SHAP), widest first."""
    parsed = []
    for item in features or []:
        name = item.get("feature") or item.get("name")
        weight = item.get("shap_value", item.get("contribution", item.get("importance")))
        if name is None or not isinstance(weight, int | float) or not math.isfinite(weight):
            continue
        parsed.append((str(name), float(weight), item.get("feature_value")))
    parsed.sort(key=lambda row: abs(row[1]), reverse=True)
    parsed = parsed[:TOP_FEATURES]
    if not parsed:
        return [html.Div("No feature explanation was stored.", style={"color": MUTED_TEXT})]
    widest = max(abs(w) for _, w, _ in parsed) or 1.0
    rows = []
    for name, weight, value in parsed:
        color = CRITICAL_RED if weight >= 0 else INFO_BLUE
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(name, className="small"),
                            html.Span(
                                f"{weight:+.3f}", className="small", style={"color": MUTED_TEXT}
                            ),
                        ],
                        className="d-flex justify-content-between",
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{max(abs(weight) / widest * 100, 2):.0f}%",
                                "height": "6px",
                                "backgroundColor": color,
                                "borderRadius": "3px",
                            }
                        ),
                        style={"backgroundColor": BACKGROUND_DARK, "borderRadius": "3px"},
                        title=f"feature value {value}" if value is not None else None,
                    ),
                ],
                className="mb-2 feature-row",
            )
        )
    return rows


def anomaly_card(alert: Mapping[str, Any] | None, unit: str | None = None) -> dbc.Card:
    """Build the detail card of one alert: what happened, the explanation and the advised action."""
    if not alert:
        return dbc.Card(
            dbc.CardBody(
                "Click a red anomaly marker on the chart to see its details.",
                style={"color": MUTED_TEXT},
            ),
            id="wh-detail-card",
        )
    expected = alert.get("expected_range")
    range_text = (
        f"{expected[0]:g} to {expected[1]:g}" if expected and len(expected) == 2 else PLACEHOLDER
    )
    status = (
        "False positive"
        if alert.get("false_positive_marked")
        else "Resolved" if alert.get("is_resolved") else "Open"
    )
    facts = [
        ("Entity", str(alert.get("entity_id") or PLACEHOLDER)),
        ("Time", f"{format_timestamp(alert.get('timestamp'), with_date=True)} UTC"),
        ("Reading", format_value_with_unit(alert.get("raw_value"), unit, 2)),
        ("Acceptable range", range_text),
        (
            "Severity score",
            (
                PLACEHOLDER
                if alert.get("severity_score") is None
                else f"{alert['severity_score']:.2f}"
            ),
        ),
        ("Detections", str(alert.get("occurrence_count") or 1)),
        ("Status", status),
    ]
    action = alert.get("recommended_action")
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    severity_badge(alert.get("severity")),
                    html.Span(
                        f"{alert.get('anomaly_type') or 'anomaly'} anomaly",
                        className="ms-2 fw-semibold",
                    ),
                ]
            ),
            dbc.CardBody(
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
                                for label, value in facts
                            ],
                            xs=12,
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    "Top contributing features",
                                    className="small text-uppercase mb-2",
                                    style={"color": MUTED_TEXT},
                                ),
                                *_feature_rows(alert.get("top_contributing_features")),
                            ],
                            xs=12,
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    "Recommended action",
                                    className="small text-uppercase mb-2",
                                    style={"color": MUTED_TEXT},
                                ),
                                html.Div(
                                    action or "No recommendation was stored.",
                                    className="p-2 rounded",
                                    style={
                                        "backgroundColor": hex_to_rgba(INFO_BLUE, 0.12),
                                        "border": f"1px solid {INFO_BLUE}",
                                        "color": TEXT_COLOR,
                                    },
                                ),
                            ],
                            xs=12,
                            md=4,
                        ),
                    ]
                )
            ),
        ],
        id="wh-detail-card",
        style={"borderColor": NEUTRAL_GREY},
    )
