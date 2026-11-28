"""Building blocks of the Inventory page (Day 20).

Pure functions (no Dash callbacks) so they can be unit-tested directly:

* :func:`top_active_skus` / :func:`stock_gauges` - the ten busiest SKUs as gauges.
* :func:`resolve_sku_selection` - the cascading warehouse -> zone -> SKU dropdowns.
* :func:`transaction_buckets` / :func:`waterfall_figure` - net stock change of one SKU per minute.
* :func:`inventory_anomaly_rows` / :func:`anomaly_table` - sudden drops and ghost stock.

Transactions come from the feature cache as points ``{"t", "v" (stock level), "d" (change),
"tx" (transaction type)}``. Times are UTC.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from src.dashboard.components.overview_widgets import hex_to_rgba
from src.dashboard.components.theme import (
    BACKGROUND_DARK,
    CRITICAL_RED,
    INFO_BLUE,
    MUTED_TEXT,
    NORMAL_GREEN,
    TEXT_COLOR,
    WARNING_AMBER,
    apply_theme,
    empty_figure,
)
from src.dashboard.utils import PLACEHOLDER, format_timestamp, parse_timestamp, severity_badge

TOP_SKUS: Final = 10
GAUGE_COLUMNS: Final = 5
GAUGE_ROW_HEIGHT: Final = 200
WINDOW_SECONDS: Final = 3600.0
BUCKET_SECONDS: Final = 60.0
#: Largest gap between an alert and the transaction it is about.
MATCH_TOLERANCE_S: Final = 1.5
#: Anomaly types of the inventory simulator and what an analyst calls them.
EVENT_LABELS: Final[dict[str, str]] = {"point": "Sudden drop", "contextual": "Ghost stock"}
STATUS_LABELS: Final[dict[str, str]] = {
    "open": "Open",
    "false_positive": "False positive",
    "resolved": "Resolved",
}
RESTOCK_COLOR: Final = NORMAL_GREEN
SHIPMENT_COLOR: Final = INFO_BLUE
ANOMALY_COLOR: Final = CRITICAL_RED
TABLE_ROWS: Final = 15


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _keep_or_first(current: str | None, values: Sequence[str]) -> str | None:
    if current in values:
        return current
    return values[0] if values else None


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def resolve_sku_selection(
    entities: Sequence[Mapping[str, Any]],
    warehouse: str | None = None,
    zone: str | None = None,
    sku: str | None = None,
) -> dict[str, Any]:
    """Compute the three cascading dropdowns from the live SKUs.

    A current choice that is still valid is kept; otherwise the first option is taken (``None``
    when there is nothing to choose).

    Returns:
        ``{"warehouse_options", "zone_options", "sku_options", "warehouse", "zone", "sku"}``.
    """
    skus = [e for e in entities if e.get("source_type") == "inventory"]
    warehouses = sorted({str(e["warehouse_id"]) for e in skus if e.get("warehouse_id")})
    chosen_warehouse = _keep_or_first(warehouse, warehouses)
    in_warehouse = [e for e in skus if str(e.get("warehouse_id")) == chosen_warehouse]
    zones = sorted({str(e["zone"]) for e in in_warehouse if e.get("zone")})
    chosen_zone = _keep_or_first(zone, zones)
    in_zone = sorted(
        str(e["entity_id"]) for e in in_warehouse if not zones or str(e.get("zone")) == chosen_zone
    )
    chosen_sku = _keep_or_first(sku, in_zone)

    def options(values: Sequence[str], prefix: str = "") -> list[dict[str, str]]:
        return [{"label": f"{prefix}{v}", "value": v} for v in values]

    return {
        "warehouse_options": options(warehouses),
        "zone_options": options(zones, "Zone "),
        "sku_options": options(in_zone),
        "warehouse": chosen_warehouse,
        "zone": chosen_zone,
        "sku": chosen_sku,
    }


# --------------------------------------------------------------------------- #
# Gauges
# --------------------------------------------------------------------------- #
def top_active_skus(
    entities: Sequence[Mapping[str, Any]],
    series: Mapping[str, Sequence[Mapping[str, Any]]],
    alerts: Sequence[Mapping[str, Any]] = (),
    n: int = TOP_SKUS,
) -> list[dict[str, Any]]:
    """Return the ``n`` SKUs with the most transactions in the window, busiest first.

    Ties go to the larger moved volume. ``has_alert`` is true when the SKU has an alert that an
    analyst has not dismissed.
    """
    flagged = {
        str(a.get("entity_id"))
        for a in alerts
        if not a.get("false_positive_marked") and not a.get("is_resolved")
    }
    rows = []
    for entity in entities:
        if entity.get("source_type") != "inventory":
            continue
        sku = str(entity["entity_id"])
        points = series.get(sku, ())
        details = entity.get("details") or {}
        low, high = _number(details.get("min_stock")), _number(details.get("max_stock"))
        if (low is None or high is None) and entity.get("expected_range"):
            low, high = entity["expected_range"]
        level = _number(entity.get("value"))
        if level is None or low is None or high is None or high <= low:
            continue
        rows.append(
            {
                "sku_id": sku,
                "warehouse_id": entity.get("warehouse_id"),
                "zone": entity.get("zone"),
                "level": level,
                "min_stock": float(low),
                "max_stock": float(high),
                "reorder_point": _number(details.get("reorder_point")),
                "transactions": len(points),
                "volume": sum(abs(_number(p.get("d")) or 0.0) for p in points),
                "has_alert": sku in flagged,
            }
        )
    rows.sort(key=lambda r: (-r["transactions"], -r["volume"], r["sku_id"]))
    return rows[: max(0, n)]


def stock_gauges(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    """One gauge per SKU in a grid: current stock on a 0..max scale.

    The bands show the minimum stock (red), the span down to the reorder point (amber) and the
    healthy range; the line marks the reorder point. SKUs with an open alert are labelled.
    """
    if not rows:
        return empty_figure("No SKUs are reporting yet", height=GAUGE_ROW_HEIGHT)
    n_rows = math.ceil(len(rows) / GAUGE_COLUMNS)
    fig = go.Figure()
    for index, row in enumerate(rows):
        low, high = row["min_stock"], row["max_stock"]
        reorder = row.get("reorder_point")
        steps = [{"range": [0, low], "color": hex_to_rgba(CRITICAL_RED, 0.35)}]
        if reorder is not None and low < reorder < high:
            steps += [
                {"range": [low, reorder], "color": hex_to_rgba(WARNING_AMBER, 0.35)},
                {"range": [reorder, high], "color": hex_to_rgba(NORMAL_GREEN, 0.22)},
            ]
        else:
            steps.append({"range": [low, high], "color": hex_to_rgba(NORMAL_GREEN, 0.22)})
        label = str(row["sku_id"])
        if row.get("has_alert"):
            label += f" <span style='color:{CRITICAL_RED}'>· ALERT</span>"
        gauge: dict[str, Any] = {
            "axis": {"range": [0, high], "tickfont": {"size": 9}, "nticks": 4},
            "bar": {"color": TEXT_COLOR, "thickness": 0.28},
            "bgcolor": BACKGROUND_DARK,
            "borderwidth": 0,
            "steps": steps,
        }
        if reorder is not None:
            gauge["threshold"] = {
                "line": {"color": WARNING_AMBER, "width": 3},
                "thickness": 0.8,
                "value": reorder,
            }
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=row["level"],
                number={"font": {"size": 22}, "valueformat": ",.0f"},
                title={"text": label, "font": {"size": 12}},
                gauge=gauge,
                domain={"row": index // GAUGE_COLUMNS, "column": index % GAUGE_COLUMNS},
            )
        )
    return apply_theme(
        fig,
        height=GAUGE_ROW_HEIGHT * n_rows + 20,
        grid={"rows": n_rows, "columns": GAUGE_COLUMNS, "pattern": "independent"},
        margin={"l": 24, "r": 24, "t": 36, "b": 8},
    )


# --------------------------------------------------------------------------- #
# Waterfall
# --------------------------------------------------------------------------- #
def _alert_times(alerts: Sequence[Mapping[str, Any]]) -> list[datetime]:
    return [
        m
        for a in alerts
        if not a.get("false_positive_marked") and (m := parse_timestamp(a.get("timestamp")))
    ]


def transaction_buckets(
    points: Sequence[Mapping[str, Any]],
    alerts: Sequence[Mapping[str, Any]] = (),
    bucket_s: float = BUCKET_SECONDS,
) -> list[dict[str, Any]]:
    """Group transactions into ``bucket_s`` buckets and chain them like a waterfall.

    Returns one dict per bucket that has transactions, oldest first: ``start`` (bucket start, UTC
    datetime), ``net`` (summed change), ``begin`` / ``end`` (stock level before and after),
    ``count``, ``anomalous`` (an alert falls in the bucket) and ``kind`` (``restock``,
    ``shipment`` or ``anomaly``). ``begin`` of the first bucket is the level before the first
    transaction; points without a change are skipped.
    """
    parsed = []
    for point in points:
        moment, change, level = (
            parse_timestamp(point.get("t")),
            _number(point.get("d")),
            _number(point.get("v")),
        )
        if moment is not None and change is not None and level is not None:
            parsed.append((moment, change, level))
    parsed.sort(key=lambda item: item[0])
    if not parsed:
        return []
    marks = _alert_times(alerts)
    buckets: dict[int, dict[str, Any]] = {}
    for moment, change, level in parsed:
        key = int(moment.timestamp() // bucket_s)
        bucket = buckets.get(key)
        if bucket is None:
            start = datetime.fromtimestamp(key * bucket_s, UTC)
            bucket = buckets[key] = {
                "start": start,
                "net": 0.0,
                "begin": level - change,
                "end": level,
                "count": 0,
                "anomalous": any(
                    start - timedelta(seconds=MATCH_TOLERANCE_S)
                    <= m
                    < start + timedelta(seconds=bucket_s + MATCH_TOLERANCE_S)
                    for m in marks
                ),
            }
        bucket["net"] += change
        bucket["end"] = level
        bucket["count"] += 1
    result = [buckets[k] for k in sorted(buckets)]
    for bucket in result:
        bucket["kind"] = (
            "anomaly"
            if bucket["anomalous"]
            else "restock" if bucket["end"] >= bucket["begin"] else "shipment"
        )
    return result


def waterfall_figure(
    buckets: Sequence[Mapping[str, Any]],
    *,
    sku_id: str | None = None,
    reorder_point: float | None = None,
    min_stock: float | None = None,
) -> go.Figure:
    """Waterfall of the net stock change per bucket.

    Each bar floats from the level before the bucket to the level after it: green for a net
    increase (restock), blue for a net decrease (shipment), red where an alert falls in the
    bucket. Bars are separate traces so the legend names them.
    """
    if not buckets:
        return empty_figure("No transactions in the last hour", height=360)
    fig = go.Figure()
    groups = (
        ("restock", "Restock (net increase)", RESTOCK_COLOR),
        ("shipment", "Shipment (net decrease)", SHIPMENT_COLOR),
        ("anomaly", "Anomaly", ANOMALY_COLOR),
    )
    for kind, name, color in groups:
        chosen = [b for b in buckets if b["kind"] == kind]
        if not chosen:
            continue
        fig.add_trace(
            go.Bar(
                x=[b["start"] for b in chosen],
                y=[b["net"] for b in chosen],
                base=[b["begin"] for b in chosen],
                name=name,
                marker={"color": color},
                width=BUCKET_SECONDS * 1000 * 0.8,
                customdata=[[b["begin"], b["end"], b["count"]] for b in chosen],
                hovertemplate=(
                    "%{x|%H:%M} UTC<br>net %{y:+,.0f} units · %{customdata[2]} transactions"
                    "<br>level %{customdata[0]:,.0f} → %{customdata[1]:,.0f}<extra>"
                    + name
                    + "</extra>"
                ),
            )
        )
    first, last = buckets[0]["start"], buckets[-1]["start"] + timedelta(seconds=BUCKET_SECONDS)
    for value, label, color in (
        (reorder_point, "Reorder point", WARNING_AMBER),
        (min_stock, "Minimum stock", CRITICAL_RED),
    ):
        if value is not None:
            fig.add_trace(
                go.Scatter(
                    x=[first, last],
                    y=[value, value],
                    mode="lines",
                    line={"color": color, "dash": "dash", "width": 1.5},
                    name=label,
                    hovertemplate=f"{label}: {value:,.0f}<extra></extra>",
                )
            )
    return apply_theme(
        fig,
        height=360,
        title={"text": f"{sku_id or 'SKU'} · stock level per minute", "font": {"size": 14}},
        barmode="overlay",
        xaxis={"title": "Time (UTC)"},
        yaxis={"title": "Units in stock"},
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 56, "r": 16, "t": 44, "b": 40},
    )


# --------------------------------------------------------------------------- #
# Anomaly table
# --------------------------------------------------------------------------- #
def change_at(points: Sequence[Mapping[str, Any]], when: Any) -> float | None:
    """Stock change of the transaction at ``when`` (within ``MATCH_TOLERANCE_S``), if known."""
    target = parse_timestamp(when)
    if target is None:
        return None
    best, best_gap = None, MATCH_TOLERANCE_S
    for point in points:
        moment = parse_timestamp(point.get("t"))
        if moment is None:
            continue
        gap = abs((moment - target).total_seconds())
        change = _number(point.get("d"))
        if change is not None and gap <= best_gap:
            best, best_gap = change, gap
    return best


def inventory_anomaly_rows(
    alerts: Sequence[Mapping[str, Any]],
    series: Mapping[str, Sequence[Mapping[str, Any]]],
    limit: int = TABLE_ROWS,
) -> list[dict[str, Any]]:
    """Rows for the anomaly table, newest first: when, SKU, what happened and by how much.

    ``change`` is looked up in the SKU's recent transactions and is ``None`` when the
    transaction is no longer in the window.
    """
    rows = []
    for alert in alerts:
        sku = str(alert.get("entity_id"))
        kind = alert.get("anomaly_type")
        status = (
            "false_positive"
            if alert.get("false_positive_marked")
            else "resolved" if alert.get("is_resolved") else "open"
        )
        rows.append(
            {
                "timestamp": alert.get("timestamp"),
                "sku_id": sku,
                "event": EVENT_LABELS.get(str(kind), str(kind or "Anomaly")),
                "change": change_at(series.get(sku, ()), alert.get("timestamp")),
                "severity": alert.get("severity"),
                "status": status,
            }
        )
    rows.sort(key=lambda r: r["timestamp"] or "", reverse=True)
    return rows[: max(0, limit)]


def anomaly_table(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Table of sudden drops and ghost-stock events (a friendly empty state when none)."""
    if not rows:
        return html.Div(
            "No inventory anomalies in the last 24 hours.",
            id="inv-anomaly-empty",
            style={"color": MUTED_TEXT},
        )
    header = html.Thead(
        html.Tr(
            [html.Th(h) for h in ("Time (UTC)", "SKU", "Event", "Magnitude", "Severity", "Status")]
        )
    )
    body = []
    for row in rows:
        change = row.get("change")
        magnitude = PLACEHOLDER if change is None else f"{change:+,.0f} units"
        body.append(
            html.Tr(
                [
                    html.Td(format_timestamp(row.get("timestamp"), with_date=False)),
                    html.Td(row["sku_id"]),
                    html.Td(row["event"]),
                    html.Td(magnitude),
                    html.Td(severity_badge(row.get("severity"))),
                    html.Td(STATUS_LABELS.get(str(row.get("status")), PLACEHOLDER)),
                ]
            )
        )
    return dbc.Table(
        [header, html.Tbody(body)],
        size="sm",
        hover=True,
        responsive=True,
        className="mb-0",
        id="inv-anomaly-table",
    )
