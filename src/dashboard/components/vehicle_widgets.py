"""Building blocks of the Vehicles page (Day 20).

Pure functions (no Dash callbacks) so they can be unit-tested directly:

* :func:`build_fleet` - merge the live vehicle readings with their alerts into one row per vehicle,
  including the on-route / deviation / delayed status.
* :func:`deviation_markers` - GPS position of each alert (taken from the vehicle's trail).
* :func:`fleet_map` - the map: planned routes, GPS trails, vehicle markers and red X alert marks.
* :func:`vehicle_info_card` / :func:`fleet_table` - the detail panel and the summary table.

The map uses ``plotly.graph_objects.Scattermap`` (MapLibre) with the OpenStreetMap tile style, which
needs no API key. It replaces ``Scattermapbox``, which Plotly 7 removed.

Status
------
A vehicle is **deviating** when it is at least ``DEVIATION_DISTANCE_M`` (500 m, the simulator's
corridor) from its route, or its adherence score falls below ``DEVIATION_ADHERENCE``. Otherwise it
is **delayed** when it has an alert from the last ``RECENT_ALERT_MINUTES`` minutes that an analyst
has not dismissed (a gradual slow-down, a stop or a speed anomaly). Otherwise it is **on route**.
Each status shows with a text label, never by colour alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Final

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from src.dashboard.components.theme import (
    CRITICAL_RED,
    GRID_COLOR,
    INFO_BLUE,
    MUTED_TEXT,
    NORMAL_GREEN,
    WARNING_AMBER,
    apply_theme,
    empty_figure,
)
from src.dashboard.utils import (
    PLACEHOLDER,
    format_age,
    format_timestamp,
    format_value_with_unit,
    parse_timestamp,
    severity_badge,
)

ON_ROUTE: Final = "on-route"
DEVIATION: Final = "deviation"
DELAYED: Final = "delayed"
STATUS_ORDER: Final[tuple[str, ...]] = (DEVIATION, DELAYED, ON_ROUTE)
STATUS_LABELS: Final[dict[str, str]] = {
    ON_ROUTE: "On route",
    DEVIATION: "Deviation",
    DELAYED: "Delayed",
}
STATUS_COLORS: Final[dict[str, str]] = {
    ON_ROUTE: NORMAL_GREEN,
    DEVIATION: CRITICAL_RED,
    DELAYED: WARNING_AMBER,
}
STATUS_BOOTSTRAP: Final[dict[str, str]] = {ON_ROUTE: "success", DEVIATION: "danger", DELAYED: "warning"}

DEVIATION_DISTANCE_M: Final = 500.0
DEVIATION_ADHERENCE: Final = 0.25
RECENT_ALERT_MINUTES: Final = 15.0
#: Longest gap between an alert and the GPS fix used as its position.
MATCH_TOLERANCE_S: Final = 30.0
TRAIL_SECONDS: Final = 1800.0
MAX_TRAILS: Final = 30
ALL_VEHICLES: Final = "all"
MAP_HEIGHT: Final = 520
ROUTE_COLOR: Final = "#7f8c8d"
FOCUS_ZOOM: Final = 9.0


# --------------------------------------------------------------------------- #
# Fleet model
# --------------------------------------------------------------------------- #
def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def vehicle_status(
    vehicle: Mapping[str, Any], recent_alerts: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Status of one vehicle: ``deviation``, ``delayed`` or ``on-route`` (see the module doc)."""
    details = vehicle.get("details") or {}
    distance = _number(details.get("distance_from_route_m"))
    adherence = _number(details.get("route_adherence_score"))
    if (distance is not None and distance >= DEVIATION_DISTANCE_M) or (
        distance is None and adherence is not None and adherence < DEVIATION_ADHERENCE
    ):
        return DEVIATION
    return DELAYED if recent_alerts else ON_ROUTE


def build_fleet(
    entities: Sequence[Mapping[str, Any]],
    alerts: Sequence[Mapping[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    """One row per live vehicle, most urgent first.

    Args:
        entities: Latest readings (``DashboardDataService.get_entities("vehicle")``).
        alerts: Alerts of any entity (the vehicle's own are picked out by ``entity_id``).
        now: Reference time; alerts newer than ``RECENT_ALERT_MINUTES`` before it set the status.

    Returns:
        Dicts with ``vehicle_id, driver_id, route_id, latitude, longitude, heading, speed_kmh,
        adherence, distance_m, status, timestamp, active_count, last_anomaly``. Vehicles without a
        position are kept (the table lists them) with ``latitude`` / ``longitude`` ``None``.
    """
    by_vehicle: dict[str, list[Mapping[str, Any]]] = {}
    for alert in alerts:
        if alert.get("entity_type") in (None, "vehicle"):
            by_vehicle.setdefault(str(alert.get("entity_id")), []).append(alert)
    cutoff = now - timedelta(minutes=RECENT_ALERT_MINUTES)
    fleet = []
    for entity in entities:
        if entity.get("source_type") != "vehicle":
            continue
        vid = str(entity["entity_id"])
        mine = by_vehicle.get(vid, [])
        live = [a for a in mine if not a.get("false_positive_marked") and not a.get("is_resolved")]
        recent = [a for a in live if (parse_timestamp(a.get("timestamp")) or cutoff) >= cutoff]
        details = entity.get("details") or {}
        newest = max(mine, key=lambda a: a.get("timestamp") or "", default=None)
        fleet.append(
            {
                "vehicle_id": vid,
                "driver_id": details.get("driver_id"),
                "route_id": details.get("route_id"),
                "latitude": _number(details.get("latitude")),
                "longitude": _number(details.get("longitude")),
                "heading": _number(details.get("heading")),
                "speed_kmh": _number(entity.get("value")),
                "adherence": _number(details.get("route_adherence_score")),
                "distance_m": _number(details.get("distance_from_route_m")),
                "status": vehicle_status(entity, recent),
                "timestamp": entity.get("timestamp"),
                "active_count": len(live),
                "last_anomaly": dict(newest) if newest is not None else None,
            }
        )
    fleet.sort(key=lambda v: (STATUS_ORDER.index(v["status"]), v["vehicle_id"]))
    return fleet


def match_trail_point(
    points: Sequence[Mapping[str, Any]], when: Any, tolerance_s: float = MATCH_TOLERANCE_S
) -> Mapping[str, Any] | None:
    """The GPS fix closest in time to ``when`` (within ``tolerance_s``) that has a position."""
    target = parse_timestamp(when)
    if target is None:
        return None
    best, best_gap = None, tolerance_s
    for point in points:
        moment = parse_timestamp(point.get("t"))
        if moment is None or _number(point.get("lat")) is None or _number(point.get("lon")) is None:
            continue
        gap = abs((moment - target).total_seconds())
        if gap <= best_gap:
            best, best_gap = point, gap
    return best


def deviation_markers(
    alerts: Sequence[Mapping[str, Any]], trails: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Map positions of the vehicle alerts that an analyst has not dismissed.

    An alert carries no coordinates, so its position is the GPS fix of that vehicle closest in
    time (within ``MATCH_TOLERANCE_S``). Alerts without such a fix are left out.
    """
    marks = []
    for alert in alerts:
        if alert.get("false_positive_marked"):
            continue
        point = match_trail_point(trails.get(str(alert.get("entity_id")), ()), alert.get("timestamp"))
        if point is None:
            continue
        marks.append(
            {
                "vehicle_id": str(alert.get("entity_id")),
                "latitude": float(point["lat"]),
                "longitude": float(point["lon"]),
                "timestamp": alert.get("timestamp"),
                "severity": str(alert.get("severity") or ""),
                "anomaly_type": alert.get("anomaly_type"),
            }
        )
    return marks


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
def fit_view(points: Sequence[tuple[float, float]]) -> tuple[dict[str, float], float]:
    """Return a map ``center`` and ``zoom`` that roughly fit all ``(lat, lon)`` points."""
    if not points:
        return {"lat": 30.0, "lon": 0.0}, 1.0
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    center = {"lat": (min(lats) + max(lats)) / 2, "lon": (min(lons) + max(lons)) / 2}
    span = max(max(lats) - min(lats), (max(lons) - min(lons)) * 0.6, 0.02)
    zoom = max(1.0, min(12.0, math.log2(360.0 / span) - 1.0))
    return center, round(zoom, 2)


def fleet_map(
    fleet: Sequence[Mapping[str, Any]],
    routes: Mapping[str, Mapping[str, Any]],
    marks: Sequence[Mapping[str, Any]] = (),
    trails: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    focus: str | None = ALL_VEHICLES,
) -> go.Figure:
    """Build the live map.

    Args:
        fleet: Rows of :func:`build_fleet`.
        routes: Planned routes by id (``{"waypoints": [(lat, lon), ...], ...}``).
        marks: Alert positions from :func:`deviation_markers` (red X).
        trails: Recent GPS fixes per vehicle (``{"t", "lat", "lon"}``), drawn as a thin trail.
        focus: A vehicle id to centre on, or ``"all"`` to fit the whole fleet. The map keeps the
            user's pan and zoom between updates as long as ``focus`` does not change.
    """
    located = [v for v in fleet if v["latitude"] is not None and v["longitude"] is not None]
    if not located:
        return empty_figure("No vehicles are reporting a position yet", height=MAP_HEIGHT)
    fig = go.Figure()
    used_routes = {str(v["route_id"]) for v in located if v.get("route_id") in routes}
    for route_id in sorted(used_routes):
        route = routes[route_id]
        way = route["waypoints"]
        fig.add_trace(
            go.Scattermap(
                lat=[p[0] for p in way],
                lon=[p[1] for p in way],
                mode="lines",
                line={"width": 3, "color": ROUTE_COLOR},
                name="Planned route",
                legendgroup="route",
                showlegend=route_id == sorted(used_routes)[0],
                hovertemplate=(
                    f"{route_id}: {route.get('origin', '')} → {route.get('destination', '')}"
                    "<extra></extra>"
                ),
            )
        )
    for vid, points in (trails or {}).items():
        fixes = [p for p in points if _number(p.get("lat")) is not None]
        if len(fixes) > 1:
            fig.add_trace(
                go.Scattermap(
                    lat=[p["lat"] for p in fixes],
                    lon=[p["lon"] for p in fixes],
                    mode="lines",
                    line={"width": 2, "color": INFO_BLUE},
                    opacity=0.55,
                    name="GPS trail",
                    legendgroup="trail",
                    showlegend=vid == next(iter(trails)),
                    hoverinfo="skip",
                )
            )
    if marks:
        fig.add_trace(
            go.Scattermap(
                lat=[m["latitude"] for m in marks],
                lon=[m["longitude"] for m in marks],
                mode="markers+text",
                marker={"size": 17, "color": CRITICAL_RED},
                text=["✕"] * len(marks),
                textfont={"size": 14, "color": "#ffffff"},
                name="Alert position (✕)",
                customdata=[m["vehicle_id"] for m in marks],
                hovertext=[
                    f"{m['vehicle_id']} · {m['severity']} {m.get('anomaly_type') or 'anomaly'}"
                    f"<br>{format_timestamp(m['timestamp'])} UTC"
                    for m in marks
                ],
                hoverinfo="text",
            )
        )
    for status in STATUS_ORDER:
        group = [v for v in located if v["status"] == status]
        if not group:
            continue
        fig.add_trace(
            go.Scattermap(
                lat=[v["latitude"] for v in group],
                lon=[v["longitude"] for v in group],
                mode="markers",
                marker={"size": 15, "color": STATUS_COLORS[status]},
                name=STATUS_LABELS[status],
                customdata=[v["vehicle_id"] for v in group],
                hovertext=[
                    f"{v['vehicle_id']} · {STATUS_LABELS[status]}"
                    f"<br>{format_value_with_unit(v['speed_kmh'], 'km/h', 0)}"
                    for v in group
                ],
                hoverinfo="text",
            )
        )
    chosen = next((v for v in located if v["vehicle_id"] == focus), None)
    if chosen is not None:
        center, zoom = {"lat": chosen["latitude"], "lon": chosen["longitude"]}, FOCUS_ZOOM
    else:
        every = [(v["latitude"], v["longitude"]) for v in located]
        for route in (routes[r] for r in used_routes):
            every.extend((p[0], p[1]) for p in route["waypoints"])
        center, zoom = fit_view(every)
    apply_theme(
        fig,
        height=MAP_HEIGHT,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        map={"style": "open-street-map", "center": center, "zoom": zoom},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(34,34,34,0.85)",
        },
        uirevision=str(focus or ALL_VEHICLES),
        clickmode="event",
    )
    return fig


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def status_badge(status: str) -> dbc.Badge:
    """Badge with the status as text (colour only reinforces it)."""
    return dbc.Badge(
        STATUS_LABELS.get(status, status), color=STATUS_BOOTSTRAP.get(status, "secondary"), pill=True
    )


def _fact(label: str, value: Any) -> html.Div:
    return html.Div(
        [html.Span(f"{label}: ", style={"color": MUTED_TEXT}), html.Span(value)],
        className="small mb-1",
    )


def vehicle_info_card(
    vehicle: Mapping[str, Any] | None,
    route: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dbc.Card:
    """Detail panel of one vehicle (or a hint to pick one)."""
    if not vehicle:
        return dbc.Card(
            dbc.CardBody(
                "Click a vehicle on the map, or choose one above, to see its details.",
                style={"color": MUTED_TEXT},
            ),
            id="veh-info-card",
        )
    last = vehicle.get("last_anomaly")
    if last:
        last_line: Any = [
            severity_badge(last.get("severity")),
            html.Span(
                f" {last.get('anomaly_type') or 'anomaly'} · "
                f"{format_timestamp(last.get('timestamp'), with_date=True)} UTC",
                className="ms-1",
            ),
        ]
    else:
        last_line = "None"
    route_text = vehicle.get("route_id") or PLACEHOLDER
    if route:
        route_text = f"{route_text} ({route.get('origin')} → {route.get('destination')})"
    adherence = vehicle.get("adherence")
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(vehicle["vehicle_id"], className="fw-semibold me-2"),
                    status_badge(str(vehicle.get("status"))),
                ]
            ),
            dbc.CardBody(
                [
                    _fact("Driver", vehicle.get("driver_id") or PLACEHOLDER),
                    _fact("Route", route_text),
                    _fact("Speed", format_value_with_unit(vehicle.get("speed_kmh"), "km/h", 1)),
                    _fact(
                        "Route adherence",
                        PLACEHOLDER if adherence is None else f"{adherence:.2f} (1.00 = on route)",
                    ),
                    _fact(
                        "Distance from route",
                        format_value_with_unit(vehicle.get("distance_m"), "m", 0),
                    ),
                    _fact("Last update", format_age(vehicle.get("timestamp"), now)),
                    _fact("Active anomalies", str(vehicle.get("active_count", 0))),
                    _fact("Last anomaly", last_line),
                ]
            ),
        ],
        id="veh-info-card",
    )


def fleet_table(fleet: Sequence[Mapping[str, Any]], now: datetime | None = None) -> Any:
    """Table of all vehicles: status, last update and number of active anomalies."""
    if not fleet:
        return html.Div("No vehicles are reporting yet.", style={"color": MUTED_TEXT})
    header = html.Thead(
        html.Tr(
            [
                html.Th(h)
                for h in ("Vehicle", "Driver", "Route", "Status", "Speed", "Last update", "Active")
            ]
        )
    )
    rows = [
        html.Tr(
            [
                html.Td(v["vehicle_id"]),
                html.Td(v.get("driver_id") or PLACEHOLDER),
                html.Td(v.get("route_id") or PLACEHOLDER),
                html.Td(status_badge(str(v["status"]))),
                html.Td(format_value_with_unit(v.get("speed_kmh"), "km/h", 0)),
                html.Td(format_age(v.get("timestamp"), now)),
                html.Td(str(v.get("active_count", 0))),
            ],
            style={"borderLeft": f"4px solid {STATUS_COLORS.get(v['status'], GRID_COLOR)}"},
        )
        for v in fleet
    ]
    return dbc.Table(
        [header, html.Tbody(rows)],
        size="sm",
        hover=True,
        responsive=True,
        className="mb-0",
        id="veh-fleet-table",
    )
