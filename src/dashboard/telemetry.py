"""Turn feature-cache windows into the plain dicts the dashboard plots (Day 18).

Shared by the in-process data source and the API's ``/api/v1/telemetry`` routes, so both return
exactly the same shape. Nothing here imports torch: reads go through
:meth:`~src.kafka.feature_cache.FeatureCache.peek`, which is not counted as a cache hit or miss.

A point is one window's newest reading (``signals[signal].last``) at the window's event time.
Times are event times (UTC ISO 8601), so replays and fast-forward simulation plot correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from src.kafka.feature_cache import FeatureCache
    from src.kafka.stream_processor import WindowedFeatures

#: The signal plotted per source type (the first signal the aggregator tracks).
PRIMARY_SIGNAL: Final[dict[str, str]] = {
    "warehouse": "value",
    "vehicle": "speed_kmh",
    "inventory": "new_level",
}
DEFAULT_UNITS: Final[dict[str, str]] = {"vehicle": "km/h", "inventory": "units"}
_MAX_POINTS: Final = 10_000


def _iso(moment: Any) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def expected_range(source_type: str, signal: str, context: dict[str, Any]) -> list[float] | None:
    """Healthy ``[low, high]`` of the plotted signal, or ``None`` if the context has none."""
    try:
        if source_type == "warehouse":
            low, high = context["acceptable_range"]
        elif source_type == "inventory":
            low, high = context["min_stock"], context["max_stock"]
        else:
            from src.models.feature_engineering import acceptable_range

            found = acceptable_range(source_type, signal, context)
            return [found[0], found[1]] if found else None
        low, high = float(low), float(high)
    except (KeyError, TypeError, ValueError, ImportError):
        return None
    return [low, high] if high > low else None


def _signal_of(window: WindowedFeatures) -> str | None:
    wanted = PRIMARY_SIGNAL.get(window.source_type)
    if wanted in window.signals:
        return wanted
    return next(iter(window.signals), None)


def _details(window: WindowedFeatures, context: dict[str, Any]) -> dict[str, Any]:
    """Source-specific extras: a vehicle's position and route, an SKU's stock band.

    Warehouse sensors have none. Missing context fields are simply left out.
    """
    if window.source_type == "vehicle":
        adherence = window.signals.get("route_adherence_score")
        wanted = {
            "latitude": context.get("latitude"),
            "longitude": context.get("longitude"),
            "heading": context.get("heading"),
            "route_id": context.get("route_id"),
            "driver_id": context.get("driver_id"),
            "distance_from_route_m": context.get("distance_from_route_m"),
            "route_adherence_score": adherence.last if adherence is not None else None,
        }
    elif window.source_type == "inventory":
        change = window.signals.get("change_amount")
        wanted = {
            "transaction_type": context.get("transaction_type"),
            "change_amount": change.last if change is not None else None,
            "min_stock": context.get("min_stock"),
            "max_stock": context.get("max_stock"),
            "reorder_point": context.get("reorder_point"),
        }
    else:
        return {}
    return {key: value for key, value in wanted.items() if value is not None}


def entity_summary(window: WindowedFeatures) -> dict[str, Any] | None:
    """Latest reading of an entity as a JSON-friendly dict (``None`` if it has no signal).

    Vehicles and SKUs add a ``details`` dict (position / route, or stock band and last
    transaction); warehouse sensors have no ``details`` key.
    """
    signal = _signal_of(window)
    if signal is None:
        return None
    context = dict(window.context)
    unit = context.get("unit") or DEFAULT_UNITS.get(window.source_type)
    extras = _details(window, context)
    return {
        **({"details": extras} if extras else {}),
        "entity_id": window.entity_id,
        "source_type": window.source_type,
        "signal": signal,
        "value": window.signals[signal].last,
        "timestamp": _iso(window.window_end),
        "unit": unit,
        "zone": context.get("zone"),
        "warehouse_id": context.get("warehouse_id"),
        "sensor_type": context.get("sensor_type") or context.get("reading_type"),
        "expected_range": expected_range(window.source_type, signal, context),
    }


def _point_extras(window: WindowedFeatures) -> dict[str, Any]:
    """Extra fields of one plotted point: a vehicle's position, an SKU's transaction."""
    if window.source_type == "vehicle":
        lat, lon = window.context.get("latitude"), window.context.get("longitude")
        return {"lat": lat, "lon": lon} if lat is not None and lon is not None else {}
    if window.source_type == "inventory":
        change = window.signals.get("change_amount")
        extras: dict[str, Any] = {}
        if change is not None:
            extras["d"] = change.last
        if window.context.get("transaction_type") is not None:
            extras["tx"] = window.context["transaction_type"]
        return extras
    return {}


def all_entities(cache: FeatureCache) -> list[dict[str, Any]]:
    """Latest reading of every live entity, sorted by entity id."""
    rows = []
    for entity_id in cache.entity_ids():
        latest = cache.peek(entity_id, 1)
        summary = entity_summary(latest[-1]) if latest else None
        if summary is not None:
            rows.append(summary)
    return sorted(rows, key=lambda row: row["entity_id"])


def entity_timeseries(
    cache: FeatureCache, entity_id: str, lookback_seconds: float = 300.0
) -> dict[str, Any] | None:
    """Recent readings of one entity, oldest first; ``None`` if the entity is unknown or expired.

    The lookback is measured back from the entity's newest event time, not from the wall clock.
    """
    windows = cache.peek(entity_id, _MAX_POINTS)
    if not windows:
        return None
    summary = entity_summary(windows[-1])
    if summary is None:
        return None
    signal = summary["signal"]
    newest = windows[-1].window_end.timestamp()
    points = [
        {"t": _iso(w.window_end), "v": w.signals[signal].last, **_point_extras(w)}
        for w in windows
        if signal in w.signals and newest - w.window_end.timestamp() <= lookback_seconds
    ]
    return {**summary, "lookback_seconds": lookback_seconds, "points": points}
