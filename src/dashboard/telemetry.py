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


def entity_summary(window: WindowedFeatures) -> dict[str, Any] | None:
    """Latest reading of an entity as a JSON-friendly dict (``None`` if it has no signal)."""
    signal = _signal_of(window)
    if signal is None:
        return None
    context = dict(window.context)
    unit = context.get("unit") or DEFAULT_UNITS.get(window.source_type)
    return {
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
        {"t": _iso(w.window_end), "v": w.signals[signal].last}
        for w in windows
        if signal in w.signals and newest - w.window_end.timestamp() <= lookback_seconds
    ]
    return {**summary, "lookback_seconds": lookback_seconds, "points": points}
