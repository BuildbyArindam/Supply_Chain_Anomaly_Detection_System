"""Small formatting helpers shared by the dashboard pages (Day 18)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import dash_bootstrap_components as dbc

from src.dashboard.components.theme import SEVERITY_BOOTSTRAP, SEVERITY_ORDER

_UNIT_SYMBOLS = {"celsius": "°C", "c": "°C", "percent": "%", "pct": "%"}
_NO_SPACE_UNITS = {"°C", "%", "°F"}
PLACEHOLDER = "—"


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 string, a ``datetime`` or epoch seconds into an aware UTC datetime.

    Returns ``None`` for anything unparseable (callers show a placeholder instead of failing).
    """
    if value is None or value == "":
        return None
    try:
        if isinstance(value, datetime):
            moment = value
        elif isinstance(value, int | float):
            moment = datetime.fromtimestamp(float(value), UTC)
        else:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, OverflowError, OSError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def format_timestamp(
    value: Any,
    *,
    tz: timezone | None = None,
    with_date: bool = False,
    with_seconds: bool = True,
) -> str:
    """Format a timestamp for display (``"11:41:02"``, or with the date); ``—`` if invalid.

    Args:
        value: ISO string, ``datetime`` or epoch seconds.
        tz: Display time zone (UTC if omitted).
        with_date: Prefix the date (``"2026-11-26 11:41:02"``).
        with_seconds: Include seconds.
    """
    moment = parse_timestamp(value)
    if moment is None:
        return PLACEHOLDER
    moment = moment.astimezone(tz or UTC)
    pattern = ("%Y-%m-%d " if with_date else "") + ("%H:%M:%S" if with_seconds else "%H:%M")
    return moment.strftime(pattern)


def format_age(value: Any, now: datetime | None = None) -> str:
    """Return a short relative age such as ``"12s ago"`` or ``"3 min ago"``."""
    moment = parse_timestamp(value)
    if moment is None:
        return PLACEHOLDER
    seconds = max(0.0, ((now or datetime.now(UTC)) - moment).total_seconds())
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.1f} d ago"


def format_value_with_unit(value: Any, unit: str | None = None, precision: int = 1) -> str:
    """Format a reading with its unit: ``3.2 °C``, ``54.0%``, ``88.5 km/h``; ``—`` if missing."""
    if value is None:
        return PLACEHOLDER
    try:
        number = float(value)
    except (TypeError, ValueError):
        return PLACEHOLDER
    if not math.isfinite(number):
        return PLACEHOLDER
    symbol = _UNIT_SYMBOLS.get((unit or "").lower(), unit or "")
    text = f"{number:,.{precision}f}"
    if not symbol:
        return text
    return f"{text}{symbol}" if symbol in _NO_SPACE_UNITS else f"{text} {symbol}"


def severity_badge(level: str | None) -> dbc.Badge:
    """Return a badge for a severity name (the text carries the meaning, colour reinforces it)."""
    name = str(level or "").upper()
    if name not in SEVERITY_ORDER:
        return dbc.Badge(name or PLACEHOLDER, color="secondary", className="severity-badge")
    return dbc.Badge(
        name,
        color=SEVERITY_BOOTSTRAP[name],
        className=f"severity-badge severity-{name.lower()}",
        pill=True,
    )


def hour_floor(moment: datetime) -> datetime:
    """Truncate ``moment`` to the start of its hour."""
    return moment.replace(minute=0, second=0, microsecond=0)


def hour_range(end: datetime, hours: int) -> list[datetime]:
    """Return ``hours`` consecutive hour starts, the last one being the hour of ``end``."""
    last = hour_floor(end)
    return [last - timedelta(hours=h) for h in range(hours - 1, -1, -1)]
