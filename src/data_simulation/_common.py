"""Small helpers shared by the Day 3 simulators.

UTC time handling and argument validation, kept in one place so the GPS,
inventory, order and orchestrator modules behave identically. Nothing here is
part of the public API of the package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime


def to_utc(moment: datetime) -> datetime:
    """Return ``moment`` as an aware UTC datetime (naive values are assumed to be UTC)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def isoformat(moment: datetime) -> str:
    """Format an aware UTC datetime as ISO 8601 with a ``Z`` suffix."""
    return moment.isoformat().replace("+00:00", "Z")


def default_start() -> datetime:
    """Return the current UTC time truncated to whole seconds."""
    return datetime.now(UTC).replace(microsecond=0)


def validate_rate(name: str, value: float) -> float:
    """Check that ``value`` is a probability in ``[0, 1]``."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")
    return float(value)


def validate_positive(name: str, value: float) -> float:
    """Check that ``value`` is strictly positive."""
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def validate_non_negative(name: str, value: float) -> float:
    """Check that ``value`` is zero or positive."""
    if not value >= 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return float(value)


def validate_length_range(name: str, value: tuple[int, int]) -> tuple[int, int]:
    """Check an inclusive ``(low, high)`` episode-length range with ``1 <= low <= high``."""
    low, high = value
    if low < 1 or high < low:
        raise ValueError(f"{name} must satisfy 1 <= low <= high, got {value}")
    return int(low), int(high)


def validate_duration(name: str, value: int | None) -> int | None:
    """Check an optional episode length (``None`` means "pick one at random")."""
    if value is not None and value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def validate_mix(name: str, mix: Sequence[float], size: int) -> tuple[float, ...]:
    """Check that ``mix`` has ``size`` non-negative shares that sum to 1."""
    shares = tuple(float(share) for share in mix)
    if len(shares) != size:
        raise ValueError(f"{name} needs {size} shares, got {len(shares)}")
    if any(share < 0 for share in shares) or not math.isclose(sum(shares), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be non-negative and sum to 1.0, got {shares}")
    return shares


def mean_length(bounds: tuple[int, int]) -> float:
    """Return the mean of an inclusive integer range."""
    return (bounds[0] + bounds[1]) / 2
