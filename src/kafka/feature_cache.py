"""In-memory feature store with a Redis-shaped interface (Day 5).

:class:`FeatureStore` is the small interface the rest of the system codes against
(:class:`~src.kafka.pipeline.StreamingFeaturePipeline` writes to it, the Day 6 sequence
builder and the Day 10/11 detectors read from it). :class:`FeatureCache` implements it with
plain dicts; a Redis implementation can replace it later without touching any caller, because
every value is a JSON-serialisable :class:`~src.kafka.stream_processor.WindowedFeatures`
(``to_dict`` / ``from_dict``) and the operations map onto Redis commands:

====================  ====================================================
``store_features``    ``LPUSH`` + ``LTRIM`` on a per-entity-and-window list, ``EXPIRE``
``get_features``      ``LINDEX 0``
``get_history``       ``LRANGE``
TTL                   ``EXPIRE`` refreshed on every store
====================  ====================================================

Layout: an entity has one history per window length (a bounded deque, oldest first) and one
expiry time for the whole entity. Every store refreshes that expiry, so an entity that stops
sending is dropped ``ttl_seconds`` after its last update. Expired data is purged lazily on
access and in bulk by :meth:`FeatureCache.purge_expired`.

``max_entities`` bounds memory: when exceeded, the entity that was stored longest ago is
dropped.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.kafka.stream_processor import WindowedFeatures
    from src.utils.metrics import ConsumerMetrics

DEFAULT_TTL_S = 3_600
DEFAULT_MAX_HISTORY = 256


@runtime_checkable
class FeatureStore(Protocol):
    """What the feature pipeline and the models need from a feature store."""

    def store_features(
        self, entity_id: str, features: WindowedFeatures, ttl_seconds: float | None = None
    ) -> None:
        """Append ``features`` to the entity's history and refresh its expiry."""

    def get_features(
        self, entity_id: str, window_seconds: int | None = None
    ) -> WindowedFeatures | None:
        """Return the latest features of one window length, or ``None``."""

    def get_history(
        self, entity_id: str, n_windows: int, window_seconds: int | None = None
    ) -> list[WindowedFeatures]:
        """Return up to ``n_windows`` most recent features, oldest first."""


@dataclass(slots=True)
class _Entry:
    expires_at: float
    histories: dict[int, deque[WindowedFeatures]] = field(default_factory=dict)


class FeatureCache:
    """Thread-safe in-memory :class:`FeatureStore`."""

    def __init__(
        self,
        *,
        default_ttl_s: float = DEFAULT_TTL_S,
        max_history: int = DEFAULT_MAX_HISTORY,
        primary_window_s: int = 60,
        max_entities: int = 50_000,
        clock: Callable[[], float] = time.monotonic,
        metrics: ConsumerMetrics | None = None,
    ) -> None:
        """Create the cache.

        Args:
            default_ttl_s: Expiry used when ``store_features`` gets no ``ttl_seconds``.
            max_history: Features kept per entity and window length (oldest dropped first).
                Must cover the longest LSTM lookback (60 by default).
            primary_window_s: Window length returned when a read does not name one.
            max_entities: Entities kept at once.
            clock: Monotonic clock in seconds (injectable for tests).
            metrics: Receives hit / miss counts; optional.

        Raises:
            ValueError: An argument is out of range.
        """
        if default_ttl_s <= 0 or max_history < 1 or max_entities < 1:
            raise ValueError("default_ttl_s, max_history and max_entities must be positive")
        self._default_ttl = float(default_ttl_s)
        self._max_history = max_history
        self.primary_window_s = primary_window_s
        self._max_entities = max_entities
        self._clock = clock
        self._metrics = metrics
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def store_features(
        self, entity_id: str, features: WindowedFeatures, ttl_seconds: float | None = None
    ) -> None:
        """Append ``features`` and refresh the entity's expiry.

        Args:
            entity_id: Must equal ``features.entity_id``.
            features: One window's features.
            ttl_seconds: Seconds until the entity expires; the cache default if ``None``.

        Raises:
            ValueError: ``entity_id`` does not match, or ``ttl_seconds`` is not positive.
        """
        if features.entity_id != entity_id:
            raise ValueError(
                f"entity_id {entity_id!r} != features.entity_id {features.entity_id!r}"
            )
        ttl = self._default_ttl if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        with self._lock:
            now = self._clock()
            entry = self._entries.get(entity_id)
            if entry is None or entry.expires_at <= now:
                entry = _Entry(expires_at=now + ttl)
                self._entries[entity_id] = entry
            history = entry.histories.get(features.window_seconds)
            if history is None:
                history = deque(maxlen=self._max_history)
                entry.histories[features.window_seconds] = history
            history.append(features)
            entry.expires_at = now + ttl
            self._entries.move_to_end(entity_id)
            while len(self._entries) > self._max_entities:
                self._entries.popitem(last=False)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_features(
        self, entity_id: str, window_seconds: int | None = None
    ) -> WindowedFeatures | None:
        """Return the latest features of ``window_seconds`` (default: the primary window)."""
        history = self._history(entity_id, window_seconds)
        found = history[-1] if history else None
        self._record(found is not None)
        return found

    def get_history(
        self, entity_id: str, n_windows: int, window_seconds: int | None = None
    ) -> list[WindowedFeatures]:
        """Return up to ``n_windows`` most recent features, oldest first.

        This is the input of the LSTM sequence builder; it may return fewer than requested
        (cold start) or an empty list.
        """
        if n_windows <= 0:
            return []
        history = self._history(entity_id, window_seconds)
        items = list(history)[-n_windows:] if history else []
        self._record(bool(items))
        return items

    def get_all_windows(self, entity_id: str) -> dict[int, WindowedFeatures]:
        """Return the latest features of every window length of the entity."""
        with self._lock:
            entry = self._live_entry(entity_id)
            latest = {w: h[-1] for w, h in sorted(entry.histories.items()) if h} if entry else {}
        self._record(bool(latest))
        return latest

    def _history(
        self, entity_id: str, window_seconds: int | None
    ) -> deque[WindowedFeatures] | None:
        with self._lock:
            entry = self._live_entry(entity_id)
            if entry is None:
                return None
            history = entry.histories.get(window_seconds or self.primary_window_s)
            return deque(history) if history else None

    def _live_entry(self, entity_id: str) -> _Entry | None:
        entry = self._entries.get(entity_id)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            del self._entries[entity_id]
            return None
        return entry

    def _record(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._hits += 1
            else:
                self._misses += 1
        if self._metrics is not None:
            (self._metrics.cache_hits if hit else self._metrics.cache_misses).inc()

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #
    def exists(self, entity_id: str) -> bool:
        """Return whether the entity has unexpired data (not counted as a hit or miss)."""
        with self._lock:
            return self._live_entry(entity_id) is not None

    def delete(self, entity_id: str) -> bool:
        """Remove an entity; return whether it existed."""
        with self._lock:
            return self._entries.pop(entity_id, None) is not None

    def entity_ids(self) -> list[str]:
        """Ids of all unexpired entities."""
        with self._lock:
            now = self._clock()
            return [eid for eid, e in self._entries.items() if e.expires_at > now]

    def purge_expired(self) -> int:
        """Remove every expired entity; return how many were removed."""
        with self._lock:
            now = self._clock()
            expired = [eid for eid, e in self._entries.items() if e.expires_at <= now]
            for eid in expired:
                del self._entries[eid]
            return len(expired)

    def clear(self) -> None:
        """Remove everything and reset hit / miss counts."""
        with self._lock:
            self._entries.clear()
            self._hits = self._misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_ratio(self) -> float:
        """Share of reads that found data since creation (``0.0`` before any read)."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict[str, float]:
        """Entities, hits, misses and hit ratio."""
        return {
            "entities": float(len(self._entries)),
            "hits": float(self._hits),
            "misses": float(self._misses),
            "hit_ratio": self.hit_ratio,
        }
