"""In-memory store of recent alerts (Day 14; the streaming engine extends it on Day 15).

:class:`AnomalyStore` is a thread-safe ring buffer of
:class:`~src.models.ensemble_detector.AnomalyEvent` records keyed by ``event_id``. When the
capacity is exceeded the oldest alert is evicted. It is the single source of truth for
``GET /api/v1/anomalies`` and, later, the dashboard.

Persistence: :meth:`snapshot` writes all alerts to a JSON file atomically (temp file + rename)
and :meth:`load_snapshot` restores them, so a restart does not empty the alert list. Day 15 adds
:meth:`start_periodic_snapshots`, a background thread that does this every few minutes (and once
more on stop), so a crash loses at most one interval of alerts.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models.ensemble_detector import AnomalyEvent, Severity
from src.utils.logger import get_logger

_log = get_logger(__name__)

DEFAULT_CAPACITY = 10_000


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


class AnomalyStore:
    """Thread-safe ring buffer of the most recent alerts."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """Create an empty store.

        Args:
            capacity: Maximum alerts kept; the oldest is evicted first.

        Raises:
            ValueError: ``capacity`` is below 1.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._events: OrderedDict[str, AnomalyEvent] = OrderedDict()
        self._lock = threading.RLock()
        self._evicted = 0
        self._snapshot_thread: threading.Thread | None = None
        self._snapshot_stop = threading.Event()
        self._snapshot_path: Path | None = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def add(self, event: AnomalyEvent) -> None:
        """Insert (or replace, e.g. after an escalation) an alert; evict the oldest if full."""
        with self._lock:
            self._events[event.event_id] = event
            self._events.move_to_end(event.event_id)
            while len(self._events) > self.capacity:
                self._events.popitem(last=False)
                self._evicted += 1

    def get(self, event_id: str) -> AnomalyEvent | None:
        """Return one alert, or ``None`` if unknown or already evicted."""
        with self._lock:
            return self._events.get(event_id)

    def set_flags(
        self,
        event_id: str,
        *,
        false_positive: bool | None = None,
        resolved: bool | None = None,
    ) -> AnomalyEvent | None:
        """Update analyst-controlled flags in place; return the alert (``None`` if unknown)."""
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                return None
            if false_positive is not None:
                event.false_positive_marked = false_positive
            if resolved is not None:
                event.is_resolved = resolved
            return event

    def filter(
        self,
        *,
        entity_id: str | None = None,
        severity: Severity | None = None,
        anomaly_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AnomalyEvent]:
        """Return matching alerts, newest event time first.

        Args:
            entity_id: Exact entity id.
            severity: Exact severity level.
            anomaly_type: ``point`` / ``contextual`` / ``collective``.
            start_time: Keep alerts at or after this time (naive values are UTC).
            end_time: Keep alerts at or before this time.
        """
        start = _aware(start_time) if start_time else None
        end = _aware(end_time) if end_time else None
        with self._lock:
            snapshot = list(self._events.values())
        out = [
            e
            for e in snapshot
            if (entity_id is None or e.entity_id == entity_id)
            and (severity is None or e.severity == severity)
            and (anomaly_type is None or e.anomaly_type == anomaly_type)
            and (start is None or e.timestamp >= start)
            and (end is None or e.timestamp <= end)
        ]
        out.sort(key=lambda e: e.timestamp, reverse=True)
        return out

    def stats(self) -> dict[str, Any]:
        """Return counts by severity, anomaly type and entity, plus store occupancy."""
        with self._lock:
            events = list(self._events.values())
            evicted = self._evicted
        by_severity: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_entity: dict[str, int] = {}
        for e in events:
            by_severity[e.severity.name] = by_severity.get(e.severity.name, 0) + 1
            key = e.anomaly_type or "unknown"
            by_type[key] = by_type.get(key, 0) + 1
            by_entity[e.entity_id] = by_entity.get(e.entity_id, 0) + 1
        return {
            "total": len(events),
            "capacity": self.capacity,
            "evicted": evicted,
            "by_severity": by_severity,
            "by_type": by_type,
            "by_entity": by_entity,
        }

    def snapshot(self, path: str | Path) -> int:
        """Write every alert to ``path`` as JSON (atomic); return how many were written."""
        target = Path(path)
        with self._lock:
            payload = [e.to_dict() for e in self._events.values()]
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "events": payload}, handle)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return len(payload)

    def start_periodic_snapshots(self, path: str | Path, interval_s: float = 300.0) -> None:
        """Snapshot to ``path`` every ``interval_s`` seconds on a daemon thread.

        A failed write is logged and retried at the next tick; it never stops the thread.
        Calling this while a schedule is already running replaces it.

        Raises:
            ValueError: ``interval_s`` is not positive.
        """
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.stop_periodic_snapshots(final=False)
        target = Path(path)
        self._snapshot_path = target
        self._snapshot_stop = threading.Event()
        stop = self._snapshot_stop

        def _loop() -> None:
            while not stop.wait(interval_s):
                self._snapshot_quietly(target)

        thread = threading.Thread(target=_loop, name="scad-anomaly-snapshots", daemon=True)
        self._snapshot_thread = thread
        thread.start()

    def stop_periodic_snapshots(self, *, final: bool = True) -> None:
        """Stop the snapshot thread; with ``final`` write one last snapshot first."""
        thread, path = self._snapshot_thread, self._snapshot_path
        if thread is None:
            return
        self._snapshot_stop.set()
        thread.join(timeout=5.0)
        self._snapshot_thread = None
        if final and path is not None:
            self._snapshot_quietly(path)

    def _snapshot_quietly(self, path: Path) -> None:
        try:
            count = self.snapshot(path)
            _log.debug("anomaly_snapshot_written", path=str(path), alerts=count)
        except OSError as error:
            _log.error("anomaly_snapshot_failed", path=str(path), error=repr(error))

    def load_snapshot(self, path: str | Path) -> int:
        """Restore alerts from a :meth:`snapshot` file; return how many were loaded.

        A missing file loads nothing. Unreadable records are skipped with a warning.
        """
        source = Path(path)
        if not source.is_file():
            return 0
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _log.warning("anomaly_snapshot_unreadable", path=str(source), error=repr(error))
            return 0
        loaded = 0
        for record in data.get("events", []):
            try:
                self.add(AnomalyEvent.from_dict(record))
                loaded += 1
            except (KeyError, ValueError, TypeError) as error:
                _log.warning("anomaly_snapshot_record_skipped", error=repr(error))
        return loaded
