"""In-memory store of analyst feedback (Day 14).

One :class:`FeedbackRecord` per ``(anomaly_event_id, analyst_id)``: a second verdict from the same
analyst on the same alert replaces the first (analysts change their minds), while different
analysts keep separate records. The whole store is JSON-serialisable (:meth:`save` /
:meth:`load`, atomic writes). Day 16 builds the feedback processor and calibrators on top.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class FeedbackRecord:
    """One analyst verdict on one alert."""

    feedback_id: str
    anomaly_event_id: str
    entity_id: str
    feedback_type: str
    analyst_id: str
    severity: str
    flagged_by: tuple[str, ...]
    notes: str | None
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "feedback_id": self.feedback_id,
            "anomaly_event_id": self.anomaly_event_id,
            "entity_id": self.entity_id,
            "feedback_type": self.feedback_type,
            "analyst_id": self.analyst_id,
            "severity": self.severity,
            "flagged_by": list(self.flagged_by),
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        """Rebuild a record from :meth:`to_dict` output."""
        return cls(
            feedback_id=str(data["feedback_id"]),
            anomaly_event_id=str(data["anomaly_event_id"]),
            entity_id=str(data["entity_id"]),
            feedback_type=str(data["feedback_type"]),
            analyst_id=str(data["analyst_id"]),
            severity=str(data.get("severity", "")),
            flagged_by=tuple(data.get("flagged_by", ())),
            notes=data.get("notes"),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
        )


def _group(records: list[FeedbackRecord], key: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        for name in key(record):
            g = groups.setdefault(name, {"true_positive": 0, "false_positive": 0})
            g[record.feedback_type] += 1
    for g in groups.values():
        g["total"] = g["true_positive"] + g["false_positive"]
        g["false_positive_rate"] = g["false_positive"] / g["total"] if g["total"] else 0.0
    return groups


class FeedbackStore:
    """Thread-safe feedback records keyed by ``(anomaly_event_id, analyst_id)``."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._records: dict[tuple[str, str], FeedbackRecord] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def add(
        self,
        *,
        anomaly_event_id: str,
        entity_id: str,
        feedback_type: str,
        analyst_id: str,
        severity: str = "",
        flagged_by: tuple[str, ...] = (),
        notes: str | None = None,
        timestamp: datetime | None = None,
    ) -> tuple[FeedbackRecord, bool]:
        """Store a verdict; return ``(record, replaced_previous)``."""
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            anomaly_event_id=anomaly_event_id,
            entity_id=entity_id,
            feedback_type=feedback_type,
            analyst_id=analyst_id,
            severity=severity,
            flagged_by=flagged_by,
            notes=notes,
            timestamp=timestamp or datetime.now(UTC),
        )
        key = (anomaly_event_id, analyst_id)
        with self._lock:
            replaced = key in self._records
            self._records[key] = record
        return record, replaced

    def for_event(self, anomaly_event_id: str) -> list[FeedbackRecord]:
        """All verdicts on one alert (oldest first)."""
        with self._lock:
            rows = [r for (eid, _), r in self._records.items() if eid == anomaly_event_id]
        return sorted(rows, key=lambda r: r.timestamp)

    def all(self) -> list[FeedbackRecord]:
        """Every record, oldest first."""
        with self._lock:
            rows = list(self._records.values())
        return sorted(rows, key=lambda r: r.timestamp)

    def stats(self) -> dict[str, Any]:
        """Tallies overall, per model that flagged the alert, per entity and per severity."""
        records = self.all()
        overall = _group(records, lambda r: ["all"]).get(
            "all",
            {"true_positive": 0, "false_positive": 0, "total": 0, "false_positive_rate": 0.0},
        )
        return {
            "overall": overall,
            "per_model": _group(records, lambda r: r.flagged_by),
            "per_entity": _group(records, lambda r: [r.entity_id]),
            "per_severity": _group(records, lambda r: [r.severity] if r.severity else []),
        }

    def save(self, path: str | Path) -> int:
        """Write all records as JSON (atomic); return the count."""
        target = Path(path)
        rows = [r.to_dict() for r in self.all()]
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "feedback": rows}, handle)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return len(rows)

    def load(self, path: str | Path) -> int:
        """Load records written by :meth:`save` (missing file: nothing); return the count."""
        source = Path(path)
        if not source.is_file():
            return 0
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _log.warning("feedback_store_unreadable", path=str(source), error=repr(error))
            return 0
        loaded = 0
        for row in data.get("feedback", []):
            try:
                record = FeedbackRecord.from_dict(row)
            except (KeyError, ValueError, TypeError) as error:
                _log.warning("feedback_record_skipped", error=repr(error))
                continue
            with self._lock:
                self._records[(record.anomaly_event_id, record.analyst_id)] = record
            loaded += 1
        return loaded
