"""Unit tests of the API building blocks: stores and the token bucket (Day 14)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest

from src.api.anomaly_store import AnomalyStore
from src.api.feedback_store import FeedbackStore
from src.api.middleware.rate_limiter import RateLimitMiddleware, TokenBucketLimiter
from src.models.ensemble_detector import AnomalyEvent, Severity

T0 = datetime(2026, 11, 24, 8, 0, tzinfo=UTC)


def event(i: int, entity: str = "s1", severity: Severity = Severity.WARNING, kind: str = "point"):
    return AnomalyEvent(
        event_id=f"e{i}",
        entity_id=entity,
        entity_type="warehouse_sensor",
        timestamp=T0 + timedelta(minutes=i),
        anomaly_type=kind,
        severity=severity,
        raw_value=float(i),
        expected_range=(2.0, 4.0),
        top_contributing_features=[],
        recommended_action="check",
    )


# --------------------------------------------------------------------------- #
# AnomalyStore
# --------------------------------------------------------------------------- #
def test_store_evicts_oldest_when_full():
    store = AnomalyStore(capacity=3)
    for i in range(5):
        store.add(event(i))
    assert len(store) == 3
    assert store.get("e0") is None and store.get("e1") is None and store.get("e4") is not None
    assert store.stats()["evicted"] == 2


def test_store_rejects_bad_capacity():
    with pytest.raises(ValueError):
        AnomalyStore(capacity=0)


def test_store_replaces_event_with_same_id():
    store = AnomalyStore()
    store.add(event(1, severity=Severity.WARNING))
    store.add(event(1, severity=Severity.CRITICAL))
    assert len(store) == 1 and store.get("e1").severity == Severity.CRITICAL


def test_store_filters_combine_and_sort_newest_first():
    store = AnomalyStore()
    store.add(event(1, "a", Severity.INFO, "point"))
    store.add(event(2, "a", Severity.CRITICAL, "collective"))
    store.add(event(3, "b", Severity.CRITICAL, "point"))
    assert [e.event_id for e in store.filter()] == ["e3", "e2", "e1"]
    assert [e.event_id for e in store.filter(entity_id="a")] == ["e2", "e1"]
    assert [e.event_id for e in store.filter(severity=Severity.CRITICAL, anomaly_type="point")] == [
        "e3"
    ]
    window = store.filter(start_time=T0 + timedelta(minutes=2), end_time=T0 + timedelta(minutes=2))
    assert [e.event_id for e in window] == ["e2"]
    # naive datetimes are read as UTC
    assert len(store.filter(start_time=datetime(2026, 11, 24, 8, 2))) == 2


def test_store_set_flags_and_unknown_id():
    store = AnomalyStore()
    store.add(event(1))
    assert store.set_flags("e1", false_positive=True, resolved=True).false_positive_marked is True
    assert store.get("e1").is_resolved is True
    assert store.set_flags("missing", false_positive=True) is None


def test_store_stats_counts():
    store = AnomalyStore()
    store.add(event(1, "a", Severity.INFO))
    store.add(event(2, "a", Severity.CRITICAL))
    stats = store.stats()
    assert stats["total"] == 2 and stats["by_severity"] == {"INFO": 1, "CRITICAL": 1}
    assert stats["by_entity"] == {"a": 2} and stats["by_type"] == {"point": 2}


def test_store_snapshot_round_trip_and_corruption(tmp_path):
    store = AnomalyStore()
    store.add(event(1))
    store.add(event(2, severity=Severity.EMERGENCY))
    path = tmp_path / "nested" / "snap.json"
    assert store.snapshot(path) == 2
    assert not list(path.parent.glob("*.tmp"))  # atomic write leaves no temp files

    restored = AnomalyStore()
    assert restored.load_snapshot(path) == 2
    assert restored.get("e2").severity == Severity.EMERGENCY

    assert AnomalyStore().load_snapshot(tmp_path / "missing.json") == 0
    path.write_text("{not json")
    assert AnomalyStore().load_snapshot(path) == 0
    path.write_text(json.dumps({"events": [{"event_id": "x"}, event(3).to_dict()]}))
    partial = AnomalyStore()
    assert partial.load_snapshot(path) == 1  # the broken record is skipped


def test_store_is_thread_safe():
    store = AnomalyStore(capacity=500)

    def worker(base: int) -> None:
        for i in range(200):
            store.add(event(base * 1000 + i))
            store.filter()
            store.stats()

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 500


# --------------------------------------------------------------------------- #
# FeedbackStore
# --------------------------------------------------------------------------- #
def add(store, event_id, analyst, kind, entity="s1", flagged=("statistical",), severity="WARNING"):
    return store.add(
        anomaly_event_id=event_id,
        entity_id=entity,
        feedback_type=kind,
        analyst_id=analyst,
        severity=severity,
        flagged_by=flagged,
    )


def test_feedback_replacement_semantics():
    store = FeedbackStore()
    _, replaced = add(store, "e1", "a1", "false_positive")
    assert replaced is False
    _, replaced = add(store, "e1", "a1", "true_positive")
    assert replaced is True and len(store) == 1
    add(store, "e1", "a2", "true_positive")
    assert len(store) == 2 and len(store.for_event("e1")) == 2


def test_feedback_stats_groups():
    store = FeedbackStore()
    add(store, "e1", "a", "false_positive", "s1", ("statistical", "isolation_forest"))
    add(store, "e2", "a", "true_positive", "s1", ("statistical",))
    add(store, "e3", "a", "false_positive", "s2", ("isolation_forest",), "CRITICAL")
    stats = store.stats()
    assert stats["overall"]["total"] == 3 and stats["overall"]["false_positive"] == 2
    assert stats["per_model"]["statistical"]["false_positive_rate"] == 0.5
    assert stats["per_model"]["isolation_forest"]["false_positive_rate"] == 1.0
    assert stats["per_entity"]["s1"]["total"] == 2
    assert set(stats["per_severity"]) == {"WARNING", "CRITICAL"}


def test_feedback_empty_stats():
    stats = FeedbackStore().stats()
    assert stats["overall"]["total"] == 0 and stats["per_model"] == {}


def test_feedback_persistence_round_trip(tmp_path):
    store = FeedbackStore()
    add(store, "e1", "a", "false_positive")
    add(store, "e2", "b", "true_positive")
    path = tmp_path / "fb.json"
    assert store.save(path) == 2
    restored = FeedbackStore()
    assert restored.load(path) == 2
    assert restored.stats() == store.stats()
    assert FeedbackStore().load(tmp_path / "none.json") == 0
    path.write_text("garbage")
    assert FeedbackStore().load(path) == 0


# --------------------------------------------------------------------------- #
# Token bucket
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_bucket_allows_burst_then_blocks_then_refills():
    clock = Clock()
    limiter = TokenBucketLimiter(clock)
    results = [limiter.acquire("k", 60) for _ in range(61)]
    assert all(r[0] for r in results[:60]) and results[60][0] is False
    allowed, retry_after, remaining = results[60]
    assert remaining == 0 and retry_after == pytest.approx(1.0)  # 60/min = 1 token per second
    clock.now += 1.0
    assert limiter.acquire("k", 60)[0] is True
    assert limiter.acquire("k", 60)[0] is False


def test_bucket_never_exceeds_capacity_after_idle():
    clock = Clock()
    limiter = TokenBucketLimiter(clock)
    limiter.acquire("k", 5)
    clock.now += 3600
    assert sum(limiter.acquire("k", 5)[0] for _ in range(10)) == 5


def test_buckets_are_independent_per_key():
    limiter = TokenBucketLimiter(Clock())
    assert limiter.acquire("a", 1)[0] and not limiter.acquire("a", 1)[0]
    assert limiter.acquire("b", 1)[0]


def test_remaining_counts_down():
    limiter = TokenBucketLimiter(Clock())
    assert [limiter.acquire("k", 3)[2] for _ in range(3)] == [2, 1, 0]


def test_idle_buckets_are_pruned():
    clock = Clock()
    limiter = TokenBucketLimiter(clock)
    for i in range(10):
        limiter.acquire(f"k{i}", 5)
    clock.now += 1000
    for _ in range(1024):
        limiter.acquire("hot", 5)
    assert len(limiter) == 1


def test_limiter_thread_safety_never_over_admits():
    limiter = TokenBucketLimiter(Clock())  # frozen clock: no refill
    admitted = []

    def worker() -> None:
        admitted.extend(limiter.acquire("k", 50)[0] for _ in range(40))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(admitted) == 50


def test_middleware_rejects_non_positive_limits():
    with pytest.raises(ValueError):
        RateLimitMiddleware(lambda *a: None, {"/x": 0})
