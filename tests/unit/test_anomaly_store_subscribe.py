"""Tests of the AnomalyStore listener hook that feeds the SSE stream (Day 18)."""

from __future__ import annotations

from datetime import UTC, datetime

from src.api.anomaly_store import AnomalyStore
from src.models.ensemble_detector import AnomalyEvent, Severity

T0 = datetime(2026, 11, 26, 8, 0, tzinfo=UTC)


def event(i: int) -> AnomalyEvent:
    return AnomalyEvent(
        event_id=f"e{i}",
        entity_id="s1",
        entity_type="warehouse_sensor",
        timestamp=T0,
        anomaly_type="point",
        severity=Severity.WARNING,
        raw_value=1.0,
        expected_range=(2.0, 4.0),
        top_contributing_features=[],
        recommended_action="check",
    )


def test_listener_is_called_for_every_added_alert():
    store, seen = AnomalyStore(), []
    store.subscribe(seen.append)
    store.add(event(1))
    store.add(event(2))
    assert [e.event_id for e in seen] == ["e1", "e2"]


def test_unsubscribe_stops_notifications_and_is_repeatable():
    store, seen = AnomalyStore(), []
    off = store.subscribe(seen.append)
    off()
    off()
    store.add(event(1))
    assert seen == []


def test_subscribing_the_same_callable_twice_notifies_once():
    store, seen = AnomalyStore(), []
    store.subscribe(seen.append)
    store.subscribe(seen.append)
    store.add(event(1))
    assert len(seen) == 1


def test_a_failing_listener_never_breaks_add_or_other_listeners():
    store, seen = AnomalyStore(), []

    def broken(_: AnomalyEvent) -> None:
        raise RuntimeError("subscriber bug")

    store.subscribe(broken)
    store.subscribe(seen.append)
    store.add(event(1))
    assert len(store) == 1 and len(seen) == 1


def test_notify_false_stores_without_telling_listeners():
    store, seen = AnomalyStore(), []
    store.subscribe(seen.append)
    store.add(event(1), notify=False)
    assert len(store) == 1 and seen == []


def test_listener_may_use_the_store_without_deadlock():
    store = AnomalyStore()
    counts = []
    store.subscribe(lambda e: counts.append(len(store)))
    store.add(event(1))
    assert counts == [1]
