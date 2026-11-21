import threading
from datetime import UTC, datetime

import pytest
from prometheus_client import CollectorRegistry

from src.kafka.feature_cache import FeatureCache, FeatureStore
from src.kafka.stream_processor import SignalStats, WindowedFeatures
from src.utils.metrics import ConsumerMetrics


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(entity="s1", window=60, n=1, last=1.0):
    stats = SignalStats(last, 0.0, last, last, 0.0, last, last)
    end = datetime(2026, 11, 20, 8, 0, n, tzinfo=UTC)
    return WindowedFeatures(
        entity_id=entity,
        source_type="warehouse",
        window_seconds=window,
        window_start=end,
        window_end=end,
        count=n,
        coverage_seconds=0.0,
        signals={"value": stats},
    )


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def cache(clock):
    return FeatureCache(default_ttl_s=100, max_history=5, clock=clock)


def test_implements_the_feature_store_protocol(cache):
    assert isinstance(cache, FeatureStore)


def test_store_and_get_latest(cache):
    assert cache.get_features("s1") is None
    cache.store_features("s1", make(n=1), 60)
    cache.store_features("s1", make(n=2), 60)
    assert cache.get_features("s1").count == 2


def test_history_is_oldest_first_and_limited_to_n(cache):
    for n in range(1, 5):
        cache.store_features("s1", make(n=n))
    assert [f.count for f in cache.get_history("s1", 3)] == [2, 3, 4]
    assert [f.count for f in cache.get_history("s1", 10)] == [1, 2, 3, 4]
    assert cache.get_history("s1", 0) == []
    assert cache.get_history("nobody", 3) == []


def test_history_is_bounded_per_entity_and_window(cache):
    for n in range(1, 9):
        cache.store_features("s1", make(n=n))
    assert [f.count for f in cache.get_history("s1", 99)] == [4, 5, 6, 7, 8]


def test_windows_are_stored_separately(cache):
    cache.store_features("s1", make(window=60, n=1))
    cache.store_features("s1", make(window=300, n=2))
    assert cache.get_features("s1").window_seconds == 60
    assert cache.get_features("s1", 300).count == 2
    assert cache.get_features("s1", 900) is None
    assert {w: f.count for w, f in cache.get_all_windows("s1").items()} == {60: 1, 300: 2}


def test_ttl_expires_entity_and_store_refreshes_it(cache, clock):
    cache.store_features("s1", make(n=1), ttl_seconds=10)
    clock.now += 9
    cache.store_features("s1", make(n=2), ttl_seconds=10)  # refresh
    clock.now += 9
    assert cache.get_features("s1").count == 2
    clock.now += 2
    assert cache.get_features("s1") is None and not cache.exists("s1")
    cache.store_features("s1", make(n=3), 10)
    assert [f.count for f in cache.get_history("s1", 9)] == [3]  # history restarted


def test_default_ttl_and_validation(cache, clock):
    cache.store_features("s1", make())
    clock.now += 99
    assert cache.exists("s1")
    clock.now += 2
    assert not cache.exists("s1")
    with pytest.raises(ValueError):
        cache.store_features("s1", make(), ttl_seconds=0)
    with pytest.raises(ValueError):
        cache.store_features("other", make(entity="s1"))
    for kwargs in ({"default_ttl_s": 0}, {"max_history": 0}, {"max_entities": 0}):
        with pytest.raises(ValueError):
            FeatureCache(**kwargs)


def test_purge_delete_clear_and_listing(cache, clock):
    cache.store_features("a", make("a"), 5)
    cache.store_features("b", make("b"), 50)
    clock.now += 10
    assert cache.entity_ids() == ["b"]
    assert cache.purge_expired() == 1 and len(cache) == 1
    assert cache.delete("b") and not cache.delete("b")
    cache.store_features("c", make("c"))
    cache.clear()
    assert len(cache) == 0 and cache.hit_ratio == 0.0


def test_max_entities_drops_oldest_stored(clock):
    small = FeatureCache(max_entities=2, clock=clock)
    for name in "abc":
        small.store_features(name, make(name))
    assert not small.exists("a") and small.exists("b") and small.exists("c")


def test_hit_ratio_counts_reads_and_feeds_prometheus(clock):
    registry = CollectorRegistry()
    metrics = ConsumerMetrics(registry)
    cache = FeatureCache(clock=clock, metrics=metrics)
    cache.store_features("s1", make())
    cache.get_features("s1")
    cache.get_features("s1")
    cache.get_features("nobody")
    cache.get_history("s1", 3)
    assert cache.hit_ratio == pytest.approx(0.75)
    assert registry.get_sample_value("scad_cache_hits_total") == 3
    assert registry.get_sample_value("scad_cache_misses_total") == 1
    assert registry.get_sample_value("scad_cache_hit_ratio") == pytest.approx(0.75)
    assert cache.stats()["entities"] == 1.0


def test_concurrent_writers_and_readers_stay_consistent(clock):
    cache = FeatureCache(max_history=1000, clock=clock)
    errors = []

    def writer(name):
        try:
            for n in range(200):
                cache.store_features(name, make(name, n=n % 59 + 1))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(400):
                for name in ("w0", "w1", "w2"):
                    cache.get_history(name, 20)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(len(cache.get_history(f"w{i}", 1000)) == 200 for i in range(3))
