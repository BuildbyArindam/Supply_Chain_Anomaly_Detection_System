import signal
import threading
import time

import pytest
from confluent_kafka import KafkaError, KafkaException
from prometheus_client import CollectorRegistry

from src.kafka.consumer import KafkaConsumerGroup, PermanentProcessingError
from src.utils.metrics import ConsumerMetrics
from tests.mock_kafka import MockBroker

TOPIC = "events"
GROUP = "g"


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def broker():
    return MockBroker({TOPIC: 4})


@pytest.fixture
def registry():
    return CollectorRegistry()


def publish(broker, n, keys=8):
    for i in range(n):
        broker.append(TOPIC, f"k{i % keys}".encode(), f"{i}".encode())


def metrics_for(registry):
    if not hasattr(registry, "_scad_metrics"):
        registry._scad_metrics = ConsumerMetrics(registry)
    return registry._scad_metrics


def make_group(broker, registry, handler, **kw):
    kw.setdefault("num_workers", 2)
    kw.setdefault("auto_offset_reset", "earliest")
    kw.setdefault("poll_timeout_s", 0.01)
    kw.setdefault("retry_backoff_s", 0.001)
    kw.setdefault("lag_interval_s", 0.0)
    return KafkaConsumerGroup(
        "localhost:9092",
        [TOPIC],
        handler,
        group_id=GROUP,
        consumer_factory=broker.consumer_factory,
        metrics=metrics_for(registry),
        **kw,
    )


def value(registry, name, **labels):
    return registry.get_sample_value(name, labels) or 0.0


def test_consumes_everything_in_parallel_and_commits_after_processing(broker, registry):
    seen, threads = [], set()

    def handler(m):
        seen.append(int(m.value))
        threads.add(threading.current_thread().name)

    group = make_group(broker, registry, handler)
    group.start()
    # Publish only once the partitions are split between the two workers, so both get work.
    assert wait_for(
        lambda: len(broker.owners) == 4 and len({id(o) for o in broker.owners.values()}) == 2
    )
    publish(broker, 200)
    assert wait_for(lambda: group.processed_count == 200)
    assert group.stop()
    assert sorted(seen) == list(range(200))
    assert len(threads) == 2  # both workers did work
    assert broker.lag(GROUP, TOPIC) == 0
    assert value(registry, "scad_messages_consumed_total", topic=TOPIC, status="ok") == 200


def test_per_key_order_is_preserved(broker, registry):
    publish(broker, 400, keys=5)
    per_key = {}

    def handler(m):
        per_key.setdefault(m.key, []).append(int(m.value))

    group = make_group(broker, registry, handler, num_workers=3)
    group.start()
    assert wait_for(lambda: group.processed_count == 400)
    group.stop()
    assert len(per_key) == 5
    assert all(seq == sorted(seq) for seq in per_key.values())


def test_offset_is_committed_only_after_the_handler_returns(broker, registry):
    publish(broker, 1)
    release, started = threading.Event(), threading.Event()

    def handler(_m):
        started.set()
        release.wait(5)

    group = make_group(broker, registry, handler, num_workers=1)
    group.start()
    assert started.wait(5)
    time.sleep(0.05)
    assert broker.commit_log == []  # handler still running: nothing committed
    release.set()
    assert wait_for(lambda: broker.total_committed(GROUP, TOPIC) == 1)
    group.stop()


def test_failed_commit_is_retried_with_the_next_batch(broker, registry):
    publish(broker, 3, keys=1)
    broker.fail_next_commits = 1
    group = make_group(broker, registry, lambda m: None, num_workers=1, max_poll_records=1)
    group.start()
    assert wait_for(lambda: broker.lag(GROUP, TOPIC) == 0)
    group.stop()
    assert value(registry, "scad_consumer_commit_failures_total") == 1


def test_transient_handler_error_is_retried_then_succeeds(broker, registry):
    publish(broker, 1)
    calls = []

    def handler(_m):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("flaky")

    group = make_group(broker, registry, handler, num_workers=1)
    group.start()
    assert wait_for(lambda: group.processed_count == 1)
    group.stop()
    assert len(calls) == 3 and group.dead_letter_count == 0


def test_persistent_error_is_dead_lettered_after_retries_and_partition_moves_on(broker, registry):
    publish(broker, 3, keys=1)
    dead, ok = [], []

    def handler(m):
        if m.value == b"1":
            raise RuntimeError("poison")
        ok.append(m.value)

    group = make_group(
        broker,
        registry,
        handler,
        num_workers=1,
        max_process_retries=2,
        dead_letter=lambda m, e, attempts: dead.append((m.value, str(e), attempts)),
    )
    group.start()
    assert wait_for(lambda: broker.lag(GROUP, TOPIC) == 0)
    group.stop()
    assert ok == [b"0", b"2"]
    assert dead == [(b"1", "poison", 3)]
    assert value(registry, "scad_messages_consumed_total", topic=TOPIC, status="dead_lettered") == 1


def test_permanent_error_skips_retries(broker, registry):
    publish(broker, 1)
    calls, dead = [], []

    def handler(_m):
        calls.append(1)
        raise PermanentProcessingError("malformed")

    group = make_group(
        broker, registry, handler, num_workers=1, dead_letter=lambda m, e, a: dead.append(a)
    )
    group.start()
    assert wait_for(lambda: broker.lag(GROUP, TOPIC) == 0)
    group.stop()
    assert len(calls) == 1 and dead == [1]


def test_failing_dead_letter_sink_does_not_stall_the_partition(broker, registry):
    publish(broker, 2, keys=1)

    def sink(*_):
        raise RuntimeError("dlq down")

    def handler(_m):
        raise PermanentProcessingError("bad")

    group = make_group(broker, registry, handler, num_workers=1, dead_letter=sink)
    group.start()
    assert wait_for(lambda: broker.lag(GROUP, TOPIC) == 0)
    group.stop()


def test_lag_is_exposed_per_partition_and_cleared_on_shutdown(broker, registry):
    publish(broker, 40)
    gate = threading.Event()
    group = make_group(broker, registry, lambda m: gate.wait(5), num_workers=1, max_poll_records=1)
    group.start()
    # With the handler blocked on the first message, the group sits behind the log end.
    assert wait_for(lambda: group.total_lag() >= 39)
    sample = [
        s
        for metric in registry.collect()
        if metric.name == "scad_kafka_consumer_lag"
        for s in metric.samples
    ]
    assert {s.labels["topic"] for s in sample} == {TOPIC}
    gate.set()
    assert wait_for(lambda: group.total_lag() == 0)
    group.stop()
    assert group.lag_snapshot() == {}
    assert value(registry, "scad_kafka_consumer_lag", topic=TOPIC, partition="0") == 0.0


def test_rebalance_commits_before_partitions_move(broker, registry):
    publish(broker, 100)
    group = make_group(broker, registry, lambda m: None, num_workers=1)
    group.start()
    assert wait_for(lambda: group.processed_count == 100)
    second = make_group(broker, registry, lambda m: None, num_workers=1)
    second.start()  # takes half of the partitions away from the first
    assert wait_for(lambda: value(registry, "scad_consumer_rebalances_total", event="revoke") >= 1)
    publish(broker, 100)
    assert wait_for(lambda: group.processed_count + second.processed_count == 200)
    group.stop()
    second.stop()
    assert broker.lag(GROUP, TOPIC) == 0


def test_worker_crash_is_restarted_and_redelivers_uncommitted_messages(broker, registry):
    publish(broker, 20)
    seen = []
    group = make_group(broker, registry, lambda m: seen.append(m.value), num_workers=1)
    original = broker.consumer_factory
    crashed = []

    def factory(config):
        consumer = original(config)
        if not crashed:
            consumer.crash_on_consume = KafkaException(KafkaError(KafkaError._FATAL, "boom"))
            crashed.append(consumer)
        return consumer

    group._factory = factory
    group.start()
    assert wait_for(lambda: group.processed_count >= 20, timeout=10)
    group.stop()
    assert value(registry, "scad_consumer_worker_restarts_total") == 1
    assert not group.failed and broker.lag(GROUP, TOPIC) == 0


def test_group_fails_after_too_many_restarts(broker, registry):
    def factory(config):
        raise RuntimeError("cannot connect")

    group = make_group(broker, registry, lambda m: None, num_workers=1, max_worker_restarts=1)
    group._factory = factory
    group.start()
    assert group.wait() is False
    assert group.failed
    assert value(registry, "scad_consumer_worker_restarts_total") == 2


def test_stop_interrupts_a_retry_backoff_without_committing_the_message(broker, registry):
    publish(broker, 1)
    attempted = threading.Event()

    def handler(_m):
        attempted.set()
        raise RuntimeError("always")

    group = make_group(
        broker, registry, handler, num_workers=1, max_process_retries=50, retry_backoff_s=5.0
    )
    group.start()
    assert attempted.wait(5)
    started = time.monotonic()
    assert group.stop(timeout=5)
    assert time.monotonic() - started < 2
    assert broker.total_committed(GROUP, TOPIC) == 0  # will be redelivered


def test_sigterm_and_sigint_trigger_graceful_shutdown(broker, registry):
    publish(broker, 10)
    group = make_group(broker, registry, lambda m: None, num_workers=1)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    assert group.install_signal_handlers()
    try:
        group.start()
        assert wait_for(lambda: group.processed_count == 10)
        signal.raise_signal(signal.SIGTERM)
        assert group.wait() is True
        assert not group.is_running
    finally:
        group._restore_signal_handlers()
    assert {s: signal.getsignal(s) for s in previous} == previous


def test_signal_handlers_are_skipped_off_the_main_thread(broker, registry):
    group = make_group(broker, registry, lambda m: None)
    result = []
    t = threading.Thread(target=lambda: result.append(group.install_signal_handlers()))
    t.start()
    t.join()
    assert result == [False]


def test_constructor_validation_and_double_start(broker, registry):
    for kwargs in ({"num_workers": 0}, {"max_poll_records": 0}, {"max_process_retries": -1}):
        with pytest.raises(ValueError):
            make_group(broker, registry, lambda m: None, **kwargs)
    with pytest.raises(ValueError):
        KafkaConsumerGroup("x", [], lambda m: None)
    group = make_group(broker, registry, lambda m: None)
    group.start()
    with pytest.raises(RuntimeError):
        group.start()
    group.stop()
