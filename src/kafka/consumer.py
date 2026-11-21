"""Kafka consumer group with parallel workers and manual offset commits (Day 5).

:class:`KafkaConsumerGroup` runs ``num_workers`` threads, each with its own
``confluent_kafka.Consumer`` in the same consumer group. Kafka spreads the partitions of the
subscribed topics across the workers; because the producer keys every message by entity id,
all events of one sensor / vehicle / SKU reach one worker, in order.

Delivery semantics: **at least once**
-------------------------------------
Offsets are committed by hand, only after the handler has returned for every message up to
that offset. A crash between handling and committing replays a few messages, so handlers
must tolerate duplicates (the pipeline drops repeated ``message_id`` values).

Failures
--------
* The handler raises :class:`PermanentProcessingError` (undecodable, schema-invalid): the
  message goes to the optional ``dead_letter`` sink and the partition moves on.
* The handler raises anything else: it is retried ``max_process_retries`` times with
  exponential backoff, then dead-lettered. The consumer never blocks a partition forever.
* A worker that dies unexpectedly is restarted with backoff, at most ``max_worker_restarts``
  times; after that the whole group stops and :meth:`wait` reports ``failed``.

Shutdown
--------
:meth:`stop` (or SIGINT / SIGTERM after :meth:`install_signal_handlers`) lets every worker
finish the message it is on, commit, leave the group and close.

Lag
---
A separate monitor thread measures, per assigned partition, ``end offset - consumer
position`` every ``lag_interval_s`` and exports it as the ``scad_kafka_consumer_lag`` gauge.
It runs independently of the workers on purpose: a handler that is slow or stuck is exactly
when lag grows, and a measurement taken between batches would go stale at that moment. The
series of a partition is removed when a rebalance moves it away.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Any, Final, Self

from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

from src.utils.logger import get_logger
from src.utils.metrics import ConsumerMetrics, get_consumer_metrics

if TYPE_CHECKING:
    from src.utils.config import KafkaConsumerConfig

ConsumerFactory = Callable[[dict[str, Any]], Any]
_MAX_BACKOFF_S: Final[float] = 30.0


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ConsumedMessage:
    """A Kafka record, decoupled from the client library's message object."""

    topic: str
    partition: int
    offset: int
    key: str | None
    value: bytes | None
    timestamp_ms: int | None = None


class PermanentProcessingError(Exception):
    """Raised by a handler when retrying the message cannot help (e.g. it is malformed)."""


MessageHandler = Callable[[ConsumedMessage], None]
#: ``(message, error, attempts)``; called once for every message that is given up on.
DeadLetterSink = Callable[[ConsumedMessage, Exception, int], None]


class _StopRequestedError(Exception):
    """Internal: the group is shutting down in the middle of a retry wait."""


# --------------------------------------------------------------------------- #
# Group
# --------------------------------------------------------------------------- #
class KafkaConsumerGroup:
    """A pool of consumer workers feeding one message handler."""

    def __init__(
        self,
        bootstrap_servers: str,
        topics: Sequence[str],
        handler: MessageHandler,
        *,
        group_id: str = "anomaly-detector",
        num_workers: int = 3,
        auto_offset_reset: str = "latest",
        max_poll_records: int = 100,
        fetch_min_bytes: int = 1,
        session_timeout_ms: int = 30_000,
        poll_timeout_s: float = 1.0,
        max_process_retries: int = 3,
        retry_backoff_s: float = 0.1,
        lag_interval_s: float = 5.0,
        max_worker_restarts: int = 5,
        dead_letter: DeadLetterSink | None = None,
        consumer_factory: ConsumerFactory = Consumer,
        metrics: ConsumerMetrics | None = None,
        client_id: str = "scad-consumer",
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        """Create the group (workers start with :meth:`start`).

        Args:
            bootstrap_servers: Comma-separated ``host:port`` broker list.
            topics: Topics every worker subscribes to.
            handler: Called once per message, on a worker thread.
            group_id: Kafka consumer group id.
            num_workers: Parallel consumers. More than the total partition count is useless.
            auto_offset_reset: ``earliest`` or ``latest`` for a group without offsets.
            max_poll_records: Messages fetched and processed per batch (one commit each).
            fetch_min_bytes: Broker waits for this many bytes before answering a fetch.
            session_timeout_ms: Time without heartbeat before a worker is considered dead.
            poll_timeout_s: How long one poll waits for messages.
            max_process_retries: Handler retries before a message is dead-lettered.
            retry_backoff_s: Base delay between handler retries (doubled each time).
            lag_interval_s: Seconds between lag measurements per worker.
            max_worker_restarts: Restarts allowed per worker before the group gives up.
            dead_letter: Receives messages that were given up on.
            consumer_factory: Builds a consumer from a librdkafka config dict (fake in tests).
            metrics: Prometheus series; defaults to the process-wide set.
            client_id: Prefix of the per-worker ``client.id``.
            extra_config: Extra librdkafka settings, applied last.

        Raises:
            ValueError: An argument is out of range.
        """
        if not topics:
            raise ValueError("topics must not be empty")
        if num_workers < 1 or max_poll_records < 1:
            raise ValueError("num_workers and max_poll_records must be >= 1")
        if max_process_retries < 0 or max_worker_restarts < 0:
            raise ValueError("max_process_retries and max_worker_restarts must be >= 0")
        self._log = get_logger(__name__)
        self._topics = list(topics)
        self._handler = handler
        self._dead_letter = dead_letter
        self._factory = consumer_factory
        self._metrics = metrics if metrics is not None else get_consumer_metrics()
        self._num_workers = num_workers
        self._max_poll = max_poll_records
        self._poll_timeout = poll_timeout_s
        self._max_retries = max_process_retries
        self._retry_backoff = retry_backoff_s
        self._lag_interval = lag_interval_s
        self._max_restarts = max_worker_restarts
        self._config: dict[str, Any] = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": False,
            "fetch.min.bytes": fetch_min_bytes,
            "session.timeout.ms": session_timeout_ms,
            "client.id": client_id,
        }
        self._config.update(extra_config or {})

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._monitor: threading.Thread | None = None
        self._workers: list[_Worker] = []
        self._workers_lock = threading.Lock()
        self._lag: dict[tuple[str, int], int] = {}
        self._lag_lock = threading.Lock()
        self._failed = threading.Event()
        self._processed = 0
        self._dead_lettered = 0
        self._count_lock = threading.Lock()
        self._previous_handlers: dict[int, Any] = {}

    @classmethod
    def from_config(
        cls,
        bootstrap_servers: str,
        cfg: KafkaConsumerConfig,
        topics: Sequence[str],
        handler: MessageHandler,
        **kwargs: Any,
    ) -> Self:
        """Build a group from the ``kafka.consumer`` section of the project config."""
        return cls(
            bootstrap_servers,
            topics,
            handler,
            group_id=cfg.group_id,
            num_workers=cfg.num_workers,
            auto_offset_reset=cfg.auto_offset_reset,
            max_poll_records=cfg.max_poll_records,
            fetch_min_bytes=cfg.fetch_min_bytes,
            session_timeout_ms=cfg.session_timeout_ms,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the worker threads (returns immediately)."""
        if self._threads:
            raise RuntimeError("consumer group already started")
        self._stop.clear()
        for index in range(self._num_workers):
            thread = threading.Thread(
                target=self._worker_main,
                args=(index,),
                name=f"scad-consumer-{index}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        self._monitor = threading.Thread(
            target=self._lag_loop, name="scad-lag-monitor", daemon=True
        )
        self._monitor.start()
        self._log.info(
            "consumer_group_started",
            workers=self._num_workers,
            topics=self._topics,
            group_id=self._config["group.id"],
        )

    def stop(self, timeout: float = 30.0) -> bool:
        """Ask all workers to finish and wait for them.

        Returns:
            ``True`` if every worker exited within ``timeout`` seconds.
        """
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(deadline - time.monotonic(), 0.0))
        if self._monitor is not None:
            self._monitor.join(max(deadline - time.monotonic(), 0.0))
        clean = not any(t.is_alive() for t in self._threads)
        self._log.info("consumer_group_stopped", clean=clean, processed=self._processed)
        return clean

    def wait(self) -> bool:
        """Block until all workers have exited. Returns ``False`` if the group failed."""
        for thread in self._threads:
            while thread.is_alive():
                thread.join(0.2)
        return not self._failed.is_set()

    def run_forever(self) -> int:
        """Start, handle SIGINT / SIGTERM, block until stopped. Returns a process exit code."""
        self.install_signal_handlers()
        try:
            self.start()
            self.wait()
        finally:
            self.stop()
            self._restore_signal_handlers()
        return 1 if self._failed.is_set() else 0

    def install_signal_handlers(self) -> bool:
        """Make SIGINT and SIGTERM stop the group. Only possible on the main thread."""
        if threading.current_thread() is not threading.main_thread():
            self._log.warning("signal_handlers_skipped_not_main_thread")
            return False

        def _handle(signum: int, _frame: FrameType | None) -> None:
            self._log.info("shutdown_signal_received", signal=signal.Signals(signum).name)
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[sig] = signal.signal(sig, _handle)
        return True

    def _restore_signal_handlers(self) -> None:
        for sig, previous in self._previous_handlers.items():
            signal.signal(sig, previous)
        self._previous_handlers.clear()

    @property
    def is_running(self) -> bool:
        """True while at least one worker thread is alive."""
        return any(t.is_alive() for t in self._threads)

    @property
    def failed(self) -> bool:
        """True if a worker exhausted its restarts."""
        return self._failed.is_set()

    @property
    def processed_count(self) -> int:
        """Messages handled successfully so far."""
        return self._processed

    @property
    def dead_letter_count(self) -> int:
        """Messages given up on so far."""
        return self._dead_lettered

    def lag_snapshot(self) -> dict[tuple[str, int], int]:
        """Latest measured lag per ``(topic, partition)``."""
        with self._lag_lock:
            return dict(self._lag)

    def total_lag(self) -> int:
        """Sum of the latest lag over all partitions."""
        return sum(self.lag_snapshot().values())

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #
    def _worker_main(self, index: int) -> None:
        restarts = 0
        while not self._stop.is_set():
            worker = _Worker(self, index)
            with self._workers_lock:
                self._workers.append(worker)
            try:
                worker.run()
                return
            except Exception:  # noqa: BLE001 - supervise: any failure restarts the worker
                with self._workers_lock:
                    self._workers.remove(worker)
                restarts += 1
                self._metrics.worker_restarts.inc()
                self._log.exception("consumer_worker_crashed", worker=index, restarts=restarts)
                if restarts > self._max_restarts:
                    self._log.critical("consumer_worker_gave_up", worker=index)
                    self._failed.set()
                    self._stop.set()
                    return
                self._stop.wait(min(0.5 * 2 ** (restarts - 1), _MAX_BACKOFF_S))

    def _lag_loop(self) -> None:
        while not self._stop.wait(max(self._lag_interval, 0.01)):
            with self._workers_lock:
                workers = list(self._workers)
            for worker in workers:
                try:
                    worker.measure_lag()
                except Exception:  # noqa: BLE001 - monitoring must never take the group down
                    self._log.debug("lag_measurement_failed", exc_info=True)

    def _set_lag(self, topic: str, partition: int, lag: int) -> None:
        with self._lag_lock:
            self._lag[(topic, partition)] = lag
        self._metrics.consumer_lag.labels(topic, str(partition)).set(lag)

    def _clear_lag(self, topic: str, partition: int) -> None:
        with self._lag_lock:
            self._lag.pop((topic, partition), None)
        with contextlib.suppress(KeyError):
            self._metrics.consumer_lag.remove(topic, str(partition))


class _Worker:
    """One consumer thread's state: its consumer, pending offsets and assignment."""

    def __init__(self, group: KafkaConsumerGroup, index: int) -> None:
        self.g = group
        self.index = index
        self.log = group._log.bind(worker=index)
        self.consumer: Any = None
        self.pending: dict[tuple[str, int], int] = {}  # next offset to commit
        self.assigned: set[tuple[str, int]] = set()

    def run(self) -> None:
        config = dict(self.g._config)
        config["client.id"] = f"{config['client.id']}-{self.index}"
        self.consumer = self.g._factory(config)
        try:
            self.consumer.subscribe(
                self.g._topics,
                on_assign=self._on_assign,
                on_revoke=self._on_revoke,
                on_lost=self._on_lost,
            )
            while not self.g._stop.is_set():
                messages = self.consumer.consume(self.g._max_poll, self.g._poll_timeout)
                try:
                    self._process_batch(messages)
                finally:
                    self._commit()
        finally:
            self._commit()
            for topic, partition in list(self.assigned):
                self.g._clear_lag(topic, partition)
            try:
                self.consumer.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                self.log.exception("consumer_close_failed")

    # -- batch ---------------------------------------------------------- #
    def _process_batch(self, messages: Sequence[Any]) -> None:
        for raw in messages:
            err = raw.error()
            if err is not None:
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                if err.fatal():
                    raise KafkaException(err)
                self.log.warning("consumer_error", error=str(err))
                continue
            message = ConsumedMessage(
                topic=raw.topic(),
                partition=raw.partition(),
                offset=raw.offset(),
                key=_decode_key(raw.key()),
                value=raw.value(),
                timestamp_ms=_timestamp_ms(raw),
            )
            try:
                status = self._handle(message)
            except _StopRequestedError:
                return  # not handled, so not committed: it is redelivered after restart
            self.g._metrics.messages_consumed.labels(message.topic, status).inc()
            self.pending[(message.topic, message.partition)] = message.offset + 1

    def _handle(self, message: ConsumedMessage) -> str:
        attempts = 0
        while True:
            try:
                self.g._handler(message)
            except PermanentProcessingError as exc:
                self._give_up(message, exc, attempts + 1)
                return "dead_lettered"
            except Exception as exc:  # noqa: BLE001 - handler bugs must not kill the worker
                attempts += 1
                if attempts > self.g._max_retries:
                    self._give_up(message, exc, attempts)
                    return "dead_lettered"
                delay = self.g._retry_backoff * 2 ** (attempts - 1)
                self.log.warning(
                    "message_retry",
                    topic=message.topic,
                    partition=message.partition,
                    offset=message.offset,
                    attempt=attempts,
                    error=str(exc),
                )
                if self.g._stop.wait(delay):
                    raise _StopRequestedError from exc
            else:
                with self.g._count_lock:
                    self.g._processed += 1
                return "ok"

    def _give_up(self, message: ConsumedMessage, exc: Exception, attempts: int) -> None:
        with self.g._count_lock:
            self.g._dead_lettered += 1
        self.log.error(
            "message_dead_lettered",
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            attempts=attempts,
            error=f"{type(exc).__name__}: {exc}",
        )
        if self.g._dead_letter is None:
            return
        try:
            self.g._dead_letter(message, exc, attempts)
        except Exception:  # noqa: BLE001 - the sink failing must not stall the partition
            self.log.critical("dead_letter_sink_failed_message_lost", offset=message.offset)

    # -- offsets -------------------------------------------------------- #
    def _commit(self, only: set[tuple[str, int]] | None = None) -> None:
        items = {tp: off for tp, off in self.pending.items() if only is None or tp in only}
        if not items:
            return
        offsets = [TopicPartition(t, p, o) for (t, p), o in items.items()]
        try:
            self.consumer.commit(offsets=offsets, asynchronous=False)
        except KafkaException as exc:
            self.g._metrics.commit_failures.inc()
            self.log.warning("offset_commit_failed", error=str(exc))
            return  # kept in `pending`; retried with the next batch
        for tp in items:
            if self.pending.get(tp) == items[tp]:
                del self.pending[tp]

    # -- rebalance ------------------------------------------------------ #
    def _on_assign(self, _consumer: Any, partitions: list[Any]) -> None:
        self.g._metrics.rebalances.labels("assign").inc()
        for p in partitions:
            self.assigned.add((p.topic, p.partition))
        self.log.info(
            "partitions_assigned", partitions=[(p.topic, p.partition) for p in partitions]
        )

    def _on_revoke(self, _consumer: Any, partitions: list[Any]) -> None:
        self.g._metrics.rebalances.labels("revoke").inc()
        revoked = {(p.topic, p.partition) for p in partitions}
        self._commit(only=revoked)  # before the partitions move to another worker
        for tp in revoked:
            self.pending.pop(tp, None)
            self.assigned.discard(tp)
            self.g._clear_lag(*tp)

    def _on_lost(self, _consumer: Any, partitions: list[Any]) -> None:
        self.g._metrics.rebalances.labels("lost").inc()
        for p in partitions:
            tp = (p.topic, p.partition)
            self.pending.pop(tp, None)  # ownership is gone; committing would be rejected
            self.assigned.discard(tp)
            self.g._clear_lag(*tp)

    # -- lag ------------------------------------------------------------ #
    def measure_lag(self) -> None:
        """Export lag of the assigned partitions (called from the group's monitor thread)."""
        consumer = self.consumer
        if consumer is None:
            return
        for topic, partition in list(self.assigned):
            try:
                tp = TopicPartition(topic, partition)
                low, high = consumer.get_watermark_offsets(tp, timeout=2.0, cached=False)
                position = consumer.position([tp])[0].offset
            except KafkaException as exc:
                self.log.debug(
                    "lag_lookup_failed", topic=topic, partition=partition, error=str(exc)
                )
                continue
            if position < 0:  # no position yet (nothing consumed, no committed offset)
                position = low
            if (topic, partition) in self.assigned:  # a rebalance may have moved it meanwhile
                self.g._set_lag(topic, partition, max(high - position, 0))


def _decode_key(key: Any) -> str | None:
    if key is None:
        return None
    return key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)


def _timestamp_ms(raw: Any) -> int | None:
    try:
        kind, value = raw.timestamp()
    except Exception:  # noqa: BLE001
        return None
    return int(value) if kind != 0 and value >= 0 else None
