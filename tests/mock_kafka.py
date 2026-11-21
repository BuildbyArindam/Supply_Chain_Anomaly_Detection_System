"""In-memory Kafka broker for tests (Day 5).

:class:`MockBroker` keeps topics as lists of partitions and implements just enough of the
broker's behaviour for the consumer layer:

* **partitioning by key** (CRC32 of the key, like a stable hash partitioner), so every message
  of one entity lands in one partition and stays ordered;
* **consumer groups**: partitions are spread over the group's members, and when a member joins
  or leaves the others see a rebalance (``on_revoke`` then ``on_assign``), delivered on the
  member's own thread inside ``consume()`` as librdkafka does;
* **committed offsets** per ``(group, topic, partition)``, with ``auto.offset.reset``;
* **fault injection**: failing commits.

:class:`MockProducer` is a drop-in ``producer_factory`` for
:class:`~src.kafka.producer.KafkaProducerManager`; :class:`MockConsumer` a ``consumer_factory``
for :class:`~src.kafka.consumer.KafkaConsumerGroup`::

    broker = MockBroker({"warehouse-sensors": 4, "dlq-events": 1})
    producer = KafkaProducerManager(..., producer_factory=broker.producer_factory)
    group = KafkaConsumerGroup(..., consumer_factory=broker.consumer_factory)
"""

from __future__ import annotations

import itertools
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from confluent_kafka import KafkaError, KafkaException, TopicPartition


@dataclass(frozen=True)
class Record:
    """One stored message."""

    key: bytes | None
    value: bytes | None
    timestamp_ms: int


class MockMessage:
    """Looks like ``confluent_kafka.Message`` for the methods the consumer uses."""

    def __init__(self, topic: str, partition: int, offset: int, record: Record) -> None:
        self._topic, self._partition, self._offset, self._record = topic, partition, offset, record

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def key(self) -> bytes | None:
        return self._record.key

    def value(self) -> bytes | None:
        return self._record.value

    def error(self) -> None:
        return None

    def timestamp(self) -> tuple[int, int]:
        return (1, self._record.timestamp_ms)  # TIMESTAMP_CREATE_TIME


class MockBroker:
    """Thread-safe in-memory broker."""

    def __init__(self, topics: dict[str, int]) -> None:
        self._lock = threading.RLock()
        self.topics: dict[str, list[list[Record]]] = {
            name: [[] for _ in range(n)] for name, n in topics.items()
        }
        self.committed: dict[tuple[str, str, int], int] = {}
        self.commit_log: list[tuple[str, str, int, int]] = []  # (group, topic, partition, offset)
        self._members: dict[str, list[MockConsumer]] = {}
        #: Current owner of every ``(group, topic, partition)``: a partition is handed to its new
        #: owner only after the previous one revoked it (eager rebalance protocol).
        self.owners: dict[tuple[str, str, int], MockConsumer] = {}
        self._ids = itertools.count()
        self.fail_next_commits = 0

    # ---------------------------------------------------------------- factories
    def producer_factory(self, config: dict[str, Any]) -> MockProducer:
        return MockProducer(self, config)

    def consumer_factory(self, config: dict[str, Any]) -> MockConsumer:
        return MockConsumer(self, config)

    # ------------------------------------------------------------------ storage
    def partition_for(self, topic: str, key: bytes | None) -> int:
        n = len(self.topics[topic])
        return 0 if key is None else zlib.crc32(key) % n

    def append(self, topic: str, key: bytes | None, value: bytes | None) -> tuple[int, int]:
        with self._lock:
            if topic not in self.topics:
                raise KafkaException(KafkaError(KafkaError._UNKNOWN_TOPIC))
            partition = self.partition_for(topic, key)
            log = self.topics[topic][partition]
            log.append(Record(key, value, int(time.time() * 1000)))
            return partition, len(log) - 1

    def end_offset(self, topic: str, partition: int) -> int:
        with self._lock:
            return len(self.topics[topic][partition])

    def messages(self, topic: str) -> list[Record]:
        """All records of a topic, partition by partition."""
        with self._lock:
            return [r for part in self.topics[topic] for r in part]

    def total_committed(self, group: str, topic: str) -> int:
        """Sum of committed offsets over the partitions of ``topic``."""
        with self._lock:
            return sum(o for (g, t, _), o in self.committed.items() if g == group and t == topic)

    def lag(self, group: str, topic: str) -> int:
        """Messages in ``topic`` not yet committed by ``group``."""
        with self._lock:
            return sum(
                self.end_offset(topic, p) - self.committed.get((group, topic, p), 0)
                for p in range(len(self.topics[topic]))
            )

    # ----------------------------------------------------------------- membership
    def _join(self, consumer: MockConsumer) -> None:
        with self._lock:
            self._members.setdefault(consumer.group_id, []).append(consumer)
            self._rebalance(consumer.group_id)

    def _leave(self, consumer: MockConsumer) -> None:
        with self._lock:
            members = self._members.get(consumer.group_id, [])
            if consumer in members:
                members.remove(consumer)
            for key in [k for k, owner in self.owners.items() if owner is consumer]:
                del self.owners[key]
            self._rebalance(consumer.group_id)

    def _rebalance(self, group: str) -> None:
        members = self._members.get(group, [])
        if not members:
            return
        wanted = sorted({t for m in members for t in m.topics})
        partitions = [(t, p) for t in wanted for p in range(len(self.topics[t]))]
        targets: dict[int, set[tuple[str, int]]] = {id(m): set() for m in members}
        for index, tp in enumerate(partitions):
            owners = [m for m in members if tp[0] in m.topics]
            if owners:
                targets[id(owners[index % len(owners)])].add(tp)
        for member in members:
            member.target = targets[id(member)]


class MockProducer:
    """Stand-in for ``confluent_kafka.Producer`` that writes to a :class:`MockBroker`."""

    def __init__(self, broker: MockBroker, config: dict[str, Any]) -> None:
        self.broker = broker
        self.config = config
        self._reports: deque[tuple[Callable[..., None] | None, Any, Any]] = deque()
        self.metadata = SimpleNamespace(
            topics={name: None for name in broker.topics}, brokers={1: object()}
        )

    def produce(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | str | None = None,
        on_delivery: Callable[..., None] | None = None,
        **_: Any,
    ) -> None:
        key_bytes = key.encode() if isinstance(key, str) else key
        partition, offset = self.broker.append(topic, key_bytes, value)
        msg = SimpleNamespace(partition=lambda: partition, offset=lambda: offset)
        self._reports.append((on_delivery, None, msg))

    def poll(self, timeout: float = 0) -> int:
        handled = 0
        while self._reports:
            callback, err, msg = self._reports.popleft()
            if callback is not None:
                callback(err, msg)
            handled += 1
        return handled

    def flush(self, timeout: float | None = None) -> int:
        self.poll()
        return 0

    def list_topics(self, timeout: float | None = None) -> Any:
        return self.metadata


class MockConsumer:
    """Stand-in for ``confluent_kafka.Consumer`` backed by a :class:`MockBroker`."""

    def __init__(self, broker: MockBroker, config: dict[str, Any]) -> None:
        self.broker = broker
        self.config = config
        self.group_id: str = config["group.id"]
        self.reset: str = config.get("auto.offset.reset", "latest")
        self.topics: list[str] = []
        self.target: set[tuple[str, int]] = set()
        self.assignment: set[tuple[str, int]] = set()
        self.positions: dict[tuple[str, int], int] = {}
        self._on_assign: Callable[..., None] | None = None
        self._on_revoke: Callable[..., None] | None = None
        self._on_lost: Callable[..., None] | None = None
        self._cursor = 0
        self._closed = False
        self.crash_on_consume: Exception | None = None  # fault injection
        self.name: str = config.get("client.id", "consumer")

    # -------------------------------------------------------------- subscription
    def subscribe(
        self,
        topics: list[str],
        on_assign: Callable[..., None] | None = None,
        on_revoke: Callable[..., None] | None = None,
        on_lost: Callable[..., None] | None = None,
    ) -> None:
        self.topics = list(topics)
        self._on_assign, self._on_revoke, self._on_lost = on_assign, on_revoke, on_lost
        self.broker._join(self)

    def _apply_rebalance(self) -> None:
        with self.broker._lock:
            target = set(self.target)
        if target == self.assignment:
            return
        revoked = self.assignment - target
        if revoked and self._on_revoke is not None:
            self._on_revoke(self, [TopicPartition(t, p) for t, p in sorted(revoked)])
        for tp in revoked:
            self.positions.pop(tp, None)
            with self.broker._lock:
                self.broker.owners.pop((self.group_id, *tp), None)
        with self.broker._lock:
            added = {
                tp
                for tp in target - self.assignment
                if self.broker.owners.setdefault((self.group_id, *tp), self) is self
            }
        target = (self.assignment - revoked) | added
        for t, p in added:
            committed = self.broker.committed.get((self.group_id, t, p))
            if committed is not None:
                self.positions[(t, p)] = committed
            elif self.reset == "earliest":
                self.positions[(t, p)] = 0
            else:
                self.positions[(t, p)] = self.broker.end_offset(t, p)
        self.assignment = target
        if added and self._on_assign is not None:
            self._on_assign(self, [TopicPartition(t, p) for t, p in sorted(added)])

    # ------------------------------------------------------------------ reading
    def consume(self, num_messages: int = 1, timeout: float = -1) -> list[MockMessage]:
        if self.crash_on_consume is not None:
            exc, self.crash_on_consume = self.crash_on_consume, None
            raise exc
        self._apply_rebalance()
        out: list[MockMessage] = []
        ordered = sorted(self.assignment)
        progressed = True
        while len(out) < num_messages and progressed and ordered:
            progressed = False
            for _ in range(len(ordered)):
                tp = ordered[self._cursor % len(ordered)]
                self._cursor += 1
                pos = self.positions[tp]
                with self.broker._lock:
                    log = self.broker.topics[tp[0]][tp[1]]
                    record = log[pos] if pos < len(log) else None
                if record is None:
                    continue
                out.append(MockMessage(tp[0], tp[1], pos, record))
                self.positions[tp] = pos + 1
                progressed = True
                if len(out) >= num_messages:
                    break
        if not out:
            time.sleep(min(max(timeout, 0.0), 0.005))
        return out

    # ------------------------------------------------------------------ offsets
    def commit(
        self,
        message: Any = None,
        offsets: list[TopicPartition] | None = None,
        asynchronous: bool = True,
    ) -> list[TopicPartition] | None:
        with self.broker._lock:
            if self.broker.fail_next_commits > 0:
                self.broker.fail_next_commits -= 1
                raise KafkaException(KafkaError(KafkaError.REQUEST_TIMED_OUT, "commit failed"))
            for tp in offsets or []:
                self.broker.committed[(self.group_id, tp.topic, tp.partition)] = tp.offset
                self.broker.commit_log.append((self.group_id, tp.topic, tp.partition, tp.offset))
        return offsets

    def position(self, partitions: list[TopicPartition]) -> list[TopicPartition]:
        return [
            TopicPartition(
                tp.topic, tp.partition, self.positions.get((tp.topic, tp.partition), -1001)
            )
            for tp in partitions
        ]

    def get_watermark_offsets(
        self, partition: TopicPartition, timeout: float | None = None, cached: bool = False
    ) -> tuple[int, int]:
        return 0, self.broker.end_offset(partition.topic, partition.partition)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.broker._leave(self)
