"""Kafka producer with async batching, retries and a dead-letter queue (Day 4).

:class:`KafkaProducerManager` wraps ``confluent_kafka.Producer`` and gives the rest
of the system one small interface::

    from src.kafka.producer import KafkaProducerManager
    from src.utils.config import get_config

    with KafkaProducerManager.from_config(get_config().kafka) as producer:
        producer.send_warehouse_event(reading)      # a Day 2 reading dict
        producer.send_event(simulation_event)       # an orchestrator envelope

How a message travels
---------------------
1. **Validate and wrap.** The record is checked against its Pydantic schema and wrapped
   in a :class:`~src.kafka.schemas.KafkaMessage`. An invalid record, or one larger than
   the topic's ``max.message.bytes``, goes straight to the DLQ - retrying cannot fix it.
2. **Send asynchronously.** ``produce()`` only queues the message; librdkafka batches
   messages per partition and ships a batch when ``batch_size`` bytes have collected or
   ``linger_ms`` has passed, whichever comes first. ``send_*`` therefore never blocks on
   the network. The entity id is the partition key, so one sensor / vehicle / SKU stays
   in order.
3. **Retry with exponential backoff.** If the broker reports a *retriable* failure the
   message is re-sent after ``retry_backoff_ms * 2**(attempt - 1)`` (100, 200, 400 ms by
   default), at most ``max_retries`` times. librdkafka also retries transient errors
   internally before it reports a failure; these retries are the application-level
   second line.
4. **Dead-letter queue.** After the last retry, or at once for a non-retriable error,
   the message is written to ``dlq-events`` as a :class:`~src.kafka.schemas.DLQMessage`:
   the original payload, the topic and key it was meant for, the number of attempts and
   the error. If even the DLQ write fails the message is lost, which is logged at
   ``critical`` level and counted in ``scad_kafka_dlq_failures_total``.

Delivery results arrive in callbacks that run inside ``Producer.poll()``. A small
background thread polls continuously, so callbacks and retries make progress while the
caller is busy generating events. Pass ``background_polling=False`` to drive it by hand
with :meth:`pump` (the unit tests do this to stay deterministic).

Prometheus counters (all labelled by ``topic``):

* ``scad_kafka_messages_sent_total`` - delivered and acknowledged
* ``scad_kafka_messages_failed_total`` - permanently failed (sent on to the DLQ)
* ``scad_kafka_retries_total`` - re-send attempts
* ``scad_kafka_dlq_messages_total`` - written to the DLQ (extra ``reason`` label)
* ``scad_kafka_dlq_failures_total`` - DLQ write itself failed (data lost)
"""

from __future__ import annotations

import heapq
import itertools
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast

from confluent_kafka import KafkaError, KafkaException, Producer
from prometheus_client import REGISTRY, CollectorRegistry, Counter
from pydantic import ValidationError

from src.kafka.schemas import (
    ENTITY_ID_FIELDS,
    SOURCE_TYPES,
    DLQErrorInfo,
    DLQMessage,
    KafkaMessage,
    SourceType,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from types import TracebackType

    from src.utils.config import KafkaConfig, KafkaProducerConfig, TopicsConfig

ProducerFactory = Callable[[dict[str, Any]], Any]

_QUEUE_FULL_WAIT_S: Final[float] = 0.5
_POLL_INTERVAL_S: Final[float] = 0.05
_DLQ_PAYLOAD_PREVIEW_CHARS: Final[int] = 2_000
#: Room kept free for the DLQ envelope around a copied original payload.
_DLQ_ENVELOPE_RESERVE_BYTES: Final[int] = 4_096


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class ProducerMetrics:
    """Prometheus counters for one producer (registered on ``registry``)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else REGISTRY
        self.sent = Counter(
            "scad_kafka_messages_sent_total",
            "Messages delivered and acknowledged by the broker.",
            ["topic"],
            registry=reg,
        )
        self.failed = Counter(
            "scad_kafka_messages_failed_total",
            "Messages that failed permanently and were routed to the DLQ.",
            ["topic"],
            registry=reg,
        )
        self.retries = Counter(
            "scad_kafka_retries_total",
            "Application-level re-send attempts.",
            ["topic"],
            registry=reg,
        )
        self.dlq = Counter(
            "scad_kafka_dlq_messages_total",
            "Messages written to the dead-letter topic.",
            ["topic", "reason"],
            registry=reg,
        )
        self.dlq_failures = Counter(
            "scad_kafka_dlq_failures_total",
            "Dead-letter writes that failed too (the message is lost).",
            ["topic"],
            registry=reg,
        )


_default_metrics: ProducerMetrics | None = None
_default_metrics_lock = threading.Lock()


def get_default_metrics() -> ProducerMetrics:
    """Return the process-wide metrics on the default registry (created once)."""
    global _default_metrics
    with _default_metrics_lock:
        if _default_metrics is None:
            _default_metrics = ProducerMetrics()
        return _default_metrics


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KafkaHealth:
    """Result of :meth:`KafkaProducerManager.check_health`."""

    healthy: bool
    latency_ms: float
    brokers: int = 0
    missing_topics: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form, ready to return from a FastAPI ``/health`` route."""
        return {
            "status": "ok" if self.healthy else "unavailable",
            "latency_ms": round(self.latency_ms, 1),
            "brokers": self.brokers,
            "missing_topics": list(self.missing_topics),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
@dataclass
class _Pending:
    """A message on its way to a topic, with the state needed to retry or dead-letter it."""

    topic: str
    key: str | None
    value: bytes
    source_type: str | None
    attempts: int = 0  # sends made so far, including the first
    is_dlq: bool = False


@dataclass(order=True)
class _Retry:
    due: float
    seq: int
    pending: _Pending = field(compare=False)


def _permanent_codes() -> frozenset[int]:
    """Error codes that no retry can fix (looked up by name so older clients still import)."""
    names = (
        "MSG_SIZE_TOO_LARGE",
        "INVALID_MSG_SIZE",
        "TOPIC_AUTHORIZATION_FAILED",
        "CLUSTER_AUTHORIZATION_FAILED",
        "_UNKNOWN_TOPIC",
        "_UNKNOWN_PARTITION",
        "_INVALID_ARG",
        "_AUTHENTICATION",
    )
    return frozenset(getattr(KafkaError, name) for name in names if hasattr(KafkaError, name))


_PERMANENT_CODES: Final[frozenset[int]] = _permanent_codes()


def _describe_error(err: Any) -> tuple[str, str | None, str, bool]:
    """Return ``(type, code, message, retriable)`` for a ``KafkaError`` or any exception.

    librdkafka's own ``retriable()`` flag is not usable here: it marks a delivery timeout
    (``_MSG_TIMED_OUT``, what a dead or unreachable broker produces) as *not* retriable,
    because the library has already retried internally. That is precisely the case the
    application-level retries exist for, so a ``KafkaError`` counts as retriable unless it
    is fatal or one of the permanent codes in :data:`_PERMANENT_CODES`.
    """
    if isinstance(err, KafkaException) and err.args:
        err = err.args[0]
    if isinstance(err, KafkaError):
        retriable = not err.fatal() and err.code() not in _PERMANENT_CODES
        return "KafkaError", err.name(), err.str(), retriable
    return type(err).__name__, None, str(err), False


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class KafkaProducerManager:
    """Producer for the three data streams, with batching, retries and a DLQ."""

    def __init__(
        self,
        bootstrap_servers: str,
        topics: TopicsConfig,
        *,
        acks: str = "all",
        batch_size: int = 16_384,
        linger_ms: int = 5,
        compression_type: str = "none",
        max_retries: int = 3,
        retry_backoff_ms: int = 100,
        client_id: str = "scad-producer",
        background_polling: bool = True,
        producer_factory: ProducerFactory = Producer,
        metrics: ProducerMetrics | None = None,
        clock: Callable[[], float] = time.monotonic,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create the producer.

        Args:
            bootstrap_servers: Comma-separated ``host:port`` broker list.
            topics: Topic definitions (names and ``max_message_bytes``).
            acks: ``"all"`` (default), ``"1"`` or ``"0"``. Idempotence is enabled
                only for ``"all"``.
            batch_size: Bytes at which librdkafka sends a partition batch.
            linger_ms: How long to wait for a batch to fill before sending it.
            compression_type: ``none``, ``gzip``, ``snappy``, ``lz4`` or ``zstd``.
            max_retries: Application-level retries before a message is dead-lettered.
            retry_backoff_ms: Base retry delay, doubled on every attempt.
            client_id: Shown in broker logs and metrics.
            background_polling: Run a thread that polls the producer. Disable it to
                drive delivery callbacks and retries by hand with :meth:`pump`.
            producer_factory: Builds the underlying producer from a librdkafka config
                dict. Replace it in tests with a fake.
            metrics: Prometheus counters. Defaults to the process-wide set.
            clock: Monotonic clock in seconds (injectable for tests).
            extra_config: Additional librdkafka settings, applied last.

        Raises:
            ValueError: An argument is out of range.
        """
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_backoff_ms < 0:
            raise ValueError(f"retry_backoff_ms must be >= 0, got {retry_backoff_ms}")
        if batch_size < 0 or linger_ms < 0:
            raise ValueError("batch_size and linger_ms must be >= 0")

        self._log = get_logger(__name__)
        self._topics = topics
        self._topic_for: dict[str, str] = {
            "warehouse": topics.warehouse_sensors.name,
            "vehicle": topics.vehicle_gps.name,
            # Order events share the inventory topic; consumers tell them apart by the
            # envelope's source_type. No dedicated order topic exists yet.
            "inventory": topics.inventory_events.name,
            "order": topics.inventory_events.name,
        }
        self._dlq_topic = topics.dlq_events.name
        self._alerts_topic = topics.anomaly_alerts.name
        self._health_topic = topics.model_health.name
        self._max_bytes: dict[str, int] = {
            topic.name: topic.max_message_bytes for topic in topics.as_list()
        }
        self._max_retries = max_retries
        self._backoff_s = retry_backoff_ms / 1000.0
        self._metrics = metrics if metrics is not None else get_default_metrics()
        self._clock = clock

        config: dict[str, Any] = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": client_id,
            "acks": acks,
            "enable.idempotence": acks == "all",
            "batch.size": batch_size,
            "linger.ms": linger_ms,
            "compression.type": compression_type,
            # Bounded delivery time so a dead broker produces a callback (and so a retry
            # or DLQ decision) within seconds instead of librdkafka's default of 5 minutes.
            "message.timeout.ms": 15_000,
            "message.max.bytes": max(self._max_bytes.values()),
        }
        config.update(extra_config or {})
        self._producer = producer_factory(config)

        self._lock = threading.Lock()
        self._retry_heap: list[_Retry] = []
        self._retry_seq = itertools.count()
        self._in_flight = 0
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if background_polling:
            self._thread = threading.Thread(
                target=self._poll_loop, name="scad-kafka-poller", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: KafkaConfig,
        *,
        batch_size: int | None = None,
        linger_ms: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build a manager from the ``kafka`` section of the project config.

        ``batch_size`` and ``linger_ms`` override the YAML values; any other keyword
        is passed to the constructor.
        """
        prod: KafkaProducerConfig = cfg.producer
        return cls(
            cfg.bootstrap_servers,
            cfg.topics,
            acks=prod.acks,
            batch_size=prod.batch_size if batch_size is None else batch_size,
            linger_ms=prod.linger_ms if linger_ms is None else linger_ms,
            compression_type=prod.compression_type,
            max_retries=prod.max_retries,
            retry_backoff_ms=prod.retry_backoff_ms,
            **kwargs,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def in_flight(self) -> int:
        """Messages accepted but not yet delivered or dead-lettered (retries included)."""
        with self._lock:
            return self._in_flight

    def topic_for(self, source_type: str) -> str:
        """Return the topic that ``source_type`` is published to.

        Raises:
            KeyError: ``source_type`` is not one of the known source types.
        """
        return self._topic_for[source_type]

    def send_warehouse_event(self, reading: Mapping[str, Any]) -> bool:
        """Publish a Day 2 warehouse sensor reading to ``warehouse-sensors``."""
        return self._send_record("warehouse", reading)

    def send_gps_event(self, reading: Mapping[str, Any]) -> bool:
        """Publish a Day 3 GPS reading to ``vehicle-gps``."""
        return self._send_record("vehicle", reading)

    def send_inventory_event(self, event: Mapping[str, Any]) -> bool:
        """Publish a Day 3 inventory event to ``inventory-events``."""
        return self._send_record("inventory", event)

    def send_order_event(self, event: Mapping[str, Any]) -> bool:
        """Publish a Day 3 order event to ``inventory-events`` (source_type ``order``)."""
        return self._send_record("order", event)

    def send_event(self, event: Mapping[str, Any]) -> bool:
        """Publish an orchestrator envelope (``source_type`` plus ``payload``).

        Returns:
            ``True`` if the message was queued, ``False`` if it went straight to the DLQ
            (unknown source type, invalid payload or oversize message).
        """
        source = event.get("source_type")
        payload = event.get("payload")
        if source not in SOURCE_TYPES or not isinstance(payload, Mapping):
            self._dead_letter_raw(
                topic=self._dlq_topic,
                key=None,
                source_type=str(source),
                raw=event,
                error=DLQErrorInfo(
                    reason="validation_error",
                    error_type="InvalidEnvelope",
                    message="event needs a known 'source_type' and a mapping 'payload'",
                ),
            )
            return False
        return self._send_record(source, payload)

    def send_batch(self, events: Iterable[Mapping[str, Any]]) -> int:
        """Publish many orchestrator envelopes; return how many were queued."""
        return sum(self.send_event(event) for event in events)

    def send_alert(self, alert: Mapping[str, Any]) -> bool:
        """Publish one alert (``AnomalyEvent.to_dict()``) to the ``anomaly-alerts`` topic.

        The record is keyed by ``entity_id`` so every alert of one entity lands in one partition
        and stays ordered. It gets the same batching, retry and dead-letter handling as the data
        streams.

        Returns:
            ``True`` if the alert was queued, ``False`` if it went straight to the DLQ (no
            ``entity_id``, not serialisable, or larger than the topic allows).
        """
        entity = alert.get("entity_id")
        key = None if entity is None else str(entity)
        try:
            value = json.dumps(alert, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            self._dead_letter_raw(
                topic=self._alerts_topic,
                key=key,
                source_type="alert",
                raw={"event_id": alert.get("event_id"), "entity_id": entity},
                error=DLQErrorInfo(
                    reason="validation_error",
                    error_type=type(exc).__name__,
                    message=f"alert is not JSON serialisable: {exc}"[:1_000],
                ),
            )
            return False
        if key is None or len(value) > self._max_bytes[self._alerts_topic]:
            too_big = key is not None
            self._dead_letter_raw(
                topic=self._alerts_topic,
                key=key,
                source_type="alert",
                raw=value,
                error=DLQErrorInfo(
                    reason="message_too_large" if too_big else "validation_error",
                    error_type="MessageTooLarge" if too_big else "InvalidAlert",
                    message=(
                        f"{len(value)} bytes exceeds max.message.bytes"
                        if too_big
                        else "alert needs an entity_id"
                    ),
                ),
            )
            return False
        pending = _Pending(topic=self._alerts_topic, key=key, value=value, source_type="alert")
        with self._lock:
            self._in_flight += 1
        self._produce(pending)
        return True

    def send_model_health(self, event: Mapping[str, Any]) -> bool:
        """Publish one model-health event (for example ``ModelDriftAlert.to_dict()``).

        The ``model-health`` topic has a single partition and low volume, so the record is keyed
        by its ``event_type`` and gets the same batching, retry and dead-letter handling as the
        data streams.

        Returns:
            ``True`` if the event was queued, ``False`` if it went straight to the DLQ (not
            serialisable, or larger than the topic allows).
        """
        key = str(event.get("event_type") or "model-health")
        try:
            value = json.dumps(event, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            self._dead_letter_raw(
                topic=self._health_topic,
                key=key,
                source_type="model_health",
                raw={"event_type": key},
                error=DLQErrorInfo(
                    reason="validation_error",
                    error_type=type(exc).__name__,
                    message=f"model-health event is not JSON serialisable: {exc}"[:1_000],
                ),
            )
            return False
        if len(value) > self._max_bytes[self._health_topic]:
            self._dead_letter_raw(
                topic=self._health_topic,
                key=key,
                source_type="model_health",
                raw=value,
                error=DLQErrorInfo(
                    reason="message_too_large",
                    error_type="MessageTooLarge",
                    message=f"{len(value)} bytes exceeds max.message.bytes",
                ),
            )
            return False
        pending = _Pending(
            topic=self._health_topic, key=key, value=value, source_type="model_health"
        )
        with self._lock:
            self._in_flight += 1
        self._produce(pending)
        return True

    def pump(self, timeout: float = 0.0) -> int:
        """Run delivery callbacks and re-send retries that are due.

        Called by the background thread; call it yourself when ``background_polling``
        is off. Returns the number of events the producer handled.
        """
        handled = self._producer.poll(timeout)
        self._send_due_retries()
        return int(handled or 0)

    def flush(self, timeout: float = 30.0) -> int:
        """Wait until every accepted message is delivered or dead-lettered.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            The number of messages still in flight (``0`` when everything finished).
        """
        # Wall time on purpose: the injectable clock only schedules retry backoff.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # librdkafka's flush(0) is non-blocking; it keeps delivery callbacks moving.
            self._producer.flush(0.05)
            self._send_due_retries()
            if self.in_flight == 0:
                return 0
            if self._thread is None:
                self._producer.poll(0)
            time.sleep(0.005)
        remaining = self.in_flight
        if remaining:
            self._log.warning("kafka_flush_timeout", in_flight=remaining, timeout_s=timeout)
        return remaining

    def close(self, timeout: float = 30.0) -> int:
        """Flush, stop the polling thread and release the producer.

        Returns the number of messages that could not be flushed in time. Safe to call
        more than once.
        """
        if self._closed:
            return 0
        remaining = self.flush(timeout)
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._log.info("kafka_producer_closed", unflushed=remaining)
        return remaining

    def check_health(self, timeout: float = 5.0) -> KafkaHealth:
        """Check broker connectivity and that every configured topic exists.

        Never raises: a failure is reported in the returned :class:`KafkaHealth`.
        """
        started = self._clock()
        try:
            metadata = self._producer.list_topics(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - any failure means "unhealthy"
            return KafkaHealth(
                healthy=False,
                latency_ms=(self._clock() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (self._clock() - started) * 1000.0
        existing = set(metadata.topics)
        required = {topic.name for topic in self._topics.as_list()}
        missing = tuple(sorted(required - existing))
        brokers = len(metadata.brokers)
        error = None
        if brokers == 0:
            error = "no brokers in cluster metadata"
        elif missing:
            error = f"topics missing (did kafka-init run?): {', '.join(missing)}"
        return KafkaHealth(
            healthy=error is None,
            latency_ms=latency_ms,
            brokers=brokers,
            missing_topics=missing,
            error=error,
        )

    # ------------------------------------------------------------------ #
    # Sending
    # ------------------------------------------------------------------ #
    def _send_record(self, source_type: SourceType | str, record: Mapping[str, Any]) -> bool:
        topic = self._topic_for[source_type]
        key = record.get(ENTITY_ID_FIELDS[source_type])
        key_text = None if key is None else str(key)
        try:
            message = KafkaMessage.from_payload(cast(SourceType, source_type), dict(record))
        except ValidationError as exc:
            self._dead_letter_raw(
                topic=topic,
                key=key_text,
                source_type=source_type,
                raw=record,
                error=DLQErrorInfo(
                    reason="validation_error",
                    error_type="ValidationError",
                    message=str(exc.errors(include_url=False, include_context=False))[:1_000],
                ),
            )
            return False

        value = message.to_json_bytes()
        limit = self._max_bytes[topic]
        if len(value) > limit:
            self._dead_letter_raw(
                topic=topic,
                key=message.entity_id,
                source_type=source_type,
                raw=value,
                error=DLQErrorInfo(
                    reason="message_too_large",
                    error_type="MessageTooLarge",
                    message=f"{len(value)} bytes exceeds max.message.bytes {limit}",
                ),
            )
            return False

        pending = _Pending(topic=topic, key=message.entity_id, value=value, source_type=source_type)
        with self._lock:
            self._in_flight += 1
        self._produce(pending)
        return True

    def _produce(self, pending: _Pending) -> None:
        """Hand ``pending`` to librdkafka; route to retry or DLQ if even that fails."""
        pending.attempts += 1
        try:
            try:
                self._producer.produce(
                    pending.topic,
                    value=pending.value,
                    key=pending.key,
                    on_delivery=lambda err, msg, p=pending: self._on_delivery(p, err, msg),
                )
            except BufferError:
                # Local queue is full: let librdkafka drain it, then try once more.
                self._producer.poll(_QUEUE_FULL_WAIT_S)
                self._producer.produce(
                    pending.topic,
                    value=pending.value,
                    key=pending.key,
                    on_delivery=lambda err, msg, p=pending: self._on_delivery(p, err, msg),
                )
        except BufferError as exc:
            self._handle_failure(pending, exc, stage="produce", retriable=True)
        except KafkaException as exc:
            self._handle_failure(pending, exc, stage="produce")
        except Exception as exc:  # noqa: BLE001 - a producer bug must not kill the stream
            self._handle_failure(pending, exc, stage="produce", retriable=False)

    def _on_delivery(self, pending: _Pending, err: Any, msg: Any) -> None:
        """Delivery callback (runs inside ``poll``/``flush``)."""
        if err is None:
            self._finish(pending)
            if pending.is_dlq:
                return
            self._metrics.sent.labels(pending.topic).inc()
            self._log.debug(
                "kafka_delivered",
                topic=pending.topic,
                key=pending.key,
                partition=_safe_call(msg, "partition"),
                offset=_safe_call(msg, "offset"),
                attempts=pending.attempts,
            )
            return
        self._handle_failure(pending, err)

    # ------------------------------------------------------------------ #
    # Failure handling: retry, then dead-letter
    # ------------------------------------------------------------------ #
    def _handle_failure(
        self,
        pending: _Pending,
        err: Any,
        *,
        stage: Literal["produce", "delivery"] = "delivery",
        retriable: bool | None = None,
    ) -> None:
        error_type, code, message, is_retriable = _describe_error(err)
        if retriable is not None:
            is_retriable = retriable

        if pending.is_dlq:
            # The dead-letter write itself failed: nothing left to fall back on.
            self._finish(pending)
            self._metrics.dlq_failures.labels(self._dlq_topic).inc()
            self._log.critical(
                "kafka_dlq_write_failed_message_lost",
                topic=self._dlq_topic,
                key=pending.key,
                error_type=error_type,
                error_code=code,
                error=message,
            )
            return

        retries_used = pending.attempts - 1
        if is_retriable and retries_used < self._max_retries:
            delay = self._backoff_s * (2**retries_used)
            self._metrics.retries.labels(pending.topic).inc()
            self._log.warning(
                "kafka_delivery_retry",
                topic=pending.topic,
                key=pending.key,
                attempt=pending.attempts,
                retry_in_s=round(delay, 3),
                error_code=code,
                error=message,
            )
            with self._lock:
                heapq.heappush(
                    self._retry_heap,
                    _Retry(self._clock() + delay, next(self._retry_seq), pending),
                )
            return

        self._metrics.failed.labels(pending.topic).inc()
        self._log.error(
            "kafka_delivery_failed",
            topic=pending.topic,
            key=pending.key,
            attempts=pending.attempts,
            error_code=code,
            error=message,
        )
        # Dead-letter first, then release the slot, so in_flight never dips to 0 in between.
        self._dead_letter_raw(
            topic=pending.topic,
            key=pending.key,
            source_type=pending.source_type,
            raw=pending.value,
            attempts=pending.attempts,
            error=DLQErrorInfo(
                reason="delivery_failed" if stage == "delivery" else "produce_failed",
                error_type=error_type,
                message=message,
                error_code=code,
                retriable=is_retriable,
            ),
        )
        self._finish(pending)

    def _send_due_retries(self) -> None:
        """Re-send every queued retry whose backoff has elapsed."""
        while True:
            with self._lock:
                if not self._retry_heap or self._retry_heap[0].due > self._clock():
                    return
                item = heapq.heappop(self._retry_heap)
            self._produce(item.pending)

    def _dead_letter_raw(
        self,
        *,
        topic: str,
        key: str | None,
        source_type: str | None,
        raw: Any,
        error: DLQErrorInfo,
        attempts: int = 0,
    ) -> None:
        """Write ``raw`` (bytes or a mapping) and ``error`` to the DLQ topic."""
        payload, truncated = self._dlq_payload(raw)
        dlq = DLQMessage(
            original_topic=topic,
            original_key=key,
            source_type=source_type,
            attempts=attempts,
            error=error,
            original_payload=payload,
            payload_truncated=truncated,
        )
        self._metrics.dlq.labels(topic, error.reason).inc()
        self._log.warning(
            "kafka_message_dead_lettered",
            original_topic=topic,
            key=key,
            reason=error.reason,
            attempts=attempts,
            dlq_id=str(dlq.dlq_id),
        )
        pending = _Pending(
            topic=self._dlq_topic,
            key=key,
            value=dlq.to_json_bytes(),
            source_type=source_type,
            is_dlq=True,
        )
        with self._lock:
            self._in_flight += 1
        self._produce(pending)

    def _dlq_payload(self, raw: Any) -> tuple[Any, bool]:
        """Return the original payload as JSON-compatible data, truncated if too big."""
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(text)
            except ValueError:
                parsed = text
        else:
            parsed = json.loads(json.dumps(raw, default=str))
        budget = self._max_bytes[self._dlq_topic] - _DLQ_ENVELOPE_RESERVE_BYTES
        if len(json.dumps(parsed).encode("utf-8")) <= budget:
            return parsed, False
        return json.dumps(parsed)[:_DLQ_PAYLOAD_PREVIEW_CHARS], True

    # ------------------------------------------------------------------ #
    # Bookkeeping and polling
    # ------------------------------------------------------------------ #
    def _finish(self, pending: _Pending) -> None:
        """Release one in-flight slot (``pending`` is unused but keeps call sites explicit)."""
        with self._lock:
            self._in_flight = max(self._in_flight - 1, 0)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.pump(_POLL_INTERVAL_S)
            except Exception:  # noqa: BLE001 - keep polling whatever happens
                self._log.exception("kafka_poll_loop_error")
                time.sleep(_POLL_INTERVAL_S)


def _safe_call(obj: Any, method: str) -> Any:
    """Call ``obj.method()`` and return ``None`` if it is missing or raises."""
    try:
        return getattr(obj, method)()
    except Exception:  # noqa: BLE001
        return None
