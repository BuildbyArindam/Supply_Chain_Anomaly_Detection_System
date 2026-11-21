"""Consumer -> stream processor -> feature cache, wired together (Day 5).

:class:`StreamingFeaturePipeline` is the message handler of the consumer group::

    Kafka topic ──> KafkaConsumerGroup ──> pipeline.handle(message)
                                              │ 1. decode + validate the envelope
                                              │ 2. drop duplicates (at-least-once replays)
                                              │ 3. WindowedAggregator.process_message
                                              └ 4. FeatureStore.store_features (every window)

Use :func:`build_pipeline` to assemble everything from ``configs/config.yaml``, or construct
the parts yourself (the integration tests do, with a fake broker).

Error policy
------------
* Bytes that are not valid JSON or do not match the schema raise
  :class:`~src.kafka.consumer.PermanentProcessingError`; the consumer dead-letters them.
* An event too far behind the watermark is *not* an error: it is counted
  (``scad_pipeline_events_total{outcome="dropped_late"}``) and skipped.
* Order events share the inventory topic; they are counted as ``unsupported`` and skipped.

Metrics: ``messages_consumed_total`` is counted by the consumer; this module adds
``processing_latency_seconds``, ``event_age_seconds``, ``pipeline_events_total`` and feeds the
cache hit / miss counters through the cache.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.kafka.consumer import (
    ConsumedMessage,
    ConsumerFactory,
    DeadLetterSink,
    KafkaConsumerGroup,
    PermanentProcessingError,
)
from src.kafka.feature_cache import FeatureCache, FeatureStore
from src.kafka.schemas import DLQErrorInfo, DLQMessage, KafkaMessage
from src.kafka.stream_processor import WindowedAggregator
from src.utils.logger import get_logger
from src.utils.metrics import ConsumerMetrics, get_consumer_metrics

if TYPE_CHECKING:
    from src.utils.config import AppConfig

_DEDUP_CAPACITY = 100_000


class StreamingFeaturePipeline:
    """Decode, de-duplicate, aggregate and cache every consumed message."""

    def __init__(
        self,
        aggregator: WindowedAggregator,
        store: FeatureStore,
        *,
        cache_ttl_s: float | None = None,
        metrics: ConsumerMetrics | None = None,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        dedup_capacity: int = _DEDUP_CAPACITY,
        idle_eviction_every: int = 10_000,
    ) -> None:
        """Create the pipeline.

        Args:
            aggregator: Produces windowed features from events.
            store: Receives every produced :class:`WindowedFeatures`.
            cache_ttl_s: TTL passed to the store; the store's default if ``None``.
            metrics: Prometheus series; defaults to the process-wide set.
            clock: Monotonic clock for latency measurement (injectable for tests).
            wall_clock: Epoch seconds, used for event age.
            dedup_capacity: Message ids remembered for duplicate detection.
            idle_eviction_every: Run :meth:`WindowedAggregator.evict_idle` every N messages.
        """
        self.aggregator = aggregator
        self.store = store
        self._ttl = cache_ttl_s
        self._metrics = metrics if metrics is not None else get_consumer_metrics()
        self._clock = clock
        self._wall = wall_clock
        self._dedup: OrderedDict[str, None] = OrderedDict()
        self._dedup_capacity = dedup_capacity
        self._dedup_lock = threading.Lock()
        self._evict_every = idle_eviction_every
        self._seen = 0
        self._log = get_logger(__name__)
        size_of_store = store.__len__ if hasattr(store, "__len__") else (lambda: 0)
        self._metrics.bind_entity_gauges(size_of_store, lambda: self.aggregator.entity_count)

    # ------------------------------------------------------------------ #
    def handle(self, message: ConsumedMessage) -> None:
        """Process one consumed message (the consumer group's handler).

        Raises:
            PermanentProcessingError: The value is missing, not JSON or not a valid envelope.
        """
        started = self._clock()
        envelope = self._decode(message)
        if self._is_duplicate(str(envelope.message_id)):
            self._metrics.pipeline_events.labels("duplicate").inc()
            return

        outcome = self.aggregator.process_message(envelope)
        self._metrics.pipeline_events.labels(outcome.status).inc()
        for features in outcome.features:
            self.store.store_features(features.entity_id, features, self._ttl)

        elapsed = self._clock() - started
        self._metrics.processing_latency.labels(envelope.source_type).observe(elapsed)
        self._metrics.event_age.observe(max(self._wall() - envelope.produced_at.timestamp(), 0.0))

        self._seen += 1
        if self._evict_every and self._seen % self._evict_every == 0:
            self.aggregator.evict_idle()

    def _decode(self, message: ConsumedMessage) -> KafkaMessage:
        if not message.value:
            raise PermanentProcessingError("empty message value")
        try:
            return KafkaMessage.model_validate_json(message.value)
        except ValidationError as exc:
            raise PermanentProcessingError(
                str(exc.errors(include_url=False, include_context=False))[:500]
            ) from exc

    def _is_duplicate(self, message_id: str) -> bool:
        with self._dedup_lock:
            if message_id in self._dedup:
                self._dedup.move_to_end(message_id)
                return True
            self._dedup[message_id] = None
            if len(self._dedup) > self._dedup_capacity:
                self._dedup.popitem(last=False)
            return False


# --------------------------------------------------------------------------- #
# Dead letters
# --------------------------------------------------------------------------- #
class DeadLetterPublisher:
    """Writes messages the consumer gave up on to the dead-letter topic.

    Uses the same :class:`~src.kafka.schemas.DLQMessage` format as the producer, so one
    replay tool can read both.
    """

    def __init__(self, producer: Any, dlq_topic: str) -> None:
        """Create the publisher.

        Args:
            producer: Anything with ``produce(topic, value=, key=)`` and ``poll(timeout)``
                (a ``confluent_kafka.Producer``).
            dlq_topic: Name of the dead-letter topic.
        """
        self._producer = producer
        self._topic = dlq_topic

    def __call__(self, message: ConsumedMessage, error: Exception, attempts: int) -> None:
        """Publish ``message`` and why it was given up on (a ``DeadLetterSink``)."""
        raw = message.value or b""
        text = raw.decode("utf-8", errors="replace")
        permanent = isinstance(error, PermanentProcessingError)
        dlq = DLQMessage(
            original_topic=message.topic,
            original_key=message.key,
            source_type=None,
            attempts=attempts,
            error=DLQErrorInfo(
                reason="validation_error" if permanent else "processing_failed",
                error_type=type(error).__name__,
                message=str(error)[:1_000],
            ),
            original_payload=text[:10_000],
            payload_truncated=len(text) > 10_000,
        )
        self._producer.produce(self._topic, value=dlq.to_json_bytes(), key=message.key)
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for queued dead letters to be delivered."""
        self._producer.flush(timeout)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
class PipelineRuntime:
    """The assembled pipeline: consumer group, pipeline, cache and aggregator."""

    def __init__(
        self,
        group: KafkaConsumerGroup,
        pipeline: StreamingFeaturePipeline,
        cache: FeatureCache,
        dead_letter: DeadLetterPublisher | None = None,
    ) -> None:
        self.group = group
        self.pipeline = pipeline
        self.cache = cache
        self.dead_letter = dead_letter

    def run_forever(self) -> int:
        """Run until SIGINT / SIGTERM; return a process exit code."""
        try:
            return self.group.run_forever()
        finally:
            if self.dead_letter is not None:
                self.dead_letter.flush()


def build_pipeline(
    cfg: AppConfig,
    *,
    consumer_factory: ConsumerFactory | None = None,
    dlq_producer: Any | None = None,
    metrics: ConsumerMetrics | None = None,
    **group_kwargs: Any,
) -> PipelineRuntime:
    """Assemble consumer group, aggregator, cache and pipeline from the project config.

    Args:
        cfg: The validated application config.
        consumer_factory: Replace the Kafka consumer (tests).
        dlq_producer: Producer used for dead letters. Created from the config if ``None``;
            pass ``False``-y by giving an object with ``produce``/``poll``/``flush`` to fake it.
        metrics: Prometheus series; defaults to the process-wide set.
        **group_kwargs: Extra arguments for :class:`KafkaConsumerGroup`.
    """
    kafka, features = cfg.kafka, cfg.features
    resolved_metrics = metrics if metrics is not None else get_consumer_metrics()

    cache = FeatureCache(
        default_ttl_s=features.cache_ttl_s,
        primary_window_s=min(features.window_seconds),
        metrics=resolved_metrics,
    )
    aggregator = WindowedAggregator(features.window_seconds, features.watermark_tolerance_s)
    pipeline = StreamingFeaturePipeline(
        aggregator, cache, cache_ttl_s=features.cache_ttl_s, metrics=resolved_metrics
    )

    publisher: DeadLetterPublisher | None = None
    if dlq_producer is None:
        from confluent_kafka import Producer

        dlq_producer = Producer(
            {"bootstrap.servers": kafka.bootstrap_servers, "client.id": "scad-consumer-dlq"}
        )
    publisher = DeadLetterPublisher(dlq_producer, kafka.topics.dlq_events.name)

    sink: DeadLetterSink = publisher
    topics = [
        kafka.topics.warehouse_sensors.name,
        kafka.topics.vehicle_gps.name,
        kafka.topics.inventory_events.name,
    ]
    factory_kwargs: dict[str, Any] = {}
    if consumer_factory is not None:
        factory_kwargs["consumer_factory"] = consumer_factory
    group = KafkaConsumerGroup.from_config(
        kafka.bootstrap_servers,
        kafka.consumer,
        topics,
        pipeline.handle,
        dead_letter=sink,
        metrics=resolved_metrics,
        **factory_kwargs,
        **group_kwargs,
    )
    return PipelineRuntime(group, pipeline, cache, publisher)


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime (helper for scripts and tests)."""
    return datetime.now(UTC)
