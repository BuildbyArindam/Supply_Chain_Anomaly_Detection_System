"""Prometheus metrics for the consumer / stream-processing side (Day 5) and inference (Day 15).

One :class:`ConsumerMetrics` object holds every series the consumer layer exports. It takes
an optional ``registry`` so tests can use a private one (the same pattern as the Day 4
``ProducerMetrics``); production code shares the process-wide default via
:func:`get_consumer_metrics`. :class:`StreamingMetrics` (Day 15) does the same for the streaming
inference engine. Day 24 extends this module with the retraining and feedback series.

Series (all prefixed ``scad_``):

* ``messages_consumed_total{topic,status}`` - ``ok`` or ``dead_lettered``
* ``processing_latency_seconds{source_type}`` - decode + aggregate + cache time per message
* ``event_age_seconds`` - producer timestamp to end of processing
* ``pipeline_events_total{outcome}`` - what the aggregator did with each event
* ``cache_hits_total`` / ``cache_misses_total`` and the derived ``cache_hit_ratio`` gauge
* ``kafka_consumer_lag{topic,partition}`` - messages behind the end of the partition
* ``consumer_worker_restarts_total``, ``consumer_commit_failures_total``,
  ``consumer_rebalances_total{event}``
* ``feature_cache_entities`` / ``aggregator_entities`` - live state sizes

Streaming inference (:class:`StreamingMetrics`, Day 15):

* ``detection_latency_seconds{outcome}`` - producer timestamp to alert published (``alert``) or
  to the "normal" verdict (``normal``)
* ``detection_latency_p99_seconds`` - rolling p99, the value the 500 ms SLO is checked against
* ``inference_events_total{outcome}`` - ``processed``, ``sampled_out``, ``failed``, ``duplicate``,
  ``unsupported``
* ``inference_queue_depth`` / ``backpressure_active`` - load on the engine
* ``alerts_published_total{severity}`` / ``alert_publish_failures_total``
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
AGE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class ConsumerMetrics:
    """All consumer-side Prometheus series, registered on ``registry``."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else REGISTRY
        self.registry = reg
        self.messages_consumed = Counter(
            "scad_messages_consumed_total",
            "Messages taken off Kafka, by topic and outcome.",
            ["topic", "status"],
            registry=reg,
        )
        self.processing_latency = Histogram(
            "scad_processing_latency_seconds",
            "Time to decode, aggregate and cache one message.",
            ["source_type"],
            buckets=LATENCY_BUCKETS,
            registry=reg,
        )
        self.event_age = Histogram(
            "scad_event_age_seconds",
            "Seconds between the producer stamping a message and its processing finishing.",
            buckets=AGE_BUCKETS,
            registry=reg,
        )
        self.pipeline_events = Counter(
            "scad_pipeline_events_total",
            "Outcome of each event in the stream processor.",
            ["outcome"],
            registry=reg,
        )
        self.cache_hits = Counter(
            "scad_cache_hits_total", "Feature cache lookups that found data.", registry=reg
        )
        self.cache_misses = Counter(
            "scad_cache_misses_total", "Feature cache lookups that found nothing.", registry=reg
        )
        self.cache_hit_ratio = Gauge(
            "scad_cache_hit_ratio", "Feature cache hits / lookups since start.", registry=reg
        )
        self.cache_hit_ratio.set_function(self._hit_ratio)
        self.consumer_lag = Gauge(
            "scad_kafka_consumer_lag",
            "Messages between the consumer position and the end of the partition.",
            ["topic", "partition"],
            registry=reg,
        )
        self.worker_restarts = Counter(
            "scad_consumer_worker_restarts_total",
            "Consumer workers restarted after an unexpected error.",
            registry=reg,
        )
        self.commit_failures = Counter(
            "scad_consumer_commit_failures_total",
            "Offset commits that failed (retried with the next batch).",
            registry=reg,
        )
        self.rebalances = Counter(
            "scad_consumer_rebalances_total",
            "Consumer group rebalance events seen by workers.",
            ["event"],
            registry=reg,
        )
        self.cache_entities = Gauge(
            "scad_feature_cache_entities", "Entities currently in the feature cache.", registry=reg
        )
        self.aggregator_entities = Gauge(
            "scad_aggregator_entities",
            "Entities with live windows in the stream processor.",
            registry=reg,
        )

    def _hit_ratio(self) -> float:
        hits = self.cache_hits._value.get()
        misses = self.cache_misses._value.get()
        total = hits + misses
        return float(hits / total) if total else 0.0

    def bind_entity_gauges(
        self, cache_size: Callable[[], int], aggregator_size: Callable[[], int]
    ) -> None:
        """Report live state sizes, evaluated at scrape time."""
        self.cache_entities.set_function(lambda: float(cache_size()))
        self.aggregator_entities.set_function(lambda: float(aggregator_size()))


_default: ConsumerMetrics | None = None
_default_lock = threading.Lock()


def get_consumer_metrics() -> ConsumerMetrics:
    """Return the process-wide metrics on the default registry (created once)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = ConsumerMetrics()
        return _default


# --------------------------------------------------------------------------- #
# Streaming inference (Day 15)
# --------------------------------------------------------------------------- #
DETECTION_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class StreamingMetrics:
    """Prometheus series of the :class:`~src.kafka.streaming_inference.StreamingInferenceEngine`."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else REGISTRY
        self.registry = reg
        self.detection_latency = Histogram(
            "scad_detection_latency_seconds",
            "End-to-end latency from the producer timestamp to the verdict (alerts: published).",
            ["outcome"],
            buckets=DETECTION_LATENCY_BUCKETS,
            registry=reg,
        )
        self.latency_p99 = Gauge(
            "scad_detection_latency_p99_seconds",
            "Rolling 99th percentile of the detection latency.",
            registry=reg,
        )
        self.events = Counter(
            "scad_inference_events_total",
            "Events seen by the inference engine, by outcome.",
            ["outcome"],
            registry=reg,
        )
        self.queue_depth = Gauge(
            "scad_inference_queue_depth",
            "Events accepted by the engine and not yet scored.",
            registry=reg,
        )
        self.backpressure_active = Gauge(
            "scad_backpressure_active",
            "1 while the engine is in sampling mode because its queue is too deep.",
            registry=reg,
        )
        self.alerts_published = Counter(
            "scad_alerts_published_total",
            "Alerts handed to the anomaly-alerts topic, by severity.",
            ["severity"],
            registry=reg,
        )
        self.alert_publish_failures = Counter(
            "scad_alert_publish_failures_total",
            "Alerts that could not be handed to the producer.",
            registry=reg,
        )


_default_streaming: StreamingMetrics | None = None


def get_streaming_metrics() -> StreamingMetrics:
    """Return the process-wide streaming metrics on the default registry (created once)."""
    global _default_streaming
    with _default_lock:
        if _default_streaming is None:
            _default_streaming = StreamingMetrics()
        return _default_streaming
