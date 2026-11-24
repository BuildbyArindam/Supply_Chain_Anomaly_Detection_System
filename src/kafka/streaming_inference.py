"""Real-time streaming inference: Kafka -> features -> ensemble -> alerts (Day 15).

:class:`StreamingInferenceEngine` connects the Day 5 consumer layer to the Day 12 ensemble::

    warehouse-sensors ┐
    vehicle-gps       ├─> KafkaConsumerGroup ─> submit() ─> shard queue ─> worker ─┐
    inventory-events  ┘     (threads)            │  backpressure / sampling        │ thread pool
                                                 ▼                                 ▼
                                      InferenceService.evaluate()   aggregate -> FeatureCache ->
                                                                    FeaturePipeline -> ensemble ->
                                                                    severity -> AnomalyEvent ->
                                                                    de-duplicate -> AnomalyStore
                                                                                   │
                                          anomaly-alerts topic <── AlertPublisher ◄┘

The scoring step itself is :meth:`InferenceService.evaluate`, the same code the REST API runs, so a
streamed event and a ``POST /predict`` produce identical verdicts.

Concurrency model
-----------------
* **Sharded queues.** Events are routed to ``crc32(entity_id) % num_workers``. One worker owns one
  entity, so an entity's events are scored strictly in arrival order (the windowed aggregator and
  the online detectors depend on that) while different entities run side by side.
* **One scoring thread behind asyncio workers.** Scoring is CPU-bound Python. Each worker awaits
  its queue, drains up to ``batch_size`` events and scores them on the scoring thread, so the event
  loop stays free for the FastAPI server that shares the process. ``InferenceService`` holds one
  lock around the stateful detectors, so extra scoring threads add no speed. Worse, every lock
  hand-off between threads needs the GIL, so under GIL pressure from other busy threads a 4-thread
  pool can fall far behind one thread (seen once, with spinning mock producer pollers on a 2-core
  machine: about 100 events/s against more than 300). Hence ``scoring_threads=1`` by default;
  throughput is bounded by single-core scoring speed.
* **Non-blocking hand-off from Kafka threads.** A consumer thread admits the event itself
  (de-duplication, sampling, depth accounting) and hands it to the loop with
  ``call_soon_threadsafe``, without waiting for the loop (see :meth:`submit_threadsafe`).

Backpressure
------------
``queue_depth`` counts events accepted but not yet scored. Above ``backpressure_depth`` the engine
enters *sampling mode*: events of **non-critical** entities are thinned to every
``sampling_every_n``-th event per entity (``scad_inference_events_total{outcome="sampled_out"}``).
Critical entities (cold-storage temperature by default, see :func:`default_is_critical`) are never
thinned. Sampled-out events are *not* ingested at all, so that entity's windows are built from
fewer points while the mode lasts. The mode ends once depth falls to half the threshold
(hysteresis, so it does not flap). If the queues themselves fill up, :meth:`submit` waits, which
blocks the Kafka consumer thread and leaves the backlog in Kafka rather than in memory.

Delivery semantics
------------------
The consumer commits an offset when its handler returns, which is when the event is *queued*, not
when it has been scored. After a crash, at most the queued events (bounded by the queue capacity)
are not scored again; the in-memory features they would have fed are lost with the process anyway.
A graceful :meth:`stop` stops consuming first and then drains every queue, so a clean shutdown
loses nothing. Replays (at-least-once redelivery) are dropped by ``message_id``.

Latency
-------
Latency is measured per scored event from the envelope's ``produced_at`` (the producer's wall clock)
to the moment the alert was *handed to the producer* (for alerts) or the verdict was reached (for
normal events). It therefore includes queueing and scoring but not broker delivery time. It assumes
the producer's clock and this process's clock agree; skew shows up directly in the numbers.
"""

from __future__ import annotations

import asyncio
import threading
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import numpy as np
from pydantic import ValidationError

from src.kafka.consumer import (
    ConsumedMessage,
    DeadLetterSink,
    KafkaConsumerGroup,
    PermanentProcessingError,
)
from src.kafka.schemas import KafkaMessage
from src.utils.logger import get_logger
from src.utils.metrics import StreamingMetrics, get_streaming_metrics

if TYPE_CHECKING:
    from src.api.service import InferenceService
    from src.kafka.producer import KafkaProducerManager
    from src.models.ensemble_detector import AnomalyEvent
    from src.utils.config import AppConfig

_log = get_logger(__name__)

SubmitResult = Literal["accepted", "sampled_out", "duplicate", "unsupported", "closed"]

DEFAULT_LATENCY_TARGET_S: Final = 0.5
DEFAULT_LATENCY_WINDOW: Final = 50_000
_DEDUP_CAPACITY: Final = 100_000
_QUEUE_CAPACITY_FACTOR: Final = 10  # hard queue limit = this many times the backpressure depth
_IDLE_POLL_S: Final = 0.01
_FULL_POLL_S: Final = 0.001


# --------------------------------------------------------------------------- #
# Alert publishing
# --------------------------------------------------------------------------- #
class AlertPublisher(Protocol):
    """Hands an :class:`AnomalyEvent` to the outside world (normally the alerts topic)."""

    def publish(self, event: AnomalyEvent) -> bool:
        """Publish ``event``; return ``True`` if it was accepted for delivery."""
        ...


class KafkaAlertPublisher:
    """Publishes alerts to the ``anomaly-alerts`` topic through a :class:`KafkaProducerManager`."""

    def __init__(self, producer: KafkaProducerManager) -> None:
        """Wrap ``producer`` (it owns batching, retries and the dead-letter queue)."""
        self._producer = producer

    def publish(self, event: AnomalyEvent) -> bool:
        """Queue the alert; ``False`` means it was dead-lettered instead of queued."""
        return self._producer.send_alert(event.to_dict())


# --------------------------------------------------------------------------- #
# Latency tracking
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatencySnapshot:
    """Latency percentiles over the tracker's rolling window (seconds)."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float]:
        """JSON-friendly form, latencies in milliseconds."""
        return {
            "count": float(self.count),
            "mean_ms": self.mean * 1000.0,
            "p50_ms": self.p50 * 1000.0,
            "p95_ms": self.p95 * 1000.0,
            "p99_ms": self.p99 * 1000.0,
            "max_ms": self.max * 1000.0,
        }


class LatencyTracker:
    """Rolling end-to-end latency percentiles, exported as a Prometheus histogram.

    :meth:`observe` is cheap and thread-safe. Percentiles are computed on demand over the last
    ``window`` samples (nearest-rank, via numpy), which is exact for the window, unlike percentiles
    interpolated from histogram buckets.
    """

    def __init__(
        self,
        target_p99_s: float = DEFAULT_LATENCY_TARGET_S,
        *,
        window: int = DEFAULT_LATENCY_WINDOW,
        min_samples: int = 100,
        metrics: StreamingMetrics | None = None,
    ) -> None:
        """Create the tracker.

        Args:
            target_p99_s: The SLO; :meth:`check_slo` warns when the rolling p99 exceeds it.
            window: Number of most recent samples the percentiles are computed over.
            min_samples: Samples needed before an SLO breach is reported (avoids noise at start).
            metrics: Prometheus series; defaults to the process-wide set.

        Raises:
            ValueError: ``target_p99_s`` is not positive or ``window`` is below 1.
        """
        if target_p99_s <= 0:
            raise ValueError("target_p99_s must be > 0")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.target_p99_s = target_p99_s
        self._min_samples = min_samples
        self._metrics = metrics if metrics is not None else get_streaming_metrics()
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total = 0
        self._breaches = 0

    def observe(self, seconds: float, outcome: str = "normal") -> None:
        """Record one latency (negative values from clock skew are clamped to zero)."""
        value = seconds if seconds > 0.0 else 0.0
        with self._lock:
            self._samples.append(value)
            self._total += 1
        self._metrics.detection_latency.labels(outcome).observe(value)

    @property
    def total(self) -> int:
        """Samples observed since creation (the window only keeps the most recent ones)."""
        with self._lock:
            return self._total

    @property
    def breaches(self) -> int:
        """How many :meth:`check_slo` calls found the p99 above the target."""
        return self._breaches

    def snapshot(self) -> LatencySnapshot:
        """Return p50 / p95 / p99 / max over the rolling window (all zero if empty)."""
        with self._lock:
            data = np.fromiter(self._samples, dtype=float, count=len(self._samples))
        if data.size == 0:
            return LatencySnapshot(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        p50, p95, p99 = (float(v) for v in np.percentile(data, [50, 95, 99], method="higher"))
        return LatencySnapshot(
            count=int(data.size),
            mean=float(np.mean(data)),
            p50=p50,
            p95=p95,
            p99=p99,
            max=float(np.max(data)),
        )

    def check_slo(self) -> bool:
        """Update the p99 gauge and log a WARNING if the p99 is above the target.

        Returns:
            ``True`` while the SLO is met (or there are too few samples to judge).
        """
        snap = self.snapshot()
        self._metrics.latency_p99.set(snap.p99)
        if snap.count < self._min_samples or snap.p99 <= self.target_p99_s:
            return True
        self._breaches += 1
        _log.warning(
            "detection_latency_slo_breached",
            p99_ms=round(snap.p99 * 1000.0, 1),
            target_ms=round(self.target_p99_s * 1000.0, 1),
            p50_ms=round(snap.p50 * 1000.0, 1),
            p95_ms=round(snap.p95 * 1000.0, 1),
            samples=snap.count,
        )
        return False


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def default_is_critical(message: KafkaMessage) -> bool:
    """Whether ``message`` is exempt from sampling: cold-storage temperature sensors.

    These carry the highest business-impact weight (1.0), so they are never thinned under load.
    """
    if message.source_type != "warehouse":
        return False
    payload = message.payload
    return (
        getattr(payload, "reading_type", None) == "temperature"
        and payload.metadata.get("storage_type") == "cold_storage"
    )


@dataclass(frozen=True)
class EngineStats:
    """A point-in-time view of the engine's counters.

    ``dropped`` is the number of events that were lost: they failed during scoring or were
    refused because the engine was closed. Events skipped on purpose (``sampled_out``), replays
    (``duplicates``) and order events (``unsupported``) are not losses.
    """

    received: int
    processed: int
    failed: int
    rejected: int
    sampled_out: int
    duplicates: int
    unsupported: int
    alerts_published: int
    alert_publish_failures: int
    queue_depth: int
    peak_queue_depth: int
    backpressure_active: bool
    backpressure_activations: int

    @property
    def dropped(self) -> int:
        """Events lost (scoring failures plus refusals)."""
        return self.failed + self.rejected

    def to_dict(self) -> dict[str, int | bool]:
        """JSON-friendly form."""
        return {
            "received": self.received,
            "processed": self.processed,
            "failed": self.failed,
            "rejected": self.rejected,
            "dropped": self.dropped,
            "sampled_out": self.sampled_out,
            "duplicates": self.duplicates,
            "unsupported": self.unsupported,
            "alerts_published": self.alerts_published,
            "alert_publish_failures": self.alert_publish_failures,
            "queue_depth": self.queue_depth,
            "peak_queue_depth": self.peak_queue_depth,
            "backpressure_active": self.backpressure_active,
            "backpressure_activations": self.backpressure_activations,
        }


@dataclass(slots=True)
class _Work:
    """One accepted event on its way through the engine."""

    message: KafkaMessage
    started_at: float  # epoch seconds the latency is measured from


class StreamingInferenceEngine:
    """Consumes events, scores them with the ensemble and publishes alerts.

    Typical use (the runner script does exactly this)::

        service = InferenceService(cfg); service.start()
        engine = StreamingInferenceEngine.from_config(service, publisher=KafkaAlertPublisher(p))
        await engine.start()
        engine.start_consuming()          # Kafka -> engine
        ...
        await engine.stop()               # stops consuming, drains the queues, final snapshot

    Without Kafka (tests, replays), feed it directly with ``await engine.submit(message)``.
    """

    def __init__(
        self,
        service: InferenceService,
        *,
        publisher: AlertPublisher | None = None,
        num_workers: int = 4,
        backpressure_depth: int = 1000,
        sampling_every_n: int = 5,
        batch_size: int = 32,
        scoring_threads: int = 1,
        max_queue: int | None = None,
        latency: LatencyTracker | None = None,
        metrics: StreamingMetrics | None = None,
        is_critical: Callable[[KafkaMessage], bool] = default_is_critical,
        snapshots: bool = True,
        snapshot_path: Path | None = None,
        snapshot_interval_s: float | None = None,
        monitor_interval_s: float = 5.0,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """Create the engine (nothing runs until :meth:`start`).

        Args:
            service: Scoring service whose models must already be loaded (``service.start()``).
            publisher: Where alerts go; ``None`` stores them but publishes nothing.
            num_workers: Parallel queue workers (also the size of the scoring thread pool).
            backpressure_depth: Queue depth above which sampling mode starts.
            sampling_every_n: In sampling mode keep every N-th event of a non-critical entity.
            batch_size: Events a worker scores per thread-pool hop.
            scoring_threads: Threads that run the scoring step. ``InferenceService`` holds one lock
                around it, so more than one adds no speed and, because every lock hand-off needs
                the GIL, makes scoring much slower when other threads (the Kafka consumers and
                producer pollers) are busy. Raise it only if the service's detectors become
                lock-free or release the GIL.
            max_queue: Hard limit on queued events; :meth:`submit` waits when it is reached
                (default: ten times ``backpressure_depth``).
            latency: Latency tracker; one with the 500 ms target is created if omitted.
            metrics: Prometheus series; defaults to the process-wide set.
            is_critical: Predicate for events that are never sampled out.
            snapshots: Persist the anomaly store periodically while running.
            snapshot_path: Snapshot file (default: ``service.anomaly_snapshot_path``).
            snapshot_interval_s: Snapshot period (default: ``inference.snapshot_interval_s``).
            monitor_interval_s: How often the SLO check runs.
            wall_clock: Epoch seconds, injectable for tests.

        Raises:
            ValueError: A numeric argument is out of range.
        """
        if num_workers < 1 or batch_size < 1 or sampling_every_n < 1 or backpressure_depth < 1:
            raise ValueError(
                "num_workers, batch_size, sampling_every_n and backpressure_depth must be >= 1"
            )
        if monitor_interval_s <= 0:
            raise ValueError("monitor_interval_s must be > 0")
        self._service = service
        self._publisher = publisher
        self._num_workers = num_workers
        self._backpressure_depth = backpressure_depth
        self._resume_depth = backpressure_depth // 2
        self._sample_every = sampling_every_n
        self._batch_size = batch_size
        if scoring_threads < 1:
            raise ValueError("scoring_threads must be >= 1")
        self._scoring_threads = scoring_threads
        self._max_queue = max_queue or backpressure_depth * _QUEUE_CAPACITY_FACTOR
        if self._max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        self._metrics = metrics if metrics is not None else get_streaming_metrics()
        self.latency = latency or LatencyTracker(metrics=self._metrics)
        self._is_critical = is_critical
        self._snapshots = snapshots
        self._snapshot_path = snapshot_path
        self._snapshot_interval_s = snapshot_interval_s
        self._monitor_interval_s = monitor_interval_s
        self._wall = wall_clock

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: list[asyncio.Queue[_Work]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._monitor_task: asyncio.Task[None] | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._group: KafkaConsumerGroup | None = None
        self._accepting = False

        self._lock = threading.Lock()  # guards every counter below
        self._seen_ids: OrderedDict[str, None] = OrderedDict()
        self._sample_counts: dict[str, int] = {}
        self._depth = 0
        self._peak_depth = 0
        self._backpressure = False
        self._activations = 0
        self._received = 0
        self._processed = 0
        self._failed = 0
        self._rejected = 0
        self._sampled = 0
        self._duplicates = 0
        self._unsupported = 0
        self._published = 0
        self._publish_failures = 0
        self._metrics.queue_depth.set_function(lambda: float(self._depth))
        self._metrics.backpressure_active.set_function(lambda: 1.0 if self._backpressure else 0.0)

    @classmethod
    def from_config(
        cls, service: InferenceService, *, publisher: AlertPublisher | None = None, **overrides: Any
    ) -> StreamingInferenceEngine:
        """Create the engine with the ``inference`` section of the service's config."""
        inference = service.cfg.inference
        params: dict[str, Any] = {
            "num_workers": inference.num_workers,
            "backpressure_depth": inference.backpressure_queue_depth,
            "sampling_every_n": inference.sampling_every_n,
            "snapshot_interval_s": inference.snapshot_interval_s,
        }
        params.update(overrides)
        return cls(service, publisher=publisher, **params)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        """Whether the workers are up and :meth:`submit` accepts events."""
        return self._accepting

    @property
    def num_workers(self) -> int:
        """Number of queue workers (and scoring threads)."""
        return self._num_workers

    async def start(self) -> None:
        """Start the workers, the SLO monitor and the periodic anomaly snapshots.

        Raises:
            RuntimeError: Already started, or the service has no models loaded.
        """
        if self._loop is not None:
            raise RuntimeError("engine already started")
        if not self._service.models.is_loaded:
            raise RuntimeError("service models are not loaded; call service.start() first")
        self._loop = asyncio.get_running_loop()
        # Unbounded on purpose: ``max_queue`` limits the global depth, checked at admission.
        self._queues = [asyncio.Queue() for _ in range(self._num_workers)]
        self._pool = ThreadPoolExecutor(
            max_workers=self._scoring_threads, thread_name_prefix="scad-infer"
        )
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"scad-infer-worker-{i}")
            for i in range(self._num_workers)
        ]
        self._monitor_task = asyncio.create_task(self._monitor(), name="scad-infer-monitor")
        if self._snapshots:
            interval = self._snapshot_interval_s or float(
                self._service.cfg.inference.snapshot_interval_s
            )
            path = self._snapshot_path or self._service.anomaly_snapshot_path
            self._service.anomaly_store.start_periodic_snapshots(path, interval)
        self._accepting = True
        _log.info(
            "streaming_engine_started",
            workers=self._num_workers,
            backpressure_depth=self._backpressure_depth,
            sampling_every_n=self._sample_every,
            max_queue=self._max_queue,
        )

    async def stop(self, drain_timeout_s: float = 30.0) -> bool:
        """Stop consuming, score everything already queued, then shut the workers down.

        Args:
            drain_timeout_s: How long to wait for the queues to empty.

        Returns:
            ``True`` if every accepted event was scored before the timeout.
        """
        if self._loop is None:
            return True
        loop = asyncio.get_running_loop()
        if self._group is not None:
            group = self._group
            await loop.run_in_executor(None, group.stop, drain_timeout_s)
            self._group = None
        self._accepting = False  # the consumer is gone; late submit() calls are refused
        drained = True
        if not await self.wait_idle(drain_timeout_s):
            drained = False
            _log.error("streaming_engine_drain_timeout", pending=self._depth)
        for task in [*self._tasks, *([self._monitor_task] if self._monitor_task else [])]:
            task.cancel()
        await asyncio.gather(
            *self._tasks, *filter(None, [self._monitor_task]), return_exceptions=True
        )
        self._tasks = []
        self._monitor_task = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._snapshots:
            self._service.anomaly_store.stop_periodic_snapshots(final=True)
        self.latency.check_slo()
        self._loop = None
        _log.info("streaming_engine_stopped", drained=drained, **self.stats().to_dict())
        return drained

    # ------------------------------------------------------------------ #
    # Kafka input
    # ------------------------------------------------------------------ #
    def start_consuming(
        self,
        *,
        group_id: str | None = None,
        consumer_factory: Any | None = None,
        dead_letter: DeadLetterSink | None = None,
        **group_kwargs: Any,
    ) -> KafkaConsumerGroup:
        """Subscribe to the three data topics and feed the engine (call after :meth:`start`).

        The group id defaults to ``<kafka.consumer.group_id>-inference`` so this consumer does not
        compete with the Day 5 feature pipeline for the same partitions.

        Args:
            group_id: Consumer group id override.
            consumer_factory: Replace the Kafka consumer (tests use the in-memory broker).
            dead_letter: Where messages the engine cannot take are sent.
            **group_kwargs: Extra :class:`KafkaConsumerGroup` arguments.

        Raises:
            RuntimeError: The engine is not running or is already consuming.
        """
        if not self._accepting or self._loop is None:
            raise RuntimeError("engine is not running; await start() first")
        if self._group is not None:
            raise RuntimeError("engine is already consuming")
        cfg: AppConfig = self._service.cfg
        topics = cfg.kafka.topics
        if consumer_factory is not None:
            group_kwargs["consumer_factory"] = consumer_factory
        consumer_cfg = cfg.kafka.consumer.model_copy(
            update={"group_id": group_id or f"{cfg.kafka.consumer.group_id}-inference"}
        )
        group = KafkaConsumerGroup.from_config(
            cfg.kafka.bootstrap_servers,
            consumer_cfg,
            [topics.warehouse_sensors.name, topics.vehicle_gps.name, topics.inventory_events.name],
            self._handle_consumed,
            dead_letter=dead_letter,
            **group_kwargs,
        )
        group.start()
        self._group = group
        return group

    @property
    def consumer_group(self) -> KafkaConsumerGroup | None:
        """The running consumer group, if :meth:`start_consuming` was called."""
        return self._group

    def _handle_consumed(self, consumed: ConsumedMessage) -> None:
        """Consumer-thread handler: decode, then queue (blocks while the queues are full)."""
        if not consumed.value:
            raise PermanentProcessingError("empty message value")
        try:
            message = KafkaMessage.model_validate_json(consumed.value)
        except ValidationError as exc:
            raise PermanentProcessingError(
                str(exc.errors(include_url=False, include_context=False))[:500]
            ) from exc
        if self.submit_threadsafe(message) == "closed":
            raise RuntimeError("inference engine is shutting down")  # retried, not committed

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    async def submit(self, message: KafkaMessage) -> SubmitResult:
        """Offer one event to the engine; waits while the engine holds ``max_queue`` events.

        Returns:
            ``accepted``: queued for scoring. ``sampled_out``: skipped on purpose because the
            engine is in sampling mode. ``duplicate``: already seen (a Kafka replay).
            ``unsupported``: order events carry no detectable signal. ``closed``: the engine is
            not accepting events (counted as a drop).
        """
        while True:
            verdict, work, shard = self._admit(message)
            if verdict != "full":
                break
            await asyncio.sleep(_FULL_POLL_S)
        if work is not None:
            self._queues[shard].put_nowait(work)
            return "accepted"
        return verdict

    def submit_threadsafe(self, message: KafkaMessage, timeout_s: float = 30.0) -> SubmitResult:
        """:meth:`submit` for threads other than the event loop (the Kafka consumer workers).

        Admission runs in the calling thread and the event is handed to the loop without waiting
        for it. (A blocking round trip per message makes each consumer thread wait for the GIL
        behind the scoring thread; under heavy GIL contention in testing that capped the Kafka
        path near 250 events/s.) While the engine holds ``max_queue`` events this call sleeps,
        which is how pressure reaches Kafka.

        Raises:
            RuntimeError: The engine has not been started.
            TimeoutError: The engine stayed full for ``timeout_s`` (the consumer will retry).
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError("engine is not running; await start() first")
        deadline = time.monotonic() + timeout_s
        while True:
            verdict, work, shard = self._admit(message)
            if verdict != "full":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"inference engine stayed full for {timeout_s}s")
            time.sleep(_FULL_POLL_S)
        if work is not None:
            loop.call_soon_threadsafe(self._queues[shard].put_nowait, work)
            return "accepted"
        return verdict

    def _admit(
        self, message: KafkaMessage
    ) -> tuple[SubmitResult | Literal["full"], _Work | None, int]:
        """Decide what happens to ``message``; thread-safe and never blocks.

        Returns ``(verdict, work, shard)``. ``work`` is set only for an accepted event, which
        the caller must put on queue ``shard``; its depth is already counted. ``"full"`` means
        nothing was recorded and the caller should wait and ask again.
        """
        if not self._accepting:
            with self._lock:
                self._received += 1
                self._rejected += 1
            self._metrics.events.labels("rejected").inc()
            return "closed", None, 0
        message_id = str(message.message_id)
        entity = message.entity_id
        shard = zlib.crc32(entity.encode("utf-8")) % self._num_workers

        verdict: SubmitResult | Literal["full"] | None = None
        with self._lock:
            if self._depth >= self._max_queue and message.source_type != "order":
                return "full", None, shard
            self._received += 1
            if message.source_type == "order":
                self._unsupported += 1
                verdict = "unsupported"
            elif message_id in self._seen_ids:
                self._seen_ids.move_to_end(message_id)
                self._duplicates += 1
                verdict = "duplicate"
            else:
                self._seen_ids[message_id] = None
                if len(self._seen_ids) > _DEDUP_CAPACITY:
                    self._seen_ids.popitem(last=False)
                self._update_backpressure()
                if self._backpressure and not self._is_critical(message):
                    seen = self._sample_counts.get(entity, 0)
                    self._sample_counts[entity] = seen + 1
                    if seen % self._sample_every != 0:
                        self._sampled += 1
                        verdict = "sampled_out"
                if verdict is None:
                    self._depth += 1
                    self._peak_depth = max(self._peak_depth, self._depth)
        if verdict is not None:
            self._metrics.events.labels(verdict).inc()
            return verdict, None, shard
        return "accepted", _Work(message, message.produced_at.timestamp()), shard

    def _update_backpressure(self) -> None:
        """Enter / leave sampling mode with hysteresis (call with ``self._lock`` held)."""
        if not self._backpressure and self._depth > self._backpressure_depth:
            self._backpressure = True
            self._activations += 1
            _log.warning(
                "backpressure_on",
                queue_depth=self._depth,
                threshold=self._backpressure_depth,
                sampling_every_n=self._sample_every,
            )
        elif self._backpressure and self._depth <= self._resume_depth:
            self._backpressure = False
            self._sample_counts.clear()
            _log.info("backpressure_off", queue_depth=self._depth)

    # ------------------------------------------------------------------ #
    # Workers
    # ------------------------------------------------------------------ #
    async def _worker(self, index: int) -> None:
        """Drain one shard: take what is queued (up to a batch) and score it on a pool thread."""
        queue = self._queues[index]
        loop = asyncio.get_running_loop()
        pool = self._pool
        assert pool is not None
        while True:
            batch = [await queue.get()]
            while len(batch) < self._batch_size:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await loop.run_in_executor(pool, self._score_batch, batch)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - one bad batch must not kill the worker
                _log.error("inference_batch_failed", error=repr(error), events=len(batch))
                with self._lock:
                    self._failed += len(batch)
                self._metrics.events.labels("failed").inc(len(batch))
            finally:
                with self._lock:
                    self._depth -= len(batch)
                    self._update_backpressure()

    def _score_batch(self, batch: list[_Work]) -> None:
        """Score ``batch`` in order, on a pool thread; a failing event never aborts the rest."""
        for work in batch:
            self._score_one(work)

    def _score_one(self, work: _Work) -> None:
        message = work.message
        try:
            evaluation = self._service.evaluate(message, message.payload.model_dump())
            alert = evaluation.alert
            if alert is not None:
                self._publish(alert)
            self.latency.observe(
                self._wall() - work.started_at, "alert" if alert is not None else "normal"
            )
        except Exception as error:  # noqa: BLE001 - count it and keep the stream moving
            with self._lock:
                self._failed += 1
            self._metrics.events.labels("failed").inc()
            _log.error(
                "inference_failed",
                entity_id=message.entity_id,
                source_type=message.source_type,
                error=repr(error),
            )
            return
        with self._lock:
            self._processed += 1
        self._metrics.events.labels("processed").inc()

    def _publish(self, alert: AnomalyEvent) -> None:
        """Hand an alert to the publisher; a failure is counted and logged, never raised."""
        if self._publisher is None:
            return
        try:
            accepted = self._publisher.publish(alert)
        except Exception as error:  # noqa: BLE001 - alerting must not stop scoring
            accepted = False
            _log.error("alert_publish_error", event_id=alert.event_id, error=repr(error))
        with self._lock:
            if accepted:
                self._published += 1
            else:
                self._publish_failures += 1
        if accepted:
            self._metrics.alerts_published.labels(alert.severity.name).inc()
        else:
            self._metrics.alert_publish_failures.inc()

    async def _monitor(self) -> None:
        """Check the latency SLO on a timer."""
        while True:
            await asyncio.sleep(self._monitor_interval_s)
            self.latency.check_slo()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def stats(self) -> EngineStats:
        """Return the current counters."""
        with self._lock:
            return EngineStats(
                received=self._received,
                processed=self._processed,
                failed=self._failed,
                rejected=self._rejected,
                sampled_out=self._sampled,
                duplicates=self._duplicates,
                unsupported=self._unsupported,
                alerts_published=self._published,
                alert_publish_failures=self._publish_failures,
                queue_depth=self._depth,
                peak_queue_depth=self._peak_depth,
                backpressure_active=self._backpressure,
                backpressure_activations=self._activations,
            )

    async def wait_idle(self, timeout_s: float = 30.0) -> bool:
        """Wait until every accepted event has been scored; ``False`` on timeout."""
        deadline = time.monotonic() + timeout_s
        while self._depth > 0:
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(_IDLE_POLL_S)
        return True
