"""Run the real-time inference pipeline and the REST API in one process (Day 15).

Starts, on a single asyncio event loop:

* the :class:`~src.kafka.streaming_inference.StreamingInferenceEngine`, consuming
  ``warehouse-sensors``, ``vehicle-gps`` and ``inventory-events`` and publishing alerts to
  ``anomaly-alerts``;
* the FastAPI service (:func:`src.api.main.create_app`), sharing the engine's
  :class:`~src.api.service.InferenceService`, so alerts raised from the stream show up in
  ``GET /api/v1/anomalies`` immediately.

``GET /api/v1/pipeline/stats`` (added here) reports the engine counters and latency percentiles;
``/metrics`` carries ``scad_detection_latency_seconds`` and friends.

Examples::

    make run-kafka                                    # broker first
    make run-simulator                                # in another terminal: events
    python scripts/run_streaming_pipeline.py          # engine + API on config api.host:port
    python scripts/run_streaming_pipeline.py --workers 8 --port 8010
    python scripts/run_streaming_pipeline.py --no-api # engine only

Ctrl+C (or SIGTERM) shuts down in order: stop consuming, score everything already queued, flush
the alert producer, stop the HTTP server (which persists alerts and feedback). Exit codes: 0 clean,
1 the queues did not drain in time or the consumer failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/run_streaming_pipeline.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import create_app  # noqa: E402
from src.api.service import InferenceService  # noqa: E402
from src.kafka.pipeline import DeadLetterPublisher  # noqa: E402
from src.kafka.producer import KafkaProducerManager  # noqa: E402
from src.kafka.streaming_inference import (  # noqa: E402
    KafkaAlertPublisher,
    StreamingInferenceEngine,
)
from src.utils.config import AppConfig, load_config  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
DRAIN_TIMEOUT_S = 30.0
FLUSH_TIMEOUT_S = 15.0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="run_streaming_pipeline.py",
        description="Run the streaming inference engine together with the FastAPI service.",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to a config YAML file")
    parser.add_argument(
        "--bootstrap-servers", default=None, help="override kafka.bootstrap_servers"
    )
    parser.add_argument("--host", default=None, help="API host (default: config api.host)")
    parser.add_argument(
        "--port", type=int, default=None, help="API port (default: config api.port)"
    )
    parser.add_argument("--workers", type=int, default=None, help="inference workers")
    parser.add_argument(
        "--backpressure-depth", type=int, default=None, help="queue depth that starts sampling"
    )
    parser.add_argument("--no-api", action="store_true", help="run the engine without the API")
    parser.add_argument(
        "--drain-timeout", type=float, default=DRAIN_TIMEOUT_S, help="seconds to drain on shutdown"
    )
    return parser


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown: asyncio.Event) -> None:
    """Set ``shutdown`` on SIGINT / SIGTERM (falls back to ``signal.signal`` on Windows)."""
    log = get_logger(__name__)

    def _request(name: str) -> None:
        log.info("shutdown_signal_received", signal=name)
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request, sig.name)
        except NotImplementedError:  # pragma: no cover - Windows event loops
            signal.signal(sig, lambda _s, _f, n=sig.name: loop.call_soon_threadsafe(_request, n))


def _add_stats_route(app: Any, engine: StreamingInferenceEngine) -> None:
    """Expose the engine counters and latency percentiles next to the API routes."""

    @app.get("/api/v1/pipeline/stats", tags=["ops"], summary="Streaming pipeline statistics")
    def pipeline_stats() -> dict[str, Any]:
        """Engine counters, queue depth, backpressure state and latency percentiles."""
        consumer = engine.consumer_group
        return {
            "engine": engine.stats().to_dict(),
            "latency": engine.latency.snapshot().to_dict(),
            "latency_target_ms": engine.latency.target_p99_s * 1000.0,
            "consumer_lag": consumer.total_lag() if consumer is not None else None,
        }


async def run(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Start everything, run until a shutdown signal, then shut down in order."""
    log = get_logger(__name__)
    kafka_cfg = cfg.kafka
    if args.bootstrap_servers:
        kafka_cfg = kafka_cfg.model_copy(update={"bootstrap_servers": args.bootstrap_servers})
        cfg = cfg.model_copy(update={"kafka": kafka_cfg})

    service = InferenceService(cfg)
    service.start()  # loads models (statistical fallback if the registry is empty)
    # Long-lived startup state (config, model weights, registry) never needs scanning again: move
    # it out of the cyclic GC's reach so full collections stay short instead of stalling the
    # stream (a full collection over a large heap showed up as ~100 ms latency spikes).
    gc.collect()
    gc.freeze()

    producer = KafkaProducerManager.from_config(kafka_cfg)
    overrides: dict[str, Any] = {}
    if args.workers is not None:
        overrides["num_workers"] = args.workers
    if args.backpressure_depth is not None:
        overrides["backpressure_depth"] = args.backpressure_depth
    engine = StreamingInferenceEngine.from_config(
        service, publisher=KafkaAlertPublisher(producer), **overrides
    )

    from confluent_kafka import Producer

    dlq_producer = Producer(
        {"bootstrap.servers": kafka_cfg.bootstrap_servers, "client.id": "scad-inference-dlq"}
    )
    dead_letter = DeadLetterPublisher(dlq_producer, kafka_cfg.topics.dlq_events.name)

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    _install_signal_handlers(loop, shutdown)

    server: Any = None
    server_task: asyncio.Task[None] | None = None
    exit_code = EXIT_OK
    try:
        await engine.start()
        group = engine.start_consuming(dead_letter=dead_letter)

        if not args.no_api:
            import uvicorn

            app = create_app(
                cfg, service=service, kafka_check=lambda: producer.check_health(2.0).healthy
            )
            _add_stats_route(app, engine)
            uv_config = uvicorn.Config(
                app,
                host=args.host or cfg.api.host,
                port=args.port or cfg.api.port,
                log_config=None,
                lifespan="on",
            )
            server = uvicorn.Server(uv_config)
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            server_task = asyncio.create_task(server.serve(), name="scad-api")

        log.info(
            "streaming_pipeline_running",
            api=not args.no_api,
            workers=engine.num_workers,
        )
        waiters: set[asyncio.Task[Any]] = {asyncio.create_task(shutdown.wait())}
        if server_task is not None:
            waiters.add(server_task)
        while not shutdown.is_set():
            done, _ = await asyncio.wait(waiters, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            if done:
                break
            if group.failed:
                log.error("consumer_group_failed")
                exit_code = EXIT_FAILURE
                break
        for waiter in waiters:
            waiter.cancel()
    finally:
        log.info("streaming_pipeline_stopping")
        drained = await engine.stop(args.drain_timeout)
        if not drained:
            exit_code = EXIT_FAILURE
        remaining = await asyncio.get_running_loop().run_in_executor(
            None, producer.close, FLUSH_TIMEOUT_S
        )
        if remaining:
            log.error("alerts_not_flushed", remaining=remaining)
            exit_code = EXIT_FAILURE
        await asyncio.get_running_loop().run_in_executor(None, dead_letter.flush, FLUSH_TIMEOUT_S)
        if server is not None and server_task is not None:
            server.should_exit = True
            await server_task  # the API lifespan persists alerts and feedback on the way out
        log.info("streaming_pipeline_stopped", **engine.stats().to_dict())
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return EXIT_USAGE if exit_.code else EXIT_OK
    if args.workers is not None and args.workers < 1:
        parser.print_usage(sys.stderr)
        print("error: --workers must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, cfg.logging.format, cfg.logging.service_name)
    try:
        return asyncio.run(run(cfg, args))
    except KeyboardInterrupt:  # pragma: no cover - the signal handler normally pre-empts this
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
