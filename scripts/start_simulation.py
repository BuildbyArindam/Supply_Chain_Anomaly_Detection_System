"""Stream the simulated supply chain into Kafka (Day 4).

Wires :class:`~src.data_simulation.orchestrator.SimulationOrchestrator` to
:class:`~src.kafka.producer.KafkaProducerManager`: every warehouse, GPS and inventory
event (and order events with ``--include-orders``) is validated, published to its topic
and, if Kafka cannot take it, dead-lettered.

Examples::

    make run-kafka                                   # start the broker first
    python scripts/start_simulation.py --check       # connectivity check only
    python scripts/start_simulation.py               # real time, until Ctrl+C
    python scripts/start_simulation.py --speed 20 --duration 3600
    python scripts/start_simulation.py --no-throttle --max-events 100000
    python -m src.main simulate                      # same, using configs/config.yaml only

Ctrl+C (or SIGTERM) stops the stream, flushes everything still queued and prints a
summary. Exit codes: 0 success, 1 Kafka unhealthy or events lost, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/start_simulation.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_simulation.orchestrator import SimulationOrchestrator  # noqa: E402
from src.kafka.producer import KafkaProducerManager  # noqa: E402
from src.utils.config import AppConfig, load_config  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
STATS_EVERY_S = 10.0
FLUSH_TIMEOUT_S = 30.0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="start_simulation.py", description="Stream simulated events into Kafka."
    )
    parser.add_argument("--config", type=Path, default=None, help="path to a config YAML file")
    parser.add_argument(
        "--bootstrap-servers", default=None, help="override kafka.bootstrap_servers"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="simulated seconds per wall second (1 = real time)",
    )
    parser.add_argument("--duration", type=float, default=None, help="simulated seconds to stream")
    parser.add_argument("--max-events", type=int, default=None, help="stop after this many events")
    parser.add_argument(
        "--no-throttle", action="store_true", help="generate as fast as possible (ignore --speed)"
    )
    parser.add_argument("--include-orders", action="store_true", help="also stream order events")
    parser.add_argument("--batch-size", type=int, default=None, help="producer batch size (bytes)")
    parser.add_argument("--linger-ms", type=int, default=None, help="producer linger time (ms)")
    parser.add_argument("--check", action="store_true", help="check Kafka connectivity and exit")
    return parser


def run_stream(
    orchestrator: SimulationOrchestrator,
    producer: KafkaProducerManager,
    *,
    duration_s: float | None = None,
    max_events: int | None = None,
    throttle: bool = True,
    stop: threading.Event | None = None,
    stats_every_s: float = STATS_EVERY_S,
) -> int:
    """Send the orchestrator's events through ``producer`` until a stop condition is met.

    Args:
        orchestrator: Source of events.
        producer: Destination.
        duration_s: Simulated seconds to cover (``None`` = unbounded).
        max_events: Stop after this many events (``None`` = unbounded).
        throttle: Pace the stream against the wall clock.
        stop: Set this event to stop early (signal handlers do).
        stats_every_s: Wall-clock seconds between progress log lines.

    Returns:
        The number of events handed to the producer.
    """
    log = get_logger(__name__)
    stop = stop or threading.Event()
    events = orchestrator.stream(duration_s, throttle=throttle)
    if max_events is not None:
        events = islice(events, max_events)

    sent = 0
    next_report = time.monotonic() + stats_every_s
    try:
        for event in events:
            if stop.is_set():
                break
            producer.send_event(event)
            sent += 1
            now = time.monotonic()
            if now >= next_report:
                next_report = now + stats_every_s
                log.info("simulation_progress", sent=sent, in_flight=producer.in_flight)
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()  # lets the orchestrator log its own "stopped" summary
    return sent


def run(cfg: AppConfig, args: argparse.Namespace | None = None) -> int:
    """Run the simulation against Kafka as configured. Returns a process exit code."""
    args = args or build_parser().parse_args([])
    log = get_logger(__name__)

    kafka_cfg = cfg.kafka
    if args.bootstrap_servers:
        kafka_cfg = kafka_cfg.model_copy(update={"bootstrap_servers": args.bootstrap_servers})

    producer = KafkaProducerManager.from_config(
        kafka_cfg, batch_size=args.batch_size, linger_ms=args.linger_ms
    )
    stop = threading.Event()
    try:
        health = producer.check_health()
        log.info("kafka_health", **health.to_dict())
        if not health.healthy:
            print(f"Kafka is not ready: {health.error}", file=sys.stderr)
            return EXIT_FAILURE
        if args.check:
            print(f"Kafka OK: {health.brokers} broker(s), {health.latency_ms:.0f} ms")
            return EXIT_OK

        orchestrator = SimulationOrchestrator.from_config(
            cfg.simulation,
            include_orders=args.include_orders or None,
            speed_multiplier=args.speed,
        )

        def _request_stop(signum: int, _frame: object) -> None:
            log.info("simulation_stop_requested", signal=signal.Signals(signum).name)
            stop.set()

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

        sent = run_stream(
            orchestrator,
            producer,
            duration_s=args.duration,
            max_events=args.max_events,
            throttle=not args.no_throttle,
            stop=stop,
        )
    finally:
        unflushed = producer.close(FLUSH_TIMEOUT_S)

    log.info("simulation_finished", events_sent=sent, unflushed=unflushed)
    print(f"Streamed {sent} events; {unflushed} unflushed.")
    return EXIT_FAILURE if unflushed else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.speed is not None and args.speed <= 0:
        print("--speed must be positive", file=sys.stderr)
        return EXIT_USAGE
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    configure_logging(
        level=cfg.logging.level, fmt=cfg.logging.format, service_name=cfg.logging.service_name
    )
    return run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
