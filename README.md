# Real-Time Supply Chain Anomaly Detection

[![CI](https://github.com/BuildbyArindam/Supply_Chain_Anomaly_Detection_System/actions/workflows/ci.yml/badge.svg)](https://github.com/BuildbyArindam/Supply_Chain_Anomaly_Detection_System/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BuildbyArindam/Supply_Chain_Anomaly_Detection_System/branch/main/graph/badge.svg)](https://codecov.io/gh/BuildbyArindam/Supply_Chain_Anomaly_Detection_System)

Streams warehouse sensor, vehicle GPS and inventory events through Kafka and flags anomalies
with a statistical + Isolation Forest + LSTM ensemble, served by FastAPI and a Dash dashboard.

Status: **Day 5 of 30 - streaming consumer and feature cache.** Models, API and dashboard are added in later days.

## Quick start

```bash
make setup        # .venv + pinned dependencies + .env (needs Python 3.11+)
make test         # all tests
make test-unit    # what CI runs: unit tests + 85% coverage gate
make test-integration  # real Kafka via testcontainers (needs Docker)
make load-test    # Locust: 0->100 users, asserts p99/error-rate/throughput SLOs
make lint         # ruff + black --check
python -m src.main info
```

Run `make help` for every target. Kafka (`make run-kafka`) needs Docker.

## Layout

```
src/
  data_simulation/   synthetic warehouse / GPS / inventory generators
  kafka/             producers, consumers, stream processing
  models/            feature engineering, detectors, ensemble
  api/               FastAPI service (routers, middleware)
  dashboard/         Dash app (pages, components)
  utils/             config.py (validated YAML + env), logger.py (structlog JSON)
  main.py            CLI entry point (python -m src.main COMMAND)
tests/               unit/, integration/, load/
configs/             config.yaml - single source of truth for all parameters
docker/              Dockerfile, docker-compose.kafka.yml (Kafka + topic init + Kafka UI)
notebooks/ scripts/ docs/
```

Dependencies flow inward: `api`/`dashboard`/`kafka` -> `models`/`data_simulation` -> `utils`.
`utils` imports nothing from the other packages. Packages must not import each other's
internals sideways (e.g. `dashboard` talks to `api` over HTTP, not by importing it).

Note: `src/kafka/` is a package name that also exists on PyPI as `kafka-python`. Always use
absolute imports (`from src.kafka import ...` for ours, `import kafka` for the library) and
run code from the project root, never with `src/` on `PYTHONPATH`.

## Streaming consumer (Day 5)

```bash
make run-kafka && make run-simulator      # produce events (Day 4)
make run-consumer                         # consume -> windowed features -> feature cache
curl localhost:8001/metrics | grep scad_  # Prometheus exporter of the consumer
```

`consumer.py` (worker pool, manual commits, lag) -> `stream_processor.py` (1/5/15-minute
sliding windows per entity: mean, std, min, max, rate of change, p95; event-time watermark of
30 s) -> `feature_cache.py` (`store_features` / `get_features` / `get_history`, Redis-shaped).
`pipeline.py` wires them; `tests/mock_kafka.py` is an in-memory broker for tests.
Delivery is at-least-once; duplicates are dropped by `message_id`. Key metrics:
`scad_messages_consumed_total`, `scad_processing_latency_seconds`, `scad_cache_hit_ratio`,
`scad_kafka_consumer_lag`.

## Real-time inference (Day 15)

```bash
make run-kafka && make run-simulator      # events in
make run-streaming                        # engine + API on one asyncio loop (Ctrl+C = graceful)
curl localhost:8000/api/v1/pipeline/stats # counters, queue depth, p50/p95/p99
make test-perf                            # 1000 ev/s for 30 s: p99 < 500 ms, 0 dropped
```

`src/kafka/streaming_inference.py`: `StreamingInferenceEngine` consumes `warehouse-sensors`,
`vehicle-gps` and `inventory-events` and scores each event through the same
`InferenceService.evaluate()` the REST API uses; alerts go to `anomaly-alerts` and into the
`AnomalyStore` (snapshotted every 5 minutes). Events are sharded by entity so each entity is scored
in order. Above 1000 queued events non-critical entities are sampled (every 5th; cold-storage
temperature never). `LatencyTracker` measures producer timestamp to alert published and logs a
WARNING when p99 > 500 ms (`scad_detection_latency_seconds`). The load test uses the statistical
detector only; trained LSTM/IF models are slower per event.

## Configuration

Parameters live in `configs/config.yaml` and are validated in `src/utils/config.py`
(unknown keys and out-of-range values fail at startup). A small allow-list of values can be
overridden from `.env` or the environment - see `.env.example`.

## Logging

`from src.utils.logger import get_logger` - JSON lines by default, `LOG_FORMAT=console` for
readable local output, `LOG_LEVEL` to change verbosity.

## Git workflow

- `main`: always releasable; protected, merged via pull request only.
- `develop`: integration branch; feature branches merge here.
- `feature/dayNN-short-name`, `fix/...`: short-lived branches cut from `develop`.
- Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
