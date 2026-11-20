# Real-Time Supply Chain Anomaly Detection

Streams warehouse sensor, vehicle GPS and inventory events through Kafka and flags anomalies
with a statistical + Isolation Forest + LSTM ensemble, served by FastAPI and a Dash dashboard.

Status: **Day 1 of 30 - project scaffold.** The simulator, models, API and dashboard are added in later days.

## Quick start

```bash
make setup        # .venv + pinned dependencies + .env (needs Python 3.11+)
make test         # unit tests
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
