# Running the system with Docker

The whole stack runs from one Compose file, `docker/docker-compose.full.yml`.

```bash
./scripts/docker-start.sh          # dev profile: DEBUG logs, 10x simulation speed
./scripts/docker-start.sh prod     # prod profile: INFO json logs, real-time simulation
# or: make docker-up PROFILE=dev
```

The script checks that Docker is running, pulls the public images, builds ours, starts everything,
waits until each service is healthy and prints the URLs. Requirements: Docker Engine with Compose v2.

> Stop the dev Kafka stack first (`make stop-kafka`): both stacks publish ports 9092 and 8080.

## Services

| Service | Image | Port | Role |
|---|---|---|---|
| `kafka` | `confluentinc/cp-kafka:7.6.0` | 9092 (host) / `kafka:29092` (containers) | Single-node KRaft broker. Healthcheck: `kafka-topics --list`. |
| `kafka-init` | same | - | One-shot job: creates the 6 topics, then exits 0. Auto-creation is off, so `kafka-ui`, `simulator` and `api` wait for it. |
| `kafka-ui` | `provectuslabs/kafka-ui:v0.7.2` | 8080 | Browse topics and messages. |
| `simulator` | built from `Dockerfile.simulator` | - | Streams warehouse, GPS and inventory events into Kafka. `restart: on-failure`. |
| `api` | built from `Dockerfile.api` | 8000 | `scripts/run_streaming_pipeline.py`: the Kafka inference engine **and** the FastAPI service in one process. Healthcheck: `/health`. |
| `dashboard` | built from `Dockerfile.dashboard` | 8050 | Dash UI. Reads from the API over HTTP; starts once the API is healthy. |
| `prometheus` | `prom/prometheus:latest` | 9090 | Scrapes `api:8000/metrics` every 15 s (`configs/prometheus.yml`). |

URLs: dashboard http://localhost:8050, API docs http://localhost:8000/docs, Prometheus
http://localhost:9090, Kafka UI http://localhost:8080.

## Images

All three are multi-stage (a builder stage creates a virtualenv; the runtime stage copies only that
and the source), run as the non-root user `appuser` (uid 1000) and are built from the project root.
Each installs only what it needs, from `docker/requirements-<service>.txt`:

| Image | Contents | Notes |
|---|---|---|
| `api` | torch (CPU wheel), scikit-learn, LightGBM, confluent-kafka, FastAPI | The largest image. `libgomp1` is installed for LightGBM. |
| `dashboard` | Dash, plotly, dash-ag-grid, httpx | No torch or Kafka client. |
| `simulator` | confluent-kafka, pydantic | Smallest. |

The base image is `python:3.12-slim`: your pinned `numpy==2.5.5` has no Python 3.11 build. The pins
in `docker/requirements-*.txt` must equal those in `requirements.txt`;
`tests/unit/test_docker_files.py` fails when they drift.

## Volumes

| Volume | Mounted at (api) | Holds |
|---|---|---|
| `model_registry` | `/app/models` | Versioned models (`{model_type}/{timestamp}/`) |
| `feedback_data` | `/app/data/feedback` | `feedback_store.json` |
| `anomaly_snapshots` | `/app/data/snapshots` | Periodic alert snapshots |
| `kafka_data` | `/var/lib/kafka/data` | Kafka logs |

The paths are set by `MODEL_REGISTRY_DIR`, `FEEDBACK_STORE_PATH` and `ANOMALY_SNAPSHOT_DIR` in the
env files.

**Models:** the `model_registry` volume starts empty, so the API runs on the statistical fallback
detector (`/health` reports `degraded`) until you put trained models in it, e.g.:

```bash
docker run --rm -v scad_model_registry:/dst -v "$PWD/model_registry":/src:ro alpine cp -a /src/. /dst/
curl -X POST http://localhost:8000/api/v1/models/reload
```

## Configuration

`SCAD_ENV_FILE` (set for you by `docker-start.sh`) picks `docker/.env.dev` or `docker/.env.prod`.

| | dev | prod |
|---|---|---|
| `LOG_LEVEL` / `LOG_FORMAT` | DEBUG / console | INFO / json |
| `SIMULATION_SPEED` | 10 | 1 |

Both files are committed, so they hold no secrets. To protect the model reload/promote endpoints,
export a key in your shell; it is passed to `api` and `dashboard`:

```bash
API_ADMIN_KEY=$(openssl rand -hex 24) ./scripts/docker-start.sh prod
```

`PUBLIC_API_URL` is the API address as seen by the **browser** (live alert stream). On a remote
host change it from `localhost` and add the dashboard's origin to `api.cors_origins` in
`configs/config.yaml`.

## Everyday commands

```bash
docker compose -f docker/docker-compose.full.yml ps
docker compose -f docker/docker-compose.full.yml logs -f api        # make docker-logs
docker compose -f docker/docker-compose.full.yml restart dashboard
docker compose -f docker/docker-compose.full.yml up -d --build api  # rebuild one service
docker compose -f docker/docker-compose.full.yml down              # keep volumes (make docker-down)
docker compose -f docker/docker-compose.full.yml down -v           # also delete volumes
```

(Prefix with `SCAD_ENV_FILE=.env.prod` when you started the prod profile.)

## Scaling

- **simulator**: `up -d --scale simulator=3` works (no published port), but every replica emits the
  same seeded entities, so the stream contains duplicates. Raise the speed instead
  (`SIMULATION_SPEED`), or give replicas different seeds.
- **api**: the published port 8000 allows one replica, and each replica keeps its own in-memory
  alert and feedback stores, so a dashboard sees only the replica it talks to. Scale it
  only behind a load balancer after moving those stores to shared storage. To spread Kafka load,
  the topics have 3-6 partitions, so up to that many engine workers can share a consumer group.
- **dashboard**: stateless; scale it behind a proxy (it runs the Dash server, which is
  fine for a demo; a WSGI server such as gunicorn would need a small module exposing the app).
- **kafka**: single node (replication factor 1). A real cluster is out of scope here.

## Troubleshooting

- **`api` stays `starting`**: check `logs api`. Torch import and model loading take 20-40 s on first start.
- **Port already in use (9092 / 8080)**: the dev Kafka stack is running; `make stop-kafka`.
- **Dashboard shows "Offline"**: the API is unreachable from the dashboard container; check `API_URL`.
- **Live alerts do not stream, but tables work**: the browser cannot reach `PUBLIC_API_URL` or CORS blocks it.
- **Prometheus target DOWN**: open http://localhost:9090/targets. The dashboard is deliberately not scraped: it has no `/metrics` endpoint.
