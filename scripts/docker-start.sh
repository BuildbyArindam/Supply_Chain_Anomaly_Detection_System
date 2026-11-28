#!/usr/bin/env bash
# Start the full stack with Docker Compose and wait until every service is healthy.
#
#   ./scripts/docker-start.sh               # dev profile  (docker/.env.dev)
#   ./scripts/docker-start.sh prod          # prod profile (docker/.env.prod)
#   ./scripts/docker-start.sh --no-pull     # skip pulling images (offline / already pulled)
#   ./scripts/docker-start.sh --no-build    # reuse the images built earlier
#   ./scripts/docker-start.sh --timeout 600 # wait up to 10 minutes (default 300 s)
#
# Exit codes: 0 all services healthy, 1 a service failed or timed out, 2 bad usage or Docker
# is not available.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="docker/docker-compose.full.yml"
PROFILE="dev"
DO_PULL=1
DO_BUILD=1
TIMEOUT_S=300
# Long-running services to wait for. kafka-init is a one-shot job: compose already makes the other
# services wait for it, so it needs no check of its own.
SERVICES=(kafka kafka-ui simulator api dashboard prometheus)

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

log() { printf '[docker-start] %s\n' "$*"; }
fail() { printf '[docker-start] ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    dev | prod) PROFILE="$1" ;;
    --no-pull) DO_PULL=0 ;;
    --no-build) DO_BUILD=0 ;;
    --timeout)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || fail "--timeout needs a number of seconds" 2
      TIMEOUT_S="$2"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown argument: $1" 2
      ;;
  esac
  shift
done

cd "$PROJECT_ROOT"

# --- Preconditions -------------------------------------------------------------
command -v docker > /dev/null 2>&1 || fail "docker is not installed or not on PATH" 2
docker info > /dev/null 2>&1 || fail "the Docker daemon is not running (start Docker Desktop or the docker service)" 2
docker compose version > /dev/null 2>&1 || fail "the 'docker compose' plugin (v2) is required" 2
[[ -f "docker/.env.${PROFILE}" ]] || fail "missing docker/.env.${PROFILE}" 2

# The dev Kafka stack (`make run-kafka`) publishes the same ports 9092 and 8080.
if docker ps --format '{{.Names}}' | grep -qx 'scad-kafka'; then
  fail "the dev Kafka stack is running and would clash on ports 9092/8080; stop it first: make stop-kafka" 2
fi

export SCAD_ENV_FILE=".env.${PROFILE}"
compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

# --- Pull, build, start --------------------------------------------------------
if ((DO_PULL)); then
  log "pulling images (kafka, kafka-ui, prometheus)"
  # --ignore-buildable skips the three images this repo builds itself.
  compose pull --ignore-buildable || log "pull failed, continuing with local images"
fi

log "starting the stack with the '${PROFILE}' profile"
if ((DO_BUILD)); then
  compose up -d --build
else
  compose up -d
fi

# --- Wait for health -----------------------------------------------------------
service_state() {
  # healthy | starting | unhealthy | running (no healthcheck) | exited | missing
  local cid
  cid="$(compose ps -q "$1" 2> /dev/null | head -n 1)"
  [[ -n "$cid" ]] || { echo "missing"; return; }
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid"
}

log "waiting up to ${TIMEOUT_S}s for: ${SERVICES[*]}"
deadline=$((SECONDS + TIMEOUT_S))
pending=("${SERVICES[@]}")
while ((${#pending[@]} > 0)); do
  still=()
  for svc in "${pending[@]}"; do
    state="$(service_state "$svc")"
    case "$state" in
      healthy | running) log "  ${svc}: ${state}" ;;
      # starting / unhealthy / exited / missing: keep waiting (the simulator restarts on failure,
      # so a short "exited" while Kafka warms up is normal).
      *) still+=("$svc") ;;
    esac
  done
  pending=(${still[@]+"${still[@]}"})
  ((${#pending[@]} == 0)) && break
  if ((SECONDS >= deadline)); then
    printf '[docker-start] ERROR: not healthy after %ss: %s\n' "$TIMEOUT_S" "${pending[*]}" >&2
    for svc in "${pending[@]}"; do
      printf '\n--- last logs of %s ---\n' "$svc" >&2
      compose logs --no-color --tail 20 "$svc" >&2 || true
    done
    printf '\nInspect with: docker compose -f %s ps\n' "$COMPOSE_FILE" >&2
    exit 1
  fi
  sleep 5
done

# --- Done ----------------------------------------------------------------------
cat << EOF

All services are up (profile: ${PROFILE}).

  Dashboard     http://localhost:8050
  API docs      http://localhost:8000/docs
  API health    http://localhost:8000/health
  Metrics       http://localhost:8000/metrics
  Prometheus    http://localhost:9090
  Kafka UI      http://localhost:8080
  Kafka (host)  localhost:9092

  Logs:  docker compose -f ${COMPOSE_FILE} logs -f api
  Stop:  docker compose -f ${COMPOSE_FILE} down     (add -v to also delete the volumes)
EOF
