"""Guards for the Day 22 Docker setup: compose file, Dockerfiles, env profiles, pinned requirements.

These tests read files only (no Docker daemon needed), so they run in CI like any unit test.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from src.utils.config import ENV_OVERRIDES, PROJECT_ROOT, load_config

DOCKER_DIR = PROJECT_ROOT / "docker"
FULL = DOCKER_DIR / "docker-compose.full.yml"
KAFKA_DEV = DOCKER_DIR / "docker-compose.kafka.yml"
PROM = PROJECT_ROOT / "configs" / "prometheus.yml"

IMAGES = {
    "api": ("Dockerfile.api", "requirements-api.txt"),
    "dashboard": ("Dockerfile.dashboard", "requirements-dashboard.txt"),
    "simulator": ("Dockerfile.simulator", "requirements-simulator.txt"),
}


def _compose() -> dict:
    return yaml.safe_load(FULL.read_text(encoding="utf-8"))


def _pins(path: Path) -> dict[str, str]:
    """Map normalised package name -> pinned version for every ``name==version`` line."""
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9._-]+)==([^\s;]+)", line)
        if match:
            pins[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return pins


def _env_file(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in (DOCKER_DIR / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key] = value
    return values


# --------------------------------------------------------------------------- #
# docker-compose.full.yml
# --------------------------------------------------------------------------- #
def test_services_ports_and_volumes():
    compose = _compose()
    services = compose["services"]
    assert set(services) == {
        "kafka",
        "kafka-init",
        "kafka-ui",
        "simulator",
        "api",
        "dashboard",
        "prometheus",
    }
    assert "9092:9092" in services["kafka"]["ports"]
    assert "8080:8080" in services["kafka-ui"]["ports"]
    assert "8000:8000" in services["api"]["ports"]
    assert "8050:8050" in services["dashboard"]["ports"]
    assert "9090:9090" in services["prometheus"]["ports"]
    assert {"model_registry", "feedback_data", "anomaly_snapshots"} <= set(compose["volumes"])
    assert "model_registry:/app/models" in services["api"]["volumes"]


def test_python_services_build_from_the_project_root_with_their_dockerfile():
    services = _compose()["services"]
    for name, (dockerfile, _) in IMAGES.items():
        build = services[name]["build"]
        assert build["context"] == ".."
        assert build["dockerfile"] == f"docker/{dockerfile}"
        assert (DOCKER_DIR / dockerfile).is_file()


def test_everything_that_uses_kafka_waits_for_healthy_broker_and_topics():
    services = _compose()["services"]
    for name in ("simulator", "api", "kafka-ui"):
        depends = services[name]["depends_on"]
        assert depends["kafka"]["condition"] == "service_healthy", name
        assert depends["kafka-init"]["condition"] == "service_completed_successfully", name
    assert services["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["simulator"]["restart"] == "on-failure"


def test_healthchecks_probe_the_documented_endpoints():
    services = _compose()["services"]
    assert services["api"]["healthcheck"]["test"][-1] == "http://127.0.0.1:8000/health"
    assert services["dashboard"]["healthcheck"]["test"][-1].startswith("http://127.0.0.1:8050")
    assert "kafka-topics" in services["kafka"]["healthcheck"]["test"][-1]


def test_kafka_matches_the_dev_compose_file():
    full = _compose()["services"]["kafka"]
    dev = yaml.safe_load(KAFKA_DEV.read_text(encoding="utf-8"))["services"]["kafka"]
    assert full["image"] == dev["image"]
    assert full["environment"] == dev["environment"]
    assert full["ports"] == dev["ports"]


def test_init_job_creates_exactly_the_configured_topics():
    script = "\n".join(_compose()["services"]["kafka-init"]["command"])
    created = dict(re.findall(r"^\s*create_topic\s+(\S+)\s+(\d+)", script, flags=re.M))
    configured = {t.name: str(t.partitions) for t in load_config(environ={}).kafka.topics.as_list()}
    assert created == configured


def test_services_use_the_compose_network_addresses():
    for profile in (".env.dev", ".env.prod"):
        env = _env_file(profile)
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:29092"
        assert env["API_URL"] == "http://api:8000"
        assert env["API_HOST"] == env["DASHBOARD_HOST"] == "0.0.0.0"


# --------------------------------------------------------------------------- #
# Env profiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", [".env.dev", ".env.prod"])
def test_env_profile_loads_into_a_valid_config(profile):
    env = _env_file(profile)
    cfg = load_config(environ=env)
    assert cfg.kafka.bootstrap_servers == "kafka:29092"
    assert cfg.paths.model_registry == "/app/models"
    assert cfg.paths.feedback_store.startswith("/app/data/feedback/")
    assert cfg.paths.anomaly_snapshots.startswith("/app/data/snapshots")


def test_env_profiles_differ_where_the_plan_says_they_should():
    dev, prod = _env_file(".env.dev"), _env_file(".env.prod")
    assert (dev["LOG_LEVEL"], dev["SIMULATION_SPEED"]) == ("DEBUG", "10")
    assert (prod["LOG_LEVEL"], prod["SIMULATION_SPEED"]) == ("INFO", "1")


@pytest.mark.parametrize("profile", [".env.dev", ".env.prod"])
def test_env_profiles_hold_no_secrets_and_only_known_variables(profile):
    known = set(ENV_OVERRIDES) | {"PUBLIC_API_URL", "API_ADMIN_KEY"}
    env = _env_file(profile)
    assert set(env) <= known, set(env) - known
    assert "API_ADMIN_KEY" not in env  # committed files must not carry the key


# --------------------------------------------------------------------------- #
# Dockerfiles and requirements
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(IMAGES))
def test_dockerfiles_are_multi_stage_and_run_as_non_root(name):
    text = (DOCKER_DIR / IMAGES[name][0]).read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.12-slim AS builder", text, re.M)
    assert re.search(r"^FROM python:3\.12-slim AS runtime", text, re.M)
    assert "COPY --from=builder /opt/venv /opt/venv" in text
    users = re.findall(r"^USER (\S+)", text, re.M)
    assert users and users[-1] == "appuser"


def test_api_and_dashboard_images_declare_a_healthcheck_and_port():
    api = (DOCKER_DIR / "Dockerfile.api").read_text(encoding="utf-8")
    dash = (DOCKER_DIR / "Dockerfile.dashboard").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in api and "/health" in api and "EXPOSE 8000" in api
    assert "HEALTHCHECK" in dash and "EXPOSE 8050" in dash


def test_api_image_uses_cpu_torch():
    text = (DOCKER_DIR / "Dockerfile.api").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in text


def test_every_image_copies_the_scripts_its_command_runs():
    expected = {
        "Dockerfile.api": "run_streaming_pipeline.py",
        "Dockerfile.simulator": "start_simulation.py",
    }
    for dockerfile, script in expected.items():
        text = (DOCKER_DIR / dockerfile).read_text(encoding="utf-8")
        assert f"scripts/{script}" in text
        assert (PROJECT_ROOT / "scripts" / script).is_file()
    for dockerfile in ("Dockerfile.api", "Dockerfile.dashboard"):
        assert "scripts/healthcheck.py" in (DOCKER_DIR / dockerfile).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", list(IMAGES))
def test_image_requirements_use_the_pins_of_requirements_txt(name):
    root = _pins(PROJECT_ROOT / "requirements.txt")
    image = _pins(DOCKER_DIR / IMAGES[name][1])
    assert image, f"no pinned packages found for {name}"
    mismatched = {pkg: (ver, root.get(pkg)) for pkg, ver in image.items() if root.get(pkg) != ver}
    assert not mismatched, f"pin differs from requirements.txt (image, root): {mismatched}"


def test_api_image_has_what_the_inference_path_imports():
    pins = _pins(DOCKER_DIR / "requirements-api.txt")
    # torch/sklearn: the models; lightgbm: the ensemble meta-learner; confluent-kafka: the engine.
    for pkg in ("torch", "scikit-learn", "lightgbm", "confluent-kafka", "fastapi", "uvicorn"):
        assert pkg in pins, pkg


# --------------------------------------------------------------------------- #
# Prometheus and the health probe
# --------------------------------------------------------------------------- #
def test_prometheus_scrapes_the_api_every_15_seconds():
    config = yaml.safe_load(PROM.read_text(encoding="utf-8"))
    assert config["global"]["scrape_interval"] == "15s"
    assert config["global"]["evaluation_interval"] == "15s"
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["scad-api"]["static_configs"][0]["targets"] == ["api:8000"]
    assert jobs["scad-api"]["metrics_path"] == "/metrics"


def _load_healthcheck():
    path = PROJECT_ROOT / "scripts" / "healthcheck.py"
    spec = importlib.util.spec_from_file_location("healthcheck", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_healthcheck_rejects_non_http_urls_and_unreachable_hosts():
    healthcheck = _load_healthcheck()
    healthy, detail = healthcheck.check("file:///etc/passwd")
    assert not healthy and "not an http" in detail
    healthy, _ = healthcheck.check("http://127.0.0.1:9/health", timeout=0.5)  # discard port
    assert not healthy
    assert healthcheck.main(["http://127.0.0.1:9/health", "--timeout", "0.5"]) == 1


def test_healthcheck_reports_http_status(tmp_path):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            self.send_response(200 if self.path == "/health" else 503)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        healthcheck = _load_healthcheck()
        assert healthcheck.main([f"{base}/health"]) == 0
        assert healthcheck.main([f"{base}/broken"]) == 1
    finally:
        server.shutdown()
        server.server_close()
