"""Guards that docker-compose.kafka.yml and configs/config.yaml describe the same topics."""

import re

import yaml

from src.utils.config import PROJECT_ROOT, load_config

COMPOSE = PROJECT_ROOT / "docker" / "docker-compose.kafka.yml"


def load_compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_services_and_images():
    services = load_compose()["services"]
    assert set(services) == {"kafka", "kafka-init", "kafka-ui"}
    assert services["kafka"]["image"] == "confluentinc/cp-kafka:7.6.0"
    assert services["kafka-ui"]["image"].startswith("provectuslabs/kafka-ui")
    assert "8080:8080" in services["kafka-ui"]["ports"]


def test_kraft_mode_without_zookeeper():
    text = COMPOSE.read_text(encoding="utf-8").lower()
    env = load_compose()["services"]["kafka"]["environment"]
    assert env["KAFKA_PROCESS_ROLES"] == "broker,controller"
    assert "zookeeper" not in text.replace("no zookeeper", "")
    assert env["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"


def test_broker_retention_and_message_size_defaults():
    env = load_compose()["services"]["kafka"]["environment"]
    assert int(env["KAFKA_LOG_RETENTION_MS"]) == 7 * 24 * 3600 * 1000
    assert int(env["KAFKA_MESSAGE_MAX_BYTES"]) == 1_048_576


def test_init_job_creates_exactly_the_configured_topics():
    script = "\n".join(load_compose()["services"]["kafka-init"]["command"])
    created = dict(re.findall(r"^\s*create_topic\s+(\S+)\s+(\d+)", script, flags=re.M))
    configured = {t.name: str(t.partitions) for t in load_config().kafka.topics.as_list()}
    assert created == configured
    assert created["warehouse-sensors"] == "6"
    assert created["dlq-events"] == "1"


def test_init_job_applies_retention_and_size_limits():
    script = "\n".join(load_compose()["services"]["kafka-init"]["command"])
    assert "RETENTION_MS=604800000" in script
    assert "MAX_MESSAGE_BYTES=1048576" in script
    assert "retention.ms=" in script
    assert "max.message.bytes=" in script
    cfg = load_config().kafka.topics
    assert all(t.retention_ms == 604_800_000 for t in cfg.as_list())
    assert all(t.max_message_bytes == 1_048_576 for t in cfg.as_list())
