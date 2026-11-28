from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import DEFAULT_CONFIG_PATH, load_config


def _load(**kwargs):
    kwargs.setdefault("environ", {})
    return load_config(**kwargs)


def test_default_config_loads():
    cfg = _load()
    assert cfg.kafka.bootstrap_servers == "localhost:9092"
    assert cfg.api.port == 8000
    assert cfg.dashboard.port == 8050
    assert cfg.kafka.topics.warehouse_sensors.name == "warehouse-sensors"
    assert cfg.kafka.topics.dlq_events.partitions == 1
    assert len(cfg.kafka.topics.as_list()) == 6


def test_yaml_anchors_inherit_topic_defaults():
    cfg = _load()
    topic = cfg.kafka.topics.vehicle_gps
    assert topic.partitions == 4
    assert topic.retention_ms == 604_800_000


def test_scientific_notation_parses_as_float():
    cfg = _load()
    assert cfg.models.lstm_autoencoder.learning_rate == pytest.approx(1e-3)
    assert cfg.retraining.finetune_learning_rate == pytest.approx(1e-5)


def test_alert_thresholds_ordered():
    t = _load().alerts.severity_thresholds
    assert t.warning < t.critical < t.emergency


def test_env_overrides_apply_with_type_coercion():
    cfg = _load(
        environ={
            "KAFKA_BOOTSTRAP_SERVERS": "broker:29092",
            "API_PORT": "9001",
            "LOG_LEVEL": "debug",
            "APP_ENV": "PROD",
            "SIMULATION_SPEED": "10",
        }
    )
    assert cfg.kafka.bootstrap_servers == "broker:29092"
    assert cfg.api.port == 9001
    assert cfg.logging.level == "DEBUG"
    assert cfg.project.environment == "prod"
    assert cfg.simulation.speed_multiplier == 10.0


def test_storage_path_env_overrides():
    cfg = _load(
        environ={
            "MODEL_REGISTRY_DIR": "/app/models",
            "FEEDBACK_STORE_PATH": "/app/data/feedback/feedback_store.json",
            "ANOMALY_SNAPSHOT_DIR": "/app/data/snapshots",
        }
    )
    assert cfg.paths.model_registry == "/app/models"
    assert cfg.paths.feedback_store == "/app/data/feedback/feedback_store.json"
    assert cfg.paths.anomaly_snapshots == "/app/data/snapshots"


def test_empty_env_values_are_ignored():
    cfg = _load(environ={"API_PORT": "", "KAFKA_BOOTSTRAP_SERVERS": "  "})
    assert cfg.api.port == 8000
    assert cfg.kafka.bootstrap_servers == "localhost:9092"


def _write(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_unknown_key_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["api"].update(prot=1))
    with pytest.raises(ValidationError):
        _load(path=path)


def test_ensemble_weights_must_sum_to_one(tmp_path):
    path = _write(tmp_path, lambda d: d["models"]["ensemble"].update(statistical_weight=0.9))
    with pytest.raises(ValidationError, match="sum to 1.0"):
        _load(path=path)


def test_severity_thresholds_must_increase(tmp_path):
    path = _write(
        tmp_path, lambda d: d["alerts"]["severity_thresholds"].update(warning=0.7, critical=0.6)
    )
    with pytest.raises(ValidationError):
        _load(path=path)


def test_invalid_port_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["api"].update(port=70000))
    with pytest.raises(ValidationError):
        _load(path=path)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load(path=tmp_path / "nope.yaml")
    bad = tmp_path / "list.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        _load(path=bad)


def test_empty_yaml_falls_back_to_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    cfg = _load(path=empty)
    assert cfg.api.port == 8000


def test_config_path_env_var(tmp_path):
    path = _write(tmp_path, lambda d: d["api"].update(port=1234))
    cfg = load_config(environ={"CONFIG_PATH": str(path)})
    assert cfg.api.port == 1234
