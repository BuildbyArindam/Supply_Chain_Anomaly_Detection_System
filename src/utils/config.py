"""Typed, validated project configuration.

All tunable parameters live in ``configs/config.yaml``. This module loads that
file, applies environment-variable overrides (including values from a local
``.env`` file) and validates everything with Pydantic. Unknown keys are
rejected so that typos fail fast instead of being silently ignored.

Typical usage::

    from src.utils.config import get_config

    cfg = get_config()
    print(cfg.kafka.bootstrap_servers, cfg.api.port)

Environment overrides are an explicit allow-list (:data:`ENV_OVERRIDES`) so the
set of "deployment knobs" stays small and documented in ``.env.example``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

#: Environment variable -> path of the config key it overrides.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "APP_ENV": ("project", "environment"),
    "LOG_LEVEL": ("logging", "level"),
    "LOG_FORMAT": ("logging", "format"),
    "SERVICE_NAME": ("logging", "service_name"),
    "KAFKA_BOOTSTRAP_SERVERS": ("kafka", "bootstrap_servers"),
    "KAFKA_CONSUMER_GROUP_ID": ("kafka", "consumer", "group_id"),
    "SIMULATION_SPEED": ("simulation", "speed_multiplier"),
    "SIMULATION_SEED": ("simulation", "seed"),
    "API_HOST": ("api", "host"),
    "API_PORT": ("api", "port"),
    "DASHBOARD_HOST": ("dashboard", "host"),
    "DASHBOARD_PORT": ("dashboard", "port"),
    "API_URL": ("dashboard", "api_url"),
    "PROMETHEUS_PORT": ("observability", "prometheus_port"),
    "MODEL_REGISTRY_DIR": ("paths", "model_registry"),
}


class _Section(BaseModel):
    """Base class for config sections: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Project and logging
# --------------------------------------------------------------------------- #
class ProjectConfig(_Section):
    """General project metadata."""

    name: str = "supply-chain-anomaly-detection"
    environment: Literal["dev", "test", "prod"] = "dev"

    @field_validator("environment", mode="before")
    @classmethod
    def _lower_environment(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value


class LoggingConfig(_Section):
    """Structured logging settings (see :mod:`src.utils.logger`)."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"
    service_name: str = Field(default="scad", min_length=1)

    @field_validator("level", mode="before")
    @classmethod
    def _upper_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @field_validator("format", mode="before")
    @classmethod
    def _lower_format(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value


# --------------------------------------------------------------------------- #
# Kafka
# --------------------------------------------------------------------------- #
class TopicConfig(_Section):
    """Definition of a single Kafka topic."""

    name: str = Field(min_length=1)
    partitions: int = Field(ge=1)
    replication_factor: int = Field(default=1, ge=1)
    retention_ms: int = Field(default=604_800_000, ge=-1)  # 7 days
    max_message_bytes: int = Field(default=1_048_576, ge=1024)  # 1 MB


def _topic(name: str, partitions: int) -> Any:
    """Return a Field whose default is a :class:`TopicConfig`."""
    return Field(default_factory=lambda: TopicConfig(name=name, partitions=partitions))


class TopicsConfig(_Section):
    """All Kafka topics used by the system."""

    warehouse_sensors: TopicConfig = _topic("warehouse-sensors", 6)
    vehicle_gps: TopicConfig = _topic("vehicle-gps", 4)
    inventory_events: TopicConfig = _topic("inventory-events", 4)
    anomaly_alerts: TopicConfig = _topic("anomaly-alerts", 3)
    dlq_events: TopicConfig = _topic("dlq-events", 1)
    model_health: TopicConfig = _topic("model-health", 1)

    def as_list(self) -> list[TopicConfig]:
        """Return every topic definition as a list."""
        return [
            self.warehouse_sensors,
            self.vehicle_gps,
            self.inventory_events,
            self.anomaly_alerts,
            self.dlq_events,
            self.model_health,
        ]


class KafkaProducerConfig(_Section):
    """Producer tuning parameters."""

    acks: str = "all"
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_ms: int = Field(default=100, ge=0)
    batch_size: int = Field(default=16_384, ge=0)
    linger_ms: int = Field(default=5, ge=0)
    compression_type: Literal["none", "gzip", "snappy", "lz4", "zstd"] = "none"

    @field_validator("acks", mode="before")
    @classmethod
    def _acks_to_str(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) else value


class KafkaConsumerConfig(_Section):
    """Consumer-group tuning parameters."""

    group_id: str = Field(default="anomaly-detector", min_length=1)
    auto_offset_reset: Literal["earliest", "latest"] = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = Field(default=100, ge=1)
    fetch_min_bytes: int = Field(default=1, ge=1)
    session_timeout_ms: int = Field(default=30_000, ge=1000)
    num_workers: int = Field(default=3, ge=1)


class KafkaConfig(_Section):
    """Kafka connection, producer, consumer and topic settings."""

    bootstrap_servers: str = Field(default="localhost:9092", min_length=1)
    producer: KafkaProducerConfig = Field(default_factory=KafkaProducerConfig)
    consumer: KafkaConsumerConfig = Field(default_factory=KafkaConsumerConfig)
    topics: TopicsConfig = Field(default_factory=TopicsConfig)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class WarehouseSimConfig(_Section):
    """Warehouse temperature/humidity sensor simulation."""

    n_warehouses: int = Field(default=2, ge=1)
    zones: list[str] = Field(default_factory=lambda: ["A", "B", "C"], min_length=1)
    sensors_per_zone: int = Field(default=2, ge=1)
    reading_interval_s: float = Field(default=1.0, gt=0)
    point_anomaly_rate: float = Field(default=0.02, ge=0, le=1)
    contextual_anomaly_rate: float = Field(default=0.01, ge=0, le=1)
    collective_anomaly_rate: float = Field(default=0.005, ge=0, le=1)
    point_duration_range: tuple[int, int] = (1, 5)
    collective_length_range: tuple[int, int] = (10, 30)
    off_hours_start: int = Field(default=0, ge=0, le=23)
    off_hours_end: int = Field(default=5, ge=0, le=23)

    @field_validator("point_duration_range", "collective_length_range")
    @classmethod
    def _valid_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        low, high = value
        if low < 1 or high < low:
            raise ValueError(f"range must satisfy 1 <= low <= high, got {value}")
        return value


class VehicleSimConfig(_Section):
    """GPS vehicle simulation."""

    n_vehicles: int = Field(default=5, ge=1)
    reading_interval_s: float = Field(default=10.0, gt=0)
    route_deviation_m: float = Field(default=500.0, gt=0)
    anomaly_rate: float = Field(default=0.02, ge=0, le=1)


class InventorySimConfig(_Section):
    """Inventory stock-level simulation."""

    n_skus: int = Field(default=50, ge=1)
    event_interval_s: float = Field(default=5.0, gt=0)
    transaction_probability: float = Field(default=0.25, gt=0, le=1)  # per SKU per interval
    anomaly_rate: float = Field(default=0.01, ge=0, le=1)


class OrderSimConfig(_Section):
    """Order-fulfilment event simulation (not wired to a Kafka topic yet)."""

    enabled: bool = False  # include order events in the orchestrator stream
    arrival_rate_per_s: float = Field(default=0.2, gt=0)
    anomaly_rate: float = Field(default=0.02, ge=0, le=1)


class SimulationConfig(_Section):
    """Top-level simulation settings."""

    speed_multiplier: float = Field(default=1.0, gt=0)
    seed: int | None = 42
    warehouse: WarehouseSimConfig = Field(default_factory=WarehouseSimConfig)
    vehicle: VehicleSimConfig = Field(default_factory=VehicleSimConfig)
    inventory: InventorySimConfig = Field(default_factory=InventorySimConfig)
    order: OrderSimConfig = Field(default_factory=OrderSimConfig)


# --------------------------------------------------------------------------- #
# Features and models
# --------------------------------------------------------------------------- #
class FeaturesConfig(_Section):
    """Stream-processing and feature-cache settings."""

    window_seconds: list[int] = Field(default_factory=lambda: [60, 300, 900], min_length=1)
    watermark_tolerance_s: int = Field(default=30, ge=0)
    cache_ttl_s: int = Field(default=3600, ge=1)


class StatisticalConfig(_Section):
    """Z-score, IQR and CUSUM detector settings."""

    zscore_threshold: float = Field(default=3.0, gt=0)
    iqr_window: int = Field(default=1000, ge=10)
    iqr_multiplier: float = Field(default=1.5, gt=0)
    cusum_k: float = Field(default=0.5, ge=0)
    cusum_h: float = Field(default=5.0, gt=0)
    cusum_cooldown_s: int = Field(default=60, ge=0)
    z_weight: float = Field(default=0.4, ge=0, le=1)
    iqr_weight: float = Field(default=0.3, ge=0, le=1)
    cusum_weight: float = Field(default=0.3, ge=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Self:
        total = self.z_weight + self.iqr_weight + self.cusum_weight
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"statistical detector weights must sum to 1.0, got {total:.3f}")
        return self


class IsolationForestConfig(_Section):
    """Isolation Forest settings."""

    n_estimators: int = Field(default=200, ge=1)
    contamination: float = Field(default=0.01, gt=0, le=0.5)
    max_features: float = Field(default=1.0, gt=0, le=1.0)
    max_samples: float = Field(default=1.0, gt=0, le=1.0)
    warmup_seconds: int = Field(default=3600, ge=0)
    reservoir_size: int = Field(default=5000, ge=1)
    refit_every: int = Field(default=1000, ge=1)
    random_state: int = 42
    best_params_path: str = "configs/isolation_forest_best_params.json"


class LSTMAutoencoderConfig(_Section):
    """LSTM autoencoder settings (collective anomalies)."""

    seq_len: int = Field(default=60, ge=2)
    hidden_sizes: tuple[int, int] = (64, 128)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    learning_rate: float = Field(default=1.0e-3, gt=0)
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    kl_weight: float = Field(default=0.1, ge=0)
    threshold_sigma: float = Field(default=3.0, gt=0)
    mc_dropout_passes: int = Field(default=10, ge=1)


class LSTMClassifierConfig(_Section):
    """LSTM sequence classifier settings (point and contextual anomalies)."""

    seq_len: int = Field(default=30, ge=2)
    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.3, ge=0, lt=1)
    learning_rate: float = Field(default=1.0e-3, gt=0)
    weight_decay: float = Field(default=1.0e-4, ge=0)
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=50, ge=1)
    early_stopping_patience: int = Field(default=8, ge=1)
    class_names: list[str] = Field(
        default_factory=lambda: ["normal", "point", "contextual"], min_length=2
    )


class EnsembleConfig(_Section):
    """Weighted-average fallback and stacking meta-learner settings."""

    statistical_weight: float = Field(default=0.2, ge=0, le=1)
    isolation_forest_weight: float = Field(default=0.25, ge=0, le=1)
    lstm_autoencoder_weight: float = Field(default=0.25, ge=0, le=1)
    lstm_classifier_weight: float = Field(default=0.3, ge=0, le=1)
    meta_learner_holdout: float = Field(default=0.2, gt=0, lt=1)
    cv_folds: int = Field(default=5, ge=2)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Self:
        total = (
            self.statistical_weight
            + self.isolation_forest_weight
            + self.lstm_autoencoder_weight
            + self.lstm_classifier_weight
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"ensemble weights must sum to 1.0, got {total:.3f}")
        return self


class ModelsConfig(_Section):
    """Settings for every detector and the ensemble."""

    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    lstm_autoencoder: LSTMAutoencoderConfig = Field(default_factory=LSTMAutoencoderConfig)
    lstm_classifier: LSTMClassifierConfig = Field(default_factory=LSTMClassifierConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


# --------------------------------------------------------------------------- #
# Alerting, inference, feedback, retraining
# --------------------------------------------------------------------------- #
class SeverityThresholds(_Section):
    """Score boundaries between INFO / WARNING / CRITICAL / EMERGENCY."""

    warning: float = Field(default=0.3, gt=0, lt=1)
    critical: float = Field(default=0.6, gt=0, lt=1)
    emergency: float = Field(default=0.85, gt=0, lt=1)

    @model_validator(mode="after")
    def _strictly_increasing(self) -> Self:
        if not self.warning < self.critical < self.emergency:
            raise ValueError("severity thresholds must satisfy warning < critical < emergency")
        return self


def _default_impact_weights() -> dict[str, float]:
    return {
        "cold_storage_temp": 1.0,
        "vehicle_deviation": 0.8,
        "inventory_drop": 0.7,
        "default": 0.5,
    }


class AlertsConfig(_Section):
    """Severity scoring, deduplication and escalation settings."""

    severity_thresholds: SeverityThresholds = Field(default_factory=SeverityThresholds)
    business_impact_weights: dict[str, float] = Field(default_factory=_default_impact_weights)
    dedup_window_s: int = Field(default=300, ge=0)
    escalation_after_s: int = Field(default=600, ge=0)

    @field_validator("business_impact_weights")
    @classmethod
    def _weights_in_unit_interval(cls, value: dict[str, float]) -> dict[str, float]:
        bad = {key: weight for key, weight in value.items() if not 0.0 <= weight <= 1.0}
        if bad:
            raise ValueError(f"business impact weights must be within [0, 1]: {bad}")
        return value


class InferenceConfig(_Section):
    """Streaming inference engine and anomaly store settings."""

    num_workers: int = Field(default=4, ge=1)
    backpressure_queue_depth: int = Field(default=1000, ge=1)
    sampling_every_n: int = Field(default=5, ge=1)
    anomaly_store_capacity: int = Field(default=10_000, ge=1)
    snapshot_interval_s: int = Field(default=300, ge=1)


class FeedbackConfig(_Section):
    """Analyst feedback loop and threshold calibration settings."""

    ema_alpha: float = Field(default=0.1, gt=0, le=1)
    fp_burst_count: int = Field(default=5, ge=1)
    fp_burst_window_minutes: int = Field(default=60, ge=1)
    fp_threshold_raise: float = Field(default=0.1, ge=0)
    missed_tp_count: int = Field(default=3, ge=1)
    missed_tp_threshold_lower: float = Field(default=0.05, ge=0)
    recalibrate_every_minutes: int = Field(default=30, ge=1)
    recalibrate_after_events: int = Field(default=50, ge=1)


class RetrainingConfig(_Section):
    """Continuous retraining and champion/challenger settings."""

    check_interval_minutes: int = Field(default=15, ge=1)
    feedback_events_trigger: int = Field(default=500, ge=1)
    scheduled_interval_hours: int = Field(default=24, ge=1)
    finetune_max_epochs: int = Field(default=10, ge=1)
    finetune_learning_rate: float = Field(default=1.0e-5, gt=0)
    replay_buffer_ratio: float = Field(default=0.5, ge=0, le=1)
    min_f1_improvement: float = Field(default=0.02, ge=0)
    rollback_error_spike: float = Field(default=0.20, gt=0)
    rollback_window_minutes: int = Field(default=60, ge=1)


# --------------------------------------------------------------------------- #
# Services, observability, paths
# --------------------------------------------------------------------------- #
class ApiConfig(_Section):
    """FastAPI service settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    predict_rate_limit_per_min: int = Field(default=100, ge=1)
    batch_predict_rate_limit_per_min: int = Field(default=10, ge=1)
    batch_max_size: int = Field(default=100, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8050"])


class DashboardConfig(_Section):
    """Dash dashboard settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8050, ge=1, le=65535)
    debug: bool = False
    api_url: str = "http://localhost:8000"
    refresh_interval_ms: int = Field(default=2000, ge=250)
    theme: str = "DARKLY"
    max_sse_connections: int = Field(default=50, ge=1)


class SLOConfig(_Section):
    """Service-level objectives validated on Day 25."""

    detection_latency_p99_ms: float = Field(default=500.0, gt=0)
    min_throughput_events_per_s: float = Field(default=500.0, gt=0)
    min_f1: float = Field(default=0.80, ge=0, le=1)
    max_false_positive_rate: float = Field(default=0.15, ge=0, le=1)
    max_dashboard_load_s: float = Field(default=2.0, gt=0)


class AlertRulesConfig(_Section):
    """Thresholds behind the Prometheus/Alertmanager alert rules."""

    latency_p99_ms: float = Field(default=500.0, gt=0)
    latency_for_minutes: int = Field(default=5, ge=1)
    kafka_lag_messages: int = Field(default=10_000, ge=1)
    min_f1: float = Field(default=0.7, ge=0, le=1)


class ObservabilityConfig(_Section):
    """Prometheus, Grafana and SLO settings."""

    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    grafana_port: int = Field(default=3000, ge=1, le=65535)
    kafka_ui_port: int = Field(default=8080, ge=1, le=65535)
    metrics_path: str = "/metrics"
    slo: SLOConfig = Field(default_factory=SLOConfig)
    alerting: AlertRulesConfig = Field(default_factory=AlertRulesConfig)


class PathsConfig(_Section):
    """Filesystem locations; relative paths resolve against the project root."""

    model_registry: str = "model_registry"
    reports: str = "reports"
    feedback_store: str = "feedback_store.json"
    anomaly_snapshots: str = "anomaly_snapshots"


class AppConfig(_Section):
    """Root configuration object."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def resolve_project_path(value: str | Path) -> Path:
    """Resolve ``value`` against the project root unless it is already absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_path(path: str | Path | None, environ: Mapping[str, str]) -> Path:
    """Pick the config file: explicit argument, then ``CONFIG_PATH``, then the default.

    An explicit argument is resolved against the current working directory (CLI
    convention); ``CONFIG_PATH`` is resolved against the project root.
    """
    if path:
        return Path(path).expanduser().resolve()
    env_value = environ.get("CONFIG_PATH", "").strip()
    if env_value:
        return resolve_project_path(env_value)
    return DEFAULT_CONFIG_PATH


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {path}")
    return data


def _apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Apply :data:`ENV_OVERRIDES` in place; empty values are treated as unset."""
    for env_name, key_path in ENV_OVERRIDES.items():
        raw = environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        node = data
        for key in key_path[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[key_path[-1]] = raw.strip()


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    load_env_file: bool = True,
) -> AppConfig:
    """Load, override and validate the configuration.

    Args:
        path: Config file to read. Defaults to ``$CONFIG_PATH`` or
            ``configs/config.yaml``.
        environ: Environment mapping to use instead of ``os.environ``. Passing
            one also skips ``.env`` loading, which keeps tests hermetic.
        load_env_file: Whether to read ``<project root>/.env`` first (only when
            ``environ`` is not given). Existing environment variables win.

    Raises:
        FileNotFoundError: The config file does not exist.
        ValueError: The YAML is not a mapping, or validation failed
            (``pydantic.ValidationError`` is a ``ValueError``).
    """
    if environ is None:
        if load_env_file:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env: Mapping[str, str] = os.environ
    else:
        env = environ
    raw = _read_yaml(resolve_config_path(path, env))
    _apply_env_overrides(raw, env)
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide configuration (loaded once, then cached)."""
    return load_config()
