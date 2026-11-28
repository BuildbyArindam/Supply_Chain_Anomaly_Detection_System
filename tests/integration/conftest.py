"""Shared Kafka fixtures for the integration tests (Day 23).

The ``kafka`` fixture gives a test an isolated Kafka environment:

* **testcontainers** (default when a Docker daemon is reachable): one real
  ``confluentinc/cp-kafka`` broker in KRaft mode is started per test session. Every test creates
  its *own uniquely named topics* on it, so tests can run in any order without seeing each other's
  messages, while the container start-up cost (~10 s) is paid once.
* **mock** (``SCAD_KAFKA_BACKEND=mock``): the in-memory broker from ``tests/mock_kafka.py``. It is
  what lets the same tests run without Docker, e.g. in a quick local check. It is a stand-in, so
  the CI job runs the tests against the real container.

Without Docker and without ``SCAD_KAFKA_BACKEND=mock`` the Kafka tests are skipped, never failed.
The modules that use these fixtures carry ``pytestmark = pytest.mark.integration`` so that CI can
run them in a Docker job of their own (``pytest -m integration``).
"""

from __future__ import annotations

import json
import os
import time
import uuid
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.utils.config import AppConfig
from tests.mock_kafka import MockBroker

KAFKA_IMAGE = "confluentinc/cp-kafka:7.6.0"
TOPIC_FIELDS = (
    "warehouse_sensors",
    "vehicle_gps",
    "inventory_events",
    "anomaly_alerts",
    "dlq_events",
    "model_health",
)


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        return False
    return True


@dataclass
class KafkaEnv:
    """A Kafka cluster plus a config whose topic names belong to one test."""

    backend: str
    cfg: AppConfig
    broker: MockBroker | None = None

    # ------------------------------------------------------------------ factories
    @property
    def consumer_factory(self) -> Callable[[dict[str, Any]], Any]:
        """Pass to the consumer classes: the mock broker's consumer, or the real Kafka one."""
        if self.broker:
            return self.broker.consumer_factory
        from confluent_kafka import Consumer

        return Consumer

    def topic(self, field: str) -> str:
        """Name of one of this test's topics (``warehouse_sensors``, ``anomaly_alerts`` ...)."""
        return str(getattr(self.cfg.kafka.topics, field).name)

    def producer(self) -> KafkaProducerManager:
        """A producer manager writing to this test's topics (close it, or use ``with``)."""
        kwargs: dict[str, Any] = {"metrics": ProducerMetrics(CollectorRegistry())}
        if self.broker:
            kwargs.update(producer_factory=self.broker.producer_factory, background_polling=False)
        return KafkaProducerManager.from_config(self.cfg.kafka, **kwargs)

    def raw_producer(self) -> Any:
        """A bare producer (produce, poll, flush), e.g. for the consumer's dead letters."""
        if self.broker:
            return self.broker.producer_factory({})
        from confluent_kafka import Producer

        return Producer({"bootstrap.servers": self.cfg.kafka.bootstrap_servers})

    # ------------------------------------------------------------------ reading
    def read(self, field: str, *, at_least: int = 0, timeout: float = 30.0) -> list[dict[str, Any]]:
        """Return every JSON message on a topic, waiting until ``at_least`` have arrived."""
        topic = self.topic(field)
        deadline = time.monotonic() + timeout
        while True:
            values = self._read_all(topic)
            if len(values) >= at_least or time.monotonic() >= deadline:
                return values
            time.sleep(0.2)

    def _read_all(self, topic: str) -> list[dict[str, Any]]:
        if self.broker:
            return [json.loads(r.value) for r in self.broker.messages(topic) if r.value]
        from confluent_kafka import Consumer

        consumer = Consumer(
            {
                "bootstrap.servers": self.cfg.kafka.bootstrap_servers,
                "group.id": f"reader-{uuid.uuid4().hex}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        out: list[dict[str, Any]] = []
        try:
            consumer.subscribe([topic])
            idle_since = time.monotonic()
            while time.monotonic() - idle_since < 3.0:  # stop after 3 s without a new record
                for msg in consumer.consume(500, timeout=0.5):
                    if msg.error() is None and msg.value():
                        out.append(json.loads(msg.value()))
                        idle_since = time.monotonic()
        finally:
            consumer.close()
        return out


@pytest.fixture(scope="session")
def _kafka_container() -> Iterator[str | None]:
    """Bootstrap address of the session's real broker, or ``None`` for the mock backend."""
    backend = os.environ.get("SCAD_KAFKA_BACKEND", "auto").lower()
    if backend == "mock":
        yield None
        return
    if not _docker_available():
        pytest.skip("no Docker daemon reachable (set SCAD_KAFKA_BACKEND=mock for the mock broker)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from testcontainers.kafka import KafkaContainer
        except ImportError:  # pragma: no cover - layout of newer testcontainers releases
            from testcontainers.community.kafka import KafkaContainer
    container = KafkaContainer(KAFKA_IMAGE).with_kraft()
    with container:
        yield container.get_bootstrap_server()


def _create_topics(bootstrap: str, names: dict[str, tuple[str, int]]) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics(
        [
            NewTopic(name, num_partitions=parts, replication_factor=1)
            for name, parts in names.values()
        ]
    )
    for name, future in futures.items():
        try:
            future.result(timeout=30)
        except Exception as exc:  # pragma: no cover - only on a broken broker
            raise RuntimeError(f"could not create topic {name}: {exc}") from exc


@pytest.fixture
def kafka(_kafka_container: str | None) -> KafkaEnv:
    """An isolated Kafka environment: fresh uniquely named topics for this test."""
    suffix = uuid.uuid4().hex[:8]
    base = AppConfig()
    topics = base.kafka.topics
    renamed = {
        field: getattr(topics, field).model_copy(
            update={"name": f"{getattr(topics, field).name}-{suffix}"}
        )
        for field in TOPIC_FIELDS
    }
    consumer = base.kafka.consumer.model_copy(
        update={"auto_offset_reset": "earliest", "group_id": f"it-{suffix}"}
    )
    kafka_cfg = base.kafka.model_copy(
        update={
            "topics": topics.model_copy(update=renamed),
            "consumer": consumer,
            "bootstrap_servers": _kafka_container or base.kafka.bootstrap_servers,
        }
    )
    cfg = base.model_copy(update={"kafka": kafka_cfg})

    if _kafka_container is None:
        broker = MockBroker({t.name: t.partitions for t in renamed.values()})
        return KafkaEnv("mock", cfg, broker)
    _create_topics(_kafka_container, {f: (t.name, t.partitions) for f, t in renamed.items()})
    return KafkaEnv("testcontainers", cfg)


@pytest.fixture
def app_config_in(tmp_path: Any) -> Callable[[AppConfig], AppConfig]:
    """Return a function that points a config's model/feedback/snapshot paths into ``tmp_path``."""

    def apply(cfg: AppConfig) -> AppConfig:
        paths = cfg.paths.model_copy(
            update={
                "model_registry": str(tmp_path / "registry"),
                "feedback_store": str(tmp_path / "feedback.json"),
                "anomaly_snapshots": str(tmp_path / "snapshots"),
            }
        )
        return cfg.model_copy(update={"paths": paths})

    return apply
