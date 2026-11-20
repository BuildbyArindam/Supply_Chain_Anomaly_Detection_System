import importlib.util
import threading
from collections import Counter
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from src.data_simulation import SimulationOrchestrator
from src.kafka.producer import KafkaProducerManager, ProducerMetrics
from src.utils.config import PROJECT_ROOT, KafkaConfig, SimulationConfig, load_config
from tests.unit.test_kafka_producer import FakeProducer

SCRIPT = PROJECT_ROOT / "scripts" / "start_simulation.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("start_simulation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_producer(fake=None):
    fake = fake or FakeProducer()
    producer = KafkaProducerManager.from_config(
        KafkaConfig(),
        background_polling=False,
        producer_factory=lambda cfg: fake,
        metrics=ProducerMetrics(CollectorRegistry()),
    )
    return producer, fake


def make_orchestrator(**kwargs):
    return SimulationOrchestrator.from_config(SimulationConfig(seed=1), **kwargs)


def test_stream_reaches_every_topic(script):
    producer, fake = make_producer()
    sent = script.run_stream(make_orchestrator(), producer, duration_s=120, throttle=False)
    producer.close(2.0)

    topics = Counter(topic for topic, _, _ in fake.produced)
    assert sent == sum(topics.values())
    assert set(topics) == {"warehouse-sensors", "vehicle-gps", "inventory-events"}
    assert "dlq-events" not in topics  # every real simulator event passes validation
    assert producer.in_flight == 0


def test_max_events_stops_the_stream(script):
    producer, fake = make_producer()
    sent = script.run_stream(make_orchestrator(), producer, max_events=25, throttle=False)
    producer.close(2.0)
    assert sent == 25
    assert len(fake.produced) == 25


def test_stop_event_ends_an_unbounded_stream(script):
    producer, _ = make_producer()
    stop = threading.Event()
    stop.set()
    assert script.run_stream(make_orchestrator(), producer, throttle=False, stop=stop) == 0
    producer.close(1.0)


def test_orders_are_published_on_the_inventory_topic(script):
    producer, fake = make_producer()
    orch = make_orchestrator(include_orders=True)
    script.run_stream(orch, producer, duration_s=600, throttle=False)
    producer.close(2.0)
    kinds = {v["source_type"] for t, _, v in fake.produced if t == "inventory-events"}
    assert kinds == {"inventory", "order"}


def test_kafka_outage_dead_letters_instead_of_crashing(script):
    from collections import deque

    from tests.unit.test_kafka_producer import FATAL_TOO_LARGE

    fake = FakeProducer()
    fake.script["warehouse-sensors"] = deque([FATAL_TOO_LARGE] * 10_000)
    producer, _ = make_producer(fake)
    script.run_stream(make_orchestrator(), producer, duration_s=5, throttle=False)
    producer.close(2.0)
    dlq = [v for t, _, v in fake.produced if t == "dlq-events"]
    assert dlq and all(m["original_topic"] == "warehouse-sensors" for m in dlq)
    assert any(t == "vehicle-gps" for t, _, _ in fake.produced)  # other streams unaffected


class FakeManager:
    """Stands in for KafkaProducerManager inside script.run()."""

    healthy = True
    instances: list = []

    def __init__(self):
        self.closed = False

    @classmethod
    def from_config(cls, cfg, **kwargs):
        cls.last_cfg, cls.last_kwargs = cfg, kwargs
        instance = cls()
        cls.instances.append(instance)
        return instance

    def check_health(self):
        from src.kafka.producer import KafkaHealth

        if self.healthy:
            return KafkaHealth(healthy=True, latency_ms=3.0, brokers=1)
        return KafkaHealth(healthy=False, latency_ms=3.0, error="all brokers down")

    def close(self, timeout=0):
        self.closed = True
        return 0


def test_check_flag_reports_ok(script, monkeypatch, capsys):
    monkeypatch.setattr(script, "KafkaProducerManager", FakeManager)
    FakeManager.healthy = True
    code = script.main(["--check", "--bootstrap-servers", "broker:9999"])
    assert code == script.EXIT_OK
    assert "Kafka OK" in capsys.readouterr().out
    assert FakeManager.last_cfg.bootstrap_servers == "broker:9999"
    assert FakeManager.instances[-1].closed


def test_unhealthy_kafka_exits_with_failure(script, monkeypatch, capsys):
    monkeypatch.setattr(script, "KafkaProducerManager", FakeManager)
    FakeManager.healthy = False
    assert script.main([]) == script.EXIT_FAILURE
    assert "all brokers down" in capsys.readouterr().err
    assert FakeManager.instances[-1].closed


def test_bad_speed_is_a_usage_error(script, capsys):
    assert script.main(["--speed", "0"]) == script.EXIT_USAGE
    assert "--speed" in capsys.readouterr().err


def test_missing_config_is_reported(script, tmp_path, capsys):
    assert script.main(["--config", str(tmp_path / "nope.yaml")]) == script.EXIT_FAILURE
    assert "Configuration error" in capsys.readouterr().err


def test_main_cli_simulate_delegates_to_script(monkeypatch):
    import src.main as cli

    monkeypatch.setattr(
        cli.runpy, "run_path", lambda path, run_name: {"run": lambda cfg: 7}, raising=True
    )
    assert cli._cmd_simulate(load_config(load_env_file=False, environ={})) == 7


def test_script_exists_at_expected_path():
    assert Path(SCRIPT).is_file()
