"""Locust load test of the inference API with SLO assertions (Day 23).

Two kinds of virtual users:

* ``EventProducer`` (90 % of users): posts sensor readings to ``POST /api/v1/predict``. Every
  request carries an ``X-Event-Timestamp`` header (the instant the event was created) and the
  user records the **end-to-end latency** ``response received - event created`` in its own metric;
  that is what the 500 ms SLO is checked against.
* ``AlertConsumer`` (10 %): polls ``GET /api/v1/anomalies`` like the dashboard does.

Load profile (``RampShape``): 0 -> 100 users over 60 s, then 100 users for 120 s.

SLO assertions run when the test ends (the process exit code is 1 if one fails):

* end-to-end p99 latency of predictions  < 500 ms
* error rate                             < 0.1 %
* sustained throughput (100-user phase)  > 800 requests/s

Run it (the API must be reachable and its rate limits raised, see ``run_load_test.py``)::

    python tests/load/run_load_test.py            # starts the API, runs this file, checks SLOs
    locust -f tests/load/locustfile.py --headless --host http://localhost:8000

Knobs (environment variables): ``LOAD_USERS`` (100), ``LOAD_RAMP_S`` (60), ``LOAD_SUSTAIN_S``
(120), ``SLO_P99_MS`` (500), ``SLO_ERROR_RATE`` (0.001), ``SLO_MIN_RPS`` (800) and
``LOAD_SUMMARY_PATH`` (where the JSON summary goes). The defaults are the Day 23 targets.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import gevent
import numpy as np
from locust import HttpUser, LoadTestShape, between, constant, events, task
from locust.env import Environment

USERS = int(os.environ.get("LOAD_USERS", "100"))
RAMP_S = int(os.environ.get("LOAD_RAMP_S", "60"))
SUSTAIN_S = int(os.environ.get("LOAD_SUSTAIN_S", "120"))
P99_TARGET_MS = float(os.environ.get("SLO_P99_MS", "500"))
MAX_ERROR_RATE = float(os.environ.get("SLO_ERROR_RATE", "0.001"))
MIN_RPS = float(os.environ.get("SLO_MIN_RPS", "800"))
SUMMARY_PATH = Path(os.environ.get("LOAD_SUMMARY_PATH", "reports/load_test_summary.json"))

EVENT_TIMESTAMP_HEADER = "X-Event-Timestamp"
SIM_START = datetime(2026, 11, 24, 8, 0, 0, tzinfo=UTC)
SPIKE_RATE = 0.01

#: End-to-end latencies (ms) of successful predictions, one entry per request.
E2E_LATENCIES_MS: list[float] = []
#: ``(monotonic time, requests so far)`` when the full user count was first reached.
SUSTAIN_START: dict[str, float] = {}
_sensor_ids = itertools.count(1)


def iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Virtual users
# --------------------------------------------------------------------------- #
class EventProducer(HttpUser):
    """Streams readings of one sensor to ``/api/v1/predict`` as fast as the API answers."""

    weight = 9
    wait_time = constant(0)

    def on_start(self) -> None:
        self.sensor_id = f"load-{next(_sensor_ids):04d}"
        self.sequence = 0
        self.rng = random.Random(self.sensor_id)

    def _event(self) -> dict[str, Any]:
        self.sequence += 1
        value = 3.0 + self.rng.gauss(0, 0.15)
        if self.rng.random() < SPIKE_RATE:
            value += self.rng.choice((-1, 1)) * self.rng.uniform(5, 12)  # an anomaly now and then
        return {
            "source_type": "warehouse",
            "payload": {
                "sensor_id": self.sensor_id,
                "warehouse_id": "wh-load",
                "timestamp": iso(SIM_START + timedelta(seconds=self.sequence)),
                "reading_type": "temperature",
                "value": value,
                "unit": "C",
                "metadata": {"zone": "A", "storage_type": "cold_storage"},
            },
        }

    @task
    def predict(self) -> None:
        created = time.time()  # the event comes into being now ...
        headers = {EVENT_TIMESTAMP_HEADER: iso(datetime.fromtimestamp(created, UTC))}
        with self.client.post(
            "/api/v1/predict",
            json=self._event(),
            headers=headers,
            name="/api/v1/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            if "is_anomaly" not in response.json():
                response.failure("response without is_anomaly")
                return
            response.success()
        # ... and is judged once the verdict is back: event created -> verdict received.
        E2E_LATENCIES_MS.append((time.time() - created) * 1000.0)


class AlertConsumer(HttpUser):
    """Polls the alert list, as the dashboard does."""

    weight = 1
    wait_time = between(0.5, 1.5)

    @task
    def poll_anomalies(self) -> None:
        self.client.get("/api/v1/anomalies", params={"limit": 50}, name="/api/v1/anomalies")


# --------------------------------------------------------------------------- #
# Load profile
# --------------------------------------------------------------------------- #
class RampShape(LoadTestShape):
    """0 -> ``USERS`` over ``RAMP_S`` seconds, then hold for ``SUSTAIN_S`` seconds."""

    def tick(self) -> tuple[int, float] | None:
        elapsed = self.get_run_time()
        if elapsed >= RAMP_S + SUSTAIN_S:
            return None
        spawn_rate = max(USERS / max(RAMP_S, 1), 1.0)
        if elapsed < RAMP_S:
            return max(1, round(USERS * elapsed / RAMP_S)), spawn_rate
        return USERS, spawn_rate


# --------------------------------------------------------------------------- #
# SLO bookkeeping
# --------------------------------------------------------------------------- #
@events.test_start.add_listener
def _watch_for_full_load(environment: Environment, **_: Any) -> None:
    E2E_LATENCIES_MS.clear()
    SUSTAIN_START.clear()

    def watch() -> None:
        while True:
            runner = environment.runner
            if runner is not None and runner.user_count >= USERS:
                SUSTAIN_START.update(
                    t=time.monotonic(), n=float(environment.stats.total.num_requests)
                )
                return
            gevent.sleep(0.5)

    gevent.spawn(watch)


def evaluate(environment: Environment) -> dict[str, Any]:
    """Collect the measurements and compare them to the SLOs."""
    stats = environment.stats.total
    e2e = np.array(E2E_LATENCIES_MS) if E2E_LATENCIES_MS else np.array([float("inf")])
    sustained_rps = 0.0
    if SUSTAIN_START:
        window = time.monotonic() - SUSTAIN_START["t"]
        if window > 0:
            sustained_rps = (stats.num_requests - SUSTAIN_START["n"]) / window
    measured = {
        "requests": stats.num_requests,
        "failures": stats.num_failures,
        "error_rate": stats.fail_ratio,
        "sustained_rps": round(sustained_rps, 1),
        "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 2),
        "e2e_p95_ms": round(float(np.percentile(e2e, 95)), 2),
        "e2e_p99_ms": round(float(np.percentile(e2e, 99)), 2),
        "server_p99_ms": stats.get_response_time_percentile(0.99),
    }
    checks = {
        f"e2e p99 < {P99_TARGET_MS:g} ms": measured["e2e_p99_ms"] < P99_TARGET_MS,
        f"error rate < {MAX_ERROR_RATE:.1%}": measured["error_rate"] < MAX_ERROR_RATE,
        f"sustained throughput > {MIN_RPS:g} req/s": sustained_rps > MIN_RPS,
        "requests were actually sent": stats.num_requests > 0 and bool(SUSTAIN_START),
    }
    return {"measured": measured, "checks": checks, "passed": all(checks.values())}


@events.quitting.add_listener
def _assert_slos(environment: Environment, **_: Any) -> None:
    result = evaluate(environment)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(result, indent=2))
    print("\n=== SLO check ===")
    for name, ok in result["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  measured: {json.dumps(result['measured'])}")
    if not result["passed"]:
        environment.process_exit_code = 1
