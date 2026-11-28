"""Start the API, run the Locust load test against it, and propagate the SLO verdict (Day 23).

The API enforces per-IP rate limits (100 req/min on ``/predict``) that a load generator on one IP
would trip at once. This runner therefore starts the API with a temporary copy of
``configs/config.yaml`` whose limits are lifted and whose storage points into a temp directory, so
nothing in the working tree is touched. Everything else is the production configuration.

Usage::

    python tests/load/run_load_test.py                       # full profile (3 minutes)
    LOAD_RAMP_S=10 LOAD_SUSTAIN_S=20 python tests/load/run_load_test.py   # quick smoke run

Exit code 0 means every SLO held; 1 means at least one was missed (see the printed summary).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCUSTFILE = ROOT / "tests" / "load" / "locustfile.py"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_load_config(directory: Path, port: int) -> Path:
    """Copy the project config with rate limits lifted and storage in ``directory``."""
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    cfg["api"].update(
        port=port,
        predict_rate_limit_per_min=10_000_000,
        batch_predict_rate_limit_per_min=10_000_000,
    )
    cfg["paths"].update(
        model_registry=str(directory / "registry"),
        feedback_store=str(directory / "feedback.json"),
        anomaly_snapshots=str(directory / "snapshots"),
    )
    cfg.setdefault("logging", {})["level"] = "WARNING"
    path = directory / "config.load.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def wait_until_healthy(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"API at {url} did not become healthy within {timeout:.0f}s")


def main() -> int:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="scad-load-") as tmp:
        directory = Path(tmp)
        env = {**os.environ, "CONFIG_PATH": str(write_load_config(directory, port))}
        env.setdefault("LOAD_SUMMARY_PATH", str(ROOT / "reports" / "load_test_summary.json"))
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            cwd=ROOT,
            env=env,
        )  # fmt: skip
        try:
            wait_until_healthy(base_url)
            locust = subprocess.run(
                [sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
                 "--host", base_url, "--only-summary"],
                cwd=ROOT,
                env=env,
                check=False,
            )  # fmt: skip
            return locust.returncode
        finally:
            api.terminate()
            try:
                api.wait(timeout=15)
            except subprocess.TimeoutExpired:
                api.kill()


if __name__ == "__main__":
    raise SystemExit(main())
