"""Playwright smoke test of the dashboard (Day 18).

Starts the real Dash server on a free port with the unit tests' fake data source (no API,
Kafka or torch needed), opens it in headless Chromium and checks that the key elements exist
and that live data arrives.

Setup, once:  ``pip install playwright && playwright install chromium``
Run:          ``pytest tests/integration/test_dashboard_smoke.py -m browser``
              (``CHROMIUM_PATH=/path/to/chrome`` uses an existing Chromium build)

The test skips itself if Playwright or a Chromium build is missing.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.dashboard.app import create_dash_app
from tests.unit.test_dashboard_data_service import FakeSource, alert

pytestmark = [pytest.mark.integration, pytest.mark.browser]

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

NAV_LINKS = [
    "nav-overview",
    "nav-warehouse",
    "nav-vehicles",
    "nav-inventory",
    "nav-alerts",
    "nav-model-health",
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_url() -> Iterator[str]:
    from werkzeug.serving import make_server

    source = FakeSource()
    source.alert_rows = [alert(1, "CRITICAL", 2), alert(2, "WARNING", 3), alert(3, "INFO", 4)]
    app = create_dash_app(source, public_api_url="", start_poller=False)
    app.scad_service.refresh()
    port = free_port()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as pw:
        try:
            instance = pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        except Exception as exc:  # no Chromium build installed
            pytest.skip(f"Chromium is not available: {str(exc).splitlines()[0]}")
        yield instance
        instance.close()


def test_dashboard_loads_with_its_key_elements(browser: Any, dashboard_url: str) -> None:
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    try:
        page.goto(dashboard_url, wait_until="networkidle")

        assert "Supply Chain Anomaly" in page.title()
        for element_id in ["sidebar", "status-strip", "page-content", "page-title"]:
            assert page.locator(f"#{element_id}").count() == 1, element_id
        for link in NAV_LINKS:
            assert page.locator(f"#{link}").is_visible(), link

        # Dark theme: the body background is dark (DARKLY), not white.
        red, green, blue = page.evaluate(
            "getComputedStyle(document.body).backgroundColor.match(/\\d+/g).map(Number)"
        )[:3]
        assert max(red, green, blue) < 90

        # Live data (this also proves dcc.Interval ticks): the refresh callback fills the stores.
        page.wait_for_function(
            "document.querySelector('#status-strip')?.innerText.includes('alerts today')",
            timeout=10_000,
        )
        assert "Live" in page.locator("#status-strip").inner_text()
        page.wait_for_function(
            "document.querySelector('#ov-active-alerts')?.innerText.trim() !== '—'", timeout=10_000
        )
        for kpi in ["ov-active-alerts", "ov-critical-24h", "ov-model-health", "ov-latency"]:
            assert page.locator(f"#{kpi}").is_visible(), kpi  # the four KPI cards
        assert page.locator("#ov-active-alerts").inner_text().strip() == "3"
        assert page.locator("#ov-feed-table tbody tr").count() == 3  # the live alert feed
        assert page.locator("#ov-donut").is_visible() and page.locator("#ov-trend").is_visible()
        assert not errors, errors
    finally:
        page.close()


class LiveWarehouseSource(FakeSource):
    """A fake source with one temperature sensor whose series gains a reading every second."""

    def __init__(self) -> None:
        super().__init__()
        self.start = datetime.now(UTC)
        self.entity_rows = [
            {
                "entity_id": "wh1-A-temp-01",
                "source_type": "warehouse",
                "signal": "value",
                "value": 3.0,
                "timestamp": self.start.isoformat(),
                "unit": "celsius",
                "zone": "A",
                "warehouse_id": "wh1",
                "sensor_type": "temperature",
                "expected_range": [2.0, 4.0],
            }
        ]

    def timeseries(self, entity_id: str, lookback_seconds: float) -> dict[str, Any] | None:
        now = datetime.now(UTC).replace(microsecond=0)
        points = [
            {"t": (now - timedelta(seconds=i)).isoformat().replace("+00:00", "Z"), "v": 3.0}
            for i in range(60, -1, -1)
        ]
        return {**self.entity_rows[0], "points": points}


def test_warehouse_chart_grows_in_place_and_shows_the_heatmap(browser: Any) -> None:
    from werkzeug.serving import make_server

    source = LiveWarehouseSource()
    app = create_dash_app(source, public_api_url="", start_poller=True)
    port = free_port()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    try:
        page.goto(f"http://127.0.0.1:{port}/warehouse", wait_until="networkidle")
        page.wait_for_selector("#wh-graph .main-svg", timeout=10_000)
        count = "document.querySelector('#wh-graph .js-plotly-plot').data[0].x.length"
        first = page.evaluate(count)
        page.evaluate("document.querySelector('#wh-graph .js-plotly-plot').__kept = true")
        page.wait_for_function(f"{count} > {first}", timeout=15_000)  # points were appended
        # extendTraces mutated the same plot element; a redraw would have replaced it
        assert page.evaluate("document.querySelector('#wh-graph .js-plotly-plot').__kept") is True
        assert "wh1-A-temp-01" in page.locator("#wh-sensor").inner_text()
        page.wait_for_selector("#wh-heatmap .main-svg", timeout=10_000)
        assert "3.0°C" in page.locator("#wh-heatmap").inner_text()
        assert not errors, errors
    finally:
        page.close()
        server.shutdown()
        app.scad_service.stop()


def test_navigation_reaches_every_page(browser: Any, dashboard_url: str) -> None:
    page = browser.new_page()
    try:
        page.goto(dashboard_url, wait_until="networkidle")
        for link, path in zip(
            NAV_LINKS,
            ["/", "/warehouse", "/vehicles", "/inventory", "/alerts", "/model-health"],
            strict=True,
        ):
            page.locator(f"#{link}").click()
            page.wait_for_url(f"**{path}", timeout=5_000)
            assert page.locator("#page-title").is_visible(), path
            assert page.locator("#sidebar").is_visible(), path
        time.sleep(0.1)
    finally:
        page.close()
