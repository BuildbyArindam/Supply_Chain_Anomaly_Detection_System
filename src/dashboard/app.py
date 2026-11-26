"""The Dash application (Day 18): multi-page shell, shared stores and the live refresh.

Run it with ``make run-dashboard`` / ``python -m src dashboard``. ``src.dashboard.app.app`` is the
ready-made instance that command uses.

How the pieces fit
------------------
* :class:`~src.dashboard.data_service.DashboardDataService` polls the source (the API over HTTP by
  default) on a background thread and keeps the latest snapshot.
* ``dcc.Interval(id="live-interval")`` ticks every ``dashboard.refresh_interval_ms`` (2 s); one
  callback copies the snapshot into two browser-side stores, ``anomaly-store`` (the 50 most severe
  active alerts) and ``metrics-store`` (headline numbers). Every page reads those stores instead
  of calling the service itself, and the callback returns ``no_update`` while nothing changed.
* ``assets/sse_client.js`` opens an ``EventSource`` on ``/api/v1/stream/alerts`` and writes each
  new alert into ``sse-latest``, which raises a toast for CRITICAL and EMERGENCY alerts.

The browser must be able to reach the SSE URL, which in Docker differs from the URL the dashboard
server uses: set ``PUBLIC_API_URL`` (default: ``dashboard.api_url``).
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):  # ``python src/dashboard/app.py``: put the repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update

from src.dashboard.components.navbar import build_sidebar
from src.dashboard.components.theme import SEVERITY_BOOTSTRAP
from src.dashboard.data_service import DashboardDataService, DashboardSource, HttpSource, slim_alert
from src.dashboard.utils import format_timestamp, format_value_with_unit
from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

_log = get_logger(__name__)
_HERE = Path(__file__).parent
APP_TITLE = "Supply Chain Anomaly Monitor"
ACTIVE_ALERTS_IN_STORE = 50


def _status_strip(metrics: dict[str, Any] | None) -> html.Div:
    """Build the connection badge plus the headline numbers shown at the bottom of the sidebar."""
    if not metrics:
        return html.Div("Connecting…")
    connected = bool(metrics.get("connected"))
    status = str(metrics.get("api_status", "unreachable"))
    badge = dbc.Badge(
        "Live" if connected and status == "ok" else "Degraded" if connected else "Offline",
        color="success" if connected and status == "ok" else "warning" if connected else "danger",
        className="me-2",
        id="status-badge",
    )
    p99 = format_value_with_unit(metrics.get("avg_latency_p99"), "ms", 0)
    lines = [
        html.Div([badge, f"updated {format_timestamp(metrics.get('updated_at'))} UTC"]),
        html.Div(f"{metrics.get('total_alerts_today', 0)} alerts today · p99 {p99}"),
    ]
    errors = metrics.get("errors") or {}
    if errors:
        lines.append(
            html.Div(f"Unavailable: {', '.join(sorted(errors))}", className="text-warning")
        )
    return html.Div(lines)


def create_dash_app(
    source: DashboardSource | None = None,
    *,
    config: AppConfig | None = None,
    public_api_url: str | None = None,
    start_poller: bool = True,
) -> Dash:
    """Build the dashboard.

    Args:
        source: Data source. Default: :class:`HttpSource` on ``dashboard.api_url``. Pass an
            :class:`~src.dashboard.data_service.InProcessSource` to run next to the pipeline.
        config: Application config (``get_config()`` if omitted).
        public_api_url: API base URL as the *browser* sees it, for the SSE connection (``None``
            reads ``PUBLIC_API_URL``, then ``dashboard.api_url``; ``""`` disables SSE).
        start_poller: Start the data poller with the first request (tests pass ``False`` and
            drive ``app.scad_service.refresh()`` themselves).

    The service is available as ``app.scad_service``.
    """
    cfg = config or get_config()
    dash_cfg = cfg.dashboard
    theme = getattr(dbc.themes, dash_cfg.theme.upper(), dbc.themes.DARKLY)
    service = DashboardDataService(source or HttpSource(dash_cfg.api_url))
    if public_api_url is None:
        public_api_url = os.environ.get("PUBLIC_API_URL") or dash_cfg.api_url

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder=str(_HERE / "pages"),
        assets_folder=str(_HERE / "assets"),
        external_stylesheets=[theme],
        suppress_callback_exceptions=True,
        title=APP_TITLE,
        update_title=None,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.scad_service = service  # type: ignore[attr-defined]

    app.layout = html.Div(
        [
            dcc.Store(id="anomaly-store"),
            dcc.Store(id="metrics-store"),
            dcc.Store(id="sse-latest"),
            dcc.Interval(id="live-interval", interval=dash_cfg.refresh_interval_ms),
            html.Div(
                id="sse-config",
                **(
                    {"data-sse-url": f"{public_api_url.rstrip('/')}/api/v1/stream/alerts"}
                    if public_api_url
                    else {}
                ),
                style={"display": "none"},
            ),
            dbc.Toast(
                id="alert-toast",
                is_open=False,
                dismissable=True,
                duration=8000,
                style={"position": "fixed", "top": 16, "right": 16, "zIndex": 2000, "width": 340},
            ),
            build_sidebar(),
            html.Main(dash.page_container, id="page-content", className="content"),
        ],
        id="app-root",
    )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("anomaly-store", "data"),
        Output("metrics-store", "data"),
        Input("live-interval", "n_intervals"),
        State("metrics-store", "data"),
    )
    def refresh_stores(_n: int | None, current: dict[str, Any] | None):
        """Copy the latest snapshot into the stores; skip the update if nothing changed."""
        snap = service.snapshot()
        if current and current.get("version") == snap.version:
            return no_update, no_update
        alerts = [slim_alert(a) for a in service.get_active_anomalies(ACTIVE_ALERTS_IN_STORE)]
        anomalies = {"version": snap.version, "alerts": alerts, "total": len(snap.alerts)}
        metrics = {**service.get_summary_metrics(), "version": snap.version}
        return anomalies, metrics

    @app.callback(Output("status-strip", "children"), Input("metrics-store", "data"))
    def show_status(metrics: dict[str, Any] | None):
        """Update the connection strip in the sidebar."""
        return _status_strip(metrics)

    @app.callback(
        Output("alert-toast", "is_open"),
        Output("alert-toast", "header"),
        Output("alert-toast", "children"),
        Output("alert-toast", "icon"),
        Input("sse-latest", "data"),
        prevent_initial_call=True,
    )
    def toast_on_alert(latest: dict[str, Any] | None):
        """Pop a toast for each CRITICAL or EMERGENCY alert pushed over SSE."""
        alert = (latest or {}).get("alert") or {}
        severity = str(alert.get("severity", "")).upper()
        if severity not in ("CRITICAL", "EMERGENCY"):
            return no_update, no_update, no_update, no_update
        header = f"{severity} · {alert.get('entity_id', 'unknown')}"
        body = (
            f"{alert.get('anomaly_type') or 'anomaly'} at "
            f"{format_timestamp(alert.get('timestamp'))} UTC. "
            f"{alert.get('recommended_action') or ''}"
        ).strip()
        return True, header, body, SEVERITY_BOOTSTRAP[severity]

    # ------------------------------------------------------------------ #
    # Poller lifecycle: start with the first request, stop at exit
    # ------------------------------------------------------------------ #
    if start_poller:
        guard = threading.Lock()

        @app.server.before_request
        def _start_poller() -> None:
            if not service.running:
                with guard:
                    service.start()

        atexit.register(service.stop)
    return app


def _default_app() -> Dash:
    return create_dash_app()


#: The instance ``python -m src dashboard`` runs (``app.run(...)``). Creating it makes no request.
app: Dash = _default_app()
server = app.server  # for gunicorn: ``gunicorn src.dashboard.app:server``

if __name__ == "__main__":  # pragma: no cover
    cfg_ = get_config()
    app.run(host=cfg_.dashboard.host, port=cfg_.dashboard.port, debug=cfg_.dashboard.debug)
