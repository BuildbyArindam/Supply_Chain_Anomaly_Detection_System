"""The dashboard's data layer (Day 18).

:class:`DashboardDataService` keeps one immutable :class:`DashboardSnapshot` of the live system and
refreshes it from a *source* on a background thread. Dash callbacks only read the latest snapshot,
so they never wait on I/O and many browsers cost nothing extra.

Sources return plain JSON-shaped dicts (the shape of ``AnomalyEvent.to_dict()`` and of the API
responses), so the service has the same behaviour in both set-ups:

* :class:`HttpSource` - the dashboard is its own process (Docker) and talks to the API.
* :class:`InProcessSource` - the dashboard runs inside the process that owns the
  :class:`~src.api.anomaly_store.AnomalyStore` and :class:`~src.kafka.feature_cache.FeatureCache`.

This module imports neither torch nor the model code, so a dashboard-only container stays small.

Time
----
"Today", "last 24 h" and the trend are measured from ``reference_now = max(wall clock, newest
event time)``. Event times can run ahead of the wall clock (10x simulation), so measuring from the
wall clock alone would hide the newest alerts. Windows are UTC.

Failure behaviour
-----------------
Each section (alerts, health, ...) is fetched independently. When a fetch fails the last good value
stays in the snapshot, the error is recorded in ``snapshot.errors`` and ``connected`` turns false
if the core sections fail; the poller backs off (1 s, 2 s, ... 10 s) and recovers by itself.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

import httpx

from src.dashboard.telemetry import all_entities, entity_timeseries
from src.dashboard.utils import hour_floor, hour_range, parse_timestamp
from src.utils.logger import get_logger

_log = get_logger(__name__)

SEVERITIES: Final[tuple[str, ...]] = ("INFO", "WARNING", "CRITICAL", "EMERGENCY")
SEVERITY_RANK: Final[dict[str, int]] = {name: i for i, name in enumerate(SEVERITIES)}
CRITICAL_RANK: Final = SEVERITY_RANK["CRITICAL"]
DEFAULT_ALERT_LIMIT: Final = 1000
DEFAULT_LATENCY_TARGET_MS: Final = 500.0
#: Consumer lag (messages) at which the status row turns amber / red (red = the Day 24 alert rule).
KAFKA_LAG_WARN: Final = 1_000
KAFKA_LAG_CRITICAL: Final = 10_000
MAX_BACKOFF_S: Final = 10.0
#: Most SKUs the inventory page reads history for (one request each).
MAX_INVENTORY_SKUS: Final = 100
#: Fields of an alert kept in the browser-side ``anomaly-store`` (details are fetched on click).
SLIM_FIELDS: Final[tuple[str, ...]] = (
    "event_id",
    "entity_id",
    "entity_type",
    "timestamp",
    "anomaly_type",
    "severity",
    "raw_value",
    "expected_range",
    "severity_score",
    "is_resolved",
    "false_positive_marked",
    "occurrence_count",
    "escalated_from",
)


def severity_rank(name: Any) -> int:
    """Rank of a severity name (``-1`` for an unknown one)."""
    return SEVERITY_RANK.get(str(name).upper(), -1)


def slim_alert(alert: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce an alert to the fields the tables and charts need."""
    return {key: alert.get(key) for key in SLIM_FIELDS}


def alert_status(alert: Mapping[str, Any]) -> str:
    """``false_positive``, ``resolved`` or ``open``; a false-positive verdict wins."""
    if alert.get("false_positive_marked"):
        return "false_positive"
    return "resolved" if alert.get("is_resolved") else "open"


def _is_active(alert: Mapping[str, Any]) -> bool:
    return not alert.get("is_resolved") and not alert.get("false_positive_marked")


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class DashboardSource(Protocol):
    """Where the live data comes from. All methods return JSON-shaped data and may raise."""

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        """Return the newest ``limit`` alerts, newest first (``AnomalyEvent.to_dict()`` shape)."""

    def alert(self, event_id: str) -> dict[str, Any] | None:
        """One full alert, or ``None``."""

    def entities(self) -> list[dict[str, Any]]:
        """Latest reading of every live entity (see ``telemetry.all_entities``)."""

    def timeseries(self, entity_id: str, lookback_seconds: float) -> dict[str, Any] | None:
        """Recent readings of one entity, or ``None`` if it is unknown."""

    def feedback_analytics(self) -> dict[str, Any]:
        """Output of the feedback loop's ``analytics()``."""

    def pipeline(self) -> dict[str, Any]:
        """Streaming engine stats (``latency``, ``consumer_lag``, ...); ``{}`` when unavailable."""

    def health(self) -> dict[str, Any]:
        """Return the ``/health`` payload."""

    def models(self) -> dict[str, Any]:
        """``{"registry": [...model types with versions...], "current": {...}}``."""

    def submit_feedback(self, body: dict[str, Any]) -> dict[str, Any]:
        """Post one analyst verdict (``FeedbackRequest`` fields); raises when it is rejected."""

    def close(self) -> None:
        """Release resources."""


class HttpSource:
    """Reads everything from the API over HTTP (``dashboard.api_url``)."""

    def __init__(self, base_url: str, *, timeout_s: float = 3.0) -> None:
        """Create the source; no request is made until a method is called."""
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def _get(self, path: str, **params: Any) -> Any:
        response = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        response.raise_for_status()
        return response.json()

    def _get_or_none(self, path: str, **params: Any) -> Any | None:
        try:
            return self._get(path, **params)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                return None
            raise

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        """See :meth:`DashboardSource.alerts`."""
        return list(self._get("/api/v1/anomalies", limit=min(limit, 1000))["items"])

    def alert(self, event_id: str) -> dict[str, Any] | None:
        """See :meth:`DashboardSource.alert`."""
        return self._get_or_none(f"/api/v1/anomalies/{event_id}")  # type: ignore[no-any-return]

    def entities(self) -> list[dict[str, Any]]:
        """See :meth:`DashboardSource.entities`."""
        return list(self._get("/api/v1/telemetry/entities")["items"])

    def timeseries(self, entity_id: str, lookback_seconds: float) -> dict[str, Any] | None:
        """See :meth:`DashboardSource.timeseries`."""
        return self._get_or_none(  # type: ignore[no-any-return]
            f"/api/v1/telemetry/{entity_id}", lookback_seconds=lookback_seconds
        )

    def feedback_analytics(self) -> dict[str, Any]:
        """See :meth:`DashboardSource.feedback_analytics`."""
        return self._get("/api/v1/feedback/analytics")  # type: ignore[no-any-return]

    def pipeline(self) -> dict[str, Any]:
        """Stats of the streaming runner; ``{}`` when the API runs without one (404)."""
        return self._get_or_none("/api/v1/pipeline/stats") or {}

    def health(self) -> dict[str, Any]:
        """See :meth:`DashboardSource.health`."""
        return self._get("/health")  # type: ignore[no-any-return]

    def models(self) -> dict[str, Any]:
        """See :meth:`DashboardSource.models`."""
        return {
            "registry": self._get("/api/v1/models")["models"],
            "current": self._get("/api/v1/models/current"),
        }

    def submit_feedback(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``/api/v1/feedback``; a 4xx/5xx raises :class:`httpx.HTTPStatusError`."""
        response = self._client.post("/api/v1/feedback", json=body)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class InProcessSource:
    """Reads the live objects directly (dashboard and pipeline in one process).

    Args:
        service: The :class:`~src.api.service.InferenceService` (alert store, feature cache,
            feedback processor, models).
        engine: Optional streaming engine, for latency and consumer-lag figures.
    """

    def __init__(self, service: Any, engine: Any | None = None) -> None:
        """Wrap ``service`` (and optionally ``engine``)."""
        self.service = service
        self.engine = engine

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        """See :meth:`DashboardSource.alerts`."""
        return [e.to_dict() for e in self.service.anomaly_store.filter()[:limit]]

    def alert(self, event_id: str) -> dict[str, Any] | None:
        """See :meth:`DashboardSource.alert`."""
        found = self.service.anomaly_store.get(event_id)
        return found.to_dict() if found is not None else None

    def entities(self) -> list[dict[str, Any]]:
        """See :meth:`DashboardSource.entities`."""
        return all_entities(self.service.cache)

    def timeseries(self, entity_id: str, lookback_seconds: float) -> dict[str, Any] | None:
        """See :meth:`DashboardSource.timeseries`."""
        return entity_timeseries(self.service.cache, entity_id, lookback_seconds)

    def feedback_analytics(self) -> dict[str, Any]:
        """See :meth:`DashboardSource.feedback_analytics`."""
        return dict(self.service.feedback_processor.analytics())

    def pipeline(self) -> dict[str, Any]:
        """Latency percentiles, engine counters and consumer lag (``{}`` without an engine)."""
        engine = self.engine
        if engine is None:
            return {}
        consumer = engine.consumer_group
        return {
            "engine": engine.stats().to_dict(),
            "latency": engine.latency.snapshot().to_dict(),
            "latency_target_ms": engine.latency.target_p99_s * 1000.0,
            "consumer_lag": consumer.total_lag() if consumer is not None else None,
        }

    def health(self) -> dict[str, Any]:
        """Return the same fields as the API's ``/health``."""
        svc = self.service
        loaded = svc.models.is_loaded
        degraded = loaded and svc.models.active.degraded
        reason = (
            "models not loaded"
            if not loaded
            else "trained models unavailable; serving statistical fallback" if degraded else None
        )
        return {
            "status": "degraded" if reason else "ok",
            "models_loaded": loaded,
            "degraded_reason": reason,
            "checks": {"models": "ok" if loaded and not degraded else "degraded"},
            "stored_anomalies": len(svc.anomaly_store),
        }

    def models(self) -> dict[str, Any]:
        """See :meth:`DashboardSource.models` (reuses the API router's builders)."""
        from src.api.routers.models import current_models, list_models

        return {
            "registry": list_models(self.service).model_dump(mode="json")["models"],
            "current": current_models(self.service).model_dump(mode="json"),
        }

    def submit_feedback(self, body: dict[str, Any]) -> dict[str, Any]:
        """Record a verdict through the service (same validation as the API endpoint)."""
        from src.api.schemas import FeedbackRequest

        outcome = self.service.record_feedback(FeedbackRequest(**body))
        return {
            "feedback_id": outcome.event.event_id,
            "anomaly_event_id": outcome.event.anomaly_event_id,
            "replaced_previous": outcome.replaced,
        }

    def close(self) -> None:
        """Nothing to release."""


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DashboardSnapshot:
    """The live state at one instant. Immutable: replaced, never mutated."""

    version: int = 0
    taken_at: float = 0.0
    reference_now: datetime = field(default_factory=lambda: datetime.now(UTC))
    connected: bool = False
    errors: Mapping[str, str] = field(default_factory=dict)
    alerts: tuple[dict[str, Any], ...] = ()
    entities: tuple[dict[str, Any], ...] = ()
    health: Mapping[str, Any] = field(default_factory=dict)
    feedback: Mapping[str, Any] = field(default_factory=dict)
    pipeline: Mapping[str, Any] = field(default_factory=dict)
    models: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _Section:
    name: str
    fetch: Callable[[], Any]
    interval_s: float
    core: bool = False
    last_run: float = 0.0
    ok: bool = False


class DashboardDataService:
    """Aggregates the live metrics for every dashboard page.

    Args:
        source: Where data comes from (:class:`HttpSource` or :class:`InProcessSource`).
        poll_interval_s: Poll tick; sections run at their own cadence (alerts, health and
            pipeline every tick, feedback every 5 s, models every 10 s).
        alert_limit: Newest alerts kept in the snapshot (the trend and "today" count see these).
        active_window_hours: Alerts older than this no longer count as active.
        clock: Wall-clock function (injectable for tests).
    """

    def __init__(
        self,
        source: DashboardSource,
        *,
        poll_interval_s: float = 1.0,
        alert_limit: int = DEFAULT_ALERT_LIMIT,
        active_window_hours: float = 24.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create the service; call :meth:`start` to begin polling."""
        if poll_interval_s <= 0 or alert_limit < 1:
            raise ValueError("poll_interval_s and alert_limit must be > 0")
        self.source = source
        self.poll_interval_s = poll_interval_s
        self.alert_limit = alert_limit
        self.active_window = timedelta(hours=active_window_hours)
        self._clock = clock or (lambda: datetime.now(UTC))
        tick = poll_interval_s
        self._sections = [
            _Section("alerts", lambda: source.alerts(alert_limit), tick, core=True),
            _Section("health", source.health, tick, core=True),
            _Section("entities", source.entities, tick),
            _Section("pipeline", source.pipeline, tick),
            _Section("feedback", source.feedback_analytics, max(tick, 5.0)),
            _Section("models", source.models, max(tick, 10.0)),
        ]
        self._values: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._snapshot = DashboardSnapshot(reference_now=self._clock())
        self._version = 0
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._series_cache: dict[tuple[str, float], tuple[float, dict[str, Any] | None]] = {}
        self._series_lock = threading.Lock()
        self._inventory_cache: tuple[float, dict[str, list[dict[str, Any]]]] | None = None
        self._inventory_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        """Whether the background poller is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background poller (idempotent). The first refresh happens at once."""
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="scad-dashboard-poller", daemon=True
            )
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the poller and close the source."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
        self._thread = None
        try:
            self.source.close()
        except Exception as error:  # noqa: BLE001 - shutdown must not fail
            _log.warning("dashboard_source_close_failed", error=repr(error))

    def refresh(self) -> DashboardSnapshot:
        """Fetch every section now (ignoring cadence) and return the new snapshot."""
        self._poll(force=True)
        return self.snapshot()

    def snapshot(self) -> DashboardSnapshot:
        """Return the latest snapshot (cheap; never does I/O)."""
        return self._snapshot

    def _run(self) -> None:
        delay = 0.0
        failures = 0
        while not self._stop.wait(delay):
            core_ok = self._poll(force=failures > 0 or self._version == 0)
            failures = 0 if core_ok else failures + 1
            delay = (
                self.poll_interval_s
                if core_ok
                else min(self.poll_interval_s * 2**failures, MAX_BACKOFF_S)
            )

    def _poll(self, *, force: bool) -> bool:
        """Run the due sections; return whether every core section is healthy."""
        with self._poll_lock:  # the poller thread and refresh() must not interleave
            return self._poll_locked(force=force)

    def _poll_locked(self, *, force: bool) -> bool:
        now = time.monotonic()
        changed = False
        for section in self._sections:
            if not force and now - section.last_run < section.interval_s:
                continue
            section.last_run = now
            try:
                value = section.fetch()
            except Exception as error:  # noqa: BLE001 - any source failure is a degraded section
                section.ok = False
                message = f"{type(error).__name__}: {error}"
                if self._errors.get(section.name) != message:
                    _log.warning("dashboard_fetch_failed", section=section.name, error=message)
                    changed = True
                self._errors[section.name] = message
                continue
            section.ok = True
            if section.name in self._errors:
                del self._errors[section.name]
                changed = True
                _log.info("dashboard_fetch_recovered", section=section.name)
            if self._values.get(section.name) != value:
                self._values[section.name] = value
                changed = True
        core_ok = all(s.ok for s in self._sections if s.core)
        if changed or self._snapshot.connected != core_ok:
            self._publish(core_ok)
        return core_ok

    def _publish(self, connected: bool) -> None:
        values = self._values
        alerts = tuple(values.get("alerts", ()))
        entities = tuple(values.get("entities", ()))
        reference = self._clock()
        for stamp in ([alerts[0].get("timestamp")] if alerts else []) + [
            e.get("timestamp") for e in entities
        ]:
            moment = parse_timestamp(stamp)
            if moment is not None and moment > reference:
                reference = moment
        self._version += 1
        self._snapshot = DashboardSnapshot(
            version=self._version,
            taken_at=time.time(),
            reference_now=reference,
            connected=connected,
            errors=dict(self._errors),
            alerts=alerts,
            entities=entities,
            health=values.get("health", {}),
            feedback=values.get("feedback", {}),
            pipeline=values.get("pipeline", {}),
            models=values.get("models", {}),
        )

    # ------------------------------------------------------------------ #
    # Queries (all read the snapshot)
    # ------------------------------------------------------------------ #
    def get_active_anomalies(self, n: int = 50) -> list[dict[str, Any]]:
        """Up to ``n`` unresolved, not-false-positive alerts, most severe first, then newest.

        Only alerts within ``active_window_hours`` of the reference time count as active.
        """
        snap = self.snapshot()
        cutoff = snap.reference_now - self.active_window
        active = [
            a
            for a in snap.alerts
            if _is_active(a) and (parse_timestamp(a.get("timestamp")) or cutoff) >= cutoff
        ]
        active.sort(key=lambda a: a.get("timestamp") or "", reverse=True)
        active.sort(key=lambda a: severity_rank(a.get("severity")), reverse=True)
        return active[: max(0, n)]

    def get_recent_alerts(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the ``n`` newest alerts of any status, newest first (the overview's live feed)."""
        rows = sorted(self.snapshot().alerts, key=lambda a: a.get("timestamp") or "", reverse=True)
        return [slim_alert(a) for a in rows[: max(0, n)]]

    def get_entity_alerts(
        self, entity_id: str, since: datetime | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Alerts of one entity, oldest first, for the overlay on its time-series chart.

        False-positive alerts are left out (an analyst said they were not anomalies); resolved
        ones stay (they were real). ``since`` drops alerts older than that time.
        """
        found = []
        for alert in self.snapshot().alerts:
            if alert.get("entity_id") != entity_id or alert.get("false_positive_marked"):
                continue
            moment = parse_timestamp(alert.get("timestamp"))
            if moment is None or (since is not None and moment < since):
                continue
            found.append((moment, slim_alert(alert)))
        found.sort(key=lambda pair: pair[0])
        return [alert for _, alert in found[-max(0, limit) :]]

    def get_status_badges(self) -> list[dict[str, str]]:
        """State of the four system components shown on the overview's status row.

        Each item is ``{"name", "state", "detail"}`` with ``state`` one of ``ok``, ``warn``,
        ``error`` or ``unknown`` (not reported by this deployment).

        * Kafka: the API's Kafka probe (``/health`` ``checks.kafka``).
        * Models: trained models serving = ok, statistical fallback = warn, none loaded = error.
        * Dashboard: the link from this dashboard to the API (error when the core data is stale).
        * Kafka lag: total consumer lag against ``KAFKA_LAG_WARN`` / ``KAFKA_LAG_CRITICAL``.
        """
        snap = self.snapshot()
        checks = (snap.health or {}).get("checks") or {}

        kafka = checks.get("kafka")
        if not snap.connected and not snap.health:
            kafka_state, kafka_detail = "unknown", "API unreachable"
        elif kafka is None:
            kafka_state, kafka_detail = "unknown", "not monitored"
        else:
            kafka_state, kafka_detail = ("ok", "reachable") if kafka == "ok" else ("error", kafka)

        if not snap.health or not snap.health.get("models_loaded", False):
            models_state, models_detail = "error", "not loaded"
        elif checks.get("models") == "ok":
            models_state, models_detail = "ok", "trained models"
        else:
            models_state, models_detail = "warn", "statistical fallback"

        if not snap.connected:
            dash_state, dash_detail = "error", "API unreachable"
        elif snap.errors:
            dash_state, dash_detail = "warn", f"no data: {', '.join(sorted(snap.errors))}"
        else:
            dash_state, dash_detail = "ok", "live"

        lag = snap.pipeline.get("consumer_lag")
        if isinstance(lag, bool) or not isinstance(lag, int | float):
            lag_state, lag_detail = "unknown", "not reported"
        else:
            lag_state = (
                "error" if lag >= KAFKA_LAG_CRITICAL else "warn" if lag >= KAFKA_LAG_WARN else "ok"
            )
            lag_detail = f"{int(lag):,} msgs"
        return [
            {"name": "Kafka", "state": kafka_state, "detail": kafka_detail},
            {"name": "Models", "state": models_state, "detail": models_detail},
            {"name": "Dashboard", "state": dash_state, "detail": dash_detail},
            {"name": "Kafka lag", "state": lag_state, "detail": lag_detail},
        ]

    def get_alert(self, event_id: str) -> dict[str, Any] | None:
        """Fetch the full alert (with contributing features), fetched from the source on demand."""
        try:
            return self.source.alert(event_id)
        except Exception as error:  # noqa: BLE001 - a failed lookup shows "not found", not a crash
            _log.warning("dashboard_alert_lookup_failed", event_id=event_id, error=repr(error))
            return None

    def get_sensor_timeseries(
        self, entity_id: str, lookback_seconds: float = 300.0
    ) -> dict[str, Any] | None:
        """Recent readings of one entity from the feature cache, or ``None`` if unavailable.

        Results are cached for 0.5 s so several browsers asking for the same series cost one
        read.
        """
        key = (entity_id, float(lookback_seconds))
        now = time.monotonic()
        with self._series_lock:
            hit = self._series_cache.get(key)
            if hit is not None and now - hit[0] < 0.5:
                return hit[1]
        try:
            series = self.source.timeseries(entity_id, lookback_seconds)
        except Exception as error:  # noqa: BLE001
            _log.warning("dashboard_timeseries_failed", entity_id=entity_id, error=repr(error))
            return None
        with self._series_lock:
            if len(self._series_cache) > 256:
                self._series_cache.clear()
            self._series_cache[key] = (now, series)
        return series

    def get_entities(self, source_type: str | None = None) -> list[dict[str, Any]]:
        """Latest reading of every live entity, optionally of one source type."""
        rows = self.snapshot().entities
        return [e for e in rows if source_type in (None, e.get("source_type"))]

    def query_alerts(
        self,
        *,
        severities: Sequence[str] | None = None,
        anomaly_types: Sequence[str] | None = None,
        entity_types: Sequence[str] | None = None,
        status: str | None = None,
        window_hours: float | None = None,
        entity_id: str | None = None,
        limit: int = DEFAULT_ALERT_LIMIT,
    ) -> list[dict[str, Any]]:
        """Filter the snapshot's alerts (full records), newest first.

        Filtering runs here on the server, so the browser only receives the matching rows. An
        empty or ``None`` list for a filter means "any". The service holds the newest
        ``alert_limit`` alerts, so older ones are out of reach.

        Args:
            severities: Keep these severity names.
            anomaly_types: Keep these anomaly types (``point`` / ``contextual`` / ``collective``).
            entity_types: Keep these entity types (``warehouse_sensor`` / ``vehicle`` /
                ``inventory_sku``).
            status: ``open``, ``false_positive`` or ``resolved``; ``None`` or ``all`` keeps all.
            window_hours: Only alerts from the last N hours (measured from the reference time).
            entity_id: Only this entity.
            limit: Most rows returned.
        """
        snap = self.snapshot()
        cutoff = snap.reference_now - timedelta(hours=window_hours) if window_hours else None
        wanted_sev = {str(v).upper() for v in severities or ()}
        wanted_type = set(anomaly_types or ())
        wanted_entity = set(entity_types or ())
        rows: list[tuple[datetime, dict[str, Any]]] = []
        for alert in snap.alerts:
            if wanted_sev and str(alert.get("severity", "")).upper() not in wanted_sev:
                continue
            if wanted_type and alert.get("anomaly_type") not in wanted_type:
                continue
            if wanted_entity and alert.get("entity_type") not in wanted_entity:
                continue
            if entity_id is not None and alert.get("entity_id") != entity_id:
                continue
            if status not in (None, "", "all") and alert_status(alert) != status:
                continue
            moment = parse_timestamp(alert.get("timestamp"))
            if cutoff is not None and (moment is None or moment < cutoff):
                continue
            rows.append((moment or datetime.min.replace(tzinfo=UTC), alert))
        rows.sort(key=lambda pair: pair[0], reverse=True)
        return [alert for _, alert in rows[: max(0, limit)]]

    def submit_feedback(
        self,
        event_ids: Sequence[str],
        feedback_type: str,
        *,
        analyst_id: str = "dashboard",
        notes: str | None = None,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """Send one verdict per alert to the API (there is no batch endpoint) and report each.

        A failure on one alert never stops the others. The snapshot is refreshed afterwards so
        the tables show the new status at once.

        Returns:
            ``{"requested", "ok", "failed": [{"event_id", "error"}], "feedback_type"}``.
        """
        unique = list(dict.fromkeys(str(e) for e in event_ids if e))

        def send(event_id: str) -> tuple[str, str | None]:
            body: dict[str, Any] = {
                "anomaly_event_id": event_id,
                "feedback_type": feedback_type,
                "analyst_id": analyst_id,
            }
            if notes:
                body["notes"] = notes
            try:
                self.source.submit_feedback(body)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                reason = "alert no longer stored" if status == 404 else f"HTTP {status}"
                return event_id, reason
            except Exception as error:  # noqa: BLE001 - one bad row must not stop the batch
                return event_id, f"{type(error).__name__}: {error}"
            return event_id, None

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique) or 1))) as pool:
            results = list(pool.map(send, unique))
        failed = [{"event_id": eid, "error": err} for eid, err in results if err is not None]
        if len(failed) < len(unique):
            try:
                self.refresh()
            except Exception as error:  # noqa: BLE001 - the verdicts are stored either way
                _log.warning("dashboard_refresh_after_feedback_failed", error=repr(error))
        return {
            "requested": len(unique),
            "ok": len(unique) - len(failed),
            "failed": failed,
            "feedback_type": feedback_type,
        }

    def get_route(self, route_id: str | None) -> dict[str, Any] | None:
        """Planned route (``origin``, ``destination``, ``waypoints``) of a built-in route id.

        Returns ``None`` for an unknown route; the vehicle map then shows only the GPS trail.
        """
        if not route_id:
            return None
        try:
            from src.data_simulation.vehicle_simulator import default_routes

            return default_routes().get(str(route_id))
        except ImportError:  # pragma: no cover - the simulator ships in every image
            return None

    def get_inventory_series(
        self, window_s: float = 3600.0, max_age_s: float = 5.0
    ) -> dict[str, list[dict[str, Any]]]:
        """Last ``window_s`` seconds of transactions of every live SKU, by SKU id.

        One read per SKU, so the result is cached for ``max_age_s`` seconds and shared by every
        browser. Each point is ``{"t", "v" (stock level), "d" (change), "tx" (type)}``.
        """
        now = time.monotonic()
        with self._inventory_lock:
            cached = self._inventory_cache
            if cached is not None and now - cached[0] < max_age_s:
                return cached[1]
            result: dict[str, list[dict[str, Any]]] = {}
            for entity in self.get_entities("inventory")[:MAX_INVENTORY_SKUS]:
                series = self.get_sensor_timeseries(str(entity["entity_id"]), window_s)
                result[str(entity["entity_id"])] = list((series or {}).get("points") or [])
            self._inventory_cache = (now, result)
            return result

    def get_alert_trend(self, hours: int = 24) -> list[dict[str, Any]]:
        """Alerts per hour by severity for the last ``hours`` hours (zero-filled, oldest first).

        Each row is ``{"hour": ISO start, "INFO": n, "WARNING": n, "CRITICAL": n,
        "EMERGENCY": n, "total": n}``. The trend sees the newest ``alert_limit`` alerts.
        """
        snap = self.snapshot()
        buckets = hour_range(snap.reference_now, max(1, hours))
        rows = {b: dict.fromkeys(SEVERITIES, 0) for b in buckets}
        for alert in snap.alerts:
            moment = parse_timestamp(alert.get("timestamp"))
            name = str(alert.get("severity", "")).upper()
            if moment is not None and name in SEVERITY_RANK:
                bucket = hour_floor(moment)
                if bucket in rows:
                    rows[bucket][name] += 1
        return [
            {"hour": b.isoformat().replace("+00:00", "Z"), **counts, "total": sum(counts.values())}
            for b, counts in rows.items()
        ]

    def get_summary_metrics(self) -> dict[str, Any]:
        """Headline numbers of the overview page.

        Keys: ``total_alerts_today`` (UTC day), ``alerts_last_hour``, ``active_alerts`` (all
        active alerts), ``active_critical`` (active alerts at CRITICAL or above), ``critical_24h``
        (CRITICAL or EMERGENCY alerts of the last 24 h, resolved or not), ``fp_rate_24h``
        (``None`` without feedback),
        ``model_health_score`` (0-1, see :meth:`model_health`), ``avg_latency_p99`` (ms, ``None``
        when the streaming engine is not reporting), ``stored_alerts``, ``connected``,
        ``api_status``, ``drift_alert_active``, ``errors`` and ``updated_at``.
        """
        snap = self.snapshot()
        ref = snap.reference_now
        stamps = [parse_timestamp(a.get("timestamp")) for a in snap.alerts]
        today = sum(1 for m in stamps if m is not None and m.date() == ref.date())
        last_hour = sum(1 for m in stamps if m is not None and ref - m <= timedelta(hours=1))
        active = self.get_active_anomalies(len(snap.alerts))
        active_critical = sum(
            1 for a in active if severity_rank(a.get("severity")) >= CRITICAL_RANK
        )
        critical_24h = sum(
            1
            for a, m in zip(snap.alerts, stamps, strict=True)
            if severity_rank(a.get("severity")) >= CRITICAL_RANK
            and m is not None
            and ref - m <= timedelta(hours=24)
        )
        window = (snap.feedback.get("windows") or {}).get("last_24h") or {}
        fp_rate = window.get("false_positive_rate") if window.get("total") else None
        p99 = (snap.pipeline.get("latency") or {}).get("p99_ms")
        health = self.model_health()
        return {
            "total_alerts_today": today,
            "alerts_last_hour": last_hour,
            "active_alerts": len(active),
            "active_critical": active_critical,
            "critical_24h": critical_24h,
            "fp_rate_24h": fp_rate,
            "model_health_score": health["score"],
            "avg_latency_p99": p99 if p99 else None,
            "stored_alerts": len(snap.alerts),
            "connected": snap.connected,
            "api_status": snap.health.get("status", "unreachable"),
            "drift_alert_active": health["drift_alert_active"],
            "errors": dict(snap.errors),
            "updated_at": snap.taken_at,
        }

    def model_health(self) -> dict[str, Any]:
        """Composite model health in ``[0, 1]`` with its components.

        ``score = 0.40 * availability + 0.25 * feedback_quality + 0.20 * latency + 0.15 * drift``

        * availability: 1 trained models serving, 0.5 statistical fallback only, 0 not loaded or
          the API unreachable
        * feedback_quality: ``1 - fp_rate_24h / 0.30`` (floored at 0); 1 while there is no verdict
        * latency: 1 while p99 is within target, then falling to 0 at twice the target; 1 if
          unknown
        * drift: 0 if the feedback drift detector fired within the last hour, else 1
        """
        snap = self.snapshot()
        health = snap.health
        if not snap.connected and not health or not health.get("models_loaded", False):
            availability = 0.0
        else:
            availability = 1.0 if health.get("checks", {}).get("models") == "ok" else 0.5

        window = (snap.feedback.get("windows") or {}).get("last_24h") or {}
        fp_rate = window.get("false_positive_rate") if window.get("total") else None
        feedback_quality = 1.0 if fp_rate is None else max(0.0, 1.0 - float(fp_rate) / 0.30)

        p99 = (snap.pipeline.get("latency") or {}).get("p99_ms")
        target = float(snap.pipeline.get("latency_target_ms") or DEFAULT_LATENCY_TARGET_MS)
        latency = (
            1.0 if not p99 or p99 <= target else max(0.0, 1.0 - (float(p99) - target) / target)
        )

        last_alert = (snap.feedback.get("drift") or {}).get("last_alert") or {}
        fired = parse_timestamp(last_alert.get("detected_at") or last_alert.get("timestamp"))
        drift_active = fired is not None and snap.reference_now - fired <= timedelta(hours=1)
        drift = 0.0 if drift_active else 1.0

        score = 0.40 * availability + 0.25 * feedback_quality + 0.20 * latency + 0.15 * drift
        return {
            "score": round(score, 4),
            "components": {
                "availability": availability,
                "feedback_quality": feedback_quality,
                "latency": latency,
                "drift": drift,
            },
            "drift_alert_active": drift_active,
        }

    def get_model_performance(self) -> dict[str, Any]:
        """Latest registry metrics of every model plus the active configuration.

        Returns ``{"models": [{"model_type", "version", "active", "created_at", "metrics"}],
        "ensemble": <metrics of the ensemble meta-learner or None>, "current": {...},
        "feedback_fp_rate": {...}}``. For each model type the active version is reported, else the
        newest. Metrics are whatever training saved (``f1``, ``precision``, ``recall``, ...).
        """
        snap = self.snapshot()
        registry: Sequence[Mapping[str, Any]] = snap.models.get("registry", ())
        rows = []
        for entry in registry:
            versions = entry.get("versions") or []
            chosen = next(
                (v for v in versions if v.get("active")), versions[0] if versions else None
            )
            if chosen is None:
                continue
            rows.append(
                {
                    "model_type": entry.get("model_type"),
                    "version": chosen.get("version"),
                    "active": bool(chosen.get("active")),
                    "created_at": chosen.get("created_at"),
                    "metrics": dict(chosen.get("metrics") or {}),
                }
            )
        ensemble = next(
            (r["metrics"] for r in rows if str(r["model_type"]).startswith("ensemble")), None
        )
        return {
            "models": rows,
            "ensemble": ensemble,
            "current": dict(snap.models.get("current") or {}),
            "feedback_fp_rate": dict(snap.feedback.get("windows") or {}),
        }

    def get_status(self) -> dict[str, Any]:
        """Describe the connection state for the status strip (connected, errors, age)."""
        snap = self.snapshot()
        return {
            "connected": snap.connected,
            "errors": dict(snap.errors),
            "age_s": max(0.0, time.time() - snap.taken_at) if snap.taken_at else None,
            "version": snap.version,
        }
