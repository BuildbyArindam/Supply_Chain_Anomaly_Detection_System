"""Inference service behind the API: model hot-swap and the per-event scoring path (Day 14).

Two classes:

* :class:`ModelManager` builds one :class:`~src.models.ensemble_detector.EnsembleDetector` per
  source type from the :class:`~src.models.model_registry.ModelRegistry` and publishes the result
  as an immutable :class:`ActiveModels` snapshot. :meth:`ModelManager.reload` builds a new
  snapshot *beside* the serving one and swaps a single reference, so in-flight requests finish on
  the old models and the next request sees the new ones: no restart, no downtime. If the build
  fails the old snapshot keeps serving.
* :class:`InferenceService` owns the streaming state (aggregator, feature cache, feature pipeline,
  stores, severity scorer, de-duplicator) and turns one request into one response::

      event -> WindowedAggregator -> FeatureCache -> FeaturePipeline -> EnsembleDetector
            -> SeverityScorer -> AnomalyEvent -> AlertDeduplicator -> AnomalyStore

Graceful degradation: the statistical detector needs no trained artifact, so a registry that is
empty (or whose artifacts are corrupt) leaves the service running on it alone; ``/health`` then
reports ``degraded``. The statistical detector instance is **kept across reloads** because it is
an online learner: replacing it would throw away every entity's learned baseline.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src.api.anomaly_store import AnomalyStore
from src.api.feedback_store import FeedbackRecord, FeedbackStore
from src.api.schemas import AnomalyDetectionResponse, FeedbackRequest, SensorEventRequest
from src.kafka.feature_cache import FeatureCache
from src.kafka.stream_processor import WindowedAggregator
from src.models.ensemble_detector import (
    ISOLATION_FOREST,
    LSTM_AUTOENCODER,
    LSTM_CLASSIFIER,
    STATISTICAL,
    AlertDeduplicator,
    EnsembleDetector,
    EntityContext,
    SeverityScorer,
    build_anomaly_event,
)
from src.models.feature_engineering import FeaturePipeline
from src.models.model_registry import ModelRegistry
from src.models.statistical_detector import (
    CUSUMDetector,
    IQRDetector,
    UnifiedStatisticalDetector,
    ZScoreDetector,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config import AppConfig

_log = get_logger(__name__)

SOURCES: Final[tuple[str, ...]] = ("warehouse", "vehicle", "inventory")
_ISOLATION_FOREST_PREFIX: Final = "isolation_forest"
_LSTM_AE_TYPE: Final = "lstm_autoencoder"
_LSTM_CLS_TYPE: Final = "lstm_classifier"
_META_TYPE: Final = "ensemble_meta_learner"


class EventNotFoundError(LookupError):
    """The referenced alert is not in the anomaly store (unknown id or already evicted)."""


class ReloadError(RuntimeError):
    """A model reload failed; the previous models keep serving."""


# --------------------------------------------------------------------------- #
# Model manager (hot swap)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActiveModels:
    """An immutable snapshot of what is serving traffic.

    Attributes:
        generation: Counter incremented by every successful build.
        loaded_at: When the snapshot was built.
        ensembles: One ensemble per source type.
        versions: Registry version of every loaded model type.
        load_errors: Model types that exist in the registry but failed to load.
        degraded: True when no trained model is serving (statistical fallback only).
    """

    generation: int
    loaded_at: datetime
    ensembles: Mapping[str, EnsembleDetector]
    versions: Mapping[str, str] = field(default_factory=dict)
    load_errors: Mapping[str, str] = field(default_factory=dict)
    degraded: bool = True


class ModelManager:
    """Builds and atomically swaps :class:`ActiveModels` from the registry."""

    def __init__(
        self,
        cfg: AppConfig,
        registry: ModelRegistry,
        statistical: UnifiedStatisticalDetector,
        sequences: Any | None = None,
    ) -> None:
        """Create the manager (nothing is loaded until :meth:`load`).

        Args:
            cfg: Application config (ensemble weights and folds).
            registry: Where trained models live.
            statistical: The shared online statistical detector (survives reloads).
            sequences: Sequence source (``FeaturePipeline.sequences``) the LSTM detectors read.
        """
        self._cfg = cfg
        self.registry = registry
        self._statistical = statistical
        self._sequences = sequences
        self._reload_lock = threading.Lock()
        self._active: ActiveModels | None = None

    @property
    def active(self) -> ActiveModels:
        """The snapshot currently serving (call once per request and keep the reference)."""
        if self._active is None:
            raise RuntimeError("models not loaded; call load() first")
        return self._active

    @property
    def is_loaded(self) -> bool:
        """Whether a snapshot has been published."""
        return self._active is not None

    def load(self) -> ActiveModels:
        """Build and publish the first snapshot (never raises: falls back to statistical)."""
        with self._reload_lock:
            self._active = self._build(generation=1)
            return self._active

    def reload(self) -> ActiveModels:
        """Build a new snapshot from the registry and swap it in.

        All-or-nothing: if a model type that loaded before now fails to load (for example a
        corrupt newest version) the reload is rejected and the previous snapshot keeps serving.

        Raises:
            ReloadError: The build failed or lost a model; the previous snapshot keeps serving.
        """
        with self._reload_lock:
            previous = self._active.generation if self._active else 0
            try:
                fresh = self._build(generation=previous + 1)
            except Exception as error:  # noqa: BLE001 - the old models must keep serving
                _log.error("model_reload_failed", error=repr(error))
                raise ReloadError(f"reload failed: {error}") from error
            if self._active is not None:
                # Errors that already existed in the serving snapshot are tolerated (that model
                # was not serving anyway); a *new* failure must not silently drop a working model.
                new_errors = sorted(set(fresh.load_errors) - set(self._active.load_errors))
                if new_errors:
                    detail = {k: fresh.load_errors[k] for k in new_errors}
                    _log.error("model_reload_rejected", errors=detail)
                    raise ReloadError(f"reload rejected, could not load {detail}")
            self._active = fresh  # single reference assignment: the hot swap
            _log.info(
                "models_reloaded",
                generation=fresh.generation,
                versions=dict(fresh.versions),
                degraded=fresh.degraded,
                load_errors=list(fresh.load_errors),
            )
            return fresh

    # ------------------------------------------------------------------ #
    def _build(self, generation: int) -> ActiveModels:
        versions: dict[str, str] = {}
        errors: dict[str, str] = {}
        members: dict[str, dict[str, Any]] = {s: {STATISTICAL: self._statistical} for s in SOURCES}

        try:
            model_types = self.registry.list_model_types()
        except OSError as error:
            errors["registry"] = repr(error)
            model_types = []

        for model_type in model_types:
            try:
                if model_type.startswith(_ISOLATION_FOREST_PREFIX):
                    loaded = self.registry.get_latest_model(model_type)
                    source = (
                        getattr(loaded.model, "source_type", None) or model_type.rsplit("_", 1)[-1]
                    )
                    if source in members:
                        members[source][ISOLATION_FOREST] = loaded.model
                        versions[model_type] = loaded.version.version
                elif model_type == _LSTM_AE_TYPE:
                    from src.models.lstm_autoencoder import LSTMAutoencoderDetector

                    det = LSTMAutoencoderDetector.load(self.registry, sequences=self._sequences)
                    members.get(det.source_type, {})[LSTM_AUTOENCODER] = det
                    versions[model_type] = self.registry.list_versions(model_type)[0].version
                elif model_type == _LSTM_CLS_TYPE:
                    from src.models.lstm_classifier import LSTMClassifierDetector

                    cls_det = LSTMClassifierDetector.load(self.registry, sequences=self._sequences)
                    members.get(cls_det.source_type, {})[LSTM_CLASSIFIER] = cls_det
                    versions[model_type] = self.registry.list_versions(model_type)[0].version
            except Exception as error:  # noqa: BLE001 - one bad artifact must not stop the rest
                errors[model_type] = repr(error)
                _log.warning("model_load_failed", model_type=model_type, error=repr(error))

        ensembles: dict[str, EnsembleDetector] = {}
        meta_version: str | None = None
        for source, source_members in members.items():
            ensemble = EnsembleDetector.from_config(self._cfg.models.ensemble, **source_members)
            if _META_TYPE in model_types:
                try:
                    ensemble.load_meta_learner(self.registry)
                    meta_version = self.registry.list_versions(_META_TYPE)[0].version
                except Exception as error:  # noqa: BLE001 - falls back to the weighted average
                    errors[_META_TYPE] = repr(error)
            ensembles[source] = ensemble
        if meta_version:
            versions[_META_TYPE] = meta_version

        trained = any(
            len([m for m in ens.available_members if m != STATISTICAL]) > 0
            for ens in ensembles.values()
        )
        return ActiveModels(
            generation=generation,
            loaded_at=datetime.now(UTC),
            ensembles=ensembles,
            versions=versions,
            load_errors=errors,
            degraded=not trained,
        )


# --------------------------------------------------------------------------- #
# Inference service
# --------------------------------------------------------------------------- #
def build_statistical_detector(cfg: AppConfig) -> UnifiedStatisticalDetector:
    """Create the online statistical detector from ``config.models.statistical``."""
    s = cfg.models.statistical
    return UnifiedStatisticalDetector(
        z_weight=s.z_weight,
        iqr_weight=s.iqr_weight,
        cusum_weight=s.cusum_weight,
        zscore=ZScoreDetector(threshold=s.zscore_threshold),
        iqr=IQRDetector(window=s.iqr_window, multiplier=s.iqr_multiplier),
        cusum=CUSUMDetector(k=s.cusum_k, h=s.cusum_h),
    )


def _clip01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(value)))


class InferenceService:
    """Scores events and owns the state behind the API."""

    def __init__(
        self,
        cfg: AppConfig,
        registry: ModelRegistry | None = None,
        *,
        anomaly_store: AnomalyStore | None = None,
        feedback_store: FeedbackStore | None = None,
        statistical: UnifiedStatisticalDetector | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Assemble the streaming state (call :meth:`start` to load models).

        Args:
            cfg: Application config.
            registry: Model registry; ``paths.model_registry`` of ``cfg`` if omitted.
            anomaly_store: Alert store; a new ring buffer sized by ``inference`` if omitted.
            feedback_store: Feedback store; empty if omitted.
            statistical: Statistical detector; built from the config if omitted.
            clock: Monotonic clock used for ``processing_ms`` (injectable for tests).
        """
        self.cfg = cfg
        self._clock = clock
        self.started_at = time.monotonic()
        features = cfg.features
        self.cache = FeatureCache(
            default_ttl_s=features.cache_ttl_s, primary_window_s=min(features.window_seconds)
        )
        self.aggregator = WindowedAggregator(
            features.window_seconds, features.watermark_tolerance_s
        )
        self.pipeline = FeaturePipeline.from_config(cfg, self.cache)
        self.registry = registry if registry is not None else ModelRegistry.from_config(cfg)
        self.models = ModelManager(
            cfg,
            self.registry,
            statistical or build_statistical_detector(cfg),
            sequences=self.pipeline.sequences,
        )
        self.anomaly_store = anomaly_store or AnomalyStore(cfg.inference.anomaly_store_capacity)
        self.feedback_store = feedback_store or FeedbackStore()
        self.severity = SeverityScorer.from_config(cfg.alerts)
        self.dedup = AlertDeduplicator.from_config(cfg.alerts)
        self._lock = threading.RLock()  # serialises the stateful detectors / aggregator

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def anomaly_snapshot_path(self) -> Path:
        """File the anomaly store is persisted to."""
        from src.utils.config import resolve_project_path

        return resolve_project_path(self.cfg.paths.anomaly_snapshots) / "anomalies.json"

    @property
    def feedback_path(self) -> Path:
        """File the feedback store is persisted to."""
        from src.utils.config import resolve_project_path

        return resolve_project_path(self.cfg.paths.feedback_store)

    def start(self) -> None:
        """Load models and restore persisted alerts and feedback."""
        self.models.load()
        restored = self.anomaly_store.load_snapshot(self.anomaly_snapshot_path)
        feedback = self.feedback_store.load(self.feedback_path)
        _log.info(
            "inference_service_started",
            degraded=self.models.active.degraded,
            restored_alerts=restored,
            restored_feedback=feedback,
        )

    def stop(self) -> None:
        """Persist alerts and feedback (best effort: a failed save is logged, not raised)."""
        for label, save in (
            ("anomalies", lambda: self.anomaly_store.snapshot(self.anomaly_snapshot_path)),
            ("feedback", lambda: self.feedback_store.save(self.feedback_path)),
        ):
            try:
                save()
            except OSError as error:
                _log.error("persist_failed", what=label, error=repr(error))

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, request: SensorEventRequest) -> AnomalyDetectionResponse:
        """Score one event and, if it is anomalous, raise (or de-duplicate) an alert."""
        started = self._clock()
        message = request.to_kafka_message()
        raw = request.to_raw_event()
        entity_id = request.entity_id
        active = self.models.active  # one reference for the whole request (hot-swap safe)
        ensemble = active.ensembles[request.source_type]

        with self._lock:
            outcome = self.aggregator.process_message(message)
            for window in outcome.features:
                self.cache.store_features(window.entity_id, window, self.cfg.features.cache_ttl_s)
            vector = self.pipeline.process(entity_id)
            score = ensemble.detect(entity_id, event=raw, vector=vector)

            alert = None
            suppressed = False
            severity = None
            if score.is_anomaly:
                context = EntityContext.from_event(raw)
                severity = self.severity.score(score, context)
                candidate = build_anomaly_event(score, severity, context, vector=vector)
                self.pipeline.record_anomaly(entity_id, score.timestamp)
                alert = self.dedup.process(candidate)
                if alert is None:
                    suppressed = True
                else:
                    self.anomaly_store.add(alert)

        method = str(score.details.get("method", "none"))
        return AnomalyDetectionResponse(
            entity_id=entity_id,
            source_type=request.source_type,
            timestamp=request.timestamp,
            is_anomaly=score.is_anomaly,
            anomaly_score=_clip01(score.ensemble_score),
            confidence=_clip01(score.confidence),
            anomaly_type=score.predicted_anomaly_type if score.is_anomaly else None,  # type: ignore[arg-type]
            severity=severity.level.name if severity else None,  # type: ignore[arg-type]
            severity_score=severity.score if severity else None,
            event_id=alert.event_id if alert else None,
            alert_suppressed=suppressed,
            method=method,
            models_used=sorted(score.individual_scores),
            individual_scores={k: _clip01(v) for k, v in score.individual_scores.items()},
            recommended_action=severity.recommended_action if severity else None,
            warming_up=not score.individual_scores,
            ingest_status=outcome.status,
            processing_ms=round((self._clock() - started) * 1000.0, 3),
        )

    # ------------------------------------------------------------------ #
    # Feedback
    # ------------------------------------------------------------------ #
    def record_feedback(self, request: FeedbackRequest) -> tuple[FeedbackRecord, bool]:
        """Store a verdict, flag the alert and tell the feature pipeline.

        Returns:
            ``(record, replaced_previous)``.

        Raises:
            EventNotFoundError: The alert is unknown or has been evicted from the store.
        """
        event = self.anomaly_store.get(request.anomaly_event_id)
        if event is None:
            raise EventNotFoundError(request.anomaly_event_id)
        is_fp = request.feedback_type == "false_positive"
        flagged_by = tuple(sorted(m for m, s in event.individual_scores.items() if s >= 0.5))
        record, replaced = self.feedback_store.add(
            anomaly_event_id=event.event_id,
            entity_id=event.entity_id,
            feedback_type=request.feedback_type,
            analyst_id=request.analyst_id,
            severity=event.severity.name,
            flagged_by=flagged_by,
            notes=request.notes,
        )
        self.anomaly_store.set_flags(event.event_id, false_positive=is_fp)
        if not replaced:
            # The history keeps only counters, which cannot be un-counted: a changed verdict from
            # the same analyst updates the store and the flag but is not counted twice.
            self.pipeline.record_feedback(event.entity_id, false_positive=is_fp)
        return record, replaced
