"""Labeled feature matrices from the simulators, for offline training and evaluation (Day 9).

Runs the production path -- simulator -> :class:`KafkaMessage` -> :class:`WindowedAggregator`
-> :class:`FeatureEngineer` -- as fast as the CPU allows (about 10 000 events/s) and returns,
per source type, one row per event::

    simulate_labeled_features(...)  ->  {"warehouse": LabeledFeatureSet(X, y, ...)}

The feature vectors are exactly what the online pipeline computes (no train / serve skew).
The simulators' ground truth is returned *next to* the features, never inside them: ``y`` and
``anomaly_type`` are for training labels and evaluation only.

Rows come out in event-time order, so a chronological train / validation / test split is a
boolean mask on ``timestamps`` (see :meth:`LabeledFeatureSet.between`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

import numpy as np

from src.data_simulation.orchestrator import SimulationOrchestrator
from src.kafka.schemas import KafkaMessage
from src.kafka.stream_processor import DEFAULT_WINDOWS, SIGNALS_BY_SOURCE, WindowedAggregator
from src.models.feature_engineering import FeatureEngineer, WindowBundle
from src.utils.config import get_config

if TYPE_CHECKING:
    from src.utils.config import SimulationConfig

SOURCES: Final[tuple[str, ...]] = tuple(SIGNALS_BY_SOURCE)


@dataclass(frozen=True)
class LabeledFeatureSet:
    """Feature matrix and ground truth of one source type.

    Attributes:
        source_type: ``warehouse``, ``vehicle`` or ``inventory``.
        feature_names: Column names of ``X`` (:func:`feature_engineering.feature_names`).
        X: Raw (unscaled) features, shape ``(n, n_features)``; ``NaN`` means unknown.
        y: ``True`` for an anomalous event (simulator ground truth).
        anomaly_type: ``None`` for normal events, else ``point`` / ``contextual`` / ``collective``
            (vehicle and inventory use their own subtypes under the same three names).
        entity_ids: Sensor, vehicle or SKU id per row.
        timestamps: Event time as UTC epoch seconds.
        zones: Zone of the entity where the source has one (``None`` otherwise).
        warm: ``True`` once the longest window of the entity is at least 90% filled; rows before
            that are aggregator cold start and are usually dropped from training and evaluation.
    """

    source_type: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    anomaly_type: np.ndarray
    entity_ids: np.ndarray
    timestamps: np.ndarray
    zones: np.ndarray
    warm: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def anomaly_rate(self) -> float:
        """Share of anomalous rows."""
        return float(self.y.mean()) if len(self) else 0.0

    def feature_index(self, name: str) -> int:
        """Return the column of a feature.

        Raises:
            KeyError: Unknown feature name.
        """
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise KeyError(f"unknown feature {name!r}") from None

    def between(self, start: datetime, end: datetime) -> np.ndarray:
        """Return a boolean mask of the rows with ``start <= timestamp < end``."""
        lo, hi = _epoch(start), _epoch(end)
        return (self.timestamps >= lo) & (self.timestamps < hi)

    def subset(self, mask: np.ndarray) -> LabeledFeatureSet:
        """Return the rows selected by a boolean mask or an index array."""
        return LabeledFeatureSet(
            self.source_type,
            self.feature_names,
            self.X[mask],
            self.y[mask],
            self.anomaly_type[mask],
            self.entity_ids[mask],
            self.timestamps[mask],
            self.zones[mask],
            self.warm[mask],
        )

    def type_counts(self) -> dict[str, int]:
        """Return the number of rows per label (``normal`` plus each anomaly type)."""
        labels = np.where(self.y, self.anomaly_type.astype(str), "normal")
        values, counts = np.unique(labels, return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts, strict=True)}


def _epoch(moment: datetime) -> float:
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp()


class _Collector:
    """Accumulates the rows of one source type."""

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.y: list[bool] = []
        self.kind: list[str | None] = []
        self.entity: list[str] = []
        self.ts: list[float] = []
        self.zone: list[str | None] = []
        self.warm: list[bool] = []

    def add(self, vector: np.ndarray, bundle: WindowBundle) -> None:
        short = bundle.short
        self.rows.append(vector)
        self.y.append(bool(short.label_is_anomaly))
        self.kind.append(short.label_anomaly_type if short.label_is_anomaly else None)
        self.entity.append(bundle.entity_id)
        self.ts.append(bundle.window_end.timestamp())
        zone = bundle.context.get("zone")
        self.zone.append(str(zone) if zone is not None else None)
        self.warm.append(bundle.long.is_warm)

    def build(self, source: str, names: Sequence[str]) -> LabeledFeatureSet:
        n = len(self.rows)
        return LabeledFeatureSet(
            source_type=source,
            feature_names=tuple(names),
            X=np.vstack(self.rows) if n else np.empty((0, len(names))),
            y=np.asarray(self.y, dtype=bool),
            anomaly_type=np.asarray(self.kind, dtype=object),
            entity_ids=np.asarray(self.entity, dtype=object),
            timestamps=np.asarray(self.ts, dtype=np.float64),
            zones=np.asarray(self.zone, dtype=object),
            warm=np.asarray(self.warm, dtype=bool),
        )


def simulate_labeled_features(
    duration_s: float,
    *,
    start_time: datetime | None = None,
    seed: int | None = None,
    sources: Sequence[str] = ("warehouse",),
    sim_config: SimulationConfig | None = None,
    window_seconds: Sequence[int] = DEFAULT_WINDOWS,
) -> dict[str, LabeledFeatureSet]:
    """Simulate ``duration_s`` seconds and return the labeled feature rows per source type.

    Args:
        duration_s: Simulated seconds (the clock starts at ``start_time``).
        start_time: Simulated start (UTC). Matters for the warehouse simulator: contextual
            anomalies are only injected during off-hours (00:00-05:00 by default).
        seed: Overrides the simulation seed of the config.
        sources: Which of ``warehouse`` / ``vehicle`` / ``inventory`` to featurise.
        sim_config: Simulation settings; defaults to ``config.simulation``.
        window_seconds: Window lengths of the aggregator (must match the production setup).

    Returns:
        One :class:`LabeledFeatureSet` per requested source (empty if it produced no rows).

    Raises:
        ValueError: ``duration_s`` is negative or a source is unknown.
    """
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; choose from {list(SOURCES)}")
    cfg = sim_config if sim_config is not None else get_config().simulation
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    start = start_time if start_time is not None else datetime(2026, 11, 20, 0, 0, tzinfo=UTC)
    orchestrator = SimulationOrchestrator.from_config(cfg, start_time=start, include_orders=False)
    wanted = set(sources)
    aggregator = WindowedAggregator(window_seconds)
    engineer = FeatureEngineer()
    needed = len(aggregator.window_seconds)
    collectors = {source: _Collector() for source in wanted}
    for event in orchestrator.stream(duration_s, throttle=False):
        source = event["source_type"]
        if source not in wanted:
            continue
        message = KafkaMessage.from_payload(source, dict(event["payload"]))
        outcome = aggregator.process_message(message)
        windows = {f.window_seconds: f for f in outcome.features}
        if len(windows) != needed:
            continue
        bundle = WindowBundle(
            message.entity_id,
            source,
            outcome.features[0].window_end,
            dict(sorted(windows.items())),
        )
        collectors[source].add(engineer.vectorize(bundle), bundle)
    return {source: collectors[source].build(source, engineer.names(source)) for source in sources}


def hours(start: datetime, count: float) -> datetime:
    """Return ``start`` plus ``count`` hours (a readability helper for split boundaries)."""
    return start + timedelta(hours=count)


__all__ = ["SOURCES", "LabeledFeatureSet", "hours", "simulate_labeled_features"]
