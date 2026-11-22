"""Tests for src/models/offline_dataset.py (Day 9)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from src.models.feature_engineering import feature_names
from src.models.offline_dataset import hours, simulate_labeled_features

START = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def sets():
    return simulate_labeled_features(
        1200, start_time=START, seed=5, sources=("warehouse", "vehicle", "inventory")
    )


def test_schema_matches_feature_pipeline(sets):
    for source, ds in sets.items():
        assert ds.feature_names == tuple(feature_names(source))
        assert ds.X.shape == (len(ds), len(ds.feature_names))
        assert len(ds.y) == len(ds.anomaly_type) == len(ds.timestamps) == len(ds)


def test_rows_in_time_order_and_labels_consistent(sets):
    ds = sets["warehouse"]
    assert np.all(np.diff(ds.timestamps) >= 0)
    assert all(t is not None for t in ds.anomaly_type[ds.y])
    assert all(t is None for t in ds.anomaly_type[~ds.y])
    counts = ds.type_counts()
    assert sum(counts.values()) == len(ds) and "normal" in counts
    assert 0 < ds.anomaly_rate < 0.5


def test_warm_flag_and_time_masks(sets):
    ds = sets["warehouse"]
    assert not ds.warm[0] and ds.warm[-1]  # cold start first
    first_10 = ds.between(START, hours(START, 10 / 60))
    assert 0 < first_10.sum() < len(ds)
    part = ds.subset(first_10)
    assert len(part) == first_10.sum() and part.feature_names == ds.feature_names
    assert ds.feature_index("value__last") == ds.feature_names.index("value__last")
    with pytest.raises(KeyError):
        ds.feature_index("nope")


def test_only_requested_sources_and_determinism():
    a = simulate_labeled_features(300, start_time=START, seed=1)
    b = simulate_labeled_features(300, start_time=START, seed=1)
    assert list(a) == ["warehouse"]
    assert np.array_equal(a["warehouse"].X, b["warehouse"].X)
    assert np.array_equal(a["warehouse"].y, b["warehouse"].y)


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="unknown sources"):
        simulate_labeled_features(10, sources=("boats",))
