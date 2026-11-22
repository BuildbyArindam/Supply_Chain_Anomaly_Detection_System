"""Tests for src/models/model_registry.py (Day 9)."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from src.models.model_registry import (
    ModelNotFoundError,
    ModelRegistry,
    RegistryIntegrityError,
)

TYPE = "isolation_forest_warehouse"


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "reg")


def test_save_and_load_roundtrip_with_metadata(registry):
    model = {"w": np.arange(5)}
    ver = registry.save(
        TYPE,
        model,
        metrics={"f1": np.float64(0.8), "bad": float("nan")},
        hyperparams={"n_estimators": 100},
        train_date=datetime(2026, 11, 20, tzinfo=UTC),
        extra={"features": ["a", "b"]},
    )
    loaded = registry.get_latest_model(TYPE)
    assert np.array_equal(loaded.model["w"], model["w"])
    meta = loaded.metadata
    assert meta["metrics"] == {"f1": 0.8, "bad": None}  # NaN is not valid JSON
    assert meta["hyperparams"] == {"n_estimators": 100}
    assert meta["train_date"].startswith("2026-11-20")
    assert meta["extra"]["features"] == ["a", "b"]
    assert meta["artifact"] == "model.joblib" and len(meta["sha256"]) == 64
    assert ver.path.joinpath("metadata.json").is_file()
    json.loads(ver.path.joinpath("metadata.json").read_text())  # strict, parseable


def test_versions_sorted_newest_first_and_latest_is_newest(registry):
    versions = [registry.save(TYPE, {"i": i}).version for i in range(3)]
    assert versions == sorted(versions) and len(set(versions)) == 3
    listed = [v.version for v in registry.list_versions(TYPE)]
    assert listed == versions[::-1]
    assert registry.get_latest_model(TYPE).model == {"i": 2}
    assert registry.load(TYPE, versions[0]).model == {"i": 0}


def test_unknown_model_and_version(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get_latest_model("nothing")
    registry.save(TYPE, 1)
    with pytest.raises(ModelNotFoundError):
        registry.load(TYPE, "20200101T000000000000Z")
    with pytest.raises(ValueError):
        registry.load(TYPE, "../../etc")
    assert registry.list_versions("nothing") == []


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", "x" * 80])
def test_invalid_model_type_rejected(registry, bad):
    with pytest.raises(ValueError):
        registry.save(bad, 1)


def test_corrupted_artifact_detected(registry):
    ver = registry.save(TYPE, {"a": 1})
    (ver.path / "model.joblib").write_bytes(b"garbage")
    with pytest.raises(RegistryIntegrityError, match="checksum"):
        registry.get_latest_model(TYPE)


def test_missing_artifact_detected(registry):
    ver = registry.save(TYPE, {"a": 1})
    (ver.path / "model.joblib").unlink()
    with pytest.raises(RegistryIntegrityError, match="missing"):
        registry.load(TYPE)


def test_stray_and_staging_dirs_are_ignored(registry):
    ver = registry.save(TYPE, 1)
    (registry.root / TYPE / ".tmp-abc").mkdir()
    (registry.root / TYPE / "notes").mkdir()
    broken = registry.root / TYPE / "29990101T000000000000Z"
    broken.mkdir()  # newer-looking but has no metadata: must not win
    assert [v.version for v in registry.list_versions(TYPE)] == [ver.version]
    assert registry.get_latest_model(TYPE).model == 1


def test_list_model_types(registry):
    assert registry.list_model_types() == []
    registry.save("b_model", 1)
    registry.save("a_model", 1)
    assert registry.list_model_types() == ["a_model", "b_model"]


def test_torch_module_uses_pt_file(registry):
    net = torch.nn.Linear(3, 2)
    ver = registry.save("lstm_ae", net)
    assert ver.metadata["serializer"] == "torch" and (ver.path / "model.pt").is_file()
    assert "torch" in ver.metadata["libraries"]
    loaded = registry.get_latest_model("lstm_ae").model
    assert torch.equal(loaded.weight, net.weight)


def test_no_staging_left_after_save(registry):
    registry.save(TYPE, 1)
    assert not [p for p in (registry.root / TYPE).iterdir() if p.name.startswith(".tmp")]


def test_version_properties(registry):
    ver = registry.save(TYPE, 1, metrics={"f1": 0.5}, hyperparams={"k": 1})
    assert ver.metrics["f1"] == 0.5 and ver.hyperparams["k"] == 1
    assert ver.created_at.tzinfo is not None
