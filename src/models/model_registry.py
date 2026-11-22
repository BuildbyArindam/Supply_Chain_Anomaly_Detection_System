"""Versioned, file-based model registry (Day 9).

Every trained model is stored under a timestamped version directory together with a
``metadata.json`` that records when and how it was trained::

    model_registry/
    └── isolation_forest_warehouse/              <- model_type
        ├── 20261122T101500123456Z/              <- version (UTC timestamp, sortable)
        │   ├── model.joblib                     <- the artifact (model.pt for PyTorch models)
        │   └── metadata.json                    <- train date, metrics, hyperparameters, checksum
        └── 20261122T114500654321Z/
            └── ...

Design decisions
----------------
* **Versions are UTC timestamps with microseconds**, so lexical order is chronological order
  and two saves in the same second never collide.
* **Saves are atomic.** The version is staged in a hidden temp directory and renamed into
  place; a crash never leaves a half-written version that :meth:`ModelRegistry.get_latest_model`
  could pick up. Directories that are not valid versions are ignored.
* **Artifacts are checksummed** (SHA-256 in ``metadata.json``) and verified on load, which
  catches truncated or corrupted files. It does *not* protect against someone who can write
  to the registry: ``joblib`` and ``torch`` artifacts are pickles, so only load from a
  registry directory you trust.
* **The registry is generic.** It stores any picklable object (``joblib``) or PyTorch object
  (``torch.save``); Days 10, 11 and 17 reuse it for the LSTM models and champion / challenger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import joblib
import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config import AppConfig

Serializer = Literal["auto", "joblib", "torch"]

DEFAULT_REGISTRY_DIR: Final[str] = "model_registry"
METADATA_FILE: Final[str] = "metadata.json"
ARTIFACT_NAMES: Final[dict[str, str]] = {"joblib": "model.joblib", "torch": "model.pt"}
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VERSION_RE: Final = re.compile(r"^\d{8}T\d{12}Z$")
_VERSION_FORMAT: Final = "%Y%m%dT%H%M%S%fZ"
_MAX_SAVE_ATTEMPTS: Final = 5
_LIBRARIES: Final[tuple[str, ...]] = ("numpy", "scikit-learn", "joblib", "torch")

_log = get_logger(__name__)


class ModelNotFoundError(LookupError):
    """The registry has no such model type or version."""


class RegistryIntegrityError(RuntimeError):
    """An artifact is missing or does not match the checksum in its metadata."""


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """One stored version of a model type (metadata only, the artifact is not loaded).

    Attributes:
        model_type: Name the model was saved under, for example ``isolation_forest_warehouse``.
        version: UTC timestamp identifier; sorts chronologically.
        path: Version directory.
        metadata: Parsed ``metadata.json`` (``created_at``, ``train_date``, ``metrics``,
            ``hyperparams``, ``sha256``, ``extra``, ...).
    """

    model_type: str
    version: str
    path: Path
    metadata: Mapping[str, Any] = field(repr=False)

    @property
    def created_at(self) -> datetime:
        """When the version was saved (UTC)."""
        return datetime.fromisoformat(str(self.metadata["created_at"]))

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Evaluation metrics recorded at save time (may be empty)."""
        return self.metadata.get("metrics") or {}

    @property
    def hyperparams(self) -> Mapping[str, Any]:
        """Hyperparameters recorded at save time (may be empty)."""
        return self.metadata.get("hyperparams") or {}


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A deserialised model together with its version record."""

    model: Any
    version: ModelVersion

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Shortcut for ``version.metadata``."""
        return self.version.metadata


def _validate_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(
            f"invalid {what} {value!r}: use 1-64 letters, digits, '_', '-' or '.', "
            "starting with a letter or digit"
        )
    return value


def _json_default(value: Any) -> Any:
    """Make numpy scalars / arrays, datetimes and paths JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _sanitize(value: Any) -> Any:
    """Recursively replace NaN / inf (invalid in strict JSON) by ``None``."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions(serializer: str) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for lib in _LIBRARIES:
        if lib == "torch" and serializer != "torch":
            continue
        try:
            versions[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _is_torch_object(obj: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency
        return False
    return isinstance(obj, torch.nn.Module | torch.Tensor)


class ModelRegistry:
    """Save and load models with timestamp versions and a metadata record.

    Example::

        registry = ModelRegistry("model_registry")
        version = registry.save(
            "isolation_forest_warehouse",
            detector,
            metrics={"f1": 0.81},
            hyperparams=detector.get_params(),
        )
        loaded = registry.get_latest_model("isolation_forest_warehouse")
        loaded.model, loaded.metadata["metrics"]

    Thread-safe within a process; saves from several processes are safe too because each
    version is staged privately and published with one atomic rename.
    """

    def __init__(self, root: str | Path = DEFAULT_REGISTRY_DIR) -> None:
        """Create the registry handle (the directory is created on the first save).

        Args:
            root: Registry directory.
        """
        self.root = Path(root)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: AppConfig) -> ModelRegistry:
        """Create the registry at ``paths.model_registry`` of the project config."""
        return cls(cfg.paths.model_registry)

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save(
        self,
        model_type: str,
        model: Any,
        *,
        metrics: Mapping[str, Any] | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        train_date: datetime | str | None = None,
        extra: Mapping[str, Any] | None = None,
        serializer: Serializer = "auto",
    ) -> ModelVersion:
        """Store ``model`` as a new version of ``model_type``.

        Args:
            model_type: Name of the model family (letters, digits, ``_ - .``).
            model: The object to store. ``auto`` uses ``torch.save`` for ``torch.nn.Module`` /
                ``torch.Tensor`` (file ``model.pt``) and ``joblib`` for everything else
                (file ``model.joblib``).
            metrics: Evaluation results (numpy scalars and NaN are handled).
            hyperparams: Hyperparameters the model was trained with.
            train_date: When the training data ends / training happened; defaults to now.
            extra: Anything else worth keeping (feature names, data description, notes).
            serializer: Force ``joblib`` or ``torch``.

        Returns:
            The new :class:`ModelVersion`.

        Raises:
            ValueError: Invalid ``model_type`` or ``serializer``.
            OSError: The registry directory cannot be written.
        """
        _validate_name(model_type, "model_type")
        if serializer not in ("auto", "joblib", "torch"):
            raise ValueError(f"unknown serializer {serializer!r}")
        chosen = (
            ("torch" if _is_torch_object(model) else "joblib")
            if serializer == "auto"
            else serializer
        )
        type_dir = self.root / model_type
        type_dir.mkdir(parents=True, exist_ok=True)
        staging = type_dir / f".tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            artifact = staging / ARTIFACT_NAMES[chosen]
            self._dump(model, artifact, chosen)
            return self._publish(
                type_dir,
                staging,
                artifact,
                model_type=model_type,
                serializer=chosen,
                metrics=metrics,
                hyperparams=hyperparams,
                train_date=train_date,
                extra=extra,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _dump(model: Any, path: Path, serializer: str) -> None:
        if serializer == "torch":
            import torch

            torch.save(model, path)
        else:
            joblib.dump(model, path, compress=3)

    def _publish(
        self,
        type_dir: Path,
        staging: Path,
        artifact: Path,
        *,
        model_type: str,
        serializer: str,
        metrics: Mapping[str, Any] | None,
        hyperparams: Mapping[str, Any] | None,
        train_date: datetime | str | None,
        extra: Mapping[str, Any] | None,
    ) -> ModelVersion:
        """Write the metadata and move the staged directory to its final, versioned name."""
        sha = _sha256(artifact)
        size = artifact.stat().st_size
        for _ in range(_MAX_SAVE_ATTEMPTS):
            with self._lock:  # keeps this process's versions strictly increasing
                created = datetime.now(UTC)
                version = created.strftime(_VERSION_FORMAT)
                metadata: dict[str, Any] = {
                    "model_type": model_type,
                    "version": version,
                    "created_at": created.isoformat(),
                    "train_date": (
                        _iso(train_date) if train_date is not None else created.isoformat()
                    ),
                    "serializer": serializer,
                    "artifact": artifact.name,
                    "sha256": sha,
                    "size_bytes": size,
                    "metrics": _sanitize(dict(metrics or {})),
                    "hyperparams": _sanitize(dict(hyperparams or {})),
                    "extra": _sanitize(dict(extra or {})),
                    "libraries": _library_versions(serializer),
                }
                (staging / METADATA_FILE).write_text(
                    json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
                )
                target = type_dir / version
                try:
                    os.rename(staging, target)
                except OSError:
                    if target.exists():  # same-microsecond clash from another process: retry
                        continue
                    raise
            _log.info("model_saved", model_type=model_type, version=version, serializer=serializer)
            return ModelVersion(model_type, version, target, metadata)
        raise OSError(f"could not allocate a unique version for {model_type!r}")

    # ------------------------------------------------------------------ #
    # Listing and loading
    # ------------------------------------------------------------------ #
    def list_model_types(self) -> list[str]:
        """Return the model types that have at least one valid version, sorted by name."""
        if not self.root.is_dir():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and _NAME_RE.fullmatch(p.name) and self.list_versions(p.name)
        )

    def list_versions(self, model_type: str) -> list[ModelVersion]:
        """Return all valid versions of ``model_type``, newest first (empty if none).

        A directory counts as a version when its name is a version timestamp and it holds a
        readable ``metadata.json``; anything else (staging leftovers, stray files) is skipped.
        """
        _validate_name(model_type, "model_type")
        type_dir = self.root / model_type
        if not type_dir.is_dir():
            return []
        versions: list[ModelVersion] = []
        for entry in sorted(type_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not (entry.is_dir() and _VERSION_RE.fullmatch(entry.name)):
                continue
            try:
                meta = json.loads((entry / METADATA_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _log.warning("registry_version_unreadable", path=str(entry))
                continue
            versions.append(ModelVersion(model_type, entry.name, entry, meta))
        return versions

    def get_version(self, model_type: str, version: str) -> ModelVersion:
        """Return the record of one specific version.

        Raises:
            ValueError: ``version`` is not a version identifier.
            ModelNotFoundError: No such version.
        """
        if not _VERSION_RE.fullmatch(version):
            raise ValueError(f"invalid version {version!r}")
        for item in self.list_versions(model_type):
            if item.version == version:
                return item
        raise ModelNotFoundError(f"{model_type!r} has no version {version!r}")

    def get_latest_model(self, model_type: str) -> LoadedModel:
        """Load the newest version of ``model_type``.

        Raises:
            ModelNotFoundError: The model type has no versions.
            RegistryIntegrityError: The artifact is missing or fails its checksum.
        """
        versions = self.list_versions(model_type)
        if not versions:
            raise ModelNotFoundError(f"no versions of {model_type!r} in {self.root}")
        return self._load(versions[0])

    def load(self, model_type: str, version: str | None = None) -> LoadedModel:
        """Load one version (the newest if ``version`` is ``None``).

        Raises:
            ModelNotFoundError: Unknown model type or version.
            RegistryIntegrityError: The artifact is missing or fails its checksum.
        """
        if version is None:
            return self.get_latest_model(model_type)
        return self._load(self.get_version(model_type, version))

    def _load(self, record: ModelVersion) -> LoadedModel:
        meta = record.metadata
        artifact = record.path / str(meta.get("artifact", ""))
        if not artifact.is_file():
            raise RegistryIntegrityError(
                f"artifact missing for {record.model_type}/{record.version}"
            )
        expected = meta.get("sha256")
        if expected and _sha256(artifact) != expected:
            raise RegistryIntegrityError(
                f"checksum mismatch for {record.model_type}/{record.version}: the artifact "
                "is corrupted or was modified after saving"
            )
        if meta.get("serializer") == "torch":
            import torch

            model = torch.load(artifact, map_location="cpu", weights_only=False)
        else:
            model = joblib.load(artifact)
        return LoadedModel(model, record)


def _iso(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


__all__ = [
    "ARTIFACT_NAMES",
    "DEFAULT_REGISTRY_DIR",
    "LoadedModel",
    "ModelNotFoundError",
    "ModelRegistry",
    "ModelVersion",
    "RegistryIntegrityError",
]
