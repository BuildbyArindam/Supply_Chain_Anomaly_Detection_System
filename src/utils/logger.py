"""Structured logging for every service in the project.

Built on ``structlog`` with output routed through the standard library
``logging`` module, so records from third-party libraries (uvicorn, kafka,
torch, ...) use the same format as our own.

* ``json`` format (default): one JSON object per line - ideal for Docker,
  Loki/ELK and ``jq``.
* ``console`` format: colourised, human-friendly output for local development.

The level and format come from the arguments, then the ``LOG_LEVEL`` /
``LOG_FORMAT`` / ``SERVICE_NAME`` environment variables, then the defaults.

Example::

    from src.utils.logger import bind_context, configure_logging, get_logger

    configure_logging(level="DEBUG", service_name="simulator")
    log = get_logger(__name__)
    bind_context(request_id="abc123")
    log.info("reading_generated", sensor_id="wh1-A-temp-01", value=3.1)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import IO, Any

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

DEFAULT_LEVEL = "INFO"
DEFAULT_FORMAT = "json"
DEFAULT_SERVICE_NAME = "scad"
_VALID_FORMATS = frozenset({"json", "console"})
_HANDLER_FLAG = "_scad_handler"


def _json_dumps(obj: Any, **_ignored: Any) -> str:
    """Serialise to JSON, falling back to ``str()`` for UUID, datetime, Path, etc.

    structlog passes its own ``default=`` (which uses ``repr``); it is ignored on
    purpose so unserialisable values render as clean strings.
    """
    return json.dumps(obj, default=str)


def _resolve_level(level: str | int | None) -> int:
    """Translate a level name or number into a ``logging`` level number."""
    raw: str | int = level if level is not None else os.environ.get("LOG_LEVEL", DEFAULT_LEVEL)
    if isinstance(raw, int):
        return raw
    name = raw.strip().upper()
    mapping = logging.getLevelNamesMapping()
    if name not in mapping:
        raise ValueError(f"Invalid log level {raw!r}; expected one of {sorted(mapping)}")
    return mapping[name]


def _add_service_name(service_name: str) -> Processor:
    """Return a processor that stamps every record with the service name."""

    def processor(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", service_name)
        return event_dict

    return processor


def configure_logging(
    level: str | int | None = None,
    fmt: str | None = None,
    service_name: str | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Configure structlog and the stdlib root logger.

    Safe to call repeatedly: the handler installed by a previous call is
    replaced instead of duplicated.

    Args:
        level: Level name (``"DEBUG"``) or number. Defaults to ``$LOG_LEVEL`` / INFO.
        fmt: ``"json"`` or ``"console"``. Defaults to ``$LOG_FORMAT`` / json.
        service_name: Value of the ``service`` field. Defaults to
            ``$SERVICE_NAME`` / ``scad``.
        stream: Output stream. Defaults to ``sys.stdout``.

    Raises:
        ValueError: The level or format is not recognised.
    """
    log_level = _resolve_level(level)
    log_format = (fmt or os.environ.get("LOG_FORMAT", DEFAULT_FORMAT)).strip().lower()
    if log_format not in _VALID_FORMATS:
        raise ValueError(
            f"Invalid log format {log_format!r}; expected one of {sorted(_VALID_FORMATS)}"
        )
    service = service_name or os.environ.get("SERVICE_NAME", DEFAULT_SERVICE_NAME)
    out: IO[str] = stream if stream is not None else sys.stdout

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_service_name(service),
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    final_processors: list[Processor] = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    if log_format == "json":
        final_processors.append(structlog.processors.format_exc_info)
        final_processors.append(structlog.processors.JSONRenderer(serializer=_json_dumps))
    else:
        use_colors = bool(getattr(out, "isatty", lambda: False)())
        final_processors.append(structlog.dev.ConsoleRenderer(colors=use_colors))

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=final_processors,
    )
    handler = logging.StreamHandler(out)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_FLAG, True)

    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _HANDLER_FLAG, False)]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(log_level)


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger, configuring logging from the environment if needed.

    Args:
        name: Logger name, usually ``__name__``.
        **initial_values: Key/value pairs bound to every record from this logger.
    """
    if not structlog.is_configured():
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name, **initial_values)
    return logger


def bind_context(**values: Any) -> None:
    """Bind key/value pairs to every log record in the current context (async-safe)."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Remove all values previously set with :func:`bind_context`."""
    structlog.contextvars.clear_contextvars()
