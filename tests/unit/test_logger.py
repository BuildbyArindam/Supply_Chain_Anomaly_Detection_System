import io
import json
import logging
import uuid

import pytest

from src.utils.logger import bind_context, clear_context, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    clear_context()
    logging.getLogger().handlers.clear()


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_json_output_has_expected_fields():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", service_name="unit", stream=stream)
    get_logger("test.logger").info("reading_generated", sensor_id="s1", value=3.5)
    (record,) = _lines(stream)
    assert record["event"] == "reading_generated"
    assert record["sensor_id"] == "s1"
    assert record["value"] == 3.5
    assert record["level"] == "info"
    assert record["logger"] == "test.logger"
    assert record["service"] == "unit"
    assert record["timestamp"].endswith("Z")


def test_level_filtering():
    stream = io.StringIO()
    configure_logging(level="WARNING", fmt="json", stream=stream)
    log = get_logger("filter")
    log.info("hidden")
    log.warning("shown")
    assert [r["event"] for r in _lines(stream)] == ["shown"]


def test_invalid_level_and_format_rejected():
    with pytest.raises(ValueError):
        configure_logging(level="LOUD")
    with pytest.raises(ValueError):
        configure_logging(fmt="xml")


def test_bound_context_included_then_cleared():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    log = get_logger("ctx")
    bind_context(request_id="abc")
    log.info("with_ctx")
    clear_context()
    log.info("without_ctx")
    with_ctx, without_ctx = _lines(stream)
    assert with_ctx["request_id"] == "abc"
    assert "request_id" not in without_ctx


def test_stdlib_records_use_same_format():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", service_name="unit", stream=stream)
    logging.getLogger("third.party").warning("from stdlib")
    (record,) = _lines(stream)
    assert record["event"] == "from stdlib"
    assert record["level"] == "warning"
    assert record["service"] == "unit"


def test_exception_rendered_as_json_and_unserialisable_values_stringified():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    log = get_logger("exc")
    ident = uuid.uuid4()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed", entity=ident)
    (record,) = _lines(stream)
    assert "RuntimeError: boom" in record["exception"]
    assert record["entity"] == str(ident)


def test_reconfigure_does_not_duplicate_output():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=io.StringIO())
    configure_logging(level="INFO", fmt="json", stream=stream)
    get_logger("dup").info("once")
    assert len(_lines(stream)) == 1


def test_console_format_is_not_json():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="console", stream=stream)
    get_logger("console").info("hello_console")
    out = stream.getvalue()
    assert "hello_console" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.splitlines()[0])
