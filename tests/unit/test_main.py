import json
import logging

import pytest

from src.main import EXIT_ERROR, EXIT_NOT_IMPLEMENTED, EXIT_OK, main


@pytest.fixture(autouse=True)
def _clean_env_and_logging(monkeypatch):
    for name in ("LOG_LEVEL", "LOG_FORMAT", "API_PORT", "CONFIG_PATH", "APP_ENV"):
        monkeypatch.delenv(name, raising=False)
    yield
    logging.getLogger().handlers.clear()


def test_info(capsys):
    assert main(["info"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "localhost:9092" in out
    assert "warehouse-sensors" in out


def test_check_config(capsys):
    assert main(["check-config"]) == EXIT_OK
    assert "Configuration OK" in capsys.readouterr().out


def test_show_config_is_valid_json(capsys):
    assert main(["show-config"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["api"]["port"] == 8000


@pytest.mark.parametrize("command", ["dashboard"])
def test_future_components_exit_cleanly(command, capsys):
    assert main([command]) == EXIT_NOT_IMPLEMENTED
    assert "not implemented yet" in capsys.readouterr().err


def test_missing_config_returns_error(capsys, tmp_path):
    assert main(["--config", str(tmp_path / "missing.yaml"), "info"]) == EXIT_ERROR
    assert "Configuration error" in capsys.readouterr().err


def test_invalid_log_level_returns_error(capsys):
    assert main(["--log-level", "LOUD", "info"]) == EXIT_ERROR
    assert "Logging error" in capsys.readouterr().err


def test_unknown_command_exits_with_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 2
