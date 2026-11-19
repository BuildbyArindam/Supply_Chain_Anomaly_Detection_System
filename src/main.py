"""Command-line entry point for the supply chain anomaly detection system.

Examples::

    python -m src.main info            # configuration summary
    python -m src.main check-config    # validate configs/config.yaml
    python -m src.main show-config     # full effective configuration as JSON
    python -m src.main simulate        # start the data simulator   (Day 4)
    python -m src.main api             # start the FastAPI service  (Day 14)
    python -m src.main dashboard       # start the Dash dashboard   (Day 18)

Commands for components that do not exist yet exit with status 2 and a short
message instead of crashing, so the Makefile targets work from day one.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from src import __version__
from src.utils.config import PROJECT_ROOT, AppConfig, load_config
from src.utils.logger import configure_logging, get_logger

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2


def _module_available(name: str) -> bool:
    """Return True if the dotted module ``name`` can be imported."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _not_implemented(component: str, day: int) -> int:
    """Report that ``component`` is scheduled for a later day of the plan."""
    print(
        f"{component} is not implemented yet (planned for Day {day} of the 30-day plan).",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _cmd_info(cfg: AppConfig) -> int:
    """Print a short, human-readable configuration summary."""
    topics = ", ".join(topic.name for topic in cfg.kafka.topics.as_list())
    lines = [
        f"{cfg.project.name} v{__version__} ({cfg.project.environment})",
        f"  Kafka brokers : {cfg.kafka.bootstrap_servers}",
        f"  Topics        : {topics}",
        f"  API           : http://{cfg.api.host}:{cfg.api.port}",
        f"  Dashboard     : http://{cfg.dashboard.host}:{cfg.dashboard.port}",
        f"  Simulation    : {cfg.simulation.speed_multiplier}x speed, seed={cfg.simulation.seed}",
        f"  Logging       : {cfg.logging.level} / {cfg.logging.format}",
    ]
    print("\n".join(lines))
    return EXIT_OK


def _cmd_check_config(cfg: AppConfig) -> int:
    """Confirm the configuration loaded and validated (loading happens in ``main``)."""
    print(f"Configuration OK ({cfg.project.name}, environment={cfg.project.environment})")
    return EXIT_OK


def _cmd_show_config(cfg: AppConfig) -> int:
    """Print the effective configuration (file + env overrides) as JSON."""
    print(cfg.model_dump_json(indent=2))
    return EXIT_OK


def _cmd_simulate(cfg: AppConfig) -> int:
    """Run ``scripts/start_simulation.py`` once it exists."""
    script = PROJECT_ROOT / "scripts" / "start_simulation.py"
    if not script.is_file():
        return _not_implemented("The simulator", 4)
    runpy.run_path(str(script), run_name="__main__")
    return EXIT_OK


def _cmd_api(cfg: AppConfig) -> int:
    """Serve ``src.api.main:app`` with uvicorn once it exists."""
    if not _module_available("src.api.main"):
        return _not_implemented("The API service", 14)
    import uvicorn

    uvicorn.run("src.api.main:app", host=cfg.api.host, port=cfg.api.port, reload=cfg.api.reload)
    return EXIT_OK


def _cmd_dashboard(cfg: AppConfig) -> int:
    """Run the Dash app from ``src.dashboard.app`` once it exists."""
    if not _module_available("src.dashboard.app"):
        return _not_implemented("The dashboard", 18)
    module = importlib.import_module("src.dashboard.app")
    module.app.run(host=cfg.dashboard.host, port=cfg.dashboard.port, debug=cfg.dashboard.debug)
    return EXIT_OK


_COMMANDS: dict[str, tuple[Callable[[AppConfig], int], str]] = {
    "info": (_cmd_info, "print a configuration summary"),
    "check-config": (_cmd_check_config, "validate the configuration file"),
    "show-config": (_cmd_show_config, "print the effective configuration as JSON"),
    "simulate": (_cmd_simulate, "start the data simulator (Day 4)"),
    "api": (_cmd_api, "start the FastAPI inference service (Day 14)"),
    "dashboard": (_cmd_dashboard, "start the Dash dashboard (Day 18)"),
}


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Real-time supply chain anomaly detection system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to a config YAML file")
    parser.add_argument(
        "--log-level",
        default=None,
        help="override the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--log-format", choices=["json", "console"], default=None)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (_, help_text) in _COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        # pydantic.ValidationError is a ValueError subclass.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        configure_logging(
            level=args.log_level or cfg.logging.level,
            fmt=args.log_format or cfg.logging.format,
            service_name=cfg.logging.service_name,
        )
    except ValueError as exc:
        print(f"Logging error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    get_logger(__name__).debug(
        "command_started", command=args.command, environment=cfg.project.environment
    )
    handler, _ = _COMMANDS[args.command]
    return handler(cfg)


if __name__ == "__main__":
    sys.exit(main())
