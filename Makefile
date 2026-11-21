.DEFAULT_GOAL := help
.PHONY: help setup run-kafka stop-kafka kafka-check run-simulator run-consumer run-api run-dashboard test lint format typecheck docker-build clean

PYTHON_BOOTSTRAP ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python
IMAGE  ?= supply-chain-anomaly-detection
COMPOSE := docker compose -f docker/docker-compose.kafka.yml

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv, install pinned dependencies, copy .env.example to .env
	@$(PYTHON_BOOTSTRAP) -c 'import sys; sys.exit("Python 3.11+ required, found " + sys.version.split()[0]) if sys.version_info < (3, 11) else None'
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	@echo "Setup complete. Try: make test"

run-kafka: ## Start Kafka (KRaft) and Kafka UI in Docker
	$(COMPOSE) up -d
	@echo "Kafka: localhost:9092   Kafka UI: http://localhost:8080"

stop-kafka: ## Stop the Kafka containers (data is kept in a volume)
	$(COMPOSE) down

kafka-check: ## Check Kafka connectivity and topics
	$(PY) scripts/start_simulation.py --check

run-simulator: ## Stream simulated events into Kafka (needs `make run-kafka`)
	$(PY) -m src.main simulate

run-consumer: ## Consume Kafka, compute windowed features (metrics on :8001)
	$(PY) -m src.main consume

run-api: ## Run the FastAPI service (Day 14)
	$(PY) -m src.main api

run-dashboard: ## Run the Dash dashboard (Day 18)
	$(PY) -m src.main dashboard

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint with ruff and check formatting with black
	$(PY) -m ruff check .
	$(PY) -m black --check .

format: ## Auto-format and apply safe lint fixes
	$(PY) -m ruff check --fix .
	$(PY) -m black .

typecheck: ## Static type check with mypy
	$(PY) -m mypy

docker-build: ## Build the application Docker image
	docker build -f docker/Dockerfile -t $(IMAGE):latest .

clean: ## Remove caches and build artifacts (keeps .venv)
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov build dist
