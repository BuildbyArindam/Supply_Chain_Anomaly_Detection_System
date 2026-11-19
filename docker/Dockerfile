# Application image. Build from the project root: make docker-build
# Default command prints the configuration summary; override per service, e.g.
#   docker run --rm -p 8000:8000 supply-chain-anomaly-detection python -m src.main api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY configs ./configs

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "src.main", "info"]
