# NIC Fraud Detection V3 — nic-fraud-server image (ADR-010)
# CPU-only. Runs both the FastAPI server and the Celery worker.
# GPU training is done on the laptop; checkpoint transferred via DVC.

FROM python:3.12-slim

WORKDIR /app

# System deps (gcc needed for some scipy/numpy wheels on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first (platform-specific wheel index — must come before -r requirements.txt)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (Linux-compatible min-version pins)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source
COPY src/       ./src/
COPY main_v3.py celeryconfig.py pyproject.toml ./

# Runtime directories must exist (volumes will overlay these at runtime)
RUN mkdir -p data/raw data/processed models/checkpoints outputs

EXPOSE 8000

# Default: FastAPI server.
# Override with: celery -A src.api.tasks.celery_app worker ... for the worker container.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
