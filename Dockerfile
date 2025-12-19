# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- deps layer (cache-friendly) ----
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

# Install project deps only (no editable install required)
RUN pip install .

# ---- app code ----
COPY . /app

EXPOSE 8000 8501

# Default command: API
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
