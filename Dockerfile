FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for pandas / ta-lib wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy project sources
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

RUN chmod +x /app/scripts/*.sh

RUN mkdir -p /app/data

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.main"]
