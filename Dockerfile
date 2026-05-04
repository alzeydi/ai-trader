FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for pandas / ta-lib wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --uid 1000 trader
WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy project sources
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

# Runtime data volume
RUN mkdir -p /app/data && chown -R trader:trader /app
VOLUME ["/app/data"]

USER trader

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.main"]
