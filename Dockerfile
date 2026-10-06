# ==============================================================================
# Multi-Agent Orchestration Platform - Production Dockerfile
# Python Baseline: 3.12-slim (Stable, slim Debian-based release)
# ==============================================================================

FROM python:3.12-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000

WORKDIR /app

# Install system dependencies and create dedicated non-root user
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/appuser --shell /bin/bash appuser

# Install Python dependencies first (leveraging Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ /app/src/

# Pre-create data directories for Chroma RAG and memory, assigning ownership to appuser
RUN mkdir -p /app/data/chroma /app/data/chroma_memory \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose standard application port
EXPOSE 8000

# Container healthcheck against the FastAPI liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server with proxy headers enabled
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
