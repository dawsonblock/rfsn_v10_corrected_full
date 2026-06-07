FROM python:3.11-slim

LABEL maintainer="RFSN Contributors"
LABEL description="RFSN v10 - Quantized KV-cache + decode-time sparse-attention runtime"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r rfsn && useradd -r -g rfsn rfsn

# Copy build metadata first for layer caching
COPY pyproject.toml README.md ./

# Install Python dependencies (editable so local changes are reflected)
RUN pip install --no-cache-dir -e ".[production]"

# Copy source code
COPY rfsn_v10/ ./rfsn_v10/
COPY agent_core/ ./agent_core/
COPY tools/ ./tools/
COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/

# Re-install to pick up any source changes
RUN pip install --no-cache-dir -e ".[production]"

# Create cache directory and fix permissions
RUN mkdir -p /app/.cache /app/artifacts && chown -R rfsn:rfsn /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RFSN_CACHE_DIR=/app/.cache
ENV RFSN_LOG_LEVEL=INFO
ENV RFSN_TELEMETRY_DIR=/app/artifacts/runtime_logs

# Switch to non-root user
USER rfsn

# Health check — runs the actual CLI healthcheck subcommand
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m rfsn_v10 healthcheck

# Default command: CLI healthcheck (no HTTP server — this is a CLI tool)
CMD ["python", "-m", "rfsn_v10", "healthcheck"]
