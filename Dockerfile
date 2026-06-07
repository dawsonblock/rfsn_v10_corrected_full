FROM python:3.12-slim

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from rfsn_v10.health import get_health_checker; checker = get_health_checker(); report = checker.get_health_report(); exit(0 if report['overall_status'] == 'healthy' else 1)"

# Default command
CMD ["python", "-m", "rfsn_v10"]
