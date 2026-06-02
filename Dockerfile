FROM python:3.11-slim

LABEL maintainer="RFSN Contributors"
LABEL description="RFSN v10 - Quantized KV-cache + decode-time sparse-attention runtime"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[mlx,production]"

# Copy source code
COPY rfsn_v10/ ./rfsn_v10/
COPY agent_core/ ./agent_core/
COPY tools/ ./tools/
COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/

# Create cache directory
RUN mkdir -p /app/.cache

# Set environment variables
ENV PYTHONPATH=/app
ENV RFSN_CACHE_DIR=/app/.cache
ENV RFSN_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from rfsn_v10.health import get_health_checker; checker = get_health_checker(); report = checker.get_health_report(); exit(0 if report['overall_status'] == 'healthy' else 1)"

# Default command
CMD ["python", "-m", "rfsn_v10"]
