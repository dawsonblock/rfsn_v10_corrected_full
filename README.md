# RFSN v10 Main 11 - Custom Metal Kernel Alpha

## Status: Main 11 Custom Metal Kernel Alpha

Verified in this package:
- static syntax
- non-MLX tests
- proof regression tooling
- generated proof plots

Requires Apple Silicon validation:
- MLX bitpack tests
- KV manager tests
- custom Metal kernel equivalence tests
- runtime audit tests
- real LLM logit/perplexity validation

Claim boundary:
- experimental custom Metal kernel path
- reference-equivalence gated
- runtime optimization alpha

## Overview
RFSN v10 is an alpha quantized KV-cache + decode-time sparse-attention runtime for MLX/Apple Silicon. This build focuses on proving numerical equivalence and quality-safe sparse behavior before any production claim.

## Components

### Core Modules
- `bitpack.py` - Bit-packed quantizer (2-8 bit widths) with exact roundtrip guarantees
- `kv_manager.py` - TurboQuant KV manager with grouped symmetric quantization and WHT preconditioning
- `attention.py` - Adaptive block-sparse attention with decode-only sparse path and prefill dense fallback
- `runtime.py` - Orchestrator integrating KV cache, sparse attention, audit mode, and telemetry
- `adaptive_sparsity.py` - Controller that adjusts top_k_ratio based on real quality signals
- `memory_guard.py` - MLX/Metal memory monitoring and protection against OOM
- `async_writer.py` - Background telemetry writer with batching and retry logic
- `clickhouse_client.py` - HTTP-based ClickHouse client for telemetry ingestion

### Test Suite
- `tests/test_bitpack.py` - bitpack roundtrip, stress, and rejection coverage
- `tests/test_kv_manager.py` - KV compression/reconstruction and cache behavior coverage
- `tests/test_metal_kernel_math.py` - bitpack, quantization, WHT, and metal-path reconstruction coverage
- `tests/test_attention.py` - sparse attention correctness, fallback, and validation coverage
- `tests/test_runtime.py` - runtime orchestration, telemetry, and audit coverage
- `tests/test_long_context.py` - long sequence smoke coverage

### Benchmarks
- `benchmarks/benchmark_bitpack.py` - Pack/unpack throughput and compression ratio
- `benchmarks/benchmark_kv_cache.py` - Store/retrieve latency and quality metrics
- `benchmarks/benchmark_attention.py` - Dense vs sparse latency, top_k_ratio sweep
- `benchmarks/benchmark_end_to_end.py` - Full pipeline benchmark with KV + attention

### Scripts
- `scripts/run_tests.sh` - Execute full test suite
- `scripts/run_benchmarks.sh` - Run all benchmarks
- `scripts/profile_memory.py` - MLX/Metal memory profiling utility

## Features
- **Bit-Packing**: Exact roundtrip for bits 2-8 with validation of edge cases
- **KV Cache**: Grouped symmetric quantization (2-8 bit widths), WHT preconditioning, sign randomization
- **Sparse Attention**: Decode-only block-sparse with proper padding handling and dense fallback
- **Runtime Orchestrator**: Composite cache keys, audit mode, latency timing, failure handling
- **Adaptive Sparsity**: Quality-based top_k_ratio adjustment using audit signals
- **Memory Safety**: MLX memory monitoring with automatic eviction under pressure
- **Telemetry**: Async writer with batching, retries, and ClickHouse backend
- **Testing**: Deterministic unit coverage across core components
- **Benchmarks**: Performance measurements with hardware/software metadata

## Requirements
- Apple Silicon Mac (ARM64)
- macOS 12.0+
- Python 3.10+
- MLX (`pip install mlx`)
- ClickHouse server (optional, for telemetry)

## Installation
```bash
pip install -e .
pip install mlx pytest  # For testing
```

## Usage
```python
import mlx.core as mx
from rfsn_v10 import (
    RFSNTurboQuantKVManager,
    RFSNRuntime,
    AdaptiveSparsityController,
    MemoryGuard
)

# Initialize KV manager
kv_manager = RFSNTurboQuantKVManager(
    k_bits=8, v_bits=3, use_incoherent=True,
    max_memory_gb=1.0, max_pinned_memory_gb=0.5
)

# Initialize runtime
runtime = RFSNRuntime(
    kv_manager=kv_manager,
    model_id="my_model",
    audit_mode=True
)

# Run decode step
q = mx.random.normal((1, 4, 1, 64))  # [B, H, T_q, D]
k = mx.random.normal((1, 4, 128, 64))  # [B, H, T_k, D]
v = mx.random.normal((1, 4, 128, 64))
output, info = runtime.execute_decode_step(
    skill_pattern="transformer",
    layer_id="layer_0",
    batch_id="batch_0",
    queries=q,
    keys=k,
    values=v
)

# Access telemetry
telemetry = runtime.get_telemetry()
```

## Testing
```bash
# Install dependencies
pip install -e .
pip install mlx pytest

# Run full test suite
./scripts/run_tests.sh
# or
pytest -v

# Run specific test suites
pytest tests/test_bitpack.py -v
pytest tests/test_kv_manager.py -v
pytest tests/test_metal_kernel_math.py -v
pytest tests/test_attention.py -v
pytest tests/test_runtime.py -v
pytest tests/test_long_context.py -v
```

## Benchmarking
```bash
# Run all benchmarks
./scripts/run_benchmarks.sh
# or run individually
python3 benchmarks/benchmark_bitpack.py
python3 benchmarks/benchmark_kv_cache.py
python3 benchmarks/benchmark_attention.py
python3 benchmarks/benchmark_end_to_end.py

# Generate proof artifacts (JSON + summary report)
./scripts/run_proof_artifacts.sh
# Optional custom output dir, iterations, and profile
./scripts/run_proof_artifacts.sh artifacts/proof/main11 3 main11

# Compare current proof run vs tracked baseline
python3 scripts/compare_proof_runs.py \
    --profile main11 \
    --baseline-dir benchmarks/proof_baselines/main10 \
    --current-dir artifacts/proof/main11 \
    --output-json artifacts/proof/main11/trend_report.json \
    --output-md artifacts/proof/main11/trend_report.md

# Enforce regression gate (non-zero exit on threshold breach)
python3 scripts/check_proof_regression.py \
    --baseline benchmarks/proof_baselines/main10 \
    --current artifacts/proof/main11 \
    --output-json artifacts/proof/main11/regression_report.json \
    --output-md artifacts/proof/main11/regression_report.md

# Generate plot artifacts from proof JSON
python3 scripts/generate_plots.py \
    --input-dir artifacts/proof/main11 \
    --output-dir results/plots
```

Policy:
- Tune thresholds in `scripts/proof_regression_thresholds.json` only when benchmark noise or hardware/runtime variance is proven to cause false positives across repeated runs.
- Refresh baseline files in `benchmarks/proof_baselines/<profile>/` when performance or quality changes are intentional and accepted after review.
- Do not update thresholds and baseline in the same change unless explicitly documenting why both are necessary.
- KV latency thresholds are intentionally looser than quality thresholds because microbenchmark timing variance is higher than quality metric variance.
- Metal kernel path is an alpha route with strict fallback to sequential reconstruction when unsupported.
- Absolute quality minima should be treated as deployment warnings unless explicitly upgraded to hard-fail policy.
- Current Main11 proof output includes `WARNING_UNSAFE_FOR_LLM_DEPLOYMENT` when sparse absolute quality is below target.

## Memory Profiling
```bash
python3 scripts/profile_memory.py
```

## Design Notes
- Tests are deterministic; wall-clock runtime depends on hardware and MLX availability
- Sparse attention is decode-only (T_q=1) with prefill dense fallback
- KV cache uses grouped symmetric quantization with WHT preconditioning
- Telemetry is written asynchronously to prevent inference stalls
- Memory guard prevents OOM by monitoring MLX/Metal usage
- Benchmarks include hardware/software metadata for reproducibility

## Implementation Status
✅ Core modules compile and integrate in alpha scope
✅ Benchmark scripts and proof plots are present
✅ Custom Metal kernel alpha route and fallback policy are implemented
✅ Telemetry layer is implemented with batched writer support
⚠ MLX-dependent quality and performance validation is environment-dependent
⚠ Sparse quality remains warning-scoped for deployment policy
⚠ Production hardening and end-to-end real-model validation remain in progress
❌ Disk persistence (planned for future)
❌ Partial dequantization (optional optimization)

## Next Steps
1. Run benchmarks to get performance numbers on your hardware
2. Validate with a real LLM (e.g., Llama 3 8B via mlx-lm)
3. Consider adding disk persistence for long-running workloads
4. Explore partial dequantization for further latency improvements