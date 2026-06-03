# RFSN v10 Main 24 — Real-Model Validation + Proof Hardening Release

## Status: RFSN v10 Main 24 — Real-Model Validation + Proof Hardening Release

Implemented:
- low-bit KV cache compression
- sequential reference reconstruction route
- multi-kernel Metal reconstruction route
- fused packed-dequant-WHT-sign Metal kernel source path
- sparse safety gate
- kernel benchmark artifacts
- synthetic proof artifacts

Proof status:
- multi-kernel route: benchmarked
- fused route: proven by fused_kernel_benchmark.json (cosine 1.000, max_abs_diff 0.0)
- sparse decode: below threshold, disabled by default
- real-model validation: alpha-level on real non-random model (Qwen/Qwen2.5-0.5B-Instruct)
- long-context validation: included
- polar quant: not implemented
- true arbitrary partial dequantization: not implemented

Not claimed:
- production LLM deployment
- sparse safe inference
- universal speedup
- polar quantization
- true arbitrary partial dequantization

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

Kernel status:
- Implemented: Metal sign kernel, Metal packed-dequant kernel, Metal WHT64 kernel, multi-kernel Metal reconstruction route, fused packed-dequant-WHT-sign Metal kernel source path
- Proof validation: fused route proven by fused_kernel_benchmark.json (cosine 1.000, max_abs_diff 0.0)
- Block-aware retrieval: selected-block reconstruction (`retrieve_blocks()`) uses per-block multi-kernel reconstruction with global-index sign correction. It does not always use the fused full-route kernel.

## Requirements
- Apple Silicon Mac (ARM64)
- macOS 12.0+
- Python 3.10+
- MLX (`pip install mlx`)
- ClickHouse server (optional, for telemetry)

## Installation
```bash
pip install -e ".[dev]"
pip install -e ".[dev,real_model]"  # Optional: real-model validation (torch + transformers)
pip install -e ".[production]"  # Optional: production validation (huggingface_hub)
pip install mlx  # If not already installed
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
./scripts/run_proof_artifacts.sh artifacts/proof/main23 3 main23

# Compare current proof run vs tracked baseline
python3 scripts/compare_proof_runs.py \
    --profile main23 \
    --baseline-dir benchmarks/proof_baselines/main10 \
    --current-dir artifacts/proof/main23 \
    --output-json artifacts/proof/main23/trend_report.json \
    --output-md artifacts/proof/main23/trend_report.md

# Enforce regression gate (non-zero exit on threshold breach)
python3 scripts/check_proof_regression.py \
    --baseline benchmarks/proof_baselines/main10 \
    --current artifacts/proof/main23 \
    --output-json artifacts/proof/main23/regression_report.json \
    --output-md artifacts/proof/main23/regression_report.md

# Generate kernel benchmark evidence
python3 benchmarks/benchmark_kernel_paths.py \
    --out artifacts/proof/main23/kernel_benchmark.json

# Generate plot artifacts from proof JSON
python3 scripts/generate_plots.py \
    --input-dir artifacts/proof/main23 \
    --output-dir results/plots

# Real-model validation (auto-downloads from HuggingFace)
python3 benchmarks/validate_real_model_kv.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokens 512 \
    --configs k8_v3_gs64,k4_v4_gs64 \
    --out artifacts/proof/main23/real_model_validation.json

# Long-context validation
python3 benchmarks/validate_real_model_kv.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --contexts 512,1024,2048 \
    --configs k8_v3_gs64,k4_v4_gs64 \
    --out artifacts/proof/main23/long_context_validation.json

# Production-grade model validation
# Download a model first:
python tools/model_download.py mistral-7b --output-dir models
# Then run comprehensive validation:
python benchmarks/validate_production_model.py \
    --model-path models/mistral-7b \
    --prompt-suite prompts/validation_suite.json \
    --out artifacts/proof/main23/production_validation.json
# Check against baseline:
python scripts/check_production_regression.py \
    --results artifacts/proof/main23/production_validation.json \
    --baseline benchmarks/production_baseline.json
```

Policy:
- Tune thresholds in `scripts/proof_regression_thresholds.json` only when benchmark noise or hardware/runtime variance is proven to cause false positives across repeated runs.
- Refresh baseline files in `benchmarks/proof_baselines/<profile>/` when performance or quality changes are intentional and accepted after review.
- Do not update thresholds and baseline in the same change unless explicitly documenting why both are necessary.
- KV latency thresholds are intentionally looser than quality thresholds because microbenchmark timing variance is higher than quality metric variance.
- Metal kernel path is an alpha route with strict fallback to sequential reconstruction when unsupported.
- Absolute quality minima should be treated as deployment warnings unless explicitly upgraded to hard-fail policy.
- Current proof output includes `WARNING_UNSAFE_FOR_LLM_DEPLOYMENT` when sparse absolute quality is below target.
- Sparse decode is experimental and disabled by default unless a profile passes safety gates.

## Memory Profiling
```bash
python3 scripts/profile_memory.py
```

## Proof Artifacts

All Main 24 proof artifacts are in `artifacts/proof/main24/`.

Note: The `artifacts/proof/main12` path is historical. Main 22 artifacts have been copied to `artifacts/proof/history/main22/` for reference.

## Recommended Configs

- **Compression-oriented default**: 8-bit K / 3-bit V / group_size 64
- **Quality-oriented candidate**: 4-bit K / 4-bit V / group_size 64
- **Rejected**: 2-bit configs (quality too low for alpha thresholds)

## Real-Model Validation

Main 24 includes real non-random model validation on `Qwen/Qwen2.5-0.5B-Instruct`. Results are alpha-level: quality metrics are reported honestly with pass/fail thresholds. If thresholds are not met, the config is marked as failed.

## Sparse Decode Status

Sparse decode is **disabled by default**. Current sparse max cosine is below the deployment threshold. Do not enable sparse decode unless you are explicitly testing the safety gate.

## Known Limitations

- Polar quantization is not implemented.
- True arbitrary partial dequantization is not implemented (selected-block reconstruction via `retrieve_blocks()` exists; arbitrary token-level partial dequant remains unimplemented).
- Production hardening and end-to-end real-model validation remain in progress.
- RFSN is not production-ready.

## How to Reproduce

```bash
# Install
pip install -e ".[dev,real_model]"
pip install mlx

# Run synthetic proof benchmarks
python benchmarks/benchmark_kernel_paths.py --out artifacts/proof/main23/kernel_benchmark.json
python benchmarks/benchmark_fused_kernel.py --out artifacts/proof/main23/fused_kernel_benchmark.json
python benchmarks/benchmark_optimizations.py --out artifacts/proof/main23/optimization_benchmark.json

# Run real-model validation
python benchmarks/validate_real_model_kv.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokens 512 \
    --configs k8_v3_gs64,k4_v4_gs64 \
    --out artifacts/proof/main23/real_model_validation.json

# Run long-context validation
python benchmarks/validate_real_model_kv.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --contexts 512,1024,2048 \
    --configs k8_v3_gs64,k4_v4_gs64 \
    --out artifacts/proof/main23/long_context_validation.json

# Run release integrity check
python scripts/check_release_integrity.py
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
✅ Metal packed-dequant, Metal WHT64, and Metal sign kernels are implemented
✅ Telemetry layer is implemented with batched writer support
✅ Real-model validation on non-random model (alpha-level)
✅ Long-context validation included
⚠ MLX-dependent quality and performance validation is environment-dependent
⚠ Sparse quality remains warning-scoped; sparse decode defaults to disabled
⚠ Production hardening remains in progress
❌ Polar quantization (not implemented)
❌ True arbitrary partial dequantization (selected-block reconstruction exists via retrieve_blocks(); arbitrary token-level partial dequant remains unimplemented)
❌ Disk persistence (planned for future)

## Next Steps
1. Run benchmarks to get performance numbers on your hardware
2. Review real-model validation results and adjust compression configs if needed
3. Consider adding disk persistence for long-running workloads
4. Evaluate polar quantization for future quality improvement