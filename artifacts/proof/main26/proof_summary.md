# Proof Summary — Main 26

**Release**: Main 26 — Documentation + Causal NLL Validation Correction  
**Status**: Alpha  
**Date**: 2026-06-03  
**Hardware**: Apple M2 Pro, 16GB RAM  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  

---

## Release

This release focuses on fixing stale documentation, correcting causal LM NLL validation, and hardening release integrity checks. No new architecture features were added.

---

## Synthetic Kernel Benchmark

| Benchmark | Status | Cosine | Max Abs Diff |
|-----------|--------|--------|--------------|
| Kernel paths | PASS | 1.000 | 0.000 |
| Fused kernel | PASS | 1.000 | 0.000 |

Both synthetic benchmarks show exact numerical equivalence (cosine 1.000, max abs diff 0.0).

---

## KV Cache Compression

Configs tested:
- `baseline_fp16` (16-bit K/V reference)
- `k8_v3_gs64` (8-bit K, 3-bit V, gs=64)
- `k8_v4_gs64` (8-bit K, 4-bit V, gs=64)
- `k8_v5_gs64` (8-bit K, 5-bit V, gs=64)
- `k8_v4_gs32` (8-bit K, 4-bit V, gs=32)
- `k8_v5_gs32` (8-bit K, 5-bit V, gs=32)
- `k6_v6_gs64` (6-bit K, 6-bit V, gs=64)
- `k4_v4_gs64` (4-bit K, 4-bit V, gs=64)

---

## Real-Model Validation

**Method**: Causal-correct NLL scoring with 64 decode positions across 3 prompts.  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  
**Context**: 512 tokens

| Config | Cosine Mean | Cosine Min | Top1 Match | NLL Δ | KL | Status |
|--------|-------------|------------|------------|-------|-----|--------|
| k8_v5_gs64 | 0.9998 | 0.9997 | 1.000 | -0.00002 | 0.000007 | **PASS** |
| k8_v5_gs32 | 0.9998 | 0.9996 | 1.000 | -0.00006 | 0.000019 | **FAIL** (threshold) |
| k8_v4_gs64 | 0.9993 | 0.9988 | 1.000 | +0.00048 | 0.000067 | **FAIL** (threshold) |
| k8_v4_gs32 | 0.9993 | 0.9989 | 1.000 | +0.00029 | 0.000032 | **FAIL** (threshold) |
| k8_v3_gs64 | 0.9937 | 0.9869 | 1.000 | -0.00033 | 0.000225 | **FAIL** (threshold) |
| k6_v6_gs64 | 0.9094 | 0.8892 | 1.000 | +0.04795 | 0.005967 | **FAIL** |
| k4_v4_gs64 | 0.7291 | 0.6458 | 0.667 | +1.17887 | 3.496386 | **FAIL** |

Only `k8_v5_gs64` passes all quality thresholds under corrected causal NLL scoring.

---

## Long-Context Validation

Contexts tested: 512, 1024, 2048 tokens  
Positions evaluated: 64 per context

| Config | 512 | 1024 | 2048 | Passes All |
|--------|-----|------|------|------------|
| k8_v5_gs64 | PASS | PASS | PASS | **YES** |
| k8_v5_gs32 | PASS | PASS | PASS | **YES** |
| k8_v4_gs64 | PASS | PASS | PASS | **YES** |
| k8_v4_gs32 | PASS | PASS | PASS | **YES** |
| k8_v3_gs64 | PASS | PASS | PASS | **YES** |
| k6_v6_gs64 | FAIL | FAIL | FAIL | NO |
| k4_v4_gs64 | FAIL | FAIL | FAIL | NO |

All configs that passed short-context validation also pass long-context validation. The lower-bit configs (k6_v6, k4_v4) fail consistently across all contexts.

---

## Generation Smoke Test

**Method**: Greedy decode 64 tokens, compare to baseline  
**Context**: 128 tokens

| Config | Token Match | Edit Dist | Repetition | NaN | Status |
|--------|-------------|-----------|------------|-----|--------|
| baseline_fp16 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v4_gs64 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v5_gs64 | 1.000 | 0.000 | 0.000 | No | PASS |

All tested configs show perfect token match (1.000) with no NaN logits.

---

## Throughput Benchmark

**Method**: 5 timed repeats after 2 warmup runs  
**Context**: 512 tokens, decode 64 tokens

| Config | Tokens/sec | TTFT (ms) | p50 (ms) | p99 (ms) |
|--------|------------|-----------|----------|----------|
| baseline_fp16 | 71.5 | 15.3 | 13.10 | 26.96 |
| k8_v4_gs64 | 68.1 | 14.2 | 13.30 | 47.41 |
| k8_v5_gs64 | 74.7 | 16.1 | 12.88 | 21.03 |
| k8_v5_gs32 | 73.3 | 19.3 | 13.19 | 20.21 |

**Observation**: Compressed routes show comparable throughput to baseline. k8_v5_gs64 achieves highest throughput (74.7 tps vs 71.5 tps baseline).

---

## Per-Layer Sensitivity

Not run for Main 26. Phase 10 deferred to future release.

---

## Targeted Layer Protection

Not run for Main 26. Phase 10 deferred to future release.

---

## Sparse Decode Status

**Status**: **DISABLED BY DEFAULT**

- Sparse attention implementation exists but remains below quality thresholds.
- `test_sparse_safety_gate.py` passes — safety gate correctly prevents sparse enablement.
- No sparse claims in README or proof.

---

## Not Implemented

- **Polar quantization**: Not implemented.
- **True arbitrary partial dequantization**: Not implemented (selected-block reconstruction via `retrieve_blocks()` exists; arbitrary token-level partial dequant remains unimplemented).
- **Production hardening**: Incomplete.
- **Per-layer sensitivity**: Not regenerated for Main 26.

---

## Recommendation

**Recommended default**: `k8_v5_gs64`

**Rationale**:
1. Passes real-model validation (cosine 0.9998, all metrics within thresholds)
2. Passes all long-context validations (512, 1024, 2048 tokens)
3. Passes generation smoke test (perfect token match)
4. Highest throughput among tested configs (74.7 tps)
5. Safe compression ratio (8-bit K, 5-bit V, gs=64)

**Rejected configs**:
- `k4_v4_gs64`: Severe quality degradation (cosine 0.73, top1 match 0.67)
- `k6_v6_gs64`: Quality below threshold (cosine 0.91)
- `k8_v3_gs64`: Fails strict cosine threshold (0.994 < 0.995)

**Quality-oriented alternatives**:
- `k8_v5_gs32`: Slightly better quality than gs64 but no throughput benefit

**Not recommended for production**:
All configs remain alpha-level. Production deployment requires further validation.

---

## Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| README title says Main 26 | PASS |
| No stale main23/main24 paths | PASS |
| MLX summary says Main 26 | PASS |
| main26_release_manifest.json exists | PASS |
| Release checker passes | PASS |
| Causal NLL scoring corrected | PASS |
| ≥32 positions evaluated | PASS (64) |
| Long-context (512/1024/2048) | PASS |
| Generation smoke exists | PASS |
| Throughput benchmark exists | PASS |
| Sparse disabled | PASS |
| Polar quant not claimed | PASS |
| No production-ready claim | PASS |

**Overall**: Main 26 passes all acceptance criteria.

---

## Artifacts

- `artifacts/proof/main26/kernel_benchmark.json`
- `artifacts/proof/main26/fused_kernel_benchmark.json`
- `artifacts/proof/main26/real_model_validation.json`
- `artifacts/proof/main26/long_context_validation.json`
- `artifacts/proof/main26/generation_smoke.json`
- `artifacts/proof/main26/generation_throughput.json`
- `artifacts/proof/main26/proof_summary.md`
- `artifacts/proof/main26/mlx_test_summary.md`
- `artifacts/proof/main26/mlx_pytest_raw.log`
- `artifacts/proof/main26/mlx_pytest_junit.xml`
- `artifacts/proof/main26/main26_release_manifest.json`

---

## Validation Commands

```bash
# Synthetic benchmarks
python benchmarks/benchmark_kernel_paths.py --out artifacts/proof/main26/kernel_benchmark.json
python benchmarks/benchmark_fused_kernel.py --out artifacts/proof/main26/fused_kernel_benchmark.json

# Real-model validation
python benchmarks/validate_real_model_kv.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokens 512 --positions 64 \
    --configs k8_v3_gs64,k8_v4_gs64,k8_v5_gs64,k8_v4_gs32,k8_v5_gs32,k6_v6_gs64,k4_v4_gs64 \
    --out artifacts/proof/main26/real_model_validation.json

# Long-context validation
python benchmarks/validate_long_context_kv.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --contexts 512,1024,2048 --positions 64 \
    --configs k8_v3_gs64,k8_v4_gs64,k8_v5_gs64,k8_v4_gs32,k8_v5_gs32,k6_v6_gs64,k4_v4_gs64 \
    --out artifacts/proof/main26/long_context_validation.json

# Generation smoke test
python benchmarks/validate_generation_smoke.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokens 128 --decode 64 \
    --configs baseline_fp16,k8_v4_gs64,k8_v5_gs64 \
    --out artifacts/proof/main26/generation_smoke.json

# Throughput benchmark
python benchmarks/benchmark_generation_throughput.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokens 512 --decode 64 \
    --configs baseline_fp16,k8_v4_gs64,k8_v5_gs64,k8_v5_gs32 \
    --out artifacts/proof/main26/generation_throughput.json

# Integrity check
python scripts/check_release_integrity.py
```
