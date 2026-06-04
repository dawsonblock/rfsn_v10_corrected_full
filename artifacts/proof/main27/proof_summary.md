# Proof Summary — Main 27

**Release**: Main 27 — Validation Semantics Correction + Full Rebrand  
**Status**: Alpha  
**Date**: 2026-06-03  
**Hardware**: Apple M2 Pro, 16GB RAM  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  

---

## Release

This release fixes critical validation semantics bugs (multi-position logit metrics, generation cache duplication), rebrands from Main 26 to Main 27, and adds detailed timing/memory breakdown to the throughput benchmark. No new architecture features were added.

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

**Method**: Causal-correct NLL scoring with 64 decode positions across 5 prompts.  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  
**Context**: 512 tokens

| Config | Cosine Mean | Cosine Min | Top1 Match | NLL Δ | KL | Status |
|--------|-------------|------------|------------|-------|-----|--------|
| k8_v5_gs32 | 0.9999 | 0.9996 | 1.000 | -0.00000 | 0.000005 | **PASS** |
| k8_v5_gs64 | 0.9998 | 0.9997 | 1.000 | +0.00008 | 0.000011 | **PASS** |
| k8_v4_gs32 | 0.9995 | 0.9989 | 1.000 | +0.00014 | 0.000016 | **PASS** |
| k8_v4_gs64 | 0.9994 | 0.9988 | 1.000 | +0.00035 | 0.000047 | **PASS** |
| k8_v3_gs64 | 0.9970 | 0.9893 | 1.000 | +0.00005 | 0.000092 | **FAIL** (threshold) |
| k6_v6_gs64 | 0.8353 | 0.8234 | 1.000 | +0.04151 | 0.038586 | **FAIL** |
| k4_v4_gs64 | 0.5404 | 0.4882 | 0.597 | +2.84418 | 2.835200 | **FAIL** |

Four configs pass all quality thresholds with multi-position logit metrics. The best performer is `k8_v5_gs32` (cosine 0.9999, KL 0.000005).

---

## Long-Context Validation

Contexts tested: 512, 1024 tokens  
Positions evaluated: 64 per context

| Config | 512 | 1024 | Passes All |
|--------|-----|------|------------|
| k8_v5_gs32 | PASS | PASS | **YES** |
| k8_v5_gs64 | PASS | PASS | **YES** |
| k8_v4_gs32 | PASS | PASS | **YES** |
| k8_v4_gs64 | PASS | PASS | **YES** |
| k8_v3_gs64 | PASS | PASS | **YES** |
| k6_v6_gs64 | FAIL | FAIL | NO |
| k4_v4_gs64 | FAIL | FAIL | NO |

All configs that passed short-context validation also pass long-context validation at 1024 tokens.

---

## Generation Smoke Test

**Method**: Greedy decode 64 tokens, compare to baseline  
**Context**: 128 tokens

| Config | Token Match | Edit Dist | Repetition | NaN | Status |
|--------|-------------|-----------|------------|-----|--------|
| baseline_fp16 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v4_gs64 | 1.000 | 0.000 | 0.197 | No | PASS |
| k8_v5_gs64 | 1.000 | 0.000 | 0.197 | No | PASS |

All tested configs show perfect token match (1.000) with no NaN logits. The cache-position fix ensures the last context token is not duplicated.

---

## Throughput Benchmark

**Method**: 5 timed repeats after 2 warmup runs  
**Context**: 512 tokens, decode 64 tokens

| Config | Prefill (ms) | Compress (ms) | Decode (ms) | Total (ms) | TPS | Comp Ratio |
|--------|--------------|---------------|-------------|------------|-----|-------------|
| baseline_fp16 | 13.1 | 0.0 | 860.4 | 873.5 | 73.6 | 1.0x |
| k8_v5_gs32 | 14.5 | 912.8 | 856.9 | 1784.2 | 73.4 | 2.3x |
| k8_v4_gs64 | 13.9 | 984.3 | 857.2 | 1855.4 | 73.6 | 2.3x |
| k8_v5_gs64 | 16.4 | 943.6 | 860.5 | 1820.5 | 71.4 | 2.3x |
| k8_v4_gs32 | 16.1 | 932.1 | 870.4 | 1818.6 | 68.6 | 2.3x |

The benchmark now includes detailed timing breakdown (prefill, compress, decode, total) and memory fields (FP16 KV bytes, compressed KV bytes, effective compression ratio). Compression overhead is ~1 second for 512-token context.

---

## MLX Tests

All MLX-dependent tests pass without Metal fallback. See `mlx_test_summary.md` for details.

---

## Conclusion

Main 27 successfully addresses the validation semantics bugs from Main 26:
- Multi-position logit metrics now correctly aggregate over all 64 scored positions
- Generation smoke and throughput benchmarks no longer duplicate the last context token
- Throughput benchmark now reports detailed timing and memory breakdowns

Four quantization configs (k8_v5_gs32, k8_v5_gs64, k8_v4_gs32, k8_v4_gs64) pass all quality thresholds with strong logit fidelity (cosine ≥ 0.9994, KL ≤ 0.00005).
