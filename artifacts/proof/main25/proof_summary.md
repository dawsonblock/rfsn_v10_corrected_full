# Proof Summary — Main 25

**Release**: Main 25 — Validation Repair
**Generated**: 2026-06-03T19:58:51.383549+00:00

## Key Fixes vs Main 24

- NaN metrics now produce `nan_fail` (was incorrectly `pass`).
- Long-context model runs at float32 on MPS (eliminates fp16 overflow at 1024/2048).
- 5-position multi-step NLL replaces single-token decode.
- `k4_v4_gs64` removed from recommendations.

## Real-Model Validation (512 tokens, 5 positions)

- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Passing configs**: 5 / 7
- **Best config**: k8_v5_gs32

| Config | Cosine | Top1 | Top5 | NLL delta | Status |
|--------|--------|------|------|-----------|--------|
| baseline_fp16 | 1.000000 | 1.000 | 1.000 | 0.000000 | reference |
| k8_v3_gs64 | 0.998214 | 1.000 | 1.000 | 0.045364 | pass |
| k8_v4_gs64 | 0.999721 | 1.000 | 1.000 | 0.037749 | pass |
| k8_v5_gs64 | 0.999873 | 1.000 | 1.000 | 0.041394 | pass |
| k6_v6_gs64 | 0.999680 | 1.000 | 1.000 | -1.337401 | fail |
| k8_v4_gs32 | 0.999768 | 1.000 | 1.000 | 0.010575 | pass |
| k8_v5_gs32 | 0.999952 | 1.000 | 1.000 | 0.037418 | pass |
| k4_v4_gs64 | 0.967693 | 1.000 | 0.800 | -4.358963 | fail |

## Long-Context Validation (float32 on MPS)

| Tokens | Config | Cosine | NLL delta | Status |
|--------|--------|--------|-----------|--------|
| 512 | baseline_fp16 | 1.000000 | 0.000000 | reference |
| 512 | k8_v3_gs64 | 0.998329 | 0.019049 | pass |
| 512 | k8_v4_gs64 | 0.999678 | 0.050525 | pass |
| 512 | k8_v5_gs64 | 0.999929 | 0.003630 | pass |
| 512 | k6_v6_gs64 | 0.999777 | -2.189975 | fail |
| 512 | k8_v4_gs32 | 0.999783 | 0.020426 | pass |
| 512 | k8_v5_gs32 | 0.999959 | 0.064736 | pass |
| 512 | k4_v4_gs64 | 0.964758 | -4.493655 | fail |
| 1024 | baseline_fp16 | 1.000000 | 0.000000 | reference |
| 1024 | k8_v3_gs64 | 0.994939 | 0.366423 | fail |
| 1024 | k8_v4_gs64 | 0.998536 | -0.019320 | pass |
| 1024 | k8_v5_gs64 | 0.999641 | -0.013003 | pass |
| 1024 | k6_v6_gs64 | 0.992746 | -1.584154 | fail |
| 1024 | k8_v4_gs32 | 0.998765 | 0.010794 | pass |
| 1024 | k8_v5_gs32 | 0.999763 | 0.022533 | pass |
| 1024 | k4_v4_gs64 | 0.558019 | -5.934061 | fail |
| 2048 | baseline_fp16 | 1.000000 | 0.000000 | reference |
| 2048 | k8_v3_gs64 | 0.999100 | 0.039114 | pass |
| 2048 | k8_v4_gs64 | 0.999873 | -0.072210 | pass |
| 2048 | k8_v5_gs64 | 0.999939 | 0.026417 | pass |
| 2048 | k6_v6_gs64 | 0.998551 | -4.894584 | fail |
| 2048 | k8_v4_gs32 | 0.999858 | 0.000570 | pass |
| 2048 | k8_v5_gs32 | 0.999916 | 0.029955 | pass |
| 2048 | k4_v4_gs64 | 0.763143 | -8.690442 | fail |

**Recommended default**: `k8_v5_gs64`

## Per-Layer Sensitivity

Analyzed 24 layers individually with `k8_v3_gs64`.

## Early-Layer Protection

Tested 4 scenarios (protect 2/4/6/8 early layers at fp16).

## Status

- Sparse decode: **disabled**
- Polar quantization: **not implemented**
- Production readiness: **no** (alpha)
