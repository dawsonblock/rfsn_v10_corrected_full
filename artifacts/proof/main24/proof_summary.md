# Proof Summary — Main 24

**Release**: Main 24 — Quality Tuning + Validation Repair
**Generated**: 2026-06-03T08:49:24.091129+00:00

## Real-Model Validation

- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Tokens tested**: 512
- **Metric**: NLL delta (replaced single-token PPL)
- **Passing configs**: 5 / 7
- **Best config**: k8_v5_gs64

### Results

| Config | Cosine | Top1 | Top5 | NLL Δ | Status |
|--------|--------|------|------|-------|--------|
| baseline_fp16 | 1.000000 | 1.000 | 1.000 | 0.000000 | reference |
| k8_v3_gs64 | 0.998785 | 1.000 | 1.000 | -0.132763 | pass |
| k8_v4_gs64 | 0.999786 | 1.000 | 1.000 | 0.335916 | pass |
| k8_v5_gs64 | 0.999948 | 1.000 | 1.000 | -0.078100 | pass |
| k6_v6_gs64 | 0.956862 | 1.000 | 0.600 | -3.414137 | fail |
| k8_v4_gs32 | 0.999795 | 1.000 | 1.000 | 0.289045 | pass |
| k8_v5_gs32 | 0.999947 | 1.000 | 1.000 | -0.062477 | pass |
| k4_v4_gs64 | 0.820265 | 1.000 | 0.600 | -5.910890 | fail |

## Long-Context Validation

Tested at 512, 1024, and 2048 tokens.

## Sparse Decode

Disabled by default. No polar quantization.

## Known Limitations

- k6_v6_gs64 and k4_v4_gs64 fail alpha thresholds (expected).
- Polar quantization not implemented.
