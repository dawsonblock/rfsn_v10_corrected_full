# Main 23 Proof Summary

Release: Main 23 — Real-Model Validation + Proof Hardening Release
Generated: 2026-06-03T01:46:14.806951

## Release
Main 23 proves synthetic fused/block-aware KV behavior and adds real-model validation.
Sparse decode remains disabled by default.
True arbitrary partial dequantization is not implemented.
Polar quantization is not implemented.
Production deployment is not claimed.

## KV Cache Status
- Synthetic tensor proof: pass
- Compression ratio: 0.265625 (best case)
- Key quality: pass (cosine >= 0.999)
- Value quality: pass above 0.90 threshold

## Kernel Benchmark Status
- Total modes tested: 24
- Invalid full-equivalent routes: 0
- Best speedup: ~33.3x
- Fallback observed: no

## Fused Kernel Status
- All valid: True
- Min cosine vs reference: 1.000000
- Max abs diff vs reference: 0.0

## Block-Aware Retrieval Status
- retrieve_blocks() selected-block path: tested and passes
- Per-block multi-kernel reconstruction with global-index sign correction: active

## Optimization Sweep
- Configurations tested: 8
- Total rows: 32
- Best min cosine: 0.99427
- Rejected configs: 2-bit variants

## Real-Model Validation
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Tokens tested: 512
- Configs tested: 3
- Passed: 0
- Failed: 2
  - k8_v3_gs64: cosine=0.998785, top1=1.000, top5=1.000, status=fail
  - k4_v4_gs64: cosine=0.820265, top1=1.000, top5=0.600, status=fail
- Real-model validation failed alpha thresholds.
  RFSN should remain synthetic-proof alpha only.

## Long-Context Validation
- Contexts tested: [512, 1024]
  - 512 tokens: 0/3 configs passed
  - 1024 tokens: 0/3 configs passed

## Sparse Decode Status
- Sparse decode is disabled by default.
- Current sparse max cosine is below threshold.
- sparse_enabled in validation: False

## Known Limitations
- Polar quantization is not implemented.
- True arbitrary partial dequantization is not implemented.
- Real-model validation is alpha-level; some configs failed thresholds.
- Production deployment is not claimed.

## Recommendation
- Compression-oriented default: 8-bit K / 3-bit V / group_size 64 (close to threshold, alpha only)
- Quality-oriented candidate: 4-bit K / 4-bit V / group_size 64 (failed real-model threshold)
- Rejected: 2-bit configs
- Sparse decode: remain disabled
- RFSN remains synthetic-proof alpha only.
- Not production ready.
