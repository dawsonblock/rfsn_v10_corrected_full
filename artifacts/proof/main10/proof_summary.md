# main10 Proof Summary

Generated: 2026-06-02T08:33:41.331643+00:00

## Files
- kv_cache_runs.json
- e2e_scenarios.json
- kernel_benchmark.json (not generated)
- real_model_validation.json (not generated)

## Highlights
- Fastest KV retrieve: 0.62ms (shape=(1, 8, 1024, 64)|k=8|v=8|incoherent=False)
- KV value quality (same run): cos=1.0000, rel_mae=0.0063, max_abs=0.0199
- Best sparse scenario: sparse_topk_075_sink1_recent2 miss=18.96ms, hit=10.13ms, sparse_cos=0.8836325407028198
- Worst sparse scenario: sparse_disabled_by_default hit_mode=dense_requested sparse_cos=None sparse_rel_mae=None
- Dense decode path: miss=10.09ms, hit=6.14ms, mode=dense_requested

## Absolute Quality
- Sparse quality: warn (min=0.4996378421783447, threshold=0.900)
- Quant quality: pass (min=0.9703969359397888, threshold=0.950)
- KV cache quality: pass (min=0.9699910283088684, threshold=0.900)
- Sparse default: disabled
- WARNING_UNSAFE_FOR_LLM_DEPLOYMENT
- Sparse deployment threshold met: no
- Recommended default: dense (sparse decode remains experimental and should default to disabled)
- Kernel benchmark: not run
- Real model validation: not run

## Next Checks
- Compare these artifacts against previous runs for trend regressions.
- Keep sparse vs dense and quant vs sparse audit metrics split in analysis.
