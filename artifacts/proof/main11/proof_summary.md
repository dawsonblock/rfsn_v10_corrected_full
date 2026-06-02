# main11 Proof Summary

Generated: 2026-06-02T04:50:32.934929+00:00

## Files
- kv_cache_runs.json
- e2e_scenarios.json

## Highlights
- Fastest KV retrieve: 0.91ms (shape=(1, 8, 1024, 64)|k=8|v=8|incoherent=False)
- KV value quality (same run): cos=1.0000, rel_mae=0.0063, max_abs=0.0199
- Sparse decode path: miss=10.46ms, hit=6.86ms, quant_cos=0.970818281173706
- Dense decode path: miss=9.02ms, hit=4.79ms, mode=dense_prefill

## Absolute Quality
- Sparse quality: warn (min=0.7554906606674194, threshold=0.900)
- Quant quality: pass (min=0.9703969359397888, threshold=0.950)
- Value quality: pass (min=0.9699910283088684, threshold=0.900)
- WARNING_UNSAFE_FOR_LLM_DEPLOYMENT

## Next Checks
- Compare these artifacts against previous runs for trend regressions.
- Keep sparse vs dense and quant vs sparse audit metrics split in analysis.
