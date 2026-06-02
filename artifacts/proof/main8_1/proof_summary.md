# main8_1 Proof Summary

Generated: 2026-06-01T21:40:47.025578+00:00

## Files
- kv_cache_runs.json
- e2e_scenarios.json

## Highlights
- Fastest KV retrieve: 0.93ms (shape=(1, 8, 1024, 64)|k=8|v=8|incoherent=False)
- KV value quality (same run): cos=1.0000, rel_mae=0.0063, max_abs=0.0199
- Sparse decode path: miss=14.63ms, hit=7.25ms, quant_cos=0.9678987264633179
- Dense decode path: miss=11.96ms, hit=6.38ms, mode=dense_prefill

## Next Checks
- Compare these artifacts against previous runs for trend regressions.
- Keep sparse vs dense and quant vs sparse audit metrics split in analysis.
