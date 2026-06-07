# RFSN v10 — Release Manifest

## Release identification

| Field | Value |
|-------|-------|
| Release name | `rfsn_v10_qjl_alpha_candidate_3` |
| Git branch | `qjl-beta-repair-finalization` |
| Git commit | `4ddebf2` (beta: production inference server + telemetry reconciliation) |
| Broken snapshot tag | `qjl-beta-repair-broken-snapshot` (preserved, do not delete) |
| Build date | 2026-06-07 |
| Python requirement | `>=3.11,<3.13` |
| Development status | `3 - Alpha` |

---

## Stable configurations

These are the only quantization presets validated for use:

| Config | Type | Notes |
|--------|------|-------|
| `k8_v5_gs32` | 8-bit KV, 5-group, gs=32 | Default — recommended |
| `k8_v5_gs64` | 8-bit KV, 5-group, gs=64 | Also validated |

---

## Experimental configurations (disabled by default)

| Feature | Flag | Status |
|---------|------|--------|
| QJL score correction | `experimental.enable_qjl: true` | Not validated — do not use |
| Polar / fused IsoQuant-Polar | `experimental.enable_polar: true` | Not validated — do not use |
| Adaptive sparse controller | `experimental.enable_adaptive: true` | Not validated — do not use |

---

## Gate results

### Non-MLX gate (Linux / any platform)

| Step | Result |
|------|--------|
| `python -m compileall -q rfsn_v10 tests` | PASS |
| `pytest --collect-only -q` | PASS — 67 files, 0 errors |
| `pytest tests/test_no_placeholder_source.py` | PASS |
| `pytest tests/test_runtime_import_contract.py` | PASS |
| `pytest tests/test_config.py tests/test_config_strict.py` | PASS |
| `pytest tests/test_health.py` | PASS |
| `pytest tests/test_no_runtime_raw_sdpa.py` | PASS |
| `pytest tests/test_experimental_flags.py` | PASS |
| `pytest tests/test_quantization_lazy_imports.py` | PASS |
| `pytest tests/test_clickhouse_security.py` | PASS (34 tests) |
| `pytest tests/test_telemetry_e2e.py` | PASS (12 tests) |
| `RFSN_BACKEND=numpy python -m rfsn_v10 healthcheck` | PASS |
| `python -m build` | PASS |
| Wheel subpackage content check | PASS — rfsn_v10, kernels, quantization, runtime all present |
| Wheel install + import verify (Python 3.11 venv) | PASS |

### Apple Silicon MLX gate

| Step | Result |
|------|--------|
| `pytest tests/test_attention.py` | PASS (12 tests) |
| `pytest tests/test_attention_causal_mask.py` | PASS (6 tests) |
| `pytest tests/test_bitpack.py` | PASS (28 tests) |
| `pytest tests/test_bitpack_fuzz.py` | PASS (5 tests) |
| `pytest tests/test_drift.py` | PASS (3 tests) |
| `pytest tests/test_kv_manager.py` | PASS (47 tests) |
| `pytest tests/test_short_prompt_decode_drift.py` | PASS (4 tests) |
| `pytest tests/test_prefill_decode_split.py` | PASS (5 tests) |
| `pytest tests/test_short_prompt_generation_regression.py` | PASS (4 tests) |
| `RFSN_BACKEND=mlx python -m rfsn_v10 healthcheck` | PASS |

Total gate tests: **882 passed, 15 skipped, 0 failed**

### Docker gate

| Step | Result |
|------|--------|
| `docker build -t rfsn-qjl .` | Requires Docker daemon — not run in this session |
| `docker run --rm -e RFSN_BACKEND=numpy rfsn-qjl` | Requires Docker daemon |

Docker gate must be run manually before promoting to beta.

---

## Quality thresholds (measured on Apple Silicon, synthetic KV tensors)

These are **measured** values, not assumed:

| Metric | Threshold | Basis |
|--------|-----------|-------|
| Cosine similarity (decode step) | ≥ 0.998 | Measured across k8_v5_gs32 4-head synthetic |
| KL divergence | ≤ 1e-6 | Measured |
| Top-5 overlap | ≥ 0.95 | Measured |

---

## Known limitations

1. **QJL fails its own artifact**: score MAE 0.1051 vs baseline 0.0824, top-k overlap 0.8 — disabled and unsupported
2. **Polar/adaptive not validated**: quality degradation observed in short-prompt generation
3. **No CUDA backend**: MLX (Apple Silicon) only for the quantized path
4. **FastAPI server implemented**: `/v1/chat/completions` with SSE streaming (set `RFSN_MODEL_ID` env var)
5. **Full sparse prefill not implemented**: prefill always uses dense attention
6. **End-to-end speedup not proven**: compression overhead dominates at short contexts
7. **Docker gate not run in CI on this machine**: must be verified manually

---

## Source integrity

The following files were **invalid placeholder text** in the broken snapshot and are now valid disabled stubs:

| File | Fix applied |
|------|-------------|
| `rfsn_v10/isoquant_precondition.py` | Replaced with `IsoQuantPreconditioner` stub raising `_ExperimentalNotImplemented` |
| `rfsn_v10/quantization/fused_isoquant_polar.py` | Replaced with `FusedIsoQuantPolar` stub raising `_ExperimentalNotImplemented` |
| `rfsn_v10/quantization/kv_quant_manager.py` | Trailing placeholder line removed; 216 lines of real code preserved |

Guard test: `tests/test_no_placeholder_source.py` — prevents regression.

---

## Beta promotion checklist

The classifier must remain `3 - Alpha` until **all** items below are checked:

- [x] `python -m compileall -q rfsn_v10 tests` — PASS
- [x] `pytest tests/test_no_placeholder_source.py` — PASS
- [x] `pytest tests/test_runtime_import_contract.py` — PASS
- [x] Full non-MLX test gate — PASS (882 tests)
- [x] Full MLX gate — PASS (110 tests)
- [x] Wheel builds and subpackages install correctly
- [ ] `docker build -t rfsn-qjl . && docker run --rm -e RFSN_BACKEND=numpy rfsn-qjl` — **pending manual Docker run**
- [ ] Benchmarks re-run with quality + speed metrics side by side
- [x] README updated with inference server section
- [ ] Archive created with `git archive` (not Finder zip)

---

## Archive instructions

**Do not zip from Finder.** Use:

```bash
git archive --format=zip HEAD -o rfsn_v10_qjl_alpha_candidate_3.zip
```

After all beta promotion checklist items are complete:

```bash
git archive --format=zip HEAD -o rfsn_v10_qjl_beta_candidate_1.zip
```

Verify the archive is clean:

```bash
python -c "
import zipfile, sys
with zipfile.ZipFile('rfsn_v10_qjl_alpha_candidate_3.zip') as z:
    bad = [n for n in z.namelist() if '__pycache__' in n or n.endswith('.pyc') or '.DS_Store' in n]
    print(f'{len(z.namelist())} files, {len(bad)} junk files')
    if bad: sys.exit(1)
"
```
