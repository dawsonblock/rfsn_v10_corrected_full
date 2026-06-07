# RFSN v10 qjl17 — Alpha Repair Build

## Status

**Alpha repair build.** This is not production ready.

| Path | Status |
|------|--------|
| Stable runtime (MLX 8-bit KV compression) | Alpha — validated on Apple Silicon |
| Sparse decode | Disabled by default — decode-only, not end-to-end proven |
| QJL score correction | Experimental — disabled by default |
| Polar / hybrid quantization | Experimental — disabled by default |
| Adaptive sparse controller | Experimental — disabled by default |
| CUDA backend | **Not implemented** |
| Full portable runtime | Not implemented — MLX required for core runtime |
| End-to-end speedup | Not proven — decode TPS comparable, total slower due to compression overhead |
| Production deployment | Not supported |

---

## Platform Support

| Platform | Status |
|----------|--------|
| Apple Silicon + MLX | Supported (primary runtime) |
| NumPy CPU | Partial — kernel validation only, not full runtime |
| CUDA | Not implemented |

---

## Quick Start

```bash
# Requires Python 3.11 and Apple Silicon with MLX installed
pip install -e .

# Check health
python -m rfsn_v10 healthcheck

# Run stable tests (no MLX required)
pytest tests/test_config.py tests/test_config_strict.py tests/test_kernels_validation.py -q

# Run MLX-dependent tests (Apple Silicon required)
pytest tests/test_attention.py tests/test_bitpack.py tests/test_kv_manager.py -q
```

---

## Architecture

### Stable Modules

- `rfsn_v10/bitpack.py` — Bit-packed quantizer (2-8 bit widths) with exact roundtrip guarantees
- `rfsn_v10/kv_manager.py` — TurboQuant KV manager with grouped symmetric quantization and WHT preconditioning
- `rfsn_v10/attention.py` — Adaptive block-sparse attention; all dense fallbacks route through `attention_reference.py`
- `rfsn_v10/attention_reference.py` — Canonical causal attention reference (always applies causal mask for T_q > 1)
- `rfsn_v10/runtime.py` — Orchestrator integrating KV cache, sparse attention, audit mode, and telemetry
- `rfsn_v10/config.py` — Strict Pydantic config (extra='forbid' on all models)
- `rfsn_v10/health.py` — Health check system; returns UNHEALTHY until checks have been run
- `rfsn_v10/clickhouse_client.py` — HMAC-SHA256 prompt sanitization, recursive sanitizer, retry queue

### Experimental Modules (disabled by default)

All experimental paths require explicit opt-in:

```yaml
# config.yaml
experimental:
  enable_qjl: false    # QJL score correction
  enable_polar: false  # Polar / hybrid quantization
  enable_adaptive: false  # Adaptive sparse controller
```

Or via environment:
```bash
RFSN_EXPERIMENTAL_QJL=true
RFSN_EXPERIMENTAL_POLAR=true
RFSN_EXPERIMENTAL_ADAPTIVE=true
```

**Warning:** Experimental features are not validated for production or quality-critical generation.

- `rfsn_v10/quantization/polar_quant.py` — Iterative hierarchical polar quantization
- `rfsn_v10/quantization/hybrid_polar_cartesian.py` — Hybrid polar-cartesian quantizer
- `rfsn_v10/quantization/qjl_score_correction.py` — QJL sketch-based score correction
- `rfsn_v10/quantization/isoquant_precondition.py` — IsoQuant quaternion preconditioner

---

## Validated Stable Configs

These configs have been validated at the alpha level on Apple Silicon:

- `k8_v5_gs32` — 8-bit keys, 5-bit values, group size 32
- `k8_v5_gs64` — 8-bit keys, 5-bit values, group size 64 (default)
- `k8_v4_gs64` — 8-bit keys, 4-bit values, group size 64

---

## Known Limitations

1. **Sparse path is decode-focused.** Prefill always uses dense attention with causal masking.
2. **QJL correction is disabled by default.** Experimental — not validated.
3. **Experimental configs may degrade logits.** Short-prompt drift under active investigation.
4. **Full runtime requires MLX.** Apple Silicon is mandatory for the core runtime.
5. **Docker service mode is not production hardened.** CLI health check only.
6. **CUDA not implemented.** Do not depend on it.
7. **End-to-end speedup not proven.** Decode TPS is comparable; compression overhead makes total time slower at short contexts.

---

## Security

- Telemetry prompt text is HMAC-SHA256 hashed before leaving the process boundary.
- Sanitization is recursive — nested dicts and message lists are also cleaned.
- Set `RFSN_TELEMETRY_HMAC_KEY` in production for keyed HMAC protection.
- HTTPS is required for remote ClickHouse hosts; HTTP is only allowed for localhost.

---

## Running Tests

```bash
# CPU-safe tests (no MLX required)
pytest tests/test_config.py tests/test_config_strict.py \
       tests/test_kernels_validation.py \
       tests/test_quantization_lazy_imports.py \
       tests/test_experimental_flags.py -q

# MLX-dependent tests (Apple Silicon required)
pytest tests/test_attention.py tests/test_bitpack.py \
       tests/test_bitpack_fuzz.py tests/test_kv_manager.py \
       tests/test_drift.py -q

# Security tests
pytest tests/test_clickhouse_security.py -q
```

---

## Docker

```bash
docker build -t rfsn-qjl17 .
docker run --rm rfsn-qjl17 python -m rfsn_v10 healthcheck
```

Requires Python 3.11 (not 3.12). Core runtime does not function in Docker without Apple Silicon + MLX.
