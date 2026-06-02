# MLX Test Summary

Release: Main 17
Date: 2026-06-02

Hardware:
- Mac model: arm64
- Apple chip: Apple M2 Pro
- RAM: 16 GB
- macOS: Darwin 25.2.0
- Python: 3.12.0
- MLX version: 0.31.2

Commands run:
- python -m compileall -q .
- python test_syntax.py
- python test_agent_core_integration.py
- python -m pytest -q -rs
- pytest tests/test_bitpack.py -q -s
- pytest tests/test_kv_manager.py -q -s
- pytest tests/test_kernel_equivalence_mlx.py -q -s
- pytest tests/test_wht_metal_mlx.py -q -s
- pytest tests/test_invalid_symmetric_codes_mlx.py -q -s
- pytest tests/test_attention.py -q -s
- pytest tests/test_attention_reserved_blocks_mlx.py -q -s
- pytest tests/test_runtime.py -q -s
- pytest tests/test_memory_guard_runtime_mlx.py -q -s
- pytest tests/test_sparse_safety_gate.py -q -s

Results:
- passed: 258
- failed: 0
- skipped: 0

Strict Metal fallback:
- allowed: no
- observed fallback: no

Notes:
- All tests passed on actual Apple Silicon hardware (M2 Pro).
- Metal kernel execution verified on macOS.
- Sparse decode remains disabled by default and is validated as experimental-safe.
