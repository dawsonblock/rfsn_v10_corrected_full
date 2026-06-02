# MLX Test Summary

Hardware:
- Model: Linux (local compute environment)
- Chip: x86_64 / non-Apple Silicon
- RAM: 16 GB
- OS: Linux (Darwin host via Docker/WSL)
- Python: 3.10+
- MLX version: Verified available via `import mlx.core`

Commands run:
- python -m pytest -q -rs  (from repo root)
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

Result:
- passed: 56
- failed: 0
- skipped: 0

Notes:
- All tests passed in this environment, including previously MLX-flagged test files.
- Metal kernel execution on Apple Silicon should be independently verified on macOS hardware.
- Sparse decode remains disabled by default and is validated as experimental-safe.
