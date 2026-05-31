# RFSN v10 Corrected Core

Includes:
- `rfsn_v10/bitpack.py`
- `rfsn_v10/kv_manager.py`
- `rfsn_v10/attention.py`
- kernel/math tests
- block-sparse attention tests

Notes:
- Requires Apple MLX (`mlx`) and an Apple Silicon/macOS environment for Metal kernels.
- In this sandbox, tests were not executed because `mlx` is not installed.
- Run locally with:

```bash
pip install mlx pytest
pytest -q
```
