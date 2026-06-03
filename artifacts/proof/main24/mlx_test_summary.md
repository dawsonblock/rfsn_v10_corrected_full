# MLX Test Summary — Main 23

## Command Run
```bash
pytest tests/test_fused_kernel_mlx.py \
       tests/test_kernel_equivalence_mlx.py \
       tests/test_kv_manager.py \
       tests/test_retrieve_blocks.py \
       tests/test_sparse_safety_gate.py \
       -vv -s --tb=short \
       | tee artifacts/proof/main23/mlx_pytest_raw.log

pytest tests/test_fused_kernel_mlx.py \
       tests/test_kernel_equivalence_mlx.py \
       tests/test_kv_manager.py \
       tests/test_retrieve_blocks.py \
       tests/test_sparse_safety_gate.py \
       --junitxml artifacts/proof/main23/mlx_pytest_junit.xml
```

## Environment
- **Hardware**: Apple M2 Pro, 16 GB RAM, macOS
- **MLX version**: 0.31.2
- **Python version**: 3.12.0

## Test Files
- `tests/test_fused_kernel_mlx.py`
- `tests/test_kernel_equivalence_mlx.py`
- `tests/test_kv_manager.py`
- `tests/test_retrieve_blocks.py`
- `tests/test_sparse_safety_gate.py`

## Results
- **Passed**: 74
- **Failed**: 0
- **Skipped**: 0

## Artifacts
- Raw log: `artifacts/proof/main23/mlx_pytest_raw.log`
- JUnit XML: `artifacts/proof/main23/mlx_pytest_junit.xml`

## Notes
All MLX-dependent tests passed without Metal fallback.
