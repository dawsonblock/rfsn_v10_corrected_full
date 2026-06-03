# MLX Test Summary — Main 26

## Command Run
```bash
pytest tests/test_fused_kernel_mlx.py \
       tests/test_kernel_equivalence_mlx.py \
       tests/test_kv_manager.py \
       tests/test_retrieve_blocks.py \
       tests/test_sparse_safety_gate.py \
       -vv -s --tb=short \
       | tee artifacts/proof/main26/mlx_pytest_raw.log

pytest tests/test_fused_kernel_mlx.py \
       tests/test_kernel_equivalence_mlx.py \
       tests/test_kv_manager.py \
       tests/test_retrieve_blocks.py \
       tests/test_sparse_safety_gate.py \
       --junitxml artifacts/proof/main26/mlx_pytest_junit.xml
```

## Environment
- **Hardware**: (to be filled after run)
- **MLX version**: (to be filled after run)
- **Python version**: (to be filled after run)

## Test Files
- `tests/test_fused_kernel_mlx.py`
- `tests/test_kernel_equivalence_mlx.py`
- `tests/test_kv_manager.py`
- `tests/test_retrieve_blocks.py`
- `tests/test_sparse_safety_gate.py`

## Results
- **Passed**: (to be filled after run)
- **Failed**: (to be filled after run)
- **Skipped**: (to be filled after run)

## Artifacts
- Raw log: `artifacts/proof/main26/mlx_pytest_raw.log`
- JUnit XML: `artifacts/proof/main26/mlx_pytest_junit.xml`

## Notes
Main 26 — Documentation + Causal NLL Validation Correction. All MLX-dependent tests must pass without Metal fallback.
