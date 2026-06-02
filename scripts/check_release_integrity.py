#!/usr/bin/env python3
"""Release integrity checker for RFSN v10."""
from __future__ import annotations

import sys
from pathlib import Path


def check() -> list[str]:
    errors: list[str] = []

    root = Path(".").resolve()

    # Check for forbidden directories
    forbidden_dirs = [
        ".tmp",
        "tmp",
        "temp",
        "release_tmp",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    ]
    for bad in forbidden_dirs:
        matches = list(root.rglob(bad))
        if matches:
            errors.append(
                f"forbidden path found: {bad} ({len(matches)} instances)"
            )

    # Check for forbidden files
    pyc = list(root.rglob("*.pyc"))
    if pyc:
        errors.append(f"pyc files found ({len(pyc)} instances)")

    ds_store = list(root.rglob(".DS_Store"))
    if ds_store:
        errors.append(f".DS_Store files found ({len(ds_store)} instances)")

    # Verify required proof artifacts exist
    artifact_dir = root / "artifacts" / "proof" / "main12"
    if not artifact_dir.exists():
        errors.append("artifacts/proof/main12 missing")
    else:
        required_artifacts = [
            "kv_cache_runs.json",
            "e2e_scenarios.json",
            "kernel_benchmark.json",
            "proof_summary.md",
            "summary.json",
            "regression_report.json",
            "regression_report.md",
            "mlx_test_summary.md",
        ]
        for artifact in required_artifacts:
            if not (artifact_dir / artifact).exists():
                errors.append(f"required artifact missing: {artifact}")

        # At least one of these must exist
        real_model_validation = artifact_dir / "real_model_validation.json"
        real_model_not_run = artifact_dir / "real_model_validation_not_run.txt"
        if not real_model_validation.exists() and not real_model_not_run.exists():
            errors.append(
                "real_model_validation.json or "
                "real_model_validation_not_run.txt must exist"
            )

        # Verify kernel plots require kernel_benchmark.json
        kernel_json = artifact_dir / "kernel_benchmark.json"
        kernel_plots = list((root / "results" / "plots").glob("kernel*.png"))
        if kernel_plots and not kernel_json.exists():
            errors.append("kernel plots exist but kernel_benchmark.json missing")

    # Verify README claims match reality
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
        if "Shipped proof artifacts" in readme and not artifact_dir.exists():
            errors.append(
                "README claims shipped proof artifacts but artifact dir missing"
            )
    except (FileNotFoundError, IOError):
        errors.append("README.md missing or unreadable")

    return errors


def main() -> int:
    errors = check()
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("release integrity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
