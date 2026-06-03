#!/usr/bin/env python3
"""Release integrity checker for RFSN v10 Main 23."""
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

    # Check for nested archives — reject all zip/tar/7z files anywhere
    for pattern in ["*.zip", "*.tar", "*.tar.gz", "*.7z"]:
        matches = list(root.rglob(pattern))
        if matches:
            errors.append(
                f"nested archive(s) found: {[str(m) for m in matches[:10]]}"
            )

    # Verify required proof artifacts exist in main24
    artifact_dir = root / "artifacts" / "proof" / "main24"
    if not artifact_dir.exists():
        errors.append("artifacts/proof/main24 missing")
    else:
        required_artifacts = [
            "kernel_benchmark.json",
            "fused_kernel_benchmark.json",
            "optimization_benchmark.json",
            "real_model_validation.json",
            "long_context_validation.json",
            "proof_summary.md",
            "summary.json",
            "mlx_test_summary.md",
            "mlx_pytest_raw.log",
            "mlx_pytest_junit.xml",
        ]
        for artifact in required_artifacts:
            if not (artifact_dir / artifact).exists():
                errors.append(f"required artifact missing: {artifact}")

    # Reject placeholder plots
    plot_dir = root / "results" / "plots"
    if plot_dir.exists():
        for pattern in ["*pending*", "*placeholder*"]:
            bad = list(plot_dir.glob(pattern))
            if bad:
                errors.append(
                    f"placeholder plot(s) found: {[str(p) for p in bad[:10]]}"
                )

    # Verify README claims match reality
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
        if "Shipped proof artifacts" in readme and not artifact_dir.exists():
            errors.append(
                "README claims shipped proof artifacts but artifact dir missing"
            )
        if (
            "fused_kernel_benchmark.json" in readme
            and artifact_dir.exists()
            and not (artifact_dir / "fused_kernel_benchmark.json").exists()
        ):
            errors.append(
                "README claims fused kernel proof but "
                "fused_kernel_benchmark.json missing"
            )
        # Hard rejection: do not claim production-ready / sparse-safe / etc.
        false_claims = [
            "production-ready",
            "production ready",
            "polar quant enabled",
            "partial dequant complete",
            "sparse-safe",
        ]
        negation_words = ["not ", "no ", "never ", "unimplemented", "disabled"]
        for line in readme.splitlines():
            lower_line = line.lower()
            for phrase in false_claims:
                if phrase in lower_line:
                    if any(nw in lower_line for nw in negation_words):
                        continue
                    errors.append(f"README positive claim detected: {phrase}")
    except (FileNotFoundError, IOError):
        errors.append("README.md missing or unreadable")

    # Verify release version markers
    expected_release = "Main 23"
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
        if expected_release not in readme:
            errors.append("README does not identify Main 23")
    except (FileNotFoundError, IOError):
        errors.append("README.md missing for version check")

    try:
        proof_path = (
            root / "artifacts" / "proof" / "main23" / "proof_summary.md"
        )
        if proof_path.exists():
            proof = proof_path.read_text(encoding="utf-8")
            if expected_release not in proof:
                errors.append("proof_summary.md does not identify Main 23")
    except (FileNotFoundError, IOError):
        pass  # Already reported if artifact missing

    # Verify real-model validation uses a real non-random model
    try:
        real_val_path = artifact_dir / "real_model_validation.json"
        if real_val_path.exists():
            import json
            data = json.loads(real_val_path.read_text(encoding="utf-8"))
            model = data.get("model", "")
            if "tiny-random" in model.lower():
                errors.append(
                    "real_model_validation.json still uses tiny-random model"
                )
    except Exception:
        pass

    # Verify sparse is not enabled by default
    try:
        real_val_path = artifact_dir / "real_model_validation.json"
        if real_val_path.exists():
            import json
            data = json.loads(real_val_path.read_text(encoding="utf-8"))
            if data.get("sparse_enabled") is True:
                errors.append("sparse_enabled is True in real_model_validation.json")
    except Exception:
        pass

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
