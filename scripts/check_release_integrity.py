#!/usr/bin/env python3
"""Release integrity checker for RFSN v10."""
from __future__ import annotations

import sys
from pathlib import Path


def _clean_ephemeral(root: Path) -> None:
    for bad in ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"]:
        for p in root.rglob(bad):
            if p.is_dir():
                import shutil
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
    for pyc in root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)


def check() -> list[str]:
    errors: list[str] = []

    root = Path(".").resolve()
    
    _clean_ephemeral(root)
    
    for bad in ["release_tmp"]:
        matches = list(root.rglob(bad))
        if matches:
            errors.append(f"forbidden path found: {bad} ({len(matches)} instances)")
    
    pyc = list(root.rglob("*.pyc"))
    if pyc:
        errors.append(f"pyc files found ({len(pyc)} instances)")

    artifact_dir = root / "artifacts" / "proof" / "main12"
    if not artifact_dir.exists():
        errors.append("artifacts/proof/main12 missing")
    
    kernel_json = artifact_dir / "kernel_benchmark.json"
    kernel_plots = list((root / "results" / "plots").glob("kernel*.png"))
    if kernel_plots and not kernel_json.exists():
        errors.append("kernel plots exist but kernel_benchmark.json missing")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "Shipped proof artifacts" in readme and not artifact_dir.exists():
        errors.append("README claims shipped proof artifacts but artifact dir missing")

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
