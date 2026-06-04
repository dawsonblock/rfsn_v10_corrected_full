#!/usr/bin/env python3
"""Release integrity checker for RFSN v10 Main 26."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def check() -> list[str]:
    errors: list[str] = []

    root = Path(".").resolve()

    # --- Forbidden filesystem artefacts ---
    forbidden_dirs = [".tmp", "tmp", "temp", "release_tmp"]
    for bad in forbidden_dirs:
        matches = [m for m in root.rglob(bad) if ".git" not in m.parts]
        if matches:
            errors.append(
                f"forbidden path found: {bad} ({len(matches)} instances)"
            )

    pycache = [m for m in root.rglob("__pycache__") if ".git" not in m.parts]
    if pycache:
        errors.append(
            f"__pycache__ directories found ({len(pycache)} instances)"
        )

    pyc = [m for m in root.rglob("*.pyc") if ".git" not in m.parts]
    if pyc:
        errors.append(f"*.pyc files found ({len(pyc)} instances)")

    ds_store = [m for m in root.rglob(".DS_Store") if ".git" not in m.parts]
    if ds_store:
        errors.append(f".DS_Store files found ({len(ds_store)} instances)")

    for pattern in ["*.zip", "*.tar", "*.tar.gz", "*.7z"]:
        matches = list(root.rglob(pattern))
        if matches:
            errors.append(
                f"nested archive(s) found: {[str(m) for m in matches[:10]]}"
            )

    # --- Reject placeholder plots ---
    plot_dir = root / "results" / "plots"
    if plot_dir.exists():
        for pattern in ["*pending*", "*placeholder*"]:
            bad = list(plot_dir.glob(pattern))
            if bad:
                errors.append(
                    f"placeholder plot(s) found: {[str(p) for p in bad[:10]]}"
                )

    # --- README strict checks ---
    readme_path = root / "README.md"
    readme: str = ""
    try:
        readme = readme_path.read_text(encoding="utf-8")
    except OSError:
        errors.append("README.md missing or unreadable")

    if readme:
        # Check first 10 non-empty lines for Main 27 title
        lines = readme.splitlines()
        non_empty = [ln for ln in lines if ln.strip()][:10]
        has_title = any(ln.startswith("# RFSN v10 Main 27") for ln in non_empty)
        if not has_title:
            errors.append("README title is not Main 27")

        for stale in [
            "artifacts/proof/main23",
            "artifacts/proof/main24",
            "artifacts/proof/main25",
            "artifacts/proof/main26",
        ]:
            # Allow historical mentions if they appear after the word
            # "historical" or inside a note about old releases — check each line
            for lineno, line in enumerate(readme.splitlines(), 1):
                if stale in line:
                    lower = line.lower()
                    if any(
                        w in lower
                        for w in (
                            "historical",
                            "history",
                            "retained",
                            "reference only",
                        )
                    ):
                        continue
                    errors.append(
                        f"README contains active stale artifact path: "
                        f"{stale} (line {lineno})"
                    )

        false_claims = [
            "production-ready",
            "production ready",
            "polar quant enabled",
            "partial dequant complete",
            "sparse-safe",
            "sparse enabled by default",
        ]
        # Use word boundaries to avoid matching partial words
        # (e.g., "noted" contains "not")
        negation_patterns = ["not ", "no ", "never ", "unimplemented", "disabled"]
        for line in readme.splitlines():
            lower_line = line.lower()
            for phrase in false_claims:
                if phrase in lower_line:
                    # Check for negation patterns with word boundaries
                    has_negation = any(
                        (" " + nw in lower_line) or lower_line.startswith(nw)
                        for nw in negation_patterns
                    )
                    if has_negation:
                        continue
                    errors.append(
                        f"README positive claim detected: {phrase!r}"
                    )

    # --- Main 27 artifact directory ---
    artifact_dir = root / "artifacts" / "proof" / "main27"
    if not artifact_dir.exists():
        errors.append("artifacts/proof/main27 missing")
    else:
        required_artifacts = [
            "kernel_benchmark.json",
            "fused_kernel_benchmark.json",
            "optimization_benchmark.json",
            "real_model_validation.json",
            "long_context_validation.json",
            "generation_smoke.json",
            "generation_throughput.json",
            "proof_summary.md",
            "mlx_test_summary.md",
            "mlx_pytest_raw.log",
            "mlx_pytest_junit.xml",
            "main27_release_manifest.json",
        ]
        for artifact in required_artifacts:
            if not (artifact_dir / artifact).exists():
                errors.append(f"required artifact missing: {artifact}")

        # Manifest must declare release = main27
        manifest_path = artifact_dir / "main27_release_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("release") != "main27":
                    errors.append(
                        "main27_release_manifest.json release field is not 'main27'"
                    )
            except Exception:
                errors.append("main27_release_manifest.json is not valid JSON")

        # MLX summary must identify Main 27
        mlx_summary_path = artifact_dir / "mlx_test_summary.md"
        if mlx_summary_path.exists():
            try:
                mlx_summary = mlx_summary_path.read_text(encoding="utf-8")
                if "Main 27" not in mlx_summary:
                    errors.append("MLX summary does not identify Main 27")
            except OSError:
                pass

        # proof_summary.md must identify Main 27
        proof_path = artifact_dir / "proof_summary.md"
        if proof_path.exists():
            try:
                proof = proof_path.read_text(encoding="utf-8")
                if "Main 27" not in proof:
                    errors.append("proof_summary.md does not identify Main 27")
            except OSError:
                pass

        # real_model_validation.json: no tiny-random model, sparse not enabled,
        # must evaluate >= 32 positions
        real_val_path = artifact_dir / "real_model_validation.json"
        if real_val_path.exists():
            try:
                data = json.loads(real_val_path.read_text(encoding="utf-8"))
                model_id = data.get("model", "")
                if "tiny-random" in model_id.lower():
                    errors.append(
                        "real_model_validation.json still uses tiny-random model"
                    )
                if data.get("sparse_enabled") is True:
                    errors.append(
                        "sparse_enabled is True in real_model_validation.json"
                    )
                for cfg in data.get("configs", []):
                    pos = cfg.get("token_positions_evaluated", 0)
                    if isinstance(pos, (int, float)) and pos < 32:
                        errors.append(
                            f"config {cfg.get('name')!r} evaluated only "
                            f"{pos} positions (minimum 32 required)"
                        )
                if data.get("release") != "main27":
                    errors.append(
                        "real_model_validation.json release field is not 'main27'"
                    )
            except Exception as exc:
                errors.append(f"real_model_validation.json parse error: {exc}")

        # long_context_validation.json: recommended config must pass all contexts
        long_ctx_path = artifact_dir / "long_context_validation.json"
        if long_ctx_path.exists():
            try:
                lc = json.loads(long_ctx_path.read_text(encoding="utf-8"))
                summary = lc.get("summary", {})
                recommended = summary.get("recommended_default", "")
                if recommended and recommended != "baseline_fp16":
                    for ctx_entry in lc.get("contexts", []):
                        for cfg in ctx_entry.get("configs", []):
                            if cfg.get("name") == recommended:
                                if cfg.get("status") != "pass":
                                    errors.append(
                                        f"recommended config {recommended!r} "
                                        f"fails context "
                                        f"{ctx_entry.get('tokens')} tokens"
                                    )
            except Exception as exc:
                errors.append(f"long_context_validation.json parse error: {exc}")

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
