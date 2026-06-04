#!/usr/bin/env python3
"""Release integrity checker for RFSN v10 Main 28."""
from __future__ import annotations

import json
import subprocess
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
        # Check first 10 non-empty lines for Main 28 title
        lines = readme.splitlines()
        non_empty = [ln for ln in lines if ln.strip()][:10]
        has_title = any(
            ln.startswith("# RFSN v10 Main 28") for ln in non_empty
        )
        if not has_title:
            errors.append("README title is not Main 28")

        # Check status section
        if "## Status: RFSN v10 Main 28" not in readme:
            errors.append("README status section is not Main 28")

        for stale in [
            "artifacts/proof/main23",
            "artifacts/proof/main24",
            "artifacts/proof/main25",
            "artifacts/proof/main26",
            "artifacts/proof/main27",
        ]:
            # Allow historical mentions if they appear after
            # the word "historical" or inside a note about old
            # releases — check each line
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
        negation_patterns = [
            "not ", "no ", "never ", "unimplemented", "disabled"
        ]
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

        # README must qualify PolarQuant status if file exists
        polar_quant_path = (
            root / "rfsn_v10" / "quantization" / "polar_quant.py"
        )
        if polar_quant_path.exists():
            readme_lower = readme.lower()
            mentions_polar = (
                "polar quantization" in readme_lower
                or "polar quant" in readme_lower
            )
            has_qualifier = (
                "stable runtime" in readme_lower
                or "experimental" in readme_lower
            )
            if mentions_polar and not has_qualifier:
                errors.append(
                    "README mentions PolarQuant but does not qualify "
                    "stable vs experimental status"
                )

    # --- Experimental branch checks ---
    exp_dir = root / "artifacts" / "proof" / "experimental"
    exp_code_present = (
        root / "rfsn_v10" / "quantization" / "polar_quant.py"
    ).exists()
    if exp_code_present:
        # If experimental code exists, certain artifacts should exist
        required_exp_artifacts = [
            "real_model_validation.json",
            "long_context_validation.json",
            "memory_accounting.json",
            "comparison_summary.json",
            "comparison_summary.md",
            "qjl_attention_score.json",
        ]
        for artifact in required_exp_artifacts:
            artifact_path = exp_dir / artifact
            if not artifact_path.exists():
                errors.append(
                    f"experimental artifact missing: {artifact}"
                )
            else:
                if artifact.endswith(".json"):
                    try:
                        data = json.loads(
                            artifact_path.read_text(encoding="utf-8")
                        )
                        # No config should claim production-ready
                        for key in ("production_ready", "production-ready"):
                            if data.get(key) is True:
                                errors.append(
                                    f"{artifact} claims production_ready=True"
                                )
                        # Check comparison_summary for rejected configs
                        if artifact == "comparison_summary.json":
                            for row in data.get("rows", []):
                                status = row.get("recommended_status", "")
                                if status == "candidate":
                                    for ctx in (512, 1024, 2048):
                                        val = row.get(f"pass_{ctx}")
                                        if val != "pass":
                                            errors.append(
                                                f"{row['config']}: candidate "
                                                f"has non-pass context {ctx}: "
                                                f"{val}"
                                            )
                    except Exception:
                        pass

        # Memory accounting consistency
        mem_path = exp_dir / "memory_accounting.json"
        if mem_path.exists():
            try:
                mem = json.loads(mem_path.read_text(encoding="utf-8"))
                rows = mem.get("rows", [])
                if not rows:
                    errors.append("memory_accounting.json has no rows")
                for row in rows:
                    cfg = row.get("config", "<unknown>")
                    ratio = row.get("actual_compression_ratio")
                    fp16 = row.get("fp16_kv_bytes")
                    comp = row.get("total_compressed_bytes")
                    basis = row.get("memory_basis")
                    if ratio is None or ratio <= 0:
                        errors.append(
                            f"{cfg}: invalid actual_compression_ratio"
                        )
                    if fp16 is None or fp16 <= 0:
                        errors.append(f"{cfg}: invalid fp16_kv_bytes")
                    if comp is None or comp <= 0:
                        errors.append(
                            f"{cfg}: invalid total_compressed_bytes"
                        )
                    if basis not in {
                        "mean_per_prompt_real_model_cache",
                        "real_model_cache",
                    }:
                        errors.append(
                            f"{cfg}: memory_basis is not real-model based: "
                            f"{basis}"
                        )
            except Exception as exc:
                errors.append(
                    f"memory_accounting.json parse error: {exc}"
                )

    # --- Main 28 artifact directory ---
    artifact_dir = root / "artifacts" / "proof" / "main28"
    if not artifact_dir.exists():
        errors.append("artifacts/proof/main28 missing")
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
            "summary.json",
            "mlx_test_summary.md",
            "mlx_pytest_raw.log",
            "mlx_pytest_junit.xml",
            "main28_release_manifest.json",
        ]
        for artifact in required_artifacts:
            artifact_path = artifact_dir / artifact
            if not artifact_path.exists():
                errors.append(f"required artifact missing: {artifact}")
                continue
            # Every JSON artifact with a "release" field must say main28
            if artifact.endswith(".json"):
                try:
                    data = json.loads(
                        artifact_path.read_text(encoding="utf-8")
                    )
                    release_field = data.get("release")
                    if release_field is not None and release_field != "main28":
                        errors.append(
                            f"{artifact} release field is "
                            f"'{release_field}' (expected 'main28')"
                        )
                except Exception:
                    pass

        # Manifest must declare release = main28
        manifest_path = artifact_dir / "main28_release_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if manifest.get("release") != "main28":
                    errors.append(
                        "main28_release_manifest.json release "
                        "field is not 'main28'"
                    )
            except Exception:
                errors.append("main28_release_manifest.json is not valid JSON")

        # MLX summary must identify Main 28
        mlx_summary_path = artifact_dir / "mlx_test_summary.md"
        if mlx_summary_path.exists():
            try:
                mlx_summary = mlx_summary_path.read_text(encoding="utf-8")
                if "Main 28" not in mlx_summary:
                    errors.append("MLX summary does not identify Main 28")
            except OSError:
                pass

        # proof_summary.md must identify Main 28
        proof_path = artifact_dir / "proof_summary.md"
        if proof_path.exists():
            try:
                proof = proof_path.read_text(encoding="utf-8")
                if "Main 28" not in proof:
                    errors.append("proof_summary.md does not identify Main 28")
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
                        "real_model_validation.json still uses "
                        "tiny-random model"
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
                if data.get("release") != "main28":
                    errors.append(
                        "real_model_validation.json release "
                        "field is not 'main28'"
                    )
            except Exception as exc:
                errors.append(f"real_model_validation.json parse error: {exc}")

        # long_context_validation.json: recommended config must
        # pass all contexts
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
                errors.append(
                    f"long_context_validation.json parse error: {exc}"
                )

    return errors


def main() -> int:
    errors = check()
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Run proof-summary / JSON consistency checker
    consistency_script = Path("scripts/check_proof_summary_consistency.py")
    if consistency_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(consistency_script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                "ERROR: proof-summary consistency check failed: "
                f"{exc.stderr}",
                file=sys.stderr,
            )
            return 1

    print("release integrity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
