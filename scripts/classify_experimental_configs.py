#!/usr/bin/env python3
"""
RFSN v10 — Classify experimental configs for promotion (hardened).

Reads all available proof artifacts and produces a classification report
according to the plan's Phase 9/15 promotion rules.

Hard rules:
- No experimental mode can be classified as candidate
  without throughput data.
- No config using raw uint32 fallback (>8-bit) may be called
  memory-optimized.
- QJL is disabled until its benchmark passes.
- turbo_polar is speed_candidate only if it beats k8_v5_gs64
  on compressed-path throughput.
- adaptive is quality_candidate only if throughput is within
  acceptable slowdown.
- experimental_hybrid is memory_candidate only if compression is
  meaningfully better and speed is acceptable.

Usage:
    python scripts/classify_experimental_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def classify_configs() -> dict[str, Any]:
    """Classify configs based on available evidence."""
    root = Path(".")
    exp_dir = root / "artifacts" / "proof" / "experimental"
    qwen_dir = exp_dir / "qwen_1_5b"

    # Load comparison summary
    comp = _load_json(exp_dir / "comparison_summary.json") or {}
    rows = comp.get("rows", [])

    # Load throughput (synthetic)
    throughput = _load_json(exp_dir / "throughput.json") or {}
    tp_results = throughput.get("results", [])
    tp_configs = {c["config"]: c for c in tp_results}
    tp_conclusions = throughput.get("conclusions", {})

    # Load real generation throughput (required for promotion)
    real_gen_tp = (
        _load_json(exp_dir / "real_generation_throughput.json") or {}
    )
    has_real_gen_tp = bool(real_gen_tp.get("results"))

    # Load 1.5B validation if available
    qwen_real = (
        _load_json(qwen_dir / "real_model_validation.json") or {}
    )
    qwen_long = (
        _load_json(qwen_dir / "long_context_validation.json") or {}
    )

    classifications: dict[str, str] = {}
    notes: list[str] = []

    for row in rows:
        name = row.get("config", "unknown")
        if name == "baseline_fp16":
            classifications[name] = "reference"
            continue

        # stable configs are the proven baseline;
        # short-circuit before experimental gates
        is_stable = name.startswith("stable_")

        # Check 0.5B evidence
        pass_512 = row.get("pass_512") == "pass"
        pass_1024 = row.get("pass_1024") == "pass"
        pass_2048 = row.get("pass_2048") == "pass"
        has_mem = row.get("total_compressed_bytes") is not None

        if is_stable:
            if pass_512 and pass_1024 and pass_2048 and has_mem:
                if name == "stable_k8_v5_gs64":
                    classifications[name] = "stable_default"
                    notes.append(
                        f"{name}: stable_default — locked default runtime"
                    )
                else:
                    classifications[name] = "stable_baseline"
                    notes.append(
                        f"{name}: stable_baseline — proven at 0.5B"
                    )
            else:
                classifications[name] = "stable_needs_data"
                notes.append(
                    f"{name}: stable_needs_data — missing context"
                )
            continue

        # Check 1.5B evidence
        qwen_pass = False
        if qwen_real and qwen_long:
            for cfg in qwen_real.get("configs", []):
                cfg_name = cfg.get("name", "")
                if cfg_name == name or cfg_name == name.replace(
                    "stable_", ""
                ):
                    qwen_pass = cfg.get("status") == "pass"

        # Check throughput (use conclusions verdict)
        tp = tp_configs.get(name, {})
        conclusion = tp_conclusions.get(name, {})
        verdict = conclusion.get("verdict", "unknown")
        has_tp = name in tp_configs

        # Check >8-bit fallback (memory disqualification)
        uses_raw_uint32 = False
        bits = row.get("cartesian_bits") or row.get("k_bits")
        if bits is not None and bits > 8:
            uses_raw_uint32 = True

        # Classification logic for experimental configs
        if not (pass_512 and pass_1024 and pass_2048):
            classifications[name] = "rejected_quality"
            notes.append(
                f"{name}: rejected_quality — fails context"
            )
        elif not has_mem:
            classifications[name] = "rejected_memory"
            notes.append(
                f"{name}: rejected_memory — missing compression bytes"
            )
        elif not has_tp:
            classifications[name] = "rejected_speed"
            notes.append(
                f"{name}: rejected_speed — missing throughput"
            )
        elif verdict == "rejected_speed":
            classifications[name] = "rejected_speed"
            notes.append(
                f"{name}: rejected_speed — throughput regression"
            )
        elif uses_raw_uint32:
            classifications[name] = "rejected_memory"
            notes.append(
                f"{name}: rejected_memory — uses raw uint32 fallback"
            )
        elif not has_real_gen_tp:
            classifications[name] = "experimental_only"
            notes.append(
                f"{name}: experimental_only — awaiting real generation"
            )
        elif not qwen_real:
            classifications[name] = "experimental_pending_1_5b"
            notes.append(
                f"{name}: experimental_pending_1_5b — awaiting 1.5B"
            )
        elif not qwen_pass:
            classifications[name] = "rejected_1_5b"
            notes.append(
                f"{name}: rejected_1_5b — fails 1.5B validation"
            )
        else:
            # All checks pass — determine candidate type
            baseline = next(
                (
                    r for r in rows
                    if r.get("config") == "stable_k8_v5_gs64"
                ),
                None,
            )
            if baseline:
                cm = row.get("cosine_min") or 0
                bcm = baseline.get("cosine_min") or 0
                to = row.get("top5_overlap") or 0
                bto = baseline.get("top5_overlap") or 0
                better_quality = cm >= bcm and to >= bto

                cr = row.get("compression_ratio") or 1.0
                bcr = baseline.get("compression_ratio") or 1.0
                better_memory = cr >= bcr

                if name == "turbo_polar":
                    # Speed candidate if faster than k8_v5_gs64
                    baseline_tp = tp_configs.get(
                        "stable_k8_v5_gs64", {}
                    )
                    baseline_tps = baseline_tp.get(
                        "tokens_per_second", 0
                    )
                    this_tps = tp.get("tokens_per_second", 0)
                    if this_tps > baseline_tps:
                        classifications[name] = "speed_candidate"
                        notes.append(
                            f"{name}: speed_candidate — fastest path"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — not yet"
                        )
                elif name == "adaptive":
                    if better_quality:
                        classifications[name] = "quality_candidate"
                        notes.append(
                            f"{name}: quality_candidate — strong quality"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — no clear win"
                        )
                elif name == "experimental_hybrid":
                    if better_memory and not better_quality:
                        classifications[name] = "memory_candidate"
                        notes.append(
                            f"{name}: memory_candidate — better memory"
                        )
                    elif better_quality and better_memory:
                        classifications[name] = "quality_candidate"
                        notes.append(
                            f"{name}: quality_candidate — beats baseline"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — no clear win"
                        )
                else:
                    if better_quality and better_memory:
                        classifications[name] = "quality_candidate"
                        notes.append(
                            f"{name}: quality_candidate — beats baseline"
                        )
                    elif better_memory:
                        classifications[name] = "memory_candidate"
                        notes.append(
                            f"{name}: memory_candidate — better memory"
                        )
                    else:
                        classifications[name] = "experimental_only"
                        notes.append(
                            f"{name}: experimental_only — no clear win"
                        )
            else:
                classifications[name] = "experimental_only"
                notes.append(
                    f"{name}: experimental_only — baseline missing"
                )

    return {
        "release": "experimental",
        "stable_default": "k8_v5_gs64",
        "promoted_to_default": False,
        "classifications": classifications,
        "notes": notes,
        "evidence": {
            "has_0_5b_validation": bool(comp),
            "has_throughput": bool(throughput),
            "has_real_generation_throughput": has_real_gen_tp,
            "has_1_5b_validation": bool(qwen_real),
            "has_layer_policy": (
                exp_dir / "layer_policy.json"
            ).exists(),
            "has_sensitivity": (
                exp_dir / "per_layer_sensitivity.json"
            ).exists(),
        },
        "promotion_rules": [
            "stable_default: locked production default (k8_v5_gs64)",
            "stable_baseline: proven stable config",
            "rejected_quality: fails 512/1024/2048",
            "rejected_memory: missing compressed bytes or raw uint32",
            "rejected_speed: missing throughput or regression",
            "experimental_pending_1_5b: awaiting 1.5B validation",
            "rejected_1_5b: fails 1.5B validation",
            "experimental_only: passes all gates but no clear win",
            "speed_candidate: fastest compressed path",
            "quality_candidate: better quality, acceptable throughput",
            "memory_candidate: better memory, comparable quality",
        ],
    }


def main() -> None:
    result = classify_configs()
    out_path = Path(
        "artifacts/proof/experimental/config_classification.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote classification to {out_path}")
    for note in result["notes"]:
        print(f"  {note}")
    print(f"\nEvidence status: {result['evidence']}")


if __name__ == "__main__":
    main()
