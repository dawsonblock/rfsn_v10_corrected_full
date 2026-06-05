#!/usr/bin/env python3
"""
RFSN v10 — Classify experimental configs for promotion.

Reads all available proof artifacts and produces a classification report
according to the plan's Phase 15 promotion rules.

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
    main28_dir = root / "artifacts" / "proof" / "main28"
    qwen_dir = exp_dir / "qwen_1_5b"

    # Load comparison summary
    comp = _load_json(exp_dir / "comparison_summary.json") or {}
    rows = comp.get("rows", [])

    # Load throughput
    throughput = _load_json(exp_dir / "throughput.json") or {}
    tp_configs = {c["config"]: c for c in throughput.get("results", [])}
    tp_conclusions = throughput.get("conclusions", {})

    # Load 1.5B validation if available
    qwen_real = _load_json(qwen_dir / "real_model_validation.json") or {}
    qwen_long = _load_json(qwen_dir / "long_context_validation.json") or {}
    qwen_mem = _load_json(qwen_dir / "memory_accounting.json") or {}

    classifications: dict[str, str] = {}
    notes: list[str] = []

    for row in rows:
        name = row.get("config", "unknown")
        if name == "baseline_fp16":
            classifications[name] = "reference"
            continue

        # stable configs are the proven baseline; short-circuit before experimental gates
        is_stable = name.startswith("stable_")

        # Check 0.5B evidence
        pass_512 = row.get("pass_512") == "pass"
        pass_1024 = row.get("pass_1024") == "pass"
        pass_2048 = row.get("pass_2048") == "pass"
        has_mem = row.get("total_compressed_bytes") is not None
        is_candidate = row.get("recommended_status") == "candidate"

        if is_stable:
            if pass_512 and pass_1024 and pass_2048 and has_mem:
                classifications[name] = "stable_baseline"
                notes.append(
                    f"{name}: stable_baseline — proven at 0.5B, default runtime"
                )
            else:
                classifications[name] = "stable_needs_data"
                notes.append(
                    f"{name}: stable_needs_data — missing context or memory proof"
                )
            continue

        # Check 1.5B evidence
        qwen_pass = False
        if qwen_real and qwen_long:
            for cfg in qwen_real.get("configs", []):
                if cfg.get("name") == name or cfg.get("name") == name.replace("stable_", ""):
                    qwen_pass = cfg.get("status") == "pass"

        # Check throughput (use conclusions verdict)
        tp = tp_configs.get(name, {})
        conclusion = tp_conclusions.get(name, {})
        verdict = conclusion.get("verdict", "unknown")
        has_tp = name in tp_configs

        # Classification logic for experimental configs
        if not (pass_512 and pass_1024 and pass_2048):
            classifications[name] = "rejected_context"
            notes.append(f"{name}: rejected — fails at least one context (512/1024/2048)")
        elif not has_mem:
            classifications[name] = "needs_memory_data"
            notes.append(f"{name}: needs_memory_data — missing compression bytes")
        elif not has_tp:
            classifications[name] = "needs_throughput_data"
            notes.append(f"{name}: needs_throughput_data — missing throughput benchmark")
        elif verdict == "rejected_speed":
            classifications[name] = "rejected_speed"
            notes.append(f"{name}: rejected_speed — throughput regression unacceptable")
        elif not qwen_real:
            classifications[name] = "experimental_pending_1_5b"
            notes.append(f"{name}: experimental — 0.5B passes, awaiting 1.5B validation")
        elif not qwen_pass:
            classifications[name] = "rejected_1_5b"
            notes.append(f"{name}: rejected_1_5b — fails 1.5B validation")
        else:
            # All checks pass — compare against k8_v5_gs64
            baseline = next(
                (r for r in rows if r.get("config") == "stable_k8_v5_gs64"), None
            )
            if baseline:
                better_quality = (
                    (row.get("cosine_min") or 0) >= (baseline.get("cosine_min") or 0)
                    and (row.get("top5_overlap") or 0) >= (baseline.get("top5_overlap") or 0)
                )
                better_memory = (row.get("compression_ratio") or 1.0) >= (
                    baseline.get("compression_ratio") or 1.0
                )
                if better_quality and better_memory:
                    classifications[name] = "promote_to_optional"
                    notes.append(
                        f"{name}: promote_to_optional — beats baseline on quality+memory"
                    )
                elif better_memory:
                    classifications[name] = "experimental_viable"
                    notes.append(
                        f"{name}: experimental_viable — better memory, comparable quality"
                    )
                else:
                    classifications[name] = "experimental"
                    notes.append(
                        f"{name}: experimental — passes all gates but no clear win"
                    )
            else:
                classifications[name] = "experimental"
                notes.append(f"{name}: experimental — baseline missing for comparison")

    return {
        "release": "experimental",
        "classifications": classifications,
        "notes": notes,
        "evidence": {
            "has_0_5b_validation": bool(comp),
            "has_throughput": bool(throughput),
            "has_1_5b_validation": bool(qwen_real),
            "has_layer_policy": (exp_dir / "layer_policy.json").exists(),
            "has_sensitivity": (exp_dir / "per_layer_sensitivity.json").exists(),
        },
        "promotion_rules": [
            "rejected_context: fails 512/1024/2048",
            "needs_memory_data: missing compressed bytes",
            "needs_throughput_data: missing throughput benchmark",
            "rejected_speed: throughput regression unacceptable",
            "experimental_pending_1_5b: 0.5B passes, 1.5B not yet run",
            "rejected_1_5b: fails 1.5B validation",
            "experimental: passes all gates but no clear win vs baseline",
            "experimental_viable: better memory, comparable quality",
            "promote_to_optional: beats baseline on quality and memory",
        ],
    }


def main() -> None:
    result = classify_configs()
    out_path = Path("artifacts/proof/experimental/config_classification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote classification to {out_path}")
    for note in result["notes"]:
        print(f"  {note}")
    print(f"\nEvidence status: {result['evidence']}")


if __name__ == "__main__":
    main()
