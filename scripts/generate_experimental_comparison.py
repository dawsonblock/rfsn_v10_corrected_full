#!/usr/bin/env python3
"""
Generate stable-vs-experimental comparison report.

Reads:
  - artifacts/proof/experimental/real_model_validation.json
  - artifacts/proof/experimental/long_context_validation.json
  - artifacts/proof/experimental/memory_accounting.json
  - Optionally: artifacts/proof/main28/real_model_validation.json
    (or other stable release dirs)

Produces:
  - artifacts/proof/experimental/comparison_summary.json
  - artifacts/proof/experimental/comparison_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_config(data: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if data is None:
        return None
    for cfg in data.get("configs", []):
        if cfg.get("name") == name:
            return cfg
    return None


def _passes_context(
    long_ctx_data: dict[str, Any] | None, name: str, ctx: int
) -> str:
    if long_ctx_data is None:
        return "unknown"
    for entry in long_ctx_data.get("contexts", []):
        if entry.get("tokens") == ctx:
            for cfg in entry.get("configs", []):
                if cfg.get("name") == name:
                    st = cfg.get("status")
                    if st == "pass":
                        return "pass"
                    if st == "oom":
                        return "oom"
                    return "fail"
    return "unknown"


def _build_row(
    name: str,
    real_data: dict[str, Any] | None,
    long_ctx_data: dict[str, Any] | None,
    mem_data: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = _find_config(real_data, name)
    if cfg is None and name.startswith("stable_"):
        # Try without stable_ prefix when reading main28 artifacts
        cfg = _find_config(real_data, name.replace("stable_", ""))

    mem_cfg = None
    if mem_data is not None:
        for m in mem_data.get("configs", []):
            if m.get("config") == name:
                mem_cfg = m
                break

    if cfg is None:
        return {
            "config": name,
            "cosine_mean": None,
            "cosine_min": None,
            "top5_overlap": None,
            "kl": None,
            "nll_delta": None,
            "pass_512": "unknown",
            "pass_1024": "unknown",
            "pass_2048": "unknown",
            "compression_ratio": None,
            "total_compressed_bytes": None,
            "recommended_status": "missing_data",
        }

    row = {
        "config": name,
        "cosine_mean": cfg.get("logit_cosine_mean"),
        "cosine_min": cfg.get("logit_cosine_min"),
        "top5_overlap": cfg.get("top5_overlap_mean"),
        "kl": cfg.get("kl_divergence_mean"),
        "nll_delta": cfg.get("avg_nll_delta"),
        "pass_512": _passes_context(long_ctx_data, name, 512),
        "pass_1024": _passes_context(long_ctx_data, name, 1024),
        "pass_2048": _passes_context(long_ctx_data, name, 2048),
        "compression_ratio": (
            mem_cfg.get("compression_ratio") if mem_cfg else None
        ),
        "total_compressed_bytes": (
            mem_cfg.get("compressed_bytes", 0)
            + mem_cfg.get("metadata_bytes", 0)
            + mem_cfg.get("qjl_bytes", 0)
            if mem_cfg
            else None
        ),
    }

    # Determine recommended status
    if name == "baseline_fp16":
        row["recommended_status"] = "reference"
        return row

    passes_q = cfg.get("status") in ("pass", "reference")
    passes_all = all(
        row[f"pass_{ctx}"] == "pass"
        for ctx in (512, 1024, 2048)
        if row[f"pass_{ctx}"] != "unknown"
    )
    has_mem = row["compression_ratio"] is not None

    if not passes_q:
        row["recommended_status"] = "rejected_quality"
    elif not passes_all:
        row["recommended_status"] = "rejected_context"
    elif not has_mem:
        row["recommended_status"] = "needs_memory_data"
    elif row["compression_ratio"] is not None and row["compression_ratio"] <= 1.0:
        row["recommended_status"] = "rejected_no_compression"
    else:
        row["recommended_status"] = "candidate"

    return row


def _make_md_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Config",
        "CosMean",
        "CosMin",
        "Top5",
        "KL",
        "NLLΔ",
        "512",
        "1024",
        "2048",
        "CompRatio",
        "Bytes",
        "Status",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        vals = [
            r["config"],
            _fmt(r.get("cosine_mean")),
            _fmt(r.get("cosine_min")),
            _fmt(r.get("top5_overlap")),
            _fmt(r.get("kl")),
            _fmt(r.get("nll_delta")),
            r.get("pass_512", "-"),
            r.get("pass_1024", "-"),
            r.get("pass_2048", "-"),
            _fmt(r.get("compression_ratio")),
            str(r.get("total_compressed_bytes", "-")) if r.get("total_compressed_bytes") is not None else "-",
            r.get("recommended_status", "-"),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stable-vs-experimental comparison report"
    )
    parser.add_argument(
        "--experimental-dir",
        default="artifacts/proof/experimental",
        help="Directory with experimental validation artifacts",
    )
    parser.add_argument(
        "--stable-dir",
        default="artifacts/proof/main28",
        help="Directory with stable validation artifacts (e.g. main28)",
    )
    parser.add_argument(
        "--configs",
        default=(
            "baseline_fp16,stable_k8_v5_gs64,stable_k8_v5_gs32,"
            "experimental_hybrid,turbo_polar,adaptive,turbo_k8r8v6"
        ),
        help="Comma-separated config names to compare",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/proof/experimental",
        help="Output directory for comparison artifacts",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experimental_dir)
    stable_dir = Path(args.stable_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load experimental artifacts
    exp_real = _load_json(exp_dir / "real_model_validation.json")
    exp_long = _load_json(exp_dir / "long_context_validation.json")
    exp_mem = _load_json(exp_dir / "memory_accounting.json")

    # Load stable artifacts (fall back to experimental if not found in stable)
    stable_real = _load_json(stable_dir / "real_model_validation.json")
    stable_long = _load_json(stable_dir / "long_context_validation.json")
    stable_mem = _load_json(stable_dir / "memory_accounting.json")

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    rows: list[dict[str, Any]] = []
    for name in config_names:
        # Use stable artifacts for stable configs, experimental for experimental
        if name.startswith("stable_"):
            real = stable_real or exp_real
            long = stable_long or exp_long
            mem = stable_mem or exp_mem
        else:
            real = exp_real
            long = exp_long
            mem = exp_mem

        row = _build_row(name, real, long, mem)
        rows.append(row)

    # JSON output
    payload = {
        "release": "experimental",
        "configs_compared": config_names,
        "stable_source": str(stable_dir),
        "experimental_source": str(exp_dir),
        "rows": rows,
        "notes": [
            "recommended_status values: reference, candidate, "
            "rejected_quality, rejected_context, "
            "rejected_no_compression, needs_memory_data, missing_data",
            "No config is recommended after failing any context.",
        ],
    }
    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    # Markdown output
    md_lines = [
        "# Stable vs Experimental Quantization Comparison",
        "",
        f"**Stable source:** `{stable_dir}`  ",
        f"**Experimental source:** `{exp_dir}`  ",
        "",
        "## Config Comparison",
        "",
        _make_md_table(rows),
        "",
        "## Legend",
        "",
        "- **reference**: baseline_fp16 (no compression)",
        "- **candidate**: passes quality + all contexts + compression > 1.0",
        "- **rejected_quality**: failed quality thresholds",
        "- **rejected_context**: failed one or more context lengths",
        "- **rejected_no_compression**: compression ratio <= 1.0",
        "- **needs_memory_data**: missing memory accounting data",
        "- **missing_data**: config not found in validation artifacts",
        "",
        "*Generated by generate_experimental_comparison.py*",
    ]
    md_path = out_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
