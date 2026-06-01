#!/usr/bin/env python3
"""Generate proof artifacts for benchmark scenarios.

Outputs:
- kv_cache_runs.json
- e2e_scenarios.json
- proof_summary.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_kv_cache import benchmark_kv
from benchmarks.benchmark_end_to_end import benchmark_e2e


KV_SHAPES = [
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 32, 4096, 128),
]

KV_CONFIGS = [
    (8, 3, True),
    (8, 3, False),
    (8, 8, False),
]

E2E_SCENARIOS = [
    ("cache_miss_full_precision_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, False),
    ("cache_miss_use_compressed_on_miss_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True),
    ("cache_hit_compressed_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True),
    ("sparse_decode_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.25, True),
    ("dense_decode_path", (1, 8, 8, 64), (1, 8, 2048, 64), 8, 3, True, 0.25, True),
]


def _metadata() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def run_kv_benchmarks(iterations: int) -> dict:
    runs = []
    for shape in KV_SHAPES:
        for k_bits, v_bits, use_incoherent in KV_CONFIGS:
            result = benchmark_kv(shape, k_bits, v_bits, use_incoherent, iterations=iterations)
            result["scenario"] = f"shape={shape}|k={k_bits}|v={v_bits}|incoherent={use_incoherent}"
            runs.append(result)
    return {"metadata": _metadata(), "iterations": iterations, "runs": runs}


def run_e2e_benchmarks(iterations: int) -> dict:
    runs = []
    for scenario, shape_q, shape_kv, k_bits, v_bits, use_incoherent, top_k_ratio, use_compressed_on_miss in E2E_SCENARIOS:
        result = benchmark_e2e(
            shape_q,
            shape_kv,
            k_bits,
            v_bits,
            use_incoherent,
            top_k_ratio,
            use_compressed_on_miss=use_compressed_on_miss,
            iterations=iterations,
        )
        result["scenario"] = scenario
        runs.append(result)
    return {"metadata": _metadata(), "iterations": iterations, "runs": runs}


def write_summary(output_dir: Path, kv_payload: dict, e2e_payload: dict, profile: str) -> None:
    kv_runs = kv_payload["runs"]
    e2e_runs = e2e_payload["runs"]

    best_kv = min(kv_runs, key=lambda r: r["retrieve_latency_ms"]) if kv_runs else None
    best_sparse = next((r for r in e2e_runs if r["scenario"] == "sparse_decode_path"), None)
    best_dense = next((r for r in e2e_runs if r["scenario"] == "dense_decode_path"), None)

    lines = [
        f"# {profile} Proof Summary",
        "",
        f"Generated: {_metadata()['timestamp']}",
        "",
        "## Files",
        "- kv_cache_runs.json",
        "- e2e_scenarios.json",
        "",
        "## Highlights",
    ]

    if best_kv is not None:
        lines.append(
            "- Fastest KV retrieve: "
            f"{best_kv['retrieve_latency_ms']:.2f}ms "
            f"({best_kv['scenario']})"
        )
        lines.append(
            "- KV value quality (same run): "
            f"cos={best_kv['value_cosine_sim']:.4f}, "
            f"rel_mae={best_kv['value_rel_mae']:.4f}, "
            f"max_abs={best_kv['value_max_abs_error']:.4f}"
        )

    if best_sparse is not None:
        lines.append(
            "- Sparse decode path: "
            f"miss={best_sparse['cache_miss_total_latency_ms']:.2f}ms, "
            f"hit={best_sparse['cache_hit_total_latency_ms']:.2f}ms, "
            f"quant_cos={best_sparse.get('quant_audit_cosine')}"
        )

    if best_dense is not None:
        lines.append(
            "- Dense decode path: "
            f"miss={best_dense['cache_miss_total_latency_ms']:.2f}ms, "
            f"hit={best_dense['cache_hit_total_latency_ms']:.2f}ms, "
            f"mode={best_dense.get('cache_hit_execution_mode')}"
        )

    lines.extend([
        "",
        "## Next Checks",
        "- Compare these artifacts against previous runs for trend regressions.",
        "- Keep sparse vs dense and quant vs sparse audit metrics split in analysis.",
    ])

    (output_dir / "proof_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate proof artifacts")
    parser.add_argument(
        "--profile",
        default="main8_1",
        help="Proof profile name for output labeling/default paths",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where JSON/Markdown artifacts are written",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per scenario",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"artifacts/proof/{args.profile}")
    output_dir.mkdir(parents=True, exist_ok=True)

    kv_payload = run_kv_benchmarks(iterations=args.iterations)
    e2e_payload = run_e2e_benchmarks(iterations=args.iterations)

    (output_dir / "kv_cache_runs.json").write_text(
        json.dumps(kv_payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "e2e_scenarios.json").write_text(
        json.dumps(e2e_payload, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(output_dir, kv_payload, e2e_payload, profile=args.profile)

    print(f"Wrote proof artifacts to {output_dir}")


if __name__ == "__main__":
    main()
