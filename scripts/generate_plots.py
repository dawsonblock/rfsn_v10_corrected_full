#!/usr/bin/env python3
"""Generate proof plots from KV and E2E benchmark artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6hQAAAAASUVORK5CYII="
)


def _load_runs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("runs", []))


def _write_placeholder_png(path: Path, note: str) -> None:
    path.write_bytes(TINY_PNG)
    path.with_suffix(".txt").write_text(note + "\n", encoding="utf-8")


def _plot_with_matplotlib(kv_runs: list[dict], e2e_runs: list[dict], output_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    kv_scenarios = [run.get("scenario", "") for run in kv_runs]
    kv_store = [float(run.get("store_latency_ms", 0.0)) for run in kv_runs]
    kv_retrieve = [float(run.get("retrieve_latency_ms", 0.0)) for run in kv_runs]
    kv_value_cos = [float(run.get("value_cosine_sim", 0.0)) for run in kv_runs]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    x = list(range(len(kv_scenarios)))
    ax.plot(x, kv_store, marker="o", label="store_latency_ms")
    ax.plot(x, kv_retrieve, marker="o", label="retrieve_latency_ms")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("shape=", "")[:28] for s in kv_scenarios], rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("KV Cache Latency by Scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "kv_latency.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, kv_value_cos, marker="o", color="#2d6a4f")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("shape=", "")[:28] for s in kv_scenarios], rotation=45, ha="right")
    ax.set_ylabel("Cosine")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("KV Value Cosine Similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "kv_quality_cosine.png", dpi=150)
    plt.close(fig)

    e2e_scenarios = [run.get("scenario", "") for run in e2e_runs]
    e2e_miss = [float(run.get("cache_miss_total_latency_ms", 0.0)) for run in e2e_runs]
    e2e_hit = [float(run.get("cache_hit_total_latency_ms", 0.0)) for run in e2e_runs]
    e2e_quant = [float(run.get("quant_audit_cosine") or 0.0) for run in e2e_runs]
    e2e_sparse = [float(run.get("sparse_audit_cosine") or 0.0) for run in e2e_runs]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    x2 = list(range(len(e2e_scenarios)))
    ax.plot(x2, e2e_miss, marker="o", label="cache_miss_total_latency_ms")
    ax.plot(x2, e2e_hit, marker="o", label="cache_hit_total_latency_ms")
    ax.set_xticks(x2)
    ax.set_xticklabels(e2e_scenarios, rotation=35, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E2E Latency by Scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "e2e_latency.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x2, e2e_quant, marker="o", label="quant_audit_cosine")
    ax.plot(x2, e2e_sparse, marker="o", label="sparse_audit_cosine")
    ax.set_xticks(x2)
    ax.set_xticklabels(e2e_scenarios, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cosine")
    ax.set_title("E2E Quality Cosine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "e2e_quality_cosine.png", dpi=150)
    plt.close(fig)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plot artifacts")
    parser.add_argument("--input-dir", default="artifacts/proof/main8_1")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kv_runs = _load_runs(input_dir / "kv_cache_runs.json")
    e2e_runs = _load_runs(input_dir / "e2e_scenarios.json")

    used_matplotlib = _plot_with_matplotlib(kv_runs, e2e_runs, output_dir)
    if not used_matplotlib:
        _write_placeholder_png(
            output_dir / "kv_latency.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "kv_quality_cosine.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "e2e_latency.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "e2e_quality_cosine.png",
            "matplotlib unavailable; placeholder only",
        )

    _write_placeholder_png(
        output_dir / "custom_kernel_alpha_pending_benchmark.png",
        "Placeholder: custom kernel benchmark plot pending dedicated benchmark dataset.",
    )

    print(f"Wrote plot artifacts to {output_dir}")


if __name__ == "__main__":
    main()
