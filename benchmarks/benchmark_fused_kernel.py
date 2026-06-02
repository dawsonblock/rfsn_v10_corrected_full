#!/usr/bin/env python3
"""Benchmark fused kernel performance vs sequential path.

Compares the new fused packed-dequant-WHT-sign kernel against
the sequential multi-kernel path to measure performance improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.kernels import (
    apply_hash_signs_metal,
    packed_dequant_metal,
    packed_dequant_wht_sign_metal,
    wht64_metal,
)
from rfsn_v10.bitpack import BitPackedQuantizer


def generate_test_data(n_values: int, bits: int, group_size: int) -> tuple:
    """Generate test data for benchmarking."""
    mx.random.seed(42)

    # Generate random values
    values = mx.random.normal((n_values,))

    # Quantize
    quantizer = BitPackedQuantizer(bits=bits, group_size=group_size)
    packed, scales = quantizer.quantize(values)

    return packed, scales, values


def benchmark_sequential_path(
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    bits: int,
    group_size: int,
    seed: int,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark the sequential multi-kernel path."""
    latencies = []

    for _ in range(iterations):
        import time

        t0 = time.perf_counter()

        deq = packed_dequant_metal(
            packed=packed,
            scales=scales,
            n_values=n_values,
            bits=bits,
            group_size=group_size,
            out_dtype=mx.float32,
        )
        wht = wht64_metal(deq)
        _ = apply_hash_signs_metal(wht, seed=seed)

        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    return {
        "path": "sequential",
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
    }


def benchmark_fused_path(
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    bits: int,
    group_size: int,
    seed: int,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark the fused single-kernel path."""
    latencies = []

    for _ in range(iterations):
        import time

        t0 = time.perf_counter()

        _ = packed_dequant_wht_sign_metal(
            packed=packed,
            scales=scales,
            n_values=n_values,
            bits=bits,
            group_size=group_size,
            seed=seed,
            out_dtype=mx.float32,
        )

        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    return {
        "path": "fused",
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
    }


def validate_equivalence(
    sequential_result: mx.array,
    fused_result: mx.array,
) -> dict[str, float]:
    """Validate numerical equivalence between paths."""
    diff = mx.abs(sequential_result - fused_result)
    max_abs_diff = float(mx.max(diff).item())
    mean_abs_diff = float(mx.mean(diff).item())

    # Cosine similarity
    seq_flat = sequential_result.flatten()
    fused_flat = fused_result.flatten()
    dot = mx.sum(seq_flat * fused_flat)
    norm = mx.sqrt(mx.sum(seq_flat * seq_flat)) * mx.sqrt(
        mx.sum(fused_flat * fused_flat)
    )
    cosine = (dot / mx.maximum(norm, mx.array(1e-8))).item()

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused kernel performance"
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main12/fused_kernel_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        {"n_values": 4096, "bits": 8, "group_size": 64},
        {"n_values": 8192, "bits": 8, "group_size": 64},
        {"n_values": 16384, "bits": 8, "group_size": 64},
        {"n_values": 32768, "bits": 8, "group_size": 64},
        {"n_values": 65536, "bits": 8, "group_size": 64},
    ]

    results = []
    seed = 42

    for config in configs:
        n_values = config["n_values"]
        bits = config["bits"]
        group_size = config["group_size"]

        print(f"Benchmarking: n_values={n_values}, bits={bits}")

        packed, scales, _ = generate_test_data(n_values, bits, group_size)

        # Benchmark sequential
        seq_stats = benchmark_sequential_path(
            packed, scales, n_values, bits, group_size, seed, args.iterations
        )
        print(f"  Sequential: {seq_stats['latency_ms_mean']:.3f}ms")

        # Benchmark fused
        fused_stats = benchmark_fused_path(
            packed, scales, n_values, bits, group_size, seed, args.iterations
        )
        print(f"  Fused: {fused_stats['latency_ms_mean']:.3f}ms")

        # Validate equivalence
        seq_result = packed_dequant_metal(
            packed, scales, n_values, bits, group_size, mx.float32
        )
        seq_result = wht64_metal(seq_result)
        seq_result = apply_hash_signs_metal(seq_result, seed=seed)

        fused_result = packed_dequant_wht_sign_metal(
            packed, scales, n_values, bits, group_size, seed, mx.float32
        )

        validation = validate_equivalence(seq_result, fused_result)
        print(f"  Cosine: {validation['cosine_similarity']:.6f}")
        print(f"  Max diff: {validation['max_abs_diff']:.6e}")

        speedup = seq_stats["latency_ms_mean"] / fused_stats["latency_ms_mean"]
        print(f"  Speedup: {speedup:.2f}x")

        results.append(
            {
                "config": config,
                "sequential": seq_stats,
                "fused": fused_stats,
                "validation": validation,
                "speedup": speedup,
            }
        )

    # Summary
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_cosine = sum(
        r["validation"]["cosine_similarity"] for r in results
    ) / len(results)

    payload = {
        "results": results,
        "summary": {
            "avg_speedup": avg_speedup,
            "avg_cosine_similarity": avg_cosine,
            "meets_target": avg_speedup >= 1.2 and avg_cosine >= 0.999,
        },
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to: {out_path}")
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Average cosine: {avg_cosine:.6f}")
    print(f"Meets target (1.2x speedup, 0.999 cosine): {payload['summary']['meets_target']}")


if __name__ == "__main__":
    main()
