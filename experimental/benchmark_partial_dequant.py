#!/usr/bin/env python3
"""Benchmark partial dequantization latency-quality tradeoffs.

Compares full dequantization vs partial dequantization to measure
latency reduction and quality impact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx

from rfsn_v10.partial_dequant import PartialDequantConfig, PartialDequantManager
from rfsn_v10.bitpack import BitPackedQuantizer


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two tensors."""
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def benchmark_full_dequant(
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    bits: int,
    group_size: int,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark full dequantization."""
    quantizer = BitPackedQuantizer(bits=bits, group_size=group_size)

    latencies = []
    for _ in range(iterations):
        import time

        t0 = time.perf_counter()
        dequant = quantizer.unpack(packed, n_values, bits)
        dequant = quantizer._dequantize_unsigned(dequant, scales, bits)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    # Get final result for comparison
    full_result = quantizer.unpack(packed, n_values, bits)
    full_result = quantizer._dequantize_unsigned(full_result, scales, bits)

    return {
        "path": "full_dequant",
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "result": full_result,
    }


def benchmark_partial_dequant(
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    bits: int,
    group_size: int,
    config: PartialDequantConfig,
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark partial dequantization."""
    manager = PartialDequantManager(config)

    latencies = []
    results = []
    for i in range(iterations):
        import time

        t0 = time.perf_counter()
        result, metadata = manager.execute_partial_dequant(
            packed=packed,
            scales=scales,
            n_values=n_values,
            bits=bits,
            group_size=group_size,
            current_position=i,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        results.append((result, metadata))

    # Use last result for comparison
    partial_result, metadata = results[-1]

    return {
        "path": "partial_dequant",
        "config": {
            "hot_block_ratio": config.hot_block_ratio,
            "cold_block_ratio": config.cold_block_ratio,
        },
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "result": partial_result,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark partial dequantization performance"
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main12/partial_dequant_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mx.random.seed(42)
    n_values = 16384
    bits = 8
    group_size = 64

    # Generate test data
    values = mx.random.normal((n_values,))
    quantizer = BitPackedQuantizer(bits=bits, group_size=group_size)
    packed, scales = quantizer.quantize(values)

    configs = [
        PartialDequantConfig(hot_block_ratio=0.1),
        PartialDequantConfig(hot_block_ratio=0.3),
        PartialDequantConfig(hot_block_ratio=0.5),
        PartialDequantConfig(hot_block_ratio=0.7),
    ]

    results = []

    # Benchmark full dequant
    print("Benchmarking full dequantization...")
    full_stats = benchmark_full_dequant(
        packed, scales, n_values, bits, group_size, args.iterations
    )
    print(f"  Full dequant: {full_stats['latency_ms_mean']:.3f}ms")

    for config in configs:
        print(f"\nBenchmarking partial dequant (hot_ratio={config.hot_block_ratio})...")
        partial_stats = benchmark_partial_dequant(
            packed, scales, n_values, bits, group_size, config, args.iterations
        )
        print(f"  Partial dequant: {partial_stats['latency_ms_mean']:.3f}ms")

        # Compute quality metrics
        cos_sim = cosine_similarity(full_stats["result"], partial_stats["result"])
        max_abs_diff = float(
            mx.max(mx.abs(full_stats["result"] - partial_stats["result"])).item()
        )

        speedup = full_stats["latency_ms_mean"] / partial_stats["latency_ms_mean"]
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Cosine: {cos_sim:.6f}")
        print(f"  Max diff: {max_abs_diff:.6e}")

        results.append(
            {
                "config": {
                    "hot_block_ratio": config.hot_block_ratio,
                    "cold_block_ratio": config.cold_block_ratio,
                },
                "partial": {
                    "latency_ms_mean": partial_stats["latency_ms_mean"],
                    "latency_ms_p50": partial_stats["latency_ms_p50"],
                },
                "quality": {
                    "cosine_similarity": cos_sim,
                    "max_abs_diff": max_abs_diff,
                },
                "speedup": speedup,
                "meets_target": speedup >= 1.15 and cos_sim >= 0.99,
            }
        )

    payload = {
        "full_dequant": {
            "latency_ms_mean": full_stats["latency_ms_mean"],
            "latency_ms_p50": full_stats["latency_ms_p50"],
        },
        "partial_results": results,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to: {out_path}")

    # Find best configuration
    best = max(results, key=lambda x: x["speedup"] if x["meets_target"] else 0)
    print(f"\nBest configuration: hot_ratio={best['config']['hot_block_ratio']}")
    print(f"  Speedup: {best['speedup']:.2f}x")
    print(f"  Cosine: {best['quality']['cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()
