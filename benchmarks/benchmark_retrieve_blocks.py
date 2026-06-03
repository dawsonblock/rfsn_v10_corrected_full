#!/usr/bin/env python3
"""Benchmark retrieve_blocks partial reconstruction vs full retrieve()."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from statistics import median

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def get_metadata() -> dict:
    return {
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_retrieve_blocks(
    shape,
    k_bits,
    v_bits,
    use_incoherent,
    block_size,
    iterations=5,
):
    mx.random.seed(42)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits,
            v_bits=v_bits,
            use_incoherent=use_incoherent,
            max_memory_gb=2.0,
            max_pinned_memory_gb=1.0,
            cache_dir=td,
            block_size=block_size,
        )
        k = mx.random.normal(shape)
        v = mx.random.normal(shape)
        mx.eval(k, v)

        mgr.store("bench", k, v, shape[2])
        cache = mgr.active_caches["bench"]
        num_blocks = cache.num_blocks

        # Full retrieve benchmark
        full_times = []
        for _ in range(iterations):
            t0 = time.monotonic()
            result = mgr.retrieve("bench", out_dtype=mx.float32)
            mx.eval(result[0], result[1])
            t1 = time.monotonic()
            full_times.append(t1 - t0)

        # Partial retrieve (first half of blocks)
        half_blocks = list(range(0, num_blocks // 2))
        partial_times = []
        for _ in range(iterations):
            t0 = time.monotonic()
            result = mgr.retrieve_blocks(
                "bench", half_blocks, block_size=block_size,
            )
            mx.eval(result[0], result[1])
            t1 = time.monotonic()
            partial_times.append(t1 - t0)

        # Partial retrieve (every other block - sparse)
        sparse_blocks = list(range(0, num_blocks, 2))
        sparse_times = []
        for _ in range(iterations):
            t0 = time.monotonic()
            result = mgr.retrieve_blocks(
                "bench", sparse_blocks, block_size=block_size,
            )
            mx.eval(result[0], result[1])
            t1 = time.monotonic()
            sparse_times.append(t1 - t0)

        return {
            "shape": str(shape),
            "k_bits": k_bits,
            "v_bits": v_bits,
            "use_incoherent": use_incoherent,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "full_retrieve_ms": float(median(full_times) * 1000.0),
            "half_blocks_ms": float(median(partial_times) * 1000.0),
            "sparse_blocks_ms": float(median(sparse_times) * 1000.0),
            "half_speedup": float(median(full_times) / median(partial_times)),
            "sparse_speedup": float(median(full_times) / median(sparse_times)),
        }


def main():
    print("=" * 60)
    print("RFSN v10 retrieve_blocks Benchmark")
    print("=" * 60)
    meta = get_metadata()
    print(json.dumps(meta, indent=2))
    print()

    shapes = [
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 8, 4096, 64),
        (1, 8, 8192, 64),
    ]
    configs = [
        (8, 3, True),
        (8, 3, False),
    ]

    results = {"metadata": meta, "runs": []}

    for shape in shapes:
        for k_bits, v_bits, use_inc in configs:
            r = benchmark_retrieve_blocks(
                shape, k_bits, v_bits, use_inc, block_size=64,
            )
            line = (
                f"  {shape} k={k_bits}b v={v_bits}b inc={use_inc}: "
                f"full={r['full_retrieve_ms']:.2f}ms "
                f"half={r['half_blocks_ms']:.2f}ms "
                f"(speedup={r['half_speedup']:.2f}x) "
                f"sparse={r['sparse_blocks_ms']:.2f}ms "
                f"(speedup={r['sparse_speedup']:.2f}x)"
            )
            print(line)
            results["runs"].append(r)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
