#!/usr/bin/env python3
"""
RFSN v10 - End-to-End Runtime Benchmarks.
Measures full decode pipeline: KV store + sparse attention + audit.
"""
from __future__ import annotations

import json
import math
import platform
import tempfile
import time
from datetime import datetime, timezone

import mlx.core as mx

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.runtime import RFSNRuntime
from rfsn_v10.attention import AdaptiveBlockSparseAttention


def get_metadata() -> dict:
    return {
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_e2e(shape_q, shape_kv, k_bits, v_bits, use_incoherent, top_k_ratio, iterations=5):
    mx.random.seed(42)
    with tempfile.TemporaryDirectory() as td:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, use_incoherent=use_incoherent,
            max_memory_gb=1.0, max_pinned_memory_gb=0.5, cache_dir=td,
        )
        runtime = RFSNRuntime(
            kv_manager=mgr, model_id="bench_model", block_size=64,
            audit_mode=True, top_k_ratio=top_k_ratio,
        )

        q = mx.random.normal(shape_q)
        k = mx.random.normal(shape_kv)
        v = mx.random.normal(shape_kv)
        mx.eval(q, k, v)

        timings = []
        for _ in range(iterations):
            output, info = runtime.execute_decode_step(
                skill_pattern="bench", layer_id="l0", batch_id="b1",
                queries=q, keys=k, values=v, top_k_ratio=top_k_ratio,
            )
            timings.append(info["total_latency_ms"])

        telemetry = runtime.get_telemetry()
        avg_cosine = None
        if telemetry and telemetry[-1].audit_cosine is not None:
            avg_cosine = telemetry[-1].audit_cosine

        return {
            "shape_q": str(shape_q),
            "shape_kv": str(shape_kv),
            "k_bits": k_bits,
            "v_bits": v_bits,
            "top_k_ratio": top_k_ratio,
            "avg_latency_ms": sum(timings) / len(timings),
            "kv_cache_hit": telemetry[-1].kv_cache_hit if telemetry else None,
            "audit_cosine": avg_cosine,
            "effective_sparsity": telemetry[-1].effective_sparsity if telemetry else None,
        }


def main():
    print("=" * 60)
    print("RFSN v10 End-to-End Benchmarks")
    print("=" * 60)
    meta = get_metadata()
    print(json.dumps(meta, indent=2))
    print()

    shape_q = (1, 8, 1, 64)
    shape_kv = (1, 8, 2048, 64)
    configs = [
        (8, 3, True, 0.25),
        (8, 3, True, 0.50),
        (8, 3, True, 0.75),
        (8, 3, True, 1.0),
        (8, 3, False, 0.50),
    ]

    results = {"metadata": meta, "runs": []}

    for k_bits, v_bits, use_inc, ratio in configs:
        r = benchmark_e2e(shape_q, shape_kv, k_bits, v_bits, use_inc, ratio)
        print(f"  k={k_bits}b v={v_bits}b incoherent={use_inc} top_k={ratio}: "
              f"latency={r['avg_latency_ms']:.2f}ms "
              f"cosine={r['audit_cosine']:.4f} "
              f"sparsity={r['effective_sparsity']:.2f}")
        results["runs"].append(r)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
