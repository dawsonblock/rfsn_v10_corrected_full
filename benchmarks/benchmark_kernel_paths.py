#!/usr/bin/env python3
"""Benchmark sequential vs metal reconstruction paths for Main11."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


SHAPES = [
    (1, 4, 512, 64),
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 32, 4096, 128),
]


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def _make_manager(cache_dir: Path, mode: str) -> RFSNTurboQuantKVManager:
    if mode == "sequential_reference":
        return RFSNTurboQuantKVManager(
            cache_dir=str(cache_dir),
            k_bits=8,
            v_bits=3,
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=False,
        )
    if mode == "metal_sign_only":
        return RFSNTurboQuantKVManager(
            cache_dir=str(cache_dir),
            k_bits=8,
            v_bits=3,
            use_wht=False,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            strict_metal=False,
        )
    if mode == "metal_packed_dequant":
        return RFSNTurboQuantKVManager(
            cache_dir=str(cache_dir),
            k_bits=8,
            v_bits=3,
            use_wht=False,
            use_incoherent_signs=False,
            prefer_metal_kernels=True,
            strict_metal=False,
        )
    if mode == "metal_packed_dequant_wht_sign":
        return RFSNTurboQuantKVManager(
            cache_dir=str(cache_dir),
            k_bits=8,
            v_bits=3,
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            strict_metal=False,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _median_retrieve_latency_ms(manager: RFSNTurboQuantKVManager, key: str, iterations: int) -> float:
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        rec = manager.retrieve(key, out_dtype=mx.float32)
        dt = (time.perf_counter() - t0) * 1000.0
        if rec is None:
            raise RuntimeError("Expected cache hit during retrieval benchmark")
        mx.eval(rec[0], rec[1])
        latencies.append(dt)
    return float(statistics.median(latencies))


def _bench_mode(shape: tuple[int, int, int, int], mode: str, iterations: int, base_dir: Path) -> dict:
    mx.random.seed(42)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)

    key = f"mode={mode}|shape={shape}"
    mode_dir = base_dir / mode / ("x".join(str(v) for v in shape))
    mode_dir.mkdir(parents=True, exist_ok=True)

    manager = _make_manager(mode_dir, mode)
    manager.store(key, keys, values, token_count=shape[2])

    latency_ms = _median_retrieve_latency_ms(manager, key, iterations=iterations)
    k_out, _ = manager.retrieve(key, out_dtype=mx.float32)
    mx.eval(k_out)

    cache = manager.active_caches[key]
    ref_k = manager._reconstruct_packed_dequant_wht(
        packed=cache.k_packed,
        scales=cache.k_scales,
        n_values=cache.k_n_values,
        shape=cache.shape,
        bits=cache.k_bits,
        seed=cache.seed,
        use_wht=cache.use_wht,
        use_incoherent_signs=cache.use_incoherent_signs,
        out_dtype=mx.float32,
    )
    mx.eval(ref_k)

    diff = mx.max(mx.abs(k_out - ref_k)).item()
    cosine = cosine_similarity(k_out, ref_k)

    return {
        "shape": str(shape),
        "mode": mode,
        "retrieve_latency_ms": latency_ms,
        "kv_reconstruction_kernel": manager.last_reconstruction_kernel,
        "max_abs_diff_vs_reference": float(diff),
        "cosine_vs_reference": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark kernel reconstruction paths")
    parser.add_argument(
        "--out",
        default="artifacts/proof/main11/kernel_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = [
        "sequential_reference",
        "metal_sign_only",
        "metal_packed_dequant",
        "metal_packed_dequant_wht_sign",
    ]

    runs: list[dict] = []
    reference_latencies: dict[str, float] = {}

    for shape in SHAPES:
        shape_label = str(shape)
        for mode in modes:
            row = _bench_mode(shape, mode, args.iterations, out_path.parent / "kernel_bench_tmp")
            runs.append(row)
            if mode == "sequential_reference":
                reference_latencies[shape_label] = row["retrieve_latency_ms"]

    for row in runs:
        ref_latency = reference_latencies.get(row["shape"], row["retrieve_latency_ms"])
        row["speedup_vs_reference"] = (
            float(ref_latency) / float(row["retrieve_latency_ms"])
            if float(row["retrieve_latency_ms"]) > 0
            else 0.0
        )

    payload = {
        "runs": runs,
        "summary": {
            "equivalence_pass_modes": [
                row["mode"]
                for row in runs
                if row["cosine_vs_reference"] > 0.999
            ],
            "speedup_modes": [
                row["mode"]
                for row in runs
                if row["mode"].startswith("metal_") and row["speedup_vs_reference"] > 1.0
            ],
        },
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote kernel benchmark to {out_path}")


if __name__ == "__main__":
    main()
