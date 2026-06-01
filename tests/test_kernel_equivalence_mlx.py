#!/usr/bin/env python3
"""Kernel routing equivalence tests for Main10 custom-kernel alpha path."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.runtime import RFSNRuntime


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def test_custom_kernel_matches_sequential_reconstruction(tmp_path):
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=True,
        use_incoherent_signs=True,
        use_custom_kernel=True,
    )

    mx.random.seed(42)
    shape = (1, 8, 256, 64)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)
    manager.store("kernel_eq", keys, values, token_count=shape[2])

    retrieved_k, _ = manager.retrieve("kernel_eq", out_dtype=mx.float32)
    assert manager.last_reconstruction_kernel == "custom"

    cache = manager.active_caches["kernel_eq"]
    sequential_k = manager._reconstruct_packed_dequant_wht(
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

    mx.eval(retrieved_k, sequential_k)
    assert cosine_similarity(retrieved_k, sequential_k) > 0.9999


def test_custom_kernel_falls_back_when_feature_combo_unsupported(tmp_path):
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=True,
        use_incoherent_signs=False,
        use_custom_kernel=True,
    )

    mx.random.seed(123)
    shape = (1, 4, 128, 64)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)
    manager.store("kernel_fb", keys, values, token_count=shape[2])

    k_rec, v_rec = manager.retrieve("kernel_fb", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert manager.last_reconstruction_kernel.startswith("sequential_fallback")


def test_runtime_flag_controls_kernel_route(tmp_path):
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=True,
        use_incoherent_signs=True,
        use_custom_kernel=True,
    )
    runtime = RFSNRuntime(
        kv_manager=manager,
        model_id="kernel_runtime",
        top_k_ratio=1.0,
        use_custom_kernel=False,
    )

    mx.random.seed(1)
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    _, info = runtime.execute_decode_step(
        skill_pattern="kernel_runtime_path",
        layer_id="l0",
        batch_id="b0",
        queries=q,
        keys=k,
        values=v,
    )

    assert info["kv_reconstruction_kernel"] == "sequential"
