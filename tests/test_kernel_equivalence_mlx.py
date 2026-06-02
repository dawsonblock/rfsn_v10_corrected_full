#!/usr/bin/env python3
"""Main12 metal reconstruction route equivalence tests."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kernels import apply_hash_signs_metal
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten().astype(mx.float32)
    b_f = b.flatten().astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def test_metal_hash_sign_is_self_inverse() -> None:
    mx.random.seed(42)
    x = mx.random.normal((1, 8, 128, 64))
    seed = 12345
    y = apply_hash_signs_metal(x, seed)
    z = apply_hash_signs_metal(y, seed)
    mx.eval(z)
    diff = mx.max(mx.abs(x - z)).item()
    assert diff < 1e-5


@pytest.mark.parametrize(
    ("use_wht", "use_signs", "expected_label"),
    [
        (False, False, "metal_dequant"),
        (True, False, "metal_dequant_wht"),
        (False, True, "metal_dequant_sign"),
        (True, True, "metal_dequant_wht_sign"),
    ],
)
def test_metal_reconstruction_matches_reference(use_wht, use_signs, expected_label, tmp_path):
    shape = (1, 8, 512, 64)
    manager = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path),
        k_bits=8,
        v_bits=3,
        use_wht=use_wht,
        use_incoherent_signs=use_signs,
        prefer_metal_kernels=True,
        strict_metal=False,
    )

    mx.random.seed(7)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)
    manager.store("kernel_eq", keys, values, token_count=shape[2])

    metal_k, metal_v = manager.retrieve("kernel_eq", out_dtype=mx.float32)
    assert manager.last_reconstruction_kernel == expected_label

    cache = manager.active_caches["kernel_eq"]
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
    ref_v = manager._reconstruct_packed_dequant_wht(
        packed=cache.v_packed,
        scales=cache.v_scales,
        n_values=cache.v_n_values,
        shape=cache.shape,
        bits=cache.v_bits,
        seed=cache.seed,
        use_wht=cache.use_wht,
        use_incoherent_signs=cache.use_incoherent_signs,
        out_dtype=mx.float32,
    )

    mx.eval(metal_k, metal_v, ref_k, ref_v)
    assert cosine_similarity(metal_k, ref_k) > 0.999
    assert cosine_similarity(metal_v, ref_v) > 0.999


def test_strict_metal_uses_metal_route(tmp_path):
    cases = [
        (False, False, "metal_dequant"),
        (True, False, "metal_dequant_wht"),
        (False, True, "metal_dequant_sign"),
        (True, True, "metal_dequant_wht_sign"),
    ]

    mx.random.seed(1)
    shape = (1, 4, 128, 64)
    keys = mx.random.normal(shape)
    values = mx.random.normal(shape)

    for idx, (use_wht, use_signs, expected_label) in enumerate(cases):
        manager = RFSNTurboQuantKVManager(
            cache_dir=str(tmp_path / f"strict_{idx}"),
            k_bits=8,
            v_bits=3,
            use_wht=use_wht,
            use_incoherent_signs=use_signs,
            prefer_metal_kernels=True,
            strict_metal=True,
        )

        key = f"strict_metal_{idx}"
        manager.store(key, keys, values, token_count=shape[2])
        k_rec, v_rec = manager.retrieve(key, out_dtype=mx.float32)
        mx.eval(k_rec, v_rec)

        assert manager.last_reconstruction_kernel == expected_label
        assert manager.last_reconstruction_kernel != "metal_failed_fallback_reference"
