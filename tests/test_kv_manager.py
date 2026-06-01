#!/usr/bin/env python3
"""
RFSN v10 - KV Manager Additional Tests.
Covers distribution tests, mode tests, corruption tests, and multi-shape tests.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager, TurboQuantKVCache


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    dot = mx.sum(a_f * b_f)
    norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
    return (dot / mx.maximum(norm, mx.array(1e-8))).item()


def rel_mae(a: mx.array, b: mx.array) -> float:
    denom = mx.maximum(mx.mean(mx.abs(a)), mx.array(1e-8))
    return (mx.mean(mx.abs(a - b)) / denom).item()


@pytest.fixture
def kv_manager(tmp_path):
    return RFSNTurboQuantKVManager(
        k_bits=8,
        v_bits=3,
        use_incoherent=True,
        max_memory_gb=0.5,
        max_pinned_memory_gb=0.1,
        cache_dir=str(tmp_path),
    )


# --- Shape tests ---

@pytest.mark.parametrize("shape", [
    (1, 8, 128, 64),
    (1, 8, 2048, 64),
    (1, 32, 4096, 128),
    (2, 16, 1024, 64),
    (4, 16, 2048, 128),
])
def test_store_retrieve_various_shapes(kv_manager, shape):
    mx.random.seed(42)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    kv_manager.store("shape_test", k, v, shape[2])
    result = kv_manager.retrieve("shape_test", out_dtype=mx.float32)
    assert result is not None
    k_rec, v_rec = result
    mx.eval(k_rec, v_rec)
    assert k_rec.shape == shape
    assert v_rec.shape == shape


# --- Distribution tests ---

@pytest.fixture
def kv_manager_plain(tmp_path):
    return RFSNTurboQuantKVManager(
        k_bits=8,
        v_bits=3,
        use_incoherent=False,
        max_memory_gb=0.5,
        max_pinned_memory_gb=0.1,
        cache_dir=str(tmp_path),
    )


def test_normal_distribution(kv_manager_plain):
    mx.random.seed(42)
    shape = (1, 8, 1024, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    kv_manager_plain.store("normal", k, v, 1024)
    k_rec, v_rec = kv_manager_plain.retrieve("normal", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert cosine_similarity(k, k_rec) > 0.95
    assert rel_mae(k, k_rec) < 0.20


def test_small_magnitude(kv_manager_plain):
    mx.random.seed(42)
    shape = (1, 8, 512, 64)
    k = mx.random.normal(shape) * 0.01
    v = mx.random.normal(shape) * 0.01
    kv_manager_plain.store("small", k, v, 512)
    k_rec, v_rec = kv_manager_plain.retrieve("small", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    # Small magnitudes should still reconstruct well with 8-bit keys
    assert cosine_similarity(k, k_rec) > 0.90


def test_large_magnitude(kv_manager_plain):
    mx.random.seed(42)
    shape = (1, 8, 512, 64)
    k = mx.random.normal(shape) * 100.0
    v = mx.random.normal(shape) * 100.0
    kv_manager_plain.store("large", k, v, 512)
    k_rec, v_rec = kv_manager_plain.retrieve("large", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert cosine_similarity(k, k_rec) > 0.95


def test_all_zeros(kv_manager_plain):
    shape = (1, 8, 256, 64)
    k = mx.zeros(shape)
    v = mx.zeros(shape)
    kv_manager_plain.store("zeros", k, v, 256)
    k_rec, v_rec = kv_manager_plain.retrieve("zeros", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert mx.max(mx.abs(k_rec)).item() < 1e-6
    assert mx.max(mx.abs(v_rec)).item() < 1e-6


def test_alternating_signs(kv_manager_plain):
    mx.random.seed(42)
    shape = (1, 8, 512, 64)
    k = mx.random.normal(shape)
    # Force alternating signs
    idx = mx.arange(k.size).reshape(k.shape)
    k = mx.where(idx % 2 == 0, k, -k)
    v = mx.random.normal(shape)
    kv_manager_plain.store("alt_signs", k, v, 512)
    k_rec, v_rec = kv_manager_plain.retrieve("alt_signs", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert cosine_similarity(k, k_rec) > 0.90


# --- Mode tests ---

def test_use_incoherent_false(tmp_path):
    manager = RFSNTurboQuantKVManager(
        k_bits=8, v_bits=3, use_incoherent=False,
        max_memory_gb=0.5, max_pinned_memory_gb=0.1,
        cache_dir=str(tmp_path),
    )
    mx.random.seed(42)
    shape = (1, 8, 512, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    manager.store("plain", k, v, 512)
    k_rec, v_rec = manager.retrieve("plain", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert cosine_similarity(k, k_rec) > 0.95


def test_different_k_v_bits(tmp_path):
    manager = RFSNTurboQuantKVManager(
        k_bits=8, v_bits=3, use_incoherent=False,
        max_memory_gb=0.5, max_pinned_memory_gb=0.1,
        cache_dir=str(tmp_path),
    )
    mx.random.seed(42)
    shape = (1, 8, 512, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    manager.store("diff_bits", k, v, 512)
    k_rec, v_rec = manager.retrieve("diff_bits", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    # 8-bit keys should be high fidelity
    assert cosine_similarity(k, k_rec) > 0.95
    # 3-bit values will be coarser
    assert cosine_similarity(v, v_rec) > 0.75


@pytest.mark.parametrize(
    "use_wht,use_incoherent_signs",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_split_transform_flags_roundtrip(tmp_path, use_wht, use_incoherent_signs):
    manager = RFSNTurboQuantKVManager(
        k_bits=8,
        v_bits=3,
        use_wht=use_wht,
        use_incoherent_signs=use_incoherent_signs,
        max_memory_gb=0.5,
        max_pinned_memory_gb=0.1,
        cache_dir=str(tmp_path),
    )

    mx.random.seed(42)
    shape = (1, 8, 256, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)

    manager.store("split_flags", k, v, 256)
    cache = manager.active_caches["split_flags"]
    assert cache.use_wht is use_wht
    assert cache.use_incoherent_signs is use_incoherent_signs

    k_rec, v_rec = manager.retrieve("split_flags", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert k_rec.shape == shape
    assert v_rec.shape == shape
    assert cosine_similarity(k, k_rec) > 0.70


def test_out_dtype_float16(kv_manager_plain):
    mx.random.seed(42)
    shape = (1, 8, 512, 64)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    kv_manager_plain.store("fp16_test", k, v, 512)
    k_rec, v_rec = kv_manager_plain.retrieve("fp16_test", out_dtype=mx.float16)
    mx.eval(k_rec, v_rec)
    assert k_rec.dtype == mx.float16
    assert v_rec.dtype == mx.float16


# --- Corruption tests ---

def test_corruption_wrong_scale_count(kv_manager):
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    kv_manager.store("corrupt_scale", k, v, 128)
    # Corrupt the scale count
    cache = kv_manager.active_caches["corrupt_scale"]
    cache.k_scales = mx.zeros((999,), dtype=mx.float32)
    with pytest.raises(ValueError, match="Scale count mismatch"):
        kv_manager.retrieve("corrupt_scale", out_dtype=mx.float32)


def test_corruption_wrong_format_version(kv_manager):
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    kv_manager.store("corrupt_fmt", k, v, 128)
    kv_manager.active_caches["corrupt_fmt"].format_version = "v9"
    with pytest.raises(ValueError, match="Unsupported cache format"):
        kv_manager.retrieve("corrupt_fmt", out_dtype=mx.float32)


def test_corruption_wrong_k_bits(kv_manager):
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    kv_manager.store("corrupt_kbits", k, v, 128)
    kv_manager.active_caches["corrupt_kbits"].k_bits = 4
    with pytest.raises(ValueError, match="metadata"):
        kv_manager.retrieve("corrupt_kbits", out_dtype=mx.float32)


# --- Cache replacement test ---

def test_store_replaces_existing_entry(kv_manager):
    shape = (1, 8, 128, 64)
    k1 = mx.random.normal(shape)
    v1 = mx.random.normal(shape)
    kv_manager.store("replace_test", k1, v1, 128)

    k2 = mx.random.normal(shape) * 2.0
    v2 = mx.random.normal(shape) * 2.0
    kv_manager.store("replace_test", k2, v2, 128)

    # Only one cache entry should exist
    assert len(kv_manager.active_caches) == 1
    assert "replace_test" in kv_manager.active_caches

    # Retrieved data should match the second store
    k_rec, v_rec = kv_manager.retrieve("replace_test", out_dtype=mx.float32)
    mx.eval(k_rec, v_rec)
    assert cosine_similarity(k2, k_rec) > 0.95


def test_estimate_compressed_bytes_for_shape_is_positive(kv_manager):
    shape = (1, 8, 512, 64)
    estimated = kv_manager.estimate_compressed_bytes_for_shape(shape)
    assert estimated > 0


def test_estimate_compressed_bytes_reduces_with_higher_bits_for_packing(kv_manager):
    shape = (1, 8, 512, 64)
    bytes_8_8 = kv_manager.estimate_compressed_bytes_for_shape(shape, k_bits=8, v_bits=8)
    bytes_8_3 = kv_manager.estimate_compressed_bytes_for_shape(shape, k_bits=8, v_bits=3)
    assert bytes_8_3 < bytes_8_8


def test_sign_cache_is_instance_scoped(tmp_path):
    manager_a = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path / "a"),
        use_incoherent=True,
    )
    manager_b = RFSNTurboQuantKVManager(
        cache_dir=str(tmp_path / "b"),
        use_incoherent=True,
    )

    shape = (1, 2, 4, 8)
    _ = manager_a._apply_signs_on_the_fly(mx.ones(shape), seed=123)

    assert len(manager_a._sign_cache) == 1
    assert len(manager_b._sign_cache) == 0
