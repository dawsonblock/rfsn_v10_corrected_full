#!/usr/bin/env python3
"""
RFSN v10 - Attention Correctness Tests.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.attention import AdaptiveBlockSparseAttention


def test_sparse_attention_dense_fallback_prefill():
    q = mx.random.normal((1, 4, 8, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    out, active, mode = AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.25)
    mx.eval(out)

    assert out.shape == q.shape
    assert active == 2  # ceil(128 / 64)
    assert mode == "dense_prefill"


def test_sparse_attention_decode_shape():
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 512, 64))
    v = mx.random.normal((1, 4, 512, 64))

    out, active, mode = AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.25)
    mx.eval(out)

    assert out.shape == q.shape
    assert active >= 1
    assert mode == "sparse_compacted"


def test_sparse_attention_rejects_invalid_top_k():
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 512, 64))
    v = mx.random.normal((1, 4, 512, 64))

    with pytest.raises(ValueError):
        AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.0)

    with pytest.raises(ValueError):
        AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=1.5)


def test_sparse_attention_rejects_shape_mismatch():
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 512, 64))
    v = mx.random.normal((1, 4, 256, 64))

    with pytest.raises(ValueError, match="shape mismatch"):
        AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.25)


def test_sparse_attention_padding_safe_decode():
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 130, 64))
    v = mx.random.normal((1, 4, 130, 64))

    out, active, mode = AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.5)
    mx.eval(out)

    assert out.shape == q.shape
    assert active >= 1


def test_sparse_attention_dense_fallback_when_kv_not_strictly_past():
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 512, 64))
    v = mx.random.normal((1, 4, 512, 64))

    out, active, mode = AdaptiveBlockSparseAttention.execute(
        q,
        k,
        v,
        top_k_ratio=0.25,
        kv_is_strictly_past=False,
    )
    mx.eval(out)

    assert out.shape == q.shape
    assert active == 8  # ceil(512 / 64)
    assert mode == "dense_not_strictly_past"


def test_sparse_attention_rejects_bad_rank():
    q = mx.random.normal((1, 4, 64))
    k = mx.random.normal((1, 4, 512, 64))
    v = mx.random.normal((1, 4, 512, 64))

    with pytest.raises(ValueError, match="queries must be"):
        AdaptiveBlockSparseAttention.execute(q, k, v, top_k_ratio=0.25)
