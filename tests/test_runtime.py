#!/usr/bin/env python3
"""
RFSN v10 - Runtime Orchestrator Tests.
"""
from __future__ import annotations

import pytest
import mlx.core as mx

from rfsn_v10.bitpack import BitPackedQuantizer
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.attention import AdaptiveBlockSparseAttention
from rfsn_v10.runtime import RFSNRuntime, TelemetryEvent


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


@pytest.fixture
def runtime(tmp_path, kv_manager):
    return RFSNRuntime(
        kv_manager=kv_manager,
        model_id="test_model",
        block_size=64,
        audit_mode=False,
        top_k_ratio=1.0,
    )


@pytest.fixture
def runtime_with_audit(tmp_path, kv_manager):
    return RFSNRuntime(
        kv_manager=kv_manager,
        model_id="test_model",
        block_size=64,
        audit_mode=True,
        top_k_ratio=0.5,
    )


# --- Cache key tests ---

def test_cache_key_is_deterministic():
    key1 = RFSNRuntime._make_cache_key(
        model_id="m", layer_id="l0", batch_id="b1",
        skill_pattern="summarize", shape=(1, 8, 128, 64),
        dtype="float32", k_bits=8, v_bits=3,
        group_size=64, use_incoherent=True, format_version="rfsn_v10",
    )
    key2 = RFSNRuntime._make_cache_key(
        model_id="m", layer_id="l0", batch_id="b1",
        skill_pattern="summarize", shape=(1, 8, 128, 64),
        dtype="float32", k_bits=8, v_bits=3,
        group_size=64, use_incoherent=True, format_version="rfsn_v10",
    )
    assert key1 == key2


def test_cache_key_differs_by_layer():
    key1 = RFSNRuntime._make_cache_key(
        model_id="m", layer_id="l0", batch_id="b1",
        skill_pattern="summarize", shape=(1, 8, 128, 64),
        dtype="float32", k_bits=8, v_bits=3,
        group_size=64, use_incoherent=True, format_version="rfsn_v10",
    )
    key2 = RFSNRuntime._make_cache_key(
        model_id="m", layer_id="l1", batch_id="b1",
        skill_pattern="summarize", shape=(1, 8, 128, 64),
        dtype="float32", k_bits=8, v_bits=3,
        group_size=64, use_incoherent=True, format_version="rfsn_v10",
    )
    assert key1 != key2


def test_cache_key_differs_by_shape():
    key1 = RFSNRuntime._make_cache_key(
        model_id="m", layer_id="l0", batch_id="b1",
        skill_pattern="summarize", shape=(1, 8, 128, 64),
        dtype="float32", k_bits=8, v_bits=3,
        group_size=64, use_incoherent=True, format_version="rfsn_v10",
    )
    key2 = RFSNRuntime._make_cache_key(
        model_id="m", layer_id="l0", batch_id="b1",
        skill_pattern="summarize", shape=(1, 8, 256, 64),
        dtype="float32", k_bits=8, v_bits=3,
        group_size=64, use_incoherent=True, format_version="rfsn_v10",
    )
    assert key1 != key2


# --- Execute decode step tests ---

def test_runtime_decode_step_returns_output_and_telemetry(runtime):
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    output, info = runtime.execute_decode_step(
        skill_pattern="test",
        layer_id="l0",
        batch_id="b1",
        queries=q,
        keys=k,
        values=v,
    )

    assert output.shape == q.shape
    assert "task_id" in info
    assert "kv_cache_hit" in info
    assert "total_latency_ms" in info


def test_runtime_cache_miss_then_hit(runtime):
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    # First call: cache miss
    _, info1 = runtime.execute_decode_step(
        skill_pattern="test", layer_id="l0", batch_id="b1",
        queries=q, keys=k, values=v,
    )
    assert info1["kv_cache_hit"] is False

    # Second call with same key: cache hit (same skill_pattern stores to same key)
    _, info2 = runtime.execute_decode_step(
        skill_pattern="test", layer_id="l0", batch_id="b1",
        queries=q, keys=k, values=v,
    )
    # Note: the cache key includes shape/dtype etc, so same params = hit
    assert info2["kv_cache_hit"] is True


def test_runtime_audit_mode_records_metrics(runtime_with_audit):
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    runtime_with_audit.execute_decode_step(
        skill_pattern="audit_test", layer_id="l0", batch_id="b1",
        queries=q, keys=k, values=v,
    )

    telemetry = runtime_with_audit.get_telemetry()
    assert len(telemetry) == 1
    event = telemetry[0]
    assert event.audit_enabled is True
    # When audit is on, sparse and dense both run, so we get audit metrics
    # (though at top_k_ratio=0.5, sparse != dense, so values may differ)


def test_runtime_rejects_bad_rank(runtime):
    q = mx.random.normal((1, 4, 64))  # 3D, should be 4D
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    with pytest.raises(ValueError, match="4D"):
        runtime.execute_decode_step(
            skill_pattern="bad", layer_id="l0", batch_id="b1",
            queries=q, keys=k, values=v,
        )


def test_runtime_rejects_kv_shape_mismatch(runtime):
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 64, 64))  # Different T dimension

    with pytest.raises(ValueError, match="shape mismatch"):
        runtime.execute_decode_step(
            skill_pattern="mismatch", layer_id="l0", batch_id="b1",
            queries=q, keys=k, values=v,
        )


def test_runtime_telemetry_clear(runtime):
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    runtime.execute_decode_step(
        skill_pattern="clear_test", layer_id="l0", batch_id="b1",
        queries=q, keys=k, values=v,
    )
    assert len(runtime.get_telemetry()) == 1

    runtime.clear_telemetry()
    assert len(runtime.get_telemetry()) == 0


def test_runtime_top_k_ratio_override(runtime):
    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 256, 64))
    v = mx.random.normal((1, 4, 256, 64))

    # Override top_k_ratio to force sparse path
    output, info = runtime.execute_decode_step(
        skill_pattern="sparse_test", layer_id="l0", batch_id="b1",
        queries=q, keys=k, values=v,
        top_k_ratio=0.25,
    )

    assert output.shape == q.shape
    # With top_k_ratio=0.25 and 256 tokens, should have fewer active blocks
    assert info["effective_sparsity"] > 0.0


def test_runtime_dense_fallback_for_prefill(runtime):
    # T_q > 1 should trigger dense fallback in sparse attention
    q = mx.random.normal((1, 4, 8, 64))  # T_q = 8 (prefill)
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    output, info = runtime.execute_decode_step(
        skill_pattern="prefill_test", layer_id="l0", batch_id="b1",
        queries=q, keys=k, values=v,
        top_k_ratio=0.25,
    )

    assert output.shape == q.shape
    # Dense fallback should succeed
    assert info["dense_success"] is True or info["sparse_success"] is True
