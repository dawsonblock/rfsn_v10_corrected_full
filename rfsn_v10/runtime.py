"""RFSN v10 Runtime Orchestrator.

Integration layer between the KV cache manager, sparse attention engine,
and telemetry subsystem.  Coordinates a single decode step by:

1. Validating input tensors
2. Generating a composite cache key for KV retrieval/storage
3. Attempting sparse block-selective attention
4. Falling back to dense attention when sparse is unsafe
5. Optionally auditing sparse vs dense output quality
6. Recording a structured TelemetryEvent for observability
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import mlx.core as mx

from .bitpack import BitPackedQuantizer
from .kv_manager import RFSNTurboQuantKVManager, TurboQuantKVCache
from .attention import AdaptiveBlockSparseAttention


@dataclass
class TelemetryEvent:
    """Single decode-step telemetry record."""

    task_id: str
    model_id: str
    layer_id: str
    batch_id: str
    skill_pattern: str
    seq_len: int
    head_count: int
    head_dim: int
    top_k_ratio: float
    block_size: int
    num_active_blocks: int
    effective_sparsity: float
    kv_cache_hit: bool
    kv_cache_store_latency_ms: float
    kv_cache_retrieve_latency_ms: float
    attention_latency_ms: float
    total_latency_ms: float
    fallback_used: bool
    sparse_success: bool
    dense_success: bool
    audit_enabled: bool
    audit_cosine: Optional[float]
    audit_rel_mae: Optional[float]
    audit_max_abs_error: Optional[float]
    termination_reason: str


class RFSNRuntime:
    """Production-grade decode-step runtime orchestrator for RFSN v10."""

    def __init__(
        self,
        kv_manager: RFSNTurboQuantKVManager,
        model_id: str = "default",
        block_size: int = 64,
        audit_mode: bool = False,
        top_k_ratio: float = 1.0,
    ):
        self.kv_manager = kv_manager
        self.model_id = model_id
        self.block_size = block_size
        self.audit_mode = audit_mode
        self.top_k_ratio = top_k_ratio
        self._telemetry_log: list[TelemetryEvent] = []

    @staticmethod
    def _make_cache_key(
        model_id: str,
        layer_id: str,
        batch_id: str,
        skill_pattern: str,
        shape: tuple,
        dtype: str,
        k_bits: int,
        v_bits: int,
        group_size: int,
        use_incoherent: bool,
        format_version: str,
    ) -> str:
        return "|".join([
            model_id, layer_id, batch_id, skill_pattern,
            "x".join(str(d) for d in shape),
            str(dtype), str(k_bits), str(v_bits),
            str(group_size), str(use_incoherent),
            format_version,
        ])

    @staticmethod
    def _cosine_similarity(a: mx.array, b: mx.array) -> float:
        a_f = a.flatten().astype(mx.float32)
        b_f = b.flatten().astype(mx.float32)
        dot = mx.sum(a_f * b_f)
        norm = mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f))
        return (dot / mx.maximum(norm, mx.array(1e-8))).item()

    @staticmethod
    def _rel_mae(a: mx.array, b: mx.array) -> float:
        a_f = a.flatten().astype(mx.float32)
        b_f = b.flatten().astype(mx.float32)
        denom = mx.maximum(mx.mean(mx.abs(a_f)), mx.array(1e-8))
        return (mx.mean(mx.abs(a_f - b_f)) / denom).item()

    @staticmethod
    def _max_abs_error(a: mx.array, b: mx.array) -> float:
        return mx.max(mx.abs(a.flatten().astype(mx.float32) - b.flatten().astype(mx.float32))).item()

    def execute_decode_step(
        self,
        skill_pattern: str,
        layer_id: str,
        batch_id: str,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: Optional[float] = None,
    ) -> tuple[mx.array, dict]:
        task_id = str(uuid.uuid4())
        t_start = time.monotonic()

        # 1. Validate tensors
        if len(queries.shape) != 4:
            raise ValueError(f"queries must be 4D, got {queries.shape}")
        if keys.shape != values.shape:
            raise ValueError(f"keys/values shape mismatch: {keys.shape} vs {values.shape}")
        if keys.shape[-1] != queries.shape[-1]:
            raise ValueError(f"head_dim mismatch: keys={keys.shape[-1]}, queries={queries.shape[-1]}")

        B, H, T_q, D = queries.shape
        T_k = keys.shape[1] if len(keys.shape) == 3 else keys.shape[2]

        effective_top_k = top_k_ratio if top_k_ratio is not None else self.top_k_ratio

        # 2. Generate composite cache key
        cache_key = self._make_cache_key(
            model_id=self.model_id,
            layer_id=layer_id,
            batch_id=batch_id,
            skill_pattern=skill_pattern,
            shape=tuple(keys.shape),
            dtype=str(keys.dtype),
            k_bits=self.kv_manager.k_bits,
            v_bits=self.kv_manager.v_bits,
            group_size=self.kv_manager.group_size,
            use_incoherent=self.kv_manager.use_incoherent,
            format_version="rfsn_v10",
        )

        # 3-4. Try retrieve or store
        t_retrieve_start = time.monotonic()
        kv_result = self.kv_manager.retrieve(cache_key, out_dtype=keys.dtype)
        retrieve_latency_ms = (time.monotonic() - t_retrieve_start) * 1000.0

        kv_cache_hit = kv_result is not None
        if not kv_cache_hit:
            t_store_start = time.monotonic()
            self.kv_manager.store(cache_key, keys, values, T_k)
            store_latency_ms = (time.monotonic() - t_store_start) * 1000.0
        else:
            store_latency_ms = 0.0
            keys, values = kv_result

        # 5. Try sparse attention
        t_attn_start = time.monotonic()
        sparse_success = False
        dense_success = False
        fallback_used = False
        attn_output = None
        num_active_blocks = 0
        termination_reason = "unknown"

        sparse_output = None
        try:
            sparse_output, num_active_blocks = AdaptiveBlockSparseAttention.execute(
                queries, keys, values,
                top_k_ratio=effective_top_k,
                block_size=self.block_size,
                kv_is_strictly_past=True,
            )
            mx.eval(sparse_output)
            sparse_success = True
            termination_reason = "sparse_success"
        except Exception as e:
            termination_reason = f"sparse_failed: {e}"

        if not sparse_success or self.audit_mode:
            # 6. Dense fallback / audit
            try:
                dense_output = mx.fast.scaled_dot_product_attention(
                    queries, keys, values,
                    scale=1.0 / math.sqrt(D),
                )
                mx.eval(dense_output)
                dense_success = True

                if not sparse_success:
                    attn_output = dense_output
                    fallback_used = True
                    termination_reason = "dense_fallback"
                    num_active_blocks = (keys.shape[2] + self.block_size - 1) // self.block_size
            except Exception as e:
                termination_reason = f"catastrophic_failure: {e}"
                raise RuntimeError(f"All attention paths failed: {e}")

            if sparse_success and attn_output is None:
                attn_output = sparse_output

        if attn_output is None:
            raise RuntimeError("No valid attention output produced")

        attention_latency_ms = (time.monotonic() - t_attn_start) * 1000.0

        # 8. Audit comparison
        audit_cosine = None
        audit_rel_mae = None
        audit_max_abs_error = None
        if self.audit_mode and sparse_success and dense_success and sparse_output is not None:
            audit_cosine = self._cosine_similarity(sparse_output, dense_output)
            audit_rel_mae = self._rel_mae(sparse_output, dense_output)
            audit_max_abs_error = self._max_abs_error(sparse_output, dense_output)

        total_latency_ms = (time.monotonic() - t_start) * 1000.0

        # Compute effective sparsity
        total_blocks = max(1, (keys.shape[2] + self.block_size - 1) // self.block_size)
        effective_sparsity = 1.0 - (num_active_blocks / total_blocks)

        # 9. Record telemetry
        event = TelemetryEvent(
            task_id=task_id,
            model_id=self.model_id,
            layer_id=layer_id,
            batch_id=batch_id,
            skill_pattern=skill_pattern,
            seq_len=int(T_k),
            head_count=int(H),
            head_dim=int(D),
            top_k_ratio=effective_top_k,
            block_size=self.block_size,
            num_active_blocks=int(num_active_blocks),
            effective_sparsity=effective_sparsity,
            kv_cache_hit=kv_cache_hit,
            kv_cache_store_latency_ms=store_latency_ms,
            kv_cache_retrieve_latency_ms=retrieve_latency_ms,
            attention_latency_ms=attention_latency_ms,
            total_latency_ms=total_latency_ms,
            fallback_used=fallback_used,
            sparse_success=sparse_success,
            dense_success=dense_success,
            audit_enabled=self.audit_mode,
            audit_cosine=audit_cosine,
            audit_rel_mae=audit_rel_mae,
            audit_max_abs_error=audit_max_abs_error,
            termination_reason=termination_reason,
        )
        self._telemetry_log.append(event)

        return attn_output, {
            "task_id": task_id,
            "kv_cache_hit": kv_cache_hit,
            "sparse_success": sparse_success,
            "fallback_used": fallback_used,
            "num_active_blocks": int(num_active_blocks),
            "effective_sparsity": effective_sparsity,
            "total_latency_ms": total_latency_ms,
        }

    def get_telemetry(self) -> list[TelemetryEvent]:
        """Return all recorded telemetry events."""
        return list(self._telemetry_log)

    def clear_telemetry(self) -> None:
        """Clear the telemetry log."""
        self._telemetry_log.clear()
