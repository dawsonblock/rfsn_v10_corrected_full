#!/usr/bin/env python3
"""
RFSN v10 - Hardware-Aware Block-Sparse Attention.

Decode-only block-sparse attention using physically compacted KV blocks.

Correct use case:
- Decode path only: queries shape [B, H, 1, D]
- KV cache contains only past tokens
- Positional information is already baked into keys, e.g. RoPE-applied keys
- Prefill uses dense attention because physical compaction breaks causal alignment

NOTE: The dense fallback path uses mx.fast.scaled_dot_product_attention without
a causal mask. Callers handling autoregressive prefill must supply their own
causal masking. This module is designed for decode, not prefill.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import mlx.core as mx

ExecutionMode = Literal[
    "sparse_compacted",
    "dense_requested",
    "dense_short_context",
    "dense_prefill",
    "dense_not_strictly_past",
]


class AdaptiveBlockSparseAttention:
    """Block-sparse attention with block selection and compacted KV dispatch."""

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        if b <= 0:
            raise ValueError(f"divisor must be positive, got {b}")
        return (a + b - 1) // b

    @staticmethod
    def _validate_inputs(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: float,
        block_size: int,
        consensus_mix: float,
    ) -> Tuple[int, int, int, int, int]:
        if len(queries.shape) != 4:
            raise ValueError(f"queries must be [B,H,T_q,D], got {queries.shape}")
        if len(keys.shape) != 4:
            raise ValueError(f"keys must be [B,H,T_k,D], got {keys.shape}")
        if len(values.shape) != 4:
            raise ValueError(f"values must be [B,H,T_k,D], got {values.shape}")
        if keys.shape != values.shape:
            raise ValueError(f"keys/values shape mismatch: {keys.shape} vs {values.shape}")

        B, H, T_k, D = keys.shape
        Bq, Hq, T_q, Dq = queries.shape

        if Bq != B:
            raise ValueError(f"batch mismatch: queries B={Bq}, keys B={B}")
        if Hq != H:
            raise ValueError(f"head mismatch: queries H={Hq}, keys H={H}")
        if Dq != D:
            raise ValueError(f"head_dim mismatch: queries D={Dq}, keys D={D}")
        if T_q <= 0 or T_k <= 0:
            raise ValueError(f"T_q and T_k must be positive, got T_q={T_q}, T_k={T_k}")
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not (0.0 < float(top_k_ratio) <= 1.0):
            raise ValueError(f"top_k_ratio must be in (0, 1], got {top_k_ratio}")
        if not math.isfinite(float(consensus_mix)):
            raise ValueError(f"consensus_mix must be finite, got {consensus_mix}")

        return B, H, T_q, T_k, D

    @staticmethod
    def _dense_unmasked(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: float,
        block_size: int,
        mode: str,
    ) -> Tuple[mx.array, int, str]:
        T_k = keys.shape[2]
        num_blocks = max(1, AdaptiveBlockSparseAttention._ceil_div(T_k, block_size))
        out = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
        )
        return out, num_blocks, mode

    @staticmethod
    def execute(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        top_k_ratio: float,
        block_size: int = 64,
        kv_is_strictly_past: bool = True,
        consensus_mix: float = 0.7,
    ) -> Tuple[mx.array, int, ExecutionMode]:
        """
        Execute hardware-aware block-sparse scaled dot-product attention.

        Args:
            queries: Query tensor [B, H, T_q, D].
            keys: Key tensor [B, H, T_k, D].
            values: Value tensor [B, H, T_k, D].
            top_k_ratio: Fraction of KV blocks to retain. Must be in (0, 1].
            block_size: KV block size. Should match TurboQuant group_size.
            kv_is_strictly_past: True when all KV tokens are valid past context.
                If false, dense fallback is used.
            consensus_mix: Blend between max-head recall and mean-head consensus.
                1.0 = pure max across heads; 0.0 = pure mean across heads.

        Returns:
            (attention_output, num_active_blocks, execution_mode)
            execution_mode is one of:
              - "sparse_compacted": actual sparse block-selective attention ran
              - "dense_requested": top_k_ratio >= 1.0
              - "dense_short_context": T_k <= block_size
              - "dense_prefill": T_q > 1 (causal mask not applied by this module)
              - "dense_not_strictly_past": kv_is_strictly_past is False
        """
        B, H, T_q, T_k, D = AdaptiveBlockSparseAttention._validate_inputs(
            queries,
            keys,
            values,
            top_k_ratio,
            block_size,
            consensus_mix,
        )

        scale = 1.0 / math.sqrt(D)

        # Dense fallback cases:
        # - User requested dense behavior
        # - Context too short to benefit
        # - Prefill path: physical compaction breaks causal mask alignment
        # - Caller cannot guarantee KV contains only past tokens
        if top_k_ratio >= 1.0:
            return AdaptiveBlockSparseAttention._dense_unmasked(
                queries, keys, values, scale, block_size, "dense_requested",
            )
        if T_k <= block_size:
            return AdaptiveBlockSparseAttention._dense_unmasked(
                queries, keys, values, scale, block_size, "dense_short_context",
            )
        if T_q > 1:
            return AdaptiveBlockSparseAttention._dense_unmasked(
                queries, keys, values, scale, block_size, "dense_prefill",
            )
        if not kv_is_strictly_past:
            return AdaptiveBlockSparseAttention._dense_unmasked(
                queries, keys, values, scale, block_size, "dense_not_strictly_past",
            )

        pad_len = (block_size - (T_k % block_size)) % block_size
        if pad_len > 0:
            k_pad = mx.concatenate(
                [keys, mx.zeros((B, H, pad_len, D), dtype=keys.dtype)],
                axis=2,
            )
            v_pad = mx.concatenate(
                [values, mx.zeros((B, H, pad_len, D), dtype=values.dtype)],
                axis=2,
            )
            T_k_padded = T_k + pad_len
        else:
            k_pad = keys
            v_pad = values
            T_k_padded = T_k

        num_blocks = T_k_padded // block_size
        k_active = max(1, int(math.ceil(num_blocks * float(top_k_ratio))))

        if k_active >= num_blocks:
            return AdaptiveBlockSparseAttention._dense_unmasked(
                queries, keys, values, scale, block_size, "dense_short_context",
            )

        # Mean pooling is less sign-biased than max pooling because max pooling
        # discards strong negative features.
        k_reshaped = k_pad.reshape(B, H, num_blocks, block_size, D)
        k_pooled = mx.mean(k_reshaped, axis=3)  # [B, H, num_blocks, D]

        # T_q == 1 here. Shape: [B, H, 1, num_blocks]
        scores_per_head = (queries @ k_pooled.transpose(0, 1, 3, 2)) * scale

        max_score = mx.max(scores_per_head, axis=1)    # [B, 1, num_blocks]
        mean_score = mx.mean(scores_per_head, axis=1)  # [B, 1, num_blocks]

        mix = max(0.0, min(1.0, float(consensus_mix)))
        global_block_scores = mix * max_score + (1.0 - mix) * mean_score

        kth = num_blocks - k_active
        unordered_topk_idx = mx.argpartition(global_block_scores, kth, axis=-1)[..., kth:]
        topk_block_idx = mx.sort(unordered_topk_idx, axis=-1)  # chronological order

        offsets = mx.arange(block_size, dtype=mx.uint32)
        base_indices = topk_block_idx.reshape(B, k_active, 1).astype(mx.uint32) * block_size
        token_indices = (base_indices + offsets).reshape(B, -1)

        # Padding-safe compaction:
        # - For B == 1, remove padded positions directly.
        # - For B > 1, padding would create ragged compact tensors, so fallback dense.
        if pad_len > 0 and B > 1:
            return AdaptiveBlockSparseAttention._dense_unmasked(
                queries, keys, values, scale, block_size, "dense_requested",
            )

        if B == 1:
            idx = token_indices[0]
            valid = idx < T_k
            # MLX lacks argwhere and boolean indexing; use where + sort
            n_valid = int(mx.sum(valid.astype(mx.int32)).item())
            sentinel = mx.array(T_k, dtype=idx.dtype)
            idx_safe = mx.where(valid, idx, sentinel)
            idx_sorted = mx.sort(idx_safe)
            idx = idx_sorted[:n_valid]
            active_tokens = n_valid
            keys_compact = k_pad[:, :, idx, :]
            values_compact = v_pad[:, :, idx, :]
            active_blocks = max(1, AdaptiveBlockSparseAttention._ceil_div(active_tokens, block_size))
        else:
            compact_keys_list = []
            compact_values_list = []
            for b in range(B):
                idx = token_indices[b]
                compact_keys_list.append(k_pad[b:b + 1, :, idx, :])
                compact_values_list.append(v_pad[b:b + 1, :, idx, :])

            keys_compact = mx.concatenate(compact_keys_list, axis=0)
            values_compact = mx.concatenate(compact_values_list, axis=0)
            active_blocks = k_active

        out = mx.fast.scaled_dot_product_attention(
            queries,
            keys_compact,
            values_compact,
            scale=scale,
        )

        return out, active_blocks, "sparse_compacted"
