"""RFSN v10 — Prepared vs packed scoring modes.

Provides five quantization-aware attention scoring paths:

* ``fp16``               — baseline without quantization.
* ``reconstructed``      — fully unpack / dequant the entire KV cache before scoring.
* ``prepared``           — reuse pre-computed dequantized (or transformed) cache blocks.
* ``packed_block``       — unpack / dequant only the blocks selected for attention.
* ``score_corrected``    — future QJL or residual score correction (stub).
"""

from __future__ import annotations

import math
from typing import Any, Callable

# Optional MLX with pytest.importorskip fallback pattern
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    try:
        import pytest

        mx = pytest.importorskip("mlx.core")
    except Exception:

        class _MissingMLX:
            def __getattr__(self, name: str) -> Any:
                raise AttributeError(
                    f"mlx.core is not installed; attribute '{name}' unavailable"
                )

        mx = _MissingMLX()  # type: ignore[misc,assignment]


def score_attention_fp16(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float | None = None,
) -> mx.array:
    """Standard FP16 baseline attention (no quantization).

    Args:
        queries: [B, H, T_q, D]
        keys:    [B, H, T_k, D]
        values:  [B, H, T_k, D]
        scale:   Optional manual scale; defaults to ``1 / sqrt(D)``.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(queries.shape[-1])
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale
    )


def score_attention_reconstructed(
    queries: mx.array,
    keys_packet: Any,
    values_packet: Any,
    dequant_fn: Callable[[Any], mx.array],
    scale: float | None = None,
) -> mx.array:
    """Full dequantization then score.

    Args:
        queries:      [B, H, T_q, D]
        keys_packet:  opaque packed key container.
        values_packet: opaque packed value container.
        dequant_fn:   callable that takes ``(keys_packet, values_packet)`` and
                      returns a tuple ``(keys, values)`` of shape ``[B, H, T_k, D]``.
        scale:        Optional manual attention scale.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    keys, values = dequant_fn(keys_packet, values_packet)
    return score_attention_fp16(queries, keys, values, scale=scale)


def score_attention_prepared(
    queries: mx.array,
    prepared_k: mx.array,
    prepared_v: mx.array,
    scale: float | None = None,
) -> mx.array:
    """Score using pre-computed dequantized or transformed cache blocks.

    Args:
        queries:    [B, H, T_q, D]
        prepared_k: [B, H, T_k, D]  (already dequantized / transformed)
        prepared_v: [B, H, T_k, D]  (already dequantized / transformed)
        scale:      Optional manual attention scale.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    return score_attention_fp16(queries, prepared_k, prepared_v, scale=scale)


def score_attention_packed_block(
    queries: mx.array,
    keys_packet: Any,
    values_packet: Any,
    block_indices: mx.array | list[int],
    block_dequant_fn: Callable[[Any, mx.array | list[int]], tuple[mx.array, mx.array]],
    scale: float | None = None,
) -> mx.array:
    """Unpack / dequant only the selected blocks needed for attention.

    Args:
        queries:          [B, H, T_q, D]
        keys_packet:      opaque packed key container.
        values_packet:    opaque packed value container.
        block_indices:    1-D array or list of integer block indices to retrieve.
        block_dequant_fn: callable ``(packet, block_indices) -> (keys, values)``
                          where the returned tensors have shape ``[B, H, len(block_indices)*Bsz, D]``.
        scale:            Optional manual attention scale.

    Returns:
        Attention output of shape [B, H, T_q, D].
    """
    keys, values = block_dequant_fn(keys_packet, values_packet, block_indices)
    return score_attention_fp16(queries, keys, values, scale=scale)


def score_attention_score_corrected(
    queries: mx.array,
    keys_packet: Any,
    values_packet: Any,
    correction_fn: Callable[..., mx.array] | None = None,
    **kwargs: Any,
) -> mx.array:
    """Future QJL or residual score correction path (stub only).

    Args:
        queries:       [B, H, T_q, D]
        keys_packet:   opaque packed key container.
        values_packet: opaque packed value container.
        correction_fn: reserved for future residual correction logic.
        **kwargs:      reserved for future keyword arguments.

    Raises:
        NotImplementedError: Always raised because this path is disabled until
                             the QJL benchmark passes.
    """
    raise NotImplementedError(
        "score_corrected disabled until QJL benchmark passes"
    )
