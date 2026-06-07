"""RFSN v10 — Quantized Metal attention scoring mode tests."""

from __future__ import annotations

import pytest

from rfsn_v10.kernels import maybe_supports_metal_kernels
from rfsn_v10.runtime.scoring_modes import score_attention_quantized_metal

mx = pytest.importorskip("mlx.core")


class TestQuantizedMetalScoring:
    """Tests for the fused quantized Metal attention scoring path."""

    def test_importable(self):
        assert callable(score_attention_quantized_metal)

    def test_raises_without_metal(self):
        """If Metal kernels are unavailable the function must raise."""
        if maybe_supports_metal_kernels():
            pytest.skip("Metal kernels are available on this platform")

        queries = mx.zeros((1, 4, 1, 64))
        packed_k = mx.zeros((4, 16), dtype=mx.uint32)
        packed_v = mx.zeros((4, 16), dtype=mx.uint32)
        scales_k = mx.ones((4, 16))
        scales_v = mx.ones((4, 16))

        with pytest.raises(Exception):
            score_attention_quantized_metal(
                queries,
                packed_k=packed_k,
                packed_v=packed_v,
                scales_k=scales_k,
                scales_v=scales_v,
                n_keys=64,
                bits=4,
            )

    @pytest.mark.skipif(
        not maybe_supports_metal_kernels(),
        reason="Metal kernels not available",
    )
    def test_smoke_with_metal(self):
        """Basic smoke test when Metal is present."""
        queries = mx.zeros((1, 4, 1, 64))
        # For 64 keys, 4 bits -> codes_per_word=8 -> 8 words per head
        packed_k = mx.zeros((4, 8), dtype=mx.uint32)
        packed_v = mx.zeros((4, 8), dtype=mx.uint32)
        # group_size=64, n_keys=64 -> 1 group per head
        scales_k = mx.ones((4, 1))
        scales_v = mx.ones((4, 1))

        out = score_attention_quantized_metal(
            queries,
            packed_k=packed_k,
            packed_v=packed_v,
            scales_k=scales_k,
            scales_v=scales_v,
            n_keys=64,
            bits=4,
            group_size=64,
        )
        assert out.shape == (1, 4, 1, 64)
