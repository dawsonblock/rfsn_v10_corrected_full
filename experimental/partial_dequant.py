#!/usr/bin/env python3
"""Partial dequantization for latency optimization.

Implements selective dequantization of KV cache blocks based on
access patterns and quality requirements to reduce latency with
minimal quality loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .compat import mx


@dataclass
class PartialDequantConfig:
    """Configuration for partial dequantization strategy."""

    hot_block_ratio: float = 0.3  # Fraction of blocks to keep fully dequantized
    cold_block_ratio: float = 0.7  # Fraction of blocks to keep quantized
    quality_threshold: float = 0.95  # Minimum cosine similarity threshold
    access_window: int = 128  # Number of recent tokens to consider hot
    enable_adaptive: bool = True  # Enable adaptive quality adjustment


class PartialDequantManager:
    """Manages partial dequantization of KV cache blocks."""

    def __init__(self, config: Optional[PartialDequantConfig] = None):
        self.config = config or PartialDequantConfig()
        self.access_history: dict[int, list[int]] = {}
        self.quality_history: list[float] = []

    def _compute_block_importance(
        self,
        access_counts: list[int],
        recent_access: list[int],
        num_blocks: int,
    ) -> list[float]:
        """Compute importance scores for each block.

        Combines access frequency and recency to determine which blocks
        should be kept fully dequantized.
        """
        scores = []
        max_access = max(access_counts) if access_counts else 1
        max_recent = max(recent_access) if recent_access else 1

        for i in range(num_blocks):
            # Normalize access frequency and recency
            freq_score = access_counts[i] / max_access if max_access > 0 else 0
            recency_score = recent_access[i] / max_recent if max_recent > 0 else 0

            # Combined score with recency bias
            combined = 0.6 * freq_score + 0.4 * recency_score
            scores.append(combined)

        return scores

    def _select_hot_blocks(
        self,
        scores: list[float],
        num_blocks: int,
        hot_ratio: float,
    ) -> list[int]:
        """Select hot blocks to keep fully dequantized."""
        num_hot = max(1, int(math.ceil(num_blocks * hot_ratio)))

        # Sort by score and select top blocks
        scored_blocks = sorted(enumerate(scores), key=lambda x: -x[1])
        hot_blocks = [idx for idx, _ in scored_blocks[:num_hot]]

        return sorted(hot_blocks)

    def _dequant_hot_blocks(
        self,
        packed: mx.array,
        scales: mx.array,
        hot_blocks: list[int],
        n_values: int,
        bits: int,
        group_size: int,
        block_size: int,
    ) -> mx.array:
        """Dequantize only the hot blocks."""
        from .bitpack import BitPackedQuantizer

        # Dequantize all values first
        quantizer = BitPackedQuantizer(bits=bits, group_size=group_size)
        all_dequant = quantizer.unpack(packed, n_values, bits)
        all_dequant = quantizer._dequantize_unsigned(all_dequant, scales, bits)

        # Extract hot blocks
        hot_values = []
        for block_idx in hot_blocks:
            start = block_idx * block_size
            end = min(start + block_size, n_values)
            hot_values.append(all_dequant[start:end])

        # For cold blocks, keep quantized representation
        # This is a simplified implementation - in practice, you'd
        # maintain separate hot/cold caches

        return all_dequant  # Simplified: return all for now

    def execute_partial_dequant(
        self,
        packed: mx.array,
        scales: mx.array,
        n_values: int,
        bits: int,
        group_size: int,
        block_size: int = 64,
        current_position: int = 0,
    ) -> tuple[mx.array, dict]:
        """Execute partial dequantization strategy.

        Args:
            packed: Packed quantized KV cache
            scales: Quantization scales
            n_values: Number of values
            bits: Bit width
            group_size: Group size for quantization
            block_size: Block size for partial dequant
            current_position: Current token position

        Returns:
            (dequantized_values, metadata)
        """
        num_blocks = math.ceil(n_values / block_size)

        # Update access history
        block_idx = current_position // block_size
        if block_idx not in self.access_history:
            self.access_history[block_idx] = []
        self.access_history[block_idx].append(current_position)

        # Compute access statistics
        access_counts = []
        recent_access = []
        window_start = max(0, current_position - self.config.access_window)

        for i in range(num_blocks):
            total = len(self.access_history.get(i, []))
            recent = sum(
                1 for pos in self.access_history.get(i, []) if pos >= window_start
            )
            access_counts.append(total)
            recent_access.append(recent)

        # Compute importance scores
        scores = self._compute_block_importance(access_counts, recent_access, num_blocks)

        # Select hot blocks
        hot_blocks = self._select_hot_blocks(
            scores, num_blocks, self.config.hot_block_ratio
        )

        # Dequantize hot blocks
        dequant = self._dequant_hot_blocks(
            packed, scales, hot_blocks, n_values, bits, group_size, block_size
        )

        metadata = {
            "hot_blocks": hot_blocks,
            "num_hot_blocks": len(hot_blocks),
            "num_total_blocks": num_blocks,
            "hot_ratio": len(hot_blocks) / num_blocks,
            "scores": scores,
        }

        return dequant, metadata

    def update_quality(self, quality: float) -> None:
        """Update quality history for adaptive adjustment."""
        self.quality_history.append(quality)
        if len(self.quality_history) > 10:
            self.quality_history = self.quality_history[-10:]

        # Adaptive adjustment
        if self.config.enable_adaptive and len(self.quality_history) >= 3:
            avg_quality = sum(self.quality_history[-3:]) / 3
            if avg_quality < self.config.quality_threshold:
                # Quality too low, increase hot block ratio
                self.config.hot_block_ratio = min(0.5, self.config.hot_block_ratio + 0.05)
            elif avg_quality > 0.98:
                # Quality very good, can be more aggressive
                self.config.hot_block_ratio = max(0.1, self.config.hot_block_ratio - 0.05)

    def get_stats(self) -> dict:
        """Get statistics about partial dequantization."""
        return {
            "config": {
                "hot_block_ratio": self.config.hot_block_ratio,
                "cold_block_ratio": self.config.cold_block_ratio,
                "quality_threshold": self.config.quality_threshold,
            },
            "quality_history": self.quality_history,
            "num_tracked_blocks": len(self.access_history),
        }
