#!/usr/bin/env python3
"""
RFSN v10 - TurboQuant KV Cache Manager.

Grouped symmetric quantization with randomized-sign Hadamard preconditioning,
packed-dequant-WHT reconstruction path, pinned cache memory budget, and active
cache LRU eviction.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

from .bitpack import BitPackedQuantizer


@dataclass
class TurboQuantKVCache:
    """Container for a single TurboQuant KV cache entry."""

    k_packed: mx.array
    k_scales: mx.array
    v_packed: mx.array
    v_scales: mx.array
    shape: tuple
    k_bits: int
    v_bits: int
    group_size: int
    use_incoherent: bool
    format_version: str
    seed: int = 0
    k_n_values: int = 0
    v_n_values: int = 0
    token_count: int = 0
    pinned: bool = False
    last_used: float = 0.0


class RFSNTurboQuantKVManager:
    """TurboQuant KV cache manager with grouped symmetric quantization."""

    def __init__(
        self,
        k_bits: int = 8,
        v_bits: int = 3,
        use_incoherent: bool = True,
        max_memory_gb: float = 1.0,
        max_pinned_memory_gb: float = 0.5,
        cache_dir: str = ".rfsn_cache",
        group_size: int = 64,
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.use_incoherent = use_incoherent
        self.max_memory_gb = max_memory_gb
        self.max_pinned_memory_gb = max_pinned_memory_gb
        self.cache_dir = Path(cache_dir)
        self.group_size = group_size
        self.active_caches: dict[str, TurboQuantKVCache] = {}
        self._total_estimated_bytes = 0
        self._pinned_bytes = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Randomized sign preconditioning (deterministic, self-inverse)
    # ------------------------------------------------------------------
    def _apply_signs_on_the_fly(self, x: mx.array, seed: int) -> mx.array:
        """Apply deterministic sign preconditioning (self-inverse).

        Sets the global MLX random seed to *seed*, generates a random
        mask of +1/-1, and multiplies element-wise.  Calling twice with
        the same seed restores the original tensor because (+1)^2 = (-1)^2 = 1.
        """
        shape = x.shape
        n = x.size

        # Set seed for deterministic sign generation
        mx.random.seed(seed)
        random_vals = mx.random.uniform(shape=(n,))
        signs = mx.where(
            random_vals < 0.5,
            mx.array(1.0, dtype=x.dtype),
            mx.array(-1.0, dtype=x.dtype),
        )

        return (x * signs.reshape(shape)).astype(x.dtype)

    # ------------------------------------------------------------------
    # Walsh-Hadamard transform (self-inverse when normalised)
    # ------------------------------------------------------------------
    def _apply_wht_pretransform(
        self, x: mx.array, wht_block: int = 64
    ) -> mx.array:
        """Apply Walsh-Hadamard transform along the last dimension.

        The transform is self-inverse when normalised by 1/sqrt(wht_block).

        Raises:
            ValueError: If wht_block != 64 or last dimension is not a
                multiple of 64.
        """
        if wht_block != 64:
            raise ValueError(
                f"wht_block must be exactly 64, got {wht_block}"
            )

        D = x.shape[-1]
        if D % wht_block != 0:
            raise ValueError(
                f"Last dimension must be a multiple of {wht_block}, got {D}"
            )

        shape = x.shape
        # Flatten all leading dims together, split into wht_block-sized chunks
        x = x.reshape(-1, wht_block)
        x = self._wht_block_recursive(x)
        x = x / math.sqrt(wht_block)
        return x.reshape(shape)

    def _wht_block_recursive(self, x: mx.array) -> mx.array:
        """Recursive Walsh-Hadamard transform along last dimension."""
        n = x.shape[-1]
        if n == 1:
            return x

        half = n // 2
        x0 = x[..., :half]
        x1 = x[..., half:]

        y0 = self._wht_block_recursive(x0)
        y1 = self._wht_block_recursive(x1)

        return mx.concatenate([y0 + y1, y0 - y1], axis=-1)

    # ------------------------------------------------------------------
    # Grouped symmetric quantization
    # ------------------------------------------------------------------
    def _quantize(
        self, x: mx.array, bits: int
    ) -> tuple[mx.array, mx.array]:
        """Grouped symmetric quantization.

        Returns (codes, scales) where codes are in [0, 2*qmax] and scales
        are per-group.  Zero always reconstructs as exactly zero.

        Raises:
            ValueError: If input is empty.
        """
        if x.size == 0:
            raise ValueError("Cannot quantize empty tensor")

        original_size = x.size
        group_size = self.group_size
        n_groups = (original_size + group_size - 1) // group_size

        # Pad to multiple of group_size (with zeros so they reconstruct as 0)
        pad_len = (group_size - (original_size % group_size)) % group_size
        if pad_len > 0:
            x = mx.concatenate([x, mx.zeros((pad_len,), dtype=x.dtype)])

        x = x.reshape(n_groups, group_size)

        qmax = (1 << (bits - 1)) - 1
        abs_max = mx.max(mx.abs(x), axis=-1)
        raw_scale = abs_max / float(qmax)
        scales = mx.maximum(raw_scale, mx.array(1e-8, dtype=x.dtype))

        # Quantise: code = round(x / scale) + qmax  →  range [0, 2*qmax]
        codes = mx.round(x / scales.reshape(-1, 1)) + qmax
        codes = mx.clip(codes.astype(mx.int32), 0, 2 * qmax).astype(mx.uint32)

        # Truncate back to the original element count
        codes = codes.reshape(-1)[:original_size]

        return codes, scales

    def _dequantize_unsigned(
        self, q: mx.array, scales: mx.array, bits: int
    ) -> mx.array:
        """Dequantize symmetric codes back to float.

        Raises:
            ValueError: If codes exceed the valid symmetric range or scale
                count does not match the number of groups.
        """
        qmax = (1 << (bits - 1)) - 1
        max_code = 2 * qmax

        # Validate codes
        if mx.any(q > max_code).item():
            raise ValueError(
                f"Invalid symmetric quant code: "
                f"max allowed is {max_code} for {bits} bits"
            )

        original_size = q.size
        n_groups = (original_size + self.group_size - 1) // self.group_size

        # Validate scale count
        if scales.size != n_groups:
            raise ValueError(
                f"Scale count mismatch: expected {n_groups}, got {scales.size}"
            )

        # Pad q to multiple of group_size
        pad_len = (self.group_size - (original_size % self.group_size)) % self.group_size
        if pad_len > 0:
            q = mx.concatenate([q, mx.zeros((pad_len,), dtype=mx.uint32)])

        q_f = q.reshape(n_groups, self.group_size).astype(mx.float32)
        q_f = q_f - qmax  # shift back to [-qmax, qmax]

        x = q_f * scales.reshape(-1, 1)

        return x.reshape(-1)[:original_size]

    # ------------------------------------------------------------------
    # Packed-dequant-WHT reconstruction path
    # ------------------------------------------------------------------
    def _reconstruct_packed_dequant_wht(
        self,
        packed: mx.array,
        scales: mx.array,
        n_values: int,
        shape: tuple,
        bits: int,
        seed: int,
        use_incoherent: bool,
        out_dtype: mx.Dtype,
    ) -> mx.array:
        """Packed-dequant-WHT reconstruction path: unpack → dequant → reshape → WHT → optional signs.

        Raises:
            ValueError: If any validation fails (out_dtype, shape product,
                packed buffer size, scale count).
        """
        # Validate out_dtype
        if out_dtype not in (mx.float32, mx.float16):
            raise ValueError(
                f"out_dtype must be float32 or float16, got {out_dtype}"
            )

        # Validate shape product
        shape_product = math.prod(shape)
        if shape_product != n_values:
            raise ValueError(
                f"Shape product {shape_product} does not match n_values {n_values}"
            )

        # Validate packed size
        codes_per_word = 32 // bits
        required_words = (n_values + codes_per_word - 1) // codes_per_word
        if packed.size < required_words:
            raise ValueError("Packed buffer too small")

        # Validate scale count
        n_groups = (n_values + self.group_size - 1) // self.group_size
        if scales.size != n_groups:
            raise ValueError("Scale count mismatch")

        # Unpack
        codes = BitPackedQuantizer.unpack(packed, n_values, bits)

        # Dequantize
        x = self._dequantize_unsigned(codes, scales, bits)

        # Reshape
        x = x.reshape(shape)

        # Apply WHT
        x = self._apply_wht_pretransform(x)

        # Apply signs (if use_incoherent)
        if use_incoherent:
            x = self._apply_signs_on_the_fly(x, seed)

        return x.astype(out_dtype)

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------
    def _estimate_cache_bytes(self, cache: TurboQuantKVCache) -> int:
        """Estimate the memory usage of a cache entry in bytes."""
        return (
            cache.k_packed.size * 4  # uint32
            + cache.k_scales.size * 4  # float32
            + cache.v_packed.size * 4  # uint32
            + cache.v_scales.size * 4  # float32
        )

    def _enforce_memory_budget(self) -> None:
        """Evict least-recently-used unpinned caches until budget is met."""
        while self._total_estimated_bytes > self.max_memory_gb * (1024 ** 3):
            # Find LRU unpinned cache
            lru_key = min(
                (
                    k
                    for k, v in self.active_caches.items()
                    if not v.pinned
                ),
                key=lambda k: self.active_caches[k].last_used,
                default=None,
            )
            if lru_key is None:
                raise MemoryError(
                    "All caches pinned, cannot evict to meet budget"
                )
            old_cache = self.active_caches.pop(lru_key)
            self._total_estimated_bytes -= self._estimate_cache_bytes(old_cache)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------
    def store(
        self,
        skill_pattern: str,
        keys: mx.array,
        values: mx.array,
        token_count: int,
    ) -> None:
        """Store KV cache with TurboQuant compression.

        Raises:
            ValueError: If shapes are invalid or mismatched.
            MemoryError: If the cache exceeds max_memory_gb.
        """
        # Validate shapes
        if len(keys.shape) != 4:
            raise ValueError(
                f"Expected KV shape [B,H,T,D], got {keys.shape}"
            )
        if keys.shape != values.shape:
            raise ValueError(
                f"keys/values shape mismatch: {keys.shape} vs {values.shape}"
            )
        if keys.shape[-1] % 64 != 0:
            raise ValueError(
                f"Head dimension must be a multiple of 64, got {keys.shape[-1]}"
            )
        if token_count <= 0:
            raise ValueError(
                f"token_count must be positive, got {token_count}"
            )

        # Flatten for quantization
        k_flat = keys.reshape(-1)
        v_flat = values.reshape(-1)

        # Deterministic seed derived from cache identity for sign preconditioning
        seed = int(
            hashlib.sha256(
                f"{skill_pattern}|{keys.shape}|{self.k_bits}|{self.v_bits}".encode()
            ).hexdigest()[:8],
            16,
        )

        # When use_incoherent, apply signs → WHT before quantization
        # so that retrieve's WHT → signs cancels them out (both self-inverse)
        if self.use_incoherent:
            k_flat = self._apply_signs_on_the_fly(k_flat, seed)
            k_flat = self._apply_wht_pretransform(k_flat)
            v_flat = self._apply_signs_on_the_fly(v_flat, seed)
            v_flat = self._apply_wht_pretransform(v_flat)

        # Quantize
        k_codes, k_scales = self._quantize(k_flat, self.k_bits)
        v_codes, v_scales = self._quantize(v_flat, self.v_bits)

        # Pack
        k_packed, k_n = BitPackedQuantizer.pack(k_codes, self.k_bits)
        v_packed, v_n = BitPackedQuantizer.pack(v_codes, self.v_bits)

        # Create cache entry
        cache = TurboQuantKVCache(
            k_packed=k_packed,
            k_scales=k_scales,
            v_packed=v_packed,
            v_scales=v_scales,
            shape=tuple(keys.shape),
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            group_size=self.group_size,
            use_incoherent=self.use_incoherent,
            format_version="rfsn_v10",
            seed=seed,
            k_n_values=k_n,
            v_n_values=v_n,
            token_count=token_count,
        )

        # Check memory budget
        cache_bytes = self._estimate_cache_bytes(cache)
        if cache_bytes > self.max_memory_gb * (1024**3):
            raise MemoryError(
                f"Cache size {cache_bytes} bytes exceeds "
                f"max_memory_gb={self.max_memory_gb}"
            )

        # If replacing an existing entry, subtract its old size
        if skill_pattern in self.active_caches:
            old_cache = self.active_caches[skill_pattern]
            old_bytes = self._estimate_cache_bytes(old_cache)
            self._total_estimated_bytes -= old_bytes
            if old_cache.pinned:
                self._pinned_bytes -= old_bytes

        self.active_caches[skill_pattern] = cache
        self._total_estimated_bytes += cache_bytes

        # Enforce memory budget with LRU eviction
        self._enforce_memory_budget()

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self,
        skill_pattern: str,
        out_dtype: mx.Dtype = mx.float32,
    ) -> Optional[tuple[mx.array, mx.array]]:
        """Retrieve and dequantize KV cache.

        Returns None if the cache key is not found.

        Raises:
            ValueError: If format version or metadata mismatches are detected.
        """
        if skill_pattern not in self.active_caches:
            return None

        cache = self.active_caches[skill_pattern]

        # Update last-used timestamp for LRU tracking
        cache.last_used = time.monotonic()

        # Validate format version
        if cache.format_version != "rfsn_v10":
            raise ValueError(
                f"Unsupported cache format version: {cache.format_version}"
            )

        # Validate metadata matches current manager settings
        if cache.k_bits != self.k_bits or cache.v_bits != self.v_bits:
            raise ValueError(
                f"metadata mismatch: stored k_bits={cache.k_bits} "
                f"v_bits={cache.v_bits}, current k_bits={self.k_bits} "
                f"v_bits={self.v_bits}"
            )

        # Dequantize K
        k_codes = BitPackedQuantizer.unpack(
            cache.k_packed, cache.k_n_values, cache.k_bits
        )
        k_rec = self._dequantize_unsigned(
            k_codes, cache.k_scales, cache.k_bits
        )
        k_rec = k_rec.reshape(cache.shape)

        if cache.use_incoherent:
            k_rec = self._apply_wht_pretransform(k_rec)
            k_rec = self._apply_signs_on_the_fly(k_rec, cache.seed)

        # Dequantize V
        v_codes = BitPackedQuantizer.unpack(
            cache.v_packed, cache.v_n_values, cache.v_bits
        )
        v_rec = self._dequantize_unsigned(
            v_codes, cache.v_scales, cache.v_bits
        )
        v_rec = v_rec.reshape(cache.shape)

        if cache.use_incoherent:
            v_rec = self._apply_wht_pretransform(v_rec)
            v_rec = self._apply_signs_on_the_fly(v_rec, cache.seed)

        return k_rec.astype(out_dtype), v_rec.astype(out_dtype)

    # ------------------------------------------------------------------
    # Pin cache (budget enforcement)
    # ------------------------------------------------------------------
    def pin_cache(self, skill_pattern: str) -> bool:
        """Pin a cache entry so it cannot be evicted.

        Returns:
            True if successfully pinned, False if the cache does not exist.

        Raises:
            MemoryError: If pinning would exceed max_pinned_memory_gb.
        """
        if skill_pattern not in self.active_caches:
            return False

        cache = self.active_caches[skill_pattern]
        if cache.pinned:
            return True

        cache_bytes = self._estimate_cache_bytes(cache)
        if (
            self._pinned_bytes + cache_bytes
            > self.max_pinned_memory_gb * (1024**3)
        ):
            raise MemoryError(
                f"Pin would exceed pinned budget: "
                f"{self._pinned_bytes + cache_bytes} > "
                f"{self.max_pinned_memory_gb * (1024**3)}"
            )

        cache.pinned = True
        self._pinned_bytes += cache_bytes
        return True

    # Backward compatibility alias
    pincache = pin_cache
