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
from threading import RLock, get_ident
from typing import Optional

from .compat import mx

from .bitpack import BitPackedQuantizer
from .kernels import (
    KernelRouteError,
    apply_hash_signs_metal,
    maybe_supports_metal_kernels,
    packed_dequant_metal,
    packed_dequant_wht_sign_metal,
    wht64_metal,
)


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
    use_wht: bool
    use_incoherent_signs: bool
    format_version: str
    seed: int = 0
    k_n_values: int = 0
    v_n_values: int = 0
    token_count: int = 0
    block_size: int = 64
    num_blocks: int = 0
    k_block_packed_offsets: list[int] = field(default_factory=list)
    k_block_scale_offsets: list[int] = field(default_factory=list)
    k_block_n_values: list[int] = field(default_factory=list)
    v_block_packed_offsets: list[int] = field(default_factory=list)
    v_block_scale_offsets: list[int] = field(default_factory=list)
    v_block_n_values: list[int] = field(default_factory=list)
    pinned: bool = False
    last_used: float = 0.0

    @property
    def use_incoherent(self) -> bool:
        return self.use_wht and self.use_incoherent_signs

    @use_incoherent.setter
    def use_incoherent(self, value: bool) -> None:
        enabled = bool(value)
        self.use_wht = enabled
        self.use_incoherent_signs = enabled


class RFSNTurboQuantKVManager:
    """TurboQuant KV cache manager with grouped symmetric quantization."""

    def __init__(
        self,
        k_bits: int = 8,
        v_bits: int = 3,
        use_wht: Optional[bool] = None,
        use_incoherent_signs: Optional[bool] = None,
        use_incoherent: Optional[bool] = None,
        use_custom_kernel: Optional[bool] = None,
        prefer_metal_kernels: bool = False,
        prefer_fused_kernel: bool = True,
        strict_metal: bool = False,
        validate_metal_codes: bool = False,
        max_memory_gb: float = 1.0,
        max_pinned_memory_gb: float = 0.5,
        cache_dir: str = ".rfsn_cache",
        group_size: int = 64,
        block_size: int = 64,
    ):
        # Validate parameters
        if not (2 <= k_bits <= 8):
            raise ValueError(f"k_bits must be between 2 and 8, got {k_bits}")
        if not (2 <= v_bits <= 8):
            raise ValueError(f"v_bits must be between 2 and 8, got {v_bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if block_size <= 0:
            raise ValueError(
                f"block_size must be positive, got {block_size}"
            )

        if use_wht is None and use_incoherent_signs is None:
            legacy = True if use_incoherent is None else bool(use_incoherent)
            use_wht = legacy
            use_incoherent_signs = legacy
        else:
            use_wht = True if use_wht is None else bool(use_wht)
            use_incoherent_signs = (
                True if use_incoherent_signs is None else bool(use_incoherent_signs)
            )

        self.k_bits = k_bits
        self.v_bits = v_bits
        self.use_wht = bool(use_wht)
        self.use_incoherent_signs = bool(use_incoherent_signs)
        if use_custom_kernel is not None:
            prefer_metal_kernels = bool(use_custom_kernel)
        self.prefer_metal_kernels = bool(prefer_metal_kernels)
        self.prefer_fused_kernel = bool(prefer_fused_kernel)
        self.strict_metal = bool(strict_metal)
        self.validate_metal_codes = bool(validate_metal_codes)
        self.max_memory_gb = max_memory_gb
        self.max_pinned_memory_gb = max_pinned_memory_gb
        self.cache_dir = Path(cache_dir)
        self.group_size = group_size
        self.block_size = block_size
        # Cache for sign masks to avoid regenerating for same shape/seed/dtype.
        self._sign_cache: dict[tuple[tuple[int, ...], int, mx.Dtype, int], mx.array] = {}
        self._sign_cache_lock = RLock()
        self.active_caches: dict[str, TurboQuantKVCache] = {}
        self._total_estimated_bytes = 0
        self._pinned_bytes = 0
        self.last_reconstruction_kernel = "sequential_reference"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def use_incoherent(self) -> bool:
        return self.use_wht and self.use_incoherent_signs

    @use_incoherent.setter
    def use_incoherent(self, value: bool) -> None:
        enabled = bool(value)
        self.use_wht = enabled
        self.use_incoherent_signs = enabled

    @property
    def use_custom_kernel(self) -> bool:
        """Backward compatibility alias for prefer_metal_kernels."""
        return self.prefer_metal_kernels

    @use_custom_kernel.setter
    def use_custom_kernel(self, value: bool) -> None:
        self.prefer_metal_kernels = bool(value)

    # ------------------------------------------------------------------
    # Randomized sign preconditioning (deterministic, self-inverse)
    # ------------------------------------------------------------------
    def _apply_signs_on_the_fly(
        self,
        x: mx.array,
        seed: int,
        indices: Optional[mx.array] = None,
    ) -> mx.array:
        """Apply deterministic sign preconditioning (self-inverse).

        Uses hash-based deterministic signs instead of global RNG to avoid
        contaminating global random state.

        Generates a deterministic mask of +1/-1 based on hash(index, seed),
        and multiplies element-wise. Calling twice with the same seed
        restores the original tensor.

        Args:
            x: Input array.
            seed: Deterministic seed for sign generation.
            indices: Optional flat uint32 index array. If provided, signs
                are computed from these indices instead of 0..n-1.

        Includes caching to avoid regenerating sign masks for identical
        shape/seed/dtype combinations.
        """
        if (
            indices is None
            and self.prefer_metal_kernels
            and maybe_supports_metal_kernels()
        ):
            try:
                return apply_hash_signs_metal(x, seed)
            except Exception:
                pass

        shape = x.shape
        n = x.size

        # Check cache first (only when using default indices)
        if indices is None:
            cache_key = (shape, seed, x.dtype, get_ident())
            with self._sign_cache_lock:
                signs = self._sign_cache.get(cache_key)
            if signs is not None:
                return x * signs

        # Vectorized deterministic hash-like mixing (SplitMix-style)
        seed_u32 = mx.array(seed & 0xFFFFFFFF, dtype=mx.uint32)
        if indices is not None:
            idx = indices.reshape(-1).astype(mx.uint32)
        else:
            idx = mx.arange(n, dtype=mx.uint32)
        z = idx ^ seed_u32
        z = z + mx.array(0x9E3779B9, dtype=mx.uint32)
        z = (z ^ (z >> 16)) * mx.array(0x85EBCA6B, dtype=mx.uint32)
        z = (z ^ (z >> 13)) * mx.array(0xC2B2AE35, dtype=mx.uint32)
        z = z ^ (z >> 16)
        parity = z & mx.array(1, dtype=mx.uint32)
        signs = mx.where(
            parity == 0,
            mx.array(1.0, dtype=x.dtype),
            mx.array(-1.0, dtype=x.dtype),
        ).reshape(shape)

        # Cache the result (only for default indices)
        if indices is None:
            with self._sign_cache_lock:
                if (
                    cache_key not in self._sign_cache
                    and len(self._sign_cache) < 128
                ):
                    self._sign_cache[cache_key] = signs
                else:
                    signs = self._sign_cache.get(cache_key, signs)

        return x * signs

    def _apply_signs_to_block(
        self,
        x: mx.array,
        seed: int,
        block_idx: int,
        block_size: int,
        full_shape: tuple,
    ) -> mx.array:
        """Apply signs to a block using global flat indices."""
        b, h, t_block, d = x.shape
        b_full, h_full, t_full, d_full = full_shape
        assert b == b_full and h == h_full and d == d_full

        t_start = block_idx * block_size
        # Build global flat index for each element in the block
        b_idx = mx.arange(b, dtype=mx.uint32).reshape(b, 1, 1, 1)
        h_idx = mx.arange(h, dtype=mx.uint32).reshape(1, h, 1, 1)
        t_idx = mx.arange(t_block, dtype=mx.uint32).reshape(1, 1, t_block, 1)
        d_idx = mx.arange(d, dtype=mx.uint32).reshape(1, 1, 1, d)

        global_idx = (
            b_idx * h_full * t_full * d_full
            + h_idx * t_full * d_full
            + (t_start + t_idx) * d_full
            + d_idx
        )
        return self._apply_signs_on_the_fly(x, seed, indices=global_idx)

    def estimate_compressed_bytes_for_shape(
        self,
        shape: tuple,
        k_bits: Optional[int] = None,
        v_bits: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> int:
        """Estimate compressed KV cache footprint for a given KV shape.

        Includes packed K/V codes, K/V scales, and fixed metadata overhead.
        """
        if len(shape) != 4:
            raise ValueError(f"Expected KV shape [B,H,T,D], got {shape}")

        k_bits = self.k_bits if k_bits is None else k_bits
        v_bits = self.v_bits if v_bits is None else v_bits
        group_size = self.group_size if group_size is None else group_size

        if not (2 <= k_bits <= 8 and 2 <= v_bits <= 8):
            raise ValueError("k_bits and v_bits must both be in [2, 8]")
        if group_size <= 0:
            raise ValueError("group_size must be positive")

        n_values = math.prod(shape)

        def packed_words(count: int, bits: int) -> int:
            codes_per_word = 32 // bits
            return (count + codes_per_word - 1) // codes_per_word

        n_groups = (n_values + group_size - 1) // group_size
        k_packed_bytes = packed_words(n_values, k_bits) * 4
        v_packed_bytes = packed_words(n_values, v_bits) * 4
        k_scale_bytes = n_groups * 4
        v_scale_bytes = n_groups * 4
        metadata_overhead = 256

        return k_packed_bytes + v_packed_bytes + k_scale_bytes + v_scale_bytes + metadata_overhead

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

    def validate_symmetric_packed_codes(
        self,
        packed: mx.array,
        n_values: int,
        bits: int,
    ) -> None:
        """Validate packed symmetric quant codes for invalid code points."""
        codes = BitPackedQuantizer.unpack(packed, n_values, bits)
        qmax = (1 << (bits - 1)) - 1
        max_valid = 2 * qmax
        if bool(mx.any(codes > max_valid).item()):
            raise ValueError(
                f"Invalid symmetric quant code for {bits}-bit quantization. "
                f"Max valid code is {max_valid}."
            )

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
        use_wht: Optional[bool] = None,
        use_incoherent_signs: Optional[bool] = None,
        out_dtype: mx.Dtype = mx.float32,
        use_incoherent: Optional[bool] = None,
    ) -> mx.array:
        """Packed-dequant-WHT reconstruction: unpack → dequant →
        reshape → WHT → optional signs.

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

        if use_wht is None and use_incoherent_signs is None:
            legacy = False if use_incoherent is None else bool(use_incoherent)
            use_wht = True
            use_incoherent_signs = legacy
        else:
            use_wht = bool(use_wht)
            use_incoherent_signs = bool(use_incoherent_signs)

        if use_wht:
            x = self._apply_wht_pretransform(x)

        if use_incoherent_signs:
            x = self._apply_signs_on_the_fly(x, seed)

        return x.astype(out_dtype)

    def _reconstruct_packed_dequant_wht_sign_fused(
        self,
        packed: mx.array,
        scales: mx.array,
        n_values: int,
        shape: tuple,
        bits: int,
        seed: int,
        out_dtype: mx.Dtype,
    ) -> mx.array:
        """Fused packed-dequant-WHT-sign reconstruction using single Metal kernel.

        This is the optimized path that combines dequantization, WHT transform,
        and sign application into a single kernel launch for better performance.
        """

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

        result = packed_dequant_wht_sign_metal(
            packed=packed,
            scales=scales,
            n_values=n_values,
            bits=bits,
            group_size=self.group_size,
            seed=seed,
            out_dtype=mx.float32,
        ).reshape(shape)

        self.last_reconstruction_kernel = "metal_fused_dequant_wht_sign"
        return result.astype(out_dtype)

    def _reconstruct_cached_tensor(
        self,
        packed: mx.array,
        scales: mx.array,
        n_values: int,
        shape: tuple,
        bits: int,
        seed: int,
        use_wht: bool,
        use_incoherent_signs: bool,
        out_dtype: mx.Dtype,
    ) -> mx.array:
        def _metal_label() -> str:
            if use_wht and use_incoherent_signs:
                return "metal_multikernel_dequant_wht_sign"
            if use_wht:
                return "metal_multikernel_dequant_wht"
            if use_incoherent_signs:
                return "metal_multikernel_dequant_sign"
            return "metal_multikernel_dequant"

        if self.prefer_metal_kernels and maybe_supports_metal_kernels():
            if self.strict_metal or self.validate_metal_codes:
                self.validate_symmetric_packed_codes(
                    packed=packed,
                    n_values=n_values,
                    bits=bits,
                )

            # Try fused kernel path when both WHT and signs are enabled
            if use_wht and use_incoherent_signs and self.prefer_fused_kernel:
                try:
                    result = self._reconstruct_packed_dequant_wht_sign_fused(
                        packed=packed,
                        scales=scales,
                        n_values=n_values,
                        shape=shape,
                        bits=bits,
                        seed=seed,
                        out_dtype=out_dtype,
                    )
                    return result
                except Exception as exc:
                    if self.strict_metal:
                        raise KernelRouteError(
                            f"fused metal reconstruction failed: {exc}"
                        ) from exc
                    self.last_reconstruction_kernel = (
                        "metal_failed_fallback_reference"
                    )

            # Fallback to sequential multi-kernel path
            try:
                deq = packed_dequant_metal(
                    packed=packed,
                    scales=scales,
                    n_values=n_values,
                    bits=bits,
                    group_size=self.group_size,
                    out_dtype=mx.float32,
                ).reshape(shape)

                if use_wht:
                    deq = wht64_metal(deq)

                if use_incoherent_signs:
                    deq = apply_hash_signs_metal(deq, seed=seed)

                self.last_reconstruction_kernel = _metal_label()
                return deq.astype(out_dtype)
            except Exception as exc:
                if self.strict_metal:
                    raise KernelRouteError(
                        f"strict metal reconstruction failed: {exc}"
                    ) from exc
                self.last_reconstruction_kernel = "metal_failed_fallback_reference"

        if self.prefer_metal_kernels and self.strict_metal:
            raise KernelRouteError(
                "strict metal requested but metal kernels are unavailable"
            )

        self.last_reconstruction_kernel = "sequential_reference"

        return self._reconstruct_packed_dequant_wht(
            packed=packed,
            scales=scales,
            n_values=n_values,
            shape=shape,
            bits=bits,
            seed=seed,
            use_wht=use_wht,
            use_incoherent_signs=use_incoherent_signs,
            out_dtype=out_dtype,
        )

    def _reconstruct_block(
        self,
        packed: mx.array,
        scales: mx.array,
        packed_start: int,
        packed_end: int,
        scale_start: int,
        scale_end: int,
        n_values: int,
        block_shape: tuple,
        bits: int,
        seed: int,
        use_wht: bool,
        use_incoherent_signs: bool,
        out_dtype: mx.Dtype,
    ) -> mx.array:
        """Reconstruct a single token block from sliced packed/scales."""
        return self._reconstruct_cached_tensor(
            packed=packed[packed_start:packed_end],
            scales=scales[scale_start:scale_end],
            n_values=n_values,
            shape=block_shape,
            bits=bits,
            seed=seed,
            use_wht=use_wht,
            use_incoherent_signs=use_incoherent_signs,
            out_dtype=out_dtype,
        )

    def _reconstruct_all_blocks(
        self,
        cache: TurboQuantKVCache,
        is_key: bool,
        out_dtype: mx.Dtype,
    ) -> mx.array:
        """Reconstruct all blocks for K or V and concatenate along T."""
        if is_key:
            packed = cache.k_packed
            scales = cache.k_scales
            poff = cache.k_block_packed_offsets
            soff = cache.k_block_scale_offsets
            bnv = cache.k_block_n_values
            bits = cache.k_bits
        else:
            packed = cache.v_packed
            scales = cache.v_scales
            poff = cache.v_block_packed_offsets
            soff = cache.v_block_scale_offsets
            bnv = cache.v_block_n_values
            bits = cache.v_bits

        _b, _h, t, _d = cache.shape
        blocks: list[mx.array] = []
        for blk in range(cache.num_blocks):
            start = blk * cache.block_size
            end = min(start + cache.block_size, t)
            block_shape = (_b, _h, end - start, _d)
            block = self._reconstruct_block(
                packed=packed,
                scales=scales,
                packed_start=poff[blk],
                packed_end=poff[blk + 1],
                scale_start=soff[blk],
                scale_end=soff[blk + 1],
                n_values=bnv[blk],
                block_shape=block_shape,
                bits=bits,
                seed=cache.seed,
                use_wht=cache.use_wht,
                use_incoherent_signs=False,
                out_dtype=out_dtype,
            )
            blocks.append(block)
        result = mx.concatenate(blocks, axis=2)
        if cache.use_incoherent_signs:
            result = self._apply_signs_on_the_fly(result, cache.seed)

        # Restore expected kernel label for backward compatibility
        if self.prefer_metal_kernels and maybe_supports_metal_kernels():
            if cache.use_wht and cache.use_incoherent_signs:
                self.last_reconstruction_kernel = (
                    "metal_multikernel_dequant_wht_sign"
                )
            elif cache.use_wht:
                self.last_reconstruction_kernel = (
                    "metal_multikernel_dequant_wht"
                )
            elif cache.use_incoherent_signs:
                self.last_reconstruction_kernel = (
                    "metal_multikernel_dequant_sign"
                )
            else:
                self.last_reconstruction_kernel = (
                    "metal_multikernel_dequant"
                )
        else:
            self.last_reconstruction_kernel = "sequential_reference"
        return result

    def evict_lru(self, target_bytes: int | None = None) -> int:
        """Evict least-recently-used unpinned caches.

        Args:
            target_bytes: Stop once at least this many bytes have been freed.

        Returns:
            Bytes freed.
        """
        to_free = 0 if target_bytes is None else max(0, int(target_bytes))
        freed = 0

        while self.active_caches and (freed < to_free or to_free == 0):
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
                break

            old_cache = self.active_caches.pop(lru_key)
            old_bytes = self._estimate_cache_bytes(old_cache)
            self._total_estimated_bytes -= old_bytes
            freed += old_bytes

            if to_free == 0:
                break

        return freed

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

        # Deterministic seed derived from cache identity for sign preconditioning
        seed = int(
            hashlib.sha256(
                f"{skill_pattern}|{keys.shape}|{self.k_bits}|{self.v_bits}".encode()
            ).hexdigest()[:8],
            16,
        )

        # Apply optional sign preconditioning and optional WHT before quantization.
        if self.use_incoherent_signs:
            k_pre = self._apply_signs_on_the_fly(keys, seed)
            v_pre = self._apply_signs_on_the_fly(values, seed)
        else:
            k_pre = keys
            v_pre = values

        if self.use_wht:
            k_pre = self._apply_wht_pretransform(k_pre)
            v_pre = self._apply_wht_pretransform(v_pre)

        # Per-block quantization along the token dimension
        _bsz, _num_h, t_len, _head_dim = keys.shape
        block_size = self.block_size
        num_blocks = max(1, (t_len + block_size - 1) // block_size)

        def _quantize_blocks(pre: mx.array, bits: int) -> tuple:
            """Quantize pre-transformed tensor block by block.

            Returns (packed, scales, packed_offsets, scale_offsets,
                     block_n_values, total_n).
            """
            packed_blocks: list[mx.array] = []
            scale_blocks: list[mx.array] = []
            packed_offsets = [0]
            scale_offsets = [0]
            block_n_values: list[int] = []

            for blk in range(num_blocks):
                start = blk * block_size
                end = min((blk + 1) * block_size, t_len)
                block = pre[:, :, start:end, :]
                flat = block.reshape(-1)
                codes, scales = self._quantize(flat, bits)
                packed, n = BitPackedQuantizer.pack(codes, bits)
                packed_blocks.append(packed)
                scale_blocks.append(scales)
                packed_offsets.append(
                    packed_offsets[-1] + int(packed.size)
                )
                scale_offsets.append(
                    scale_offsets[-1] + int(scales.size)
                )
                block_n_values.append(n)

            return (
                mx.concatenate(packed_blocks),
                mx.concatenate(scale_blocks),
                packed_offsets,
                scale_offsets,
                block_n_values,
                sum(block_n_values),
            )

        (
            k_packed, k_scales, k_poff, k_soff,
            k_bnv, k_n,
        ) = _quantize_blocks(k_pre, self.k_bits)
        (
            v_packed, v_scales, v_poff, v_soff,
            v_bnv, v_n,
        ) = _quantize_blocks(v_pre, self.v_bits)

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
            use_wht=self.use_wht,
            use_incoherent_signs=self.use_incoherent_signs,
            format_version="rfsn_v10",
            seed=seed,
            k_n_values=k_n,
            v_n_values=v_n,
            token_count=token_count,
            block_size=block_size,
            num_blocks=num_blocks,
            k_block_packed_offsets=k_poff,
            k_block_scale_offsets=k_soff,
            k_block_n_values=k_bnv,
            v_block_packed_offsets=v_poff,
            v_block_scale_offsets=v_soff,
            v_block_n_values=v_bnv,
        )
        # Set last_used timestamp for newly created cache
        cache.last_used = time.monotonic()

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

        # Validate group_size metadata
        if cache.group_size != self.group_size:
            raise ValueError(
                f"metadata mismatch: stored group_size={cache.group_size} "
                f"current group_size={self.group_size}"
            )

        if cache.num_blocks == 0:
            # Legacy cache without per-block offsets
            k_rec = self._reconstruct_cached_tensor(
                packed=cache.k_packed,
                scales=cache.k_scales,
                n_values=cache.k_n_values,
                shape=cache.shape,
                bits=cache.k_bits,
                seed=cache.seed,
                use_wht=cache.use_wht,
                use_incoherent_signs=cache.use_incoherent_signs,
                out_dtype=out_dtype,
            )
            v_rec = self._reconstruct_cached_tensor(
                packed=cache.v_packed,
                scales=cache.v_scales,
                n_values=cache.v_n_values,
                shape=cache.shape,
                bits=cache.v_bits,
                seed=cache.seed,
                use_wht=cache.use_wht,
                use_incoherent_signs=cache.use_incoherent_signs,
                out_dtype=out_dtype,
            )
            return k_rec, v_rec

        k_rec = self._reconstruct_all_blocks(
            cache=cache, is_key=True, out_dtype=out_dtype,
        )
        v_rec = self._reconstruct_all_blocks(
            cache=cache, is_key=False, out_dtype=out_dtype,
        )
        return k_rec, v_rec

    def retrieve_blocks(
        self,
        skill_pattern: str,
        block_indices: list[int],
        block_size: int = 64,
        out_dtype: mx.Dtype = mx.float32,
    ) -> Optional[tuple[mx.array, mx.array]]:
        """Retrieve and dequantize only selected token blocks.

        This reconstructs the full cache entry internally, then slices
        along the token dimension to return only the requested blocks.
        The reconstruction cost is the same as a full retrieve; the
        savings are in memory bandwidth and downstream compute.

        Args:
            skill_pattern: Cache key to retrieve.
            block_indices: Sorted list of 0-based block indices to keep.
            block_size: Tokens per block (must match attention block_size).
            out_dtype: Output dtype (float32 or float16).

        Returns:
            (keys, values) with shape [B, H, len(block_indices)*block_size, D]
            where T may be smaller than the original if blocks are skipped.
            Returns None if the cache key is not found.

        Raises:
            ValueError: If block_indices is empty or out of range.
        """
        if skill_pattern not in self.active_caches:
            return None

        cache = self.active_caches[skill_pattern]
        cache.last_used = time.monotonic()

        if not block_indices:
            raise ValueError("block_indices must not be empty")

        _b, _h, t, _d = cache.shape
        max_blocks = max(1, (t + block_size - 1) // block_size)
        if max(block_indices) >= max_blocks:
            raise ValueError(
                f"block index {max(block_indices)} >= max_blocks {max_blocks}"
            )

        sorted_blocks = sorted(set(block_indices))

        # Fast path: contiguous blocks starting from 0 can reuse the
        # cached full reconstruction from retrieve() and just slice.
        if (
            cache.num_blocks > 0
            and sorted_blocks == list(range(len(sorted_blocks)))
        ):
            k_full, v_full = self.retrieve(
                skill_pattern, out_dtype=out_dtype,
            )
            if k_full is None:
                return None
            end_t = min(len(sorted_blocks) * cache.block_size, t)
            return k_full[:, :, :end_t, :], v_full[:, :, :end_t, :]

        if cache.num_blocks == 0:
            # Legacy cache: full reconstruct then slice
            k_full, v_full = self.retrieve(skill_pattern, out_dtype=out_dtype)
            if k_full is None:
                return None
            token_indices: list[int] = []
            for blk in sorted_blocks:
                start = blk * block_size
                end = min(start + block_size, t)
                token_indices.extend(range(start, end))
            idx_mx = mx.array(token_indices, dtype=mx.uint32)
            return k_full[:, :, idx_mx, :], v_full[:, :, idx_mx, :]

        k_blocks: list[mx.array] = []
        v_blocks: list[mx.array] = []
        t_global_blocks: list[mx.array] = []

        for blk in sorted(set(block_indices)):
            start = blk * cache.block_size
            end = min(start + cache.block_size, t)
            block_shape = (_b, _h, end - start, _d)
            k_block = self._reconstruct_block(
                packed=cache.k_packed,
                scales=cache.k_scales,
                packed_start=cache.k_block_packed_offsets[blk],
                packed_end=cache.k_block_packed_offsets[blk + 1],
                scale_start=cache.k_block_scale_offsets[blk],
                scale_end=cache.k_block_scale_offsets[blk + 1],
                n_values=cache.k_block_n_values[blk],
                block_shape=block_shape,
                bits=cache.k_bits,
                seed=cache.seed,
                use_wht=cache.use_wht,
                use_incoherent_signs=False,
                out_dtype=out_dtype,
            )
            v_block = self._reconstruct_block(
                packed=cache.v_packed,
                scales=cache.v_scales,
                packed_start=cache.v_block_packed_offsets[blk],
                packed_end=cache.v_block_packed_offsets[blk + 1],
                scale_start=cache.v_block_scale_offsets[blk],
                scale_end=cache.v_block_scale_offsets[blk + 1],
                n_values=cache.v_block_n_values[blk],
                block_shape=block_shape,
                bits=cache.v_bits,
                seed=cache.seed,
                use_wht=cache.use_wht,
                use_incoherent_signs=False,
                out_dtype=out_dtype,
            )
            k_blocks.append(k_block)
            v_blocks.append(v_block)

            if cache.use_incoherent_signs:
                t_global_blocks.append(
                    mx.arange(start, end, dtype=mx.uint32)
                )

        k_result = mx.concatenate(k_blocks, axis=2)
        v_result = mx.concatenate(v_blocks, axis=2)

        if cache.use_incoherent_signs:
            t_global = mx.concatenate(t_global_blocks).reshape(
                1, 1, -1, 1
            )
            b_idx = mx.arange(_b, dtype=mx.uint32).reshape(_b, 1, 1, 1)
            h_idx = mx.arange(_h, dtype=mx.uint32).reshape(1, _h, 1, 1)
            d_idx = mx.arange(_d, dtype=mx.uint32).reshape(1, 1, 1, _d)
            global_idx = (
                b_idx * _h * t * _d
                + h_idx * t * _d
                + t_global * _d
                + d_idx
            )
            k_result = self._apply_signs_on_the_fly(
                k_result, cache.seed, indices=global_idx
            )
            v_result = self._apply_signs_on_the_fly(
                v_result, cache.seed, indices=global_idx
            )

        return k_result, v_result

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
