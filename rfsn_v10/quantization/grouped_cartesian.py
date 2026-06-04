#!/usr/bin/env python3
"""
Grouped Cartesian quantization reference implementation.
Uses symmetric signed quantization:
    q_signed = round(x / scale)
    code = q_signed + qmax
For bits=5:
    qmax = 15
    signed range = [-15, 15]
    code range = [0, 30]
    one code value remains unused. This preserves exact zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class CartesianPacked:
    codes: mx.array
    scale: mx.array
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int


class GroupedCartesianQuantizer:
    def __init__(
        self,
        bits: int = 6,
        group_size: int = 64,
        eps: float = 1e-8,
    ):
        if not (2 <= bits <= 16):
            raise ValueError(f"bits must be in [2,16]. Got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.eps = eps

    def quantize(self, x: mx.array) -> CartesianPacked:
        original_shape = tuple(x.shape)
        flat = x.astype(mx.float32).reshape(-1)
        original_size = int(flat.size)
        pad = (self.group_size - (original_size % self.group_size))
        pad %= self.group_size
        if pad:
            flat = mx.concatenate(
                [flat, mx.zeros((pad,), dtype=flat.dtype)], axis=0
            )
        padded_size = int(flat.size)
        grouped = flat.reshape(-1, self.group_size)
        qmax = (1 << (self.bits - 1)) - 1
        max_abs = mx.maximum(
            mx.max(mx.abs(grouped), axis=1),
            mx.array(self.eps, dtype=mx.float32),
        )
        scale = max_abs / float(qmax)
        q_signed = mx.round(grouped / scale[:, None])
        q_signed = mx.clip(q_signed, -qmax, qmax)
        codes = (q_signed + qmax).astype(mx.uint32).reshape(-1)
        return CartesianPacked(
            codes=codes,
            scale=scale,
            bits=self.bits,
            group_size=self.group_size,
            original_shape=original_shape,
            original_size=original_size,
            padded_size=padded_size,
        )

    def dequantize(self, packed: CartesianPacked) -> mx.array:
        if packed.bits != self.bits:
            raise ValueError(
                f"Packed bits={packed.bits}, quantizer bits={self.bits}"
            )
        flat = packed.codes.astype(mx.float32).reshape(-1)
        if int(flat.size) != packed.padded_size:
            raise ValueError(
                f"Expected {packed.padded_size} codes, got {flat.size}"
            )
        qmax = (1 << (packed.bits - 1)) - 1
        grouped_codes = flat.reshape(-1, packed.group_size)
        q_signed = grouped_codes - float(qmax)
        restored = q_signed * packed.scale[:, None]
        restored = restored.reshape(-1)[: packed.original_size]
        return restored.reshape(packed.original_shape)

    def estimate_bytes(self, packed: CartesianPacked) -> int:
        code_bits = int(packed.codes.size) * packed.bits
        code_bytes = (code_bits + 7) // 8
        scale_bytes = int(packed.scale.size) * 4
        return code_bytes + scale_bytes
