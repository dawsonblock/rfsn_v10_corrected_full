"""Kernel routing helpers for packed-dequant-WHT reconstruction."""

from __future__ import annotations

from typing import Callable

from .compat import mx
from .bitpack import BitPackedQuantizer


class KernelRouteError(RuntimeError):
    """Raised when custom-kernel routing is unsupported for an input."""


def custom_kernel_supported(
    *,
    shape: tuple,
    out_dtype: mx.Dtype,
    use_wht: bool,
    use_incoherent_signs: bool,
) -> tuple[bool, str]:
    if not use_wht:
        return False, "custom_kernel_requires_wht"
    if not use_incoherent_signs:
        return False, "custom_kernel_requires_incoherent_signs"
    if out_dtype not in (mx.float32, mx.float16):
        return False, "custom_kernel_out_dtype_unsupported"
    if len(shape) != 4:
        return False, "custom_kernel_shape_rank_unsupported"
    if shape[-1] % 64 != 0:
        return False, "custom_kernel_head_dim_unsupported"
    return True, "custom_kernel_supported"


def reconstruct_packed_dequant_wht_custom(
    *,
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    shape: tuple,
    bits: int,
    seed: int,
    out_dtype: mx.Dtype,
    dequantize_fn: Callable[[mx.array, mx.array, int], mx.array],
    wht_fn: Callable[[mx.array], mx.array],
    signs_fn: Callable[[mx.array, int], mx.array],
) -> mx.array:
    supported, reason = custom_kernel_supported(
        shape=shape,
        out_dtype=out_dtype,
        use_wht=True,
        use_incoherent_signs=True,
    )
    if not supported:
        raise KernelRouteError(reason)

    codes = BitPackedQuantizer.unpack(packed, n_values, bits)
    x = dequantize_fn(codes, scales, bits).reshape(shape)
    x = wht_fn(x)
    x = signs_fn(x, seed)
    return x.astype(out_dtype)
