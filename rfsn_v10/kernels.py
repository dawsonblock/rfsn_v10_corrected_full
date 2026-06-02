"""MLX metal-kernel helpers for KV reconstruction routes."""

from __future__ import annotations

import math

from .compat import MLX_AVAILABLE, ensure_mlx_available, mx


class KernelRouteError(RuntimeError):
    """Raised when a requested reconstruction route cannot run."""


def sequential_reference_route_supported(
    *,
    shape: tuple,
    out_dtype: mx.Dtype,
    use_wht: bool,
    use_incoherent_signs: bool,
) -> tuple[bool, str]:
    if not use_wht:
        return False, "sequential_reference_requires_wht"
    if not use_incoherent_signs:
        return False, "sequential_reference_requires_incoherent_signs"
    if out_dtype not in (mx.float32, mx.float16):
        return False, "sequential_reference_out_dtype_unsupported"
    if len(shape) != 4:
        return False, "sequential_reference_shape_rank_unsupported"
    if shape[-1] % 64 != 0:
        return False, "sequential_reference_head_dim_unsupported"
    return True, "sequential_reference_supported"


def _run_metal_kernel(
    *,
    kernel,
    inputs: list[mx.array],
    output_shape: tuple[int, ...],
    output_dtype: mx.Dtype,
    n_threads: int,
) -> mx.array:
    threadgroup_x = 256 if n_threads >= 256 else max(1, int(n_threads))
    outputs = kernel(
        inputs=inputs,
        template=[("T", output_dtype)],
        grid=(int(n_threads), 1, 1),
        threadgroup=(threadgroup_x, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def _wht_transform_mx(x: mx.array, block: int = 64) -> mx.array:
    if block != 64:
        raise KernelRouteError(f"wht block must be 64, got {block}")
    if x.shape[-1] % block != 0:
        raise KernelRouteError(
            f"wht requires head dim multiple of {block}, got {x.shape[-1]}"
        )

    def _wht_block_recursive(y: mx.array) -> mx.array:
        n = y.shape[-1]
        if n == 1:
            return y

        half = n // 2
        y0 = y[..., :half]
        y1 = y[..., half:]

        a = _wht_block_recursive(y0)
        b = _wht_block_recursive(y1)
        return mx.concatenate([a + b, a - b], axis=-1)

    shape = x.shape
    y = x.reshape(-1, block)
    y = _wht_block_recursive(y)
    y = y / math.sqrt(block)
    return y.reshape(shape)


def wht64_metal(
    x: mx.array,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Apply normalized WHT over contiguous 64-value blocks with Metal."""
    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")
    if x.size == 0:
        raise ValueError("Cannot WHT-transform empty tensor.")
    if x.shape[-1] % 64 != 0:
        raise ValueError("Last dimension must be a multiple of 64.")

    source = """
        uint tgid = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint gid = tgid * 64u + lid;
        uint n = n_buf[0];

        threadgroup float smem[64];
        float val = 0.0f;
        if (gid < n) {
            val = float(x[gid]);
        }

        smem[lid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = 1u; step < 64u; step *= 2u) {
            uint partner = lid ^ step;
            float a = smem[lid];
            float b = smem[partner];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            smem[lid] = ((lid & step) == 0u) ? (a + b) : (b - a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (gid < n) {
            out[gid] = T(smem[lid] / 8.0f);
        }
    """

    kernel = mx.fast.metal_kernel(
        name="rfsn_wht64",
        input_names=["x", "n_buf"],
        output_names=["out"],
        source=source,
    )

    shape = x.shape
    flat = mx.array(x.reshape(-1))
    n = int(flat.size)
    n_buf = mx.array([n], dtype=mx.uint32)
    outputs = kernel(
        inputs=[flat, n_buf],
        template=[("T", out_dtype)],
        grid=(n, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[out_dtype],
    )

    return outputs[0].reshape(shape)


def apply_hash_signs_metal(x: mx.array, seed: int) -> mx.array:
    """Apply deterministic +/-1 signs with an MLX metal kernel."""
    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")

    source = """
        uint gid = thread_position_in_grid.x;
        uint n = n_buf[0];
        uint seed_val = seed_buf[0];

        if (gid >= n) { return; }

        uint state = gid ^ seed_val;
        state ^= state >> 15;
        state *= 0x85ebca6bu;
        state ^= state >> 13;
        state *= 0xc2b2ae35u;
        state ^= state >> 16;

        T sign = (state & 1u) ? T(-1.0f) : T(1.0f);
        out[gid] = x[gid] * sign;
    """

    kernel = mx.fast.metal_kernel(
        name="rfsn_hash_sign",
        input_names=["x", "seed_buf", "n_buf"],
        output_names=["out"],
        source=source,
    )

    flat = mx.array(x.reshape(-1))
    n = int(flat.size)
    seed_buf = mx.array([seed & 0xFFFFFFFF], dtype=mx.uint32)
    n_buf = mx.array([n], dtype=mx.uint32)

    out = _run_metal_kernel(
        kernel=kernel,
        inputs=[flat, seed_buf, n_buf],
        output_shape=(n,),
        output_dtype=flat.dtype,
        n_threads=n,
    )
    return out.reshape(x.shape)


def packed_dequant_metal(
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    bits: int,
    group_size: int = 64,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize packed symmetric codes using a custom metal kernel."""
    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")
    if bits < 2 or bits > 8:
        raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
    if group_size <= 0:
        raise KernelRouteError(f"group_size must be positive, got {group_size}")

    source = """
        uint gid = thread_position_in_grid.x;
        uint n = n_buf[0];
        if (gid >= n) { return; }

        uint bits = bits_buf[0];
        uint group_size = group_buf[0];
        uint qmax = qmax_buf[0];

        uint codes_per_word = 32u / bits;
        uint word_idx = gid / codes_per_word;
        uint offset = (gid % codes_per_word) * bits;
        uint mask = (1u << bits) - 1u;

        uint code = (packed[word_idx] >> offset) & mask;
        uint group_idx = gid / group_size;
        float scale = float(scales[group_idx]);

        float deq = (float(code) - float(qmax)) * scale;
        out[gid] = T(deq);
    """

    kernel = mx.fast.metal_kernel(
        name="rfsn_packed_dequant",
        input_names=["packed", "scales", "n_buf", "bits_buf", "group_buf", "qmax_buf"],
        output_names=["out"],
        source=source,
    )

    qmax = (1 << (bits - 1)) - 1
    n_buf = mx.array([int(n_values)], dtype=mx.uint32)
    bits_buf = mx.array([int(bits)], dtype=mx.uint32)
    group_buf = mx.array([int(group_size)], dtype=mx.uint32)
    qmax_buf = mx.array([int(qmax)], dtype=mx.uint32)

    return _run_metal_kernel(
        kernel=kernel,
        inputs=[packed, scales.astype(mx.float32), n_buf, bits_buf, group_buf, qmax_buf],
        output_shape=(int(n_values),),
        output_dtype=out_dtype,
        n_threads=int(n_values),
    )


def reconstruct_packed_dequant_wht_sign_metal(
    *,
    packed: mx.array,
    scales: mx.array,
    n_values: int,
    shape: tuple,
    bits: int,
    seed: int,
    out_dtype: mx.Dtype,
    group_size: int,
) -> mx.array:
    """Metal path: packed dequant kernel -> WHT transform -> sign kernel."""
    if not MLX_AVAILABLE:
        raise KernelRouteError("mlx_not_available")
    if out_dtype not in (mx.float32, mx.float16):
        raise KernelRouteError(f"out dtype unsupported: {out_dtype}")
    if math.prod(shape) != n_values:
        raise KernelRouteError(
            f"shape product {math.prod(shape)} does not match n_values {n_values}"
        )

    deq = packed_dequant_metal(
        packed=packed,
        scales=scales,
        n_values=n_values,
        bits=bits,
        group_size=group_size,
        out_dtype=out_dtype,
    )
    x = deq.reshape(shape)
    x = _wht_transform_mx(x, block=64)
    x = apply_hash_signs_metal(x, seed=seed)
    return x.astype(out_dtype)


def maybe_supports_metal_kernels() -> bool:
    if not MLX_AVAILABLE:
        return False
    return hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
