"""RFSN v10 package surface with lazy MLX-dependent imports."""

from __future__ import annotations

from importlib import import_module

from .compat import MLX_AVAILABLE

_LAZY_IMPORTS = {
    "BitPackedQuantizer": ".bitpack",
    "RFSNTurboQuantKVManager": ".kv_manager",
    "TurboQuantKVCache": ".kv_manager",
    "AdaptiveBlockSparseAttention": ".attention",
    "ExecutionMode": ".attention",
    "RFSNRuntime": ".runtime",
    "TelemetryEvent": ".runtime",
    "AdaptiveSparsityController": ".adaptive_sparsity",
    "QualitySample": ".adaptive_sparsity",
    "MemoryGuard": ".memory_guard",
    "AsyncWriter": ".async_writer",
    "ClickHouseClient": ".clickhouse_client",
    "KernelRouteError": ".kernels",
    "custom_kernel_supported": ".kernels",
}

__all__ = [
    "MLX_AVAILABLE",
    "BitPackedQuantizer",
    "RFSNTurboQuantKVManager",
    "TurboQuantKVCache",
    "AdaptiveBlockSparseAttention",
    "ExecutionMode",
    "RFSNRuntime",
    "TelemetryEvent",
    "AdaptiveSparsityController",
    "QualitySample",
    "MemoryGuard",
    "AsyncWriter",
    "ClickHouseClient",
    "KernelRouteError",
    "custom_kernel_supported",
]


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value