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
    "AdaptiveDecision": ".adaptive_sparsity",
    "MemoryGuard": ".memory_guard",
    "AsyncWriter": ".async_writer",
    "ClickHouseClient": ".clickhouse_client",
    "KernelRouteError": ".kernels",
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
    "AdaptiveDecision",
    "MemoryGuard",
    "AsyncWriter",
    "ClickHouseClient",
    "KernelRouteError",
]


def __getattr__(name: str):
    # RFSNRuntime and TelemetryEvent live in the sibling runtime.py module,
    # which is shadowed by the runtime/ package.  Load the module file
    # directly so that accessing them does not trigger an MLX import via
    # the runtime package __init__.
    if name in {"RFSNRuntime", "TelemetryEvent"}:
        import importlib.util
        import sys
        from pathlib import Path

        runtime_py = Path(__file__).with_name("runtime.py")
        spec = importlib.util.spec_from_file_location(
            "rfsn_v10._runtime_module", runtime_py
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot find rfsn_v10/runtime.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rfsn_v10._runtime_module"] = mod
        spec.loader.exec_module(mod)
        value = getattr(mod, name)
        globals()[name] = value
        return value

    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
