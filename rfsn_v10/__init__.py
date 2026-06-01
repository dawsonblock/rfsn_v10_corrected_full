"""RFSN v10 — Quantized KV-cache + decode-time sparse-attention runtime for MLX/Apple Silicon."""

from .bitpack import BitPackedQuantizer
from .kv_manager import RFSNTurboQuantKVManager, TurboQuantKVCache
from .attention import AdaptiveBlockSparseAttention
from .runtime import RFSNRuntime, TelemetryEvent
from .adaptive_sparsity import AdaptiveSparsityController, QualitySample
from .memory_guard import MemoryGuard

__all__ = [
    "BitPackedQuantizer",
    "RFSNTurboQuantKVManager",
    "TurboQuantKVCache",
    "AdaptiveBlockSparseAttention",
    "RFSNRuntime",
    "TelemetryEvent",
    "AdaptiveSparsityController",
    "QualitySample",
    "MemoryGuard",
]
