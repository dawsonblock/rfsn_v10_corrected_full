"""RFSN v10 — Runtime sub-package.

Provides experimental scoring modes, quality audit, and quant runtime
integration layers that wrap :class:`RFSNTurboQuantKVManager` without
mutating existing production code paths.

The stable production runtime lives in the sibling ``rfsn_v10/runtime.py``
module.  It is **not** re-exported here so that importing submodules of
``rfsn_v10.runtime`` (e.g. ``adaptive_controller``) does not force an
MLX import on non-MLX systems.
"""

from __future__ import annotations

from .adaptive_controller import AdaptiveQuantController
from .audit import (
    AuditEvent,
    AuditMetrics,
    audit_decode_step,
    check_drift,
    log_audit_event,
)
from .scoring_modes import (
    score_attention_fp16,
    score_attention_packed_block,
    score_attention_prepared,
    score_attention_reconstructed,
    score_attention_score_corrected,
)

__all__ = [
    # scoring_modes
    "score_attention_fp16",
    "score_attention_reconstructed",
    "score_attention_prepared",
    "score_attention_packed_block",
    "score_attention_score_corrected",
    # audit
    "AuditEvent",
    "AuditMetrics",
    "audit_decode_step",
    "check_drift",
    "log_audit_event",
    # adaptive_controller
    "AdaptiveQuantController",
    # experimental_quant_runtime (lazy)
    "ExperimentalQuantRuntime",
    "ExperimentalQuantState",
    "LayerQuantPolicy",
    "QuantTelemetryEvent",
]

_LAZY_RUNTIME_NAMES = {
    "ExperimentalQuantRuntime",
    "ExperimentalQuantState",
    "LayerQuantPolicy",
    "QuantTelemetryEvent",
}


def __getattr__(name: str):
    if name in _LAZY_RUNTIME_NAMES:
        from .experimental_quant_runtime import (
            ExperimentalQuantRuntime,
            ExperimentalQuantState,
            LayerQuantPolicy,
            QuantTelemetryEvent,
        )
        value = {
            "ExperimentalQuantRuntime": ExperimentalQuantRuntime,
            "ExperimentalQuantState": ExperimentalQuantState,
            "LayerQuantPolicy": LayerQuantPolicy,
            "QuantTelemetryEvent": QuantTelemetryEvent,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
