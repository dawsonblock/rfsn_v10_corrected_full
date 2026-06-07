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

from .audit import (
    AuditEvent,
    AuditMetrics,
    audit_decode_step,
    check_drift,
    log_audit_event,
)
from .experimental_quant_runtime import (
    ExperimentalQuantRuntime,
    ExperimentalQuantState,
    LayerQuantPolicy,
    QuantTelemetryEvent,
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
    # experimental_quant_runtime
    "ExperimentalQuantRuntime",
    "ExperimentalQuantState",
    "LayerQuantPolicy",
    "QuantTelemetryEvent",
]
