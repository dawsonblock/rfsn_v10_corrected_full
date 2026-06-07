"""RFSN v10 — Runtime sub-package.

Provides experimental scoring modes, quality audit, and quant runtime
integration layers that wrap :class:`RFSNTurboQuantKVManager` without
mutating existing production code paths.

Also re-exports the stable production runtime from ``rfsn_v10/runtime.py``.
Because a ``runtime/`` package shadows the sibling ``runtime.py`` module,
we load the original module explicitly via ``importlib.util``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the original rfsn_v10/runtime.py module (shadowed by this package)
# ---------------------------------------------------------------------------
_runtime_py = Path(__file__).parent.with_name("runtime.py")
_spec = importlib.util.spec_from_file_location(
    "rfsn_v10._runtime_original", _runtime_py
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot find rfsn_v10/runtime.py")
_runtime_original = importlib.util.module_from_spec(_spec)
sys.modules["rfsn_v10._runtime_original"] = _runtime_original
_spec.loader.exec_module(_runtime_original)
RFSNRuntime = _runtime_original.RFSNRuntime
TelemetryEvent = _runtime_original.TelemetryEvent
AdaptiveBlockSparseAttention = _runtime_original.AdaptiveBlockSparseAttention

# ---------------------------------------------------------------------------
# Experimental runtime exports
# ---------------------------------------------------------------------------
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
    # stable runtime (loaded from runtime.py)
    "RFSNRuntime",
    "TelemetryEvent",
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
    # experimental_quant_runtime
    "ExperimentalQuantRuntime",
    "ExperimentalQuantState",
    "LayerQuantPolicy",
    "QuantTelemetryEvent",
]
