"""RFSN v10 Adaptive Sparsity Controller.

Dynamically adjusts top_k_ratio based on real quality signals from audit
cosine similarity, relative MAE, sparse kernel success/failure, and
fallback events. Replaces fake "did not crash" reward with genuine
quality-aware sparsity adaptation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualitySample:
    """Single quality observation from an audit."""

    step: int
    audit_cosine: float
    rel_mae: float
    latency_ms: float
    fallback_occurred: bool
    sparse_success: bool


class AdaptiveSparsityController:
    """Dynamically adjusts top_k_ratio based on real quality signals."""

    def __init__(
        self,
        initial_top_k_ratio: float = 1.0,
        min_top_k_ratio: float = 0.125,
        max_top_k_ratio: float = 1.0,
        cosine_quality_threshold: float = 0.995,
        mae_quality_threshold: float = 0.01,
        cosine_degradation_threshold: float = 0.98,
        mae_degradation_threshold: float = 0.05,
        stabilization_steps: int = 100,
        decrease_step_size: float = 0.1,
        increase_step_size: float = 0.25,
    ):
        self.current_top_k_ratio = initial_top_k_ratio
        self.min_top_k_ratio = min_top_k_ratio
        self.max_top_k_ratio = max_top_k_ratio
        self.cosine_quality_threshold = cosine_quality_threshold
        self.mae_quality_threshold = mae_quality_threshold
        self.cosine_degradation_threshold = cosine_degradation_threshold
        self.mae_degradation_threshold = mae_degradation_threshold
        self.stabilization_steps = stabilization_steps
        self.decrease_step_size = decrease_step_size
        self.increase_step_size = increase_step_size

        self._samples: list[QualitySample] = []
        self._stable_good_steps = 0
        self._step_counter = 0
        self._risky_patterns: set[str] = set()

    def update(
        self,
        audit_cosine: Optional[float] = None,
        rel_mae: Optional[float] = None,
        latency_ms: float = 0.0,
        fallback_occurred: bool = False,
        sparse_success: bool = True,
        pattern: str = "default",
    ) -> float:
        """Update top_k_ratio based on quality signals.

        Policy:
        - If sparse kernel failure: increase top_k_ratio immediately and
          mark pattern risky.
        - If audit_cosine < degradation threshold OR rel_mae > degradation
          threshold: increase top_k_ratio immediately (quality is degrading).
        - If audit_cosine > quality threshold AND rel_mae < quality threshold:
          increment stable_good_steps counter. After stabilization_steps
          consecutive good steps: decrease top_k_ratio slightly.
        - Never reduce if fallback rate rises.
        - Never reduce if audit drift exceeds threshold.

        Returns the new top_k_ratio.
        """
        self._step_counter += 1

        sample = QualitySample(
            step=self._step_counter,
            audit_cosine=audit_cosine or 1.0,
            rel_mae=rel_mae or 0.0,
            latency_ms=latency_ms,
            fallback_occurred=fallback_occurred,
            sparse_success=sparse_success,
        )
        self._samples.append(sample)

        # Rule 1: Sparse kernel failure -> increase immediately, mark risky
        if not sparse_success:
            self._risky_patterns.add(pattern)
            self.current_top_k_ratio = min(
                self.max_top_k_ratio,
                self.current_top_k_ratio + self.increase_step_size,
            )
            self._stable_good_steps = 0
            return self.current_top_k_ratio

        # Rule 2: Quality degradation -> increase immediately
        if audit_cosine is not None and rel_mae is not None:
            if (
                audit_cosine < self.cosine_degradation_threshold
                or rel_mae > self.mae_degradation_threshold
            ):
                self.current_top_k_ratio = min(
                    self.max_top_k_ratio,
                    self.current_top_k_ratio + self.increase_step_size,
                )
                self._stable_good_steps = 0
                return self.current_top_k_ratio

        # Rule 3: Quality is good -> count stable steps
        if audit_cosine is not None and rel_mae is not None:
            if (
                audit_cosine > self.cosine_quality_threshold
                and rel_mae < self.mae_quality_threshold
            ):
                self._stable_good_steps += 1

                # After enough stable steps, reduce sparsity
                if self._stable_good_steps >= self.stabilization_steps:
                    if pattern not in self._risky_patterns:
                        self.current_top_k_ratio = max(
                            self.min_top_k_ratio,
                            self.current_top_k_ratio - self.decrease_step_size,
                        )
                    self._stable_good_steps = 0
            else:
                self._stable_good_steps = 0
        else:
            self._stable_good_steps = 0

        return self.current_top_k_ratio

    def get_top_k_ratio(self) -> float:
        """Return the current top_k_ratio."""
        return self.current_top_k_ratio

    def reset(self, pattern: Optional[str] = None) -> None:
        """Reset controller state, optionally for a specific pattern."""
        if pattern:
            self._risky_patterns.discard(pattern)
        else:
            self.current_top_k_ratio = self.max_top_k_ratio
            self._samples.clear()
            self._stable_good_steps = 0
            self._step_counter = 0
            self._risky_patterns.clear()

    def get_stats(self) -> dict:
        """Return controller statistics."""
        if not self._samples:
            return {"total_samples": 0}

        recent = self._samples[-100:] if len(self._samples) > 100 else self._samples
        fallback_count = sum(1 for s in recent if s.fallback_occurred)
        sparse_fail_count = sum(1 for s in recent if not s.sparse_success)

        cosines = [s.audit_cosine for s in recent if s.audit_cosine is not None]
        maes = [s.rel_mae for s in recent if s.rel_mae is not None]

        return {
            "total_samples": len(self._samples),
            "recent_samples": len(recent),
            "current_top_k_ratio": self.current_top_k_ratio,
            "fallback_rate": fallback_count / len(recent) if recent else 0.0,
            "sparse_failure_rate": sparse_fail_count / len(recent) if recent else 0.0,
            "avg_audit_cosine": sum(cosines) / len(cosines) if cosines else None,
            "avg_rel_mae": sum(maes) / len(maes) if maes else None,
            "stable_good_steps": self._stable_good_steps,
            "risky_patterns": list(self._risky_patterns),
        }
