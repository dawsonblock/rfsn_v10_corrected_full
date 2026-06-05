#!/usr/bin/env python3
"""
RFSN v10 — Adaptive layer policy.

Loads a per-layer quantization policy from JSON and validates it.
A policy maps layer IDs to quantization modes, with a default fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KNOWN_MODES = frozenset(
    {
        "baseline_fp16",
        "k8_v5_gs64",
        "k8_v5_gs32",
        "k8_v4_gs64",
        "adaptive",
        "experimental_hybrid",
        "turbo_polar",
        "turbo_k8r8v6",
        "cartesian",
        "polar",
        "hybrid",
        "none",
    }
)

# Bit widths that genuinely use word-level bit-packing.
_TRUE_BITPACK_BITS = frozenset(range(2, 9))


class LayerPolicy:
    """Layer-wise quantization policy with default fallback."""

    def __init__(self, default_config: dict[str, Any]):
        """Args:
            default_config: Dict with at least a "mode" key and any extra
                quant hyperparameters (e.g. bits, group_size).
        """
        mode = default_config.get("mode", "")
        if mode not in KNOWN_MODES:
            raise ValueError(f"Unknown default mode: {mode!r}")
        self._default = dict(default_config)
        self._layers: dict[int, dict[str, Any]] = {}

    def set_layer(self, layer_id: int, **kwargs: Any) -> None:
        """Assign a config dict to a specific layer.

        Args:
            layer_id: Non-negative integer layer index.
            **kwargs: Config keys (must include ``mode``).

        Raises:
            ValueError: On invalid layer_id or unknown mode.
        """
        if not isinstance(layer_id, int):
            raise ValueError(f"layer_id must be int, got {type(layer_id).__name__}")
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}")
        mode = kwargs.get("mode", "")
        if mode not in KNOWN_MODES:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._layers[layer_id] = dict(kwargs)

    def get_config(self, layer_id: int) -> dict[str, Any]:
        """Return the quantization config for a given layer ID.

        Falls back to the default config if the layer is not explicitly listed.

        Args:
            layer_id: Integer layer ID.

        Returns:
            Dict with at least a "mode" key.
        """
        if not isinstance(layer_id, int):
            raise ValueError(f"layer_id must be int, got {type(layer_id).__name__}")
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}")
        explicit = self._layers.get(layer_id)
        if explicit is not None:
            return dict(explicit)
        return dict(self._default)


def validate_layer_policy(policy: dict[str, Any]) -> list[str]:
    """Validate a layer policy dict and return a list of error messages."""
    import warnings

    errors: list[str] = []
    if not isinstance(policy, dict):
        return ["Policy must be a dict"]
    default = policy.get("default")
    if default is None:
        errors.append("Policy missing 'default' key")
    elif not isinstance(default, dict):
        errors.append("Policy 'default' must be a dict")
    else:
        mode = default.get("mode", "")
        if mode not in KNOWN_MODES:
            errors.append(f"Unknown default mode: {mode!r}")
    layers = policy.get("layers")
    if layers is not None:
        if not isinstance(layers, dict):
            errors.append("Policy 'layers' must be a dict")
        else:
            for layer_id_str, cfg in layers.items():
                try:
                    layer_id = int(layer_id_str)
                except (ValueError, TypeError):
                    errors.append(f"Invalid layer ID: {layer_id_str!r}")
                    continue
                if layer_id < 0:
                    errors.append(f"Negative layer ID: {layer_id}")
                if not isinstance(cfg, dict):
                    errors.append(f"Layer {layer_id}: config must be a dict")
                    continue
                mode = cfg.get("mode", "")
                if mode not in KNOWN_MODES:
                    errors.append(f"Layer {layer_id}: unknown mode {mode!r}")
    # Warn about configs that claim bit-packing but use >8-bit widths
    for cfg_name, cfg in [("default", default), *(layers or {}).items()]:
        if not isinstance(cfg, dict):
            continue
        bits = cfg.get("bits") or cfg.get("cartesian_bits")
        if bits is not None and bits not in _TRUE_BITPACK_BITS:
            warnings.warn(
                f"{cfg_name}: bits={bits} exceeds 8-bit pack limit; "
                f"falls back to raw uint32 (not truly bit-packed)",
                stacklevel=2,
            )

    return errors


def load_policy(path: str | Path) -> LayerPolicy:
    """Load and validate a layer policy from JSON.

    Args:
        path: Path to the JSON policy file.

    Returns:
        Validated :class:`LayerPolicy` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the policy is structurally invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Layer policy not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Policy JSON must be a dict")
    default = data.get("default")
    if default is None:
        raise ValueError("Policy missing 'default' key")
    if not isinstance(default, dict):
        raise ValueError("Policy 'default' must be a dict")
    policy = LayerPolicy(default_config=default)
    layers = data.get("layers")
    if layers is not None:
        if not isinstance(layers, dict):
            raise ValueError("Policy 'layers' must be a dict")
        for layer_id_str, cfg in layers.items():
            try:
                layer_id = int(layer_id_str)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid layer ID: {layer_id_str!r}")
            if not isinstance(cfg, dict):
                raise ValueError(f"Layer {layer_id}: config must be a dict")
            policy.set_layer(layer_id, **cfg)
    return policy
