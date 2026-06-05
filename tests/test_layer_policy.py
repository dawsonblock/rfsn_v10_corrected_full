#!/usr/bin/env python3
"""
RFSN v10 — Layer policy acceptance tests.

Verifies that LayerPolicy falls back to defaults, validates inputs,
produces deterministic results, and loads correctly from JSON.
"""
from __future__ import annotations

import json

import pytest

from rfsn_v10.quantization.layer_policy import LayerPolicy, load_policy


# ------------------------------------------------------------------
# LayerPolicy tests
# ------------------------------------------------------------------

class TestLayerPolicy:
    def test_missing_layer_falls_back_to_default(self):
        """Layer not explicitly in policy should return the default config."""
        default = {"mode": "cartesian", "bits": 4, "group_size": 64}
        policy = LayerPolicy(default_config=default)
        policy.set_layer(0, mode="polar", bits=8, group_size=32)

        # Layer 0 is registered
        cfg_0 = policy.get_config(0)
        assert cfg_0["mode"] == "polar"
        assert cfg_0["bits"] == 8
        assert cfg_0["group_size"] == 32

        # Layer 99 is not registered — should fall back
        cfg_99 = policy.get_config(99)
        assert cfg_99["mode"] == "cartesian"
        assert cfg_99["bits"] == 4
        assert cfg_99["group_size"] == 64

    def test_invalid_layer_rejected(self):
        """Non-integer layer IDs should raise ValueError."""
        policy = LayerPolicy(default_config={"mode": "cartesian", "bits": 4})

        with pytest.raises(ValueError, match="layer_id"):
            policy.get_config("not_an_int")

        with pytest.raises(ValueError, match="layer_id"):
            policy.set_layer("not_an_int", mode="polar", bits=8)

        with pytest.raises(ValueError, match="layer_id"):
            policy.get_config(-1)

    def test_invalid_mode_rejected(self):
        """Unknown quantization modes should raise ValueError."""
        policy = LayerPolicy(default_config={"mode": "cartesian", "bits": 4})

        with pytest.raises(ValueError, match="mode"):
            policy.set_layer(0, mode="unknown_mode", bits=8)

        with pytest.raises(ValueError, match="mode"):
            policy.set_layer(1, mode="", bits=4)

    def test_policy_produces_deterministic_config(self):
        """Same layer ID should always return the same config object."""
        policy = LayerPolicy(default_config={"mode": "cartesian", "bits": 4, "group_size": 64})
        policy.set_layer(2, mode="hybrid", bits=6, group_size=128)

        cfg_a = policy.get_config(2)
        cfg_b = policy.get_config(2)
        assert cfg_a == cfg_b
        assert cfg_a is not None
        assert cfg_a["mode"] == "hybrid"
        assert cfg_a["bits"] == 6

    def test_load_valid_policy(self, tmp_path):
        """Valid JSON policy should load correctly."""
        data = {
            "default": {"mode": "cartesian", "bits": 4, "group_size": 64},
            "layers": {
                "0": {"mode": "polar", "bits": 8, "group_size": 32},
                "1": {"mode": "hybrid", "bits": 6, "group_size": 64},
            },
        }
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(data))

        policy = load_policy(str(path))
        assert policy.get_config(0)["mode"] == "polar"
        assert policy.get_config(1)["bits"] == 6
        assert policy.get_config(99)["mode"] == "cartesian"

    def test_load_invalid_policy_raises(self, tmp_path):
        """Invalid policy structure should raise ValueError."""
        # Missing "default" key
        bad_data = {"layers": {"0": {"mode": "polar", "bits": 8}}}
        path = tmp_path / "bad_policy.json"
        path.write_text(json.dumps(bad_data))

        with pytest.raises(ValueError, match="default"):
            load_policy(str(path))

        # Non-dict "layers" value
        bad_data_2 = {"default": {"mode": "cartesian"}, "layers": "not_a_dict"}
        path_2 = tmp_path / "bad_policy2.json"
        path_2.write_text(json.dumps(bad_data_2))

        with pytest.raises(ValueError, match="layers"):
            load_policy(str(path_2))
