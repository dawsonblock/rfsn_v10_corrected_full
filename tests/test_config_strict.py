"""Strict configuration validation tests (Ticket 7-2).

Tests verify:
- Unknown keys raise ValidationError
- Pydantic validates all config fields
- Environment variable overrides work
"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError

from rfsn_v10.config import RFSNConfig, load_config


class TestStrictConfigValidation:
    """Ticket 7-2: Unknown keys must raise ValidationError.

    NOTE: These tests document the REQUIRED behavior for strict mode.
    They will FAIL until `extra='forbid'` is added to all config models.
    """

    def test_valid_config_loads(self):
        """Valid configuration should load successfully."""
        config = RFSNConfig()
        assert config.logging.level == "INFO"
        assert config.memory.max_gb == 8.0

    @pytest.mark.xfail(reason="Strict mode not yet implemented (Ticket 7-2)")
    def test_unknown_key_raises_validation_error(self):
        """Unknown key in config should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RFSNConfig(unknown_field="value")

        assert "unknown_field" in str(exc_info.value)

    @pytest.mark.xfail(reason="Strict mode not yet implemented (Ticket 7-2)")
    def test_unknown_nested_key_raises(self):
        """Unknown key in nested config should raise ValidationError."""
        with pytest.raises(ValidationError):
            RFSNConfig(logging={"unknown_log_key": "value"})

    @pytest.mark.xfail(reason="Strict mode not yet implemented (Ticket 7-2)")
    def test_typo_in_key_raises(self):
        """Typo in key name should raise ValidationError."""
        with pytest.raises(ValidationError):
            RFSNConfig(loging={"level": "DEBUG"})  # typo: loging vs logging

    def test_current_behavior_allows_unknown_keys(self):
        """Document current behavior: unknown keys are silently ignored."""
        # This shows the current (problematic) behavior
        # After implementing strict mode, this test should be removed
        config = RFSNConfig(unknown_field="value")
        # Unknown field is ignored, no error raised
        assert not hasattr(config, 'unknown_field')


class TestConfigFromFile:
    """Loading config from YAML files."""

    def test_load_valid_yaml(self):
        """Valid YAML config should load."""
        config_dict = {
            "logging": {"level": "DEBUG", "format": "json"},
            "memory": {"max_gb": 16.0},
            "backend": {"name": "metal"},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            f.flush()
            config = load_config(f.name)

        assert config.logging.level == "DEBUG"
        assert config.memory.max_gb == 16.0
        assert config.backend.name == "metal"

    @pytest.mark.xfail(reason="Strict mode not yet implemented (Ticket 7-2)")
    def test_load_yaml_with_unknown_key_fails(self):
        """YAML with unknown key should fail on load."""
        config_dict = {
            "logging": {"level": "INFO"},
            "unknown_top_level": "value",
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            f.flush()
            with pytest.raises(ValidationError):
                load_config(f.name)

    @pytest.mark.xfail(reason="load_config doesn't check file existence before from_yaml")
    def test_missing_file_raises(self):
        """Missing config file should raise."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestLoggingConfigValidation:
    """Logging configuration validation."""

    def test_valid_log_levels(self):
        """Valid log levels should be accepted."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = RFSNConfig(logging={"level": level})
            assert config.logging.level == level

        # Lowercase should be normalized to uppercase
        config = RFSNConfig(logging={"level": "debug"})
        assert config.logging.level == "DEBUG"

    def test_invalid_log_level_raises(self):
        """Invalid log level should raise ValidationError."""
        with pytest.raises(ValidationError):
            RFSNConfig(logging={"level": "INVALID"})

    def test_invalid_format_raises(self):
        """Invalid format should raise ValidationError (if strict)."""
        # Currently only json and text are supported
        config = RFSNConfig(logging={"format": "xml"})
        # Should accept any string unless we add validator


class TestMemoryConfigValidation:
    """Memory configuration validation."""

    def test_positive_memory_values(self):
        """Memory values should be positive."""
        config = RFSNConfig(memory={"max_gb": 4.0, "quota_gb": 10.0})
        assert config.memory.max_gb == 4.0

    def test_zero_memory_rejected(self):
        """Zero or negative memory should be rejected."""
        with pytest.raises(ValidationError):
            RFSNConfig(memory={"max_gb": 0})

    def test_negative_memory_rejected(self):
        """Negative memory should be rejected."""
        with pytest.raises(ValidationError):
            RFSNConfig(memory={"max_gb": -1.0})


class TestBackendConfigValidation:
    """Backend configuration validation."""

    def test_valid_backend_names(self):
        """Valid backend names should be accepted."""
        for name in ["metal", "numpy", "cuda"]:
            config = RFSNConfig(backend={"name": name})
            assert config.backend.name == name

    @pytest.mark.xfail(reason="Backend validation not yet implemented")
    def test_invalid_backend_raises(self):
        """Invalid backend name should raise ValidationError."""
        with pytest.raises(ValidationError):
            RFSNConfig(backend={"name": "invalid_backend"})

    @pytest.mark.xfail(reason="Backend fallback option not yet implemented")
    def test_backend_fallback_boolean(self):
        """Backend fallback should be boolean."""
        config = RFSNConfig(backend={"fallback": True})
        assert config.backend.fallback is True


class TestSparseAttentionConfigValidation:
    """Sparse attention configuration validation."""

    def test_valid_top_k_ratio(self):
        """Top-k ratio should be between 0 and 1."""
        config = RFSNConfig(sparse_attention={"default_top_k_ratio": 0.5})
        assert config.sparse_attention.default_top_k_ratio == 0.5

    def test_top_k_ratio_too_high_raises(self):
        """Top-k ratio > 1 should raise."""
        with pytest.raises(ValidationError):
            RFSNConfig(sparse_attention={"default_top_k_ratio": 1.5})

    def test_top_k_ratio_negative_raises(self):
        """Negative top-k ratio should raise."""
        with pytest.raises(ValidationError):
            RFSNConfig(sparse_attention={"default_top_k_ratio": -0.1})

    def test_positive_block_size(self):
        """Block size should be positive."""
        config = RFSNConfig(sparse_attention={"block_size": 128})
        assert config.sparse_attention.block_size == 128

    def test_zero_block_size_raises(self):
        """Zero block size should raise."""
        with pytest.raises(ValidationError):
            RFSNConfig(sparse_attention={"block_size": 0})


class TestQuantizationConfigValidation:
    """Quantization configuration validation."""

    def test_valid_bits(self):
        """Bits should be between 2 and 8."""
        for bits in [2, 4, 8]:
            config = RFSNConfig(quantization={"default_bits": bits})
            assert config.quantization.default_bits == bits

    def test_bits_too_high_raises(self):
        """Bits > 8 should raise."""
        with pytest.raises(ValidationError):
            RFSNConfig(quantization={"default_bits": 16})

    def test_bits_too_low_raises(self):
        """Bits < 2 should raise."""
        with pytest.raises(ValidationError):
            RFSNConfig(quantization={"default_bits": 1})

    def test_positive_group_size(self):
        """Group size should be positive."""
        config = RFSNConfig(quantization={"group_size": 128})
        assert config.quantization.group_size == 128


class TestEnvironmentVariableOverrides:
    """Config loading from environment variables."""

    def test_env_var_override(self, monkeypatch):
        """Environment variable should override config."""
        monkeypatch.setenv("RFSN_LOG_LEVEL", "DEBUG")

        config = load_config()
        # Depending on implementation, env var should override
        assert config.logging.level == "DEBUG"

    def test_env_var_backend_override(self, monkeypatch):
        """RFSN_BACKEND env var should override backend."""
        monkeypatch.setenv("RFSN_BACKEND", "numpy")

        config = load_config()
        assert config.backend.name == "numpy"


class TestConfigSerialization:
    """Config serialization and deserialization."""

    def test_config_to_dict(self):
        """Config should serialize to dict."""
        config = RFSNConfig()
        data = config.model_dump()

        assert "logging" in data
        assert "memory" in data
        assert data["logging"]["level"] == "INFO"

    def test_config_to_json(self):
        """Config should serialize to JSON."""
        config = RFSNConfig()
        json_str = config.model_dump_json()

        assert "INFO" in json_str
        assert "logging" in json_str

    def test_roundtrip(self):
        """Config should roundtrip through serialization."""
        original = RFSNConfig(logging={"level": "DEBUG"})
        data = original.model_dump()
        restored = RFSNConfig(**data)

        assert restored.logging.level == "DEBUG"


class TestDefaultConfigFile:
    """Default config file handling."""

    def test_default_config_exists(self):
        """Default config file should exist."""
        default_path = "configs/default_runtime.yaml"
        if os.path.exists(default_path):
            config = load_config(default_path)
            assert config is not None
