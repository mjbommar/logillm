"""Test hyperparameter configuration system."""

import pytest

from logillm.core.config_utils import (
    ensure_config,
    get_hyperparameter,
    set_hyperparameter,
    update_config,
)
from logillm.core.hyperparameters import (
    HyperparameterConfig,
    ensure_hyperparameter_config,
    merge_configs,
)


class TestHyperparameterConfig:
    """Test HyperparameterConfig class."""

    def test_initialization_with_defaults(self):
        """Test config initializes with default values."""
        config = HyperparameterConfig()
        assert config["temperature"] == 0.7
        assert config["top_p"] == 1.0
        assert config["max_tokens"] == 150

    def test_initialization_with_values(self):
        """Test config initializes with provided values."""
        config = HyperparameterConfig({"temperature": 1.5, "top_p": 0.8})
        assert config["temperature"] == 1.5
        assert config["top_p"] == 0.8

    def test_dict_like_access(self):
        """Test dict-like access patterns."""
        config = HyperparameterConfig()

        # Set with bracket notation
        config["temperature"] = 1.2
        assert config["temperature"] == 1.2

        # Get with get method
        assert config.get("temperature") == 1.2
        assert config.get("nonexistent", "default") == "default"

        # Contains check
        assert "temperature" in config
        assert "nonexistent" not in config

    def test_attribute_access(self):
        """Test attribute-like access patterns."""
        config = HyperparameterConfig()

        # Set with attribute notation
        config.temperature = 0.5
        assert config.temperature == 0.5

        # Get with attribute notation
        assert config.top_p == 1.0

    def test_validation_and_clamping(self):
        """Test parameter validation and clamping."""
        config = HyperparameterConfig()

        # Temperature should be clamped to valid range
        config["temperature"] = -1.0
        assert config["temperature"] == 0.0  # Clamped to min

        config["temperature"] = 3.0
        assert config["temperature"] == 2.0  # Clamped to max

        # Top-p should be clamped
        config["top_p"] = 1.5
        assert config["top_p"] == 1.0

        config["top_p"] = -0.1
        assert config["top_p"] == 0.0

    def test_type_coercion(self):
        """Test automatic type coercion."""
        config = HyperparameterConfig()

        # Float coercion
        config["temperature"] = "0.8"
        assert config["temperature"] == 0.8
        assert isinstance(config["temperature"], float)

        # Int coercion
        config["max_tokens"] = 100.5
        assert config["max_tokens"] == 100
        assert isinstance(config["max_tokens"], int)

    def test_update_method(self):
        """Test batch updates."""
        config = HyperparameterConfig()
        config.update({"temperature": 1.0, "top_p": 0.9, "max_tokens": 200})
        assert config["temperature"] == 1.0
        assert config["top_p"] == 0.9
        assert config["max_tokens"] == 200

    def test_copy_method(self):
        """Test copying configuration."""
        config = HyperparameterConfig({"temperature": 1.5})
        copy = config.copy()
        assert copy["temperature"] == 1.5
        assert isinstance(copy, dict)

    def test_adjust_method(self):
        """Test parameter adjustment for optimization."""
        config = HyperparameterConfig({"temperature": 0.7})
        config.adjust("temperature", 0.2)
        assert abs(config["temperature"] - 0.9) < 0.001  # Allow for floating point precision

        # Should still respect bounds
        config.adjust("temperature", 1.5)
        assert config["temperature"] == 2.0  # Clamped to max

    def test_interpolation(self):
        """Test configuration interpolation."""
        config1 = HyperparameterConfig({"temperature": 0.5, "top_p": 0.8})
        config2 = HyperparameterConfig({"temperature": 1.5, "top_p": 1.0})

        # Interpolate halfway
        interpolated = config1.interpolate(config2, weight=0.5)
        assert interpolated["temperature"] == 1.0
        assert interpolated["top_p"] == 0.9

    def test_provider_params_conversion(self):
        """Test conversion to provider-specific parameters."""
        config = HyperparameterConfig({"temperature": 0.7, "max_tokens": 100, "stop": ["\n\n"]})

        # OpenAI conversion
        openai_params = config.to_provider_params("openai")
        assert openai_params["temperature"] == 0.7
        assert "max_completion_tokens" in openai_params
        assert openai_params["max_completion_tokens"] == 100

        # Anthropic conversion
        anthropic_params = config.to_provider_params("anthropic")
        assert anthropic_params["temperature"] == 0.7
        assert "stop_sequences" in anthropic_params

    def test_change_history(self):
        """Test parameter change tracking."""
        config = HyperparameterConfig()
        config["temperature"] = 0.8
        config["temperature"] = 0.9

        changes = config.get_changes()
        assert len(changes) >= 2
        assert changes[-1]["param"] == "temperature"
        assert changes[-1]["new"] == 0.9


class TestConfigUtils:
    """Test config utility functions with hyperparameter configs."""

    def test_ensure_config(self):
        """Test ensure_config creates HyperparameterConfig."""

        class MockModule:
            pass

        module = MockModule()
        ensure_config(module)
        assert hasattr(module, "config")
        assert isinstance(module.config, HyperparameterConfig)

    def test_ensure_config_converts_dict(self):
        """Test ensure_config converts dict to HyperparameterConfig."""

        class MockModule:
            def __init__(self):
                self.config = {"temperature": 0.5}

        module = MockModule()
        ensure_config(module)
        assert isinstance(module.config, HyperparameterConfig)
        assert module.config["temperature"] == 0.5

    def test_set_get_hyperparameter(self):
        """Test hyperparameter setting and getting."""

        class MockModule:
            pass

        module = MockModule()

        # Set creates config if needed
        set_hyperparameter(module, "temperature", 0.8)
        assert get_hyperparameter(module, "temperature") == 0.8

        # Update existing
        set_hyperparameter(module, "temperature", 0.9)
        assert get_hyperparameter(module, "temperature") == 0.9

    def test_update_config(self):
        """Test batch config updates."""

        class MockModule:
            pass

        module = MockModule()
        update_config(module, {"temperature": 1.2, "top_p": 0.95, "max_tokens": 250})

        assert get_hyperparameter(module, "temperature") == 1.2
        assert get_hyperparameter(module, "top_p") == 0.95
        assert get_hyperparameter(module, "max_tokens") == 250


class TestUtilityFunctions:
    """Test utility functions."""

    def test_ensure_hyperparameter_config(self):
        """Test ensuring config is HyperparameterConfig."""
        # Already a HyperparameterConfig
        config = HyperparameterConfig()
        assert ensure_hyperparameter_config(config) is config

        # Dict conversion
        dict_config = {"temperature": 0.7}
        result = ensure_hyperparameter_config(dict_config)
        assert isinstance(result, HyperparameterConfig)
        assert result["temperature"] == 0.7

        # None creates empty config
        result = ensure_hyperparameter_config(None)
        assert isinstance(result, HyperparameterConfig)

    def test_merge_configs(self):
        """Test merging multiple configurations."""
        config1 = {"temperature": 0.5}
        config2 = HyperparameterConfig({"top_p": 0.9})
        config3 = {"temperature": 0.8, "max_tokens": 200}

        merged = merge_configs(config1, config2, config3)
        assert isinstance(merged, HyperparameterConfig)
        assert merged["temperature"] == 0.8  # Later takes precedence
        assert merged["top_p"] == 0.9
        assert merged["max_tokens"] == 200


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_strict_mode(self):
        """Test strict mode validation."""
        config = HyperparameterConfig(strict=True)

        # Should raise on invalid parameter in strict mode
        with pytest.raises(ValueError):
            config["invalid_param"] = 123

    def test_unknown_parameters(self):
        """Test handling of unknown parameters."""
        config = HyperparameterConfig(strict=False)

        # Should allow unknown parameters in non-strict mode
        config["custom_param"] = "value"
        assert config["custom_param"] == "value"

    def test_none_values(self):
        """Test handling of None values."""
        config = HyperparameterConfig()
        config["seed"] = None
        assert config["seed"] is None  # None is valid for seed

    def test_complex_stop_sequences(self):
        """Test handling of stop sequences."""
        config = HyperparameterConfig()

        # String stop sequence
        config["stop"] = "\n\n"
        assert config["stop"] == "\n\n"

        # List of stop sequences
        config["stop"] = ["\n\n", "END"]
        assert config["stop"] == ["\n\n", "END"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
