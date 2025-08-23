"""Tests for parameter specification and management system."""

import pytest

from logillm.core.parameters import (
    STANDARD_PARAM_SPECS,
    STANDARD_PRESETS,
    ParamConstraint,
    ParamDomain,
    ParameterHistory,
    ParameterTrace,
    ParamPreset,
    ParamSpec,
    ParamType,
    SearchSpace,
)
from logillm.core.providers import MockProvider


class TestParamSpec:
    """Test parameter specifications."""

    def test_param_spec_creation(self):
        """Test creating parameter specs."""
        spec = ParamSpec(
            name="temperature",
            param_type=ParamType.FLOAT,
            domain=ParamDomain.GENERATION,
            description="Controls randomness",
            default=0.7,
            range=(0.0, 1.0),
        )

        assert spec.name == "temperature"
        assert spec.param_type == ParamType.FLOAT
        assert spec.default == 0.7
        assert spec.range == (0.0, 1.0)

    def test_param_validation(self):
        """Test parameter validation."""
        spec = ParamSpec(
            name="temperature",
            param_type=ParamType.FLOAT,
            domain=ParamDomain.GENERATION,
            description="Temperature",
            default=0.7,
            range=(0.0, 1.0),
        )

        # Valid values
        assert spec.validate(0.5)
        assert spec.validate(0.0)
        assert spec.validate(1.0)

        # Invalid values
        assert not spec.validate(-0.1)
        assert not spec.validate(1.1)
        assert not spec.validate("not a number")

    def test_param_sampling(self):
        """Test parameter sampling."""
        spec = ParamSpec(
            name="temperature",
            param_type=ParamType.FLOAT,
            domain=ParamDomain.GENERATION,
            description="Temperature",
            default=0.7,
            range=(0.0, 1.0),
        )

        # Sample should be within range
        for _ in range(10):
            value = spec.sample()
            assert 0.0 <= value <= 1.0

    def test_categorical_param(self):
        """Test categorical parameters."""
        spec = ParamSpec(
            name="mode",
            param_type=ParamType.CATEGORICAL,
            domain=ParamDomain.BEHAVIOR,
            description="Operation mode",
            default="balanced",
            choices=["fast", "balanced", "quality"],
        )

        # Valid choices
        assert spec.validate("fast")
        assert spec.validate("balanced")

        # Invalid choice
        assert not spec.validate("invalid")

        # Sampling
        for _ in range(10):
            value = spec.sample()
            assert value in ["fast", "balanced", "quality"]


class TestParamConstraint:
    """Test parameter constraints."""

    def test_range_constraint(self):
        """Test range constraints."""
        constraint = ParamConstraint(type="range", value=(0.0, 1.0))

        assert constraint.validate(0.5)
        assert constraint.validate(0.0)
        assert constraint.validate(1.0)
        assert not constraint.validate(-0.1)
        assert not constraint.validate(1.1)

    def test_choices_constraint(self):
        """Test choices constraint."""
        constraint = ParamConstraint(type="choices", value=["a", "b", "c"])

        assert constraint.validate("a")
        assert constraint.validate("b")
        assert not constraint.validate("d")

    def test_custom_constraint(self):
        """Test custom validator constraint."""

        def is_even(x):
            return x % 2 == 0

        constraint = ParamConstraint(type="custom", value=None, validator=is_even)

        assert constraint.validate(2)
        assert constraint.validate(4)
        assert not constraint.validate(3)


class TestSearchSpace:
    """Test search space functionality."""

    def test_search_space_creation(self):
        """Test creating search space."""
        param_specs = {
            "temperature": ParamSpec(
                name="temperature",
                param_type=ParamType.FLOAT,
                domain=ParamDomain.GENERATION,
                description="Temperature",
                default=0.7,
                range=(0.0, 1.0),
            ),
            "max_tokens": ParamSpec(
                name="max_tokens",
                param_type=ParamType.INT,
                domain=ParamDomain.EFFICIENCY,
                description="Max tokens",
                default=100,
                range=(10, 1000),
            ),
        }

        space = SearchSpace(param_specs)
        assert len(space.param_specs) == 2
        assert "temperature" in space.param_specs
        assert "max_tokens" in space.param_specs

    def test_fixed_params(self):
        """Test fixing parameters."""
        param_specs = {
            "temperature": STANDARD_PARAM_SPECS["temperature"],
            "top_p": STANDARD_PARAM_SPECS["top_p"],
        }

        space = SearchSpace(param_specs)
        space.fix_param("temperature", 0.5)

        # Sample should always have fixed value
        for _ in range(5):
            config = space.sample()
            assert config["temperature"] == 0.5
            assert "top_p" in config

    def test_conditional_params(self):
        """Test conditional parameters."""
        param_specs = {
            "use_cache": ParamSpec(
                name="use_cache",
                param_type=ParamType.BOOL,
                domain=ParamDomain.EFFICIENCY,
                description="Use cache",
                default=True,
            ),
            "cache_size": ParamSpec(
                name="cache_size",
                param_type=ParamType.INT,
                domain=ParamDomain.EFFICIENCY,
                description="Cache size",
                default=100,
                range=(10, 1000),
            ),
        }

        space = SearchSpace(param_specs)
        # Only include cache_size if use_cache is True
        space.add_conditional("cache_size", lambda config: config.get("use_cache", False))

        # Test sampling
        configs_with_cache = []
        configs_without_cache = []

        for _ in range(20):
            config = space.sample()
            if config.get("use_cache"):
                configs_with_cache.append(config)
            else:
                configs_without_cache.append(config)

        # When use_cache is True, cache_size should be present
        for config in configs_with_cache:
            if config["use_cache"]:
                assert "cache_size" in config

        # When use_cache is False, cache_size should not be present
        for config in configs_without_cache:
            if not config["use_cache"]:
                assert "cache_size" not in config


class TestParameterHistory:
    """Test parameter history tracking."""

    def test_history_tracking(self):
        """Test tracking parameter history."""
        history = ParameterHistory()

        # Add some traces
        trace1 = ParameterTrace(
            module_name="TestModule",
            parameters={"temperature": 0.5, "top_p": 0.9},
            score=0.7,
            timestamp=1.0,
        )
        history.add_trace(trace1)

        trace2 = ParameterTrace(
            module_name="TestModule",
            parameters={"temperature": 0.7, "top_p": 0.8},
            score=0.8,
            timestamp=2.0,
        )
        history.add_trace(trace2)

        trace3 = ParameterTrace(
            module_name="TestModule",
            parameters={"temperature": 0.3, "top_p": 0.95},
            score=0.6,
            timestamp=3.0,
        )
        history.add_trace(trace3)

        # Check best tracking
        assert history.best_score == 0.8
        assert history.best_config["temperature"] == 0.7
        assert history.best_config["top_p"] == 0.8

        # Check trajectory
        temp_trajectory = history.get_trajectory("temperature")
        assert len(temp_trajectory) == 3
        assert temp_trajectory[0] == (1.0, 0.5)
        assert temp_trajectory[1] == (2.0, 0.7)
        assert temp_trajectory[2] == (3.0, 0.3)


class TestProviderIntegration:
    """Test provider parameter integration."""

    def test_provider_param_specs(self):
        """Test provider parameter specifications."""
        provider = MockProvider()

        # Should have standard param specs
        specs = provider.get_param_specs()
        assert "temperature" in specs
        assert "top_p" in specs
        assert "max_tokens" in specs

        # Check spec properties
        temp_spec = specs["temperature"]
        assert temp_spec.param_type == ParamType.FLOAT
        assert temp_spec.range == (0.0, 2.0)

    def test_provider_presets(self):
        """Test provider presets."""
        provider = MockProvider()

        presets = provider.get_param_presets()
        assert ParamPreset.CREATIVE in presets
        assert ParamPreset.BALANCED in presets
        assert ParamPreset.PRECISE in presets

        # Check preset values
        creative = presets[ParamPreset.CREATIVE]
        assert creative["temperature"] > 0.7

        precise = presets[ParamPreset.PRECISE]
        assert precise["temperature"] < 0.3

    def test_provider_param_validation(self):
        """Test provider parameter validation."""
        provider = MockProvider()

        # Valid parameters
        valid_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        is_valid, errors = provider.validate_params(valid_params)
        assert is_valid
        assert len(errors) == 0

        # Invalid parameters
        invalid_params = {
            "temperature": 3.0,  # Out of range
            "top_p": -0.1,  # Out of range
        }
        is_valid, errors = provider.validate_params(invalid_params)
        assert not is_valid
        assert len(errors) == 2


class TestStandardSpecs:
    """Test standard parameter specifications."""

    def test_standard_specs_available(self):
        """Test that standard specs are defined."""
        assert "temperature" in STANDARD_PARAM_SPECS
        assert "top_p" in STANDARD_PARAM_SPECS
        assert "max_tokens" in STANDARD_PARAM_SPECS

        # Check they're proper ParamSpecs
        for name, spec in STANDARD_PARAM_SPECS.items():
            assert isinstance(spec, ParamSpec)
            assert spec.name == name
            assert spec.param_type in ParamType
            assert spec.domain in ParamDomain

    def test_standard_presets_available(self):
        """Test that standard presets are defined."""
        assert ParamPreset.CREATIVE in STANDARD_PRESETS
        assert ParamPreset.BALANCED in STANDARD_PRESETS
        assert ParamPreset.PRECISE in STANDARD_PRESETS

        # Check they contain parameter values
        for _preset, config in STANDARD_PRESETS.items():
            assert "temperature" in config
            assert isinstance(config["temperature"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
