"""Integration tests for hyperparameter handling in optimizers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from logillm.core.config_utils import ensure_config, get_hyperparameter
from logillm.core.hyperparameters import HyperparameterConfig
from logillm.core.modules import Module
from logillm.optimizers.hybrid_optimizer import HybridOptimizer
from logillm.optimizers.reflective_evolution import ReflectiveEvolutionOptimizer


@pytest.mark.integration
class TestOptimizerHyperparameterHandling:
    """Test that optimizers properly handle hyperparameters."""

    def test_reflective_evolution_preserves_hyperparameters(self):
        """Test that ReflectiveEvolution properly handles hyperparameter configs."""
        # Create a mock module with hyperparameter config
        module = MagicMock(spec=Module)
        module.config = HyperparameterConfig({"temperature": 0.8, "top_p": 0.95})

        # Create optimizer
        metric = MagicMock()
        ReflectiveEvolutionOptimizer(
            metric=metric, include_hyperparameters=True, n_iterations=1
        )

        # Verify module config is preserved and accessible
        assert get_hyperparameter(module, "temperature") == 0.8
        assert get_hyperparameter(module, "top_p") == 0.95

    @pytest.mark.asyncio
    async def test_reflective_evolution_improvements(self):
        """Test that ReflectiveEvolution can apply hyperparameter improvements."""
        # Create a mock module
        module = MagicMock(spec=Module)
        module.config = {}
        ensure_config(module)

        # Create optimizer
        metric = MagicMock()
        optimizer = ReflectiveEvolutionOptimizer(
            metric=metric, include_hyperparameters=True, n_iterations=1
        )

        # Test applying improvements
        improvements = {"temperature": 0.2, "top_p": -0.1}
        improved = await optimizer._apply_improvements(module, improvements, [])

        # Check that hyperparameters were adjusted properly
        assert abs(get_hyperparameter(improved, "temperature") - 0.9) < 0.001  # 0.7 default + 0.2
        assert abs(get_hyperparameter(improved, "top_p") - 0.9) < 0.001  # 1.0 default - 0.1

    @pytest.mark.asyncio
    async def test_reflective_evolution_merging(self):
        """Test that ReflectiveEvolution properly merges hyperparameters."""
        # Create mock modules with different configs
        module1 = MagicMock(spec=Module)
        module1.config = HyperparameterConfig({"temperature": 0.6})

        module2 = MagicMock(spec=Module)
        module2.config = HyperparameterConfig({"temperature": 1.0})

        module3 = MagicMock(spec=Module)
        module3.config = HyperparameterConfig({"temperature": 0.8})

        # Create optimizer
        metric = MagicMock()
        optimizer = ReflectiveEvolutionOptimizer(metric=metric)

        # Merge candidates
        merged = await optimizer._merge_candidates([module1, module2, module3])

        # Check averaged temperature
        expected_temp = (0.6 + 1.0 + 0.8) / 3
        assert abs(get_hyperparameter(merged, "temperature") - expected_temp) < 0.01

    def test_hybrid_optimizer_config_handling(self):
        """Test that HybridOptimizer properly applies configurations."""
        # Create a mock module
        module = MagicMock(spec=Module)
        module.config = {}

        # Create optimizer
        metric = MagicMock()
        optimizer = HybridOptimizer(metric=metric, strategy="alternating")

        # Mock the _apply_joint_config to test it's called correctly
        optimizer._apply_joint_config = AsyncMock(return_value=module)

        # Verify module gets config initialized
        ensure_config(module)
        assert isinstance(module.config, HyperparameterConfig)

    @pytest.mark.asyncio
    async def test_hybrid_optimizer_applies_hyperparameters(self):
        """Test that HybridOptimizer applies hyperparameter configs correctly."""

        # Create a base module
        module = MagicMock(spec=Module)
        module.config = {}
        module.signature = MagicMock()
        module.parameters = {}

        # Make deepcopy work with mock
        module.__deepcopy__ = lambda memo: MagicMock(
            spec=Module, config={}, signature=MagicMock(), parameters={}
        )

        # Create optimizer
        metric = MagicMock()
        optimizer = HybridOptimizer(metric=metric)

        # Apply configuration
        config = {
            "temperature": 1.2,
            "top_p": 0.85,
            "max_tokens": 200,
            "num_demos": 0,  # Skip demo optimization for this test
        }

        result = await optimizer._apply_joint_config(module, config, [])

        # Verify hyperparameters were applied
        assert get_hyperparameter(result, "temperature") == 1.2
        assert get_hyperparameter(result, "top_p") == 0.85
        assert get_hyperparameter(result, "max_tokens") == 200

    def test_config_coercion_and_validation(self):
        """Test that configs properly coerce and validate values."""
        module = MagicMock(spec=Module)
        module.config = HyperparameterConfig()

        # Test temperature clamping
        module.config["temperature"] = 3.0
        assert module.config["temperature"] == 2.0  # Clamped to max

        module.config["temperature"] = -0.5
        assert module.config["temperature"] == 0.0  # Clamped to min

        # Test type coercion
        module.config["max_tokens"] = "150"
        assert module.config["max_tokens"] == 150
        assert isinstance(module.config["max_tokens"], int)

        # Test top_p validation
        module.config["top_p"] = 1.5
        assert module.config["top_p"] == 1.0  # Clamped to max

    def test_provider_param_conversion(self):
        """Test conversion to provider-specific parameters."""
        config = HyperparameterConfig({"temperature": 0.7, "max_tokens": 100, "stop": ["\n\n"]})

        # Test OpenAI conversion
        openai_params = config.to_provider_params("openai")
        assert "max_completion_tokens" in openai_params
        assert openai_params["max_completion_tokens"] == 100
        assert "max_tokens" not in openai_params

        # Test Anthropic conversion
        anthropic_params = config.to_provider_params("anthropic")
        assert "stop_sequences" in anthropic_params
        assert anthropic_params["stop_sequences"] == ["\n\n"]
        assert "stop" not in anthropic_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
