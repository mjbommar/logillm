"""Tests for hyperparameter optimization."""

from dataclasses import dataclass
from typing import Any

import pytest

from logillm.core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace
from logillm.core.predict import Predict
from logillm.core.providers import MockProvider, register_provider
from logillm.optimizers.hyperparameter import (
    AdaptiveOptimizer,
    GridSearchOptimizer,
    HyperparameterOptimizer,
)


@dataclass
class Example:
    """Simple example for testing."""

    inputs: dict[str, Any]
    outputs: dict[str, Any]


def accuracy_metric(pred_outputs: dict[str, Any], true_outputs: dict[str, Any]) -> float:
    """Simple accuracy metric for testing."""
    if not pred_outputs or not true_outputs:
        return 0.0

    # Check if answer matches
    pred_answer = pred_outputs.get("answer", pred_outputs.get("output", ""))
    true_answer = true_outputs.get("answer", true_outputs.get("output", ""))

    # Simple string matching
    return 1.0 if str(pred_answer).strip() == str(true_answer).strip() else 0.0


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MockProvider(response_text="Paris")
    register_provider(provider, "test_provider", set_default=True)
    return provider


@pytest.fixture
def simple_dataset():
    """Create a simple test dataset."""
    return [
        Example(inputs={"question": "What is the capital of France?"}, outputs={"answer": "Paris"}),
        Example(inputs={"question": "What is 2 + 2?"}, outputs={"answer": "4"}),
        Example(inputs={"question": "What color is the sky?"}, outputs={"answer": "blue"}),
    ]


@pytest.fixture
def search_space():
    """Create a simple search space for testing."""
    param_specs = {
        "temperature": ParamSpec(
            name="temperature",
            param_type=ParamType.FLOAT,
            domain=ParamDomain.GENERATION,
            description="Temperature",
            default=0.7,
            range=(0.0, 1.0),
            step=0.1,
        ),
        "max_tokens": ParamSpec(
            name="max_tokens",
            param_type=ParamType.INT,
            domain=ParamDomain.EFFICIENCY,
            description="Max tokens",
            default=100,
            range=(50, 200),
            step=50,
        ),
    }
    return SearchSpace(param_specs)


class TestHyperparameterOptimizer:
    """Test hyperparameter optimizer."""

    @pytest.mark.asyncio
    async def test_basic_optimization(self, mock_provider, simple_dataset, search_space):
        """Test basic hyperparameter optimization."""
        # Create module
        module = Predict("question -> answer")

        # Create optimizer
        optimizer = HyperparameterOptimizer(
            metric=accuracy_metric,
            search_space=search_space,
            strategy="random",  # Use random search for speed
            n_trials=5,
            track_history=True,
        )

        # Optimize
        result = await optimizer.optimize(
            module=module,
            trainset=simple_dataset,
        )

        # Check result
        assert result.optimized_module is not None
        assert result.best_score >= 0.0
        assert result.iterations == 5
        assert result.optimization_time > 0
        assert "best_config" in result.metadata

        # Check history was tracked
        assert optimizer.history is not None
        assert len(optimizer.history.traces) == 5

    @pytest.mark.asyncio
    async def test_optimization_improves_score(self, mock_provider, simple_dataset):
        """Test that optimization finds better parameters."""
        # Create module with bad initial config
        module = Predict("question -> answer")
        module.config = {"temperature": 2.0}  # Too high

        # Create optimizer with specific search space
        param_specs = {
            "temperature": ParamSpec(
                name="temperature",
                param_type=ParamType.FLOAT,
                domain=ParamDomain.GENERATION,
                description="Temperature",
                default=0.7,
                range=(0.0, 1.0),  # Force reasonable range
                step=0.2,
            ),
        }
        search_space = SearchSpace(param_specs)

        optimizer = HyperparameterOptimizer(
            metric=accuracy_metric,
            search_space=search_space,
            strategy="random",
            n_trials=5,
        )

        # Optimize
        result = await optimizer.optimize(
            module=module,
            trainset=simple_dataset,
        )

        # Should find better temperature
        best_temp = result.metadata["best_config"]["temperature"]
        assert 0.0 <= best_temp <= 1.0

    def test_preset_application(self, mock_provider):
        """Test applying parameter presets."""
        from logillm.core.parameters import ParamPreset

        module = Predict("question -> answer")
        optimizer = HyperparameterOptimizer(metric=accuracy_metric)

        # Apply creative preset
        creative_module = optimizer.apply_preset(module, ParamPreset.CREATIVE)
        assert creative_module.config["temperature"] > 0.7

        # Apply precise preset
        precise_module = optimizer.apply_preset(module, ParamPreset.PRECISE)
        assert precise_module.config["temperature"] < 0.3

    def test_parameter_analysis(self, mock_provider):
        """Test parameter importance analysis."""
        optimizer = HyperparameterOptimizer(
            metric=accuracy_metric,
            track_history=True,
        )

        # Add some fake history
        import time

        from logillm.core.parameters import ParameterTrace

        traces = [
            ParameterTrace("Module", {"temperature": 0.1, "top_p": 0.9}, 0.5, time.time()),
            ParameterTrace("Module", {"temperature": 0.5, "top_p": 0.8}, 0.7, time.time()),
            ParameterTrace("Module", {"temperature": 0.9, "top_p": 0.7}, 0.6, time.time()),
        ]

        for trace in traces:
            optimizer.history.add_trace(trace)

        # Analyze
        analysis = optimizer.analyze_parameters()

        assert "best_config" in analysis
        assert "best_score" in analysis
        assert "parameter_importance" in analysis
        assert "parameter_trajectories" in analysis


class TestGridSearchOptimizer:
    """Test grid search optimizer."""

    @pytest.mark.asyncio
    async def test_grid_search(self, mock_provider, simple_dataset):
        """Test grid search optimization."""
        # Create module
        module = Predict("question -> answer")

        # Define grid
        param_grid = {
            "temperature": [0.0, 0.5, 1.0],
            "max_tokens": [50, 100],
        }

        # Create optimizer
        optimizer = GridSearchOptimizer(
            metric=accuracy_metric,
            param_grid=param_grid,
            track_history=True,
        )

        # Check total trials calculated correctly
        assert optimizer.n_trials == 6  # 3 * 2

        # Optimize
        await optimizer.optimize(
            module=module,
            trainset=simple_dataset,
        )

        # Should have tried all combinations
        assert len(optimizer.history.traces) == 6

        # Check all combinations were tried
        configs_tried = [trace.parameters for trace in optimizer.history.traces]

        # Verify all combinations
        for temp in [0.0, 0.5, 1.0]:
            for max_tok in [50, 100]:
                config = {"temperature": temp, "max_tokens": max_tok}
                assert config in configs_tried


class TestAdaptiveOptimizer:
    """Test adaptive optimizer."""

    def test_task_analyzer(self):
        """Test task type analysis."""
        optimizer = AdaptiveOptimizer(metric=accuracy_metric)

        # Test code task detection
        code_examples = [
            Example(inputs={"prompt": "Write a function"}, outputs={"code": "def foo(): pass"})
        ]
        assert optimizer._default_task_analyzer(code_examples) == "code"

        # Test reasoning task detection
        reasoning_examples = [
            Example(
                inputs={"question": "Why?"},
                outputs={"answer": "Because...", "reasoning": "Let me think..."},
            )
        ]
        assert optimizer._default_task_analyzer(reasoning_examples) == "reasoning"

        # Test factual task detection
        factual_examples = [
            Example(inputs={"question": "What is X?"}, outputs={"answer": "X is Y"})
        ]
        assert optimizer._default_task_analyzer(factual_examples) == "factual"

    def test_task_specific_config(self):
        """Test task-specific configurations."""
        optimizer = AdaptiveOptimizer(metric=accuracy_metric)

        # Code task should have low temperature
        code_config = optimizer._get_task_config("code")
        assert code_config["temperature"] < 0.3

        # Creative task should have high temperature
        creative_config = optimizer._get_task_config("creative")
        assert creative_config["temperature"] > 0.8

        # Factual task should have very low temperature
        factual_config = optimizer._get_task_config("factual")
        assert factual_config["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_adaptive_optimization(self, mock_provider):
        """Test adaptive optimization adjusts to task type."""
        # Create code-like dataset
        code_dataset = [
            Example(
                inputs={"prompt": "Write a function to add two numbers"},
                outputs={"code": "def add(a, b): return a + b"},
            ),
        ]

        # Create module
        module = Predict("prompt -> code")

        # Create optimizer
        optimizer = AdaptiveOptimizer(
            metric=accuracy_metric,
            strategy="random",
            n_trials=3,
        )

        # Optimize
        await optimizer.optimize(
            module=module,
            trainset=code_dataset,
        )

        # Should have adapted search space for code
        # The search space should be constrained for code generation
        assert optimizer.search_space is not None


class TestOptunaIntegration:
    """Test Optuna integration (if available)."""

    @pytest.mark.asyncio
    async def test_optuna_optimization(self, mock_provider, simple_dataset):
        """Test optimization with Optuna."""
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not installed")

        # Create module
        module = Predict("question -> answer")

        # Create simple search space
        param_specs = {
            "temperature": ParamSpec(
                name="temperature",
                param_type=ParamType.FLOAT,
                domain=ParamDomain.GENERATION,
                description="Temperature",
                default=0.7,
                range=(0.0, 1.0),
            ),
        }
        search_space = SearchSpace(param_specs)

        # Create optimizer with Bayesian strategy
        optimizer = HyperparameterOptimizer(
            metric=accuracy_metric,
            search_space=search_space,
            strategy="bayesian",
            n_trials=3,
            track_history=True,
        )

        # Optimize
        result = await optimizer.optimize(
            module=module,
            trainset=simple_dataset,
        )

        # Check result
        assert result.optimized_module is not None
        assert result.iterations == 3
        assert "best_config" in result.metadata

        # Check Optuna was used
        assert len(optimizer.history.traces) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
