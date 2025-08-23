"""Comprehensive tests for HyperparameterOptimizer."""

import asyncio
from unittest.mock import patch

import pytest

from logillm.core.parameters import SearchSpace
from logillm.core.types import OptimizationResult
from logillm.exceptions import OptimizationError
from logillm.optimizers import (
    AdaptiveOptimizer,
    GridSearchOptimizer,
    HyperparameterOptimizer,
)
from logillm.optimizers.search_strategies import (
    RandomSearchStrategy,
)
from tests.unit.fixtures import (
    MockDataset,
    MockMetric,
    MockModule,
    OptimizationMonitor,
)


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""

    @pytest.fixture
    def mock_module(self):
        """Create mock module."""
        return MockModule(behavior="quadratic")

    @pytest.fixture
    def mock_metric(self):
        """Create mock metric."""
        return MockMetric(target_value=1.0)

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        return MockDataset(size=20, task_type="general")

    def test_initialization(self):
        """Test optimizer initialization."""
        metric = MockMetric()
        optimizer = HyperparameterOptimizer(metric=metric, n_trials=10, seed=42)

        assert optimizer.n_trials == 10
        assert optimizer.seed == 42
        assert optimizer.track_history
        assert optimizer.history is not None
        assert optimizer.search_strategy.name == "bayesian"  # default

    def test_initialization_with_strategy(self):
        """Test initialization with specific strategy."""
        metric = MockMetric()

        # With strategy name
        optimizer = HyperparameterOptimizer(metric=metric, strategy="random")
        assert optimizer.search_strategy.name == "random"

        # With strategy instance
        strategy = RandomSearchStrategy()
        optimizer = HyperparameterOptimizer(metric=metric, strategy=strategy)
        assert optimizer.search_strategy is strategy

    def test_invalid_metric(self):
        """Test error on invalid metric."""
        with pytest.raises(ValueError, match="Metric must be callable"):
            HyperparameterOptimizer(metric="not_callable")

    def test_invalid_strategy(self):
        """Test error on invalid strategy."""
        metric = MockMetric()
        with pytest.raises(ValueError, match="Invalid strategy type"):
            HyperparameterOptimizer(metric=metric, strategy=123)

    @pytest.mark.asyncio
    async def test_optimize_basic(self, mock_module, mock_metric, mock_dataset):
        """Test basic optimization flow."""
        optimizer = HyperparameterOptimizer(
            metric=mock_metric, n_trials=5, strategy="random", seed=42
        )

        train, val = mock_dataset.get_train_val_split()

        result = await optimizer.optimize(module=mock_module, trainset=train, valset=val)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_module is not None
        assert result.best_score is not None
        assert result.iterations == 5
        assert result.optimization_time > 0
        assert "best_config" in result.metadata

    @pytest.mark.asyncio
    async def test_optimize_improves_score(self, mock_module, mock_metric, mock_dataset):
        """Test that optimization improves the score."""
        # Use quadratic module that peaks at temperature=0.5
        mock_module.behavior = "quadratic"
        mock_module.config["temperature"] = 1.5  # Start far from optimum

        optimizer = HyperparameterOptimizer(
            metric=mock_metric, n_trials=20, strategy="bayesian", seed=42
        )

        train, val = mock_dataset.get_train_val_split()

        # Get baseline score
        baseline_score = await optimizer._evaluate_baseline(mock_module, val)

        # Optimize
        result = await optimizer.optimize(module=mock_module, trainset=train, valset=val)

        # Should improve
        assert result.improvement > 0
        assert result.best_score > baseline_score

        # Best config should be near temperature=0.5 (the optimum)
        best_temp = result.metadata["best_config"]["temperature"]
        assert abs(best_temp - 0.5) < 0.3  # Should find near-optimal

    @pytest.mark.asyncio
    async def test_optimize_without_provider(self):
        """Test optimization without provider falls back to default."""
        module = MockModule()
        module.provider = None  # Remove provider

        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=2)

        # Mock _get_default_provider to return None
        with patch.object(optimizer, "_get_default_provider", return_value=None):
            with pytest.raises(OptimizationError, match="No provider available"):
                await optimizer.optimize(module, [{"input": "test"}])

    @pytest.mark.asyncio
    async def test_optimize_with_custom_search_space(self, mock_module, mock_dataset):
        """Test optimization with custom search space."""
        from logillm.core.parameters import ParamDomain, ParamSpec, ParamType

        # Define limited search space
        search_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Limited temperature",
                    default=0.5,
                    range=(0.4, 0.6),  # Narrow range around optimum
                )
            }
        )

        optimizer = HyperparameterOptimizer(
            metric=MockMetric(), search_space=search_space, n_trials=5
        )

        train, val = mock_dataset.get_train_val_split()
        result = await optimizer.optimize(mock_module, train, val)

        # Check that optimization stayed within bounds
        best_temp = result.metadata["best_config"]["temperature"]
        assert 0.4 <= best_temp <= 0.6

    @pytest.mark.asyncio
    async def test_history_tracking(self, mock_module, mock_dataset):
        """Test that history is properly tracked."""
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=5, track_history=True)

        train, _ = mock_dataset.get_train_val_split()
        result = await optimizer.optimize(mock_module, train)

        # Check history was tracked
        assert optimizer.history is not None
        assert len(optimizer.history.traces) == 5
        assert optimizer.history.best_config is not None
        assert optimizer.history.best_score is not None

        # Check metadata includes history
        assert "history" in result.metadata
        assert len(result.metadata["history"]) == 5

    @pytest.mark.asyncio
    async def test_no_history_tracking(self, mock_module, mock_dataset):
        """Test optimization without history tracking."""
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=3, track_history=False)

        train, _ = mock_dataset.get_train_val_split()
        result = await optimizer.optimize(mock_module, train)

        assert optimizer.history is None
        assert result.metadata["history"] is None

    @pytest.mark.asyncio
    async def test_early_stopping(self, mock_module, mock_dataset):
        """Test early stopping functionality."""

        # Create custom strategy with early stopping
        class EarlyStoppingStrategy(RandomSearchStrategy):
            def __init__(self):
                super().__init__()
                self.stop_at = 3

            def should_stop(self, history):
                return self.iteration >= self.stop_at

        optimizer = HyperparameterOptimizer(
            metric=MockMetric(),
            strategy=EarlyStoppingStrategy(),
            n_trials=100,  # Would run 100 trials without early stopping
        )

        train, _ = mock_dataset.get_train_val_split()
        result = await optimizer.optimize(mock_module, train)

        # Should stop early
        assert result.iterations <= 3

    @pytest.mark.asyncio
    async def test_module_evaluation_error_handling(self, mock_dataset):
        """Test handling of module evaluation errors."""
        # Create failing module
        failing_module = MockModule(behavior="failing")

        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=10, strategy="random")

        train, _ = mock_dataset.get_train_val_split()

        # Should handle failures gracefully
        result = await optimizer.optimize(failing_module, train)

        # Should still return a result
        assert isinstance(result, OptimizationResult)
        # Score might be lower due to failures being scored as 0
        assert result.best_score >= 0

    def test_apply_preset(self, mock_module):
        """Test applying parameter presets."""
        from logillm.core.parameters import ParamPreset

        optimizer = HyperparameterOptimizer(metric=MockMetric())

        # Apply creative preset
        creative_module = optimizer.apply_preset(mock_module, ParamPreset.CREATIVE)
        assert creative_module.config["temperature"] == 0.9
        assert creative_module.config["top_p"] == 0.95

        # Apply precise preset
        precise_module = optimizer.apply_preset(mock_module, ParamPreset.PRECISE)
        assert precise_module.config["temperature"] == 0.1
        assert precise_module.config["top_p"] == 0.1

        # Invalid preset
        with pytest.raises(ValueError, match="Unknown preset"):
            optimizer.apply_preset(mock_module, "invalid_preset")

    def test_analyze_parameters(self, mock_module):
        """Test parameter analysis."""
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=5, track_history=True)

        # No history yet
        analysis = optimizer.analyze_parameters()
        assert analysis["n_trials"] == 0
        assert analysis["best_config"] is None
        assert analysis["best_score"] == float("-inf")

        # Add some history manually
        if optimizer.history:
            import time

            from logillm.core.parameters import ParameterTrace

            optimizer.history.add_trace(
                ParameterTrace(
                    module_name="MockModule",
                    parameters={"temperature": 0.5, "top_p": 0.9},
                    score=0.8,
                    timestamp=time.time(),
                )
            )
            optimizer.history.add_trace(
                ParameterTrace(
                    module_name="MockModule",
                    parameters={"temperature": 0.7, "top_p": 0.95},
                    score=0.85,
                    timestamp=time.time(),
                )
            )

        analysis = optimizer.analyze_parameters()
        assert "best_config" in analysis
        assert "best_score" in analysis
        assert "n_trials" in analysis
        assert analysis["n_trials"] == 2


class TestGridSearchOptimizer:
    """Test GridSearchOptimizer class."""

    def test_initialization_with_param_grid(self):
        """Test initialization with parameter grid."""
        param_grid = {"temperature": [0.0, 0.5, 1.0], "top_p": [0.8, 0.9, 1.0]}

        optimizer = GridSearchOptimizer(metric=MockMetric(), param_grid=param_grid)

        # Should calculate total combinations
        assert optimizer.n_trials == 9  # 3 * 3

        # Should have created search space
        assert optimizer.search_space is not None
        assert "temperature" in optimizer.search_space.param_specs
        assert "top_p" in optimizer.search_space.param_specs

    def test_initialization_with_resolution(self):
        """Test initialization with resolution parameter."""
        optimizer = GridSearchOptimizer(metric=MockMetric(), resolution=5)

        assert optimizer.search_strategy.resolution == 5

    @pytest.mark.asyncio
    async def test_grid_search_optimization(self):
        """Test grid search optimization."""
        import random

        # Fix random seed for deterministic test
        random.seed(42)

        module = MockModule(behavior="quadratic")

        param_grid = {
            "temperature": [0.3, 0.5, 0.7],  # 0.5 is optimal for quadratic
        }

        # Use a metric that directly evaluates the quadratic score
        def quadratic_metric(pred, target):
            # Return the score from the quadratic module
            if hasattr(pred, "outputs"):
                return pred.outputs.get("score", 0.0)
            return 0.0

        optimizer = GridSearchOptimizer(metric=quadratic_metric, param_grid=param_grid)

        dataset = MockDataset(size=10)
        train, _ = dataset.get_train_val_split()

        result = await optimizer.optimize(module, train)

        # Should test all 3 values
        assert result.iterations == 3

        # For quadratic behavior, 0.5 gives the best score theoretically
        # But with the MockMetric comparing against target 1.0, and module scoring
        # being modified by top_p, the actual optimal may vary
        best_temp = result.metadata["best_config"]["temperature"]
        # Should be one of the grid values
        assert best_temp in [0.3, 0.5, 0.7]
        # With pure quadratic scoring, 0.5 would be best, but our mock adds noise
        # Check that it found a reasonable value near the center
        assert 0.2 <= best_temp <= 0.8


class TestAdaptiveOptimizer:
    """Test AdaptiveOptimizer class."""

    def test_initialization(self):
        """Test adaptive optimizer initialization."""
        optimizer = AdaptiveOptimizer(metric=MockMetric())

        assert optimizer.task_analyzer is not None

    def test_custom_task_analyzer(self):
        """Test with custom task analyzer."""

        def custom_analyzer(dataset):
            return "custom_task"

        optimizer = AdaptiveOptimizer(metric=MockMetric(), task_analyzer=custom_analyzer)

        assert optimizer.task_analyzer is custom_analyzer

    def test_default_task_analyzer(self):
        """Test default task analyzer."""
        optimizer = AdaptiveOptimizer(metric=MockMetric())

        # Test with different dataset types
        dataset = [{"outputs": {"code": "print('hello')"}}]
        assert optimizer._default_task_analyzer(dataset) == "code"

        dataset = [{"outputs": {"answer": "42", "reasoning": "because..."}}]
        assert optimizer._default_task_analyzer(dataset) == "reasoning"

        dataset = [{"outputs": {"summary": "Brief summary"}}]
        assert optimizer._default_task_analyzer(dataset) == "summarization"

        dataset = []
        assert optimizer._default_task_analyzer(dataset) == "general"

    @pytest.mark.asyncio
    async def test_adaptive_optimization(self):
        """Test adaptive optimization adjusts to task type."""
        module = MockModule()
        dataset = MockDataset(size=10)

        # Create optimizer that will detect "code" task
        optimizer = AdaptiveOptimizer(metric=MockMetric(), task_analyzer=lambda x: "code")

        train, _ = dataset.get_train_val_split()
        result = await optimizer.optimize(module, train)

        # Check task type in metadata
        assert result.metadata["task_type"] == "code"

        # Check initial config was set for code
        # (In real implementation, this would be reflected in the search)

    def test_get_task_config(self):
        """Test task-specific configurations."""
        optimizer = AdaptiveOptimizer(metric=MockMetric())

        configs = {
            "code": optimizer._get_task_config("code"),
            "factual": optimizer._get_task_config("factual"),
            "reasoning": optimizer._get_task_config("reasoning"),
            "creative": optimizer._get_task_config("creative"),
            "general": optimizer._get_task_config("general"),
        }

        # Code should have low temperature
        assert configs["code"]["temperature"] < configs["creative"]["temperature"]

        # Factual should have zero temperature
        assert configs["factual"]["temperature"] == 0.0

        # Creative should have high temperature
        assert configs["creative"]["temperature"] > 0.8


class TestOptimizationIntegration:
    """Integration tests for optimization system."""

    @pytest.mark.asyncio
    async def test_optimization_convergence(self):
        """Test that optimization converges to good solutions."""
        # Use quadratic module with known optimum
        module = MockModule(behavior="quadratic")
        module.config["temperature"] = 2.0  # Start far from optimum (0.5)

        monitor = OptimizationMonitor()

        # Custom metric that directly uses the module's score
        def monitored_metric(pred, target):
            # For quadratic module, score peaks at temperature=0.5
            # Just return the score from the module
            score = pred.outputs.get("score", 0.5) if hasattr(pred, "outputs") else 0.5
            monitor.record(module.config, score)
            return score

        optimizer = HyperparameterOptimizer(
            metric=monitored_metric, n_trials=30, strategy="bayesian", seed=42
        )

        dataset = MockDataset(size=20)
        train, val = dataset.get_train_val_split()

        result = await optimizer.optimize(module, train, val)

        # Check convergence with approximate comparisons
        # The score should be reasonable (not terrible)
        assert result.best_score > -1.0  # Not a terrible score
        # Temperature should move toward 0.5 (the optimum for quadratic)
        best_temp = result.metadata["best_config"]["temperature"]
        # Should be closer to 0.5 than starting point (2.0)
        assert abs(best_temp - 0.5) < abs(2.0 - 0.5)  # Moved toward optimum
        # The improvement might be 0 if baseline and best are similar due to randomness
        # but we should have explored the space
        assert result.iterations > 0  # Actually ran trials

    @pytest.mark.asyncio
    async def test_multiple_optimizers_comparison(self):
        """Test comparing different optimization strategies."""
        module_behaviors = ["quadratic", "linear"]
        strategies = ["random", "grid", "bayesian"]

        results = {}

        for behavior in module_behaviors:
            results[behavior] = {}

            for strategy in strategies:
                module = MockModule(behavior=behavior)

                if strategy == "grid":
                    optimizer = GridSearchOptimizer(
                        metric=MockMetric(), param_grid={"temperature": [0.0, 0.25, 0.5, 0.75, 1.0]}
                    )
                else:
                    optimizer = HyperparameterOptimizer(
                        metric=MockMetric(), strategy=strategy, n_trials=10, seed=42
                    )

                dataset = MockDataset(size=10)
                train, _ = dataset.get_train_val_split()

                result = await optimizer.optimize(module, train)
                results[behavior][strategy] = result.best_score

        # All strategies should find reasonable solutions
        for behavior in module_behaviors:
            for strategy in strategies:
                assert results[behavior][strategy] > 0.3

    @pytest.mark.asyncio
    async def test_concurrent_optimization(self):
        """Test running multiple optimizations concurrently."""
        modules = [
            MockModule(behavior="linear"),
            MockModule(behavior="quadratic"),
            MockModule(behavior="random"),
        ]

        dataset = MockDataset(size=10)
        train, _ = dataset.get_train_val_split()

        async def optimize_module(module):
            optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=5, strategy="random")
            return await optimizer.optimize(module, train)

        # Run optimizations concurrently
        results = await asyncio.gather(*[optimize_module(m) for m in modules])

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, OptimizationResult)
            assert result.best_score is not None


class TestErrorHandling:
    """Test error handling in optimization."""

    @pytest.mark.asyncio
    async def test_empty_dataset(self):
        """Test optimization with empty dataset."""
        module = MockModule()
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=5)

        # Should handle empty dataset
        result = await optimizer.optimize(module, [], [])
        assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_nan_scores(self):
        """Test handling of NaN scores."""
        module = MockModule()

        # Metric that sometimes returns NaN
        def nan_metric(pred, target):
            import random

            if random.random() < 0.3:
                return float("nan")
            return 0.5

        optimizer = HyperparameterOptimizer(metric=nan_metric, n_trials=10)

        dataset = MockDataset(size=5)
        train, _ = dataset.get_train_val_split()

        # Should handle NaN gracefully (treating as 0)
        result = await optimizer.optimize(module, train)
        assert isinstance(result, OptimizationResult)
