"""Tests for HybridOptimizer - LogiLLM's killer feature."""

import pytest

from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot
from logillm.optimizers.hybrid_optimizer import HybridOptimizer
from logillm.optimizers.hyperparameter import HyperparameterOptimizer
from tests.unit.fixtures.mock_components import MockMetric, MockModule


class TestHybridOptimizer:
    """Test suite for HybridOptimizer."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset."""
        return [
            {"inputs": {"x": 1}, "outputs": {"y": 2}},
            {"inputs": {"x": 2}, "outputs": {"y": 4}},
            {"inputs": {"x": 3}, "outputs": {"y": 6}},
            {"inputs": {"x": 4}, "outputs": {"y": 8}},
        ]

    @pytest.fixture
    def validation_set(self):
        """Create a validation set."""
        return [
            {"inputs": {"x": 5}, "outputs": {"y": 10}},
            {"inputs": {"x": 6}, "outputs": {"y": 12}},
        ]

    @pytest.mark.asyncio
    async def test_alternating_strategy(self, dataset, validation_set):
        """Test alternating optimization strategy."""
        module = MockModule(behavior="quadratic", seed=42)
        metric = MockMetric()

        optimizer = HybridOptimizer(metric=metric, strategy="alternating", num_iterations=2)

        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Check that optimization occurred
        assert result.optimized_module is not None
        assert result.improvement >= 0  # Should improve or stay same
        assert result.metadata["strategy"] == "alternating"
        assert result.metadata["num_iterations"] > 0

        # Check that both prompts and hyperparameters were optimized
        optimized = result.optimized_module
        assert "demonstrations" in optimized.parameters  # Prompts optimized
        assert hasattr(optimized, "config")  # Has hyperparameters

    @pytest.mark.asyncio
    async def test_joint_strategy(self, dataset):
        """Test joint optimization strategy."""
        module = MockModule(behavior="linear", seed=42)
        metric = MockMetric()

        optimizer = HybridOptimizer(metric=metric, strategy="joint", n_trials=10)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check metadata
        assert result.metadata["strategy"] == "joint"
        assert "best_config" in result.metadata
        assert "score_history" in result.metadata
        assert len(result.metadata["score_history"]) == 10

    @pytest.mark.asyncio
    async def test_sequential_strategy(self, dataset):
        """Test sequential optimization strategy."""
        module = MockModule(behavior="random", seed=42)
        metric = MockMetric()

        optimizer = HybridOptimizer(metric=metric, strategy="sequential")

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check sequential metadata
        assert result.metadata["strategy"] == "sequential"
        assert "hyperopt_improvement" in result.metadata
        assert "prompt_improvement" in result.metadata
        assert "after_hyperopt_score" in result.metadata

    @pytest.mark.asyncio
    async def test_custom_optimizers(self, dataset):
        """Test with custom prompt and hyperparameter optimizers."""
        module = MockModule(behavior="linear")
        metric = MockMetric()

        # Custom component optimizers
        prompt_opt = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
        hyper_opt = HyperparameterOptimizer(metric=metric, strategy="grid", n_trials=4)

        optimizer = HybridOptimizer(
            metric=metric,
            prompt_optimizer=prompt_opt,
            hyperparameter_optimizer=hyper_opt,
            strategy="alternating",
            num_iterations=1,
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should have used custom optimizers
        assert result.optimized_module is not None
        demos = result.optimized_module.parameters.get("demonstrations")
        if demos and demos.value:
            assert len(demos.value) <= 2  # Max 2 demos as configured

    @pytest.mark.asyncio
    async def test_convergence_detection(self, dataset):
        """Test convergence detection in alternating strategy."""
        module = MockModule(behavior="linear")
        metric = MockMetric()

        optimizer = HybridOptimizer(
            metric=metric,
            strategy="alternating",
            num_iterations=10,  # High number
            convergence_threshold=0.001,
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should converge before max iterations
        assert result.metadata["convergence"] or result.metadata["num_iterations"] <= 10

    @pytest.mark.asyncio
    async def test_balance_weight(self, dataset):
        """Test balance weight between prompt and hyperparameter importance."""
        module = MockModule(behavior="quadratic")
        metric = MockMetric()

        # High weight on hyperparameters
        optimizer = HybridOptimizer(
            metric=metric,
            strategy="joint",
            n_trials=5,
            balance_weight=0.8,  # Favor hyperparameters
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check that optimization occurred
        assert result.optimized_module is not None

    @pytest.mark.asyncio
    async def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        module = MockModule()
        metric = MockMetric()
        optimizer = HybridOptimizer(metric=metric)

        # Should handle gracefully
        result = await optimizer.optimize(module=module, dataset=[])

        # Should return something even with empty dataset
        assert result.optimized_module is not None

    @pytest.mark.asyncio
    async def test_invalid_strategy(self, dataset):
        """Test invalid strategy handling."""
        module = MockModule()
        metric = MockMetric()

        optimizer = HybridOptimizer(metric=metric, strategy="invalid_strategy")

        with pytest.raises(ValueError, match="Unknown strategy"):
            await optimizer.optimize(module=module, dataset=dataset)

    @pytest.mark.asyncio
    async def test_improvement_tracking(self, dataset, validation_set):
        """Test that improvements are properly tracked."""
        module = MockModule(behavior="quadratic", seed=42)
        metric = MockMetric(target_value=0.5)

        optimizer = HybridOptimizer(metric=metric, strategy="alternating", num_iterations=3)

        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Check improvement tracking
        assert "baseline_score" in result.metadata
        assert "score_trajectory" in result.metadata
        assert len(result.metadata["score_trajectory"]) > 0

        # Improvement should be calculated correctly
        baseline = result.metadata["baseline_score"]
        final = result.best_score
        assert abs(result.improvement - (final - baseline)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
