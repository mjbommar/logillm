"""Tests for MultiObjectiveOptimizer - balancing multiple goals simultaneously."""

from typing import Any

import pytest

from logillm.optimizers.multi_objective import MultiObjectiveOptimizer
from tests.unit.fixtures.mock_components import MockMetric, MockModule


class TestMultiObjectiveOptimizer:
    """Test suite for MultiObjectiveOptimizer."""

    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        return [
            {"inputs": {"text": "Hello"}, "outputs": {"response": "Hi"}},
            {"inputs": {"text": "How are you?"}, "outputs": {"response": "I'm good"}},
            {"inputs": {"text": "What's 2+2?"}, "outputs": {"response": "4"}},
            {"inputs": {"text": "Goodbye"}, "outputs": {"response": "Bye"}},
        ]

    @pytest.fixture
    def metrics(self):
        """Create multiple metrics for multi-objective optimization."""
        return {
            "accuracy": MockMetric(target_value=0.8),
            "speed": MockMetric(target_value=0.7),  # Higher = faster
            "consistency": MockMetric(target_value=0.9),
        }

    @pytest.mark.asyncio
    async def test_basic_multi_objective(self, dataset, metrics):
        """Test basic multi-objective optimization."""
        module = MockModule(behavior="linear", seed=42)

        optimizer = MultiObjectiveOptimizer(
            metrics=metrics,
            weights={"accuracy": 0.5, "speed": 0.3, "consistency": 0.2},
            n_trials=5,
            maintain_pareto=False,
            strategy="weighted",
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check result structure
        assert result.optimized_module is not None
        assert result.iterations == 5
        assert "final_scores" in result.metadata
        assert "baseline_scores" in result.metadata
        assert "improvements" in result.metadata

        # Check all objectives were evaluated
        for metric_name in metrics:
            assert metric_name in result.metadata["final_scores"]
            assert metric_name in result.metadata["baseline_scores"]
            assert metric_name in result.metadata["improvements"]

    @pytest.mark.asyncio
    async def test_weighted_strategy(self, dataset, metrics):
        """Test weighted combination strategy."""
        module = MockModule(behavior="quadratic")

        # Heavy weight on accuracy
        weights = {"accuracy": 0.8, "speed": 0.1, "consistency": 0.1}

        optimizer = MultiObjectiveOptimizer(
            metrics=metrics, weights=weights, n_trials=10, strategy="weighted"
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check weights were normalized
        total_weight = sum(optimizer.weights.values())
        assert abs(total_weight - 1.0) < 0.01

        # Check weighted strategy was used
        assert result.metadata["strategy"] == "weighted"
        assert result.metadata["weights"] == optimizer.weights

    @pytest.mark.asyncio
    async def test_pareto_frontier(self, dataset, metrics):
        """Test Pareto frontier maintenance."""
        module = MockModule(behavior="linear")

        optimizer = MultiObjectiveOptimizer(
            metrics=metrics, maintain_pareto=True, pareto_size=10, n_trials=15, strategy="pareto"
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check Pareto frontier
        assert "pareto_frontier_size" in result.metadata
        assert result.metadata["pareto_frontier_size"] > 0
        assert result.metadata["pareto_frontier_size"] <= 10

        # Get frontier for inspection
        frontier = optimizer.get_pareto_frontier()
        assert len(frontier) <= 10

        for solution in frontier:
            assert "scores" in solution
            assert "weighted_score" in solution

    @pytest.mark.asyncio
    async def test_constraint_strategy(self, dataset, metrics):
        """Test constraint-based optimization."""
        module = MockModule(behavior="random", seed=42)

        # Set minimum constraints
        constraints = {
            "accuracy": 0.6,  # Minimum accuracy
            "consistency": 0.5,  # Minimum consistency
        }

        optimizer = MultiObjectiveOptimizer(
            metrics=metrics, constraints=constraints, n_trials=10, strategy="constraint"
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check constraints were applied
        assert result.metadata["constraints"] == constraints
        assert result.metadata["strategy"] == "constraint"

    @pytest.mark.asyncio
    async def test_latency_evaluation(self, dataset):
        """Test latency objective evaluation."""
        module = MockModule(behavior="linear")  # Linear behavior

        # Create optimizer with latency objective
        metrics = {
            "accuracy": MockMetric(),
            "latency": MockMetric(),  # Special handling for latency
        }

        optimizer = MultiObjectiveOptimizer(metrics=metrics, n_trials=3)

        # Test latency evaluation
        latency_score = await optimizer._evaluate_latency(module, dataset[:2])

        # Score should be between 0 and 1 (converted from time)
        assert 0 <= latency_score <= 1
        # Lower latency = higher score
        assert latency_score > 0

    @pytest.mark.asyncio
    async def test_cost_evaluation(self, dataset):
        """Test cost objective evaluation."""
        module = MockModule(behavior="linear")

        metrics = {
            "accuracy": MockMetric(),
            "cost": MockMetric(),  # Special handling for cost
        }

        optimizer = MultiObjectiveOptimizer(metrics=metrics, n_trials=3)

        # Test cost evaluation
        cost_score = await optimizer._evaluate_cost(module, dataset[:2])

        # Score should be between 0 and 1
        assert 0 <= cost_score <= 1
        # Lower cost = higher score
        assert cost_score > 0

    @pytest.mark.asyncio
    async def test_consistency_evaluation(self, dataset):
        """Test consistency objective evaluation."""
        # Test with deterministic module (high consistency)
        module_consistent = MockModule(behavior="linear", seed=42)

        # Test with random module (low consistency)
        module_random = MockModule(behavior="random", seed=None)

        metrics = {"consistency": MockMetric()}
        optimizer = MultiObjectiveOptimizer(metrics=metrics, n_trials=1)

        # Consistent module should score higher
        consistent_score = await optimizer._evaluate_consistency(module_consistent, dataset[:1])

        random_score = await optimizer._evaluate_consistency(module_random, dataset[:1])

        # Scores should be valid
        assert 0 <= consistent_score <= 1
        assert 0 <= random_score <= 1

        # Deterministic should be more consistent
        # (This might not always hold for Mock, but testing the concept)
        assert consistent_score >= random_score

    @pytest.mark.asyncio
    async def test_dominance_checking(self):
        """Test Pareto dominance checking."""
        metrics = {"accuracy": MockMetric(), "speed": MockMetric()}

        optimizer = MultiObjectiveOptimizer(metrics=metrics, maintain_pareto=True)

        # Test dominance
        scores1 = {"accuracy": 0.9, "speed": 0.8}
        scores2 = {"accuracy": 0.8, "speed": 0.7}
        scores3 = {"accuracy": 0.85, "speed": 0.85}

        # scores1 dominates scores2 (better on both)
        assert optimizer._dominates(scores1, scores2) is True
        assert optimizer._dominates(scores2, scores1) is False

        # scores3 doesn't dominate scores1 (worse on accuracy)
        assert optimizer._dominates(scores3, scores1) is False

        # Equal scores don't dominate
        assert optimizer._dominates(scores1, scores1) is False

    @pytest.mark.asyncio
    async def test_search_space_creation(self, dataset):
        """Test multi-objective search space creation."""
        module = MockModule()
        metrics = {"accuracy": MockMetric()}

        optimizer = MultiObjectiveOptimizer(metrics=metrics)
        search_space = optimizer._create_multi_objective_search_space(module)

        # Check expected parameters
        assert "temperature" in search_space.param_specs
        assert "top_p" in search_space.param_specs
        assert "max_tokens" in search_space.param_specs
        assert "num_demos" in search_space.param_specs

        # Check parameter ranges
        temp_spec = search_space.param_specs["temperature"]
        assert temp_spec.range == (0.0, 2.0)

        top_p_spec = search_space.param_specs["top_p"]
        assert top_p_spec.range == (0.1, 1.0)

    @pytest.mark.asyncio
    async def test_config_application(self):
        """Test applying configuration to module."""
        module = MockModule()
        module.config = {"temperature": 0.7}

        optimizer = MultiObjectiveOptimizer(metrics={"accuracy": MockMetric()})

        config = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 200, "num_demos": 3}

        updated = optimizer._apply_config(module, config)

        # Check config was applied
        assert updated.config["temperature"] == 0.5
        assert updated.config["top_p"] == 0.9
        assert updated.config["max_tokens"] == 200

        # Check demo count was recorded
        assert "demo_count" in updated.parameters
        assert updated.parameters["demo_count"].value == 3

    @pytest.mark.asyncio
    async def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        metrics = {"accuracy": MockMetric(), "speed": MockMetric(), "cost": MockMetric()}

        weights = {"accuracy": 0.5, "speed": 0.3, "cost": 0.2}

        optimizer = MultiObjectiveOptimizer(metrics=metrics, weights=weights)

        scores = {"accuracy": 0.8, "speed": 0.6, "cost": 0.9}

        weighted = optimizer._calculate_weighted_score(scores)

        # Manual calculation
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.9
        assert abs(weighted - expected) < 0.01

    @pytest.mark.asyncio
    async def test_constraint_satisfaction(self):
        """Test constraint satisfaction checking."""
        metrics = {"accuracy": MockMetric(), "speed": MockMetric()}
        constraints = {"accuracy": 0.7, "speed": 0.5}

        optimizer = MultiObjectiveOptimizer(metrics=metrics, constraints=constraints)

        # Test satisfying constraints
        good_scores = {"accuracy": 0.8, "speed": 0.6}
        assert optimizer._satisfies_constraints(good_scores) is True

        # Test violating accuracy constraint
        bad_accuracy = {"accuracy": 0.6, "speed": 0.6}
        assert optimizer._satisfies_constraints(bad_accuracy) is False

        # Test violating speed constraint
        bad_speed = {"accuracy": 0.8, "speed": 0.4}
        assert optimizer._satisfies_constraints(bad_speed) is False

    @pytest.mark.asyncio
    async def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        module = MockModule()
        metrics = {"accuracy": MockMetric()}

        optimizer = MultiObjectiveOptimizer(metrics=metrics, n_trials=2)

        # Should handle empty dataset gracefully
        result = await optimizer.optimize(module=module, dataset=[])

        assert result.optimized_module is not None

    @pytest.mark.asyncio
    async def test_pareto_selection(self, dataset):
        """Test selection from Pareto frontier."""
        module = MockModule(behavior="linear")
        metrics = {"accuracy": MockMetric(), "speed": MockMetric()}

        optimizer = MultiObjectiveOptimizer(
            metrics=metrics, maintain_pareto=True, strategy="pareto", n_trials=5
        )

        # Run optimization to populate frontier
        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should select from Pareto frontier
        assert result.optimized_module is not None
        assert result.metadata["strategy"] == "pareto"

    @pytest.mark.asyncio
    async def test_multiple_metric_types(self, dataset):
        """Test with different metric types (callable vs Metric)."""

        def custom_metric(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
            """Custom callable metric."""
            if predicted.get("response") == expected.get("response"):
                return 1.0
            return 0.0

        module = MockModule(behavior="linear")

        # Mix of Metric objects and callables
        metrics = {"accuracy": MockMetric(), "custom": custom_metric}

        optimizer = MultiObjectiveOptimizer(metrics=metrics, n_trials=3)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should handle both metric types
        assert "accuracy" in result.metadata["final_scores"]
        assert "custom" in result.metadata["final_scores"]

    def test_pareto_frontier_export(self):
        """Test exporting Pareto frontier."""
        metrics = {"accuracy": MockMetric(), "speed": MockMetric()}
        optimizer = MultiObjectiveOptimizer(metrics=metrics, maintain_pareto=True)

        # Manually add some solutions to frontier
        module1 = MockModule()
        module2 = MockModule()

        optimizer.pareto_frontier = [
            {"module": module1, "scores": {"accuracy": 0.8, "speed": 0.7}, "weighted_score": 0.75},
            {"module": module2, "scores": {"accuracy": 0.7, "speed": 0.9}, "weighted_score": 0.8},
        ]

        # Export frontier
        frontier = optimizer.get_pareto_frontier()

        assert len(frontier) == 2
        assert frontier[0]["scores"]["accuracy"] == 0.8
        assert frontier[1]["weighted_score"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
