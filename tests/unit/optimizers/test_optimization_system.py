"""Integration tests for optimization system."""

import json
import tempfile
from pathlib import Path

import pytest

from logillm.core.modules import Module
from logillm.core.parameters import (
    SearchSpace,
)
from logillm.core.types import Prediction, Usage
from logillm.optimizers import HyperparameterOptimizer
from logillm.optimizers.search_strategies import (
    GridSearchStrategy,
    LatinHypercubeStrategy,
    StrategyConfig,
    create_strategy,
)
from tests.unit.fixtures.mock_components import (
    MockDataset,
    MockMetric,
    MockModule,
    OptimizationMonitor,
)


class TestEndToEndOptimization:
    """Test complete optimization workflows."""

    @pytest.mark.asyncio
    async def test_complete_optimization_pipeline(self):
        """Test full optimization pipeline from start to finish."""
        # 1. Create module with suboptimal settings
        module = MockModule(behavior="quadratic")
        module.config = {"temperature": 1.8, "top_p": 0.5}  # Far from optimal

        # 2. Create dataset
        dataset = MockDataset(size=50, task_type="general")
        train, val = dataset.get_train_val_split()

        # 3. Create optimizer with Bayesian strategy
        optimizer = HyperparameterOptimizer(
            metric=MockMetric(target_value=1.0),
            strategy="bayesian",
            n_trials=25,
            track_history=True,
            seed=42,
        )

        # 4. Run optimization
        result = await optimizer.optimize(module, train, val)

        # 5. Verify results
        assert result.optimized_module is not None
        assert result.improvement > 0
        assert result.best_score > 0.7

        # 6. Check optimized configuration
        optimized_config = result.optimized_module.config
        assert 0.3 <= optimized_config["temperature"] <= 0.7  # Near optimal (0.5)

        # 7. Analyze parameters
        analysis = optimizer.analyze_parameters()
        assert analysis["n_trials"] == 25
        assert "parameter_importance" in analysis
        assert "temperature" in analysis["parameter_importance"]

    @pytest.mark.asyncio
    async def test_strategy_comparison(self):
        """Compare different optimization strategies on same problem."""
        strategies_to_test = [
            ("random", {"config": StrategyConfig(seed=42)}),
            ("grid", {"resolution": 5}),
            ("bayesian", {"config": StrategyConfig(n_warmup=5, seed=42)}),
            ("latin_hypercube", {"n_samples": 20}),
        ]

        results = {}
        MockModule(behavior="quadratic", seed=42)
        dataset = MockDataset(size=30)
        train, val = dataset.get_train_val_split()

        for strategy_name, _kwargs in strategies_to_test:
            # Fresh module for each strategy with fixed seed
            module = MockModule(behavior="quadratic", seed=42)
            module.config = {"temperature": 1.5, "top_p": 0.5}

            # Create correct strategy configuration
            if strategy_name == "random":
                strategy = create_strategy("random", config=StrategyConfig(seed=42))
            elif strategy_name == "grid":
                strategy = GridSearchStrategy(resolution=5)
            elif strategy_name == "bayesian":
                strategy = create_strategy("bayesian", config=StrategyConfig(n_warmup=5, seed=42))
            elif strategy_name == "latin_hypercube":
                strategy = LatinHypercubeStrategy(n_samples=20)

            optimizer = HyperparameterOptimizer(
                metric=MockMetric(target_value=1.0),
                strategy=strategy,
                n_trials=20,
                track_history=True,
            )

            result = await optimizer.optimize(module, train, val)

            results[strategy_name] = {
                "best_score": result.best_score,
                "best_temp": result.metadata["best_config"]["temperature"],
                "improvement": result.improvement,
                "time": result.optimization_time,
            }

        # All strategies should find reasonable solutions
        for strategy_name in results:
            assert results[strategy_name]["best_score"] > 0.6
            # Temperature should be near optimal (0.5)
            assert abs(results[strategy_name]["best_temp"] - 0.5) < 0.5

        # Bayesian should generally perform well
        assert results["bayesian"]["best_score"] >= results["random"]["best_score"] - 0.1

    @pytest.mark.asyncio
    async def test_optimization_with_constraints(self):
        """Test optimization with parameter constraints."""
        from logillm.core.parameters import ParamDomain, ParamSpec, ParamType

        # Create constrained search space
        constrained_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Constrained temperature",
                    default=0.45,
                    range=(0.4, 0.6),  # Tight constraint around optimum
                    step=0.05,
                ),
                "top_p": ParamSpec(
                    name="top_p",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Constrained top-p",
                    default=0.9,
                    range=(0.85, 0.95),  # Tight constraint
                    step=0.05,
                ),
            }
        )

        module = MockModule(behavior="quadratic")
        optimizer = HyperparameterOptimizer(
            metric=MockMetric(),
            search_space=constrained_space,
            strategy="grid",  # Grid will respect constraints exactly
            n_trials=100,  # Will be limited by grid size
        )

        dataset = MockDataset(size=20)
        train, _ = dataset.get_train_val_split()

        await optimizer.optimize(module, train)

        # Check all tried configurations respect constraints
        if optimizer.history:
            for trace in optimizer.history.traces:
                assert 0.4 <= trace.parameters["temperature"] <= 0.6
                assert 0.85 <= trace.parameters["top_p"] <= 0.95

    @pytest.mark.asyncio
    async def test_optimization_reproducibility(self):
        """Test that optimization is reproducible with same seed."""

        async def run_optimization(seed):
            module = MockModule(behavior="quadratic")
            module.config = {"temperature": 1.5}

            optimizer = HyperparameterOptimizer(
                metric=MockMetric(), strategy="random", n_trials=10, seed=seed
            )

            dataset = MockDataset(size=10)
            train, _ = dataset.get_train_val_split()

            result = await optimizer.optimize(module, train)
            return result.metadata["best_config"]

        # Run twice with same seed
        config1 = await run_optimization(42)
        config2 = await run_optimization(42)

        # Should get same results
        assert config1["temperature"] == config2["temperature"]
        assert config1["top_p"] == config2["top_p"]

        # Run with different seed
        config3 = await run_optimization(123)

        # Should get different results (very likely)
        assert (
            config3["temperature"] != config1["temperature"] or config3["top_p"] != config1["top_p"]
        )


class TestOptimizationPersistence:
    """Test saving and loading optimization results."""

    @pytest.mark.asyncio
    async def test_save_load_history(self):
        """Test saving and loading optimization history."""
        # Run optimization
        module = MockModule(behavior="quadratic")
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=10, track_history=True)

        dataset = MockDataset(size=20)
        train, _ = dataset.get_train_val_split()

        await optimizer.optimize(module, train)

        # Save history
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            history_data = {
                "traces": [
                    {
                        "module_name": t.module_name,
                        "parameters": t.parameters,
                        "score": t.score,
                        "timestamp": t.timestamp,
                        "metadata": t.metadata,
                    }
                    for t in optimizer.history.traces
                ],
                "best_config": optimizer.history.best_config,
                "best_score": optimizer.history.best_score,
            }
            json.dump(history_data, f)
            temp_path = f.name

        try:
            # Load history
            with open(temp_path) as f:
                loaded_data = json.load(f)

            # Verify
            assert len(loaded_data["traces"]) == 10
            assert loaded_data["best_config"] == optimizer.history.best_config
            assert loaded_data["best_score"] == optimizer.history.best_score

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_resume_optimization(self):
        """Test resuming optimization from previous state."""
        module = MockModule(behavior="quadratic")
        dataset = MockDataset(size=20)
        train, _ = dataset.get_train_val_split()

        # First optimization run
        optimizer1 = HyperparameterOptimizer(
            metric=MockMetric(), strategy="bayesian", n_trials=10, track_history=True, seed=42
        )

        result1 = await optimizer1.optimize(module, train)
        history1 = optimizer1.history

        # Create new optimizer with same strategy
        optimizer2 = HyperparameterOptimizer(
            metric=MockMetric(), strategy="bayesian", n_trials=10, track_history=True, seed=42
        )

        # Manually restore history to simulate resume
        if history1 and optimizer2.search_strategy.name == "bayesian":
            # Bayesian strategy can use history
            optimizer2.search_strategy._rebuild_from_history(history1)

        # Continue optimization
        result2 = await optimizer2.optimize(module, train)

        # Should build on previous results
        assert result2.best_score >= result1.best_score - 0.1  # Allow small variance


class TestOptimizationMonitoring:
    """Test monitoring and debugging optimization."""

    @pytest.mark.asyncio
    async def test_optimization_monitoring(self):
        """Test monitoring optimization progress."""
        OptimizationMonitor()

        # Create custom metric that uses monitor
        def monitored_metric(pred, target):
            if hasattr(pred, "outputs"):
                score = pred.outputs.get("score", 0.5)
            else:
                score = 0.5
            # Record would normally be called differently
            # This is just for testing
            return score

        module = MockModule(behavior="quadratic")
        optimizer = HyperparameterOptimizer(
            metric=monitored_metric, n_trials=20, track_history=True
        )

        dataset = MockDataset(size=20)
        train, _ = dataset.get_train_val_split()

        await optimizer.optimize(module, train)

        # Check optimizer's built-in history
        assert len(optimizer.history.traces) == 20
        assert optimizer.history.best_score > 0

        # Could check convergence if we had access to monitor
        # In real use, monitor would be integrated differently

    @pytest.mark.asyncio
    async def test_optimization_with_callbacks(self):
        """Test optimization with progress callbacks."""
        callback_data = {"iterations": 0, "best_score": 0.0}

        # Create metric that acts as callback
        def callback_metric(pred, target):
            callback_data["iterations"] += 1
            score = pred.outputs.get("score", 0.5) if hasattr(pred, "outputs") else 0.5
            callback_data["best_score"] = max(callback_data["best_score"], score)
            return score

        module = MockModule(behavior="quadratic")
        optimizer = HyperparameterOptimizer(metric=callback_metric, n_trials=15)

        dataset = MockDataset(size=10)
        train, _ = dataset.get_train_val_split()

        await optimizer.optimize(module, train)

        # Callback should have been called for each evaluation
        assert callback_data["iterations"] > 0
        assert callback_data["best_score"] > 0


class TestRobustness:
    """Test robustness and edge cases."""

    @pytest.mark.asyncio
    async def test_single_trial(self):
        """Test optimization with single trial."""
        module = MockModule()
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=1)

        dataset = MockDataset(size=5)
        train, _ = dataset.get_train_val_split()

        result = await optimizer.optimize(module, train)
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_zero_trials(self):
        """Test optimization with zero trials."""
        module = MockModule()
        optimizer = HyperparameterOptimizer(metric=MockMetric(), n_trials=0)

        dataset = MockDataset(size=5)
        train, _ = dataset.get_train_val_split()

        result = await optimizer.optimize(module, train)
        assert result.iterations == 0

    @pytest.mark.asyncio
    async def test_identical_scores(self):
        """Test optimization when all scores are identical."""

        # Module that always returns same score
        class ConstantModule(Module):
            async def forward(self, **inputs):
                return Prediction(outputs={"score": 0.5}, usage=Usage(), success=True)

        module = ConstantModule()
        module.provider = MockModule().provider  # Add provider

        optimizer = HyperparameterOptimizer(
            metric=lambda p, t: 0.5,  # Constant metric
            n_trials=10,
        )

        dataset = MockDataset(size=5)
        train, _ = dataset.get_train_val_split()

        result = await optimizer.optimize(module, train)

        # Should still complete
        assert result.best_score == 0.5
        assert result.improvement == 0.0

    @pytest.mark.asyncio
    async def test_extreme_parameter_values(self):
        """Test optimization with extreme parameter values."""
        from logillm.core.parameters import ParamDomain, ParamSpec, ParamType

        # Create search space with extreme values
        extreme_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Extreme temperature",
                    default=1000.0,
                    range=(0.0, 10000.0),
                ),
                "tiny_param": ParamSpec(
                    name="tiny_param",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Tiny parameter",
                    default=0.0001,
                    range=(0.0, 0.001),
                    step=0.0001,
                ),
            }
        )

        module = MockModule()
        optimizer = HyperparameterOptimizer(
            metric=MockMetric(), search_space=extreme_space, strategy="random", n_trials=10
        )

        dataset = MockDataset(size=5)
        train, _ = dataset.get_train_val_split()

        result = await optimizer.optimize(module, train)

        # Should handle extreme values
        assert result.optimized_module is not None
        assert "temperature" in result.metadata["best_config"]
        assert "tiny_param" in result.metadata["best_config"]
