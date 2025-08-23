"""Tests for search strategies."""

import math

import pytest

from logillm.core.parameters import (
    STANDARD_PARAM_SPECS,
    ParamDomain,
    ParamSpec,
    ParamType,
    SearchSpace,
)
from logillm.optimizers.search_strategies import (
    AcquisitionType,
    GridSearchStrategy,
    LatinHypercubeStrategy,
    RandomSearchStrategy,
    SimpleBayesianStrategy,
    StrategyConfig,
    create_strategy,
)


class TestRandomSearchStrategy:
    """Test random search strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = RandomSearchStrategy()
        assert strategy.name == "random"
        assert not strategy.requires_history
        assert not strategy.is_initialized

    def test_suggest_next(self):
        """Test suggesting next configuration."""
        strategy = RandomSearchStrategy(config=StrategyConfig(seed=42))
        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # Get multiple suggestions
        configs = [strategy.suggest_next() for _ in range(10)]

        # Check all configs are valid
        for config in configs:
            assert isinstance(config, dict)
            assert "temperature" in config
            assert 0.0 <= config["temperature"] <= 2.0

        # Check randomness (with fixed seed, should be deterministic)
        strategy2 = RandomSearchStrategy(config=StrategyConfig(seed=42))
        strategy2.initialize(search_space)
        configs2 = [strategy2.suggest_next() for _ in range(10)]

        for c1, c2 in zip(configs, configs2):
            assert c1["temperature"] == c2["temperature"]

    def test_update_does_nothing(self):
        """Test that update has no effect for random search."""
        strategy = RandomSearchStrategy()
        # Should not raise
        strategy.update({"temperature": 0.5}, 0.9)


class TestGridSearchStrategy:
    """Test grid search strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = GridSearchStrategy(resolution=5)
        assert strategy.name == "grid"
        assert not strategy.requires_history
        assert strategy.resolution == 5

    def test_grid_generation(self):
        """Test grid point generation."""
        strategy = GridSearchStrategy(resolution=3)

        # Create simple search space
        search_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Temperature for generation",
                    default=0.5,
                    range=(0.0, 1.0),
                ),
                "use_cot": ParamSpec(
                    name="use_cot",
                    param_type=ParamType.BOOL,
                    domain=ParamDomain.GENERATION,
                    description="Use chain of thought",
                    default=False,
                ),
            }
        )

        strategy.initialize(search_space)

        # Should have 3 * 2 = 6 grid points
        assert len(strategy.grid_points) == 6

        # Check all combinations are covered
        configs = [strategy.suggest_next() for _ in range(6)]
        temperatures = sorted({c["temperature"] for c in configs})
        bools = sorted({c["use_cot"] for c in configs})

        assert len(temperatures) == 3
        assert len(bools) == 2
        assert bools == [False, True]

        # Check temperatures are evenly spaced
        assert temperatures[0] == 0.0
        assert temperatures[-1] == 1.0
        assert abs(temperatures[1] - 0.5) < 0.01

    def test_categorical_parameter(self):
        """Test grid search with categorical parameters."""
        strategy = GridSearchStrategy()

        search_space = SearchSpace(
            {
                "format": ParamSpec(
                    name="format",
                    param_type=ParamType.CATEGORICAL,
                    domain=ParamDomain.GENERATION,
                    description="Output format",
                    default="json",
                    choices=["json", "xml", "text"],
                ),
            }
        )

        strategy.initialize(search_space)
        assert len(strategy.grid_points) == 3

        configs = [strategy.suggest_next() for _ in range(3)]
        formats = sorted({c["format"] for c in configs})
        assert formats == ["json", "text", "xml"]

    def test_wraparound(self):
        """Test that grid search wraps around when exhausted."""
        strategy = GridSearchStrategy(resolution=2)

        search_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Temperature for generation",
                    default=0.5,
                    range=(0.0, 1.0),
                ),
            }
        )

        strategy.initialize(search_space)
        assert len(strategy.grid_points) == 2

        # Get more configs than grid points
        configs = [strategy.suggest_next() for _ in range(5)]

        # Should wrap around
        assert configs[0] == configs[2]
        assert configs[1] == configs[3]


class TestSimpleBayesianStrategy:
    """Test Bayesian optimization strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SimpleBayesianStrategy()
        assert strategy.name == "bayesian"
        assert strategy.requires_history
        assert strategy.config.n_warmup == 10

    def test_warmup_phase(self):
        """Test warmup phase uses random sampling."""
        strategy = SimpleBayesianStrategy(config=StrategyConfig(seed=42, n_warmup=3))

        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # First n_warmup suggestions should be random
        configs = []
        for i in range(3):
            config = strategy.suggest_next()
            configs.append(config)
            strategy.update(config, 0.5 + i * 0.1)

        # All should be different (very likely with continuous parameter)
        temperatures = [c["temperature"] for c in configs]
        assert len(set(temperatures)) == 3

    def test_acquisition_function(self):
        """Test acquisition function scoring."""
        strategy = SimpleBayesianStrategy(
            config=StrategyConfig(acquisition_type=AcquisitionType.EXPECTED_IMPROVEMENT)
        )

        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # Add some observations
        strategy.update({"temperature": 0.5}, 0.7)
        strategy.update({"temperature": 1.0}, 0.6)
        strategy.update({"temperature": 0.2}, 0.8)

        # Test acquisition scoring
        score1 = strategy._acquisition_score({"temperature": 0.3})
        score2 = strategy._acquisition_score({"temperature": 0.5})

        # Both should be valid scores
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    def test_update_tracking(self):
        """Test that updates are tracked correctly."""
        strategy = SimpleBayesianStrategy()

        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # Add observations
        strategy.update({"temperature": 0.5}, 0.7)
        strategy.update({"temperature": 1.0}, 0.9)

        assert len(strategy.observations) == 2
        assert strategy.best_score == 0.9
        assert strategy.best_config == {"temperature": 1.0}

    def test_distance_calculation(self):
        """Test configuration distance calculation."""
        strategy = SimpleBayesianStrategy()

        search_space = SearchSpace(
            {
                "temperature": STANDARD_PARAM_SPECS["temperature"],
                "top_p": STANDARD_PARAM_SPECS["top_p"],
            }
        )
        strategy.initialize(search_space)

        config1 = {"temperature": 0.5, "top_p": 0.9}
        config2 = {"temperature": 0.5, "top_p": 0.9}
        config3 = {"temperature": 1.0, "top_p": 0.5}

        # Same configs should have distance 0
        dist1 = strategy._config_distance(config1, config2)
        assert dist1 == 0.0

        # Different configs should have positive distance
        dist2 = strategy._config_distance(config1, config3)
        assert dist2 > 0.0

    def test_different_acquisition_types(self):
        """Test different acquisition function types."""
        for acq_type in [
            AcquisitionType.EXPECTED_IMPROVEMENT,
            AcquisitionType.UPPER_CONFIDENCE_BOUND,
            AcquisitionType.PROBABILITY_OF_IMPROVEMENT,
        ]:
            strategy = SimpleBayesianStrategy(config=StrategyConfig(acquisition_type=acq_type))

            search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
            strategy.initialize(search_space)

            # Add an observation
            strategy.update({"temperature": 0.5}, 0.7)

            # Should be able to score
            score = strategy._acquisition_score({"temperature": 0.8})
            assert isinstance(score, float)
            assert not math.isnan(score)


class TestLatinHypercubeStrategy:
    """Test Latin Hypercube sampling strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = LatinHypercubeStrategy(n_samples=20)
        assert strategy.name == "latin_hypercube"
        assert not strategy.requires_history
        assert strategy.n_samples == 20

    def test_sample_generation(self):
        """Test Latin Hypercube sample generation."""
        strategy = LatinHypercubeStrategy(config=StrategyConfig(seed=42), n_samples=10)

        search_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Temperature for generation",
                    default=0.5,
                    range=(0.0, 1.0),
                ),
                "top_p": ParamSpec(
                    name="top_p",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Top-p sampling parameter",
                    default=0.9,
                    range=(0.0, 1.0),
                ),
            }
        )

        strategy.initialize(search_space)
        assert len(strategy.samples) == 10

        # Get all samples
        configs = [strategy.suggest_next() for _ in range(10)]

        # Check Latin Hypercube property: each parameter should have
        # one sample in each of 10 equal intervals
        temperatures = sorted(c["temperature"] for c in configs)
        top_ps = sorted(c["top_p"] for c in configs)

        # Check that values are well-distributed
        for i in range(10):
            expected_min = i * 0.1
            expected_max = (i + 1) * 0.1

            # Each interval should have exactly one sample
            temps_in_interval = sum(
                1 for t in temperatures if expected_min <= t < expected_max or (i == 9 and t == 1.0)
            )
            assert temps_in_interval == 1

            top_ps_in_interval = sum(
                1 for p in top_ps if expected_min <= p < expected_max or (i == 9 and p == 1.0)
            )
            assert top_ps_in_interval == 1

    def test_regeneration_on_exhaustion(self):
        """Test that samples regenerate when exhausted."""
        strategy = LatinHypercubeStrategy(config=StrategyConfig(seed=42), n_samples=3)

        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # Get more samples than n_samples
        configs = [strategy.suggest_next() for _ in range(5)]

        # Should have regenerated after 3
        assert len(configs) == 5
        assert strategy.current_index == 2  # Reset and advanced to 2


class TestStrategyFactory:
    """Test strategy factory function."""

    def test_create_strategy_by_name(self):
        """Test creating strategies by name."""
        strategies = {
            "random": RandomSearchStrategy,
            "grid": GridSearchStrategy,
            "bayesian": SimpleBayesianStrategy,
            "latin_hypercube": LatinHypercubeStrategy,
        }

        for name, expected_class in strategies.items():
            strategy = create_strategy(name)
            assert isinstance(strategy, expected_class)
            assert strategy.name == name

    def test_create_with_config(self):
        """Test creating strategy with configuration."""
        config = StrategyConfig(seed=123, n_warmup=5)
        strategy = create_strategy("bayesian", config=config)

        assert strategy.config.seed == 123
        assert strategy.config.n_warmup == 5

    def test_create_with_kwargs(self):
        """Test creating strategy with keyword arguments."""
        strategy = create_strategy("grid", resolution=15)
        assert isinstance(strategy, GridSearchStrategy)
        assert strategy.resolution == 15

    def test_invalid_strategy_name(self):
        """Test error on invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("invalid_strategy")


class TestStrategyReset:
    """Test strategy reset functionality."""

    def test_random_reset(self):
        """Test resetting random search."""
        strategy = RandomSearchStrategy()
        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # Make some calls
        for _ in range(5):
            strategy.suggest_next()

        assert strategy.iteration == 5

        strategy.reset()
        assert strategy.iteration == 0

    def test_bayesian_reset(self):
        """Test resetting Bayesian strategy."""
        strategy = SimpleBayesianStrategy()
        search_space = SearchSpace({"temperature": STANDARD_PARAM_SPECS["temperature"]})
        strategy.initialize(search_space)

        # Add observations
        strategy.update({"temperature": 0.5}, 0.7)
        strategy.update({"temperature": 1.0}, 0.9)

        assert len(strategy.observations) == 2
        assert strategy.best_score == 0.9

        strategy.reset()
        assert strategy.iteration == 0
        assert len(strategy.observations) == 0
        assert strategy.best_score == float("-inf")
        assert strategy.best_config is None
