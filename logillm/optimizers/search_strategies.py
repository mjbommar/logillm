"""Search strategies for hyperparameter optimization.

This module provides pluggable search strategies for hyperparameter optimization,
all implemented with zero external dependencies.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.parameters import ParameterHistory, SearchSpace
from ..core.types import Configuration


class AcquisitionType(Enum):
    """Types of acquisition functions for Bayesian optimization."""

    EXPECTED_IMPROVEMENT = "ei"
    UPPER_CONFIDENCE_BOUND = "ucb"
    PROBABILITY_OF_IMPROVEMENT = "pi"
    THOMPSON_SAMPLING = "thompson"


@dataclass
class StrategyConfig:
    """Configuration for search strategies."""

    seed: int | None = None
    n_warmup: int = 10
    exploration_weight: float = 0.1
    acquisition_type: AcquisitionType = AcquisitionType.EXPECTED_IMPROVEMENT
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchStrategy(ABC):
    """Base class for all hyperparameter search strategies.

    This defines the interface that all search strategies must implement,
    allowing them to be used interchangeably in the optimizer.
    """

    def __init__(self, config: StrategyConfig | None = None):
        """Initialize the search strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()
        self.search_space: SearchSpace | None = None
        self.rng = random.Random(self.config.seed)
        self.iteration = 0
        self.is_initialized = False

    def initialize(self, search_space: SearchSpace) -> None:
        """Initialize the strategy with a search space.

        Args:
            search_space: The parameter search space
        """
        self.search_space = search_space
        self.is_initialized = True
        self._on_initialize()

    @abstractmethod
    def _on_initialize(self) -> None:
        """Hook for strategy-specific initialization."""
        ...

    @abstractmethod
    def suggest_next(self, history: ParameterHistory | None = None) -> Configuration:
        """Suggest the next configuration to evaluate.

        Args:
            history: Optional parameter history for informed strategies

        Returns:
            Next configuration to evaluate
        """
        ...

    @abstractmethod
    def update(
        self, config: Configuration, score: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """Update strategy with evaluation result.

        Args:
            config: Configuration that was evaluated
            score: Score achieved by the configuration
            metadata: Optional metadata about the evaluation
        """
        ...

    def should_stop(self, history: ParameterHistory | None = None) -> bool:
        """Check if optimization should stop early.

        Args:
            history: Parameter history

        Returns:
            True if optimization should stop
        """
        # Default: never stop early
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        ...

    @property
    def requires_history(self) -> bool:
        """Whether strategy needs access to full history."""
        return False

    def reset(self) -> None:
        """Reset strategy state."""
        self.iteration = 0
        self._on_reset()

    @abstractmethod
    def _on_reset(self) -> None:
        """Hook for strategy-specific reset."""
        ...


class RandomSearchStrategy(SearchStrategy):
    """Random search strategy.

    Samples configurations uniformly at random from the search space.
    Simple but surprisingly effective for many problems.
    """

    def _on_initialize(self) -> None:
        """No special initialization needed."""
        pass

    def suggest_next(self, history: ParameterHistory | None = None) -> Configuration:
        """Suggest random configuration."""
        if not self.search_space:
            raise ValueError("Strategy not initialized with search space")

        self.iteration += 1
        return self.search_space.sample(self.rng)

    def update(
        self, config: Configuration, score: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """No update needed for random search."""
        pass

    @property
    def name(self) -> str:
        """Strategy name."""
        return "random"

    def _on_reset(self) -> None:
        """No state to reset."""
        pass


class GridSearchStrategy(SearchStrategy):
    """Grid search strategy.

    Systematically explores all combinations of parameter values.
    Guarantees coverage but can be expensive for large spaces.
    """

    def __init__(self, config: StrategyConfig | None = None, resolution: int = 10):
        """Initialize grid search.

        Args:
            config: Strategy configuration
            resolution: Number of points per continuous parameter
        """
        super().__init__(config)
        self.resolution = resolution
        self.grid_points: list[Configuration] = []
        self.current_index = 0

    def _on_initialize(self) -> None:
        """Build the grid of configurations."""
        if not self.search_space:
            return

        # Generate grid points for each parameter
        param_grids = {}
        for name, spec in self.search_space.param_specs.items():
            if name in self.search_space.fixed_params:
                param_grids[name] = [self.search_space.fixed_params[name]]
            elif spec.param_type.value == "float" and spec.range:
                # Create grid for continuous parameter
                min_val, max_val = spec.range
                step = (max_val - min_val) / (self.resolution - 1)
                param_grids[name] = [min_val + i * step for i in range(self.resolution)]
            elif spec.param_type.value == "int" and spec.range:
                # Create grid for integer parameter
                min_val, max_val = int(spec.range[0]), int(spec.range[1])
                step = max(1, (max_val - min_val) // (self.resolution - 1))
                param_grids[name] = list(range(min_val, max_val + 1, step))
            elif spec.param_type.value == "categorical" and spec.choices:
                param_grids[name] = spec.choices
            elif spec.param_type.value == "bool":
                param_grids[name] = [True, False]
            else:
                # Use default value
                param_grids[name] = [spec.default]

        # Generate all combinations
        import itertools

        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]

        self.grid_points = [
            dict(zip(param_names, values)) for values in itertools.product(*param_values)
        ]

        # Shuffle if seed is set for randomized grid search
        if self.config.seed is not None:
            self.rng.shuffle(self.grid_points)

    def suggest_next(self, history: ParameterHistory | None = None) -> Configuration:
        """Suggest next grid point."""
        if not self.grid_points:
            raise ValueError("No grid points available")

        if self.current_index >= len(self.grid_points):
            # Wrap around if we've exhausted the grid
            self.current_index = 0

        config = self.grid_points[self.current_index]
        self.current_index += 1
        self.iteration += 1

        return config

    def update(
        self, config: Configuration, score: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """No update needed for grid search."""
        pass

    @property
    def name(self) -> str:
        """Strategy name."""
        return "grid"

    def _on_reset(self) -> None:
        """Reset grid position."""
        self.current_index = 0


class SimpleBayesianStrategy(SearchStrategy):
    """Simple Bayesian optimization strategy.

    Uses a simplified Gaussian Process model with RBF kernel for
    surrogate modeling. Balances exploration and exploitation using
    acquisition functions.

    This is a zero-dependency alternative to Optuna's TPE.
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        kernel_bandwidth: float = 0.1,
    ):
        """Initialize Bayesian strategy.

        Args:
            config: Strategy configuration
            kernel_bandwidth: RBF kernel bandwidth parameter
        """
        super().__init__(config)
        self.kernel_bandwidth = kernel_bandwidth
        self.observations: list[tuple[Configuration, float]] = []
        self.best_score = float("-inf")
        self.best_config: Configuration | None = None

    def _on_initialize(self) -> None:
        """Initialize strategy."""
        self.observations = []
        self.best_score = float("-inf")
        self.best_config = None

    def suggest_next(self, history: ParameterHistory | None = None) -> Configuration:
        """Suggest next configuration using Bayesian optimization."""
        if not self.search_space:
            raise ValueError("Strategy not initialized")

        self.iteration += 1

        # Use history if available to rebuild observations
        if history and self.requires_history:
            self._rebuild_from_history(history)

        # Warm-up phase: random exploration
        if len(self.observations) < self.config.n_warmup:
            return self.search_space.sample(self.rng)

        # Generate candidate configurations
        candidates = self._generate_candidates(n=100)

        # Score candidates using acquisition function
        scores = [self._acquisition_score(c) for c in candidates]

        # Select best candidate
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return candidates[best_idx]

    def update(
        self, config: Configuration, score: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """Update with evaluation result."""
        self.observations.append((config, score))

        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()

    def _generate_candidates(self, n: int) -> list[Configuration]:
        """Generate candidate configurations."""
        if not self.search_space:
            return []

        candidates = []

        # Mix of random and local search around good points
        n_random = n // 2
        n_local = n - n_random

        # Random candidates
        for _ in range(n_random):
            candidates.append(self.search_space.sample(self.rng))

        # Local search around best observations
        if self.observations:
            top_k = min(5, len(self.observations))
            top_configs = sorted(self.observations, key=lambda x: x[1], reverse=True)[:top_k]

            for _ in range(n_local):
                # Pick a good configuration
                base_config = self.rng.choice(top_configs)[0]

                # Add noise to create variation
                new_config = self._mutate_config(base_config, noise=0.2)
                candidates.append(new_config)

        return candidates

    def _mutate_config(self, config: Configuration, noise: float) -> Configuration:
        """Create variation of a configuration."""
        if not self.search_space:
            return config

        mutated = {}
        for name, value in config.items():
            if name in self.search_space.fixed_params:
                mutated[name] = value
                continue

            spec = self.search_space.param_specs.get(name)
            if not spec:
                mutated[name] = value
                continue

            if spec.param_type.value == "float" and spec.range:
                # Add Gaussian noise
                min_val, max_val = spec.range
                std_dev = (max_val - min_val) * noise
                new_value = value + self.rng.gauss(0, std_dev)
                mutated[name] = max(min_val, min(max_val, new_value))

            elif spec.param_type.value == "int" and spec.range:
                # Add discrete noise
                min_val, max_val = int(spec.range[0]), int(spec.range[1])
                std_dev = (max_val - min_val) * noise
                new_value = int(value + self.rng.gauss(0, std_dev))
                mutated[name] = max(min_val, min(max_val, new_value))

            elif spec.param_type.value == "categorical" and spec.choices:
                # Occasionally switch to different choice
                if self.rng.random() < noise:
                    mutated[name] = self.rng.choice(spec.choices)
                else:
                    mutated[name] = value

            else:
                mutated[name] = value

        return mutated

    def _acquisition_score(self, config: Configuration) -> float:
        """Compute acquisition score for a configuration."""
        if not self.observations:
            return 0.0

        # Compute predicted mean and uncertainty using kernel density
        mean, std = self._predict(config)

        if self.config.acquisition_type == AcquisitionType.EXPECTED_IMPROVEMENT:
            # Expected Improvement
            if std == 0:
                return 0.0

            z = (mean - self.best_score) / std
            # Approximate normal CDF and PDF
            ei = std * (z * self._normal_cdf(z) + self._normal_pdf(z))

            # Add exploration bonus
            return ei + self.config.exploration_weight * std

        elif self.config.acquisition_type == AcquisitionType.UPPER_CONFIDENCE_BOUND:
            # Upper Confidence Bound
            beta = 2.0  # Exploration parameter
            return mean + beta * std

        else:  # PROBABILITY_OF_IMPROVEMENT
            # Probability of Improvement
            if std == 0:
                return 0.0

            z = (mean - self.best_score) / std
            return self._normal_cdf(z)

    def _predict(self, config: Configuration) -> tuple[float, float]:
        """Predict mean and std using kernel regression."""
        if not self.observations:
            return 0.0, 1.0

        # Compute kernel weights
        weights = []
        values = []

        for obs_config, obs_score in self.observations:
            # Compute distance between configurations
            distance = self._config_distance(config, obs_config)

            # RBF kernel
            weight = math.exp(-distance / (2 * self.kernel_bandwidth**2))
            weights.append(weight)
            values.append(obs_score)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 1.0

        weights = [w / total_weight for w in weights]

        # Weighted mean
        mean = sum(w * v for w, v in zip(weights, values))

        # Weighted std (uncertainty based on distance to observations)
        variance = sum(w * (v - mean) ** 2 for w, v in zip(weights, values))
        std = math.sqrt(variance) if variance > 0 else 0.1

        # Increase uncertainty for regions far from observations
        min_weight = max(weights) if weights else 0
        exploration_factor = 1.0 - min_weight
        std = std * (1 + exploration_factor)

        return mean, std

    def _config_distance(self, config1: Configuration, config2: Configuration) -> float:
        """Compute distance between two configurations."""
        if not self.search_space:
            return 0.0

        distance = 0.0
        n_params = 0

        for name in config1:
            if name not in config2:
                continue

            spec = self.search_space.param_specs.get(name)
            if not spec:
                continue

            val1 = config1[name]
            val2 = config2[name]

            if spec.param_type.value in ["float", "int"] and spec.range:
                # Normalized distance for numeric parameters
                min_val, max_val = spec.range
                if max_val > min_val:
                    normalized_dist = abs(val1 - val2) / (max_val - min_val)
                    distance += normalized_dist**2
                    n_params += 1

            elif spec.param_type.value == "categorical":
                # Hamming distance for categorical
                if val1 != val2:
                    distance += 1.0
                n_params += 1

            elif spec.param_type.value == "bool":
                # Boolean distance
                if val1 != val2:
                    distance += 1.0
                n_params += 1

        # Return Euclidean distance
        return math.sqrt(distance / max(1, n_params))

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function."""
        # Using approximation: cdf(x) ≈ 0.5 * (1 + erf(x / sqrt(2)))
        return 0.5 * (1 + self._erf(x / math.sqrt(2)))

    def _normal_pdf(self, x: float) -> float:
        """Normal PDF."""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    def _erf(self, x: float) -> float:
        """Approximate error function."""
        # Using Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return sign * y

    def _rebuild_from_history(self, history: ParameterHistory) -> None:
        """Rebuild observations from history."""
        self.observations = []
        self.best_score = float("-inf")
        self.best_config = None

        for trace in history.traces:
            config = trace.parameters
            score = trace.score
            self.observations.append((config, score))

            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()

    @property
    def name(self) -> str:
        """Strategy name."""
        return "bayesian"

    @property
    def requires_history(self) -> bool:
        """Bayesian strategy benefits from history."""
        return True

    def _on_reset(self) -> None:
        """Reset strategy state."""
        self.observations = []
        self.best_score = float("-inf")
        self.best_config = None


class LatinHypercubeStrategy(SearchStrategy):
    """Latin Hypercube sampling strategy.

    Generates a space-filling design that ensures good coverage
    of the parameter space with fewer samples than grid search.
    """

    def __init__(self, config: StrategyConfig | None = None, n_samples: int = 100):
        """Initialize Latin Hypercube strategy.

        Args:
            config: Strategy configuration
            n_samples: Number of samples to generate
        """
        super().__init__(config)
        self.n_samples = n_samples
        self.samples: list[Configuration] = []
        self.current_index = 0

    def _on_initialize(self) -> None:
        """Generate Latin Hypercube samples."""
        if not self.search_space:
            return

        # Get active parameters
        active_params = [
            (name, spec)
            for name, spec in self.search_space.param_specs.items()
            if name not in self.search_space.fixed_params
        ]

        if not active_params:
            return

        # Generate Latin Hypercube design
        # Each parameter gets n_samples evenly spaced values
        param_samples = {}

        for name, spec in active_params:
            if spec.param_type.value == "float" and spec.range:
                min_val, max_val = spec.range
                # Divide range into n_samples intervals
                intervals = []
                for i in range(self.n_samples):
                    low = min_val + i * (max_val - min_val) / self.n_samples
                    high = min_val + (i + 1) * (max_val - min_val) / self.n_samples
                    # Sample within interval
                    value = self.rng.uniform(low, high)
                    intervals.append(value)
                self.rng.shuffle(intervals)
                param_samples[name] = intervals

            elif spec.param_type.value == "int" and spec.range:
                min_val, max_val = int(spec.range[0]), int(spec.range[1])
                values = []
                for i in range(self.n_samples):
                    # Map to integer range
                    frac = (i + self.rng.random()) / self.n_samples
                    value = int(min_val + frac * (max_val - min_val + 1))
                    value = min(max_val, max(min_val, value))
                    values.append(value)
                self.rng.shuffle(values)
                param_samples[name] = values

            elif spec.param_type.value == "categorical" and spec.choices:
                # Repeat choices to fill n_samples
                values = []
                for _ in range(self.n_samples):
                    values.append(self.rng.choice(spec.choices))
                param_samples[name] = values

            else:
                # Use default
                param_samples[name] = [spec.default] * self.n_samples

        # Combine into configurations
        self.samples = []
        for i in range(self.n_samples):
            config = {}

            # Add fixed params
            config.update(self.search_space.fixed_params)

            # Add sampled params
            for name, _ in active_params:
                config[name] = param_samples[name][i]

            self.samples.append(config)

    def suggest_next(self, history: ParameterHistory | None = None) -> Configuration:
        """Suggest next Latin Hypercube sample."""
        if not self.samples:
            raise ValueError("No samples available")

        if self.current_index >= len(self.samples):
            # Regenerate if exhausted
            self._on_initialize()
            self.current_index = 0

        config = self.samples[self.current_index]
        self.current_index += 1
        self.iteration += 1

        return config

    def update(
        self, config: Configuration, score: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """No update needed."""
        pass

    @property
    def name(self) -> str:
        """Strategy name."""
        return "latin_hypercube"

    def _on_reset(self) -> None:
        """Reset sampling."""
        self.current_index = 0
        self.samples = []


# Strategy factory
def create_strategy(name: str, config: StrategyConfig | None = None, **kwargs) -> SearchStrategy:
    """Create a search strategy by name.

    Args:
        name: Strategy name (random, grid, bayesian, latin_hypercube)
        config: Strategy configuration
        **kwargs: Additional strategy-specific arguments

    Returns:
        Search strategy instance
    """
    strategies = {
        "random": RandomSearchStrategy,
        "grid": GridSearchStrategy,
        "bayesian": SimpleBayesianStrategy,
        "latin_hypercube": LatinHypercubeStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {', '.join(strategies.keys())}")

    strategy_class = strategies[name]
    return strategy_class(config=config, **kwargs)


__all__ = [
    "SearchStrategy",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "SimpleBayesianStrategy",
    "LatinHypercubeStrategy",
    "StrategyConfig",
    "AcquisitionType",
    "create_strategy",
]
