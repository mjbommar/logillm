"""Hyperparameter-aware optimizer for LogiLLM.

This module provides hyperparameter optimization with pluggable search strategies,
all implemented with zero external dependencies.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..core.modules import Module
from ..core.optimizers import Optimizer
from ..core.parameters import (
    STANDARD_PRESETS,
    ParameterHistory,
    ParameterTrace,
    ParamPreset,
    SearchSpace,
)
from ..core.types import Configuration, OptimizationResult
from ..exceptions import OptimizationError
from .search_strategies import (
    GridSearchStrategy,
    SearchStrategy,
    SimpleBayesianStrategy,
    StrategyConfig,
    create_strategy,
)


class HyperparameterOptimizer(Optimizer):
    """Optimizer that tunes both program and LLM hyperparameters.

    Supports multiple search strategies through a pluggable interface,
    with zero external dependencies. Default strategy is Bayesian optimization.
    """

    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        search_space: SearchSpace | None = None,
        strategy: str | SearchStrategy | None = None,
        n_trials: int = 100,
        track_history: bool = True,
        seed: int | None = None,
        strategy_config: StrategyConfig | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize hyperparameter optimizer.

        Args:
            metric: Evaluation metric
            search_space: Parameter search space
            strategy: Search strategy name or instance (default: "bayesian")
            n_trials: Number of optimization trials
            track_history: Whether to track optimization history
            seed: Random seed
            strategy_config: Configuration for the search strategy
            verbose: Enable step-by-step logging
            **kwargs: Additional strategy-specific arguments
        """
        from ..core.optimizers import CustomMetric
        from ..core.types import OptimizationStrategy

        # Create metric wrapper if needed
        if not callable(metric):
            raise ValueError("Metric must be callable")

        # Wrap the metric if it's a simple function
        if not hasattr(metric, "name"):
            metric_wrapper = CustomMetric(metric, "custom")
        else:
            metric_wrapper = metric

        super().__init__(
            strategy=OptimizationStrategy.HYBRID,
            metric=metric_wrapper,
            verbose=verbose,
        )

        # Initialize search strategy
        if strategy is None:
            strategy = "bayesian"  # Default to Bayesian optimization

        if isinstance(strategy, str):
            # Create strategy from name
            config = strategy_config or StrategyConfig(seed=seed)
            self.search_strategy = create_strategy(strategy, config=config, **kwargs)
        elif isinstance(strategy, SearchStrategy):
            # Use provided strategy instance
            self.search_strategy = strategy
        else:
            raise ValueError(f"Invalid strategy type: {type(strategy)}")

        self.search_space = search_space
        self.n_trials = n_trials
        self.track_history = track_history
        self.seed = seed
        self.history = ParameterHistory() if track_history else None

    async def optimize(
        self, module: Module, trainset: list[Any], valset: list[Any] | None = None, **kwargs
    ) -> OptimizationResult:
        """Optimize module hyperparameters.

        Args:
            module: Module to optimize
            trainset: Training examples
            valset: Validation examples
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with optimized module
        """
        start_time = time.time()

        # Initialize logging timestamp
        if self.verbose and not hasattr(self, "_start_time_seconds"):
            self._start_time_seconds = start_time

        # Get provider and its parameter specs
        provider = getattr(module, "provider", None) or self._get_default_provider()
        if not provider:
            raise OptimizationError(
                "No provider available for hyperparameter optimization",
                context={"module": module.__class__.__name__},
            )

        # Build search space if not provided
        if self.search_space is None:
            self.search_space = self._build_search_space(provider)

        # Initialize search strategy with search space
        self.search_strategy.initialize(self.search_space)

        # Run optimization
        best_config, best_score, actual_trials = await self._optimize_with_strategy(
            module, trainset, valset
        )

        # Apply best configuration
        optimized_module = module.deepcopy()
        if best_config:
            optimized_module.config.update(best_config)

        optimization_time = time.time() - start_time

        return OptimizationResult(
            optimized_module=optimized_module,
            improvement=best_score - await self._evaluate_baseline(module, valset or trainset),
            iterations=actual_trials,
            best_score=best_score,
            optimization_time=optimization_time,
            metadata={
                "best_config": best_config,
                "search_space": list(self.search_space.param_specs),
                "strategy": self.search_strategy.name,
                "history": self.history.traces if self.history else None,
            },
        )

    async def _optimize_with_strategy(
        self,
        module: Module,
        trainset: list[Any],
        valset: list[Any] | None,
    ) -> tuple[Configuration, float, int]:
        """Optimize using the configured search strategy."""
        eval_set = valset or trainset
        best_config = None
        best_score = float("-inf")
        actual_trials = 0

        for trial_num in range(self.n_trials):
            # Get next configuration from strategy
            config = self.search_strategy.suggest_next(
                history=self.history if self.search_strategy.requires_history else None
            )

            # Log parameter trial
            if self.verbose:
                param_str = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in config.items()
                )
                self._log_step(trial_num + 1, self.n_trials, f"Testing params: {param_str}")

            # Apply configuration
            test_module = module.deepcopy()
            test_module.config.update(config)

            # Evaluate
            score = await self._evaluate_module(test_module, eval_set)

            # Log results
            if self.verbose:
                if score > best_score:
                    self._log_step(trial_num + 1, self.n_trials, f"🎯 NEW BEST! Score: {score:.4f}")
                else:
                    self._log_step(trial_num + 1, self.n_trials, f"Score: {score:.4f}")

            # Update strategy with result
            self.search_strategy.update(config, score, metadata={"trial": trial_num})

            # Track history
            if self.history:
                trace = ParameterTrace(
                    module_name=module.__class__.__name__,
                    parameters=config,
                    score=score,
                    timestamp=time.time(),
                    metadata={"trial": trial_num, "strategy": self.search_strategy.name},
                )
                self.history.add_trace(trace)

            # Update best
            if score > best_score:
                best_score = score
                best_config = config.copy()

            actual_trials += 1

            # Check for early stopping
            if self.search_strategy.should_stop(self.history if self.track_history else None):
                break

        return best_config, best_score, actual_trials

    async def _evaluate_module(
        self,
        module: Module,
        dataset: list[Any],
    ) -> float:
        """Evaluate module on dataset."""
        scores = []

        for example in dataset:
            try:
                # Assume example has input/output structure
                if hasattr(example, "inputs") and hasattr(example, "outputs"):
                    prediction = await module(**example.inputs)
                    score = self.metric(prediction.outputs, example.outputs)
                else:
                    # Try to use example as dict
                    prediction = await module(**example.get("inputs", example))
                    score = self.metric(prediction.outputs, example.get("outputs", {}))

                scores.append(score)
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    async def _evaluate_baseline(self, module: Module, dataset: list[Any]) -> float:
        """Evaluate baseline performance."""
        return await self._evaluate_module(module, dataset)

    def _build_search_space(self, provider) -> SearchSpace:
        """Build search space from provider specs."""
        from ..core.parameters import STANDARD_PARAM_SPECS

        # Try to get provider-specific specs
        if hasattr(provider, "get_param_specs"):
            param_specs = provider.get_param_specs()
        else:
            # Use standard specs
            param_specs = STANDARD_PARAM_SPECS.copy()

        return SearchSpace(param_specs)

    def _get_default_provider(self):
        """Get default provider."""
        from ..core.providers import get_provider

        try:
            return get_provider()
        except Exception:
            return None

    def apply_preset(self, module: Module, preset: ParamPreset) -> Module:
        """Apply a parameter preset to a module.

        Args:
            module: Module to configure
            preset: Preset to apply

        Returns:
            Module with preset applied
        """
        if preset not in STANDARD_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")

        configured = module.deepcopy()
        configured.config.update(STANDARD_PRESETS[preset])
        return configured

    def analyze_parameters(self) -> dict[str, Any]:
        """Analyze parameter importance and correlations.

        Returns:
            Analysis results including best configuration, parameter importance,
            and optimization trajectories.
        """
        if not self.history or not self.history.traces:
            # Return minimal analysis if no history
            return {
                "best_config": self.history.best_config if self.history else None,
                "best_score": self.history.best_score if self.history else float("-inf"),
                "n_trials": len(self.history.traces) if self.history else 0,
                "strategy": self.search_strategy.name,
                "parameter_importance": {},
                "parameter_trajectories": {},
            }

        analysis = {
            "best_config": self.history.best_config,
            "best_score": self.history.best_score,
            "n_trials": len(self.history.traces),
            "strategy": self.search_strategy.name,
            "parameter_importance": {},
            "parameter_trajectories": {},
        }

        # Analyze each parameter
        if self.search_space:
            for param_name in self.search_space.param_specs:
                # Get correlation with score
                correlation = self.history.get_correlation(param_name)
                analysis["parameter_importance"][param_name] = abs(correlation)

                # Get optimization trajectory
                trajectory = self.history.get_trajectory(param_name)
                if trajectory:
                    analysis["parameter_trajectories"][param_name] = trajectory

        return analysis


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search optimizer for systematic parameter exploration.

    Convenience class that uses GridSearchStrategy internally.
    """

    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        param_grid: dict[str, list[Any]] | None = None,
        resolution: int = 10,
        **kwargs,
    ):
        """Initialize grid search optimizer.

        Args:
            metric: Evaluation metric
            param_grid: Grid of parameters to search (optional)
            resolution: Number of points per continuous parameter
            **kwargs: Additional arguments for HyperparameterOptimizer
        """
        # Create GridSearchStrategy
        strategy = GridSearchStrategy(resolution=resolution)

        # If param_grid provided, create search space from it
        search_space = None
        if param_grid:
            from ..core.parameters import ParamDomain, ParamSpec, ParamType

            param_specs = {}
            for name, values in param_grid.items():
                # For grid search, treat all discrete values as categorical
                # This ensures we only test the exact values provided
                param_type = ParamType.CATEGORICAL

                param_specs[name] = ParamSpec(
                    name=name,
                    param_type=param_type,
                    domain=ParamDomain.GENERATION,
                    description=f"Grid search parameter {name}",
                    default=values[0],
                    choices=values,  # Always use provided values as choices
                    range=None,  # Don't use ranges for grid search
                )

            search_space = SearchSpace(param_specs)

            # Calculate total combinations
            n_trials = 1
            for values in param_grid.values():
                n_trials *= len(values)
        else:
            n_trials = kwargs.get("n_trials", 100)

        super().__init__(
            metric=metric, strategy=strategy, search_space=search_space, n_trials=n_trials, **kwargs
        )


class AdaptiveOptimizer(HyperparameterOptimizer):
    """Adaptive optimizer that adjusts parameters based on task characteristics.

    Analyzes the task type and adjusts the search strategy and parameter
    ranges accordingly.
    """

    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        task_analyzer: Callable[[list[Any]], str] | None = None,
        **kwargs,
    ):
        """Initialize adaptive optimizer.

        Args:
            metric: Evaluation metric
            task_analyzer: Function to analyze task type
            **kwargs: Additional arguments
        """
        super().__init__(metric=metric, **kwargs)
        self.task_analyzer = task_analyzer or self._default_task_analyzer

    def _default_task_analyzer(self, dataset: list[Any]) -> str:
        """Default task analyzer - tries to infer task type."""
        # Simple heuristics
        if not dataset:
            return "general"

        # Check first example
        example = dataset[0]

        # Try to detect task type from structure
        if hasattr(example, "outputs"):
            output = example.outputs
        elif isinstance(example, dict) and "outputs" in example:
            output = example["outputs"]
        else:
            return "general"

        # Check output characteristics
        if isinstance(output, dict):
            if any(key in output for key in ["code", "program", "function"]):
                return "code"
            elif any(key in output for key in ["answer", "solution"]):
                if any(key in output for key in ["reasoning", "explanation"]):
                    return "reasoning"
                return "factual"
            elif any(key in output for key in ["summary", "tldr"]):
                return "summarization"

        return "general"

    async def optimize(
        self, module: Module, trainset: list[Any], valset: list[Any] | None = None, **kwargs
    ) -> OptimizationResult:
        """Optimize with task-aware parameter selection."""
        # Analyze task
        task_type = self.task_analyzer(trainset)

        # Set initial configuration based on task
        initial_config = self._get_task_config(task_type)
        module.config.update(initial_config)

        # Adjust search space based on task
        self.search_space = self._get_task_search_space(task_type)

        # Choose strategy based on task
        if task_type == "code":
            # Code generation benefits from more focused search
            self.search_strategy = SimpleBayesianStrategy(
                config=StrategyConfig(n_warmup=5, exploration_weight=0.05)
            )
        elif task_type == "creative":
            # Creative tasks benefit from more exploration
            self.search_strategy = SimpleBayesianStrategy(
                config=StrategyConfig(n_warmup=15, exploration_weight=0.2)
            )
        else:
            # Default strategy
            self.search_strategy = SimpleBayesianStrategy()

        # Run optimization
        result = await super().optimize(module, trainset, valset, **kwargs)

        # Add task type to metadata
        result.metadata["task_type"] = task_type

        return result

    def _get_task_config(self, task_type: str) -> Configuration:
        """Get initial configuration for task type."""
        configs = {
            "code": {"temperature": 0.2, "top_p": 0.95},
            "factual": {"temperature": 0.0, "top_p": 0.1},
            "reasoning": {"temperature": 0.7, "top_p": 0.9},
            "creative": {"temperature": 0.9, "top_p": 0.95},
            "summarization": {"temperature": 0.3, "top_p": 0.8},
            "general": {"temperature": 0.7, "top_p": 0.9},
        }
        return configs.get(task_type, configs["general"])

    def _get_task_search_space(self, task_type: str) -> SearchSpace:
        """Get search space for task type."""
        from ..core.parameters import ParamDomain, ParamSpec, ParamType

        # Different search ranges for different tasks
        if task_type == "code":
            param_specs = {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Temperature for code generation",
                    default=0.2,
                    range=(0.0, 0.5),  # Lower range for code
                    step=0.05,
                ),
                "top_p": ParamSpec(
                    name="top_p",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Top-p for code generation",
                    default=0.95,
                    range=(0.8, 1.0),
                    step=0.05,
                ),
            }
        elif task_type == "creative":
            param_specs = {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Temperature for creative tasks",
                    default=0.9,
                    range=(0.7, 1.5),  # Higher range for creativity
                    step=0.1,
                ),
                "top_p": ParamSpec(
                    name="top_p",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Top-p for creative tasks",
                    default=0.95,
                    range=(0.9, 1.0),
                    step=0.02,
                ),
            }
        else:
            from ..core.parameters import STANDARD_PARAM_SPECS

            param_specs = STANDARD_PARAM_SPECS.copy()

        return SearchSpace(param_specs)


__all__ = [
    "HyperparameterOptimizer",
    "GridSearchOptimizer",
    "AdaptiveOptimizer",
]
