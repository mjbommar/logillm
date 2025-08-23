"""Hybrid optimizer - LogiLLM's killer feature that optimizes both prompts and hyperparameters.

This is impossible in DSPy because they have no hyperparameter optimization infrastructure.
"""

import copy
import time
from typing import Any, Optional

from ..core.modules import Module
from ..core.optimizers import Metric, Optimizer
from ..core.parameters import SearchSpace
from ..core.types import OptimizationResult, OptimizationStrategy
from .base import PromptOptimizer
from .bootstrap_fewshot import BootstrapFewShot
from .format_optimizer import FormatOptimizer, FormatOptimizerConfig
from .hyperparameter import HyperparameterOptimizer
from .optimizer_config import HybridOptimizerConfig
from .search_strategies import SimpleBayesianStrategy, StrategyConfig


class HybridOptimizer(Optimizer):
    """Simultaneously optimizes prompts AND hyperparameters.

    This is LogiLLM's unique value proposition - the ability to optimize
    both prompt components (instructions, demonstrations) and LLM
    hyperparameters (temperature, top_p, etc.) together.

    DSPy can't do this because they have no hyperparameter optimization
    infrastructure. This gives LogiLLM a significant performance advantage.

    Strategies:
    - alternating: Alternate between prompt and hyperparameter optimization
    - joint: Optimize both in a unified search space
    - sequential: First optimize hyperparameters, then prompts (or vice versa)
    """

    def __init__(
        self,
        metric: Metric,
        prompt_optimizer: Optional[PromptOptimizer] = None,
        hyperparameter_optimizer: Optional[Optimizer] = None,
        format_optimizer: Optional[FormatOptimizer] = None,
        strategy: str = "alternating",  # alternating, joint, sequential
        config: Optional[HybridOptimizerConfig] = None,
        optimize_format: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """Initialize HybridOptimizer.

        Args:
            metric: Evaluation metric
            prompt_optimizer: Optimizer for prompts (default: BootstrapFewShot)
            hyperparameter_optimizer: Optimizer for hyperparameters
            format_optimizer: Optimizer for prompt formats (optional)
            strategy: Optimization strategy (alternating, joint, sequential)
            config: Configuration object (defaults to HybridOptimizerConfig())
            optimize_format: Whether to include format optimization
            verbose: Enable step-by-step logging
            **kwargs: Override specific config values
        """
        super().__init__(
            strategy=OptimizationStrategy.HYBRID,
            metric=metric,
            verbose=verbose,
        )

        # Initialize config (use provided or create default)
        self.config = config or HybridOptimizerConfig()

        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize component optimizers
        self.prompt_opt = prompt_optimizer or BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=4
        )

        # Adjust trial allocation based on whether format is being optimized
        n_components = 3 if optimize_format else 2
        self.hyper_opt = hyperparameter_optimizer or HyperparameterOptimizer(
            metric=metric,
            strategy="bayesian",
            n_trials=self.config.n_trials // n_components,
            verbose=verbose,
        )

        # Optional format optimizer
        self.optimize_format = optimize_format
        self.format_opt = format_optimizer
        if optimize_format and not self.format_opt:
            self.format_opt = FormatOptimizer(metric=metric, config=FormatOptimizerConfig())

        self.optimization_strategy = strategy
        self.convergence_history = []

        # Set up logging if enabled
        if self.config.enable_logging:
            import logging

            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(getattr(logging, self.config.log_level))

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize both prompts and hyperparameters.

        Args:
            module: Module to optimize
            dataset: Training dataset
            validation_set: Validation dataset
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with optimized module
        """
        if self.optimization_strategy == "alternating":
            return await self._alternating_optimization(module, dataset, validation_set, **kwargs)
        elif self.optimization_strategy == "joint":
            return await self._joint_optimization(module, dataset, validation_set, **kwargs)
        elif self.optimization_strategy == "sequential":
            return await self._sequential_optimization(module, dataset, validation_set, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.optimization_strategy}")

    async def _alternating_optimization(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Alternates between optimizing prompts and hyperparameters.

        Each optimization benefits from the other's improvements,
        leading to better overall performance.
        """
        start_time = time.time()

        # Initialize logging timestamp
        if self.verbose and not hasattr(self, "_start_time_seconds"):
            self._start_time_seconds = start_time

        current = copy.deepcopy(module)
        eval_set = validation_set or dataset

        # Track scores for convergence
        scores = []
        best_module = None
        best_score = -float("inf")
        best_config = {}  # Track the best hyperparameter configuration

        # Log start
        if self.verbose:
            total_steps = self.config.num_iterations * 2  # Prompt + hyperparameter per iteration
            if self.optimize_format:
                total_steps += 1
            self._log_step(0, total_steps, "Starting alternating optimization...")

        # Get baseline score
        baseline_score, _ = await self.evaluate(module, eval_set)
        scores.append(baseline_score)

        if self.verbose:
            self._log_step(0, total_steps, f"Baseline score: {baseline_score:.4f}")

        iteration = 0
        for iteration in range(self.config.num_iterations):
            # Optional: Optimize format first (if enabled)
            if self.optimize_format and iteration == 0:
                try:
                    format_result = await self.format_opt.optimize(
                        current, dataset[:50], validation_set[:50] if validation_set else None
                    )
                    current = format_result.optimized_module
                    format_score, _ = await self.evaluate(current, eval_set)
                    scores.append(format_score)

                    if format_score > best_score:
                        best_score = format_score
                        best_module = copy.deepcopy(current)

                    if self.config.enable_logging:
                        self.logger.info(
                            f"Format optimization: {format_result.metadata.get('best_format')} (score: {format_score:.3f})"
                        )
                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Format optimization failed: {e}")

            # Calculate step numbers for logging
            step_offset = 1 if self.optimize_format and iteration == 0 else 0
            hyper_step = iteration * 2 + 1 + step_offset
            prompt_step = iteration * 2 + 2 + step_offset

            # Optimize hyperparameters with current prompts
            if self.verbose:
                self._log_step(
                    hyper_step,
                    total_steps,
                    f"Iteration {iteration + 1}: Optimizing hyperparameters...",
                )

            try:
                # If param_space is provided, create a new hyperparameter optimizer with the search space
                if "param_space" in kwargs and kwargs["param_space"]:
                    from ..core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace

                    # Build search space from param_space dict
                    param_specs = {}
                    for name, value_range in kwargs["param_space"].items():
                        if isinstance(value_range, tuple) and len(value_range) == 2:
                            # It's a range for a float parameter
                            param_specs[name] = ParamSpec(
                                name=name,
                                param_type=ParamType.FLOAT,
                                domain=ParamDomain.GENERATION,
                                description=f"Hyperparameter {name}",
                                default=(value_range[0] + value_range[1]) / 2,
                                range=value_range,
                            )

                    # Set the search space on the hyperparameter optimizer
                    if param_specs:
                        self.hyper_opt.search_space = SearchSpace(param_specs)

                hyper_result = await self.hyper_opt.optimize(current, dataset, validation_set)
                current = hyper_result.optimized_module

                # Extract the best configuration from hyperparameter optimization
                if hyper_result.metadata and "best_config" in hyper_result.metadata:
                    current_config = hyper_result.metadata["best_config"]
                    if current_config:  # Only update if we got a valid config
                        best_config.update(current_config)
            except Exception as e:
                # Log the error but continue with current module
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Hyperparameter optimization failed in iteration {iteration}: {e}")
                # Continue with current module

            # Evaluate after hyperparameter optimization
            hyper_score, _ = await self.evaluate(current, eval_set)

            if self.verbose:
                self._log_step(hyper_step, total_steps, f"Hyperparameter score: {hyper_score:.4f}")

            # Optimize prompts with optimal hyperparameters
            if self.verbose:
                self._log_step(
                    prompt_step, total_steps, f"Iteration {iteration + 1}: Optimizing prompts..."
                )

            try:
                prompt_result = await self.prompt_opt.optimize(current, dataset, validation_set)
                current = prompt_result.optimized_module
            except Exception as e:
                # Log the error but continue with current module
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Prompt optimization failed in iteration {iteration}: {e}")
                # Continue with current module

            # Evaluate after prompt optimization
            final_score, _ = await self.evaluate(current, eval_set)
            scores.append(final_score)

            if self.verbose:
                if final_score > best_score:
                    self._log_step(
                        prompt_step, total_steps, f"🎯 NEW BEST! Score: {final_score:.4f}"
                    )
                else:
                    self._log_step(prompt_step, total_steps, f"Prompt score: {final_score:.4f}")

            # Track best
            if final_score > best_score:
                best_score = final_score
                best_module = copy.deepcopy(current)

            # Check for convergence
            if self._has_converged(scores):
                break

        return OptimizationResult(
            optimized_module=best_module or current,
            improvement=best_score - baseline_score,
            iterations=iteration + 1,
            best_score=best_score,
            optimization_time=time.time() - start_time,
            metadata={
                "strategy": "alternating",
                "num_iterations": iteration + 1,
                "score_trajectory": scores,
                "baseline_score": baseline_score,
                "convergence": self._has_converged(scores),
                "best_config": best_config,  # Include the best hyperparameter configuration
            },
        )

    async def _joint_optimization(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimizes both simultaneously in a unified search space.

        Most powerful but computationally expensive approach.
        """
        start_time = time.time()
        eval_set = validation_set or dataset

        # Get baseline score
        baseline_score, _ = await self.evaluate(module, eval_set)

        # Create unified search space (pass param_space if provided)
        search_space = self._create_joint_search_space(module, kwargs.get("param_space"))

        # Use advanced search strategy
        strategy = SimpleBayesianStrategy(
            config=StrategyConfig(n_warmup=self.config.n_warmup_joint)
        )
        strategy.initialize(search_space)

        best_config = None
        best_module = None
        best_score = -float("inf")
        score_history = []

        for _trial in range(self.config.n_trials):
            # Get next configuration (both prompts and hyperparams)
            config = strategy.suggest_next()

            # Apply configuration
            test_module = await self._apply_joint_config(module, config, dataset)

            # Evaluate
            score, _ = await self.evaluate(test_module, eval_set)
            score_history.append(score)

            # Update strategy
            strategy.update(config, score)

            if score > best_score:
                best_score = score
                best_config = config
                best_module = copy.deepcopy(test_module)

        return OptimizationResult(
            optimized_module=best_module,
            improvement=best_score - baseline_score,
            iterations=self.config.n_trials,
            best_score=best_score,
            optimization_time=time.time() - start_time,
            metadata={
                "strategy": "joint",
                "trials": self.config.n_trials,
                "best_config": best_config,
                "score_history": score_history,
                "baseline_score": baseline_score,
            },
        )

    async def _sequential_optimization(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Sequential optimization: first one, then the other.

        By default, optimizes hyperparameters first, then prompts.
        """
        start_time = time.time()
        eval_set = validation_set or dataset

        # Get baseline score
        baseline_score, _ = await self.evaluate(module, eval_set)

        # First: optimize hyperparameters
        # Set up search space if param_space provided
        if "param_space" in kwargs and kwargs["param_space"]:
            from ..core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace

            param_specs = {}
            for name, value_range in kwargs["param_space"].items():
                if isinstance(value_range, tuple) and len(value_range) == 2:
                    param_specs[name] = ParamSpec(
                        name=name,
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description=f"Hyperparameter {name}",
                        default=(value_range[0] + value_range[1]) / 2,
                        range=value_range,
                    )

            if param_specs:
                self.hyper_opt.search_space = SearchSpace(param_specs)

        hyper_result = await self.hyper_opt.optimize(module, dataset, validation_set)
        intermediate = hyper_result.optimized_module
        hyper_score, _ = await self.evaluate(intermediate, eval_set)

        # Extract best config from hyperparameter optimization
        best_config = {}
        if hyper_result.metadata and "best_config" in hyper_result.metadata:
            best_config = hyper_result.metadata["best_config"] or {}

        # Second: optimize prompts with optimal hyperparameters
        prompt_result = await self.prompt_opt.optimize(intermediate, dataset, validation_set)
        final = prompt_result.optimized_module
        final_score, _ = await self.evaluate(final, eval_set)

        return OptimizationResult(
            optimized_module=final,
            improvement=final_score - baseline_score,
            iterations=2,  # Two sequential steps
            best_score=final_score,
            optimization_time=time.time() - start_time,
            metadata={
                "strategy": "sequential",
                "baseline_score": baseline_score,
                "after_hyperopt_score": hyper_score,
                "final_score": final_score,
                "hyperopt_improvement": hyper_score - baseline_score,
                "prompt_improvement": final_score - hyper_score,
                "best_config": best_config,  # Include the best hyperparameter configuration
            },
        )

    def _has_converged(self, scores: list[float]) -> bool:
        """Check if optimization has converged."""
        if len(scores) < 3:
            return False

        # Check if recent improvements are below threshold
        recent_improvements = []
        for i in range(len(scores) - 2, len(scores)):
            improvement = abs(scores[i] - scores[i - 1])
            recent_improvements.append(improvement)

        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        return avg_improvement < self.config.convergence_threshold

    def _create_joint_search_space(
        self, module: Module, param_space: dict[str, Any] | None = None
    ) -> SearchSpace:
        """Create unified search space for joint optimization."""
        from ..core.parameters import ParamDomain, ParamSpec, ParamType

        param_specs = {}

        # If param_space is provided, use it for hyperparameters
        if param_space:
            for name, value_range in param_space.items():
                if isinstance(value_range, tuple) and len(value_range) == 2:
                    param_specs[name] = ParamSpec(
                        name=name,
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description=f"Hyperparameter {name}",
                        default=(value_range[0] + value_range[1]) / 2,
                        range=value_range,
                    )
        else:
            # Add hyperparameter specs from provider
            if hasattr(module, "provider") and module.provider:
                if hasattr(module.provider, "get_param_specs"):
                    # Type checking: provider has get_param_specs method
                    provider = module.provider
                    if callable(getattr(provider, "get_param_specs", None)):
                        param_specs.update(provider.get_param_specs())  # type: ignore[attr-defined]

        # Add prompt-related parameters
        # Number of demonstrations
        param_specs["num_demos"] = ParamSpec(
            name="num_demos",
            param_type=ParamType.INT,
            domain=ParamDomain.GENERATION,
            description="Number of demonstrations to use",
            default=4,
            range=self.config.num_demos_range,
        )

        # Instruction style (simplified as categorical for joint optimization)
        param_specs["instruction_style"] = ParamSpec(
            name="instruction_style",
            param_type=ParamType.CATEGORICAL,
            domain=ParamDomain.GENERATION,
            description="Style of instruction",
            default="concise",
            choices=self.config.instruction_styles,
        )

        return SearchSpace(param_specs)

    async def _apply_joint_config(
        self, module: Module, config: dict[str, Any], dataset: list[dict[str, Any]]
    ) -> Module:
        """Apply joint configuration to module."""
        result = copy.deepcopy(module)

        # Apply hyperparameters
        hyperparam_config = {
            k: v for k, v in config.items() if k not in ["num_demos", "instruction_style"]
        }
        if hyperparam_config:
            result.config.update(hyperparam_config)

        # Apply prompt parameters
        if "num_demos" in config and config["num_demos"] > 0 and dataset:
            # Use bootstrap to get demonstrations
            try:
                demo_optimizer = BootstrapFewShot(
                    metric=self.metric,
                    max_bootstrapped_demos=config["num_demos"],
                    max_labeled_demos=config["num_demos"],
                )
                # Use configurable subset size
                subset_size = (
                    self.config.demo_subset_size if self.config.max_demo_subset else len(dataset)
                )
                demo_result = await demo_optimizer.optimize(
                    result, dataset[: min(subset_size, len(dataset))]
                )
                result = demo_result.optimized_module
            except Exception as e:
                # Log the error if bootstrap fails
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Bootstrap demonstration generation failed: {e}")
                # Continue without demonstrations

        if "instruction_style" in config:
            # Apply instruction style
            style = config["instruction_style"]
            if hasattr(result, "signature") and result.signature:
                if style == "detailed":
                    result.signature.instructions = (
                        f"Please carefully {result.signature.instructions}. "
                        "Provide thorough analysis."
                    )
                elif style == "step_by_step":
                    result.signature.instructions = (
                        f"Step by step, {result.signature.instructions}. Think through each step."
                    )
                elif style == "formal":
                    result.signature.instructions = (
                        f"You are required to {result.signature.instructions}. Ensure accuracy."
                    )

        return result


__all__ = ["HybridOptimizer"]
