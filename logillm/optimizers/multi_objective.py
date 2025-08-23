"""Multi-objective optimizer for balancing multiple goals simultaneously.

Optimizes for multiple objectives like accuracy, latency, and cost at once.
"""

import copy
import time
from typing import Any, Callable, Optional

from ..core.modules import Module
from ..core.optimizers import Metric, Optimizer
from ..core.types import OptimizationResult, OptimizationStrategy, Usage
from ..exceptions import OptimizationError


class MultiObjectiveOptimizer(Optimizer):
    """Optimizes for multiple objectives simultaneously.

    Examples of objectives:
    - Accuracy: How correct the outputs are
    - Latency: How fast the model responds
    - Cost: Token usage and API costs
    - Consistency: Variance in outputs
    - Safety: Avoiding harmful outputs

    Uses Pareto frontier tracking to find non-dominated solutions
    that represent different trade-offs between objectives.
    """

    def __init__(
        self,
        metrics: dict[str, Metric | Callable],
        weights: Optional[dict[str, float]] = None,
        maintain_pareto: bool = True,
        n_trials: int = 50,
        pareto_size: int = 20,
        strategy: str = "weighted",  # weighted, pareto, constraint
        constraints: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ):
        """Initialize MultiObjectiveOptimizer.

        Args:
            metrics: Dictionary of objective names to metrics
            weights: Weights for each objective (for weighted strategy)
            maintain_pareto: Whether to maintain Pareto frontier
            n_trials: Number of optimization trials
            pareto_size: Maximum size of Pareto frontier
            strategy: Multi-objective strategy (weighted, pareto, constraint)
            constraints: Constraints for each objective (min values)
        """
        # Use first metric as primary for base class
        primary_metric = list(metrics.values())[0]
        super().__init__(
            strategy=OptimizationStrategy.MULTI_OBJECTIVE,
            metric=primary_metric,
        )

        self.metrics = metrics
        self.weights = weights or dict.fromkeys(metrics, 1.0)
        self.maintain_pareto = maintain_pareto
        self.n_trials = n_trials
        self.pareto_size = pareto_size
        self.optimization_strategy = strategy
        self.constraints = constraints or {}

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Initialize Pareto frontier
        self.pareto_frontier = [] if maintain_pareto else None

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize for multiple objectives.

        Args:
            module: Module to optimize
            dataset: Training dataset
            validation_set: Validation dataset
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with Pareto-optimal or weighted-best module
        """
        start_time = time.time()
        eval_set = validation_set or dataset

        # Get baseline scores
        baseline_scores = await self._evaluate_all_objectives(module, eval_set)

        # Create search space
        search_space = self._create_multi_objective_search_space(module)

        # Initialize search strategy
        from .search_strategies import SimpleBayesianStrategy, StrategyConfig

        search_strategy = SimpleBayesianStrategy(config=StrategyConfig(n_warmup=10))
        search_strategy.initialize(search_space)

        # Track all evaluations
        all_evaluations = []
        best_weighted_module = None
        best_weighted_score = -float("inf")

        for _trial in range(self.n_trials):
            # Get next configuration
            config = search_strategy.suggest_next()

            # Apply configuration
            test_module = self._apply_config(module, config)

            # Evaluate all objectives
            scores = await self._evaluate_all_objectives(test_module, eval_set)

            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(scores)

            # Track all evaluations (for constraint strategy)
            all_evaluations.append(
                {
                    "config": config,
                    "scores": scores,
                    "weighted_score": weighted_score,
                    "module": test_module,
                }
            )

            # Check constraints
            if self._satisfies_constraints(scores):
                # Update best weighted
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_weighted_module = copy.deepcopy(test_module)

                # Update Pareto frontier
                if self.pareto_frontier is not None:
                    self._update_pareto_frontier(test_module, scores)

            # Update search strategy with weighted score
            search_strategy.update(config, weighted_score)

        # Select final module based on strategy
        if self.optimization_strategy == "weighted":
            final_module = best_weighted_module
            final_scores = await self._evaluate_all_objectives(final_module, eval_set)
        elif self.optimization_strategy == "pareto":
            final_module = self._select_from_pareto()
            final_scores = await self._evaluate_all_objectives(final_module, eval_set)
        elif self.optimization_strategy == "constraint":
            final_module = self._select_best_constrained(all_evaluations)
            final_scores = await self._evaluate_all_objectives(final_module, eval_set)
        else:
            raise ValueError(f"Unknown strategy: {self.optimization_strategy}")

        # Calculate improvements
        improvements = {name: final_scores[name] - baseline_scores[name] for name in self.metrics}

        return OptimizationResult(
            optimized_module=final_module,
            improvement=sum(improvements.values()) / len(improvements),  # Average improvement
            iterations=self.n_trials,
            best_score=best_weighted_score,
            optimization_time=time.time() - start_time,
            metadata={
                "final_scores": final_scores,
                "baseline_scores": baseline_scores,
                "improvements": improvements,
                "pareto_frontier_size": len(self.pareto_frontier) if self.pareto_frontier else 0,
                "strategy": self.optimization_strategy,
                "weights": self.weights,
                "constraints": self.constraints,
            },
        )

    async def _evaluate_all_objectives(
        self, module: Module, dataset: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate module on all objectives."""
        scores = {}

        for name, metric in self.metrics.items():
            if name == "latency":
                # Special handling for latency
                scores[name] = await self._evaluate_latency(module, dataset)
            elif name == "cost":
                # Special handling for cost
                scores[name] = await self._evaluate_cost(module, dataset)
            elif name == "consistency":
                # Special handling for consistency
                scores[name] = await self._evaluate_consistency(module, dataset)
            else:
                # Standard accuracy-like metric
                score, _ = await self.evaluate_with_metric(module, dataset, metric)
                scores[name] = score

        return scores

    async def evaluate_with_metric(
        self, module: Module, dataset: list[dict[str, Any]], metric: Metric | Callable
    ) -> tuple[float, list]:
        """Evaluate with a specific metric."""
        scores = []
        traces = []

        for example in dataset:
            try:
                inputs = example.get("inputs", {})
                expected = example.get("outputs", {})

                prediction = await module(**inputs)

                # Calculate metric score
                if hasattr(prediction, "outputs"):
                    score = metric(prediction.outputs, expected)
                else:
                    score = metric(prediction, expected)

                scores.append(score)
                traces.append(prediction)

            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Evaluation failed for example: {e}")
                scores.append(0.0)
                traces.append(None)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, traces

    async def _evaluate_latency(self, module: Module, dataset: list[dict[str, Any]]) -> float:
        """Evaluate latency objective (lower is better)."""
        latencies = []

        for example in dataset[:10]:  # Sample for latency testing
            inputs = example.get("inputs", {})

            start = time.time()
            try:
                await module(**inputs)
                latency = time.time() - start
                latencies.append(latency)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Latency evaluation failed: {e}")
                latencies.append(10.0)  # Penalty for failures

        # Return negative average latency (so higher is better)
        avg_latency = sum(latencies) / len(latencies) if latencies else 10.0
        return 1.0 / (1.0 + avg_latency)  # Convert to 0-1 range, higher is better

    async def _evaluate_cost(self, module: Module, dataset: list[dict[str, Any]]) -> float:
        """Evaluate cost objective (lower is better)."""
        total_tokens = 0

        for example in dataset[:10]:  # Sample for cost estimation
            inputs = example.get("inputs", {})

            try:
                prediction = await module(**inputs)

                # Extract token usage
                if hasattr(prediction, "usage") and isinstance(prediction.usage, Usage):
                    total_tokens += prediction.usage.tokens.total_tokens
                else:
                    total_tokens += 100  # Default estimate

            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Cost evaluation failed: {e}")
                total_tokens += 100  # Penalty

        # Convert to cost score (higher is better, lower cost)
        # Assuming $0.01 per 1K tokens
        cost = total_tokens * 0.00001
        return 1.0 / (1.0 + cost)  # Convert to 0-1 range

    async def _evaluate_consistency(self, module: Module, dataset: list[dict[str, Any]]) -> float:
        """Evaluate output consistency (lower variance is better)."""
        # Test same input multiple times
        if not dataset:
            return 0.0

        test_example = dataset[0]
        inputs = test_example.get("inputs", {})

        outputs = []
        for _ in range(3):  # Run 3 times
            try:
                prediction = await module(**inputs)
                if hasattr(prediction, "outputs"):
                    outputs.append(str(prediction.outputs))
                else:
                    outputs.append(str(prediction))
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Consistency evaluation failed: {e}")
                outputs.append("")

        # Calculate consistency (1.0 if all same, lower if different)
        if len(set(outputs)) == 1:
            return 1.0  # Perfect consistency
        else:
            # Calculate similarity ratio
            unique_outputs = len(set(outputs))
            return 1.0 / unique_outputs

    def _calculate_weighted_score(self, scores: dict[str, float]) -> float:
        """Calculate weighted combination of objectives."""
        weighted_sum = 0.0
        for name, score in scores.items():
            weight = self.weights.get(name, 1.0)
            weighted_sum += weight * score
        return weighted_sum

    def _satisfies_constraints(self, scores: dict[str, float]) -> bool:
        """Check if scores satisfy constraints."""
        for name, min_value in self.constraints.items():
            if name in scores and scores[name] < min_value:
                return False
        return True

    def _update_pareto_frontier(self, module: Module, scores: dict[str, float]):
        """Update Pareto frontier with new solution."""
        # Check if dominated by any existing
        dominated = False
        for existing in self.pareto_frontier:
            if self._dominates(existing["scores"], scores):
                dominated = True
                break

        if not dominated:
            # Remove any dominated by new solution
            self.pareto_frontier = [
                pf for pf in self.pareto_frontier if not self._dominates(scores, pf["scores"])
            ]

            # Add new solution
            self.pareto_frontier.append(
                {
                    "module": copy.deepcopy(module),
                    "scores": scores,
                    "weighted_score": self._calculate_weighted_score(scores),
                }
            )

        # Limit size (keep best weighted scores)
        if len(self.pareto_frontier) > self.pareto_size:
            self.pareto_frontier.sort(key=lambda x: x["weighted_score"], reverse=True)
            self.pareto_frontier = self.pareto_frontier[: self.pareto_size]

    def _dominates(self, scores1: dict[str, float], scores2: dict[str, float]) -> bool:
        """Check if scores1 dominates scores2 (better or equal on all, better on at least one)."""
        better_on_at_least_one = False

        for name in self.metrics:
            if name not in scores1 or name not in scores2:
                continue

            if scores1[name] < scores2[name]:
                return False  # Worse on this objective
            elif scores1[name] > scores2[name]:
                better_on_at_least_one = True

        return better_on_at_least_one

    def _select_from_pareto(self) -> Module:
        """Select best module from Pareto frontier."""
        if not self.pareto_frontier:
            raise OptimizationError(
                "No solutions in Pareto frontier", optimizer_type=self.strategy.value
            )

        # Select based on weighted score
        best = max(self.pareto_frontier, key=lambda x: x["weighted_score"])
        return best["module"]

    def _select_best_constrained(self, evaluations: list[dict[str, Any]]) -> Module:
        """Select best module that satisfies constraints."""
        # Filter to those satisfying constraints
        valid = [e for e in evaluations if self._satisfies_constraints(e["scores"])]

        if not valid:
            # Relax constraints and find best effort
            if evaluations:
                return max(evaluations, key=lambda x: x["weighted_score"])["module"]
            else:
                raise OptimizationError(
                    "No valid solutions found", optimizer_type=self.strategy.value
                )

        # Return best among valid
        best = max(valid, key=lambda x: x["weighted_score"])
        return best["module"]

    def _create_multi_objective_search_space(self, module: Module):
        """Create search space for multi-objective optimization."""
        from ..core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace

        param_specs = {}

        # Hyperparameters that affect different objectives

        # Temperature affects accuracy vs creativity
        param_specs["temperature"] = ParamSpec(
            name="temperature",
            param_type=ParamType.FLOAT,
            domain=ParamDomain.GENERATION,
            description="Temperature for randomness",
            default=0.7,
            range=(0.0, 2.0),
        )

        # Top-p affects consistency
        param_specs["top_p"] = ParamSpec(
            name="top_p",
            param_type=ParamType.FLOAT,
            domain=ParamDomain.GENERATION,
            description="Nucleus sampling threshold",
            default=0.9,
            range=(0.1, 1.0),
        )

        # Max tokens affects cost and latency
        param_specs["max_tokens"] = ParamSpec(
            name="max_tokens",
            param_type=ParamType.INT,
            domain=ParamDomain.GENERATION,
            description="Maximum output tokens",
            default=150,
            range=(50, 500),
        )

        # Number of demonstrations affects accuracy but increases cost
        param_specs["num_demos"] = ParamSpec(
            name="num_demos",
            param_type=ParamType.INT,
            domain=ParamDomain.GENERATION,
            description="Number of demonstrations",
            default=3,
            range=(0, 8),
        )

        return SearchSpace(param_specs)

    def _apply_config(self, module: Module, config: dict[str, Any]) -> Module:
        """Apply configuration to module."""
        result = copy.deepcopy(module)

        # Apply hyperparameters
        if hasattr(result, "config"):
            for key in ["temperature", "top_p", "max_tokens"]:
                if key in config:
                    result.config[key] = config[key]

        # Apply demonstration count (simplified)
        if "num_demos" in config and hasattr(result, "parameters"):
            # This is simplified - in practice would bootstrap demos
            from ..core.modules import Parameter

            result.parameters["demo_count"] = Parameter(
                value=config["num_demos"], learnable=False, metadata={"source": "multi_objective"}
            )

        return result

    def get_pareto_frontier(self) -> list[dict[str, Any]]:
        """Get current Pareto frontier."""
        if self.pareto_frontier is None:
            return []
        return [
            {"scores": pf["scores"], "weighted_score": pf["weighted_score"]}
            for pf in self.pareto_frontier
        ]

    def plot_pareto_frontier(self, objective1: str, objective2: str) -> None:
        """Plot Pareto frontier for two objectives (requires matplotlib)."""
        if not self.pareto_frontier:
            print("No Pareto frontier to plot")
            return

        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]

            # Extract scores
            x = [pf["scores"][objective1] for pf in self.pareto_frontier]
            y = [pf["scores"][objective2] for pf in self.pareto_frontier]

            # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, s=100, alpha=0.7)
            plt.xlabel(objective1)
            plt.ylabel(objective2)
            plt.title(f"Pareto Frontier: {objective1} vs {objective2}")
            plt.grid(True, alpha=0.3)
            plt.show()

        except ImportError:
            print("Matplotlib not installed. Cannot plot Pareto frontier.")


__all__ = ["MultiObjectiveOptimizer"]
