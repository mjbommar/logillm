"""Optimizer abstractions for automatic improvement of modules."""

from __future__ import annotations

import copy
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar

from ..exceptions import OptimizationError
from ..protocols.runtime import Configurable, Monitorable
from .modules import Module, Parameter
from .types import (
    Configuration,
    ExecutionTrace,
    Metadata,
    OptimizationResult,
    OptimizationStrategy,
    TraceStep,
    Usage,
)

M = TypeVar("M", bound=Module)


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def __call__(self, prediction: Any, reference: Any, **kwargs: Any) -> float:
        """Compute metric score."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        ...

    def is_better(self, score1: float, score2: float) -> bool:
        """Check if score1 is better than score2."""
        # Default: higher is better
        return score1 > score2


class Trace:
    """Execution trace for learning."""

    def __init__(
        self,
        module: Module,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        score: float,
        metadata: Metadata | None = None,
    ):
        self.module = module
        self.inputs = inputs
        self.outputs = outputs
        self.score = score
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_execution_trace(self) -> ExecutionTrace:
        """Convert to ExecutionTrace format."""
        trace = ExecutionTrace()
        trace.add_step(
            TraceStep(
                module_name=self.module.__class__.__name__,
                inputs=self.inputs,
                outputs=self.outputs,
                usage=Usage(),
                success=self.score > 0,
                metadata=self.metadata,
            )
        )
        return trace


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    strategy: OptimizationStrategy = OptimizationStrategy.BOOTSTRAP
    max_iterations: int = 100
    target_score: float | None = None
    early_stopping: bool = True
    patience: int = 10
    batch_size: int = 10
    learning_rate: float = 0.1
    temperature: float = 1.0
    exploration_rate: float = 0.1
    metadata: Metadata = field(default_factory=dict)


class Optimizer(ABC, Configurable, Monitorable, Generic[M]):
    """Abstract base class for optimizers."""

    def __init__(
        self,
        strategy: OptimizationStrategy,
        metric: Metric,
        config: OptimizationConfig | None = None,
        verbose: bool = False,
    ):
        self.strategy = strategy
        self.metric = metric
        self.config = config or OptimizationConfig(strategy=strategy)
        self._metrics: dict[str, float] = {}
        self._best_score = float("-inf")
        self._iteration = 0
        self._start_time: datetime | None = None
        self.verbose = verbose
        self._log_step = self._create_logger() if verbose else lambda *args, **kwargs: None

    def _create_logger(self):
        """Create simple stdout logger using only stdlib."""

        def log(step: int, total: int, message: str, **kwargs):
            elapsed = (
                time.time() - self._start_time_seconds
                if hasattr(self, "_start_time_seconds")
                else 0.0
            )
            print(
                f"[{elapsed:6.1f}s] Step {step:3d}/{total:3d} | {message}",
                flush=True,
                file=sys.stdout,
            )
            if kwargs:
                for key, val in kwargs.items():
                    if isinstance(val, float):
                        print(f"    {key}: {val:.4f}", flush=True)
                    else:
                        print(f"    {key}: {val}", flush=True)

        return log

    @abstractmethod
    async def optimize(
        self,
        module: M,
        dataset: list[dict[str, Any]],
        validation_set: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize module on dataset."""
        ...

    def compile(self, module: M) -> M:
        """Compile module for optimization."""
        compiled = copy.deepcopy(module)
        compiled.state = module.state  # Preserve state
        return compiled

    async def evaluate(
        self,
        module: M,
        dataset: list[dict[str, Any]],
    ) -> tuple[float, list[Trace]]:
        """Evaluate module on dataset."""
        traces = []
        scores = []

        # Initialize start time if not set
        if not hasattr(self, "_start_time_seconds"):
            self._start_time_seconds = time.time()

        for i, example in enumerate(dataset):
            # Log evaluation progress for larger datasets
            if self.verbose and len(dataset) > 5 and i % max(1, len(dataset) // 10) == 0:
                self._log_step(i + 1, len(dataset), "Evaluating samples...")

            try:
                # Extract inputs and expected outputs
                inputs = example.get("inputs", {})
                expected = example.get("outputs", {})

                # Run module
                prediction = await module(**inputs)

                # Compute score
                score = self.metric(prediction.outputs, expected)
                scores.append(score)

                # Create trace
                trace = Trace(
                    module=module,
                    inputs=inputs,
                    outputs=prediction.outputs,
                    score=score,
                    metadata={"expected": expected},
                )
                traces.append(trace)

            except Exception as e:
                # Failed execution gets zero score
                scores.append(0.0)
                trace = Trace(
                    module=module, inputs=inputs, outputs={}, score=0.0, metadata={"error": str(e)}
                )
                traces.append(trace)

        # Return average score and traces
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, traces

    def should_stop(self, iteration: int, score: float) -> bool:
        """Check early stopping criteria."""
        if not self.config.early_stopping:
            return False

        # Check max iterations
        if iteration >= self.config.max_iterations:
            return True

        # Check target score
        if self.config.target_score and score >= self.config.target_score:
            return True

        # Could add patience logic here
        return False

    # Configurable protocol
    def configure(self, config: Configuration) -> None:
        """Apply configuration."""
        if isinstance(config, OptimizationConfig):
            self.config = config
        elif isinstance(config, dict):
            # Update config fields from dict
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            raise OptimizationError(
                "Invalid configuration type",
                optimizer_type=self.strategy.value,
                context={"config_type": type(config)},
            )

    def get_config(self) -> Configuration:
        """Get current configuration."""
        return self.config

    def validate_config(self, config: Configuration) -> bool:
        """Validate configuration."""
        if isinstance(config, OptimizationConfig):
            return True
        if isinstance(config, dict):
            # Check for valid keys
            valid_keys = {
                "strategy",
                "max_iterations",
                "target_score",
                "early_stopping",
                "patience",
                "batch_size",
                "learning_rate",
                "temperature",
                "exploration_rate",
            }
            return all(key in valid_keys for key in config.keys())
        return False

    # Monitorable protocol
    def get_metrics(self) -> dict[str, float]:
        """Get optimization metrics."""
        metrics = self._metrics.copy()
        metrics["best_score"] = self._best_score
        metrics["iteration"] = self._iteration
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            metrics["elapsed_time"] = elapsed
        return metrics

    def get_health_status(self) -> bool:
        """Get health status."""
        return True  # Optimizers are always "healthy"


class BootstrapOptimizer(Optimizer[M]):
    """Bootstrap few-shot optimizer."""

    def __init__(self, **kwargs):
        super().__init__(strategy=OptimizationStrategy.BOOTSTRAP, **kwargs)
        self.bootstrapped_demos: list[dict[str, Any]] = []

    async def optimize(
        self,
        module: M,
        dataset: list[dict[str, Any]],
        validation_set: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize using bootstrapped few-shot examples."""
        self._start_time = datetime.now()
        self._iteration = 0

        # Use module as teacher to generate examples
        teacher = self.compile(module)

        # Collect successful traces
        successful_demos = []

        for example in dataset[: self.config.batch_size]:
            try:
                inputs = example.get("inputs", {})
                expected = example.get("outputs", {})

                # Run teacher
                prediction = await teacher(**inputs)

                # Check if successful
                score = self.metric(prediction.outputs, expected)
                if score > 0.5:  # Threshold for "successful"
                    demo = {"inputs": inputs, "outputs": prediction.outputs, "score": score}
                    successful_demos.append(demo)

            except Exception:
                continue

        if not successful_demos:
            raise OptimizationError(
                "No successful demonstrations generated",
                optimizer_type=self.strategy.value,
                iteration=0,
            )

        # Create student module with demos
        student = self.compile(module)

        # Add demos as parameters
        demo_param = Parameter(
            value=successful_demos, learnable=True, metadata={"type": "demonstrations"}
        )
        student.parameters["demos"] = demo_param

        # Evaluate on validation set
        val_set = validation_set or dataset[self.config.batch_size :]
        final_score, _ = await self.evaluate(student, val_set)

        self._best_score = final_score
        self._metrics["num_demos"] = len(successful_demos)

        return OptimizationResult(
            optimized_module=student,
            improvement=final_score - 0.0,  # Assuming baseline is 0
            iterations=1,
            best_score=final_score,
            optimization_time=(datetime.now() - self._start_time).total_seconds(),
            metadata={"demos": successful_demos},
        )


class RandomSearchOptimizer(Optimizer[M]):
    """Random search optimizer."""

    async def optimize(
        self,
        module: M,
        dataset: list[dict[str, Any]],
        validation_set: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize using random search."""
        self._start_time = datetime.now()
        best_module = None
        best_score = float("-inf")

        for iteration in range(self.config.max_iterations):
            self._iteration = iteration

            # Create random variant
            variant = self.compile(module)

            # Randomly modify parameters
            for _name, param in variant.parameters.items():
                if param.learnable:
                    # Add random noise to parameter value
                    if isinstance(param.value, (int, float)):
                        noise = random.gauss(0, self.config.temperature)
                        param.value += noise
                    elif isinstance(param.value, str):
                        # Could do random string mutations
                        pass

            # Evaluate variant
            score, _ = await self.evaluate(variant, validation_set or dataset)

            # Track best
            if score > best_score:
                best_score = score
                best_module = variant
                self._best_score = score

            # Check stopping criteria
            if self.should_stop(iteration, best_score):
                break

        if best_module is None:
            raise OptimizationError("No valid module found", optimizer_type=self.strategy.value)

        return OptimizationResult(
            optimized_module=best_module,
            improvement=best_score - 0.0,
            iterations=self._iteration + 1,
            best_score=best_score,
            optimization_time=(datetime.now() - self._start_time).total_seconds(),
        )


# Concrete metric implementations


class ExactMatchMetric(Metric):
    """Exact match metric."""

    def __call__(self, prediction: Any, reference: Any, **kwargs: Any) -> float:
        """Check exact match."""
        # Handle dict outputs
        if isinstance(prediction, dict) and isinstance(reference, dict):
            # Check all keys match
            if set(prediction.keys()) != set(reference.keys()):
                return 0.0
            # Check all values match
            for key in reference.keys():
                if prediction.get(key) != reference.get(key):
                    return 0.0
            return 1.0
        else:
            return 1.0 if prediction == reference else 0.0

    def name(self) -> str:
        """Get metric name."""
        return "exact_match"


class AccuracyMetric(Metric):
    """Accuracy metric for classification."""

    def __init__(self, key: str | None = None):
        self.key = key

    def __call__(self, prediction: Any, reference: Any, **kwargs: Any) -> float:
        """Compute accuracy."""
        if self.key and isinstance(prediction, dict):
            pred_value = prediction.get(self.key)
            ref_value = reference.get(self.key) if isinstance(reference, dict) else reference
        else:
            pred_value = prediction
            ref_value = reference

        return 1.0 if pred_value == ref_value else 0.0

    def name(self) -> str:
        """Get metric name."""
        return f"accuracy_{self.key}" if self.key else "accuracy"


class F1Metric(Metric):
    """F1 score metric."""

    def __call__(self, prediction: Any, reference: Any, **kwargs: Any) -> float:
        """Compute F1 score."""
        # Simplified F1 for token overlap
        if isinstance(prediction, str) and isinstance(reference, str):
            pred_tokens = set(prediction.lower().split())
            ref_tokens = set(reference.lower().split())

            if not pred_tokens or not ref_tokens:
                return 0.0

            intersection = pred_tokens & ref_tokens
            precision = len(intersection) / len(pred_tokens)
            recall = len(intersection) / len(ref_tokens)

            if precision + recall == 0:
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            return f1

        # For other types, fall back to exact match
        return 1.0 if prediction == reference else 0.0

    def name(self) -> str:
        """Get metric name."""
        return "f1"


class CustomMetric(Metric):
    """Custom metric from callable."""

    def __init__(
        self,
        eval_fn: Callable[[Any, Any], float],
        metric_name: str = "custom",
    ):
        self.eval_fn = eval_fn
        self.metric_name = metric_name

    def __call__(self, prediction: Any, reference: Any, **kwargs: Any) -> float:
        """Apply custom metric."""
        return self.eval_fn(prediction, reference)

    def name(self) -> str:
        """Get metric name."""
        return self.metric_name


# Factory function
def create_optimizer(
    strategy: OptimizationStrategy | str, metric: Metric, **kwargs: Any
) -> Optimizer:
    """Create optimizer instance."""
    if isinstance(strategy, str):
        strategy = OptimizationStrategy(strategy)

    optimizer_map = {
        OptimizationStrategy.BOOTSTRAP: BootstrapOptimizer,
        OptimizationStrategy.INSTRUCTION: RandomSearchOptimizer,  # Placeholder
        OptimizationStrategy.ENSEMBLE: RandomSearchOptimizer,  # Placeholder
        OptimizationStrategy.EVOLUTION: RandomSearchOptimizer,  # Placeholder
        OptimizationStrategy.REFLECTION: RandomSearchOptimizer,  # Placeholder
        OptimizationStrategy.HYBRID: RandomSearchOptimizer,  # Placeholder
    }

    optimizer_class = optimizer_map.get(strategy, RandomSearchOptimizer)
    return optimizer_class(metric=metric, **kwargs)


__all__ = [
    "Metric",
    "Trace",
    "OptimizationConfig",
    "Optimizer",
    "BootstrapOptimizer",
    "RandomSearchOptimizer",
    "ExactMatchMetric",
    "AccuracyMetric",
    "F1Metric",
    "CustomMetric",
    "create_optimizer",
]
