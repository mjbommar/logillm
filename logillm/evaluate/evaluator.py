"""Main evaluation framework for LogiLLM.

Based on reference/dspy/notes.md lines 375-380.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.modules import Module

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""

    score: float  # Average score
    scores: list[float]  # Individual scores
    predictions: list[Any]  # Predictions made
    successes: int  # Number of successful predictions
    failures: int  # Number of failed predictions

    # Detailed results
    examples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Timing
    total_time: float = 0.0
    avg_time_per_example: float = 0.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"EvaluationResult(score={self.score:.3f}, "
            f"successes={self.successes}/{self.successes + self.failures})"
        )

    def summary(self) -> str:
        """Get detailed summary."""
        total = self.successes + self.failures
        return f"""
Evaluation Summary:
  Score: {self.score:.3f}
  Successes: {self.successes}/{total} ({100 * self.successes / total:.1f}%)
  Failures: {self.failures}/{total} ({100 * self.failures / total:.1f}%)
  Total Time: {self.total_time:.2f}s
  Avg Time: {self.avg_time_per_example:.3f}s
"""


class Evaluate:
    """Orchestrates evaluation over datasets with metrics.

    Based on DSPy's Evaluate class (reference/dspy/notes.md:375-380).
    Provides parallel execution, progress tracking, and detailed results.
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        metric: Callable[[dict, dict], float],
        num_threads: int = 1,
        display_progress: bool = True,
        display_results: bool = False,
        return_all_scores: bool = False,
    ):
        """Initialize evaluator.

        Args:
            dataset: List of examples with 'inputs' and 'outputs' keys
            metric: Metric function (pred_outputs, true_outputs) -> float
            num_threads: Number of parallel threads (1 for sequential)
            display_progress: Whether to show progress bar
            display_results: Whether to display results table
            return_all_scores: Whether to return all individual scores
        """
        self.dataset = dataset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_results = display_results
        self.return_all_scores = return_all_scores

    async def __call__(
        self,
        module: Module,
        dataset: Optional[list[dict[str, Any]]] = None,
    ) -> EvaluationResult:
        """Evaluate module on dataset.

        Args:
            module: Module to evaluate
            dataset: Optional dataset override

        Returns:
            EvaluationResult with scores and details
        """
        dataset = dataset or self.dataset

        if not dataset:
            raise ValueError("No dataset provided for evaluation")

        logger.info(f"Evaluating on {len(dataset)} examples")
        start_time = time.time()

        # Run evaluation
        if self.num_threads > 1:
            results = await self._parallel_evaluate(module, dataset)
        else:
            results = await self._sequential_evaluate(module, dataset)

        # Aggregate results
        scores = []
        predictions = []
        examples = []
        errors = []
        successes = 0
        failures = 0

        for result in results:
            if result["success"]:
                scores.append(result["score"])
                successes += 1
            else:
                scores.append(0.0)
                failures += 1
                if result.get("error"):
                    errors.append(result["error"])

            predictions.append(result.get("prediction"))

            if self.return_all_scores:
                examples.append(result)

        # Calculate statistics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        total_time = time.time() - start_time
        avg_time = total_time / len(dataset) if dataset else 0.0

        # Create result
        eval_result = EvaluationResult(
            score=avg_score,
            scores=scores if self.return_all_scores else [],
            predictions=predictions,
            successes=successes,
            failures=failures,
            examples=examples,
            errors=errors,
            total_time=total_time,
            avg_time_per_example=avg_time,
            metadata={
                "num_threads": self.num_threads,
                "dataset_size": len(dataset),
                "metric_name": getattr(self.metric, "__name__", "custom"),
            },
        )

        # Display results if requested
        if self.display_results:
            print(eval_result.summary())

        logger.info(f"Evaluation complete: {eval_result}")

        return eval_result

    async def _sequential_evaluate(
        self, module: Module, dataset: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Evaluate sequentially."""
        results = []

        for i, example in enumerate(dataset):
            if self.display_progress and i % 10 == 0:
                logger.info(f"Progress: {i}/{len(dataset)}")

            result = await self._evaluate_example(module, example)
            results.append(result)

        return results

    async def _parallel_evaluate(
        self, module: Module, dataset: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Evaluate in parallel."""
        # Create tasks
        tasks = []
        for example in dataset:
            task = self._evaluate_example(module, example)
            tasks.append(task)

        # Run with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.num_threads)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]

        # Gather results
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "score": 0.0,
                        "error": str(result),
                        "example_idx": i,
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _evaluate_example(self, module: Module, example: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single example."""
        result = {
            "success": False,
            "score": 0.0,
            "prediction": None,
            "example": example,
            "error": None,
        }

        try:
            # Get inputs and expected outputs
            inputs = example.get("inputs", {})
            expected_outputs = example.get("outputs", {})

            # Run module
            prediction = await module.forward(**inputs)

            if prediction.success:
                # Evaluate with metric
                score = self.metric(prediction.outputs, expected_outputs)

                result["success"] = True
                result["score"] = score
                result["prediction"] = prediction.outputs
            else:
                result["error"] = "Module prediction failed"

        except Exception as e:
            result["error"] = f"Evaluation error: {str(e)}"
            logger.debug(f"Error evaluating example: {e}")

        return result

    def run_sync(
        self, module: Module, dataset: Optional[list[dict[str, Any]]] = None
    ) -> EvaluationResult:
        """Synchronous wrapper for evaluation."""
        return asyncio.run(self(module, dataset))
