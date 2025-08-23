"""LabeledFewShot optimizer - simplest baseline that just adds demos."""

import copy
from typing import Any, Optional

from ..core.modules import Module, Parameter
from ..core.optimizers import Metric
from ..core.types import OptimizationResult, OptimizationStrategy
from .base import Demonstration, DemoSelector, PromptOptimizationConfig, PromptOptimizer
from .demo_selectors import BestDemoSelector


class LabeledFewShot(PromptOptimizer):
    """Simplest optimizer - just adds provided demonstrations.

    This is not really an optimizer but provides the baseline
    for demo management. It simply adds labeled examples to the module.

    This matches DSPy's LabeledFewShot behavior: no learning,
    just adding examples as context.
    """

    def __init__(
        self,
        metric: Metric,
        max_demos: int = 4,
        demo_selector: Optional[DemoSelector] = None,
        config: Optional[PromptOptimizationConfig] = None,
        **kwargs: Any,
    ):
        """Initialize LabeledFewShot.

        Args:
            metric: Evaluation metric
            max_demos: Maximum number of demonstrations to add
            demo_selector: Strategy for selecting demos (default: BestDemoSelector)
            config: Optimization configuration
        """
        config = config or PromptOptimizationConfig(
            strategy=OptimizationStrategy.BOOTSTRAP, max_demos=max_demos
        )
        super().__init__(strategy=OptimizationStrategy.BOOTSTRAP, metric=metric, config=config)
        self.max_demos = max_demos
        self.demo_selector = demo_selector or BestDemoSelector()

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Add demonstrations to module.

        This simply takes examples from the dataset and adds them
        as demonstrations to the module. No actual optimization occurs.

        Args:
            module: Module to optimize
            dataset: Training examples to use as demonstrations
            validation_set: Optional validation set for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with module containing demonstrations
        """
        import time

        start_time = time.time()

        # Create demonstrations from dataset
        demonstrations = []
        for example in dataset[: self.max_demos * 2]:  # Take more than needed for selection
            demo = Demonstration.from_example(example)
            # Score is 1.0 for labeled data (assumed correct)
            demo.score = 1.0
            demonstrations.append(demo)

        # Select demonstrations
        selected_demos = self.demo_selector.select(demonstrations, self.max_demos)

        # Create new module with demonstrations
        optimized_module = copy.deepcopy(module)

        # Add demonstrations to the demo_manager (for Predict modules)
        if hasattr(optimized_module, "demo_manager"):
            # Clear existing demos and add new ones
            demo_manager = optimized_module.demo_manager
            demo_manager.clear()  # type: ignore[attr-defined]
            for demo in selected_demos:
                demo_manager.add({"inputs": demo.inputs, "outputs": demo.outputs})  # type: ignore[attr-defined]

        # Also add demonstrations as a parameter for tracking
        demo_param = Parameter(
            value=[d.to_dict() for d in selected_demos],
            learnable=True,
            metadata={
                "type": "demonstrations",
                "source": "labeled",
                "selection_strategy": self.demo_selector.__class__.__name__,
            },
        )
        optimized_module.parameters["demonstrations"] = demo_param

        # Evaluate if validation set provided
        final_score = 0.0
        eval_traces = []
        if validation_set:
            final_score, eval_traces = await self.evaluate(optimized_module, validation_set)

        # Also evaluate baseline for improvement calculation
        baseline_score = 0.0
        if validation_set:
            baseline_score, _ = await self.evaluate(module, validation_set)

        return OptimizationResult(
            optimized_module=optimized_module,
            improvement=final_score - baseline_score,
            iterations=1,  # Single pass
            best_score=final_score,
            optimization_time=time.time() - start_time,
            metadata={
                "num_demos": len(selected_demos),
                "demo_scores": [d.score for d in selected_demos],
                "baseline_score": baseline_score,
                "selection_strategy": self.demo_selector.__class__.__name__,
            },
        )


__all__ = ["LabeledFewShot"]
