"""Random search prompt optimizer - tries random prompt variations."""

import copy
import random
from typing import Any, Callable, Optional

from ..core.modules import Module, Parameter
from ..core.optimizers import Metric
from ..core.types import OptimizationResult, OptimizationStrategy
from ..exceptions import OptimizationError
from .base import PromptOptimizationConfig, PromptOptimizer


class RandomPromptOptimizer(PromptOptimizer):
    """Random search over prompt variations.

    Generates random prompt variations and selects the best one.
    This is a simple baseline for prompt optimization.

    Strategies:
    - Template-based: Uses predefined templates with variations
    - Paraphrase: Rephrases instructions in different ways
    - Augmentation: Adds/removes prompt components
    """

    def __init__(
        self,
        metric: Metric,
        num_candidates: int = 10,
        prompt_generator: Optional[Callable] = None,
        include_demos: bool = False,
        config: Optional[PromptOptimizationConfig] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize RandomPromptOptimizer.

        Args:
            metric: Evaluation metric
            num_candidates: Number of random prompts to try
            prompt_generator: Function to generate random prompts
            include_demos: Whether to also randomly select demonstrations
            config: Optimization configuration
            seed: Random seed for reproducibility
        """
        config = config or PromptOptimizationConfig(
            strategy=OptimizationStrategy.INSTRUCTION, max_iterations=num_candidates
        )
        super().__init__(strategy=OptimizationStrategy.INSTRUCTION, metric=metric, config=config)
        self.num_candidates = num_candidates
        self.prompt_generator = prompt_generator or self._default_prompt_generator
        self.include_demos = include_demos
        self.rng = random.Random(seed)

    def _default_prompt_generator(self, base_instruction: str = "") -> str:
        """Generate a random prompt variation using templates."""
        templates = [
            "Please {task}",
            "You should {task}",
            "Your goal is to {task}",
            "{task}. Be precise.",
            "Carefully {task}",
            "Step by step, {task}",
            "{task}. Think carefully.",
            "Given the input, {task}",
            "Your task: {task}",
            "I need you to {task}",
            "{task}. Show your work.",
            "Let's {task}",
            "Can you {task}?",
            "{task}. Be thorough.",
            "Please help me {task}",
        ]

        # Add variation with prefixes
        prefixes = [
            "",
            "Important: ",
            "Note: ",
            "Task: ",
            "Objective: ",
        ]

        # Add variation with suffixes
        suffixes = [
            "",
            " Be accurate.",
            " Think step by step.",
            " Be concise.",
            " Provide details.",
            " Double-check your work.",
        ]

        # Use base instruction or extract from it
        if base_instruction:
            # Try to extract the core task
            task = base_instruction.lower()
            # Remove common prefixes
            for prefix in ["please ", "you should ", "your goal is to "]:
                if task.startswith(prefix):
                    task = task[len(prefix) :]
                    break
        else:
            task = "complete the task"

        # Select random components
        template = self.rng.choice(templates)
        prefix = self.rng.choice(prefixes)
        suffix = self.rng.choice(suffixes)

        # Combine
        prompt = prefix + template.format(task=task) + suffix
        return prompt.strip()

    def _generate_demo_variations(
        self, dataset: list[dict[str, Any]], n_demos: int
    ) -> list[list[dict[str, Any]]]:
        """Generate different demo selections."""
        if len(dataset) <= n_demos:
            return [dataset]  # Only one possibility

        variations = []
        for _ in range(min(5, self.num_candidates // 2)):  # Generate up to 5 demo variations
            selected = self.rng.sample(dataset, n_demos)
            variations.append(selected)

        return variations

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize by trying random prompt variations.

        Args:
            module: Module to optimize
            dataset: Training dataset
            validation_set: Validation dataset for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with best random prompt
        """
        import time

        start_time = time.time()

        # Get base instruction if available
        base_instruction = self._extract_instruction(module)

        # Generate and evaluate candidates
        candidates = []
        eval_set = validation_set or dataset

        # Generate demo variations if requested
        demo_variations = []
        if self.include_demos and dataset:
            n_demos = getattr(self.config, "max_demos", 4)
            demo_variations = self._generate_demo_variations(dataset, n_demos)

        for i in range(self.num_candidates):
            # Generate random prompt
            prompt = self.prompt_generator(base_instruction)

            # Create module variant with new prompt
            variant = copy.deepcopy(module)

            # Add instruction as parameter
            instruction_param = Parameter(
                value=prompt, learnable=True, metadata={"type": "instruction", "iteration": i}
            )
            variant.parameters["instruction"] = instruction_param

            # Update signature if available
            if hasattr(variant, "signature") and variant.signature:
                variant.signature.instructions = prompt

            # Optionally add random demos
            if demo_variations:
                demos = self.rng.choice(demo_variations)
                demo_param = Parameter(
                    value=demos,
                    learnable=True,
                    metadata={"type": "demonstrations", "source": "random"},
                )
                variant.parameters["demonstrations"] = demo_param

            # Evaluate
            score, traces = await self.evaluate(variant, eval_set)

            candidates.append(
                {
                    "module": variant,
                    "prompt": prompt,
                    "score": score,
                    "demos": len(demo_variations) if demo_variations else 0,
                    "traces": traces,
                }
            )

        # Select best
        if not candidates:
            raise OptimizationError("No candidates generated", optimizer_type=self.strategy.value)

        best = max(candidates, key=lambda c: c["score"])

        # Calculate improvement
        baseline_score = 0.0
        if validation_set:
            baseline_score, _ = await self.evaluate(module, validation_set)

        return OptimizationResult(
            optimized_module=best["module"],
            improvement=best["score"] - baseline_score,
            iterations=self.num_candidates,
            best_score=best["score"],
            optimization_time=time.time() - start_time,
            metadata={
                "best_prompt": best["prompt"],
                "all_scores": [c["score"] for c in candidates],
                "avg_score": sum(c["score"] for c in candidates) / len(candidates),
                "baseline_score": baseline_score,
                "included_demos": self.include_demos,
                "num_demo_variations": best["demos"],
            },
        )


__all__ = ["RandomPromptOptimizer"]
