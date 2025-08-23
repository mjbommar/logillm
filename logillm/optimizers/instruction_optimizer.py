"""Instruction optimizer - optimizes instruction text based on task analysis."""

import copy
import random
from typing import Any, Callable, Optional

from ..core.modules import Module, Parameter
from ..core.optimizers import Metric
from ..core.types import OptimizationResult, OptimizationStrategy
from ..exceptions import OptimizationError
from .base import PromptOptimizationConfig, PromptOptimizer


class InstructionOptimizer(PromptOptimizer):
    """Optimizes just the instruction component of prompts.

    This optimizer focuses solely on improving the instruction text,
    using various strategies to generate and select better instructions.
    It analyzes the dataset to understand the task pattern and generates
    appropriate instructions.
    """

    def __init__(
        self,
        metric: Metric,
        num_candidates: int = 5,
        instruction_generator: Optional[Callable] = None,
        selection_strategy: str = "best",  # best, weighted, tournament
        analyze_examples: bool = True,
        config: Optional[PromptOptimizationConfig] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize InstructionOptimizer.

        Args:
            metric: Evaluation metric
            num_candidates: Number of instruction candidates to generate
            instruction_generator: Custom instruction generation function
            selection_strategy: How to select the best instruction
            analyze_examples: Whether to analyze dataset examples
            config: Optimization configuration
            seed: Random seed for reproducibility
        """
        config = config or PromptOptimizationConfig(
            strategy=OptimizationStrategy.INSTRUCTION, max_iterations=num_candidates
        )
        super().__init__(strategy=OptimizationStrategy.INSTRUCTION, metric=metric, config=config)
        self.num_candidates = num_candidates
        self.instruction_generator = instruction_generator or self._generate_instruction
        self.selection_strategy = selection_strategy
        self.analyze_examples = analyze_examples
        self.rng = random.Random(seed)

    def _analyze_task(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze examples to understand the task.

        Returns:
            Dictionary with task analysis:
            - input_keys: Set of input field names
            - output_keys: Set of output field names
            - input_types: Inferred types of inputs
            - output_types: Inferred types of outputs
            - patterns: Any detected patterns
        """
        if not examples:
            return {
                "input_keys": set(),
                "output_keys": set(),
                "input_types": {},
                "output_types": {},
                "patterns": [],
            }

        # Collect all keys
        input_keys = set()
        output_keys = set()
        input_types = {}
        output_types = {}

        for ex in examples[:10]:  # Analyze first 10 examples
            if "inputs" in ex:
                for key, value in ex["inputs"].items():
                    input_keys.add(key)
                    if key not in input_types:
                        input_types[key] = type(value).__name__

            if "outputs" in ex:
                for key, value in ex["outputs"].items():
                    output_keys.add(key)
                    if key not in output_types:
                        output_types[key] = type(value).__name__

        # Detect patterns
        patterns = []

        # Check if it's a transformation task
        if input_keys and output_keys:
            if len(input_keys) == 1 and len(output_keys) == 1:
                patterns.append("single_transform")
            elif len(output_keys) > len(input_keys):
                patterns.append("expansion")
            elif len(output_keys) < len(input_keys):
                patterns.append("reduction")

        # Check for specific task types
        common_nlp_keys = {"text", "question", "context", "prompt", "query"}
        if input_keys & common_nlp_keys:
            patterns.append("nlp_task")

        if "label" in output_keys or "class" in output_keys:
            patterns.append("classification")

        if "answer" in output_keys:
            patterns.append("qa_task")

        return {
            "input_keys": input_keys,
            "output_keys": output_keys,
            "input_types": input_types,
            "output_types": output_types,
            "patterns": patterns,
        }

    def _generate_instruction(self, task_analysis: dict[str, Any], iteration: int = 0) -> str:
        """Generate an instruction based on task analysis.

        Args:
            task_analysis: Analysis of the task from examples
            iteration: Current iteration number for variation

        Returns:
            Generated instruction text
        """
        input_keys = task_analysis.get("input_keys", set())
        output_keys = task_analysis.get("output_keys", set())
        patterns = task_analysis.get("patterns", [])

        # Handle empty analysis
        if not input_keys and not output_keys:
            return "Complete the following task accurately."

        # Create key descriptions
        input_desc = ", ".join(sorted(input_keys)) if input_keys else "the input"
        output_desc = ", ".join(sorted(output_keys)) if output_keys else "the output"

        # Generate instructions based on patterns
        instructions = []

        # Base templates
        base_templates = [
            f"Given {input_desc}, produce {output_desc}.",
            f"Process {input_desc} to generate {output_desc}.",
            f"Analyze {input_desc} and determine {output_desc}.",
            f"Based on {input_desc}, provide {output_desc}.",
            f"Transform {input_desc} into {output_desc}.",
        ]

        # Pattern-specific templates
        if "classification" in patterns:
            instructions.extend(
                [
                    f"Classify {input_desc} to determine {output_desc}.",
                    f"Categorize {input_desc} and output {output_desc}.",
                    f"Identify the correct {output_desc} for {input_desc}.",
                ]
            )

        if "qa_task" in patterns:
            instructions.extend(
                [
                    f"Answer questions based on {input_desc}.",
                    f"Using {input_desc}, provide {output_desc}.",
                    f"Read {input_desc} and answer with {output_desc}.",
                ]
            )

        if "single_transform" in patterns:
            instructions.extend(
                [
                    f"Convert {input_desc} to {output_desc}.",
                    f"Map {input_desc} to corresponding {output_desc}.",
                    f"Translate {input_desc} into {output_desc}.",
                ]
            )

        if "nlp_task" in patterns:
            instructions.extend(
                [
                    f"Process the text in {input_desc} to produce {output_desc}.",
                    f"Analyze the textual {input_desc} and generate {output_desc}.",
                    f"Understanding {input_desc}, create appropriate {output_desc}.",
                ]
            )

        # If no specific patterns, use base templates
        if not instructions:
            instructions = base_templates

        # Select instruction based on iteration
        if iteration < len(instructions):
            base = instructions[iteration]
        else:
            # Random selection with seed based on iteration
            local_rng = random.Random(iteration)
            base = local_rng.choice(instructions)

        # Add modifiers for variation
        modifiers = [
            "",
            " Be precise.",
            " Think step by step.",
            " Ensure accuracy.",
            " Follow the pattern.",
            " Be consistent.",
            " Pay attention to details.",
        ]

        modifier_idx = iteration % len(modifiers)
        instruction = base + modifiers[modifier_idx]

        return instruction.strip()

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize the instruction component.

        Args:
            module: Module to optimize
            dataset: Training dataset to analyze
            validation_set: Validation set for evaluation
            **kwargs: Additional arguments (e.g., task_description)

        Returns:
            OptimizationResult with optimized instruction
        """
        import time

        start_time = time.time()

        # Analyze task from dataset if requested
        task_analysis = {}
        if self.analyze_examples:
            task_analysis = self._analyze_task(dataset)

        # Allow override via kwargs
        if "task_analysis" in kwargs:
            task_analysis.update(kwargs["task_analysis"])

        # Generate candidate instructions
        candidates = []
        eval_set = validation_set or dataset

        for i in range(self.num_candidates):
            # Generate instruction
            if self.instruction_generator == self._generate_instruction:
                # Use built-in generator with task analysis
                instruction = self.instruction_generator(task_analysis, i)
            else:
                # Use custom generator
                instruction = self.instruction_generator(
                    task_description=kwargs.get("task_description", ""),
                    examples=dataset,
                    iteration=i,
                )

            # Create module variant
            variant = copy.deepcopy(module)

            # Add instruction parameter
            instruction_param = Parameter(
                value=instruction,
                learnable=True,
                metadata={
                    "type": "instruction",
                    "iteration": i,
                    "patterns": task_analysis.get("patterns", []),
                },
            )
            variant.parameters["instruction"] = instruction_param

            # Update signature if available
            if hasattr(variant, "signature") and variant.signature:
                variant.signature.instructions = instruction

            # Evaluate
            score, traces = await self.evaluate(variant, eval_set)

            candidates.append(
                {
                    "module": variant,
                    "instruction": instruction,
                    "score": score,
                    "iteration": i,
                    "traces": traces,
                }
            )

        # Select best based on strategy
        if not candidates:
            raise OptimizationError(
                "No instruction candidates generated", optimizer_type=self.strategy.value
            )

        if self.selection_strategy == "best":
            best = max(candidates, key=lambda c: c["score"])
        elif self.selection_strategy == "weighted":
            # Weight by score for probabilistic selection
            scores = [c["score"] for c in candidates]
            total = sum(scores)
            if total > 0:
                weights = [s / total for s in scores]
                best = self.rng.choices(candidates, weights=weights)[0]
            else:
                best = candidates[0]
        elif self.selection_strategy == "tournament":
            # Tournament selection
            tournament_size = min(3, len(candidates))
            tournament = self.rng.sample(candidates, tournament_size)
            best = max(tournament, key=lambda c: c["score"])
        else:
            # Default to best
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
                "best_instruction": best["instruction"],
                "all_scores": [c["score"] for c in candidates],
                "selection_strategy": self.selection_strategy,
                "task_analysis": task_analysis,
                "baseline_score": baseline_score,
                "avg_score": sum(c["score"] for c in candidates) / len(candidates),
            },
        )


__all__ = ["InstructionOptimizer"]
