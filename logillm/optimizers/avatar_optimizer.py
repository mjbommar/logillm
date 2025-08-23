"""Avatar optimizer for LogiLLM based on DSPy's implementation."""

from __future__ import annotations

import asyncio
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from random import sample
from typing import Any, Callable

from ..core.avatar import ActionOutput, Avatar
from ..core.predict import Predict
from ..core.signatures import BaseSignature, FieldSpec
from ..core.types import FieldType, Prediction

logger = logging.getLogger(__name__)

DEFAULT_MAX_EXAMPLES = 10


@dataclass
class EvalResult:
    """Result from evaluating an Avatar on an example."""

    example: dict[str, Any]
    score: float
    actions: list[ActionOutput] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ComparatorSignature(BaseSignature):
    """Signature for comparing positive and negative Avatar performance."""

    def __init__(self):
        super().__init__(
            input_fields={
                "instruction": FieldSpec(
                    name="instruction",
                    field_type=FieldType.INPUT,
                    python_type=str,
                    description="Current instruction for the Avatar agent",
                    required=True,
                ),
                "actions": FieldSpec(
                    name="actions",
                    field_type=FieldType.INPUT,
                    python_type=list,
                    description="Available actions/tools for the agent",
                    required=True,
                ),
                "pos_input_with_metrics": FieldSpec(
                    name="pos_input_with_metrics",
                    field_type=FieldType.INPUT,
                    python_type=list,
                    description="Positive examples with their scores and actions taken",
                    required=True,
                ),
                "neg_input_with_metrics": FieldSpec(
                    name="neg_input_with_metrics",
                    field_type=FieldType.INPUT,
                    python_type=list,
                    description="Negative examples with their scores and actions taken",
                    required=True,
                ),
            },
            output_fields={
                "feedback": FieldSpec(
                    name="feedback",
                    field_type=FieldType.OUTPUT,
                    python_type=str,
                    description="Feedback for improving Avatar performance on negative examples",
                    required=True,
                ),
            },
            instructions=(
                "After executing the given actions on user inputs using the given instruction, "
                "some inputs have yielded good results, while others have not. I'll provide you "
                "the inputs along with their corresponding evaluation metrics.\n\n"
                "Task:\n"
                "(1) Firstly, identify and contrast the patterns of inputs that have achieved good "
                "results with those that have not.\n"
                "(2) Then, review the computational logic for any inconsistencies in the previous actions.\n"
                "(3) Lastly, specify the modification in tools used that can lead to improved "
                "performance on the negative inputs."
            ),
        )


class FeedbackBasedInstructionSignature(BaseSignature):
    """Signature for generating improved instructions based on feedback."""

    def __init__(self):
        super().__init__(
            input_fields={
                "previous_instruction": FieldSpec(
                    name="previous_instruction",
                    field_type=FieldType.INPUT,
                    python_type=str,
                    description="Previous instruction for the Avatar agent",
                    required=True,
                ),
                "feedback": FieldSpec(
                    name="feedback",
                    field_type=FieldType.INPUT,
                    python_type=str,
                    description="Feedback for improving Avatar performance",
                    required=True,
                ),
            },
            output_fields={
                "new_instruction": FieldSpec(
                    name="new_instruction",
                    field_type=FieldType.OUTPUT,
                    python_type=str,
                    description="Improved instruction for the Avatar agent",
                    required=True,
                ),
            },
            instructions=(
                "There is a task that needs to be completed for which one can use multiple tools "
                "to achieve the desired outcome. A group's performance was evaluated on a dataset "
                "of inputs, the inputs that did well are positive inputs, and the inputs that did "
                "not do well are negative inputs.\n\n"
                "You received feedback on how they can better use the tools to improve your "
                "performance on the negative inputs. You have been provided with the previous "
                "instruction, that they followed to use tools to complete the task, and the "
                "feedback on your performance.\n\n"
                "Your task is to incorporate the feedback and generate a detailed instruction for "
                "the group to follow to improve their performance on the task.\n\n"
                "Make sure that the new instruction talks about how to use the tools effectively "
                "and should be no more than 3 paragraphs long. The previous instruction contains "
                "general guidelines that you must retain in the new instruction."
            ),
        )


class AvatarOptimizer:
    """Optimizer for Avatar modules that improves tool usage through feedback.

    This optimizer:
    1. Evaluates Avatar performance on a dataset
    2. Separates examples into positive (high score) and negative (low score)
    3. Uses an LLM to analyze patterns and generate feedback
    4. Creates improved instructions based on the feedback
    5. Iteratively refines over multiple rounds

    This is LogiLLM's implementation of DSPy's AvatarOptimizer,
    adapted for our async-first, zero-dependency architecture.
    """

    def __init__(
        self,
        metric: Callable[[dict, Prediction], float],
        max_iters: int = 10,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        max_positive_inputs: int | None = None,
        max_negative_inputs: int | None = None,
        optimize_for: str = "max",
        num_threads: int = 4,
        **kwargs: Any,
    ):
        """Initialize AvatarOptimizer.

        Args:
            metric: Function to evaluate Avatar performance (example, prediction) -> score
            max_iters: Maximum optimization iterations
            lower_bound: Minimum score for negative examples
            upper_bound: Minimum score for positive examples
            max_positive_inputs: Maximum positive examples to use for feedback
            max_negative_inputs: Maximum negative examples to use for feedback
            optimize_for: "max" or "min" optimization direction
            num_threads: Number of threads for parallel evaluation
            **kwargs: Additional arguments for base Optimizer
        """
        # No parent class to initialize

        if metric is None:
            raise ValueError("`metric` argument cannot be None. Please provide a metric function.")

        self.metric = metric
        self.max_iters = max_iters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_positive_inputs = max_positive_inputs or DEFAULT_MAX_EXAMPLES
        self.max_negative_inputs = max_negative_inputs or DEFAULT_MAX_EXAMPLES
        self.optimize_for = optimize_for
        self.num_threads = num_threads

        # Create predictor modules for optimization
        self.comparator = Predict(signature=ComparatorSignature())  # type: ignore[arg-type]
        self.feedback_instruction = Predict(signature=FeedbackBasedInstructionSignature())  # type: ignore[arg-type]

    def _process_example(
        self, avatar: Avatar, example: dict[str, Any], return_outputs: bool = False
    ) -> tuple[dict, Prediction | None, float] | float:
        """Process a single example with the Avatar."""
        avatar_copy = copy.deepcopy(avatar)

        try:
            # Extract inputs from example
            if "inputs" in example:
                inputs = example["inputs"]
            else:
                # Assume the example itself contains the inputs
                inputs = {
                    k: v for k, v in example.items() if k not in {"outputs", "score", "metadata"}
                }

            # Get prediction
            prediction = avatar_copy.call_sync(**inputs)
            score = self.metric(example, prediction)

            if return_outputs:
                return example, prediction, score
            else:
                return score

        except Exception as e:
            logger.error(f"Error processing example: {e}")
            if return_outputs:
                return example, None, 0.0
            else:
                return 0.0

    async def _evaluate_avatar(
        self, avatar: Avatar, dataset: list[dict[str, Any]], return_outputs: bool = False
    ) -> tuple[float, list[tuple[dict, Prediction | None, float]]] | float:
        """Evaluate Avatar on dataset with optional threading."""
        total_score = 0.0
        total_examples = len(dataset)
        results = []

        # Use ThreadPoolExecutor for CPU-bound evaluation
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self._process_example, avatar, example, return_outputs)
                for example in dataset
            ]

            for future in asyncio.as_completed([asyncio.wrap_future(f) for f in futures]):
                try:
                    result = await future
                    if return_outputs:
                        example, prediction, score = result
                        total_score += score
                        results.append((example, prediction, score))
                    else:
                        total_score += result
                except Exception as e:
                    logger.error(f"Error in evaluation future: {e}")
                    if return_outputs:
                        results.append((None, None, 0.0))

        avg_score = total_score / total_examples if total_examples > 0 else 0.0

        if return_outputs:
            return avg_score, results
        else:
            return avg_score

    async def _get_pos_neg_results(
        self, avatar: Avatar, trainset: list[dict[str, Any]]
    ) -> tuple[float, list[EvalResult], list[EvalResult]]:
        """Separate examples into positive and negative based on performance."""
        pos_inputs = []
        neg_inputs = []

        avg_score, results = await self._evaluate_avatar(avatar, trainset, return_outputs=True)
        logger.info(f"Average Score: {avg_score}")

        for example, prediction, score in results:
            if example is None:  # Error case
                continue

            actions = getattr(prediction, "actions", None) if prediction else None

            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example,
                        score=score,
                        actions=actions,
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example,
                        score=score,
                        actions=actions,
                    )
                )

        if len(pos_inputs) == 0:
            raise ValueError(
                "No positive examples found. Try lowering the upper_bound or providing more training data."
            )
        if len(neg_inputs) == 0:
            raise ValueError(
                "No negative examples found. Try raising the lower_bound or providing more training data."
            )

        return avg_score, pos_inputs, neg_inputs

    async def optimize(
        self,
        module: Avatar,
        dataset: list[dict[str, Any]],
        validation_set: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Avatar:
        """Optimize Avatar module through iterative feedback.

        Args:
            module: Avatar module to optimize
            dataset: Training dataset for optimization
            validation_set: Optional validation set (not used in this implementation)
            **kwargs: Additional optimization parameters

        Returns:
            Optimized Avatar module
        """
        if not isinstance(module, Avatar):
            raise ValueError("AvatarOptimizer only works with Avatar modules")

        best_avatar = copy.deepcopy(module)
        best_score = -999 if self.optimize_for == "max" else 999

        logger.info(f"Starting Avatar optimization with {len(dataset)} examples")

        for iteration in range(self.max_iters):
            logger.info("=" * 50)
            logger.info(f"Iteration {iteration + 1}/{self.max_iters}")

            # Evaluate current avatar and get positive/negative examples
            try:
                score, pos_inputs, neg_inputs = await self._get_pos_neg_results(
                    best_avatar, dataset
                )
            except ValueError as e:
                logger.warning(f"Stopping optimization: {e}")
                break

            logger.info(f"Positive examples: {len(pos_inputs)}")
            logger.info(f"Negative examples: {len(neg_inputs)}")
            logger.info(
                f"Sampling {self.max_positive_inputs} positive and {self.max_negative_inputs} negative examples"
            )

            # Sample examples if too many
            if self.max_positive_inputs and len(pos_inputs) > self.max_positive_inputs:
                pos_inputs = sample(pos_inputs, self.max_positive_inputs)

            if self.max_negative_inputs and len(neg_inputs) > self.max_negative_inputs:
                neg_inputs = sample(neg_inputs, self.max_negative_inputs)

            # Get current instruction
            current_instruction = (
                best_avatar.signature.instructions if best_avatar.signature else ""
            )

            # Format examples for comparator
            pos_formatted = []
            for result in pos_inputs:
                actions_str = (
                    [
                        f"{a.tool_name}({a.tool_input_query}) -> {a.tool_output}"
                        for a in result.actions
                    ]
                    if result.actions
                    else ["No actions recorded"]
                )
                pos_formatted.append(
                    {
                        "example": result.example,
                        "score": result.score,
                        "actions": actions_str,
                    }
                )

            neg_formatted = []
            for result in neg_inputs:
                actions_str = (
                    [
                        f"{a.tool_name}({a.tool_input_query}) -> {a.tool_output}"
                        for a in result.actions
                    ]
                    if result.actions
                    else ["No actions recorded"]
                )
                neg_formatted.append(
                    {
                        "example": result.example,
                        "score": result.score,
                        "actions": actions_str,
                    }
                )

            # Generate feedback using comparator
            try:
                feedback_prediction = await self.comparator(
                    instruction=current_instruction,
                    actions=[str(tool) for tool in best_avatar.tools],
                    pos_input_with_metrics=pos_formatted,
                    neg_input_with_metrics=neg_formatted,
                )
                feedback = feedback_prediction.outputs.get("feedback", "")
                logger.info(f"Generated feedback: {feedback[:200]}...")
            except Exception as e:
                logger.error(f"Error generating feedback: {e}")
                feedback = (
                    "Unable to generate specific feedback. Consider reviewing tool usage patterns."
                )

            # Generate new instruction based on feedback
            try:
                instruction_prediction = await self.feedback_instruction(
                    previous_instruction=current_instruction,
                    feedback=feedback,
                )
                new_instruction = instruction_prediction.outputs.get(
                    "new_instruction", current_instruction
                )
                logger.info(f"Generated new instruction: {new_instruction[:200]}...")
            except Exception as e:
                logger.error(f"Error generating new instruction: {e}")
                new_instruction = current_instruction

            # Update avatar with new instruction if score improved
            if (self.optimize_for == "max" and score > best_score) or (
                self.optimize_for == "min" and score < best_score
            ):
                # Create new signature with updated instruction
                new_signature = copy.deepcopy(best_avatar.signature)
                if new_signature:
                    new_signature.instructions = new_instruction

                # Update the avatar
                best_avatar.signature = new_signature
                if hasattr(best_avatar, "actor") and best_avatar.actor:
                    best_avatar.actor.signature.instructions = new_instruction

                # Update the actor clone
                best_avatar.actor_clone = copy.deepcopy(best_avatar.actor)
                best_score = score

                logger.info(f"Avatar improved! New score: {best_score}")
            else:
                logger.info(f"No improvement. Keeping previous avatar (score: {best_score})")

        logger.info(f"Optimization complete. Best avatar score: {best_score}")
        return best_avatar


__all__ = [
    "AvatarOptimizer",
    "EvalResult",
    "ComparatorSignature",
    "FeedbackBasedInstructionSignature",
]
