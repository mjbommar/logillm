"""BootstrapFewShot optimizer - DSPy's most important optimizer."""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..core.modules import Module, Parameter
from ..core.optimizers import Metric
from ..core.types import OptimizationResult, OptimizationStrategy
from ..exceptions import OptimizationError
from .base import Demonstration, DemoSelector, PromptOptimizationConfig, PromptOptimizer
from .demo_selectors import BestDemoSelector

logger = logging.getLogger(__name__)


@dataclass
class BootstrapFewShotConfig(PromptOptimizationConfig):
    """Configuration for BootstrapFewShot optimizer with temperature scheduling."""

    # Temperature scheduling parameters
    initial_teacher_temperature: float = 1.0
    temperature_decay: float = 0.9
    min_temperature: float = 0.3
    rescue_mode_threshold: float = 0.2

    # Bootstrap-specific parameters
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    max_rounds: int = 1
    metric_threshold: Optional[float] = None

    # Rescue mode parameters
    rescue_initial_temperature: float = 1.5
    rescue_max_attempts_multiplier: float = 2.0

    # Diversity scoring parameters
    use_diversity_scoring: bool = True
    diversity_weight: float = 0.3  # 0=accuracy only, 1=diversity only


class BootstrapFewShot(PromptOptimizer):
    """Bootstrap few-shot optimizer using teacher-student architecture.

    This is DSPy's most important optimizer. It uses a teacher model
    (usually with higher temperature) to generate successful examples,
    then selects the best ones as demonstrations for the student model.

    The key insight: LLMs can often solve problems correctly sometimes,
    so we can bootstrap demonstrations from successful runs.

    Includes temperature scheduling and rescue mode for better performance
    on challenging baselines.
    """

    def __init__(
        self,
        metric: Metric,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        teacher_settings: Optional[dict[str, Any]] = None,
        max_rounds: int = 1,
        metric_threshold: Optional[float] = None,
        demo_selector: Optional[DemoSelector] = None,
        config: Optional[BootstrapFewShotConfig] = None,
        **kwargs: Any,
    ):
        """Initialize BootstrapFewShot optimizer.

        Args:
            metric: Evaluation metric
            max_bootstrapped_demos: Max number of bootstrapped demos
            max_labeled_demos: Max number of labeled demos to use
            teacher_settings: Settings for teacher model (e.g., higher temperature)
            max_rounds: Number of bootstrapping rounds
            metric_threshold: Minimum score for a demo to be considered successful
            demo_selector: Strategy for selecting demonstrations
            config: Bootstrap-specific optimization configuration
        """
        # Create or update config with bootstrap-specific parameters
        if config is None:
            config = BootstrapFewShotConfig(
                strategy=OptimizationStrategy.BOOTSTRAP,
                max_demos=max_bootstrapped_demos,
                teacher_settings=teacher_settings or {},
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=max_rounds,
                metric_threshold=metric_threshold,
            )
        else:
            # Update config with provided parameters
            config.max_bootstrapped_demos = max_bootstrapped_demos
            config.max_labeled_demos = max_labeled_demos
            config.max_rounds = max_rounds
            if metric_threshold is not None:
                config.metric_threshold = metric_threshold

        super().__init__(strategy=OptimizationStrategy.BOOTSTRAP, metric=metric, config=config)

        # Store config as typed attribute
        self.config: BootstrapFewShotConfig = config
        self.teacher_settings = teacher_settings or {
            "temperature": config.initial_teacher_temperature
        }
        self.demo_selector = demo_selector or BestDemoSelector()

        # Properties for backward compatibility
        self.max_bootstrapped_demos = config.max_bootstrapped_demos
        self.max_labeled_demos = config.max_labeled_demos
        self.max_rounds = config.max_rounds
        self.metric_threshold = config.metric_threshold

    async def _bootstrap_demonstrations(
        self,
        teacher: Module,
        dataset: list[dict[str, Any]],
        max_attempts: int = 100,
        current_temperature: float = 0.7,
        rescue_mode: bool = False,
    ) -> list[Demonstration]:
        """Bootstrap demonstrations from teacher model.

        Args:
            teacher: Teacher module to generate examples
            dataset: Training dataset
            max_attempts: Maximum attempts to generate enough demos
            current_temperature: Current teacher temperature
            rescue_mode: Whether rescue mode is active

        Returns:
            List of successful demonstrations
        """
        demonstrations: list[Demonstration] = []
        attempts = 0
        example_idx = 0
        successful_bootstraps = 0

        target_demos = self.config.max_bootstrapped_demos
        if rescue_mode:
            # In rescue mode, try harder
            max_attempts = int(max_attempts * self.config.rescue_max_attempts_multiplier)
            logger.info(f"Rescue mode active - increased max_attempts to {max_attempts}")

        while len(demonstrations) < target_demos and attempts < max_attempts:
            if example_idx >= len(dataset):
                # Wrap around if we run out of examples
                example_idx = 0

            example = dataset[example_idx]
            example_idx += 1
            attempts += 1

            try:
                # Extract inputs and expected outputs
                inputs = example.get("inputs", {})
                expected = example.get("outputs", {})

                # Run teacher
                prediction = await teacher(**inputs)

                # Evaluate prediction
                if prediction.success and prediction.outputs:
                    score = self.metric(prediction.outputs, expected)

                    # Check if successful
                    threshold = (
                        self.config.metric_threshold
                        if self.config.metric_threshold is not None
                        else 0.5
                    )
                    if score >= threshold:
                        demo = Demonstration(
                            inputs=inputs,
                            outputs=prediction.outputs,
                            score=score,
                            metadata={
                                "expected": expected,
                                "teacher": True,
                                "attempt": attempts,
                                "temperature": current_temperature,
                                "rescue_mode": rescue_mode,
                            },
                        )
                        demonstrations.append(demo)
                        successful_bootstraps += 1

            except Exception:
                # Continue on errors
                continue

        logger.info(
            f"Bootstrap round completed: {successful_bootstraps} successful demos "
            f"from {attempts} attempts (temperature={current_temperature:.2f}, "
            f"rescue_mode={rescue_mode})"
        )

        return demonstrations

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0

    def _calculate_diversity_score(
        self, candidate: Demonstration, selected: list[Demonstration], alpha: float = 0.5
    ) -> float:
        """Calculate combined score balancing accuracy and diversity.

        Args:
            candidate: Demo to evaluate
            selected: Already selected demos
            alpha: Weight for accuracy vs diversity (0.5 = balanced)

        Returns:
            Combined score [0, 1]
        """
        if not selected:
            return candidate.score  # First demo selected by score alone

        # Calculate minimum similarity to any selected demo
        similarities = []
        for demo in selected:
            # Compare inputs
            input_sim = 0.0
            for key in candidate.inputs:
                if key in demo.inputs:
                    input_sim += self._text_similarity(
                        str(candidate.inputs[key]), str(demo.inputs[key])
                    )
            input_sim /= max(len(candidate.inputs), 1)

            # Compare outputs similarly
            output_sim = 0.0
            for key in candidate.outputs:
                if key in demo.outputs:
                    output_sim += self._text_similarity(
                        str(candidate.outputs[key]), str(demo.outputs[key])
                    )
            output_sim /= max(len(candidate.outputs), 1)

            similarities.append((input_sim + output_sim) / 2)

        # Diversity is inverse of maximum similarity
        diversity = 1.0 - max(similarities)

        # Combine accuracy and diversity
        return alpha * candidate.score + (1 - alpha) * diversity

    def _select_demonstrations(
        self, demonstrations: list[Demonstration], max_demos: int
    ) -> list[Demonstration]:
        """Select demonstrations using diversity-aware algorithm.

        Args:
            demonstrations: List of candidate demonstrations
            max_demos: Maximum number of demos to select

        Returns:
            Selected demonstrations balancing accuracy and diversity
        """
        if not self.config.use_diversity_scoring:
            # Original behavior
            sorted_demos = sorted(demonstrations, key=lambda d: d.score, reverse=True)
            return sorted_demos[:max_demos]

        logger.info(
            f"Using diversity scoring with weight {self.config.diversity_weight:.2f} "
            f"to select {max_demos} demos from {len(demonstrations)} candidates"
        )

        # Diversity-aware selection
        selected: list[Demonstration] = []
        remaining = list(demonstrations)

        for selection_idx in range(min(max_demos, len(demonstrations))):
            if not remaining:
                break

            # Score each remaining demo
            scores = []
            for demo in remaining:
                score = self._calculate_diversity_score(
                    demo, selected, alpha=1.0 - self.config.diversity_weight
                )
                scores.append((score, demo))

            # Select best and add to selected
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_demo = scores[0]
            selected.append(best_demo)
            remaining.remove(best_demo)

            # Log selection details
            logger.debug(
                f"Selected demo {selection_idx + 1}: accuracy={best_demo.score:.3f}, "
                f"combined_score={best_score:.3f}"
            )

        # Log final selection summary
        accuracy_scores = [d.score for d in selected]
        avg_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        logger.info(
            f"Selected {len(selected)} demos with accuracy scores: "
            f"[{', '.join(f'{s:.3f}' for s in accuracy_scores)}], "
            f"avg: {avg_score:.3f}"
        )

        return selected

    async def _add_labeled_demos(
        self, demonstrations: list[Demonstration], dataset: list[dict[str, Any]]
    ) -> list[Demonstration]:
        """Add labeled demonstrations from the dataset.

        Args:
            demonstrations: Current list of demonstrations
            dataset: Training dataset

        Returns:
            Combined list of demonstrations
        """
        # Add some labeled examples if we don't have enough bootstrapped
        num_to_add = min(
            self.config.max_labeled_demos,
            max(0, self.config.max_bootstrapped_demos - len(demonstrations)),
        )

        if num_to_add > 0:
            logger.info(
                f"Adding {num_to_add} labeled demonstrations to supplement bootstrapped ones"
            )
            for example in dataset[:num_to_add]:
                demo = Demonstration(
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    score=1.0,  # Labeled data assumed correct
                    metadata={"source": "labeled", "expected": example.get("outputs", {})},
                )
                demonstrations.append(demo)

        return demonstrations

    def _update_teacher_temperature(self, teacher: Module, temperature: float) -> None:
        """Update teacher temperature in both config and provider."""
        from ..core.config_utils import set_hyperparameter

        # Use the utility function for consistent parameter setting
        set_hyperparameter(teacher, "temperature", temperature)

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        teacher: Optional[Module] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize module using bootstrapped few-shot examples with temperature scheduling.

        Args:
            module: Module to optimize (becomes student)
            dataset: Training dataset
            validation_set: Validation set for evaluation
            teacher: Optional teacher module (defaults to module with teacher settings)
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with bootstrapped demonstrations
        """
        import time

        start_time = time.time()

        # Calculate baseline score to determine if rescue mode is needed
        eval_set = validation_set or dataset
        baseline_score, _ = await self.evaluate(module, eval_set)

        # Determine if rescue mode is needed
        rescue_mode = baseline_score < self.config.rescue_mode_threshold
        if rescue_mode:
            logger.warning(
                f"Baseline score {baseline_score:.2%} below rescue threshold "
                f"{self.config.rescue_mode_threshold:.2%} - activating rescue mode"
            )

        # Initialize temperature scheduling
        if rescue_mode:
            current_temperature = self.config.rescue_initial_temperature
        else:
            current_temperature = self.config.initial_teacher_temperature

        logger.info(
            f"Starting bootstrap optimization with initial teacher temperature: {current_temperature:.2f} "
            f"(rescue_mode={rescue_mode})"
        )

        # Create teacher if not provided
        if teacher is None:
            teacher = copy.deepcopy(module)
            # Apply initial teacher settings
            teacher_settings = dict(self.teacher_settings)
            teacher_settings["temperature"] = current_temperature

            if hasattr(teacher, "config"):
                if hasattr(teacher.config, "update"):
                    teacher.config.update(teacher_settings)
                else:
                    for key, value in teacher_settings.items():
                        setattr(teacher.config, key, value)

            # Update provider settings
            if hasattr(teacher, "provider") and teacher.provider:
                for key, value in teacher_settings.items():
                    setattr(teacher.provider, key, value)

        # Bootstrap demonstrations with temperature scheduling
        all_demonstrations = []

        for round_idx in range(self.config.max_rounds):
            logger.info(
                f"Bootstrap round {round_idx + 1}/{self.config.max_rounds} (temperature={current_temperature:.2f})"
            )

            # Update teacher temperature for this round
            self._update_teacher_temperature(teacher, current_temperature)

            # Generate demonstrations from teacher
            round_demos = await self._bootstrap_demonstrations(
                teacher,
                dataset,
                max_attempts=len(dataset) * 2,  # Try each example up to twice
                current_temperature=current_temperature,
                rescue_mode=rescue_mode,
            )
            all_demonstrations.extend(round_demos)

            # Apply temperature decay for next round
            current_temperature = max(
                self.config.min_temperature, current_temperature * self.config.temperature_decay
            )

            logger.info(
                f"Round {round_idx + 1} completed: {len(round_demos)} demos generated, "
                f"{len(all_demonstrations)} total. Next temperature: {current_temperature:.2f}"
            )

            # If we have enough, stop
            if len(all_demonstrations) >= self.config.max_bootstrapped_demos:
                break

        # Add labeled demos if needed
        if len(all_demonstrations) < self.config.max_bootstrapped_demos:
            all_demonstrations = await self._add_labeled_demos(all_demonstrations, dataset)

        # Select best demonstrations using diversity-aware selection
        if self.config.use_diversity_scoring:
            selected_demos = self._select_demonstrations(
                all_demonstrations, min(self.config.max_bootstrapped_demos, len(all_demonstrations))
            )
        else:
            # Use traditional demo selector
            selected_demos = self.demo_selector.select(
                all_demonstrations, min(self.config.max_bootstrapped_demos, len(all_demonstrations))
            )

        if not selected_demos:
            raise OptimizationError(
                "No successful demonstrations could be generated",
                optimizer_type=self.strategy.value,
                context={
                    "attempts": len(dataset) * 2 * self.config.max_rounds,
                    "threshold": self.config.metric_threshold,
                    "rescue_mode": rescue_mode,
                    "baseline_score": baseline_score,
                    "temperature_range": f"{self.config.initial_teacher_temperature:.2f} → {self.config.min_temperature:.2f}",
                },
            )

        # Create student module with demonstrations
        student = copy.deepcopy(module)

        # Add demonstrations to the demo_manager (for Predict modules)
        if hasattr(student, "demo_manager"):
            # Clear existing demos and add new ones
            student.demo_manager.clear()  # type: ignore[attr-defined]
            for demo in selected_demos:
                student.demo_manager.add({"inputs": demo.inputs, "outputs": demo.outputs})  # type: ignore[attr-defined]

        # Also add demonstrations as parameter for tracking
        demo_param = Parameter(
            value=[d.to_dict() for d in selected_demos],
            learnable=True,
            metadata={
                "type": "demonstrations",
                "source": "bootstrapped",
                "num_bootstrapped": sum(
                    1 for d in selected_demos if d.metadata.get("teacher", False)
                ),
                "num_labeled": sum(
                    1 for d in selected_demos if d.metadata.get("source") == "labeled"
                ),
                "temperature_schedule": {
                    "initial": self.config.initial_teacher_temperature,
                    "final": current_temperature,
                    "decay": self.config.temperature_decay,
                },
                "rescue_mode": rescue_mode,
            },
        )
        student.parameters["demonstrations"] = demo_param

        # Evaluate student
        final_score, _ = await self.evaluate(student, eval_set)

        improvement = final_score - baseline_score
        logger.info(
            f"Bootstrap optimization completed: {baseline_score:.2%} → {final_score:.2%} "
            f"(improvement: {improvement:+.2%})"
        )

        return OptimizationResult(
            optimized_module=student,
            improvement=improvement,
            iterations=self.config.max_rounds,
            best_score=final_score,
            optimization_time=time.time() - start_time,
            metadata={
                "num_demos": len(selected_demos),
                "demo_scores": [d.score for d in selected_demos],
                "avg_demo_score": sum(d.score for d in selected_demos) / len(selected_demos),
                "num_bootstrapped": sum(
                    1 for d in selected_demos if d.metadata.get("teacher", False)
                ),
                "num_labeled": sum(
                    1 for d in selected_demos if d.metadata.get("source") == "labeled"
                ),
                "teacher_settings": self.teacher_settings,
                "baseline_score": baseline_score,
                "total_attempts": len(all_demonstrations),
                "rescue_mode": rescue_mode,
                "temperature_schedule": {
                    "initial": self.config.initial_teacher_temperature,
                    "final": current_temperature,
                    "decay": self.config.temperature_decay,
                    "min": self.config.min_temperature,
                },
                "config": {
                    "rescue_threshold": self.config.rescue_mode_threshold,
                    "initial_temp": self.config.initial_teacher_temperature,
                    "rescue_temp": self.config.rescue_initial_temperature,
                    "max_attempts_multiplier": self.config.rescue_max_attempts_multiplier,
                    "diversity_scoring": {
                        "enabled": self.config.use_diversity_scoring,
                        "weight": self.config.diversity_weight,
                    },
                },
            },
        )


__all__ = ["BootstrapFewShot", "BootstrapFewShotConfig"]
