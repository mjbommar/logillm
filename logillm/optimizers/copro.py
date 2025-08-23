"""COPRO (Collaborative Prompt Optimization) optimizer for LogiLLM.

Based on DSPy's COPRO implementation but adapted for LogiLLM's architecture.
Optimizes instructions and output prefixes through iterative generation and evaluation.
"""

from __future__ import annotations

import copy
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..core.modules import Module, Parameter
from ..core.optimizers import Metric
from ..core.predict import Predict
from ..core.providers import Provider, get_provider
from ..core.types import OptimizationResult, OptimizationStrategy
from ..exceptions import OptimizationError
from .base import PromptOptimizationConfig, PromptOptimizer
from .instruction_signatures import BasicGenerateInstruction, GenerateInstructionGivenAttempts

logger = logging.getLogger(__name__)


@dataclass
class COPROConfig(PromptOptimizationConfig):
    """Configuration for COPRO optimizer."""

    breadth: int = 10  # Number of candidates per iteration
    depth: int = 3  # Number of refinement iterations
    init_temperature: float = 1.4  # Temperature for generation
    track_stats: bool = False  # Track optimization statistics
    prompt_model: Any = None  # Model for prompt generation
    dedupe_candidates: bool = True  # Remove duplicate candidates
    min_score_threshold: float = 0.0  # Minimum score to consider a candidate


@dataclass
class COPROStats:
    """Statistics tracking for COPRO optimization."""

    results_best: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    results_latest: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    total_calls: int = 0


class InstructionCandidate:
    """A candidate instruction with associated metadata."""

    def __init__(
        self,
        instruction: str | None,
        prefix: str | None,
        score: float = 0.0,
        depth: int = 0,
        module: Module | None = None,
    ):
        self.instruction = instruction.strip().strip('"') if instruction else ""
        self.prefix = prefix.strip().strip('"') if prefix else ""
        self.score = score
        self.depth = depth
        self.module = module

    def __eq__(self, other) -> bool:
        """Check if two candidates are equivalent."""
        if not isinstance(other, InstructionCandidate):
            return False
        return self.instruction == other.instruction and self.prefix == other.prefix

    def __hash__(self) -> int:
        """Hash for deduplication."""
        return hash((self.instruction, self.prefix))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instruction": self.instruction,
            "prefix": self.prefix,
            "score": self.score,
            "depth": self.depth,
        }


class COPRO(PromptOptimizer):
    """Collaborative Prompt Optimization (COPRO) optimizer.

    Optimizes instructions and output prefixes through breadth-first search
    followed by iterative refinement using previous attempts as feedback.

    Key features:
    - Breadth-first search for instruction generation
    - Depth iterations for progressive refinement
    - Temperature-based creativity control
    - Duplicate detection and removal
    - Statistics tracking for optimization process
    """

    def __init__(
        self,
        metric: Metric,
        breadth: int = 10,
        depth: int = 3,
        init_temperature: float = 1.4,
        track_stats: bool = False,
        prompt_model: Provider | None = None,
        config: COPROConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize COPRO optimizer.

        Args:
            metric: Evaluation metric
            breadth: Number of candidates per iteration
            depth: Number of refinement iterations
            init_temperature: Temperature for generation
            track_stats: Whether to track optimization statistics
            prompt_model: Model for prompt generation (uses default if None)
            config: COPRO configuration
            **kwargs: Additional configuration
        """
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")

        # Create config if not provided
        if config is None:
            config = COPROConfig(
                strategy=OptimizationStrategy.INSTRUCTION,
                breadth=breadth,
                depth=depth,
                init_temperature=init_temperature,
                track_stats=track_stats,
                prompt_model=prompt_model,
                **kwargs,
            )

        super().__init__(strategy=OptimizationStrategy.INSTRUCTION, metric=metric, config=config)

        self.breadth = config.breadth
        self.depth = config.depth
        self.init_temperature = config.init_temperature
        self.track_stats = config.track_stats
        self.prompt_model = config.prompt_model
        self.dedupe_candidates = config.dedupe_candidates
        self.min_score_threshold = config.min_score_threshold

        # Statistics tracking
        self.stats = COPROStats() if track_stats else None

        # Create instruction generators with the prompt model
        self.basic_generator = Predict(BasicGenerateInstruction, provider=self.prompt_model)
        self.advanced_generator = Predict(
            GenerateInstructionGivenAttempts, provider=self.prompt_model
        )

    def _get_instruction(self, module: Module) -> str:
        """Extract current instruction from module."""
        # Check parameters first
        if hasattr(module, "parameters") and "instruction" in module.parameters:
            return module.parameters["instruction"].value

        # Check signature
        if hasattr(module, "signature") and module.signature:
            return getattr(module.signature, "instructions", "")

        return "Complete the following task."

    def _get_output_prefix(self, module: Module) -> str:
        """Extract output prefix from module."""
        # For now, return empty string - could enhance to extract from signature
        return ""

    def _set_instruction(self, module: Module, instruction: str, prefix: str) -> None:
        """Set instruction and prefix on module."""
        # Update parameters
        if hasattr(module, "parameters"):
            module.parameters["instruction"] = Parameter(
                value=instruction,
                learnable=True,
                metadata={"type": "instruction", "prefix": prefix},
            )

        # Update signature if available
        if hasattr(module, "signature") and module.signature:
            # Create updated signature with new instructions
            if hasattr(module.signature, "with_instructions"):
                module.signature = module.signature.with_instructions(instruction)
            else:
                module.signature.instructions = instruction

    async def _generate_initial_candidates(
        self,
        module: Module,
        basic_instruction: str,
        basic_prefix: str,
    ) -> list[InstructionCandidate]:
        """Generate initial instruction candidates."""
        candidates = []

        # Get provider for generation
        provider = self.prompt_model or get_provider()

        # Generate breadth-1 new instructions
        try:
            # Configure generation parameters
            config = {"temperature": self.init_temperature, "max_tokens": 500}
            if hasattr(provider, "supports_n") and provider.supports_n:
                config["n"] = self.breadth - 1

            # Generate candidates
            prediction = await self.basic_generator(
                basic_instruction=basic_instruction, _config=config
            )

            # Extract completions
            if hasattr(prediction, "completions") and prediction.completions:
                # Multiple completions
                instructions = prediction.completions.get("proposed_instruction", [])
                prefixes = prediction.completions.get("proposed_prefix", [])
            else:
                # Single completion
                instructions = [prediction.outputs.get("proposed_instruction", "")]
                prefixes = [prediction.outputs.get("proposed_prefix", "")]

        except Exception as e:
            logger.warning(f"Failed to generate instructions: {e}")
            instructions = []
            prefixes = []

        # Add generated candidates
        for inst, prefix in zip(instructions, prefixes):
            inst_str = str(inst) if inst is not None else ""
            prefix_str = str(prefix) if prefix is not None else ""

            # Skip undefined or empty values
            if inst_str and inst_str.strip() and "Undefined" not in inst_str:
                candidates.append(
                    InstructionCandidate(instruction=inst_str, prefix=prefix_str, depth=0)
                )

        # Add original instruction as candidate
        candidates.append(
            InstructionCandidate(instruction=basic_instruction, prefix=basic_prefix, depth=0)
        )

        return candidates

    async def _generate_refined_candidates(
        self,
        best_candidates: list[InstructionCandidate],
        depth: int,
    ) -> list[InstructionCandidate]:
        """Generate refined candidates based on previous attempts."""
        # Prepare attempt history
        attempts = []

        # Use up to breadth best candidates, sorted by score
        sorted_candidates = sorted(best_candidates, key=lambda c: c.score, reverse=True)
        use_candidates = sorted_candidates[: min(self.breadth, len(sorted_candidates))]

        for i, candidate in enumerate(use_candidates):
            attempts.extend(
                [
                    f"Instruction #{i + 1}: {candidate.instruction}",
                    f"Prefix #{i + 1}: {candidate.prefix}",
                    f"Resulting Score #{i + 1}: {candidate.score:.3f}",
                ]
            )

        # Generate new candidates
        candidates = []
        provider = self.prompt_model or get_provider()

        try:
            # Configure generation
            config = {"temperature": self.init_temperature, "max_tokens": 500}
            if hasattr(provider, "supports_n") and provider.supports_n:
                config["n"] = self.breadth

            # Generate using attempt history
            prediction = await self.advanced_generator(
                attempted_instructions=attempts, _config=config
            )

            # Extract completions
            if hasattr(prediction, "completions") and prediction.completions:
                instructions = prediction.completions.get("proposed_instruction", [])
                prefixes = prediction.completions.get("proposed_prefix", [])
            else:
                instructions = [prediction.outputs.get("proposed_instruction", "")]
                prefixes = [prediction.outputs.get("proposed_prefix", "")]

        except Exception as e:
            logger.warning(f"Failed to generate refined instructions: {e}")
            instructions = []
            prefixes = []

        # Create candidates
        for inst, prefix in zip(instructions, prefixes):
            inst_str = str(inst) if inst is not None else ""
            prefix_str = str(prefix) if prefix is not None else ""

            # Skip undefined or empty values
            if inst_str and inst_str.strip() and "Undefined" not in inst_str:
                candidates.append(
                    InstructionCandidate(instruction=inst_str, prefix=prefix_str, depth=depth)
                )

        return candidates

    def _dedupe_candidates(
        self, candidates: list[InstructionCandidate]
    ) -> list[InstructionCandidate]:
        """Remove duplicate candidates, keeping the best score for each."""
        if not self.dedupe_candidates:
            return candidates

        seen = {}
        for candidate in candidates:
            key = (candidate.instruction, candidate.prefix)
            if key not in seen or candidate.score > seen[key].score:
                seen[key] = candidate

        return list(seen.values())

    def _track_stats(
        self,
        predictor_id: str,
        depth: int,
        latest_scores: list[float],
        all_scores: list[float],
    ) -> None:
        """Track optimization statistics."""
        if not self.stats:
            return

        # Initialize tracking for this predictor
        if predictor_id not in self.stats.results_latest:
            self.stats.results_latest[predictor_id] = {
                "depth": [],
                "max": [],
                "average": [],
                "min": [],
                "std": [],
            }
        if predictor_id not in self.stats.results_best:
            self.stats.results_best[predictor_id] = {
                "depth": [],
                "max": [],
                "average": [],
                "min": [],
                "std": [],
            }

        # Track latest scores
        if latest_scores:
            self.stats.results_latest[predictor_id]["depth"].append(depth)
            self.stats.results_latest[predictor_id]["max"].append(max(latest_scores))
            self.stats.results_latest[predictor_id]["average"].append(
                statistics.mean(latest_scores)
            )
            self.stats.results_latest[predictor_id]["min"].append(min(latest_scores))
            self.stats.results_latest[predictor_id]["std"].append(
                statistics.stdev(latest_scores) if len(latest_scores) > 1 else 0.0
            )

        # Track best scores (top 10)
        if all_scores:
            top_scores = sorted(all_scores, reverse=True)[:10]
            self.stats.results_best[predictor_id]["depth"].append(depth)
            self.stats.results_best[predictor_id]["max"].append(max(top_scores))
            self.stats.results_best[predictor_id]["average"].append(statistics.mean(top_scores))
            self.stats.results_best[predictor_id]["min"].append(min(top_scores))
            self.stats.results_best[predictor_id]["std"].append(
                statistics.stdev(top_scores) if len(top_scores) > 1 else 0.0
            )

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize module instructions using COPRO algorithm.

        Args:
            module: Module to optimize
            dataset: Training dataset for evaluation
            validation_set: Optional validation set
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with optimized module
        """
        import time

        start_time = time.time()

        # Use dataset for evaluation if no validation set
        eval_set = validation_set or dataset

        # Initialize statistics
        if self.stats:
            self.stats.total_calls = 0

        # Get initial instruction and prefix
        basic_instruction = self._get_instruction(module)
        basic_prefix = self._get_output_prefix(module)

        logger.info(f"Starting COPRO optimization with breadth={self.breadth}, depth={self.depth}")
        logger.info(f"Initial instruction: {basic_instruction}")

        # Generate initial candidates
        logger.info(f"Generating {self.breadth} initial instruction candidates...")
        candidates = await self._generate_initial_candidates(
            module, basic_instruction, basic_prefix
        )

        # Evaluate initial candidates
        logger.info(f"Evaluating {len(candidates)} initial candidates...")
        evaluated_candidates = []

        for i, candidate in enumerate(candidates):
            # Create module variant
            variant = copy.deepcopy(module)
            self._set_instruction(variant, candidate.instruction, candidate.prefix)

            # Evaluate
            logger.info(f"Evaluating initial candidate {i + 1}/{len(candidates)}")
            score, traces = await self.evaluate(variant, eval_set)

            candidate.score = score
            candidate.module = variant
            evaluated_candidates.append(candidate)

            if self.stats:
                self.stats.total_calls += 1

        # Track initial stats
        predictor_id = f"{module.__class__.__name__}_{id(module)}"
        initial_scores = [c.score for c in evaluated_candidates]
        if self.stats:
            self._track_stats(predictor_id, 0, initial_scores, initial_scores)

        # Iterative refinement
        all_candidates = evaluated_candidates.copy()

        for depth in range(1, self.depth + 1):
            logger.info(f"Starting refinement iteration {depth}/{self.depth}")

            # Generate refined candidates based on best so far
            best_so_far = sorted(all_candidates, key=lambda c: c.score, reverse=True)
            refined = await self._generate_refined_candidates(best_so_far, depth)

            if not refined:
                logger.warning(f"No refined candidates generated at depth {depth}")
                continue

            # Evaluate refined candidates
            logger.info(f"Evaluating {len(refined)} refined candidates...")
            latest_scores = []

            for i, candidate in enumerate(refined):
                # Create module variant
                variant = copy.deepcopy(module)
                self._set_instruction(variant, candidate.instruction, candidate.prefix)

                # Evaluate
                logger.info(f"Evaluating refined candidate {i + 1}/{len(refined)} at depth {depth}")
                score, traces = await self.evaluate(variant, eval_set)

                candidate.score = score
                candidate.module = variant
                latest_scores.append(score)

                if self.stats:
                    self.stats.total_calls += 1

            # Add to all candidates
            all_candidates.extend(refined)

            # Track stats for this depth
            all_scores = [c.score for c in all_candidates]
            if self.stats:
                self._track_stats(predictor_id, depth, latest_scores, all_scores)

        # Remove duplicates and filter by threshold
        all_candidates = self._dedupe_candidates(all_candidates)
        all_candidates = [c for c in all_candidates if c.score >= self.min_score_threshold]

        if not all_candidates:
            raise OptimizationError(
                "No valid candidates found after optimization", optimizer_type=self.strategy.value
            )

        # Select best candidate
        best_candidate = max(all_candidates, key=lambda c: c.score)

        logger.info(f"Optimization complete. Best instruction: {best_candidate.instruction}")
        logger.info(f"Best score: {best_candidate.score:.3f}")

        # Calculate baseline score for improvement
        baseline_score, _ = await self.evaluate(module, eval_set)

        # Create result
        result = OptimizationResult(
            optimized_module=best_candidate.module,
            improvement=best_candidate.score - baseline_score,
            iterations=self.depth + 1,  # Initial + depth refinements
            best_score=best_candidate.score,
            optimization_time=time.time() - start_time,
            metadata={
                "best_instruction": best_candidate.instruction,
                "best_prefix": best_candidate.prefix,
                "num_candidates": len(all_candidates),
                "baseline_score": baseline_score,
                "breadth": self.breadth,
                "depth": self.depth,
                "total_evaluations": self.stats.total_calls if self.stats else len(all_candidates),
                "candidate_scores": [c.score for c in all_candidates],
                "avg_score": statistics.mean([c.score for c in all_candidates]),
                "std_score": statistics.stdev([c.score for c in all_candidates])
                if len(all_candidates) > 1
                else 0.0,
            },
        )

        # Add statistics if tracking
        if self.stats:
            result.metadata["results_best"] = self.stats.results_best
            result.metadata["results_latest"] = self.stats.results_latest

        return result


__all__ = [
    "COPRO",
    "COPROConfig",
    "COPROStats",
    "InstructionCandidate",
]
