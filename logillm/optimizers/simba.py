"""SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer for LogiLLM."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ..core.modules import Module, Parameter
from ..core.optimizers import Metric, OptimizationResult
from ..core.types import OptimizationStrategy
from ..exceptions import OptimizationError
from .base import Demonstration, PromptOptimizationConfig, PromptOptimizer
from .simba_utils import (
    OfferFeedback,
    inspect_modules,
    prepare_models_for_resampling,
    recursive_mask,
    wrap_program,
)

logger = logging.getLogger(__name__)


@dataclass
class SIMBAConfig(PromptOptimizationConfig):
    """Configuration for SIMBA optimizer."""

    bsize: int = 32
    num_candidates: int = 6
    max_steps: int = 8
    demo_input_field_maxlen: int = 100_000
    num_threads: int | None = None
    temperature_for_sampling: float = 0.2
    temperature_for_candidates: float = 0.2


class SIMBA(PromptOptimizer):
    """SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer.

    SIMBA is an optimization algorithm that uses mini-batch sampling,
    introspective rule generation, and demo appending to improve LLM programs.

    Key features:
    - Mini-batch sampling (bsize parameter)
    - Introspective rule generation (append_a_rule)
    - Demo appending (append_a_demo)
    - Multi-candidate generation (num_candidates)
    - Temperature-based sampling for trajectory selection
    - Parallel evaluation support
    """

    def __init__(
        self,
        metric: Metric,
        *,
        bsize: int = 32,
        num_candidates: int = 6,
        max_steps: int = 8,
        max_demos: int = 4,
        demo_input_field_maxlen: int = 100_000,
        num_threads: int | None = None,
        temperature_for_sampling: float = 0.2,
        temperature_for_candidates: float = 0.2,
        config: SIMBAConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SIMBA optimizer.

        Args:
            metric: Metric function that takes prediction and reference
            bsize: Mini-batch size
            num_candidates: Number of new candidate programs per iteration
            max_steps: Number of optimization steps to run
            max_demos: Maximum demos per predictor before dropping some
            demo_input_field_maxlen: Max characters in demo input fields
            num_threads: Number of threads for parallel execution
            temperature_for_sampling: Temperature for trajectory sampling
            temperature_for_candidates: Temperature for candidate selection
            config: Full configuration object (overrides individual params)
        """
        # Create or update config
        if config is None:
            # Create parent config first
            base_config = PromptOptimizationConfig(
                strategy=OptimizationStrategy.REFLECTION,
                max_iterations=max_steps,
                max_demos=max_demos,
            )

            # Create SIMBA-specific config with base values
            config = SIMBAConfig(
                # Copy from base config
                strategy=base_config.strategy,
                max_iterations=base_config.max_iterations,
                max_demos=base_config.max_demos,
                target_score=base_config.target_score,
                early_stopping=base_config.early_stopping,
                patience=base_config.patience,
                batch_size=base_config.batch_size,
                learning_rate=base_config.learning_rate,
                temperature=base_config.temperature,
                exploration_rate=base_config.exploration_rate,
                metadata=base_config.metadata,
                max_instructions=base_config.max_instructions,
                teacher_settings=base_config.teacher_settings,
                demo_selection_strategy=base_config.demo_selection_strategy,
                instruction_generation_strategy=base_config.instruction_generation_strategy,
                use_validation_for_selection=base_config.use_validation_for_selection,
                min_demo_score=base_config.min_demo_score,
                # SIMBA-specific params
                bsize=bsize,
                num_candidates=num_candidates,
                max_steps=max_steps,
                demo_input_field_maxlen=demo_input_field_maxlen,
                num_threads=num_threads,
                temperature_for_sampling=temperature_for_sampling,
                temperature_for_candidates=temperature_for_candidates,
            )

        super().__init__(
            strategy=OptimizationStrategy.REFLECTION, metric=metric, config=config, **kwargs
        )

        # SIMBA-specific attributes
        self.bsize = config.bsize
        self.num_candidates = config.num_candidates
        self.max_steps = config.max_steps
        self.max_demos = config.max_demos
        self.demo_input_field_maxlen = config.demo_input_field_maxlen
        self.num_threads = config.num_threads
        self.temperature_for_sampling = config.temperature_for_sampling
        self.temperature_for_candidates = config.temperature_for_candidates

        # Set up optimization strategies
        if self.max_demos > 0:
            self.strategies = [self._append_demo, self._append_rule]
        else:
            self.strategies = [self._append_rule]

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: list[dict[str, Any]] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize module using SIMBA algorithm.

        Args:
            module: Module to optimize
            dataset: Training dataset
            validation_set: Optional validation dataset
            seed: Random seed for reproducibility

        Returns:
            OptimizationResult with optimized module and metadata
        """
        # Basic checks
        if len(dataset) < self.bsize:
            raise OptimizationError(
                f"Dataset too small: {len(dataset)} < {self.bsize}",
                optimizer_type="SIMBA",
                context={"dataset_size": len(dataset), "required_size": self.bsize},
            )

        # Initialize RNG
        rng = random.Random(seed)

        # Initialize tracking variables
        programs = []
        program_scores: dict[int, list[float]] = {}
        next_program_idx = 0
        self._start_time = datetime.now()

        # Helper functions
        def calc_average_score(prog_idx: int) -> float:
            scores = program_scores.get(prog_idx, [])
            return sum(scores) / len(scores) if scores else 0.0

        def top_k_plus_baseline(k: int) -> list[int]:
            # Sort all programs by descending average score
            scored_programs = sorted(
                programs, key=lambda p: calc_average_score(p.simba_idx), reverse=True
            )
            top_k = [p.simba_idx for p in scored_programs[:k]]
            # Ensure baseline=0 is in there
            if 0 not in top_k and len(top_k) > 0:
                top_k[-1] = 0
            return list(dict.fromkeys(top_k))

        def register_new_program(prog: Module, score_list: list[float]) -> None:
            nonlocal next_program_idx
            next_program_idx += 1
            new_idx = next_program_idx
            prog.simba_idx = new_idx  # type: ignore
            programs.append(prog)
            program_scores[new_idx] = score_list

        # Initialize the baseline program: index=0
        student = module.deepcopy()
        student.simba_idx = 0  # type: ignore
        programs.append(student)
        program_scores[0] = []

        winning_programs = [student]

        # Data shuffling
        data_indices = list(range(len(dataset)))
        rng.shuffle(data_indices)
        instance_idx = 0

        trial_logs = {}

        # Main optimization loop
        for batch_idx in range(self.max_steps):
            self._iteration = batch_idx
            trial_logs[batch_idx] = {}

            logger.info(f"Starting batch {batch_idx + 1} of {self.max_steps}.")

            # STEP 1: Get next mini-batch
            if instance_idx + self.bsize > len(dataset):
                rng.shuffle(data_indices)
                instance_idx = 0

            batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
            batch = [dataset[i] for i in batch_indices]
            instance_idx += self.bsize

            # STEP 2: Generate program trajectories
            models = await prepare_models_for_resampling(programs[0], self.num_candidates)
            top_programs = top_k_plus_baseline(self.num_candidates)

            exec_pairs = []
            predictor2name = {}

            # For each model, for each example, pick a program from the pool via softmax
            for model in models:
                for example in batch:
                    chosen_prog_idx = self._softmax_sample(
                        rng, top_programs, self.temperature_for_sampling, calc_average_score
                    )
                    candidate_system = programs[chosen_prog_idx].deepcopy()

                    # Set the model (simplified - in real DSPy this sets LM)
                    candidate_system.metadata["temperature"] = getattr(model, "temperature", 0.7)

                    # Track predictor names for later use
                    for name, predictor in self._named_predictors(candidate_system):
                        predictor2name[id(predictor)] = name

                    # Wrap program for execution
                    wrapped_candidate = wrap_program(candidate_system, self.metric)
                    exec_pairs.append((wrapped_candidate, example))

            # STEP 3: Execute trajectories in parallel
            logger.info(
                f"Sampling program trajectories on {self.bsize} examples x {self.num_candidates} samples."
            )
            outputs = await self._evaluate_batch(exec_pairs)

            # STEP 4: Sort training buckets by performance gaps
            buckets = []
            batch_scores = [float(o["score"]) for o in outputs]
            batch_10th_percentile = (
                sorted(batch_scores)[int(0.1 * len(batch_scores))] if batch_scores else 0.0
            )
            batch_90th_percentile = (
                sorted(batch_scores)[int(0.9 * len(batch_scores))] if batch_scores else 0.0
            )

            # Group outputs by example
            for idx in range(self.bsize):
                # Gather all results for this example
                bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
                bucket.sort(key=lambda x: x["score"], reverse=True)

                max_score = float(bucket[0]["score"])
                min_score = float(bucket[-1]["score"])
                avg_score = sum(x["score"] for x in bucket) / len(bucket)
                max_to_min_gap = max_score - min_score
                max_to_avg_gap = max_score - avg_score

                buckets.append((bucket, (max_to_min_gap, max_score, max_to_avg_gap)))

            # Sort buckets by performance gaps
            buckets.sort(key=lambda x: x[1], reverse=True)

            baseline_score = sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
            logger.info(f"Batch {batch_idx + 1}: Baseline mini-batch score: {baseline_score}")

            # STEP 5: Build new candidate programs
            system_candidates = []
            for bucket_idx, (bucket, bucket_stats) in enumerate(buckets):
                max_to_min_gap, max_score, max_to_avg_gap = bucket_stats
                logger.info(
                    f"Batch {batch_idx + 1}: Processing bucket #{bucket_idx + 1}, "
                    f"max score {max_score}, gaps: min={max_to_min_gap:.3f}, avg={max_to_avg_gap:.3f}"
                )

                # Pick source program
                src_prog_idx = self._softmax_sample(
                    rng,
                    top_k_plus_baseline(self.num_candidates),
                    self.temperature_for_candidates,
                    calc_average_score,
                )
                system_candidate = programs[src_prog_idx].deepcopy()

                # Drop some demos from each predictor
                self._drop_random_demos(system_candidate, rng)

                # Pick and apply strategy
                strategy = rng.choice(self.strategies)
                logger.info(f"Batch {batch_idx + 1}: Applying strategy: {strategy.__name__}")

                try:
                    success = await strategy(
                        bucket,
                        system_candidate,
                        predictor2name=predictor2name,
                        batch_10p_score=batch_10th_percentile,
                        batch_90p_score=batch_90th_percentile,
                    )
                    if success:
                        system_candidates.append(system_candidate)
                except Exception as e:
                    logger.error(f"Strategy {strategy.__name__} failed: {e}")
                    continue

                if len(system_candidates) >= self.num_candidates + 1:
                    break

            # STEP 6: Evaluate new candidates
            if system_candidates:
                logger.info(
                    f"Batch {batch_idx + 1}: Evaluating {len(system_candidates)} new programs."
                )

                eval_pairs = [
                    (wrap_program(sys, self.metric), ex)
                    for sys in system_candidates
                    for ex in batch
                ]
                eval_outputs = await self._evaluate_batch(eval_pairs)

                # Compute candidate scores
                candidate_scores = []
                for idx_cand in range(len(system_candidates)):
                    start = idx_cand * self.bsize
                    end = (idx_cand + 1) * self.bsize
                    sys_scores = [eval_outputs[i]["score"] for i in range(start, end)]
                    avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
                    candidate_scores.append(avg_score)

                logger.info(
                    f"Candidate scores: {candidate_scores}, Best: {max(candidate_scores) if candidate_scores else 'N/A'}"
                )

                # Track best candidate
                if candidate_scores:
                    best_idx = candidate_scores.index(max(candidate_scores))
                    best_program = system_candidates[best_idx]
                    winning_programs.append(best_program.deepcopy())

                # Register all new candidates
                for idx_cand, cand_sys in enumerate(system_candidates):
                    start = idx_cand * self.bsize
                    end = (idx_cand + 1) * self.bsize
                    sys_scores = [eval_outputs[i]["score"] for i in range(start, end)]
                    register_new_program(cand_sys, sys_scores)

        # STEP 7: Final evaluation and selection
        M = len(winning_programs) - 1
        N = self.num_candidates + 1
        if M < 1:
            program_idxs = [0] * N
        else:
            program_idxs = [round(i * M / (N - 1)) for i in range(N)]

        program_idxs = list(dict.fromkeys(program_idxs))
        candidate_programs = [winning_programs[i].deepcopy() for i in program_idxs]

        logger.info(
            f"Final evaluation: Testing {len(candidate_programs)} programs on full dataset."
        )

        # Evaluate all candidates on full dataset
        final_eval_pairs = [
            (wrap_program(sys, self.metric), ex) for sys in candidate_programs for ex in dataset
        ]
        final_outputs = await self._evaluate_batch(final_eval_pairs)

        scores = []
        for idx_prog in range(len(candidate_programs)):
            start = idx_prog * len(dataset)
            end = (idx_prog + 1) * len(dataset)
            sys_scores = [final_outputs[i]["score"] for i in range(start, end)]
            avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
            scores.append(avg_score)

        # Find best program
        best_idx = scores.index(max(scores)) if scores else 0
        best_program = candidate_programs[best_idx].deepcopy()
        best_score = max(scores) if scores else 0.0

        logger.info(f"Final scores: {scores}, Best: {best_score} (at index {best_idx})")

        # Attach metadata
        candidate_data = [{"score": s, "program": p} for s, p in zip(scores, candidate_programs)]
        candidate_data.sort(key=lambda x: x["score"], reverse=True)

        best_program.candidate_programs = candidate_data  # type: ignore
        best_program.trial_logs = trial_logs  # type: ignore

        optimization_time = (datetime.now() - self._start_time).total_seconds()

        return OptimizationResult(
            optimized_module=best_program,
            improvement=best_score - scores[0] if len(scores) > 1 else best_score,
            iterations=self.max_steps,
            best_score=best_score,
            optimization_time=optimization_time,
            metadata={
                "trial_logs": trial_logs,
                "candidate_programs": candidate_data,
                "final_scores": scores,
            },
        )

    def _softmax_sample(
        self,
        rng: random.Random,
        program_idxs: list[int],
        temperature: float,
        score_fn: Callable[[int], float],
    ) -> int:
        """Sample program index using softmax with temperature."""
        if not program_idxs:
            raise ValueError("No programs available for softmax sampling")

        scores = [score_fn(idx) for idx in program_idxs]

        # Apply temperature and compute softmax
        import math

        max_score = max(scores) if scores else 0
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        sum_exp = sum(exp_scores)

        if sum_exp <= 0:
            return rng.choice(program_idxs)

        probs = [exp_s / sum_exp for exp_s in exp_scores]
        return rng.choices(program_idxs, weights=probs, k=1)[0]

    def _named_predictors(self, module: Module) -> list[tuple[str, Any]]:
        """Extract named predictors from module."""
        predictors = []

        # Look for common predictor patterns in LogiLLM
        if hasattr(module, "parameters"):
            for name, param in module.parameters.items():
                if "predictor" in name.lower() or "predict" in name.lower():
                    predictors.append((name, param))

        # If no predictors found, use the module itself
        if not predictors:
            predictors.append((module.__class__.__name__, module))

        return predictors

    def _drop_random_demos(self, module: Module, rng: random.Random) -> None:
        """Randomly drop some demonstrations from predictors."""
        if self.max_demos <= 0:
            return

        max_demos_tmp = self.max_demos if self.max_demos > 0 else 3

        # Get current demo counts
        demo_counts = []
        for _name, predictor in self._named_predictors(module):
            if hasattr(predictor, "parameters") and "demonstrations" in predictor.parameters:
                demo_param = predictor.parameters["demonstrations"]
                if demo_param.value:
                    demo_counts.append(len(demo_param.value))

        if not demo_counts:
            return

        num_demos = max(demo_counts)
        if num_demos < max_demos_tmp:
            return

        # Use simple random approach instead of Poisson (to avoid numpy dependency)
        # Approximate Poisson behavior with simple probability
        avg_to_drop = num_demos / max_demos_tmp
        num_demos_to_drop = max(
            int(avg_to_drop + rng.gauss(0, avg_to_drop**0.5)) if avg_to_drop > 0 else 0,
            int(num_demos >= max_demos_tmp),
        )
        num_demos_to_drop = min(num_demos_to_drop, num_demos)

        if num_demos_to_drop > 0:
            indices_to_drop = [rng.randrange(num_demos) for _ in range(num_demos_to_drop)]

            # Remove demos from all predictors
            for _name, predictor in self._named_predictors(module):
                if hasattr(predictor, "parameters") and "demonstrations" in predictor.parameters:
                    demo_param = predictor.parameters["demonstrations"]
                    if demo_param.value:
                        new_demos = [
                            demo
                            for i, demo in enumerate(demo_param.value)
                            if i not in indices_to_drop
                        ]
                        demo_param.value = new_demos

    async def _append_demo(
        self, bucket: list[dict[str, Any]], system: Module, **kwargs: Any
    ) -> bool:
        """Append successful demonstration to predictors."""
        predictor2name = kwargs.get("predictor2name", {})

        if not bucket:
            return False

        # Get the best trace from the bucket
        best_result = bucket[0]
        trace = best_result.get("trace", [])

        if not trace:
            return False

        name2demo = {}

        # Extract demonstrations from trace
        for step in trace:
            predictor, inputs, outputs = step

            # Truncate long inputs
            if self.demo_input_field_maxlen:
                for k, v in inputs.items():
                    if isinstance(v, str) and len(v) > self.demo_input_field_maxlen:
                        inputs[k] = (
                            f"{v[: self.demo_input_field_maxlen]}\n\t\t... <TRUNCATED FOR BREVITY>"
                        )

            # Create demonstration
            demo = Demonstration(
                inputs=inputs,
                outputs=dict(outputs),
                score=float(best_result["score"]),
                metadata={"augmented": True},
            )

            predictor_name = predictor2name.get(id(predictor), "default")
            name2demo[predictor_name] = demo

        # Add demos to predictors
        demo_count = 0
        for name, predictor in self._named_predictors(system):
            if name in name2demo:
                demo = name2demo[name]

                # Add to demonstrations parameter
                if not hasattr(predictor, "parameters"):
                    predictor.parameters = {}

                if "demonstrations" not in predictor.parameters:
                    predictor.parameters["demonstrations"] = Parameter(value=[], learnable=True)

                predictor.parameters["demonstrations"].value.append(demo.to_dict())
                demo_count += 1

        logger.info(f"Added {demo_count} demonstrations across predictors.")
        return demo_count > 0

    async def _append_rule(
        self, bucket: list[dict[str, Any]], system: Module, **kwargs: Any
    ) -> bool:
        """Generate and append introspective rule to predictors."""
        predictor2name = kwargs.get("predictor2name", {})
        batch_10p_score = kwargs.get("batch_10p_score", 0.0)
        batch_90p_score = kwargs.get("batch_90p_score", 1.0)

        if len(bucket) < 2:
            logger.info("Not enough examples in bucket for rule generation.")
            return False

        good, bad = bucket[0], bucket[-1]
        example = good.get("example")

        if not example:
            logger.info("No example found in bucket.")
            return False

        # Check if scores are in reasonable range
        if good["score"] < batch_10p_score or bad["score"] > batch_90p_score:
            logger.info(
                f"Skipping rule generation: good score {good['score']} < 10p {batch_10p_score} "
                f"or bad score {bad['score']} > 90p {batch_90p_score}"
            )
            return False

        if good["score"] <= bad["score"]:
            # Handle edge case where good isn't better than bad
            if good["score"] > batch_90p_score:
                bad["trace"] = []
                bad["score"] = "N/A"
                bad["prediction"] = {"N/A": "Prediction not available"}
            else:
                good["trace"] = []
                good["score"] = "N/A"
                good["prediction"] = {"N/A": "Prediction not available"}

        # Build trajectories for introspection
        better_trajectory = [
            {"module_name": predictor2name.get(id(p), "unknown"), "inputs": i, "outputs": dict(o)}
            for p, i, o in good.get("trace", [])
        ]

        worse_trajectory = [
            {"module_name": predictor2name.get(id(p), "unknown"), "inputs": i, "outputs": dict(o)}
            for p, i, o in bad.get("trace", [])
        ]

        # Prepare introspection inputs
        module_names = [name for name, _ in self._named_predictors(system)]

        try:
            # Get program source (simplified)
            program_code = f"class {system.__class__.__name__}(Module): pass"

            introspection_inputs = {
                "program_code": program_code,
                "modules_defn": inspect_modules(system),
                "program_inputs": json.dumps(recursive_mask(example.get("inputs", {})), indent=2),
                "oracle_metadata": json.dumps(recursive_mask(example.get("outputs", {})), indent=2),
                "better_program_trajectory": json.dumps(better_trajectory, indent=2),
                "better_program_outputs": json.dumps(
                    recursive_mask(good.get("prediction", {})), indent=2
                ),
                "worse_program_trajectory": json.dumps(worse_trajectory, indent=2),
                "worse_program_outputs": json.dumps(
                    recursive_mask(bad.get("prediction", {})), indent=2
                ),
                "worse_reward_value": float(bad["score"])
                if isinstance(bad["score"], (int, float))
                else 0.0,
                "better_reward_value": float(good["score"]),
                "module_names": module_names,
            }

            # Generate advice using introspective module
            from ..core.predict import Predict

            feedback_module = Predict(OfferFeedback)
            feedback_result = await feedback_module(**introspection_inputs)

            if not feedback_result.success:
                logger.error(f"Feedback generation failed: {feedback_result.error}")
                return False

            advice = feedback_result.outputs.get("module_advice", {})
            if isinstance(advice, str):
                try:
                    advice = json.loads(advice)
                except json.JSONDecodeError:
                    logger.error("Could not parse advice as JSON")
                    return False

            # Apply advice to predictors
            rules_applied = 0
            for name, predictor in self._named_predictors(system):
                if name in advice:
                    rule_text = advice[name]
                    logger.info(f"Applying rule to {name}: {rule_text}")

                    # Add rule to instruction parameter
                    if not hasattr(predictor, "parameters"):
                        predictor.parameters = {}

                    if "instruction" not in predictor.parameters:
                        # Get existing instruction from signature if available
                        existing_instruction = ""
                        if hasattr(predictor, "signature") and predictor.signature:
                            existing_instruction = getattr(predictor.signature, "instructions", "")
                        predictor.parameters["instruction"] = Parameter(
                            value=existing_instruction, learnable=True
                        )

                    # Append the new rule
                    current_instruction = predictor.parameters["instruction"].value
                    updated_instruction = current_instruction + "\n\n" + rule_text
                    predictor.parameters["instruction"].value = updated_instruction
                    rules_applied += 1

            logger.info(f"Applied {rules_applied} introspective rules.")
            return rules_applied > 0

        except Exception as e:
            logger.error(f"Rule generation failed: {e}")
            return False

    async def _evaluate_batch(
        self, exec_pairs: list[tuple[Callable, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Evaluate batch of (wrapped_program, example) pairs in parallel."""
        # Limit concurrency to avoid overwhelming the system
        max_concurrent = self.num_threads or 10
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_single(
            wrapped_program: Callable, example: dict[str, Any]
        ) -> dict[str, Any]:
            async with semaphore:
                try:
                    return wrapped_program(example)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    return {
                        "prediction": None,
                        "trace": [],
                        "score": 0.0,
                        "example": example,
                        "error": str(e),
                    }

        # Execute all evaluations concurrently
        tasks = [evaluate_single(wrapped_prog, ex) for wrapped_prog, ex in exec_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that slipped through
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "prediction": None,
                        "trace": [],
                        "score": 0.0,
                        "example": {},
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results


__all__ = ["SIMBA", "SIMBAConfig"]
