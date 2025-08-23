"""Reflective Evolution Optimizer - LogiLLM's version of GEPA with hyperparameter awareness.

Uses reflection on execution traces to evolve both prompts and hyperparameters.
This is more advanced than DSPy's GEPA because it also considers hyperparameters.
"""

import copy
import time
from typing import Any, Optional

from ..core.modules import Module
from ..core.optimizers import Metric, Optimizer
from ..core.providers import Provider
from ..core.types import OptimizationResult, OptimizationStrategy


class ReflectiveEvolutionOptimizer(Optimizer):
    """LogiLLM's version of GEPA with hyperparameter awareness.

    Uses reflection on execution traces to evolve both prompts
    and hyperparameters. Key innovations over DSPy's GEPA:

    1. Reflects on hyperparameter impact (temperature, top_p, etc.)
    2. Uses textual feedback to improve both prompts and params
    3. Maintains Pareto frontier for multi-dimensional optimization
    4. Can merge successful candidates

    This optimizer uses an LLM to reflect on what went wrong and
    propose improvements, making it more adaptive than traditional
    optimization methods.
    """

    def __init__(
        self,
        metric: Metric,
        reflection_lm: Optional[Provider] = None,
        use_textual_feedback: bool = True,
        maintain_pareto: bool = True,
        n_iterations: int = 10,
        minibatch_size: int = 5,
        merge_candidates: bool = True,
        include_hyperparameters: bool = True,  # Our addition!
        pareto_size_limit: int = 10,
        **kwargs: Any,
    ):
        """Initialize ReflectiveEvolutionOptimizer.

        Args:
            metric: Evaluation metric
            reflection_lm: LLM for reflection (defaults to main provider)
            use_textual_feedback: Whether to use textual feedback
            maintain_pareto: Whether to maintain Pareto frontier
            n_iterations: Number of evolution iterations
            minibatch_size: Size of minibatch for evaluation
            merge_candidates: Whether to merge successful candidates
            include_hyperparameters: Reflect on hyperparameters (LogiLLM feature!)
            pareto_size_limit: Maximum size of Pareto frontier
        """
        super().__init__(
            strategy=OptimizationStrategy.EVOLUTION,
            metric=metric,
        )

        self.reflection_lm = reflection_lm
        self.use_textual_feedback = use_textual_feedback
        self.maintain_pareto = maintain_pareto
        self.n_iterations = n_iterations
        self.minibatch_size = minibatch_size
        self.merge_candidates = merge_candidates
        self.include_hyperparameters = include_hyperparameters
        self.pareto_size_limit = pareto_size_limit

        # Initialize Pareto frontier if maintaining
        self.pareto_frontier = [] if maintain_pareto else None

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize using reflective evolution.

        Args:
            module: Module to optimize
            dataset: Training dataset
            validation_set: Validation dataset
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with evolved module
        """
        start_time = time.time()

        # Initialize reflection LM if not provided
        if self.reflection_lm is None:
            self.reflection_lm = self._get_default_reflection_lm(module)

        # Initialize candidates
        candidates = [module]
        if self.pareto_frontier is not None:
            self.pareto_frontier = [(module, 0.0, {})]  # (module, score, metadata)

        eval_set = validation_set or dataset
        best_module = module
        best_score = 0.0
        evolution_history = []

        for iteration in range(self.n_iterations):
            # Select candidate (from Pareto frontier if maintaining)
            candidate = self._select_candidate(candidates)

            # Execute with trace collection on minibatch
            minibatch = dataset[: self.minibatch_size]
            traces = await self._execute_with_traces(candidate, minibatch)

            # Get feedback (textual if available)
            feedback = self._collect_feedback(traces, minibatch)

            # Reflect on execution to propose improvements
            improvements = await self._reflect_on_execution(
                candidate, traces, feedback, include_hyperparameters=self.include_hyperparameters
            )

            # Create new candidate with improvements
            new_candidate = await self._apply_improvements(
                candidate,
                improvements,
                dataset[:20],  # Use small subset for demo generation
            )

            # Evaluate new candidate
            score, _ = await self.evaluate(new_candidate, eval_set)

            # Update Pareto frontier or candidates
            if self.pareto_frontier is not None:
                self._update_pareto_frontier(new_candidate, score, improvements)
                candidates = [c[0] for c in self.pareto_frontier]
            else:
                candidates.append(new_candidate)
                if score > best_score:
                    best_score = score
                    best_module = new_candidate

            # Optionally merge candidates
            if self.merge_candidates and len(candidates) > 2:
                merged = await self._merge_candidates(candidates[:3])
                merged_score, _ = await self.evaluate(merged, eval_set)
                if merged_score > best_score:
                    best_score = merged_score
                    best_module = merged
                    candidates.append(merged)

            # Track evolution
            evolution_history.append(
                {
                    "iteration": iteration,
                    "score": score,
                    "improvements": improvements,
                    "pareto_size": len(self.pareto_frontier) if self.pareto_frontier else 0,
                }
            )

        # Get best from Pareto frontier or candidates
        if self.pareto_frontier:
            best_module, best_score, _ = max(self.pareto_frontier, key=lambda x: x[1])

        # Calculate baseline for improvement
        baseline_score, _ = await self.evaluate(module, eval_set)

        return OptimizationResult(
            optimized_module=best_module,
            improvement=best_score - baseline_score,
            iterations=self.n_iterations,
            best_score=best_score,
            optimization_time=time.time() - start_time,
            metadata={
                "evolution_history": evolution_history,
                "pareto_frontier_size": len(self.pareto_frontier) if self.pareto_frontier else 0,
                "baseline_score": baseline_score,
                "included_hyperparameters": self.include_hyperparameters,
                "used_textual_feedback": self.use_textual_feedback,
            },
        )

    def _select_candidate(self, candidates: list) -> Optional[Module]:
        """Select candidate for next iteration."""
        if self.pareto_frontier:
            # Stochastic selection from Pareto frontier
            # Prefer better scores but allow exploration
            import random

            weights = [c[1] + 0.1 for c in self.pareto_frontier]  # Add small base weight
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                selected = random.choices(self.pareto_frontier, weights=weights)[0]
                return selected[0]

        # Select most recent if no Pareto frontier
        return candidates[-1] if candidates else None

    async def _execute_with_traces(
        self, module: Module, dataset: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute module and collect traces."""
        traces = []

        for example in dataset:
            try:
                # Execute and collect trace
                inputs = example.get("inputs", {})
                expected = example.get("outputs", {})

                prediction = await module(**inputs)

                # Create trace
                trace = {
                    "inputs": inputs,
                    "outputs": prediction.outputs if hasattr(prediction, "outputs") else prediction,
                    "expected": expected,
                    "metadata": {
                        "success": prediction.success if hasattr(prediction, "success") else True,
                        "config": module.config.copy() if hasattr(module, "config") else {},
                        "temperature": module.config.get("temperature", 0.7)
                        if hasattr(module, "config")
                        else 0.7,
                        "top_p": module.config.get("top_p", 0.9)
                        if hasattr(module, "config")
                        else 0.9,
                    },
                }
                traces.append(trace)

            except Exception as e:
                # Include failed traces for learning
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Execution failed for example: {e}")

                trace = {
                    "inputs": inputs,
                    "outputs": {},
                    "expected": expected,
                    "metadata": {"error": str(e), "success": False, "error_type": type(e).__name__},
                }
                traces.append(trace)

        return traces

    def _collect_feedback(
        self, traces: list[dict[str, Any]], dataset: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Collect feedback from traces."""
        feedback = []

        for trace, example in zip(traces, dataset):
            expected = example.get("outputs", {})

            # Calculate score
            if trace["metadata"].get("success", False):
                score = self.metric(trace["outputs"], expected)
            else:
                score = 0.0

            # Create feedback entry
            feedback_entry = {
                "score": score,
                "success": trace["metadata"].get("success", False),
                "error": trace["metadata"].get("error"),
            }

            # Add textual feedback if enabled
            if self.use_textual_feedback:
                if score < 0.5:
                    feedback_entry["text"] = self._generate_textual_feedback(trace, expected, score)

            feedback.append(feedback_entry)

        return feedback

    def _generate_textual_feedback(
        self, trace: dict[str, Any], expected: dict[str, Any], score: float
    ) -> str:
        """Generate textual feedback for a trace."""
        if not trace["metadata"].get("success", False):
            return f"Execution failed: {trace['metadata'].get('error', 'Unknown error')}"

        # Analyze what went wrong
        feedback_parts = []

        if score < 0.3:
            feedback_parts.append("Output is significantly different from expected.")
        elif score < 0.5:
            feedback_parts.append("Output is partially correct but needs improvement.")

        # Check specific issues
        if trace["outputs"] and expected:
            for key in expected:
                if key not in trace["outputs"]:
                    feedback_parts.append(f"Missing expected field: {key}")
                elif trace["outputs"][key] != expected[key]:
                    feedback_parts.append(f"Incorrect value for {key}")

        # Check hyperparameter impact
        if self.include_hyperparameters:
            temp = trace["metadata"].get("temperature", 0.7)
            if temp > 1.0 and score < 0.5:
                feedback_parts.append("High temperature may be causing inconsistency")
            elif temp < 0.3 and not trace["outputs"]:
                feedback_parts.append("Low temperature may be too restrictive")

        return ". ".join(feedback_parts) if feedback_parts else "Output needs improvement"

    async def _reflect_on_execution(
        self,
        module: Module,
        traces: list[dict[str, Any]],
        feedback: list[dict[str, Any]],
        include_hyperparameters: bool = True,
    ) -> dict[str, Any]:
        """Reflect on execution to propose improvements.

        This is our key innovation: reflecting on hyperparameters too!
        """
        # Prepare reflection prompt
        reflection_prompt = self._build_reflection_prompt(
            module, traces, feedback, include_hyperparameters
        )

        # Use reflection LM to propose improvements
        if self.reflection_lm:
            # Convert string prompt to messages format
            messages = [{"role": "user", "content": reflection_prompt}]
            response = await self.reflection_lm.complete(messages)
            improvements = self._parse_improvements(response)
        else:
            # Fallback to heuristic improvements
            improvements = self._heuristic_improvements(traces, feedback)

        return improvements

    def _build_reflection_prompt(
        self,
        module: Module,
        traces: list[dict[str, Any]],
        feedback: list[dict[str, Any]],
        include_hyperparameters: bool,
    ) -> str:
        """Build prompt for reflection."""
        prompt_parts = [
            "Analyze these execution traces and feedback to suggest improvements.\n",
            "\nExecution Traces:",
        ]

        for i, (trace, fb) in enumerate(zip(traces[:3], feedback[:3])):  # Limit to 3 for context
            prompt_parts.append(f"\nExample {i + 1}:")
            prompt_parts.append(f"  Input: {trace['inputs']}")
            prompt_parts.append(f"  Output: {trace['outputs']}")
            prompt_parts.append(f"  Expected: {trace['expected']}")
            prompt_parts.append(f"  Score: {fb['score']:.2f}")
            if "text" in fb:
                prompt_parts.append(f"  Feedback: {fb['text']}")

        prompt_parts.append("\n\nCurrent Configuration:")
        if hasattr(module, "signature") and module.signature:
            prompt_parts.append(f"  Instructions: {module.signature.instructions}")
        if hasattr(module, "config"):
            prompt_parts.append(f"  Config: {module.config}")

        prompt_parts.append("\n\nSuggest improvements to:")
        prompt_parts.append("1. Instructions: How to clarify the task")
        prompt_parts.append("2. Demonstrations: Which examples would help")

        if include_hyperparameters:
            prompt_parts.append(
                "3. Hyperparameters: Temperature, top_p adjustments based on error patterns"
            )

        prompt_parts.append("\nProvide specific, actionable improvements in JSON format.")

        return "\n".join(prompt_parts)

    def _parse_improvements(self, response: Any) -> dict[str, Any]:
        """Parse improvements from reflection response."""
        improvements = {"instruction": None, "num_demos": None, "temperature": None, "top_p": None}

        # Extract text from Completion object if needed
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = str(response)

        # Try to parse JSON response
        try:
            import json

            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response_text[start:end])
                improvements.update(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to text parsing
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"JSON parsing failed, falling back to text parsing: {e}")
            if "temperature" in response_text.lower():
                if "increase" in response_text.lower() or "higher" in response_text.lower():
                    improvements["temperature"] = 0.1  # Increase by 0.1
                elif "decrease" in response_text.lower() or "lower" in response_text.lower():
                    improvements["temperature"] = -0.1  # Decrease by 0.1

            if "instruction" in response_text.lower():
                # Extract instruction improvements (simplified)
                improvements["instruction"] = "improved"

        return improvements

    def _heuristic_improvements(
        self, traces: list[dict[str, Any]], feedback: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate heuristic improvements without LLM reflection."""
        improvements = {}

        # Calculate average score
        scores = [fb["score"] for fb in feedback]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Suggest hyperparameter adjustments based on patterns
        if self.include_hyperparameters:
            if avg_score < 0.3:
                # Poor performance - try adjusting temperature
                improvements["temperature"] = -0.2  # Reduce randomness
            elif avg_score < 0.5:
                # Moderate performance - fine-tune
                improvements["temperature"] = -0.1

            # Check for consistency issues
            score_variance = (
                sum((s - avg_score) ** 2 for s in scores) / len(scores) if scores else 0
            )
            if score_variance > 0.1:
                # High variance - reduce temperature
                improvements["temperature"] = -0.15
                improvements["top_p"] = -0.1

        # Suggest more demonstrations if performance is poor
        if avg_score < 0.5:
            improvements["num_demos"] = 2  # Add 2 more demos

        return improvements

    async def _apply_improvements(
        self, module: Module, improvements: dict[str, Any], dataset: list[dict[str, Any]]
    ) -> Module:
        """Apply improvements to create new candidate."""
        improved = copy.deepcopy(module)

        # Apply hyperparameter improvements
        if improvements.get("temperature") is not None:
            current_temp = (
                improved.config.get("temperature", 0.7) if hasattr(improved, "config") else 0.7
            )
            new_temp = max(0.0, min(2.0, current_temp + improvements["temperature"]))
            if hasattr(improved, "config"):
                improved.config["temperature"] = new_temp

        if improvements.get("top_p") is not None:
            current_top_p = (
                improved.config.get("top_p", 0.9) if hasattr(improved, "config") else 0.9
            )
            new_top_p = max(0.1, min(1.0, current_top_p + improvements["top_p"]))
            if hasattr(improved, "config"):
                improved.config["top_p"] = new_top_p

        # Apply instruction improvements
        if improvements.get("instruction"):
            if hasattr(improved, "signature") and improved.signature:
                # Enhance instruction (simplified)
                current = improved.signature.instructions
                improved.signature.instructions = f"{current}. Be precise and thorough."

        # Apply demonstration improvements
        if improvements.get("num_demos"):
            from .bootstrap_fewshot import BootstrapFewShot

            # Add more demonstrations
            demo_opt = BootstrapFewShot(
                metric=self.metric, max_bootstrapped_demos=improvements["num_demos"]
            )
            demo_result = await demo_opt.optimize(improved, dataset)
            improved = demo_result.optimized_module

        return improved

    def _update_pareto_frontier(self, module: Module, score: float, metadata: dict[str, Any]):
        """Update Pareto frontier with new candidate."""
        if not self.pareto_frontier:
            self.pareto_frontier = [(module, score, metadata)]
            return

        # Check if dominated by any existing
        dominated = False
        for _existing_module, existing_score, _ in self.pareto_frontier:
            if existing_score >= score:
                # Check if also dominates on other dimensions (e.g., complexity)
                # For now, just use score
                dominated = True
                break

        if not dominated:
            # Remove any dominated by new candidate
            self.pareto_frontier = [
                (m, s, meta)
                for m, s, meta in self.pareto_frontier
                if s > score  # Keep if better score
            ]
            self.pareto_frontier.append((module, score, metadata))

        # Limit size
        if len(self.pareto_frontier) > self.pareto_size_limit:
            # Keep best ones
            self.pareto_frontier.sort(key=lambda x: x[1], reverse=True)
            self.pareto_frontier = self.pareto_frontier[: self.pareto_size_limit]

    async def _merge_candidates(self, candidates: list[Module]) -> Module:
        """Merge multiple successful candidates."""
        if not candidates:
            raise ValueError("No candidates to merge")

        # Start with first candidate
        merged = copy.deepcopy(candidates[0])

        # Merge configurations (average hyperparameters)
        if hasattr(merged, "config"):
            temps = []
            top_ps = []
            for candidate in candidates:
                if hasattr(candidate, "config"):
                    temps.append(candidate.config.get("temperature", 0.7))
                    top_ps.append(candidate.config.get("top_p", 0.9))

            if temps:
                merged.config["temperature"] = sum(temps) / len(temps)
            if top_ps:
                merged.config["top_p"] = sum(top_ps) / len(top_ps)

        # Merge demonstrations (take best from each)
        all_demos = []
        for candidate in candidates:
            if hasattr(candidate, "parameters") and "demonstrations" in candidate.parameters:
                demos = candidate.parameters["demonstrations"].value
                if demos:
                    all_demos.extend(demos[:2])  # Take top 2 from each

        if all_demos and hasattr(merged, "parameters"):
            from ..core.modules import Parameter

            merged.parameters["demonstrations"] = Parameter(
                value=all_demos[:6],  # Limit total
                learnable=True,
                metadata={"source": "merged"},
            )

        return merged

    def _get_default_reflection_lm(self, module: Module) -> Optional[Provider]:
        """Get default reflection LM from module or environment."""
        # Try to use module's provider
        if hasattr(module, "provider") and module.provider:
            provider = module.provider
            # Check if it's actually a Provider instance
            if isinstance(provider, Provider):
                return provider

        # Try to get default provider
        try:
            from ..core.providers import get_provider

            return get_provider()
        except ImportError as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Could not import get_provider: {e}")
            return None
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Could not get default provider: {e}")
            return None


__all__ = ["ReflectiveEvolutionOptimizer"]
