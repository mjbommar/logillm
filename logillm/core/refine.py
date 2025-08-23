"""Refine module for LogiLLM - iteratively improves outputs through multiple attempts."""

from __future__ import annotations

import copy
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .modules import Module
from .predict import Predict
from .signatures import Signature
from .signatures.fields import InputField, OutputField
from .types import Configuration, Metadata, Prediction, Usage


@dataclass
class RefinementAttempt:
    """Information about a refinement attempt."""

    attempt_number: int
    timestamp: float
    temperature: float
    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    success: bool = False
    duration: float = 0.0
    trace: list[Any] = field(default_factory=list)


@dataclass
class RefinementHistory:
    """Complete history of refinement attempts."""

    attempts: list[RefinementAttempt] = field(default_factory=list)
    best_attempt: RefinementAttempt | None = None
    best_reward: float = float("-inf")

    def add_attempt(self, attempt: RefinementAttempt) -> None:
        """Add an attempt to the history."""
        self.attempts.append(attempt)

        if attempt.reward > self.best_reward:
            self.best_reward = attempt.reward
            self.best_attempt = attempt


class OfferFeedback(Signature):
    """Signature for generating feedback and improvement advice.

    This signature is used internally by the Refine module to generate
    advice on how modules should improve their performance based on
    execution traces and reward feedback.
    """

    program_code: str = InputField(description="The code of the program that we are analyzing")
    modules_defn: str = InputField(
        description="The definition of each module in the program, including its I/O"
    )
    program_inputs: str = InputField(description="The inputs to the program that we are analyzing")
    program_trajectory: str = InputField(
        description="The trajectory of the program's execution, showing each module's I/O"
    )
    program_outputs: str = InputField(
        description="The outputs of the program that we are analyzing"
    )
    reward_code: str = InputField(
        description="The code of the reward function that we are analyzing"
    )
    target_threshold: float = InputField(description="The target threshold for the reward function")
    reward_value: float = InputField(
        description="The reward value assigned to the program's outputs"
    )
    module_names: list[str] = InputField(
        description="The names of the modules in the program, for which we seek advice"
    )

    discussion: str = OutputField(
        description="Discussing blame of where each module went wrong, if it did"
    )
    advice: dict[str, str] = OutputField(
        description="For each module, describe very concretely, in this order: the specific scenarios in which it has made mistakes in the past and what each mistake was, followed by what it should do differently in that kind of scenario in the future. If the module is not to blame, write N/A."
    )

    __doc__ = """
    In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
    any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
    it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
    The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
    to avoid the same mistake on the same or similar inputs.
    """


class Refine(Module):
    """Module that refines outputs through multiple attempts with varying temperatures.

    This module runs the provided module multiple times with different temperature settings
    and selects either the first prediction that exceeds the specified threshold or the one
    with the highest reward. If no prediction meets the threshold, it automatically generates
    feedback to improve future predictions.

    Key features:
    - Temperature-based variation for diverse outputs
    - Reward function evaluation
    - Threshold-based early stopping
    - Automatic feedback generation using LLM
    - Execution trace analysis
    - Support for custom reward functions

    Example:
        ```python
        import asyncio
        from logillm.core import Predict, Refine

        # Define a QA module
        qa = Predict("question -> answer")

        # Define a reward function that prefers shorter answers
        def brevity_reward(inputs, prediction):
            if not prediction.success or not prediction.outputs.get('answer'):
                return 0.0
            answer_length = len(prediction.outputs['answer'].split())
            return max(0.0, 1.0 - (answer_length - 1) * 0.1)  # Prefer 1-word answers

        # Create a refined module that tries up to 3 times
        refined_qa = Refine(
            module=qa,
            N=3,
            reward_fn=brevity_reward,
            threshold=0.8,
            fail_count=2
        )

        # Use the refined module
        result = await refined_qa(question="What is the capital of Belgium?")
        print(result.outputs['answer'])  # Should prefer shorter answers like "Brussels"
        ```
    """

    def __init__(
        self,
        module: Module,
        N: int,
        reward_fn: Callable[[dict[str, Any], Prediction], float],
        threshold: float,
        fail_count: int | None = None,
        *,
        config: Configuration | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        """Initialize the Refine module.

        Args:
            module: The module to refine
            N: The number of times to run the module (must be >= 1)
            reward_fn: Function that evaluates prediction quality (inputs, prediction) -> float
            threshold: Target threshold for the reward function
            fail_count: Number of failures allowed before giving up (defaults to N)
            config: Additional configuration
            metadata: Module metadata
        """
        super().__init__(signature=module.signature, config=config, metadata=metadata)

        if N < 1:
            raise ValueError("N must be at least 1")

        self.module = module
        self.N = N
        self.reward_fn = reward_fn
        self.threshold = threshold
        self.fail_count = fail_count if fail_count is not None else N

        # Track refinement history
        self.history = RefinementHistory()

        # Cache module code for feedback generation
        try:
            self.module_code = inspect.getsource(module.__class__)
        except (TypeError, OSError):
            self.module_code = f"<Module: {module.__class__.__name__}>"

        # Cache reward function code for feedback generation
        try:
            self.reward_fn_code = inspect.getsource(reward_fn)
        except (TypeError, OSError):
            try:
                self.reward_fn_code = inspect.getsource(reward_fn.__class__)
            except (TypeError, OSError):
                self.reward_fn_code = f"<Function: {reward_fn.__name__ if hasattr(reward_fn, '__name__') else 'reward_function'}>"

    def _generate_temperature_sequence(self, base_temperature: float = 0.7) -> list[float]:
        """Generate a sequence of temperatures for refinement attempts."""
        if self.N == 1:
            return [base_temperature]

        # Start with base temperature, then create a sequence
        temps = [base_temperature]

        # Add varied temperatures
        for i in range(1, self.N):
            # Create temperature variation: some lower (more focused), some higher (more creative)
            temp = 0.5 + i * (0.5 / self.N)
            temps.append(temp)

        # Remove duplicates while preserving order
        seen = set()
        unique_temps = []
        for temp in temps:
            if temp not in seen:
                seen.add(temp)
                unique_temps.append(temp)

        return unique_temps[: self.N]

    def _inspect_modules(self, module: Module) -> str:
        """Generate inspection string for module definitions."""
        separator = "-" * 80
        output = [separator]

        # For now, just inspect the main module
        # In a more complex implementation, this would walk the module tree
        output.append(f"Module {module.__class__.__name__}")

        if module.signature:
            output.append("\n\tInput Fields:")
            if hasattr(module.signature, "input_fields"):
                for name, field in module.signature.input_fields.items():
                    desc = getattr(field, "description", "No description")
                    output.append(f"\t\t{name}: {desc}")

            output.append("\tOutput Fields:")
            if hasattr(module.signature, "output_fields"):
                for name, field in module.signature.output_fields.items():
                    desc = getattr(field, "description", "No description")
                    output.append(f"\t\t{name}: {desc}")

            instructions = getattr(module.signature, "instructions", None)
            if not instructions and hasattr(module.signature, "__doc__"):
                instructions = module.signature.__doc__
            if instructions:
                output.append(f"\tInstructions: {instructions}")

        output.append(separator)
        return "\n".join(output)

    def _recursive_mask(self, obj: Any) -> Any:
        """Recursively mask non-serializable objects for JSON encoding."""
        try:
            # Test if object is JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass

        # Handle different types
        if isinstance(obj, dict):
            return {k: self._recursive_mask(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_mask(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._recursive_mask(v) for v in obj)
        else:
            # Replace non-serializable with placeholder
            return f"<non-serializable: {type(obj).__name__}>"

    async def _generate_feedback(
        self, inputs: dict[str, Any], outputs: dict[str, Any], reward: float, trace: list[Any]
    ) -> dict[str, str]:
        """Generate improvement feedback based on execution trace and reward."""
        try:
            # Prepare feedback generation inputs
            modules_defn = self._inspect_modules(self.module)

            # Create simplified trace representation
            trajectory = []
            if trace:
                for step in trace:
                    if hasattr(step, "__dict__"):
                        step_dict = {
                            "module_name": getattr(step, "module_name", "Unknown"),
                            "inputs": self._recursive_mask(getattr(step, "inputs", {})),
                            "outputs": self._recursive_mask(getattr(step, "outputs", {})),
                        }
                        trajectory.append(step_dict)

            feedback_inputs = {
                "program_code": self.module_code,
                "modules_defn": modules_defn,
                "program_inputs": json.dumps(self._recursive_mask(inputs), indent=2),
                "program_trajectory": json.dumps(trajectory, indent=2),
                "program_outputs": json.dumps(self._recursive_mask(outputs), indent=2),
                "reward_code": self.reward_fn_code,
                "target_threshold": self.threshold,
                "reward_value": reward,
                "module_names": [self.module.__class__.__name__],
            }

            # Generate advice using an LLM
            advisor = Predict(OfferFeedback)
            advice_result = await advisor(**feedback_inputs)

            if advice_result.success and advice_result.outputs.get("advice"):
                return advice_result.outputs["advice"]
            else:
                # Fallback advice
                return {
                    self.module.__class__.__name__: f"Current reward ({reward:.2f}) is below threshold ({self.threshold}). Consider being more precise and following instructions more carefully."
                }

        except Exception as e:
            # Fallback to generic advice if feedback generation fails
            return {
                self.module.__class__.__name__: f"Feedback generation failed: {str(e)}. Please improve output quality to exceed threshold of {self.threshold}."
            }

    async def forward(self, **inputs: Any) -> Prediction:
        """Execute refinement with multiple temperature attempts."""
        # Get base temperature from provider config if available
        from .config_utils import get_config_value
        from .providers import get_provider

        base_temp = 0.7
        try:
            provider = get_provider()
            base_temp = get_config_value(provider, "temperature", 0.7)
        except Exception:
            pass

        # Generate temperature sequence
        temperatures = self._generate_temperature_sequence(base_temp)

        best_prediction = None
        best_reward = float("-inf")
        advice = None
        current_fail_count = self.fail_count

        for idx, temperature in enumerate(temperatures):
            start_time = time.time()

            try:
                # Create a copy of the module for this attempt
                module_copy = copy.deepcopy(self.module)

                # Set temperature using the utility for consistent handling
                from .config_utils import set_hyperparameter

                set_hyperparameter(module_copy, "temperature", temperature)

                # Prepare inputs for this attempt
                attempt_inputs = inputs.copy()

                # Add advice as hint if we have it
                if advice and hasattr(module_copy.signature, "input_fields"):
                    # Check if we can add a hint field
                    hint_advice = advice.get(module_copy.__class__.__name__, "N/A")
                    if hint_advice != "N/A":
                        attempt_inputs["hint_"] = hint_advice

                        # Temporarily modify signature to include hint
                        if module_copy.signature and hasattr(module_copy.signature, "input_fields"):
                            # This is a simplified approach - in practice, signature modification
                            # would need more sophisticated handling
                            pass

                # Enable tracing for this attempt
                if hasattr(module_copy, "enable_tracing"):
                    module_copy.enable_tracing()

                # Execute the module
                prediction = await module_copy(**attempt_inputs)
                duration = time.time() - start_time

                # Get trace if available
                trace = []
                if hasattr(module_copy, "get_trace") and module_copy.get_trace():
                    trace = module_copy.get_trace().steps

                # Calculate reward
                try:
                    reward = self.reward_fn(inputs, prediction)
                except Exception as e:
                    # If reward calculation fails, assign very low reward
                    reward = float("-inf")
                    if prediction.success:
                        prediction.success = False
                        prediction.error = f"Reward calculation failed: {str(e)}"

                # Record this attempt
                attempt = RefinementAttempt(
                    attempt_number=idx + 1,
                    timestamp=start_time,
                    temperature=temperature,
                    inputs=attempt_inputs,
                    outputs=prediction.outputs if prediction.outputs else {},
                    reward=reward,
                    success=prediction.success,
                    duration=duration,
                    trace=trace,
                )
                self.history.add_attempt(attempt)

                # Update best if this is better
                if reward > best_reward:
                    best_reward = reward
                    best_prediction = prediction

                # Check if we've met the threshold
                if self.threshold is not None and reward >= self.threshold:
                    # Success! Add refinement metadata and return this prediction
                    prediction.metadata = prediction.metadata or {}
                    prediction.metadata.update(
                        {
                            "refinement_attempts": len(self.history.attempts),
                            "best_reward": reward,
                            "refinement_history": self.history.attempts,
                        }
                    )
                    return prediction

                # If this is our last attempt, don't generate more feedback
                if idx == len(temperatures) - 1:
                    break

                # Generate feedback for next attempt
                advice = await self._generate_feedback(
                    inputs, prediction.outputs if prediction.outputs else {}, reward, trace
                )

            except Exception as e:
                duration = time.time() - start_time

                # Record failed attempt
                attempt = RefinementAttempt(
                    attempt_number=idx + 1,
                    timestamp=start_time,
                    temperature=temperature,
                    inputs=inputs,
                    success=False,
                    duration=duration,
                )
                self.history.add_attempt(attempt)

                current_fail_count -= 1
                if current_fail_count <= 0:
                    # Too many failures, give up
                    return Prediction(
                        success=False,
                        error=f"Refine failed after {self.fail_count} failures. Last error: {str(e)}",
                        usage=Usage(),
                        outputs={},
                        metadata={
                            "refinement_attempts": self.history.attempts,
                            "best_reward": best_reward,
                        },
                    )

                # Generate feedback about the error
                advice = {
                    self.module.__class__.__name__: f"Previous attempt failed with error: {str(e)}. Please be more careful with input validation and error handling."
                }

        # Return the best prediction we found
        if best_prediction:
            # Add refinement metadata
            best_prediction.metadata = best_prediction.metadata or {}
            best_prediction.metadata.update(
                {
                    "refinement_attempts": len(self.history.attempts),
                    "best_reward": best_reward,
                    "refinement_history": self.history.attempts,
                }
            )
            return best_prediction
        else:
            # All attempts failed
            return Prediction(
                success=False,
                error=f"All {len(temperatures)} refinement attempts failed",
                usage=Usage(),
                outputs={},
                metadata={
                    "refinement_attempts": self.history.attempts,
                    "best_reward": best_reward,
                },
            )

    def reset_history(self) -> None:
        """Reset the refinement history."""
        self.history = RefinementHistory()

    def get_average_reward(self) -> float:
        """Get the average reward across all attempts."""
        if not self.history.attempts:
            return 0.0

        rewards = [attempt.reward for attempt in self.history.attempts if attempt.success]
        if not rewards:
            return 0.0

        return sum(rewards) / len(rewards)

    def get_improvement_rate(self) -> float:
        """Get the rate of improvement from first to best attempt."""
        if len(self.history.attempts) < 2:
            return 0.0

        first_reward = self.history.attempts[0].reward
        best_reward = self.history.best_reward

        if first_reward == 0:
            return float("inf") if best_reward > 0 else 0.0

        return (best_reward - first_reward) / abs(first_reward)


__all__ = [
    "Refine",
    "OfferFeedback",
    "RefinementAttempt",
    "RefinementHistory",
]
