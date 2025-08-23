"""Retry module for LogiLLM - wraps modules with error-aware retry logic."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .modules import Module, Parameter
from .signatures import Signature
from .signatures.fields import InputField, OutputField
from .types import Configuration, Metadata, Prediction, Usage


class RetryStrategy(Enum):
    """Available retry strategies."""

    IMMEDIATE = "immediate"  # Retry immediately
    LINEAR = "linear"  # Linear backoff
    EXPONENTIAL = "exponential"  # Exponential backoff


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt_number: int
    timestamp: float
    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    success: bool = False
    duration: float = 0.0


@dataclass
class RetryHistory:
    """Complete history of retry attempts."""

    attempts: list[RetryAttempt] = field(default_factory=list)
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0

    def add_attempt(self, attempt: RetryAttempt) -> None:
        """Add an attempt to the history."""
        self.attempts.append(attempt)
        self.total_attempts += 1
        if attempt.success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1

    def get_last_errors(self, n: int = 3) -> list[str]:
        """Get the last N error messages."""
        errors = []
        for attempt in reversed(self.attempts):
            if attempt.error and len(errors) < n:
                errors.append(attempt.error)
        return list(reversed(errors))


class Retry(Module):
    """Module wrapper that adds retry logic with error feedback.

    This module wraps another module and automatically retries execution
    when failures occur, providing feedback about previous attempts to
    help the wrapped module learn from its mistakes.

    Key features:
    - Automatic signature transformation (adds past_{field} and feedback fields)
    - Configurable retry strategies (immediate, linear backoff, exponential backoff)
    - Error feedback generation
    - Retry history tracking
    - Support for custom retry conditions

    Example:
        ```python
        # Create a base module
        qa = Predict("question -> answer")

        # Wrap with retry logic
        retry_qa = Retry(qa, max_retries=3, strategy=RetryStrategy.EXPONENTIAL)

        # Use normally - retries happen automatically on failures
        result = await retry_qa(question="What is the capital of France?")
        ```
    """

    def __init__(
        self,
        module: Module,
        *,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        retry_condition: Callable[[Prediction], bool] | None = None,
        feedback_generator: Callable[[RetryHistory], str] | None = None,
        config: Configuration | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        """Initialize the Retry module.

        Args:
            module: The module to wrap with retry logic
            max_retries: Maximum number of retry attempts
            strategy: Retry strategy to use
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Multiplier for exponential backoff
            retry_condition: Custom condition to determine if retry is needed
            feedback_generator: Custom function to generate feedback from history
            config: Additional configuration
            metadata: Module metadata
        """
        # Store the original signature
        self.original_signature = module.signature

        # Create enhanced signature with retry fields
        enhanced_signature = (
            self._create_retry_signature(module.signature) if module.signature else None
        )

        # Initialize with the enhanced signature
        super().__init__(signature=enhanced_signature, config=config, metadata=metadata)

        self.module = module
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retry_condition = retry_condition or self._default_retry_condition
        self.feedback_generator = feedback_generator or self._default_feedback_generator

        # Track retry history
        self.history = RetryHistory()

        # Add parameters for optimization
        self.parameters["max_retries"] = Parameter(
            value=max_retries, learnable=True, metadata={"type": "retry_limit"}
        )
        self.parameters["strategy"] = Parameter(
            value=strategy, learnable=True, metadata={"type": "retry_strategy"}
        )
        self.parameters["base_delay"] = Parameter(
            value=base_delay, learnable=True, metadata={"type": "retry_timing"}
        )

    def _create_retry_signature(self, original_signature: Signature) -> type[Signature] | None:
        """Create enhanced signature with past outputs and feedback fields."""
        if not original_signature:
            return None

        # Start with a copy of the original signature
        fields_dict = {}

        # Add all original input fields - convert to InputField objects
        if hasattr(original_signature, "input_fields"):
            for name, field_info in original_signature.input_fields.items():
                # Convert FieldSpec to InputField for make_signature
                desc = getattr(field_info, "description", None) or f"Input field {name}"
                default = getattr(field_info, "default", None)
                fields_dict[name] = InputField(description=desc, default=default)

        # Add past_{field} inputs for each output field
        if hasattr(original_signature, "output_fields"):
            for name, field_info in original_signature.output_fields.items():
                past_name = f"past_{name}"

                # Get the original prefix and description
                if hasattr(field_info, "description"):
                    desc = f"Previous {name} attempt that had errors"
                else:
                    desc = f"Past {name} with errors"

                # Create the past field as an input field
                past_field = InputField(
                    description=desc,
                    default=None,  # Optional field
                )
                fields_dict[past_name] = past_field

        # Add feedback field
        feedback_field = InputField(
            description="Instructions and feedback based on previous failed attempts",
            default=None,  # Optional field
        )
        fields_dict["feedback"] = feedback_field

        # Add all original output fields - convert to OutputField objects
        if hasattr(original_signature, "output_fields"):
            for name, field_info in original_signature.output_fields.items():
                # Convert FieldSpec to OutputField for make_signature
                desc = getattr(field_info, "description", None) or f"Output field {name}"
                default = getattr(field_info, "default", None)
                fields_dict[name] = OutputField(description=desc, default=default)

        # Create new signature class dynamically
        from .signatures.factory import make_signature

        instructions = getattr(original_signature, "instructions", None)
        if not instructions and hasattr(original_signature, "__doc__"):
            instructions = original_signature.__doc__

        signature_name = (
            f"Retry{getattr(original_signature, '__name__', original_signature.__class__.__name__)}"
        )
        return make_signature(  # type: ignore[return-value]
            fields_dict,
            instructions=instructions,
            signature_name=signature_name,
        )

    def _default_retry_condition(self, prediction: Prediction) -> bool:
        """Default condition to determine if retry is needed."""
        return not prediction.success

    def _default_feedback_generator(self, history: RetryHistory) -> str:
        """Default feedback generator based on retry history."""
        if not history.attempts:
            return "This is your first attempt. Please be careful and accurate."

        last_errors = history.get_last_errors(3)
        if not last_errors:
            return "Previous attempts succeeded but may need improvement."

        feedback_parts = [
            f"You have made {history.failed_attempts} failed attempts so far.",
            "Previous errors encountered:",
        ]

        for i, error in enumerate(last_errors, 1):
            feedback_parts.append(f"{i}. {error}")

        feedback_parts.extend(
            [
                "",
                "Please learn from these errors and avoid making the same mistakes.",
                "Be more careful and double-check your work before responding.",
            ]
        )

        return "\n".join(feedback_parts)

    async def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        else:
            delay = self.base_delay

        return min(delay, self.max_delay)

    async def forward(self, **inputs: Any) -> Prediction:
        """Execute the module with retry logic."""
        # Extract the original inputs (without past fields and feedback)
        original_inputs = {}
        past_outputs = {}
        feedback = None

        # Separate original inputs from retry-specific fields
        for key, value in inputs.items():
            if key.startswith("past_"):
                past_outputs[key] = value
            elif key == "feedback":
                feedback = value
            else:
                # This is an original input field
                original_inputs[key] = value

        last_prediction = None
        module_inputs = original_inputs.copy()

        for attempt in range(
            1, self.max_retries + 2
        ):  # +1 for initial attempt, +1 for inclusive range
            start_time = time.time()

            try:
                # Call the wrapped module
                prediction = await self.module(**module_inputs)
                duration = time.time() - start_time

                # Record this attempt
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    timestamp=start_time,
                    inputs=module_inputs,
                    outputs=prediction.outputs if prediction.outputs else {},
                    success=prediction.success,
                    duration=duration,
                )
                self.history.add_attempt(attempt_record)

                # Check if we should retry
                if not self.retry_condition(prediction):
                    # Success! Return the prediction
                    return prediction

                last_prediction = prediction

                # If this was our last attempt, break
                if attempt > self.max_retries:
                    break

                # Prepare inputs for next attempt with feedback and past outputs
                if self.original_signature and hasattr(self.original_signature, "output_fields"):
                    # Add past outputs for the next attempt
                    module_inputs = original_inputs.copy()

                    # Add past outputs if available
                    if prediction.outputs:
                        for field_name in self.original_signature.output_fields:
                            if field_name in prediction.outputs:
                                past_field_name = f"past_{field_name}"
                                module_inputs[past_field_name] = prediction.outputs[field_name]

                    # Generate and add feedback
                    feedback = self.feedback_generator(self.history)
                    if feedback:
                        module_inputs["feedback"] = feedback

                # Wait before retry
                delay = await self._calculate_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

            except Exception as e:
                duration = time.time() - start_time

                # Record failed attempt
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    timestamp=start_time,
                    inputs=module_inputs,
                    error=str(e),
                    success=False,
                    duration=duration,
                )
                self.history.add_attempt(attempt_record)

                last_prediction = Prediction(success=False, error=str(e), usage=Usage(), outputs={})

                # If this was our last attempt, break
                if attempt > self.max_retries:
                    break

                # Prepare inputs for next attempt with feedback about the error
                if self.original_signature:
                    module_inputs = original_inputs.copy()

                    # Generate and add feedback about the error
                    feedback = self.feedback_generator(self.history)
                    if feedback:
                        module_inputs["feedback"] = feedback

                # Wait before retry
                delay = await self._calculate_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        # All retries exhausted - return the last prediction with failure info
        if last_prediction:
            # Enhance the error message with retry info
            error_msg = f"All {self.max_retries + 1} attempts failed. "
            if last_prediction.error:
                error_msg += f"Last error: {last_prediction.error}"
            else:
                error_msg += "Module returned unsuccessful predictions."

            return Prediction(
                success=False,
                error=error_msg,
                usage=last_prediction.usage if last_prediction.usage else Usage(),
                outputs=last_prediction.outputs if last_prediction.outputs else {},
                metadata={
                    "retry_attempts": self.history.total_attempts,
                    "retry_history": self.history.attempts,
                },
            )
        else:
            return Prediction(
                success=False,
                error="Retry module failed to execute",
                usage=Usage(),
                outputs={},
            )

    def reset_history(self) -> None:
        """Reset the retry history."""
        self.history = RetryHistory()

    def get_success_rate(self) -> float:
        """Get the overall success rate of retry attempts."""
        if self.history.total_attempts == 0:
            return 0.0
        return self.history.successful_attempts / self.history.total_attempts

    def get_average_attempts(self) -> float:
        """Get the average number of attempts needed for success."""
        if self.history.successful_attempts == 0:
            return 0.0

        successful_sessions = []
        current_session = []

        for attempt in self.history.attempts:
            current_session.append(attempt)
            if attempt.success:
                successful_sessions.append(len(current_session))
                current_session = []
            elif attempt.attempt_number == 1 and current_session:
                # New session started, reset
                current_session = [attempt]

        if successful_sessions:
            return sum(successful_sessions) / len(successful_sessions)
        return 0.0


__all__ = [
    "Retry",
    "RetryStrategy",
    "RetryAttempt",
    "RetryHistory",
]
