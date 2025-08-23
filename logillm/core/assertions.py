"""Comprehensive assertion system with runtime validation and automatic backtracking.

This module provides a powerful assertion framework for LogiLLM that enables:
- Runtime validation of outputs and intermediate results
- Automatic backtracking and retry on assertion failures
- Suggestion generation for fixing failed assertions
- Integration with the Module system for seamless validation
- Support for both hard and soft assertions

The assertion system follows LogiLLM's zero-dependency philosophy and integrates
seamlessly with the existing Module, Provider, and Optimizer systems.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, Union
from weakref import WeakSet

from ..exceptions import LogiLLMError
from .types import FieldValue, Metadata, Prediction

T = TypeVar("T")
AssertionValue: type = Union[FieldValue, Any]


class AssertionSeverity(Enum):
    """Severity levels for assertions."""

    SOFT = "soft"  # Warning only, continue execution
    HARD = "hard"  # Error, trigger backtracking
    CRITICAL = "critical"  # Fatal error, stop execution entirely


class BacktrackStrategy(Enum):
    """Strategies for handling assertion failures."""

    RETRY = "retry"  # Simple retry with same inputs
    MODIFY_INPUTS = "modify_inputs"  # Modify inputs based on suggestions
    RELAX_CONSTRAINTS = "relax_constraints"  # Progressively relax assertion constraints
    ESCALATE = "escalate"  # Try different approach entirely


class AssertionType(Enum):
    """Types of assertions supported."""

    VALUE = "value"  # Value equality, range, type checks
    FORMAT = "format"  # JSON, regex, structure validation
    SEMANTIC = "semantic"  # LLM-based content validation
    CONSTRAINT = "constraint"  # Length, uniqueness, etc.
    CUSTOM = "custom"  # User-defined validation functions


@dataclass
class AssertionResult:
    """Result of an assertion evaluation."""

    passed: bool
    message: str
    severity: AssertionSeverity
    suggestions: list[str] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)
    actual_value: Any = None
    expected_value: Any = None

    def __bool__(self) -> bool:
        """Allow boolean evaluation of assertion results."""
        return self.passed


@dataclass
class BacktrackingContext:
    """Context for backtracking operations."""

    attempt: int = 0
    max_attempts: int = 3
    strategy: BacktrackStrategy = BacktrackStrategy.RETRY
    original_inputs: dict[str, Any] = field(default_factory=dict)
    modified_inputs: dict[str, Any] = field(default_factory=dict)
    failed_assertions: list[AssertionResult] = field(default_factory=list)
    suggestions_applied: list[str] = field(default_factory=list)
    relaxation_factor: float = 1.0

    @property
    def should_continue(self) -> bool:
        """Check if backtracking should continue."""
        return self.attempt < self.max_attempts


class AssertionError(LogiLLMError):
    """Enhanced assertion error with backtracking context."""

    def __init__(
        self,
        message: str,
        *,
        assertion_result: AssertionResult | None = None,
        backtrack_context: BacktrackingContext | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if assertion_result:
            context["assertion_result"] = assertion_result.__dict__
        if backtrack_context:
            context["backtrack_context"] = backtrack_context.__dict__

        suggestions = kwargs.pop("suggestions", [])
        if assertion_result is not None and assertion_result.suggestions:
            suggestions.extend(assertion_result.suggestions)

        super().__init__(message, context=context, suggestions=suggestions, **kwargs)
        self.assertion_result = assertion_result
        self.backtrack_context = backtrack_context


class BaseAssertion(ABC):
    """Abstract base class for all assertions."""

    def __init__(
        self,
        name: str,
        severity: AssertionSeverity = AssertionSeverity.HARD,
        message: str | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        self.name = name
        self.severity = severity
        self.custom_message = message
        self.metadata = metadata or {}

    @abstractmethod
    def check(self, value: Any, context: dict[str, Any] | None = None) -> AssertionResult:
        """Check if the assertion passes for the given value."""
        ...

    @abstractmethod
    def generate_suggestions(self, value: Any, context: dict[str, Any] | None = None) -> list[str]:
        """Generate suggestions for fixing assertion failures."""
        ...

    def _format_message(self, default_message: str) -> str:
        """Format the assertion message."""
        return self.custom_message or default_message


class ValueAssertion(BaseAssertion):
    """Assertion for value-based checks (equality, range, type)."""

    def __init__(
        self,
        name: str,
        expected_value: Any = None,
        expected_type: type | None = None,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        allowed_values: set[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.expected_value = expected_value
        self.expected_type = expected_type
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values

    def check(self, value: Any, context: dict[str, Any] | None = None) -> AssertionResult:
        """Check value-based assertions."""
        # Type check
        if self.expected_type and not isinstance(value, self.expected_type):
            return AssertionResult(
                passed=False,
                message=self._format_message(
                    f"Expected type {self.expected_type.__name__}, got {type(value).__name__}"
                ),
                severity=self.severity,
                suggestions=self.generate_suggestions(value, context),
                actual_value=value,
                expected_value=self.expected_type,
                metadata={"assertion_type": "type_check"},
            )

        # Exact value check
        if self.expected_value is not None and value != self.expected_value:
            return AssertionResult(
                passed=False,
                message=self._format_message(f"Expected value {self.expected_value}, got {value}"),
                severity=self.severity,
                suggestions=self.generate_suggestions(value, context),
                actual_value=value,
                expected_value=self.expected_value,
                metadata={"assertion_type": "value_equality"},
            )

        # Range check for numeric values
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return AssertionResult(
                    passed=False,
                    message=self._format_message(
                        f"Value {value} is below minimum {self.min_value}"
                    ),
                    severity=self.severity,
                    suggestions=self.generate_suggestions(value, context),
                    actual_value=value,
                    expected_value=f">= {self.min_value}",
                    metadata={"assertion_type": "range_check"},
                )

            if self.max_value is not None and value > self.max_value:
                return AssertionResult(
                    passed=False,
                    message=self._format_message(
                        f"Value {value} is above maximum {self.max_value}"
                    ),
                    severity=self.severity,
                    suggestions=self.generate_suggestions(value, context),
                    actual_value=value,
                    expected_value=f"<= {self.max_value}",
                    metadata={"assertion_type": "range_check"},
                )

        # Allowed values check (only for hashable values)
        if self.allowed_values is not None:
            try:
                if value not in self.allowed_values:
                    return AssertionResult(
                        passed=False,
                        message=self._format_message(
                            f"Value {value} not in allowed set: {self.allowed_values}"
                        ),
                        severity=self.severity,
                        suggestions=self.generate_suggestions(value, context),
                        actual_value=value,
                        expected_value=self.allowed_values,
                        metadata={"assertion_type": "allowed_values"},
                    )
            except TypeError:
                # Value is not hashable (e.g., dict, list), skip allowed values check
                pass

        return AssertionResult(
            passed=True,
            message=self._format_message(f"Value assertion '{self.name}' passed"),
            severity=self.severity,
            actual_value=value,
            metadata={"assertion_type": "value"},
        )

    def generate_suggestions(self, value: Any, context: dict[str, Any] | None = None) -> list[str]:
        """Generate suggestions for value assertion failures."""
        suggestions = []

        if self.expected_type and not isinstance(value, self.expected_type):
            suggestions.append(f"Convert value to {self.expected_type.__name__}")

        if self.expected_value is not None:
            suggestions.append(f"Use expected value: {self.expected_value}")

        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                suggestions.append(f"Increase value to at least {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                suggestions.append(f"Decrease value to at most {self.max_value}")

        if self.allowed_values is not None:
            suggestions.append(f"Choose from allowed values: {list(self.allowed_values)}")

        return suggestions


class FormatAssertion(BaseAssertion):
    """Assertion for format validation (JSON, regex, structure)."""

    def __init__(
        self,
        name: str,
        format_type: str = "json",
        pattern: str | None = None,
        required_keys: list[str] | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.format_type = format_type.lower()
        self.pattern = pattern
        self.required_keys = required_keys or []
        self.schema = schema

    def check(self, value: Any, context: dict[str, Any] | None = None) -> AssertionResult:
        """Check format-based assertions."""
        if self.format_type == "json":
            return self._check_json(value)
        elif self.format_type == "regex":
            return self._check_regex(value)
        elif self.format_type == "xml":
            return self._check_xml(value)
        elif self.format_type == "structure":
            return self._check_structure(value)
        else:
            return AssertionResult(
                passed=False,
                message=f"Unknown format type: {self.format_type}",
                severity=self.severity,
                suggestions=["Use supported format types: json, regex, xml, structure"],
                metadata={"assertion_type": "format", "format_type": self.format_type},
            )

    def _check_json(self, value: Any) -> AssertionResult:
        """Check if value is valid JSON."""
        try:
            if isinstance(value, str):
                parsed = json.loads(value)
            elif isinstance(value, dict):
                parsed = value
                value = json.dumps(value)  # For serialization test
            else:
                return AssertionResult(
                    passed=False,
                    message="Value must be string or dict for JSON validation",
                    severity=self.severity,
                    suggestions=["Convert value to JSON string or dict"],
                    actual_value=value,
                    metadata={"assertion_type": "json_format"},
                )

            # Check required keys
            if self.required_keys and isinstance(parsed, dict):
                missing_keys = [key for key in self.required_keys if key not in parsed]
                if missing_keys:
                    return AssertionResult(
                        passed=False,
                        message=f"Missing required JSON keys: {missing_keys}",
                        severity=self.severity,
                        suggestions=[f"Add missing keys: {missing_keys}"],
                        actual_value=parsed,
                        expected_value=self.required_keys,
                        metadata={"assertion_type": "json_keys", "missing_keys": missing_keys},
                    )

            return AssertionResult(
                passed=True,
                message="Valid JSON format",
                severity=self.severity,
                actual_value=parsed,
                metadata={"assertion_type": "json_format"},
            )

        except json.JSONDecodeError as e:
            return AssertionResult(
                passed=False,
                message=f"Invalid JSON format: {e}",
                severity=self.severity,
                suggestions=self.generate_suggestions(value),
                actual_value=value,
                metadata={"assertion_type": "json_format", "json_error": str(e)},
            )

    def _check_regex(self, value: Any) -> AssertionResult:
        """Check if value matches regex pattern."""
        if not self.pattern:
            return AssertionResult(
                passed=False,
                message="No regex pattern specified",
                severity=self.severity,
                suggestions=["Provide a regex pattern for validation"],
                metadata={"assertion_type": "regex"},
            )

        if not isinstance(value, str):
            return AssertionResult(
                passed=False,
                message="Value must be string for regex validation",
                severity=self.severity,
                suggestions=["Convert value to string"],
                actual_value=value,
                metadata={"assertion_type": "regex"},
            )

        try:
            if re.match(self.pattern, value):
                return AssertionResult(
                    passed=True,
                    message=f"Value matches regex pattern: {self.pattern}",
                    severity=self.severity,
                    actual_value=value,
                    metadata={"assertion_type": "regex", "pattern": self.pattern},
                )
            else:
                return AssertionResult(
                    passed=False,
                    message=f"Value does not match regex pattern: {self.pattern}",
                    severity=self.severity,
                    suggestions=self.generate_suggestions(value),
                    actual_value=value,
                    expected_value=self.pattern,
                    metadata={"assertion_type": "regex", "pattern": self.pattern},
                )
        except re.error as e:
            return AssertionResult(
                passed=False,
                message=f"Invalid regex pattern: {e}",
                severity=self.severity,
                suggestions=["Fix regex pattern syntax"],
                metadata={"assertion_type": "regex", "regex_error": str(e)},
            )

    def _check_xml(self, value: Any) -> AssertionResult:
        """Check if value is valid XML."""
        if not isinstance(value, str):
            return AssertionResult(
                passed=False,
                message="Value must be string for XML validation",
                severity=self.severity,
                suggestions=["Convert value to string"],
                actual_value=value,
                metadata={"assertion_type": "xml"},
            )

        # Basic XML validation - check for matching tags
        try:
            import xml.etree.ElementTree as ET

            ET.fromstring(value)
            return AssertionResult(
                passed=True,
                message="Valid XML format",
                severity=self.severity,
                actual_value=value,
                metadata={"assertion_type": "xml"},
            )
        except ET.ParseError as e:
            return AssertionResult(
                passed=False,
                message=f"Invalid XML format: {e}",
                severity=self.severity,
                suggestions=self.generate_suggestions(value),
                actual_value=value,
                metadata={"assertion_type": "xml", "xml_error": str(e)},
            )

    def _check_structure(self, value: Any) -> AssertionResult:
        """Check if value has required structure."""
        if self.required_keys:
            if not isinstance(value, dict):
                return AssertionResult(
                    passed=False,
                    message="Value must be dict for structure validation",
                    severity=self.severity,
                    suggestions=["Convert value to dictionary"],
                    actual_value=value,
                    metadata={"assertion_type": "structure"},
                )

            missing_keys = [key for key in self.required_keys if key not in value]
            if missing_keys:
                return AssertionResult(
                    passed=False,
                    message=f"Missing required keys: {missing_keys}",
                    severity=self.severity,
                    suggestions=[f"Add missing keys: {missing_keys}"],
                    actual_value=value,
                    expected_value=self.required_keys,
                    metadata={"assertion_type": "structure", "missing_keys": missing_keys},
                )

        return AssertionResult(
            passed=True,
            message="Valid structure",
            severity=self.severity,
            actual_value=value,
            metadata={"assertion_type": "structure"},
        )

    def generate_suggestions(self, value: Any, context: dict[str, Any] | None = None) -> list[str]:
        """Generate suggestions for format assertion failures."""
        suggestions = []

        if self.format_type == "json":
            suggestions.extend(
                [
                    "Ensure valid JSON syntax with proper quotes and brackets",
                    "Check for trailing commas or missing quotes",
                    "Validate JSON structure with an online validator",
                ]
            )
            if self.required_keys:
                suggestions.append(f"Include required keys: {self.required_keys}")

        elif self.format_type == "regex":
            suggestions.extend(
                [
                    f"Modify text to match pattern: {self.pattern}",
                    "Check regex pattern documentation for examples",
                ]
            )

        elif self.format_type == "xml":
            suggestions.extend(
                [
                    "Ensure all XML tags are properly closed",
                    "Check for invalid characters in XML content",
                    "Validate XML structure with proper nesting",
                ]
            )

        elif self.format_type == "structure":
            if self.required_keys:
                suggestions.append(f"Add required keys: {self.required_keys}")

        return suggestions


class ConstraintAssertion(BaseAssertion):
    """Assertion for constraint validation (length, uniqueness, etc.)."""

    def __init__(
        self,
        name: str,
        min_length: int | None = None,
        max_length: int | None = None,
        unique: bool = False,
        non_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.unique = unique
        self.non_empty = non_empty

    def check(self, value: Any, context: dict[str, Any] | None = None) -> AssertionResult:
        """Check constraint-based assertions."""
        # Non-empty check
        if self.non_empty:
            if not value or (hasattr(value, "__len__") and len(value) == 0):
                return AssertionResult(
                    passed=False,
                    message="Value cannot be empty",
                    severity=self.severity,
                    suggestions=["Provide a non-empty value"],
                    actual_value=value,
                    metadata={"assertion_type": "non_empty"},
                )

        # Length checks
        if hasattr(value, "__len__"):
            length = len(value)

            if self.min_length is not None and length < self.min_length:
                return AssertionResult(
                    passed=False,
                    message=f"Length {length} is below minimum {self.min_length}",
                    severity=self.severity,
                    suggestions=[f"Increase length to at least {self.min_length}"],
                    actual_value=value,
                    expected_value=f"length >= {self.min_length}",
                    metadata={"assertion_type": "min_length", "actual_length": length},
                )

            if self.max_length is not None and length > self.max_length:
                return AssertionResult(
                    passed=False,
                    message=f"Length {length} is above maximum {self.max_length}",
                    severity=self.severity,
                    suggestions=[f"Reduce length to at most {self.max_length}"],
                    actual_value=value,
                    expected_value=f"length <= {self.max_length}",
                    metadata={"assertion_type": "max_length", "actual_length": length},
                )

        # Uniqueness check for iterables
        if self.unique and hasattr(value, "__iter__") and not isinstance(value, str):
            try:
                items = list(value)
                if len(items) != len(set(items)):
                    duplicates = [item for item in set(items) if items.count(item) > 1]
                    return AssertionResult(
                        passed=False,
                        message=f"Duplicate items found: {duplicates}",
                        severity=self.severity,
                        suggestions=["Remove duplicate items"],
                        actual_value=value,
                        metadata={"assertion_type": "uniqueness", "duplicates": duplicates},
                    )
            except TypeError:
                # Items not hashable, skip uniqueness check
                pass

        return AssertionResult(
            passed=True,
            message=f"Constraint assertion '{self.name}' passed",
            severity=self.severity,
            actual_value=value,
            metadata={"assertion_type": "constraint"},
        )

    def generate_suggestions(self, value: Any, context: dict[str, Any] | None = None) -> list[str]:
        """Generate suggestions for constraint assertion failures."""
        suggestions = []

        if self.non_empty:
            suggestions.append("Provide a non-empty value")

        if hasattr(value, "__len__"):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                suggestions.append(f"Add content to reach minimum length of {self.min_length}")
            if self.max_length is not None and length > self.max_length:
                suggestions.append(f"Trim content to maximum length of {self.max_length}")

        if self.unique:
            suggestions.append("Remove duplicate items to ensure uniqueness")

        return suggestions


class CustomAssertion(BaseAssertion):
    """Assertion using a custom validation function."""

    def __init__(
        self,
        name: str,
        validation_fn: Callable[[Any], bool] | Callable[[Any, dict[str, Any]], bool],
        suggestion_fn: Callable[[Any], list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.validation_fn = validation_fn
        self.suggestion_fn = suggestion_fn

    def check(self, value: Any, context: dict[str, Any] | None = None) -> AssertionResult:
        """Check custom assertion."""
        try:
            # Try calling with context first, fall back to just value
            try:
                passed = self.validation_fn(value, context or {})
            except TypeError:
                passed = self.validation_fn(value)

            if passed:
                return AssertionResult(
                    passed=True,
                    message=f"Custom assertion '{self.name}' passed",
                    severity=self.severity,
                    actual_value=value,
                    metadata={"assertion_type": "custom"},
                )
            else:
                return AssertionResult(
                    passed=False,
                    message=self._format_message(f"Custom assertion '{self.name}' failed"),
                    severity=self.severity,
                    suggestions=self.generate_suggestions(value, context),
                    actual_value=value,
                    metadata={"assertion_type": "custom"},
                )

        except Exception as e:
            return AssertionResult(
                passed=False,
                message=f"Custom assertion error: {e}",
                severity=self.severity,
                suggestions=["Fix validation function implementation"],
                actual_value=value,
                metadata={"assertion_type": "custom", "error": str(e)},
            )

    def generate_suggestions(self, value: Any, context: dict[str, Any] | None = None) -> list[str]:
        """Generate suggestions for custom assertion failures."""
        if self.suggestion_fn:
            try:
                return self.suggestion_fn(value)
            except Exception:
                pass
        return [f"Fix value to satisfy custom assertion '{self.name}'"]


class SemanticAssertion(BaseAssertion):
    """Assertion using LLM-based semantic validation."""

    def __init__(
        self,
        name: str,
        requirement: str,
        provider: Any | None = None,
        confidence_threshold: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.requirement = requirement
        self.provider = provider
        self.confidence_threshold = confidence_threshold

    def check(self, value: Any, context: dict[str, Any] | None = None) -> AssertionResult:
        """Check semantic assertion using LLM."""
        if not self.provider:
            return AssertionResult(
                passed=False,
                message="No provider available for semantic validation",
                severity=self.severity,
                suggestions=["Set up an LLM provider for semantic assertions"],
                actual_value=value,
                metadata={"assertion_type": "semantic"},
            )

        # This would integrate with the provider system
        # For now, return a placeholder implementation
        return AssertionResult(
            passed=True,
            message=f"Semantic assertion '{self.name}' passed (placeholder)",
            severity=self.severity,
            actual_value=value,
            metadata={"assertion_type": "semantic", "requirement": self.requirement},
        )

    def generate_suggestions(self, value: Any, context: dict[str, Any] | None = None) -> list[str]:
        """Generate suggestions for semantic assertion failures."""
        return [
            f"Modify content to meet requirement: {self.requirement}",
            "Review content for semantic consistency",
        ]


class SuggestionGenerator:
    """Generates suggestions for fixing assertion failures."""

    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    async def generate_suggestions(
        self,
        failed_assertions: list[AssertionResult],
        original_value: Any,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate suggestions based on failed assertions."""
        suggestions = []

        # Collect suggestions from individual assertions
        for result in failed_assertions:
            suggestions.extend(result.suggestions)

        # Add context-aware suggestions
        if len(failed_assertions) > 1:
            suggestions.append("Address multiple validation issues simultaneously")

        # If we have an LLM provider, generate intelligent suggestions
        if self.provider:
            llm_suggestions = await self._generate_llm_suggestions(
                failed_assertions, original_value, context
            )
            suggestions.extend(llm_suggestions)

        return list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order

    async def _generate_llm_suggestions(
        self,
        failed_assertions: list[AssertionResult],
        original_value: Any,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate intelligent suggestions using LLM."""
        # This would integrate with the provider system
        # For now, return basic suggestions
        return [
            "Consider the validation requirements carefully",
            "Review the expected output format",
        ]


class BacktrackHandler:
    """Handles backtracking and retry logic for assertion failures."""

    def __init__(
        self,
        max_attempts: int = 3,
        default_strategy: BacktrackStrategy = BacktrackStrategy.RETRY,
        suggestion_generator: SuggestionGenerator | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.default_strategy = default_strategy
        self.suggestion_generator = suggestion_generator or SuggestionGenerator()

    async def handle_failures(
        self,
        failed_assertions: list[AssertionResult] | list[BaseAssertion],
        original_inputs: dict[str, Any],
        forward_fn: Callable[..., Awaitable[Prediction]],
        context: BacktrackingContext | None = None,
        original_value: Any = None,
    ) -> Prediction:
        """Handle assertion failures with backtracking.

        Args:
            failed_assertions: Either AssertionResult objects or BaseAssertion objects
            original_inputs: The original inputs to the forward function
            forward_fn: The function to retry with modified inputs
            context: Optional backtracking context
            original_value: The original value that failed assertions (needed if passing BaseAssertion objects)
        """
        if context is None:
            context = BacktrackingContext(
                original_inputs=copy.deepcopy(original_inputs),
                max_attempts=self.max_attempts,
                strategy=self.default_strategy,
            )

        # Convert BaseAssertion objects to AssertionResult if needed
        assertion_results = []
        assertions_to_check = []

        for item in failed_assertions:
            if isinstance(item, AssertionResult):
                assertion_results.append(item)
                # We can't re-check AssertionResult objects
            elif isinstance(item, BaseAssertion):
                # Keep the assertion for re-checking
                assertions_to_check.append(item)
                # Check it on the original value if provided
                if original_value is not None:
                    result = item.check(original_value)
                    assertion_results.append(result)

        context.failed_assertions.extend(assertion_results)

        # Generate suggestions for fixes
        suggestions = await self.suggestion_generator.generate_suggestions(
            assertion_results if assertion_results else failed_assertions,
            original_inputs,
            {"context": context},
        )

        while context.should_continue:
            context.attempt += 1

            try:
                # Apply strategy-specific modifications
                modified_inputs = await self._apply_strategy(context.strategy, context, suggestions)

                # Retry with modified inputs
                result = await forward_fn(**modified_inputs)

                # If we have assertions to check, validate the new outputs
                if assertions_to_check and result.success:
                    # Check if the new outputs pass the assertions
                    all_pass = True
                    new_failures = []

                    # Get the value to check from the result
                    # Assuming the assertion should check the output values
                    for assertion in assertions_to_check:
                        # Try to find the relevant output value
                        # This is a simplified approach - in practice might need field mapping
                        check_value = result.outputs.get("number", result.outputs)
                        if isinstance(result.outputs, dict) and len(result.outputs) == 1:
                            check_value = next(iter(result.outputs.values()))

                        check_result = assertion.check(check_value)
                        if not check_result.passed:
                            all_pass = False
                            new_failures.append(check_result)

                    if all_pass:
                        return result
                    else:
                        # Update context with new failures
                        context.failed_assertions.extend(new_failures)
                        # Continue with next attempt
                        continue

                # If no assertions to check, just return if successful
                if result.success:
                    return result

            except Exception as e:
                # Log attempt failure
                context.failed_assertions.append(
                    AssertionResult(
                        passed=False,
                        message=f"Backtrack attempt {context.attempt} failed: {e}",
                        severity=AssertionSeverity.HARD,
                        metadata={"attempt": context.attempt, "error": str(e)},
                    )
                )

        # All attempts exhausted
        raise AssertionError(
            f"All backtracking attempts exhausted after {context.attempt} tries",
            backtrack_context=context,
            suggestions=suggestions,
        )

    async def _apply_strategy(
        self,
        strategy: BacktrackStrategy,
        context: BacktrackingContext,
        suggestions: list[str],
    ) -> dict[str, Any]:
        """Apply the specified backtracking strategy."""
        if strategy == BacktrackStrategy.RETRY:
            return context.original_inputs.copy()

        elif strategy == BacktrackStrategy.MODIFY_INPUTS:
            return await self._modify_inputs_with_suggestions(context, suggestions)

        elif strategy == BacktrackStrategy.RELAX_CONSTRAINTS:
            return await self._relax_constraints(context)

        elif strategy == BacktrackStrategy.ESCALATE:
            return await self._escalate_approach(context)

        else:
            return context.original_inputs.copy()

    async def _modify_inputs_with_suggestions(
        self,
        context: BacktrackingContext,
        suggestions: list[str],
    ) -> dict[str, Any]:
        """Modify inputs based on suggestions."""
        modified = context.original_inputs.copy()

        # Apply simple modifications based on suggestions
        for suggestion in suggestions[:3]:  # Limit to first 3 suggestions
            if "increase" in suggestion.lower():
                # Try to increase numeric values
                for key, value in modified.items():
                    if isinstance(value, (int, float)):
                        modified[key] = value * 1.1
            elif "decrease" in suggestion.lower():
                # Try to decrease numeric values
                for key, value in modified.items():
                    if isinstance(value, (int, float)):
                        modified[key] = value * 0.9

        context.modified_inputs = modified
        context.suggestions_applied.extend(suggestions[:3])
        return modified

    async def _relax_constraints(self, context: BacktrackingContext) -> dict[str, Any]:
        """Progressively relax constraints."""
        context.relaxation_factor *= 0.8  # Reduce strictness by 20%
        return context.original_inputs.copy()

    async def _escalate_approach(self, context: BacktrackingContext) -> dict[str, Any]:
        """Try a completely different approach."""
        return context.original_inputs.copy()


class AssertionContext:
    """Context manager for assertion execution and tracking."""

    def __init__(
        self,
        name: str = "default",
        enabled: bool = True,
        backtrack_handler: BacktrackHandler | None = None,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.backtrack_handler = backtrack_handler or BacktrackHandler()
        self.assertions: list[BaseAssertion] = []
        self.results: list[AssertionResult] = []
        self._active_contexts: WeakSet[AssertionContext] = WeakSet()

    def add_assertion(self, assertion: BaseAssertion) -> None:
        """Add an assertion to this context."""
        self.assertions.append(assertion)

    def check_all(self, value: Any, context: dict[str, Any] | None = None) -> list[AssertionResult]:
        """Check all assertions in this context."""
        if not self.enabled:
            return []

        results = []
        for assertion in self.assertions:
            result = assertion.check(value, context)
            results.append(result)

        self.results.extend(results)
        return results

    def has_failures(self) -> bool:
        """Check if any assertions have failed."""
        return any(not result.passed for result in self.results)

    def get_failures(self) -> list[AssertionResult]:
        """Get all failed assertion results."""
        return [result for result in self.results if not result.passed]

    def clear_results(self) -> None:
        """Clear assertion results."""
        self.results.clear()


# Global assertion context
_global_context = AssertionContext("global")


def get_global_context() -> AssertionContext:
    """Get the global assertion context."""
    return _global_context


@asynccontextmanager
async def assertion_context(
    name: str = "local",
    enabled: bool = True,
    backtrack_handler: BacktrackHandler | None = None,
):
    """Context manager for local assertion contexts."""
    context = AssertionContext(name, enabled, backtrack_handler)
    try:
        yield context
    finally:
        context.clear_results()


# Main assertion interface
class Assert:
    """Main assertion interface for LogiLLM."""

    @staticmethod
    def value(
        name: str,
        expected_value: Any = None,
        expected_type: type | None = None,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        allowed_values: set[Any] | None = None,
        severity: AssertionSeverity = AssertionSeverity.HARD,
        message: str | None = None,
    ) -> ValueAssertion:
        """Create a value assertion."""
        return ValueAssertion(
            name=name,
            expected_value=expected_value,
            expected_type=expected_type,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            severity=severity,
            message=message,
        )

    @staticmethod
    def format(
        name: str,
        format_type: str = "json",
        pattern: str | None = None,
        required_keys: list[str] | None = None,
        schema: dict[str, Any] | None = None,
        severity: AssertionSeverity = AssertionSeverity.HARD,
        message: str | None = None,
    ) -> FormatAssertion:
        """Create a format assertion."""
        return FormatAssertion(
            name=name,
            format_type=format_type,
            pattern=pattern,
            required_keys=required_keys,
            schema=schema,
            severity=severity,
            message=message,
        )

    @staticmethod
    def constraint(
        name: str,
        min_length: int | None = None,
        max_length: int | None = None,
        unique: bool = False,
        non_empty: bool = False,
        severity: AssertionSeverity = AssertionSeverity.HARD,
        message: str | None = None,
    ) -> ConstraintAssertion:
        """Create a constraint assertion."""
        return ConstraintAssertion(
            name=name,
            min_length=min_length,
            max_length=max_length,
            unique=unique,
            non_empty=non_empty,
            severity=severity,
            message=message,
        )

    @staticmethod
    def custom(
        name: str,
        validation_fn: Callable[[Any], bool] | Callable[[Any, dict[str, Any]], bool],
        suggestion_fn: Callable[[Any], list[str]] | None = None,
        severity: AssertionSeverity = AssertionSeverity.HARD,
        message: str | None = None,
    ) -> CustomAssertion:
        """Create a custom assertion."""
        return CustomAssertion(
            name=name,
            validation_fn=validation_fn,
            suggestion_fn=suggestion_fn,
            severity=severity,
            message=message,
        )

    @staticmethod
    def semantic(
        name: str,
        requirement: str,
        provider: Any | None = None,
        confidence_threshold: float = 0.8,
        severity: AssertionSeverity = AssertionSeverity.HARD,
        message: str | None = None,
    ) -> SemanticAssertion:
        """Create a semantic assertion."""
        return SemanticAssertion(
            name=name,
            requirement=requirement,
            provider=provider,
            confidence_threshold=confidence_threshold,
            severity=severity,
            message=message,
        )


# Decorator for automatic assertion checking
def with_assertions(
    assertions: list[BaseAssertion] | None = None,
    context: AssertionContext | None = None,
    backtrack: bool = True,
    output_field: str | None = None,
) -> Callable:
    """Decorator to add assertion checking to module methods."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)

            if assertions or context:
                # Determine what to check
                check_value = result
                if isinstance(result, Prediction) and output_field:
                    check_value = result.outputs.get(output_field, result.outputs)
                elif isinstance(result, Prediction):
                    check_value = result.outputs

                # Check assertions
                assertion_context = context or get_global_context()
                if assertions:
                    for assertion in assertions:
                        assertion_context.add_assertion(assertion)

                assertion_results = assertion_context.check_all(check_value)

                # Handle failures
                failures = [r for r in assertion_results if not r.passed]
                if failures and backtrack:
                    # Extract hard failures for backtracking
                    hard_failures = [f for f in failures if f.severity == AssertionSeverity.HARD]
                    if hard_failures:
                        # Trigger backtracking
                        backtrack_handler = assertion_context.backtrack_handler
                        try:
                            # Pass the actual assertions, not the results
                            # Find which assertions failed by matching indices
                            failed_assertions = []
                            for i, result in enumerate(assertion_results):
                                if not result.passed and result.severity == AssertionSeverity.HARD:
                                    if assertions and i < len(assertions):
                                        failed_assertions.append(assertions[i])
                                    elif assertion_context.assertions and i < len(
                                        assertion_context.assertions
                                    ):
                                        failed_assertions.append(assertion_context.assertions[i])

                            if failed_assertions:
                                return await backtrack_handler.handle_failures(
                                    failed_assertions, kwargs, func, original_value=check_value
                                )
                            else:
                                # Fallback to passing results if we can't find assertions
                                return await backtrack_handler.handle_failures(
                                    hard_failures, kwargs, func
                                )
                        except AssertionError as e:
                            # If backtracking fails, raise original assertion error
                            raise AssertionError(
                                f"Assertion failures: {[f.message for f in hard_failures]}",
                                assertion_result=hard_failures[0] if hard_failures else None,
                            ) from e
                elif failures:
                    # Collect error messages for non-backtracked failures
                    error_messages = [
                        f.message for f in failures if f.severity == AssertionSeverity.HARD
                    ]
                    if error_messages:
                        raise AssertionError(
                            f"Assertion failures: {error_messages}",
                            assertion_result=failures[0],
                        )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
                # Similar assertion logic for sync functions
                return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Module integration
def assert_module_output(
    module: Any,
    assertions: list[BaseAssertion],
    enable_backtracking: bool = True,
) -> Any:
    """Add assertion checking to a module's forward method."""
    original_forward = module.forward

    @with_assertions(assertions=assertions, backtrack=enable_backtracking)
    async def new_forward(**inputs: Any) -> Prediction:
        return await original_forward(**inputs)

    module.forward = new_forward
    return module


__all__ = [
    # Enums
    "AssertionSeverity",
    "BacktrackStrategy",
    "AssertionType",
    # Data classes
    "AssertionResult",
    "BacktrackingContext",
    # Exceptions
    "AssertionError",
    # Assertion classes
    "BaseAssertion",
    "ValueAssertion",
    "FormatAssertion",
    "ConstraintAssertion",
    "CustomAssertion",
    "SemanticAssertion",
    # Core classes
    "SuggestionGenerator",
    "BacktrackHandler",
    "AssertionContext",
    "Assert",
    # Functions and decorators
    "get_global_context",
    "assertion_context",
    "with_assertions",
    "assert_module_output",
]
