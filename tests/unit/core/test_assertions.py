"""Comprehensive unit tests for the assertion system."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from logillm.core.assertions import (
    Assert,
    AssertionContext,
    AssertionError,
    AssertionResult,
    AssertionSeverity,
    BacktrackHandler,
    BacktrackingContext,
    BacktrackStrategy,
    SuggestionGenerator,
    assert_module_output,
    assertion_context,
    get_global_context,
    with_assertions,
)
from logillm.core.types import Prediction


class TestAssertionResult:
    """Test AssertionResult data class."""

    def test_boolean_evaluation(self):
        """Test boolean evaluation of assertion results."""
        passed_result = AssertionResult(
            passed=True,
            message="Test passed",
            severity=AssertionSeverity.HARD,
        )
        failed_result = AssertionResult(
            passed=False,
            message="Test failed",
            severity=AssertionSeverity.HARD,
        )

        assert bool(passed_result) is True
        assert bool(failed_result) is False

    def test_result_creation(self):
        """Test creation of assertion results."""
        result = AssertionResult(
            passed=True,
            message="Test message",
            severity=AssertionSeverity.SOFT,
            suggestions=["suggestion1", "suggestion2"],
            metadata={"key": "value"},
            actual_value="actual",
            expected_value="expected",
        )

        assert result.passed is True
        assert result.message == "Test message"
        assert result.severity == AssertionSeverity.SOFT
        assert result.suggestions == ["suggestion1", "suggestion2"]
        assert result.metadata == {"key": "value"}
        assert result.actual_value == "actual"
        assert result.expected_value == "expected"


class TestValueAssertion:
    """Test ValueAssertion class."""

    def test_type_check_success(self):
        """Test successful type checking."""
        assertion = Assert.value("test", expected_type=str)
        result = assertion.check("hello")

        assert result.passed is True
        assert "type assertion" in result.message.lower() or "passed" in result.message.lower()

    def test_type_check_failure(self):
        """Test failed type checking."""
        assertion = Assert.value("test", expected_type=str)
        result = assertion.check(42)

        assert result.passed is False
        assert "Expected type str, got int" in result.message
        assert "Convert value to str" in result.suggestions

    def test_value_equality_success(self):
        """Test successful value equality."""
        assertion = Assert.value("test", expected_value=42)
        result = assertion.check(42)

        assert result.passed is True

    def test_value_equality_failure(self):
        """Test failed value equality."""
        assertion = Assert.value("test", expected_value=42)
        result = assertion.check(24)

        assert result.passed is False
        assert "Expected value 42, got 24" in result.message
        assert "Use expected value: 42" in result.suggestions

    def test_range_check_success(self):
        """Test successful range checking."""
        assertion = Assert.value("test", min_value=0, max_value=100)
        result = assertion.check(50)

        assert result.passed is True

    def test_range_check_below_minimum(self):
        """Test range check below minimum."""
        assertion = Assert.value("test", min_value=10)
        result = assertion.check(5)

        assert result.passed is False
        assert "below minimum 10" in result.message
        assert "Increase value to at least 10" in result.suggestions

    def test_range_check_above_maximum(self):
        """Test range check above maximum."""
        assertion = Assert.value("test", max_value=10)
        result = assertion.check(15)

        assert result.passed is False
        assert "above maximum 10" in result.message
        assert "Decrease value to at most 10" in result.suggestions

    def test_allowed_values_success(self):
        """Test successful allowed values check."""
        assertion = Assert.value("test", allowed_values={"a", "b", "c"})
        result = assertion.check("b")

        assert result.passed is True

    def test_allowed_values_failure(self):
        """Test failed allowed values check."""
        assertion = Assert.value("test", allowed_values={"a", "b", "c"})
        result = assertion.check("d")

        assert result.passed is False
        assert "not in allowed set" in result.message
        assert "Choose from allowed values" in result.suggestions[0]


class TestFormatAssertion:
    """Test FormatAssertion class."""

    def test_json_format_success_string(self):
        """Test successful JSON format validation with string."""
        assertion = Assert.format("test", format_type="json")
        result = assertion.check('{"key": "value"}')

        assert result.passed is True
        assert result.actual_value == {"key": "value"}

    def test_json_format_success_dict(self):
        """Test successful JSON format validation with dict."""
        assertion = Assert.format("test", format_type="json")
        result = assertion.check({"key": "value"})

        assert result.passed is True

    def test_json_format_failure(self):
        """Test failed JSON format validation."""
        assertion = Assert.format("test", format_type="json")
        result = assertion.check('{"invalid": json}')

        assert result.passed is False
        assert "Invalid JSON format" in result.message
        assert "valid JSON syntax" in result.suggestions[0]

    def test_json_required_keys_success(self):
        """Test successful JSON required keys validation."""
        assertion = Assert.format("test", format_type="json", required_keys=["name", "age"])
        result = assertion.check({"name": "John", "age": 30, "city": "NYC"})

        assert result.passed is True

    def test_json_required_keys_failure(self):
        """Test failed JSON required keys validation."""
        assertion = Assert.format("test", format_type="json", required_keys=["name", "age"])
        result = assertion.check({"name": "John"})

        assert result.passed is False
        assert "Missing required JSON keys: ['age']" in result.message
        assert "Add missing keys: ['age']" in result.suggestions

    def test_regex_format_success(self):
        """Test successful regex format validation."""
        assertion = Assert.format("test", format_type="regex", pattern=r"^\d{3}-\d{2}-\d{4}$")
        result = assertion.check("123-45-6789")

        assert result.passed is True

    def test_regex_format_failure(self):
        """Test failed regex format validation."""
        assertion = Assert.format("test", format_type="regex", pattern=r"^\d{3}-\d{2}-\d{4}$")
        result = assertion.check("123-456-7890")

        assert result.passed is False
        assert "does not match regex pattern" in result.message

    def test_regex_invalid_pattern(self):
        """Test regex with invalid pattern."""
        assertion = Assert.format("test", format_type="regex", pattern=r"[invalid")
        result = assertion.check("test")

        assert result.passed is False
        assert "Invalid regex pattern" in result.message

    def test_xml_format_success(self):
        """Test successful XML format validation."""
        assertion = Assert.format("test", format_type="xml")
        result = assertion.check("<root><child>value</child></root>")

        assert result.passed is True

    def test_xml_format_failure(self):
        """Test failed XML format validation."""
        assertion = Assert.format("test", format_type="xml")
        result = assertion.check("<root><child>value</root>")  # Mismatched tags

        assert result.passed is False
        assert "Invalid XML format" in result.message

    def test_structure_format_success(self):
        """Test successful structure format validation."""
        assertion = Assert.format("test", format_type="structure", required_keys=["id", "name"])
        result = assertion.check({"id": 1, "name": "test", "extra": "data"})

        assert result.passed is True

    def test_structure_format_failure(self):
        """Test failed structure format validation."""
        assertion = Assert.format("test", format_type="structure", required_keys=["id", "name"])
        result = assertion.check({"id": 1})

        assert result.passed is False
        assert "Missing required keys: ['name']" in result.message

    def test_unknown_format_type(self):
        """Test unknown format type."""
        assertion = Assert.format("test", format_type="unknown")
        result = assertion.check("value")

        assert result.passed is False
        assert "Unknown format type: unknown" in result.message


class TestConstraintAssertion:
    """Test ConstraintAssertion class."""

    def test_non_empty_success(self):
        """Test successful non-empty validation."""
        assertion = Assert.constraint("test", non_empty=True)
        result = assertion.check("hello")

        assert result.passed is True

    def test_non_empty_failure_empty_string(self):
        """Test failed non-empty validation with empty string."""
        assertion = Assert.constraint("test", non_empty=True)
        result = assertion.check("")

        assert result.passed is False
        assert "cannot be empty" in result.message

    def test_non_empty_failure_empty_list(self):
        """Test failed non-empty validation with empty list."""
        assertion = Assert.constraint("test", non_empty=True)
        result = assertion.check([])

        assert result.passed is False
        assert "cannot be empty" in result.message

    def test_min_length_success(self):
        """Test successful minimum length validation."""
        assertion = Assert.constraint("test", min_length=3)
        result = assertion.check("hello")

        assert result.passed is True

    def test_min_length_failure(self):
        """Test failed minimum length validation."""
        assertion = Assert.constraint("test", min_length=5)
        result = assertion.check("hi")

        assert result.passed is False
        assert "below minimum 5" in result.message
        assert "Increase length to at least 5" in result.suggestions

    def test_max_length_success(self):
        """Test successful maximum length validation."""
        assertion = Assert.constraint("test", max_length=10)
        result = assertion.check("hello")

        assert result.passed is True

    def test_max_length_failure(self):
        """Test failed maximum length validation."""
        assertion = Assert.constraint("test", max_length=3)
        result = assertion.check("hello")

        assert result.passed is False
        assert "above maximum 3" in result.message
        assert "Reduce length to at most 3" in result.suggestions

    def test_uniqueness_success(self):
        """Test successful uniqueness validation."""
        assertion = Assert.constraint("test", unique=True)
        result = assertion.check([1, 2, 3, 4])

        assert result.passed is True

    def test_uniqueness_failure(self):
        """Test failed uniqueness validation."""
        assertion = Assert.constraint("test", unique=True)
        result = assertion.check([1, 2, 2, 3])

        assert result.passed is False
        assert "Duplicate items found: [2]" in result.message
        assert "Remove duplicate items" in result.suggestions


class TestCustomAssertion:
    """Test CustomAssertion class."""

    def test_custom_assertion_success(self):
        """Test successful custom assertion."""

        def is_even(value):
            return isinstance(value, int) and value % 2 == 0

        assertion = Assert.custom("even_check", is_even)
        result = assertion.check(4)

        assert result.passed is True

    def test_custom_assertion_failure(self):
        """Test failed custom assertion."""

        def is_even(value):
            return isinstance(value, int) and value % 2 == 0

        assertion = Assert.custom("even_check", is_even)
        result = assertion.check(3)

        assert result.passed is False
        assert "Custom assertion 'even_check' failed" in result.message

    def test_custom_assertion_with_context(self):
        """Test custom assertion with context parameter."""

        def context_aware_check(value, context):
            threshold = context.get("threshold", 10)
            return value > threshold

        assertion = Assert.custom("context_check", context_aware_check)
        result = assertion.check(15, {"threshold": 12})

        assert result.passed is True

    def test_custom_assertion_with_suggestions(self):
        """Test custom assertion with suggestion function."""

        def is_positive(value):
            return isinstance(value, (int, float)) and value > 0

        def suggest_positive(value):
            return ["Use a positive number", f"Try {abs(value)} instead"]

        assertion = Assert.custom("positive_check", is_positive, suggest_positive)
        result = assertion.check(-5)

        assert result.passed is False
        assert "Try 5 instead" in result.suggestions

    def test_custom_assertion_exception(self):
        """Test custom assertion that raises exception."""

        def failing_check(value):
            raise ValueError("Something went wrong")

        assertion = Assert.custom("failing_check", failing_check)
        result = assertion.check("anything")

        assert result.passed is False
        assert "Custom assertion error" in result.message


class TestSemanticAssertion:
    """Test SemanticAssertion class."""

    def test_semantic_assertion_no_provider(self):
        """Test semantic assertion without provider."""
        assertion = Assert.semantic("relevance", "Must be relevant to the topic")
        result = assertion.check("Some content")

        assert result.passed is False
        assert "No provider available" in result.message

    def test_semantic_assertion_with_provider(self):
        """Test semantic assertion with provider (placeholder)."""
        mock_provider = MagicMock()
        assertion = Assert.semantic("relevance", "Must be relevant", provider=mock_provider)
        result = assertion.check("Some content")

        # Currently returns placeholder success
        assert result.passed is True
        assert "placeholder" in result.message


class TestBacktrackingContext:
    """Test BacktrackingContext class."""

    def test_context_creation(self):
        """Test creation of backtracking context."""
        context = BacktrackingContext(
            max_attempts=5,
            strategy=BacktrackStrategy.MODIFY_INPUTS,
        )

        assert context.attempt == 0
        assert context.max_attempts == 5
        assert context.strategy == BacktrackStrategy.MODIFY_INPUTS
        assert context.should_continue is True

    def test_should_continue_logic(self):
        """Test should_continue property."""
        context = BacktrackingContext(max_attempts=3)

        assert context.should_continue is True

        context.attempt = 2
        assert context.should_continue is True

        context.attempt = 3
        assert context.should_continue is False


class TestSuggestionGenerator:
    """Test SuggestionGenerator class."""

    @pytest.mark.asyncio
    async def test_generate_suggestions_basic(self):
        """Test basic suggestion generation."""
        generator = SuggestionGenerator()

        failed_assertions = [
            AssertionResult(
                passed=False,
                message="Test failed",
                severity=AssertionSeverity.HARD,
                suggestions=["Fix the value", "Try again"],
            )
        ]

        suggestions = await generator.generate_suggestions(failed_assertions, "test_value")

        assert "Fix the value" in suggestions
        assert "Try again" in suggestions

    @pytest.mark.asyncio
    async def test_generate_suggestions_multiple_failures(self):
        """Test suggestion generation with multiple failures."""
        generator = SuggestionGenerator()

        failed_assertions = [
            AssertionResult(
                passed=False,
                message="Test 1 failed",
                severity=AssertionSeverity.HARD,
                suggestions=["Fix 1"],
            ),
            AssertionResult(
                passed=False,
                message="Test 2 failed",
                severity=AssertionSeverity.HARD,
                suggestions=["Fix 2"],
            ),
        ]

        suggestions = await generator.generate_suggestions(failed_assertions, "test_value")

        assert "Fix 1" in suggestions
        assert "Fix 2" in suggestions
        assert any("multiple" in s.lower() for s in suggestions)


class TestBacktrackHandler:
    """Test BacktrackHandler class."""

    def test_handler_creation(self):
        """Test creation of backtrack handler."""
        handler = BacktrackHandler(
            max_attempts=5,
            default_strategy=BacktrackStrategy.RELAX_CONSTRAINTS,
        )

        assert handler.max_attempts == 5
        assert handler.default_strategy == BacktrackStrategy.RELAX_CONSTRAINTS

    @pytest.mark.asyncio
    async def test_apply_retry_strategy(self):
        """Test retry strategy application."""
        handler = BacktrackHandler()
        context = BacktrackingContext(original_inputs={"key": "value"})

        result = await handler._apply_strategy(BacktrackStrategy.RETRY, context, [])

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_apply_modify_inputs_strategy(self):
        """Test modify inputs strategy."""
        handler = BacktrackHandler()
        context = BacktrackingContext(original_inputs={"number": 10})
        suggestions = ["increase the value"]

        result = await handler._apply_strategy(
            BacktrackStrategy.MODIFY_INPUTS, context, suggestions
        )

        # Should modify numeric values
        assert result["number"] != 10
        assert "increase the value" in context.suggestions_applied

    @pytest.mark.asyncio
    async def test_relax_constraints_strategy(self):
        """Test relax constraints strategy."""
        handler = BacktrackHandler()
        context = BacktrackingContext(
            original_inputs={"key": "value"},
            relaxation_factor=1.0,
        )

        await handler._apply_strategy(BacktrackStrategy.RELAX_CONSTRAINTS, context, [])

        assert context.relaxation_factor == 0.8


class TestAssertionContext:
    """Test AssertionContext class."""

    def test_context_creation(self):
        """Test creation of assertion context."""
        context = AssertionContext("test_context", enabled=False)

        assert context.name == "test_context"
        assert context.enabled is False
        assert len(context.assertions) == 0
        assert len(context.results) == 0

    def test_add_assertion(self):
        """Test adding assertions to context."""
        context = AssertionContext()
        assertion = Assert.value("test", expected_type=str)

        context.add_assertion(assertion)

        assert len(context.assertions) == 1
        assert context.assertions[0] == assertion

    def test_check_all_enabled(self):
        """Test checking all assertions when enabled."""
        context = AssertionContext(enabled=True)
        assertion1 = Assert.value("test1", expected_type=str)
        assertion2 = Assert.value("test2", expected_type=int)  # Both type checks

        context.add_assertion(assertion1)
        context.add_assertion(assertion2)

        results = context.check_all("hello")

        assert len(results) == 2
        assert results[0].passed is True  # String check passes
        assert results[1].passed is False  # Int check fails on string

    def test_check_all_disabled(self):
        """Test checking assertions when disabled."""
        context = AssertionContext(enabled=False)
        assertion = Assert.value("test", expected_type=str)
        context.add_assertion(assertion)

        results = context.check_all(42)

        assert len(results) == 0

    def test_has_failures(self):
        """Test failure detection."""
        context = AssertionContext()
        assertion = Assert.value("test", expected_type=str)
        context.add_assertion(assertion)

        context.check_all(42)  # This should fail

        assert context.has_failures() is True

    def test_get_failures(self):
        """Test getting failed results."""
        context = AssertionContext()
        assertion = Assert.value("test", expected_type=str)
        context.add_assertion(assertion)

        context.check_all(42)  # This should fail
        failures = context.get_failures()

        assert len(failures) == 1
        assert failures[0].passed is False

    def test_clear_results(self):
        """Test clearing results."""
        context = AssertionContext()
        assertion = Assert.value("test", expected_type=str)
        context.add_assertion(assertion)

        context.check_all("test")
        assert len(context.results) == 1

        context.clear_results()
        assert len(context.results) == 0


class TestGlobalContext:
    """Test global assertion context."""

    def test_get_global_context(self):
        """Test getting global context."""
        context = get_global_context()

        assert context.name == "global"
        assert isinstance(context, AssertionContext)


class TestAssertionContextManager:
    """Test assertion context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test assertion context manager."""
        async with assertion_context("test", enabled=True) as context:
            assert context.name == "test"
            assert context.enabled is True

            assertion = Assert.value("test", expected_type=str)
            context.add_assertion(assertion)

            results = context.check_all("hello")
            assert len(results) == 1
            assert results[0].passed is True

        # Context should be cleared after exit
        assert len(context.results) == 0


class TestWithAssertionsDecorator:
    """Test with_assertions decorator."""

    def setup_method(self):
        """Clear global context before each test."""
        from logillm.core.assertions import get_global_context

        get_global_context().assertions.clear()
        get_global_context().results.clear()

    @pytest.mark.asyncio
    async def test_decorator_with_passing_assertions(self):
        """Test decorator with passing assertions."""
        assertions = [Assert.value("test", expected_type=dict)]  # Check the dict type

        @with_assertions(assertions=assertions, backtrack=False)
        async def test_func():
            return Prediction(outputs={"result": "hello"})

        result = await test_func()

        assert result.outputs["result"] == "hello"

    @pytest.mark.asyncio
    async def test_decorator_with_failing_assertions_no_backtrack(self):
        """Test decorator with failing assertions without backtracking."""
        assertions = [Assert.value("test", expected_type=str)]

        @with_assertions(assertions=assertions, backtrack=False)
        async def test_func():
            return Prediction(outputs={"result": 42})

        with pytest.raises(AssertionError):
            await test_func()

    @pytest.mark.asyncio
    async def test_decorator_with_output_field(self):
        """Test decorator with specific output field."""
        assertions = [Assert.value("test", expected_type=str)]

        @with_assertions(assertions=assertions, output_field="result", backtrack=False)
        async def test_func():
            return Prediction(outputs={"result": "hello", "other": 42})

        result = await test_func()
        assert result.outputs["result"] == "hello"


class TestAssertModuleOutput:
    """Test assert_module_output function."""

    def setup_method(self):
        """Clear global context before each test."""
        from logillm.core.assertions import get_global_context

        get_global_context().assertions.clear()
        get_global_context().results.clear()

    @pytest.mark.asyncio
    async def test_assert_module_output(self):
        """Test adding assertions to module output."""
        # Create mock module
        module = MagicMock()
        module.forward = AsyncMock(return_value=Prediction(outputs={"result": "hello"}))

        assertions = [Assert.value("test", expected_type=dict)]  # Check the dict type

        # Add assertions to module
        modified_module = assert_module_output(module, assertions, enable_backtracking=False)

        # Should work with valid output
        result = await modified_module.forward(input="test")
        assert result.outputs["result"] == "hello"


class TestAssertionError:
    """Test AssertionError exception."""

    def test_assertion_error_creation(self):
        """Test creation of assertion error."""
        result = AssertionResult(
            passed=False,
            message="Test failed",
            severity=AssertionSeverity.HARD,
            suggestions=["Fix it"],
        )

        error = AssertionError(
            "Assertion failed",
            assertion_result=result,
        )

        assert "Assertion failed" in str(error)
        assert "Fix it" in error.suggestions
        assert error.assertion_result == result

    def test_assertion_error_with_context(self):
        """Test assertion error with backtracking context."""
        context = BacktrackingContext(attempt=2)

        error = AssertionError(
            "Backtracking failed",
            backtrack_context=context,
        )

        assert error.backtrack_context == context


if __name__ == "__main__":
    pytest.main([__file__])
