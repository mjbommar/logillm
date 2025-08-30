"""Integration tests for float field handling to prevent regression of the 0-1 range bug."""

import asyncio
import pytest
from typing import Optional

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers.mock import MockProvider


class NumberAddition(Signature):
    """Add numbers together."""

    numbers: str = InputField(desc="Numbers to add")
    result: float = OutputField(desc="Sum of the numbers")


class MathProblem(Signature):
    """Solve a math problem."""

    problem: str = InputField(desc="Math problem to solve")
    answer: float = OutputField(desc="Numerical answer")
    explanation: Optional[str] = OutputField(desc="Step by step explanation")


class PercentageCalculation(Signature):
    """Calculate a percentage."""

    base: str = InputField(desc="Base amount")
    percentage: str = InputField(desc="Percentage to calculate")
    result: float = OutputField(desc="Result of the percentage calculation")


@pytest.mark.asyncio
async def test_float_fields_not_treated_as_percentages():
    """Test that float fields are NOT interpreted as 0-1 range values."""
    # Test with various numeric responses
    test_cases = [
        ("3", 3.0),  # Should be 3.0, not 0.03
        ("10", 10.0),  # Should be 10.0, not 0.10
        ("50", 50.0),  # Should be 50.0, not 0.50
        ("100", 100.0),  # Should be 100.0, not 1.0
        ("150", 150.0),  # Should stay 150.0
        ("0.5", 0.5),  # Actual decimal should stay as-is
        ("1.5", 1.5),  # Should stay 1.5
    ]

    for response_text, expected_value in test_cases:
        provider = MockProvider(responses=[f"result: {response_text}"])
        predictor = Predict(NumberAddition, provider=provider)
        
        result = await predictor(numbers="test input")
        
        assert result.result == expected_value, (
            f"Expected {expected_value}, got {result.result} for response '{response_text}'"
        )


@pytest.mark.asyncio
async def test_large_numbers_preserved():
    """Test that large numbers are not scaled down."""
    provider = MockProvider(responses=["answer: 1000"])
    predictor = Predict(MathProblem, provider=provider)
    
    result = await predictor(problem="What is 500 + 500?")
    
    assert result.answer == 1000.0, f"Expected 1000.0, got {result.answer}"
    # Should NOT be 10.0 or 1.0


@pytest.mark.asyncio
async def test_decimal_numbers_preserved():
    """Test that decimal numbers are correctly preserved."""
    test_cases = [
        ("3.14159", 3.14159),  # Pi
        ("2.71828", 2.71828),  # e
        ("0.25", 0.25),  # Quarter
        ("0.001", 0.001),  # Small decimal
    ]
    
    for response_text, expected_value in test_cases:
        provider = MockProvider(responses=[f"result: {response_text}"])
        predictor = Predict(NumberAddition, provider=provider)
        
        result = await predictor(numbers="test")
        
        assert abs(result.result - expected_value) < 0.00001, (
            f"Expected {expected_value}, got {result.result}"
        )


@pytest.mark.asyncio
async def test_prompt_formatting_no_range_hints():
    """Test that prompts don't contain confusing 0-1 range hints."""
    from logillm.core.adapters.chat import ChatAdapter
    from logillm.core.format_adapters import ChatAdapter as FormatChatAdapter
    
    # Test both chat adapters
    adapters = [ChatAdapter(), FormatChatAdapter()]
    
    for adapter in adapters:
        # Format a prompt with float fields
        try:
            prompt = adapter.format_prompt({"numbers": "1, 2, 3"}, NumberAddition)
            
            # Convert to string if it's a list of messages
            if isinstance(prompt, list):
                prompt_text = str(prompt)
            else:
                prompt_text = prompt
            
            # Check that problematic phrases are NOT present
            assert "between 0.0 and 1.0" not in prompt_text, (
                f"Found 'between 0.0 and 1.0' in prompt from {adapter.__class__.__name__}"
            )
            assert "between 0 and 1" not in prompt_text, (
                f"Found 'between 0 and 1' in prompt from {adapter.__class__.__name__}"
            )
            
            # The word "decimal" alone is OK, but not with range constraints
            if "decimal" in prompt_text:
                # Make sure it's not followed by range constraints
                assert "decimal number between" not in prompt_text.lower()
                
        except AttributeError:
            # Skip if adapter doesn't have format_prompt
            continue


@pytest.mark.asyncio
async def test_percentage_fields_when_explicitly_needed():
    """Test that actual percentage calculations still work correctly."""
    # When we DO want a percentage, it should work
    provider = MockProvider(responses=["result: 25"])  # 25% of 100 = 25
    predictor = Predict(PercentageCalculation, provider=provider)
    
    result = await predictor(base="100", percentage="25")
    
    # 25% of 100 should be 25, not 0.25
    assert result.result == 25.0, f"Expected 25.0, got {result.result}"


@pytest.mark.asyncio
async def test_confidence_scores_still_work():
    """Test that actual confidence scores (0-1) still work when appropriate."""
    
    class ConfidenceScore(Signature):
        """Analyze with confidence."""
        text: str = InputField(desc="Text to analyze")
        confidence: float = OutputField(desc="Confidence score from 0 to 1")
    
    provider = MockProvider(responses=["confidence: 0.85"])
    predictor = Predict(ConfidenceScore, provider=provider)
    
    result = await predictor(text="test")
    
    # Actual confidence scores should work
    assert result.confidence == 0.85, f"Expected 0.85, got {result.confidence}"


@pytest.mark.asyncio
async def test_math_operations_produce_correct_scale():
    """Test that math operations produce correctly scaled results."""
    test_cases = [
        ("10 + 20", "30", 30.0),
        ("100 - 50", "50", 50.0),
        ("5 * 6", "30", 30.0),
        ("100 / 4", "25", 25.0),
        ("2^8", "256", 256.0),
    ]
    
    for problem, response, expected in test_cases:
        provider = MockProvider(responses=[f"answer: {response}"])
        predictor = Predict(MathProblem, provider=provider)
        
        result = await predictor(problem=problem)
        
        assert result.answer == expected, (
            f"For '{problem}': expected {expected}, got {result.answer}"
        )


@pytest.mark.asyncio
async def test_word_numbers_to_float():
    """Test that word numbers convert to correct float values."""
    # Mock responses that simulate what an LLM might return
    test_cases = [
        ("one plus two", "3", 3.0),
        ("five times ten", "50", 50.0),
        ("one hundred", "100", 100.0),
        ("negative ten", "-10", -10.0),
    ]
    
    for problem, response, expected in test_cases:
        provider = MockProvider(responses=[f"result: {response}"])
        predictor = Predict(NumberAddition, provider=provider)
        
        result = await predictor(numbers=problem)
        
        assert result.result == expected, (
            f"For '{problem}': expected {expected}, got {result.result}"
        )


@pytest.mark.asyncio 
async def test_edge_cases():
    """Test edge cases that previously failed."""
    # These are the exact cases that were failing before the fix
    edge_cases = [
        ("one, two", "3", 3.0),  # Was returning 0.003
        ("5, 10, 15", "30", 30.0),  # Was returning 0.3
        ("dog, bowl", "0", 0.0),  # No numbers case
    ]
    
    for input_text, response, expected in edge_cases:
        provider = MockProvider(responses=[f"result: {response}"])
        predictor = Predict(NumberAddition, provider=provider)
        
        result = await predictor(numbers=input_text)
        
        assert result.result == expected, (
            f"Edge case '{input_text}' failed: expected {expected}, got {result.result}"
        )


def test_sync_integration():
    """Synchronous test wrapper for integration tests."""
    asyncio.run(test_float_fields_not_treated_as_percentages())
    asyncio.run(test_large_numbers_preserved())
    asyncio.run(test_decimal_numbers_preserved())
    asyncio.run(test_prompt_formatting_no_range_hints())
    asyncio.run(test_edge_cases())
    print("✅ All float field integration tests passed!")


if __name__ == "__main__":
    test_sync_integration()