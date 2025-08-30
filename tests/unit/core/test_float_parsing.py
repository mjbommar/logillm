"""Test float parsing to ensure numbers aren't incorrectly treated as percentages."""

import asyncio
import pytest

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers.mock import MockProvider


class NumberSignature(Signature):
    """Test signature for number extraction."""

    input: str = InputField(desc="Input text")
    number: float = OutputField(desc="Output number")


@pytest.mark.asyncio
async def test_float_parsing_no_percentage_conversion():
    """Test that regular numbers aren't converted as percentages."""
    # Test small number (3) - should NOT become 0.03
    provider = MockProvider(responses=["number: 3"])
    predictor = Predict(NumberSignature, provider=provider)
    result = await predictor(input="test")
    assert result.number == 3.0, f"Expected 3.0, got {result.number}"

    # Test medium number (50) - should NOT become 0.5
    provider = MockProvider(responses=["number: 50"])
    predictor = Predict(NumberSignature, provider=provider)
    result = await predictor(input="test")
    assert result.number == 50.0, f"Expected 50.0, got {result.number}"

    # Test large number (150) - should stay as-is
    provider = MockProvider(responses=["number: 150"])
    predictor = Predict(NumberSignature, provider=provider)
    result = await predictor(input="test")
    assert result.number == 150.0, f"Expected 150.0, got {result.number}"

    # Test decimal (0.5) - should stay as-is
    provider = MockProvider(responses=["number: 0.5"])
    predictor = Predict(NumberSignature, provider=provider)
    result = await predictor(input="test")
    assert result.number == 0.5, f"Expected 0.5, got {result.number}"


@pytest.mark.asyncio
async def test_float_formatting_prompt():
    """Test that float fields don't force 0-1 range in prompts."""
    from logillm.core.format_adapters import ChatAdapter

    adapter = ChatAdapter()
    
    # Create a mock signature-like object
    class MockSignature:
        def __init__(self):
            self.output_fields = {
                "number": type('FieldSpec', (), {
                    'desc': 'output number',
                    'annotation': float
                })()
            }
            self.input_fields = {
                "input": type('FieldSpec', (), {
                    'desc': 'input text',
                    'annotation': str
                })()
            }
    
    signature = MockSignature()
    
    # Format a prompt
    prompt = adapter.format_prompt({"input": "test"}, signature)
    
    # Ensure it doesn't contain the problematic "between 0.0 and 1.0" text
    if isinstance(prompt, str):
        assert "between 0.0 and 1.0" not in prompt, "Float fields shouldn't force 0-1 range"
        # Should just say "decimal number" without range restriction
        assert "decimal number)" in prompt or "number" in prompt.lower()
    elif isinstance(prompt, list):
        # For message format
        content = str(prompt)
        assert "between 0.0 and 1.0" not in content, "Float fields shouldn't force 0-1 range"


def test_sync_wrapper():
    """Test synchronous wrapper for the async tests."""
    asyncio.run(test_float_parsing_no_percentage_conversion())
    asyncio.run(test_float_formatting_prompt())


if __name__ == "__main__":
    test_sync_wrapper()
    print("✅ All float parsing tests passed!")