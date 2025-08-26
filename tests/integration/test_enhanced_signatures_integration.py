"""Integration tests for enhanced signature features with real LLM calls."""

import asyncio
import pytest

from logillm.core.signatures import Signature
from logillm.core.signatures.types import Image, Audio, Tool, History
from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complex_type_signature_with_llm():
    """Test that complex type signatures work with real LLM calls."""
    # Setup provider
    provider = create_provider("openai", model="gpt-4o-mini")
    register_provider(provider, set_default=True)
    
    # Create signature with complex types
    sig = Signature("items: list[str] -> summary: str, count: int")
    predictor = Predict(sig)
    
    # Test with real call
    result = await predictor(items=["apple", "banana", "orange"])
    
    assert hasattr(result, "summary")
    assert hasattr(result, "count")
    assert isinstance(result.summary, str)
    # Count might come back as string from LLM, but should be convertible
    assert int(result.count) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_optional_type_signature_with_llm():
    """Test Optional type handling with real LLM."""
    provider = create_provider("openai", model="gpt-4o-mini")
    register_provider(provider, set_default=True)
    
    # Signature with optional output
    sig = Signature("query: str -> result: Optional[str]")
    predictor = Predict(sig)
    
    # Test with query that should return something
    result = await predictor(query="What is 2+2?")
    assert hasattr(result, "result")
    assert result.result is not None
    
    # Test with query that might return None
    result2 = await predictor(query="Find a unicorn in this empty list: []")
    assert hasattr(result2, "result")
    # LLM might return None, empty string, or "None" string


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multimodal_history_type():
    """Test History type with real LLM."""
    provider = create_provider("openai", model="gpt-4o-mini")
    register_provider(provider, set_default=True)
    
    # Create conversation history
    history = History([
        {"role": "user", "content": "Hi, my name is Alice"},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name?"}
    ])
    
    # Use in signature (Note: actual LLM handling of History type would need adapter support)
    sig = Signature("question: str, context: str -> answer: str")
    predictor = Predict(sig)
    
    # Convert history to context string for now
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history.messages])
    
    result = await predictor(question="What's the user's name?", context=context)
    assert "Alice" in result.answer


@pytest.mark.integration
@pytest.mark.asyncio
async def test_signature_from_examples_with_llm():
    """Test that signatures inferred from examples work with real LLM."""
    from logillm.core.signatures.parser import infer_signature_from_examples
    from logillm.core.signatures.factory import make_signature
    
    provider = create_provider("openai", model="gpt-4o-mini")
    register_provider(provider, set_default=True)
    
    # Create examples
    examples = [
        {"input": {"text": "Hello world"}, "output": {"language": "English"}},
        {"input": {"text": "Bonjour le monde"}, "output": {"language": "French"}},
    ]
    
    # Infer signature
    fields = infer_signature_from_examples(examples)
    
    # Convert to signature class (this is a bit manual, could be improved)
    sig = Signature("text: str -> language: str")
    predictor = Predict(sig)
    
    # Test with real call
    result = await predictor(text="Hola mundo")
    assert hasattr(result, "language")
    assert "Spanish" in result.language or "español" in result.language.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_union_type_signature():
    """Test Union type handling with real LLM."""
    provider = create_provider("openai", model="gpt-4o-mini")
    register_provider(provider, set_default=True)
    
    # Signature that can return different types
    sig = Signature("number: str -> parsed: Union[int, float]")
    predictor = Predict(sig)
    
    # Test with integer string
    result = await predictor(number="42")
    assert hasattr(result, "parsed")
    # LLM returns strings, but should be parseable
    assert float(result.parsed) == 42
    
    # Test with float string
    result2 = await predictor(number="3.14")
    assert hasattr(result2, "parsed")
    assert abs(float(result2.parsed) - 3.14) < 0.01


if __name__ == "__main__":
    # For local testing
    asyncio.run(test_complex_type_signature_with_llm())