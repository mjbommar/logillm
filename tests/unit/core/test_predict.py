"""Tests for the Predict module."""

import asyncio

import pytest

from logillm.core.demos import Demo
from logillm.core.predict import ChainOfThought, Predict
from logillm.core.providers import MockProvider, register_provider


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MockProvider(response_text="The capital of France is Paris.")
    register_provider(provider, "test_provider", set_default=True)
    return provider


@pytest.mark.asyncio
async def test_predict_basic(mock_provider):
    """Test basic prediction without demos."""
    # Create a predict module
    predict = Predict("question -> answer")

    # Make a prediction
    result = await predict(question="What is the capital of France?")

    # Check the result
    assert result.success
    assert "output" in result.outputs or "answer" in result.outputs
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.output_tokens > 0
    assert result.metadata["provider"] == "mock"
    assert result.metadata["demos_used"] == 0


@pytest.mark.asyncio
async def test_predict_with_demos(mock_provider):
    """Test prediction with few-shot demonstrations."""
    # Create a predict module
    predict = Predict("question -> answer")

    # Add some demos
    predict.add_demo(
        Demo(
            inputs={"question": "What is the capital of Germany?"},
            outputs={"answer": "Berlin"},
            score=1.0,
            source="manual",
        )
    )

    predict.add_demo(
        {
            "inputs": {"question": "What is the capital of Spain?"},
            "outputs": {"answer": "Madrid"},
        }
    )

    # Make a prediction
    result = await predict(question="What is the capital of France?")

    # Check the result
    assert result.success
    assert result.metadata["demos_used"] == 2
    assert len(predict.demo_manager) == 2


def test_predict_sync(mock_provider):
    """Test synchronous prediction."""
    # Create a predict module
    predict = Predict("question -> answer")

    # Make a synchronous prediction
    result = predict.call_sync(question="What is the capital of France?")

    # Check the result
    assert result.success
    assert "output" in result.outputs or "answer" in result.outputs


@pytest.mark.asyncio
async def test_chain_of_thought():
    """Test ChainOfThought module."""
    # Create a mock provider that returns properly formatted JSON
    provider = MockProvider(response_text='{"reasoning": "2 plus 2 equals 4", "answer": "4"}')
    register_provider(provider, "test_cot_provider", set_default=True)

    # Create a chain of thought module
    cot = ChainOfThought("question -> answer")

    # Make a prediction
    result = await cot(question="What is 2+2?")

    # Check the result
    assert result.success
    # Should have reasoning field (added by ChainOfThought)
    assert "reasoning" in result.outputs
    # Should have answer field (from original signature)
    assert "answer" in result.outputs


def test_demo_manager():
    """Test demo management functionality."""
    from logillm.core.demos import DemoManager

    manager = DemoManager(max_demos=3)

    # Add demos
    for i in range(5):
        manager.add(
            Demo(
                inputs={"x": i},
                outputs={"y": i * 2},
                score=float(i),
            )
        )

    # Should only keep top 3
    assert len(manager) == 3

    # Should be sorted by score
    best_demos = manager.get_best()
    assert best_demos[0].score == 4.0
    assert best_demos[1].score == 3.0
    assert best_demos[2].score == 2.0


def test_signature_validation():
    """Test signature validation in Predict."""
    predict = Predict("question: str -> answer: str")

    # Valid inputs should work
    predict.signature.validate_inputs(question="What is 2+2?")

    # Invalid inputs should raise
    with pytest.raises(Exception):
        predict.signature.validate_inputs()  # Missing required field


def test_predict_serialization():
    """Test Predict serialization."""
    # Create a predict module with demos
    predict = Predict("question -> answer")
    predict.add_demo(
        {
            "inputs": {"question": "What is 2+2?"},
            "outputs": {"answer": "4"},
        }
    )

    # Serialize
    data = predict.to_dict()
    assert "demos" in data
    assert len(data["demos"]) == 1

    # Deserialize
    restored = Predict.from_dict(data)
    assert len(restored.demo_manager) == 1
    assert restored.demo_manager.demos[0].inputs["question"] == "What is 2+2?"


if __name__ == "__main__":
    # Run a simple test
    provider = MockProvider(response_text="Test response")
    register_provider(provider, set_default=True)

    predict = Predict("input -> output")
    result = asyncio.run(predict(input="test"))
    print(f"Success: {result.success}")
    print(f"Output: {result.outputs}")
    print(f"Usage: {result.usage}")
