"""Unit tests for debug mode functionality."""

import os

import pytest

from logillm.core.modules import Module
from logillm.core.predict import ChainOfThought, Predict
from logillm.core.types import Prediction
from logillm.providers.mock import MockProvider


# Test implementation of Module for testing
class _TestModule(Module):
    """Concrete Module for testing.

    Named with underscore prefix to prevent pytest from collecting it as a test class.
    """

    async def forward(self, **inputs):
        """Simple forward implementation."""
        return Prediction(outputs=inputs)


class TestDebugMode:
    """Test debug mode functionality in modules."""

    def test_module_debug_init_explicit(self):
        """Test explicit debug parameter in module initialization."""
        # Debug enabled
        module_debug = _TestModule(signature=None, debug=True)
        assert module_debug.is_debugging() is True

        # Debug disabled
        module_no_debug = _TestModule(signature=None, debug=False)
        assert module_no_debug.is_debugging() is False

        # Default (no debug parameter)
        module_default = _TestModule(signature=None)
        assert module_default.is_debugging() is False

    def test_module_debug_environment_variable(self):
        """Test debug mode from environment variable."""
        # Set environment variable
        os.environ["LOGILLM_DEBUG"] = "1"
        try:
            module = _TestModule(signature=None)
            assert module.is_debugging() is True
        finally:
            del os.environ["LOGILLM_DEBUG"]

        # Without environment variable
        module = _TestModule(signature=None)
        assert module.is_debugging() is False

    def test_module_debug_explicit_overrides_env(self):
        """Test that explicit debug parameter overrides environment variable."""
        os.environ["LOGILLM_DEBUG"] = "1"
        try:
            # Explicit False should override environment
            module = _TestModule(signature=None, debug=False)
            assert module.is_debugging() is False

            # Explicit True when env is set
            module = _TestModule(signature=None, debug=True)
            assert module.is_debugging() is True
        finally:
            del os.environ["LOGILLM_DEBUG"]

    def test_module_debug_methods(self):
        """Test enable/disable debug methods."""
        module = _TestModule(signature=None, debug=False)
        assert module.is_debugging() is False

        # Enable debug
        module.enable_debug_mode()
        assert module.is_debugging() is True

        # Disable debug
        module.disable_debug_mode()
        assert module.is_debugging() is False

    @pytest.mark.asyncio
    async def test_predict_debug_captures_prompt(self):
        """Test that Predict captures prompt when debug is enabled."""
        # Create mock provider
        mock_provider = MockProvider()

        # Create Predict with debug enabled
        predict = Predict("question -> answer", provider=mock_provider, debug=True)

        # Execute
        result = await predict(question="What is 2+2?")

        # Check that prompt was captured
        assert result.prompt is not None
        assert "messages" in result.prompt
        assert "adapter" in result.prompt
        assert "demos_count" in result.prompt
        assert "provider" in result.prompt
        assert "model" in result.prompt

        # Check content
        assert result.prompt["adapter"] == "chat"
        assert result.prompt["demos_count"] == 0
        assert result.prompt["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_predict_no_debug_no_prompt(self):
        """Test that Predict doesn't capture prompt when debug is disabled."""
        mock_provider = MockProvider()
        predict = Predict("question -> answer", provider=mock_provider, debug=False)

        result = await predict(question="What is 2+2?")

        # Prompt should not be captured
        assert result.prompt is None

    @pytest.mark.asyncio
    async def test_predict_debug_with_demos(self):
        """Test that debug mode correctly reports demo count."""
        mock_provider = MockProvider()
        predict = Predict("question -> answer", provider=mock_provider, debug=True)

        # Add demonstrations
        predict.add_demo({"inputs": {"question": "1+1"}, "outputs": {"answer": "2"}})
        predict.add_demo({"inputs": {"question": "2+2"}, "outputs": {"answer": "4"}})

        result = await predict(question="3+3")

        # Check demo count
        assert result.prompt is not None
        assert result.prompt["demos_count"] == 2

    @pytest.mark.asyncio
    async def test_predict_toggle_debug(self):
        """Test toggling debug mode on Predict."""
        mock_provider = MockProvider()
        predict = Predict("question -> answer", provider=mock_provider)

        # Initially no debug
        result1 = await predict(question="Test 1")
        assert result1.prompt is None

        # Enable debug
        predict.enable_debug_mode()
        result2 = await predict(question="Test 2")
        assert result2.prompt is not None

        # Disable debug
        predict.disable_debug_mode()
        result3 = await predict(question="Test 3")
        assert result3.prompt is None

    @pytest.mark.asyncio
    async def test_chain_of_thought_debug(self):
        """Test that ChainOfThought passes debug parameter correctly."""
        mock_provider = MockProvider()

        # ChainOfThought with debug
        cot = ChainOfThought("problem -> answer", provider=mock_provider, debug=True)
        result = await cot(problem="Test problem")

        # Should capture prompt
        assert result.prompt is not None
        assert "messages" in result.prompt

    def test_prediction_prompt_field(self):
        """Test that Prediction dataclass has prompt field."""
        # Create prediction without prompt
        pred1 = Prediction(outputs={"test": "value"})
        assert pred1.prompt is None

        # Create prediction with prompt
        prompt_data = {
            "messages": [{"role": "user", "content": "test"}],
            "adapter": "chat",
            "demos_count": 0,
        }
        pred2 = Prediction(outputs={"test": "value"}, prompt=prompt_data)
        assert pred2.prompt == prompt_data

        # Ensure prompt field doesn't interfere with output access
        assert pred2.test == "value"  # Should access outputs["test"]

        # Setting prompt shouldn't go to outputs
        pred3 = Prediction(outputs={})
        pred3.prompt = prompt_data
        assert "prompt" not in pred3.outputs
        assert pred3.prompt == prompt_data

    @pytest.mark.asyncio
    async def test_debug_backward_compatibility(self):
        """Test that debug mode maintains backward compatibility."""
        mock_provider = MockProvider()

        # Predict without debug parameter should work
        predict = Predict("question -> answer", provider=mock_provider)
        result = await predict(question="Test")

        # Should have all expected fields
        assert hasattr(result, "outputs")
        assert hasattr(result, "usage")
        assert hasattr(result, "metadata")
        assert hasattr(result, "success")
        assert hasattr(result, "error")
        assert hasattr(result, "prompt")

        # Metadata should still contain provider info (backward compat)
        assert "provider" in result.metadata
        assert "model" in result.metadata
        assert "adapter" in result.metadata
        assert "demos_used" in result.metadata

    @pytest.mark.asyncio
    async def test_debug_metadata_contains_prompt(self):
        """Test that prompt is also in metadata for backward compatibility."""
        mock_provider = MockProvider()
        predict = Predict("question -> answer", provider=mock_provider, debug=True)

        result = await predict(question="Test")

        # Should be in both places
        assert result.prompt is not None
        assert "prompt" in result.metadata
        assert result.metadata["prompt"] == result.prompt["messages"]
