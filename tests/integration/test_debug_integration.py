"""Integration tests for debug request/response logging functionality.

These tests verify that debug mode works correctly with real LLM providers
and captures complete request/response data.
"""

import os

import pytest

from logillm.core.avatar import Avatar
from logillm.core.predict import Predict
from logillm.core.react import ReAct
from logillm.core.retry import Retry
from logillm.core.tools import Tool
from logillm.providers import create_provider

# Skip all tests if no API keys are set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not any(
            [
                os.environ.get("OPENAI_API_KEY"),
                os.environ.get("ANTHROPIC_API_KEY"),
                os.environ.get("GOOGLE_API_KEY"),
            ]
        ),
        reason="No API keys set for any provider",
    ),
]


class TestDebugIntegration:
    """Integration tests for debug functionality with real providers."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider if API key is available."""
        if os.environ.get("OPENAI_API_KEY"):
            return create_provider("openai", model="gpt-4.1-mini")
        pytest.skip("OPENAI_API_KEY not set")

    @pytest.fixture
    def anthropic_provider(self):
        """Create Anthropic provider if API key is available."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            return create_provider("anthropic", model="claude-3-haiku-20240307")
        pytest.skip("ANTHROPIC_API_KEY not set")

    @pytest.fixture
    def google_provider(self):
        """Create Google provider if API key is available."""
        if os.environ.get("GOOGLE_API_KEY"):
            return create_provider("google", model="gemini-pro")
        pytest.skip("GOOGLE_API_KEY not set")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_predict_debug_openai(self, openai_provider):
        """Test Predict debug mode with OpenAI."""
        predict = Predict("question -> answer", provider=openai_provider, debug=True)

        result = await predict(question="What is 2+2? Answer briefly.")

        # Verify debug data is captured
        assert result.request is not None
        assert result.response is not None
        assert result.prompt is not None

        # Check request structure
        assert "messages" in result.request
        assert "provider" in result.request
        assert "model" in result.request
        assert "timestamp" in result.request

        # Check response structure
        assert "text" in result.response
        assert "usage" in result.response
        assert "cost" in result.response
        assert "latency" in result.response

        # Check usage details
        usage = result.response["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage

        # Verify actual values
        assert result.response["text"].strip()
        assert result.response["usage"]["total_tokens"] > 0
        assert result.response["cost"] >= 0
        assert result.response["latency"] > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_predict_debug_anthropic(self, anthropic_provider):
        """Test Predict debug mode with Anthropic."""
        predict = Predict("question -> answer", provider=anthropic_provider, debug=True)

        result = await predict(question="What is the capital of France?")

        # Verify debug data is captured
        assert result.request is not None
        assert result.response is not None
        assert result.prompt is not None

        # Check request structure
        assert "messages" in result.request
        assert "provider" in result.request
        assert "model" in result.request

        # Check response structure
        assert "text" in result.response
        assert "usage" in result.response

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_predict_debug_google(self, google_provider):
        """Test Predict debug mode with Google."""
        predict = Predict("question -> answer", provider=google_provider, debug=True)

        result = await predict(question="Hello, how are you?")

        # Verify debug data is captured
        assert result.request is not None
        assert result.response is not None
        assert result.prompt is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_debug_toggle_runtime(self, openai_provider):
        """Test runtime debug mode toggling."""
        predict = Predict("question -> answer", provider=openai_provider, debug=False)

        # Initially disabled
        result1 = await predict(question="Test 1")
        assert result1.request is None
        assert result1.response is None

        # Enable debug
        predict.enable_debug_mode()
        result2 = await predict(question="Test 2")
        assert result2.request is not None
        assert result2.response is not None

        # Disable debug
        predict.disable_debug_mode()
        result3 = await predict(question="Test 3")
        assert result3.request is None
        assert result3.response is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_avatar_debug_inheritance(self, openai_provider):
        """Test that Avatar inherits debug functionality."""

        # Create a simple tool for testing
        def add_numbers(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        tool = Tool(add_numbers, name="add", desc="Add two numbers")

        avatar = Avatar(
            "goal -> result",
            tools=[tool],
            max_iters=2,  # Keep it short for testing
        )

        # Enable debug on the internal actor
        avatar.actor.enable_debug_mode()

        result = await avatar(goal="What is 5 + 3?")

        # Avatar should have debug data from its internal Predict usage
        assert result.request is not None
        assert result.response is not None

    @pytest.mark.skip(reason="ReAct simplified - debug not propagated to top level")
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_react_debug_inheritance(self, openai_provider):
        """Test that ReAct inherits debug functionality."""

        # Create a simple tool for testing
        def search_info(query: str) -> str:
            """Search for information."""
            return f"Found info about: {query}"

        tool = Tool(search_info, name="search", desc="Search for information")

        react = ReAct(
            "question -> answer",
            tools=[tool],
            max_iters=2,  # Keep it short for testing
            config={"provider": openai_provider},
        )

        # Enable debug on the internal predict modules
        react.react.enable_debug_mode()
        react.extract.enable_debug_mode()

        result = await react(question="What is the weather like?")

        # ReAct should have debug data from its internal Predict usage
        assert result.request is not None
        assert result.response is not None
        assert "cost" in result.response
        assert isinstance(result.response["cost"], (int, float))
        assert result.response["cost"] >= 0

        # Verify usage details
        usage = result.response["usage"]
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_debug_provider_config_capture(self, openai_provider):
        """Test that debug mode captures provider configuration."""
        predict = Predict("question -> answer", provider=openai_provider, debug=True)

        result = await predict(question="Say hello")

        # Verify provider config is captured
        assert result.request is not None
        assert "provider_config" in result.request
        assert isinstance(result.request["provider_config"], dict)

        # Should contain basic provider info
        assert "model" in result.request or "model" in result.request.get("provider_config", {})

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_debug_backward_compatibility(self, openai_provider):
        """Test that debug mode maintains backward compatibility."""
        predict = Predict("question -> answer", provider=openai_provider, debug=True)

        result = await predict(question="Test")

        # Should have both old and new fields
        assert result.prompt is not None  # Backward compatibility
        assert result.request is not None  # New functionality
        assert result.response is not None  # New functionality

        # Prompt should be in metadata for backward compatibility
        assert "prompt" in result.metadata
        assert result.metadata["prompt"] == result.prompt["messages"]
