"""Integration tests for Anthropic provider.

These tests use the real Anthropic API. Set ANTHROPIC_API_KEY environment variable
to run these tests. They are marked with @pytest.mark.integration so they
can be skipped in CI.
"""

import os

import pytest
from pydantic import BaseModel

from logillm.core.predict import Predict
from logillm.providers import create_provider
from logillm.providers.anthropic import AnthropicProvider

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


class TestAnthropicProvider:
    """Test Anthropic provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create Anthropic provider with Claude Opus 4.1."""
        return create_provider(
            "anthropic",
            model="claude-opus-4-1",
        )

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_basic_completion(self, provider):
        """Test basic text completion."""
        messages = [{"role": "user", "content": "Say 'Hello, LogiLLM!' exactly."}]

        completion = await provider.complete(messages)

        assert completion.text
        assert "Hello, LogiLLM!" in completion.text
        assert completion.usage.tokens.input_tokens > 0
        assert completion.usage.tokens.output_tokens > 0
        assert completion.provider == "anthropic"
        assert completion.model  # Should be set

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_system_message(self, provider):
        """Test system message handling."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        completion = await provider.complete(messages)

        assert completion.text
        # Claude should respond with the answer, possibly in pirate speak
        assert "4" in completion.text or "four" in completion.text.lower()

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_temperature_control(self, provider):
        """Test temperature parameter."""
        messages = [{"role": "user", "content": "Write exactly: 'test'"}]

        # Low temperature for deterministic output
        completion = await provider.complete(messages, temperature=0.0)

        assert completion.text
        assert "test" in completion.text.lower()

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_max_tokens(self, provider):
        """Test max_tokens parameter."""
        messages = [{"role": "user", "content": "Count from 1 to 100, writing out each number"}]

        # Limit response length
        completion = await provider.complete(messages, max_tokens=50)

        assert completion.text
        assert len(completion.text) > 0
        # Should be truncated - check it doesn't reach high numbers
        # The model should start counting but get cut off
        assert "50" not in completion.text or "60" not in completion.text

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_conversation(self, provider):
        """Test multi-turn conversation."""
        messages = [
            {"role": "user", "content": "My name is Alice. Remember it."},
            {"role": "assistant", "content": "Hello Alice! I'll remember your name."},
            {"role": "user", "content": "What's my name?"},
        ]

        completion = await provider.complete(messages)

        assert completion.text
        assert "Alice" in completion.text

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_streaming(self, provider):
        """Test streaming responses."""
        messages = [{"role": "user", "content": "Count from 1 to 5"}]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 0
        # Full response should contain numbers
        full_text = "".join(chunks)
        assert "1" in full_text
        assert "2" in full_text

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_with_predict_module(self, provider):
        """Test using provider with Predict module."""
        from logillm.core import register_provider

        # Register as default provider
        register_provider(provider, set_default=True)

        # Create predictor
        qa = Predict("question -> answer")

        # Test prediction
        result = await qa.forward(question="What is the capital of France?")

        assert result.answer
        assert "Paris" in result.answer

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_structured_output_basic(self, provider):
        """Test structured output with Pydantic model."""

        class MathAnswer(BaseModel):
            reasoning: str
            answer: int

        # Anthropic doesn't have native structured output like OpenAI
        # But we can ask for JSON format
        messages_with_format = [
            {
                "role": "user",
                "content": (
                    "What is 15 + 27? Think step by step. "
                    "Respond with JSON containing 'reasoning' (string) and 'answer' (integer)."
                ),
            }
        ]

        completion = await provider.complete(messages_with_format)

        assert completion.text
        # Should contain JSON-like structure
        assert "{" in completion.text
        assert "}" in completion.text

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_retry_on_rate_limit(self, provider):
        """Test retry logic (won't actually hit rate limit in normal testing)."""
        messages = [{"role": "user", "content": "Say 'test'"}]

        # This should succeed without retries
        completion = await provider.complete_with_retry(messages)

        assert completion.text
        assert "test" in completion.text.lower()

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_cache_behavior(self, provider):
        """Test caching behavior."""
        from logillm.core.types import CacheLevel

        # Create provider with caching enabled
        cached_provider = AnthropicProvider(model="claude-opus-4-1", cache_level=CacheLevel.MEMORY)

        messages = [{"role": "user", "content": "What is 2+2?"}]

        # First call
        completion1 = await cached_provider.complete(messages)
        tokens1 = completion1.usage.tokens.input_tokens

        # Second call (should be cached)
        completion2 = await cached_provider.complete(messages)

        # Should get same response
        assert completion1.text == completion2.text
        # Cache hit should have same token count
        assert completion2.usage.tokens.input_tokens == tokens1

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_embedding_not_supported(self, provider):
        """Test that embeddings raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Anthropic does not provide an embeddings API"
        ):
            await provider.embed(["test text"])

    @pytest.mark.integration
    @pytest.mark.anthropic
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_model_variants(self):
        """Test different Claude model variants."""
        models_to_test = [
            "claude-opus-4-1",  # Latest Opus 4.1 (best for complex tasks)
            "claude-sonnet-4",  # Sonnet 4 (balanced performance/cost)
            "claude-opus-4",  # Original Opus 4
        ]

        for model in models_to_test:
            try:
                provider = AnthropicProvider(model=model)
                messages = [{"role": "user", "content": "Say 'Hi'"}]

                completion = await provider.complete(messages, max_tokens=10)

                assert completion.text
                assert len(completion.text) > 0
                # Check cost calculation differs by model
                if "sonnet" in model.lower():
                    # Sonnet is cheaper than Opus
                    assert completion.usage.cost < 0.001  # Less than $0.001
                else:
                    # Opus costs more (but still small for 10 tokens)
                    assert completion.usage.cost >= 0.0001  # At least $0.0001
            except Exception as e:
                # Model might not be available on this account
                if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                    pytest.skip(f"Model {model} not available")
                else:
                    raise
