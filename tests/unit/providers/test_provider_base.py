"""Unit tests for Provider base class and MockProvider.

These tests use only mock objects and test the provider interface logic.
"""

import pytest

from logillm.core.providers import MockProvider, Provider, get_provider, register_provider
from logillm.core.types import Completion, TokenUsage, Usage


class TestProviderBase:
    """Test the Provider base class interface."""

    def test_provider_abstract_methods(self):
        """Test that Provider is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            Provider()

    def test_provider_interface_methods(self):
        """Test that provider interface methods exist."""

        # Create a concrete implementation
        class TestProvider(Provider):
            def __init__(self):
                self.name = "test"
                self.model = "test-model"

            async def complete(self, messages, **kwargs):
                return Completion(
                    text="test response",
                    usage=Usage(
                        tokens=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
                    ),
                    provider="test",
                    model="test-model",
                )

            async def embed(self, texts, **kwargs):
                return [[0.1, 0.2, 0.3] for _ in texts]

        provider = TestProvider()
        assert hasattr(provider, "complete")
        assert hasattr(provider, "embed")
        assert hasattr(provider, "name")
        assert hasattr(provider, "model")


class TestMockProvider:
    """Test MockProvider implementation."""

    def test_mock_provider_creation(self):
        """Test MockProvider can be created with default settings."""
        provider = MockProvider()

        assert provider.name == "mock"
        assert provider.model == "mock-model"
        assert provider.response_text == "Mock response"

    def test_mock_provider_custom_response(self):
        """Test MockProvider with custom response text."""
        custom_response = "Custom test response"
        provider = MockProvider(response_text=custom_response)

        assert provider.response_text == custom_response

    @pytest.mark.asyncio
    async def test_mock_provider_complete(self):
        """Test MockProvider completion."""
        provider = MockProvider(response_text="Test completion")

        messages = [{"role": "user", "content": "Test message"}]
        result = await provider.complete(messages)

        assert isinstance(result, Completion)
        assert result.text == "Test completion"
        assert result.provider == "mock"
        assert result.model == "mock-model"
        assert result.usage.tokens.input_tokens > 0
        assert result.usage.tokens.output_tokens > 0

    @pytest.mark.asyncio
    async def test_mock_provider_complete_with_params(self):
        """Test MockProvider accepts parameters (ignores them gracefully)."""
        provider = MockProvider()

        messages = [{"role": "user", "content": "Test"}]
        result = await provider.complete(messages, temperature=0.5, max_tokens=100, top_p=0.9)

        assert isinstance(result, Completion)
        # Mock provider should complete successfully even with parameters
        assert result.text == "Mock response"
        assert result.metadata["mock"]
        assert "call_number" in result.metadata

    @pytest.mark.asyncio
    async def test_mock_provider_embed(self):
        """Test MockProvider embedding."""
        provider = MockProvider()

        texts = ["text1", "text2", "text3"]
        embeddings = await provider.embed(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(emb[0], float) for emb in embeddings)

    @pytest.mark.asyncio
    async def test_mock_provider_stream(self):
        """Test MockProvider streaming."""
        provider = MockProvider(response_text="Hello world")

        messages = [{"role": "user", "content": "Test"}]
        tokens = []

        async for token in provider.stream(messages):
            tokens.append(token)

        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert "Hello world" in full_text

    def test_mock_provider_param_specs(self):
        """Test MockProvider parameter specifications."""
        provider = MockProvider()

        param_specs = provider.get_param_specs()

        assert "temperature" in param_specs
        assert "top_p" in param_specs
        assert "max_tokens" in param_specs

        # Check parameter spec structure
        temp_spec = param_specs["temperature"]
        assert temp_spec.name == "temperature"
        assert temp_spec.param_type.value == "float"
        assert temp_spec.range == (0.0, 2.0)


class TestProviderRegistry:
    """Test provider registration and retrieval."""

    def test_register_provider(self, mock_provider):
        """Test registering a provider."""
        register_provider(mock_provider, "test_provider")

        retrieved = get_provider("test_provider")
        assert retrieved is mock_provider

    def test_register_provider_as_default(self, mock_provider):
        """Test registering provider as default."""
        register_provider(mock_provider, "test_provider", set_default=True)

        default_provider = get_provider()
        assert default_provider is mock_provider

    def test_get_nonexistent_provider(self):
        """Test getting non-existent provider raises error."""
        from logillm.providers.base import ProviderError

        with pytest.raises(ProviderError, match="not found"):
            get_provider("nonexistent_provider")

    def test_get_provider_no_default(self):
        """Test getting default when none set raises error."""
        # Clear any existing providers properly
        from logillm.providers.registry import clear_registry

        clear_registry()

        from logillm.providers.base import ProviderError

        with pytest.raises(ProviderError, match="No default provider"):
            get_provider()

    def test_provider_registry_isolation(self):
        """Test that provider registry is properly isolated between tests."""
        # This test verifies the conftest.py fixture works
        from logillm.providers.registry import list_providers

        # Registry should be empty at start of test
        assert len(list_providers()) == 0


class TestProviderErrorHandling:
    """Test provider error handling."""

    @pytest.mark.asyncio
    async def test_provider_completion_error_handling(self):
        """Test provider handles completion errors gracefully."""

        class FailingProvider(Provider):
            def __init__(self):
                self.name = "failing"
                self.model = "failing-model"

            async def complete(self, messages, **kwargs):
                raise Exception("Simulated API failure")

            async def embed(self, texts, **kwargs):
                return [[0.1] for _ in texts]

        provider = FailingProvider()

        with pytest.raises(Exception, match="Simulated API failure"):
            await provider.complete([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_mock_provider_with_custom_responses(self):
        """Test MockProvider with custom response list."""
        responses = ["Response 1", "Response 2", "Response 3"]
        provider = MockProvider(responses=responses)

        # First call should get first response
        result1 = await provider.complete([{"role": "user", "content": "test"}])
        assert result1.text == "Response 1"

        # Second call should get second response
        result2 = await provider.complete([{"role": "user", "content": "test"}])
        assert result2.text == "Response 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
