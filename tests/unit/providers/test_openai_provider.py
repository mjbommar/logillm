"""Unit tests for OpenAI provider.

These tests use mocks and don't require API keys. They test the provider logic,
parameter handling, and response parsing.
"""

from unittest.mock import Mock, patch

import pytest

from logillm.core.parameters import ParamDomain, ParamType
from logillm.core.types import Completion, TokenUsage, Usage
from logillm.providers.base import ProviderError, RateLimitError, TimeoutError
from logillm.providers.openai import OpenAIProvider


class TestOpenAIProviderInit:
    """Test OpenAI provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = OpenAIProvider(model="gpt-4.1", api_key="test-key")

        assert provider.name == "openai"
        assert provider.model == "gpt-4.1"
        assert provider.api_key == "test-key"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    def test_init_with_env_key(self):
        """Test initialization with environment variable."""
        provider = OpenAIProvider(model="gpt-4o")

        assert provider.api_key == "env-key"
        assert provider.model == "gpt-4o"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_no_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(model="gpt-4.1")

    @patch("logillm.providers.openai.HAS_OPENAI", False)
    def test_init_no_openai_package_raises_error(self):
        """Test initialization without openai package raises ImportError."""
        with pytest.raises(ImportError, match="OpenAI provider requires"):
            OpenAIProvider(model="gpt-4.1", api_key="test-key")


class TestOpenAIProviderCapabilities:
    """Test OpenAI provider capability reporting."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(model="gpt-4.1", api_key="test-key")

    def test_supports_streaming(self, provider):
        """Test streaming capability."""
        assert provider.supports_streaming() is True

    def test_supports_structured_output(self, provider):
        """Test structured output capability."""
        assert provider.supports_structured_output() is True

    def test_supports_function_calling(self, provider):
        """Test function calling capability."""
        assert provider.supports_function_calling() is True

    def test_supports_vision_gpt4o(self):
        """Test vision support for GPT-4o."""
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        assert provider.supports_vision() is True

    def test_supports_vision_gpt4_turbo(self):
        """Test vision support for GPT-4 Turbo."""
        provider = OpenAIProvider(model="gpt-4-turbo", api_key="test-key")
        assert provider.supports_vision() is True

    def test_no_vision_support_regular_models(self):
        """Test no vision support for regular models."""
        provider = OpenAIProvider(model="gpt-4.1-mini", api_key="test-key")
        assert provider.supports_vision() is False


class TestOpenAIProviderParameters:
    """Test OpenAI provider parameter handling."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(model="gpt-4.1", api_key="test-key")

    @pytest.fixture
    def o1_provider(self):
        """Create o1 reasoning model provider."""
        return OpenAIProvider(model="o1-preview", api_key="test-key")

    def test_get_param_specs(self, provider):
        """Test parameter specifications."""
        specs = provider.get_param_specs()

        assert "temperature" in specs
        assert "top_p" in specs
        assert "max_tokens" in specs
        assert "presence_penalty" in specs
        assert "frequency_penalty" in specs

        # OpenAI doesn't support top_k
        assert "top_k" not in specs

        # Check spec details
        temp_spec = specs["temperature"]
        assert temp_spec.param_type == ParamType.FLOAT
        assert temp_spec.domain == ParamDomain.GENERATION
        assert temp_spec.range == (0.0, 2.0)

    def test_o1_param_specs(self, o1_provider):
        """Test o1 reasoning model parameter specifications."""
        specs = o1_provider.get_param_specs()

        # o1 models don't support these parameters
        assert "temperature" not in specs
        assert "top_p" not in specs
        assert "presence_penalty" not in specs
        assert "frequency_penalty" not in specs

        # But they do support max_completion_tokens
        assert "max_completion_tokens" in specs

        spec = specs["max_completion_tokens"]
        assert spec.param_type == ParamType.INT
        assert spec.domain == ParamDomain.EFFICIENCY

    def test_is_reasoning_model(self, provider, o1_provider):
        """Test reasoning model detection."""
        assert provider._is_reasoning_model() is False
        assert o1_provider._is_reasoning_model() is True

        # Test other o1 variants
        o1_mini = OpenAIProvider(model="o1-mini", api_key="test-key")
        assert o1_mini._is_reasoning_model() is True

    def test_prepare_params_regular_model(self, provider):
        """Test parameter preparation for regular models."""
        params = provider._prepare_params(
            {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
                "stop": ["\n"],
                "invalid_param": "should_be_ignored",
            }
        )

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9
        assert params["stop"] == ["\n"]

    def test_prepare_params_o1_model(self, o1_provider):
        """Test parameter preparation for o1 models."""
        params = o1_provider._prepare_params(
            {
                "temperature": 0.7,  # Should be removed
                "max_tokens": 100,  # Should become max_completion_tokens
                "top_p": 0.9,  # Should be removed
                "stop": ["\n"],  # Should remain
            }
        )

        assert "temperature" not in params
        assert "top_p" not in params
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 100
        assert params["stop"] == ["\n"]

    def test_prepare_params_structured_output(self, provider):
        """Test parameter preparation with structured output."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        params = provider._prepare_params({"response_format": TestModel, "temperature": 0.0})

        assert "response_format" in params
        response_format = params["response_format"]
        assert response_format["type"] == "json_schema"
        assert "json_schema" in response_format
        assert response_format["json_schema"]["name"] == "TestModel"

    def test_clean_params(self, provider):
        """Test parameter cleaning and validation."""
        raw_params = {
            "temperature": "0.7",  # String that can be converted
            "max_tokens": 100,
            "invalid_range": 5.0,  # Assuming this would be out of range
            "unknown_param": "ignored",
        }

        cleaned = provider.clean_params(raw_params)

        assert cleaned["temperature"] == 0.7  # Converted to float
        assert cleaned["max_tokens"] == 100
        # unknown_param should be filtered out by the base clean_params logic


class TestOpenAIProviderResponseParsing:
    """Test OpenAI response parsing logic."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(model="gpt-4.1", api_key="test-key")

    def test_parse_completion_basic(self, provider):
        """Test parsing basic completion response."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4.1"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens_details = None

        completion = provider._parse_completion(mock_response)

        assert isinstance(completion, Completion)
        assert completion.text == "Hello, world!"
        assert completion.model == "gpt-4.1"
        assert completion.provider == "openai"
        assert completion.finish_reason == "stop"
        assert completion.usage.tokens.input_tokens == 10
        assert completion.usage.tokens.output_tokens == 5
        assert completion.usage.tokens.total_tokens == 15

    def test_parse_completion_with_tool_calls(self, provider):
        """Test parsing completion with tool calls."""
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "NYC"}'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.model = "gpt-4.1"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 8
        mock_response.usage.prompt_tokens_details = None

        completion = provider._parse_completion(mock_response)

        assert completion.finish_reason == "tool_calls"
        assert "tool_calls" in completion.metadata
        tool_calls = completion.metadata["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["function"]["name"] == "get_weather"

    def test_parse_completion_with_cached_tokens(self, provider):
        """Test parsing completion with cached tokens."""
        mock_details = Mock()
        mock_details.cached_tokens = 5

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Cached response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4.1"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = mock_details

        completion = provider._parse_completion(mock_response)

        assert completion.usage.tokens.cached_tokens == 5

    def test_parse_completion_empty_response(self, provider):
        """Test parsing empty response."""
        mock_response = Mock()
        mock_response.choices = []
        mock_response.model = "gpt-4.1"
        mock_response.usage = None

        completion = provider._parse_completion(mock_response)

        assert completion.text == ""
        assert completion.finish_reason == "unknown"
        assert completion.usage.tokens.total_tokens == 0


class TestOpenAIProviderErrorHandling:
    """Test OpenAI provider error handling."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(model="gpt-4.1", api_key="test-key")

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, provider):
        """Test rate limit error conversion."""
        import openai

        with patch.object(provider.async_client.chat.completions, "create") as mock_create:
            mock_create.side_effect = openai.RateLimitError(
                "Rate limit exceeded", response=Mock(), body={}
            )

            with pytest.raises(RateLimitError, match="OpenAI rate limit exceeded"):
                await provider.complete([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, provider):
        """Test timeout error conversion."""
        import openai

        with patch.object(provider.async_client.chat.completions, "create") as mock_create:
            mock_create.side_effect = openai.APITimeoutError("Request timed out")

            with pytest.raises(TimeoutError, match="OpenAI request timed out"):
                await provider.complete([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, provider):
        """Test authentication error conversion."""
        import openai

        with patch.object(provider.async_client.chat.completions, "create") as mock_create:
            mock_create.side_effect = openai.AuthenticationError(
                "Invalid API key", response=Mock(), body={}
            )

            with pytest.raises(ProviderError, match="OpenAI authentication failed"):
                await provider.complete([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_generic_api_error_handling(self, provider):
        """Test generic API error conversion."""
        import openai

        with patch.object(provider.async_client.chat.completions, "create") as mock_create:
            mock_create.side_effect = openai.BadRequestError(
                "Bad request error", response=Mock(), body={}
            )

            with pytest.raises(ProviderError, match="OpenAI bad request"):
                await provider.complete([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self, provider):
        """Test unexpected error handling."""
        with patch.object(provider.async_client.chat.completions, "create") as mock_create:
            mock_create.side_effect = ValueError("Unexpected error")

            with pytest.raises(ProviderError, match="Unexpected error"):
                await provider.complete([{"role": "user", "content": "test"}])


class TestOpenAIProviderMetrics:
    """Test OpenAI provider metrics tracking."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(model="gpt-4.1", api_key="test-key")

    def test_update_metrics(self, provider):
        """Test metrics update from completion."""
        completion = Completion(
            text="Test response",
            usage=Usage(tokens=TokenUsage(input_tokens=10, output_tokens=5, cached_tokens=2)),
            provider="openai",
            model="gpt-4.1",
        )

        provider._update_metrics(completion)
        metrics = provider.get_metrics()

        assert metrics["total_tokens"] == 15
        assert metrics["input_tokens"] == 10
        assert metrics["output_tokens"] == 5
        assert metrics["cached_tokens"] == 2
        assert metrics["total_calls"] == 1

    def test_reset_metrics(self, provider):
        """Test metrics reset."""
        # Add some metrics first
        provider._metrics["test"] = 42

        provider.reset_metrics()
        metrics = provider.get_metrics()

        assert len(metrics) == 0


class TestOpenAIProviderStructuredOutput:
    """Test OpenAI structured output functionality."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(model="gpt-4.1", api_key="test-key")

    @pytest.mark.asyncio
    async def test_create_structured_completion(self, provider):
        """Test structured completion creation."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            result: int
            explanation: str

        # Mock the completion response
        mock_completion = Completion(
            text='{"result": 42, "explanation": "The answer"}',
            usage=Usage(tokens=TokenUsage(input_tokens=10, output_tokens=5)),
            provider="openai",
            model="gpt-4.1",
        )

        with patch.object(provider, "complete", return_value=mock_completion):
            result = await provider.create_structured_completion(
                messages=[{"role": "user", "content": "Test"}], response_format=TestModel
            )

            assert isinstance(result, TestModel)
            assert result.result == 42
            assert result.explanation == "The answer"

    @pytest.mark.asyncio
    async def test_create_structured_completion_invalid_json(self, provider):
        """Test structured completion with invalid JSON."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            result: int

        # Mock invalid JSON response
        mock_completion = Completion(
            text="invalid json", usage=Usage(), provider="openai", model="gpt-4.1"
        )

        with patch.object(provider, "complete", return_value=mock_completion):
            with pytest.raises(ProviderError, match="Failed to parse structured output"):
                await provider.create_structured_completion(
                    messages=[{"role": "user", "content": "Test"}], response_format=TestModel
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
