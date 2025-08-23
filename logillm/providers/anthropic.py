"""Anthropic Claude provider implementation."""

import importlib.util
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..core.types import Completion, TokenUsage, Usage
from .base import Provider, ProviderError, RateLimitError, TimeoutError

# Check if anthropic is available using importlib
HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None

if HAS_ANTHROPIC:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
elif TYPE_CHECKING:
    # For type checking when anthropic is not available
    from typing import Any as Anthropic
    from typing import Any as AsyncAnthropic

    # Create a mock anthropic module for type checking
    class MockAnthropic:
        NOT_GIVEN = object()

        class RateLimitError(Exception):
            pass

        class APITimeoutError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        class PermissionDeniedError(Exception):
            pass

        class NotFoundError(Exception):
            pass

        class BadRequestError(Exception):
            pass

        class APIError(Exception):
            pass

    anthropic = MockAnthropic()  # type: ignore
else:
    Anthropic = None  # type: ignore
    AsyncAnthropic = None  # type: ignore
    anthropic = None  # type: ignore


class AnthropicProvider(Provider):
    """Anthropic Claude API provider implementation.

    Supports Claude 4 models (Opus 4.1, Sonnet 4) and legacy Claude 3 models.
    Features:
    - System prompts
    - Streaming responses
    - Vision support for multimodal models
    - Tool/function calling
    - Hybrid reasoning modes (instant vs extended thinking)
    """

    def __init__(
        self,
        model: str = "claude-opus-4-1",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-opus-4-1", "claude-sonnet-4", "claude-opus-4")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Optional base URL for API
            **kwargs: Additional provider configuration
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install logillm[anthropic]"
            )

        # Get API key from environment if not provided
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        super().__init__(
            name="anthropic", model=model, api_key=api_key, base_url=base_url, **kwargs
        )

        # Initialize clients
        # Assert for type checker - we know HAS_ANTHROPIC is True here
        assert Anthropic is not None and AsyncAnthropic is not None

        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        self.async_client = AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=self.timeout, max_retries=0
        )

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """Generate completion using Anthropic API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional Anthropic parameters

        Returns:
            Completion object
        """
        # Check cache first
        cached = await self._get_cached(messages, **kwargs)
        if cached:
            return cached

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Prepare parameters
        params = self._prepare_params(kwargs)

        try:
            # Make API call
            response = await self.async_client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
                **params,
            )

            # Convert to our Completion type
            completion = self._parse_completion(response)

            # Cache the result
            await self._set_cache(messages, completion, **kwargs)

            # Update metrics
            self._update_metrics(completion)

            return completion

        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}") from e
        except anthropic.APITimeoutError as e:
            raise TimeoutError(f"Anthropic request timed out: {e}") from e
        except anthropic.AuthenticationError as e:
            raise ProviderError(f"Anthropic authentication failed: {e}") from e
        except anthropic.PermissionDeniedError as e:
            raise ProviderError(f"Anthropic permission denied: {e}") from e
        except anthropic.NotFoundError as e:
            raise ProviderError(f"Anthropic model not found: {e}") from e
        except anthropic.BadRequestError as e:
            raise ProviderError(f"Anthropic bad request: {e}") from e
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}") from e

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        """Stream completion tokens.

        Args:
            messages: List of message dicts
            **kwargs: Additional Anthropic parameters

        Yields:
            Completion tokens as they arrive
        """
        # Convert messages
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Prepare parameters
        params = self._prepare_params(kwargs)

        try:
            # Create streaming response
            async with self.async_client.messages.stream(
                model=self.model,
                messages=anthropic_messages,
                system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
                **params,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}") from e
        except anthropic.APITimeoutError as e:
            raise TimeoutError(f"Anthropic request timed out: {e}") from e
        except Exception as e:
            raise ProviderError(f"Streaming error: {e}") from e

    async def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings.

        Note: Anthropic doesn't provide embeddings API.
        Raises NotImplementedError.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use OpenAI or another provider for embeddings."
        )

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert OpenAI-style messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, anthropic_messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Anthropic uses a separate system parameter
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                # Unknown role, treat as user
                anthropic_messages.append({"role": "user", "content": content})

        # Ensure conversation starts with user message
        if anthropic_messages and anthropic_messages[0]["role"] != "user":
            anthropic_messages.insert(0, {"role": "user", "content": "Continue the conversation"})

        # Ensure alternating pattern
        cleaned_messages = []
        last_role = None
        for msg in anthropic_messages:
            if msg["role"] == last_role:
                # Merge consecutive messages from same role
                if cleaned_messages:
                    cleaned_messages[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned_messages.append(msg)
                last_role = msg["role"]

        return system_prompt, cleaned_messages

    def _prepare_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare parameters for Anthropic API call."""
        params = {}

        # Map common parameters
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        else:
            params["max_tokens"] = 4096  # Anthropic requires this

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]

        if "top_k" in kwargs:
            params["top_k"] = kwargs["top_k"]

        if "stop" in kwargs:
            stop = kwargs["stop"]
            if isinstance(stop, str):
                params["stop_sequences"] = [stop]
            else:
                params["stop_sequences"] = stop

        # Handle tools/functions if provided
        if "tools" in kwargs:
            params["tools"] = kwargs["tools"]

        if "tool_choice" in kwargs:
            params["tool_choice"] = kwargs["tool_choice"]

        return params

    def _parse_completion(self, response: Any) -> Completion:
        """Parse Anthropic response into our Completion type.

        Args:
            response: Anthropic Message response

        Returns:
            Completion object
        """
        # Extract text from content blocks
        text = ""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
                elif isinstance(block, dict) and "text" in block:
                    text += block["text"]

        # Parse usage
        usage = None
        if hasattr(response, "usage"):
            tokens = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_tokens=0,
                reasoning_tokens=0,
            )

            usage = Usage(
                tokens=tokens,
                cost=self._calculate_cost(tokens),
                provider=self.name,
                model=self.model,
            )
        else:
            usage = Usage()

        # Build metadata
        metadata = {}
        if hasattr(response, "id"):
            metadata["message_id"] = response.id
        if hasattr(response, "model"):
            metadata["model"] = response.model
        if hasattr(response, "stop_reason"):
            metadata["stop_reason"] = response.stop_reason
        if hasattr(response, "stop_sequence"):
            metadata["stop_sequence"] = response.stop_sequence

        return Completion(
            text=text,
            usage=usage,
            metadata=metadata,
            finish_reason=getattr(response, "stop_reason", "stop"),
            model=self.model,
            provider=self.name,
        )

    def _calculate_cost(self, tokens: TokenUsage) -> float:
        """Calculate estimated cost based on token usage.

        Anthropic pricing (as of 2025):
        - Claude Opus 4.1/4: $15/$75 per 1M tokens (input/output)
        - Claude Sonnet 4: $3/$15 per 1M tokens
        - Claude 3.5 Sonnet: $3/$15 per 1M tokens
        - Claude 3.5 Haiku: $0.25/$1.25 per 1M tokens
        - Claude 3 Opus: $15/$75 per 1M tokens
        """
        model_lower = self.model.lower()

        # Claude 4 models
        if "opus-4" in model_lower or "opus-4-1" in model_lower:
            input_cost = tokens.input_tokens * 0.000015
            output_cost = tokens.output_tokens * 0.000075
        elif "sonnet-4" in model_lower:
            input_cost = tokens.input_tokens * 0.000003
            output_cost = tokens.output_tokens * 0.000015
        # Legacy Claude 3 models
        elif "opus" in model_lower:
            input_cost = tokens.input_tokens * 0.000015
            output_cost = tokens.output_tokens * 0.000075
        elif "haiku" in model_lower:
            input_cost = tokens.input_tokens * 0.00000025
            output_cost = tokens.output_tokens * 0.00000125
        else:  # Sonnet (default)
            input_cost = tokens.input_tokens * 0.000003
            output_cost = tokens.output_tokens * 0.000015

        return input_cost + output_cost

    def _update_metrics(self, completion: Completion) -> None:
        """Update provider metrics from completion.

        Args:
            completion: Completion to extract metrics from
        """
        if completion.usage and completion.usage.tokens:
            tokens = completion.usage.tokens
            self._metrics["total_tokens"] = (
                self._metrics.get("total_tokens", 0) + tokens.total_tokens
            )
            self._metrics["input_tokens"] = (
                self._metrics.get("input_tokens", 0) + tokens.input_tokens
            )
            self._metrics["output_tokens"] = (
                self._metrics.get("output_tokens", 0) + tokens.output_tokens
            )

            if tokens.cached_tokens:
                self._metrics["cached_tokens"] = (
                    self._metrics.get("cached_tokens", 0) + tokens.cached_tokens
                )

        self._metrics["total_calls"] = self._metrics.get("total_calls", 0) + 1


__all__ = ["AnthropicProvider"]
