"""Base provider interface and utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..core.parameters import (
    STANDARD_PARAM_SPECS,
    STANDARD_PRESETS,
    ParameterProvider,
    ParamGroup,
    ParamPreset,
    ParamSpec,
)
from ..core.types import CacheLevel, Completion, Configuration


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""

    pass


class TimeoutError(ProviderError):
    """Request timeout error."""

    pass


class Provider(ParameterProvider):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        model: str | None = None,
        *,
        name: str | None = None,  # Alternative to provider_name
        api_key: str | None = None,
        base_url: str | None = None,
        config: Configuration | None = None,
        cache_level: CacheLevel = CacheLevel.NONE,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs: Any,  # Accept additional provider-specific parameters
    ) -> None:
        """Initialize provider.

        Args:
            provider_name: Provider name (e.g., "openai", "anthropic")
            model: Model identifier (e.g., "gpt-4", "claude-3")
            name: Alternative to provider_name for backward compatibility
            api_key: API key for authentication
            base_url: Optional base URL override
            config: Additional configuration
            cache_level: Caching level for responses
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
        """
        # Handle both provider_name and name for flexibility
        self.name = provider_name or name or "unknown"
        self.model = model or "unknown"
        self.api_key = api_key
        self.base_url = base_url
        self.config = config or {}
        self.cache_level = cache_level
        self._max_retries = max_retries
        self.timeout = timeout
        self._cache: dict[str, Completion] = {}
        self._metrics: dict[str, float] = {}
        self._health_status = True

        # Store additional provider-specific parameters
        self.provider_params = kwargs

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """Generate completion from messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional provider-specific parameters

        Returns:
            Completion object with text and metadata
        """
        ...

    @abstractmethod
    async def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            **kwargs: Additional provider-specific parameters

        Returns:
            List of embedding vectors
        """
        ...

    def complete_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """Synchronous completion interface.

        This method provides a synchronous wrapper around the async complete method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.complete(messages, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self.complete(messages, **kwargs))
        except RuntimeError:
            # No event loop
            return asyncio.run(self.complete(messages, **kwargs))

    async def complete_with_retry(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Completion:
        """Complete with automatic retry on failure.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional provider-specific parameters

        Returns:
            Completion object

        Raises:
            ProviderError: After all retries exhausted
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                return await self.complete(messages, **kwargs)
            except RateLimitError as e:
                # Exponential backoff for rate limits
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)
                last_error = e
            except TimeoutError as e:
                # Quick retry for timeouts
                await asyncio.sleep(0.5)
                last_error = e
            except Exception as e:
                # Log and retry for other errors
                last_error = e
                if attempt == self._max_retries - 1:
                    break
                await asyncio.sleep(1)

        raise ProviderError(
            f"Failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        """Stream completion tokens.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional provider-specific parameters

        Yields:
            Completion tokens as they become available
        """
        # Default implementation: return complete response at once
        completion = await self.complete(messages, **kwargs)
        yield completion.text

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses."""
        return False

    def supports_structured_output(self) -> bool:
        """Check if provider supports structured output (JSON mode, etc.)."""
        return False

    def supports_function_calling(self) -> bool:
        """Check if provider supports function/tool calling."""
        return False

    def supports_vision(self) -> bool:
        """Check if provider supports image inputs."""
        return False

    # Cache management
    def _cache_key(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generate cache key for request."""
        # Create deterministic key from messages and kwargs
        key_data = {
            "messages": messages,
            "kwargs": kwargs,
            "model": self.model,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def _get_cached(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion | None:
        """Get cached completion if available."""
        if self.cache_level == CacheLevel.NONE:
            return None

        key = self._cache_key(messages, **kwargs)
        return self._cache.get(key)

    async def _set_cache(
        self, messages: list[dict[str, str]], completion: Completion, **kwargs: Any
    ) -> None:
        """Cache completion result."""
        if self.cache_level == CacheLevel.NONE:
            return

        key = self._cache_key(messages, **kwargs)
        self._cache[key] = completion

    # Metrics and monitoring
    def get_metrics(self) -> dict[str, float]:
        """Get provider metrics."""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset provider metrics."""
        self._metrics.clear()

    def get_health_status(self) -> bool:
        """Get provider health status."""
        return self._health_status

    async def health_check(self) -> bool:
        """Perform health check on provider."""
        try:
            # Simple health check: try a minimal completion
            await self.complete(messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
            self._health_status = True
            return True
        except Exception:
            self._health_status = False
            return False

    # Parameter management (from ParameterProvider)
    def get_param_specs(self) -> dict[str, ParamSpec]:
        """Get specifications for all supported parameters."""
        # Return standard OpenAI-style parameters by default
        # Providers can override to add custom parameters
        return STANDARD_PARAM_SPECS.copy()

    def get_param_groups(self) -> list[ParamGroup]:
        """Get parameter groups for organization."""
        return [
            ParamGroup(
                name="generation",
                description="Text generation parameters",
                params=[
                    "temperature",
                    "top_p",
                    "max_tokens",
                    "frequency_penalty",
                    "presence_penalty",
                ],
            ),
            ParamGroup(
                name="efficiency",
                description="Efficiency and cost parameters",
                params=["max_tokens", "seed"],
            ),
            ParamGroup(
                name="advanced",
                description="Advanced parameters",
                params=["logit_bias", "stop", "n"],
            ),
        ]

    def get_param_presets(self) -> dict[ParamPreset, Configuration]:
        """Get available parameter presets."""
        return STANDARD_PRESETS.copy()

    def validate_params(self, params: Configuration) -> tuple[bool, list[str]]:
        """Validate parameters and return validation status."""
        specs = self.get_param_specs()
        errors = []

        # Map ParamType values to Python types
        type_map = {
            "float": float,
            "int": int,
            "bool": bool,
            "string": str,
            "str": str,
            "list": list,
            "dict": dict,
        }

        for key, value in params.items():
            if key in specs:
                spec = specs[key]
                expected_type = type_map.get(spec.param_type.value)

                if expected_type:
                    # Validate type
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"Parameter {key} must be {spec.param_type.value}, got {type(value).__name__}"
                        )

                    # Validate range
                    if spec.range and isinstance(value, (int, float)):
                        min_val, max_val = spec.range
                        if value < min_val or value > max_val:
                            errors.append(
                                f"Parameter {key} must be between {min_val} and {max_val}, got {value}"
                            )

        return len(errors) == 0, errors

    def clean_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clean and validate parameters for use."""
        specs = self.get_param_specs()
        validated = {}

        # Map ParamType values to Python types
        type_map = {
            "float": float,
            "int": int,
            "bool": bool,
            "string": str,
            "str": str,
            "list": list,
            "dict": dict,
        }

        for key, value in params.items():
            if key in specs:
                spec = specs[key]
                expected_type = type_map.get(spec.param_type.value)

                if expected_type:
                    # Try to coerce type
                    if not isinstance(value, expected_type):
                        try:
                            if expected_type is float:
                                value = float(value)
                            elif expected_type is int:
                                value = int(value)
                            elif expected_type is str:
                                value = str(value)
                            elif expected_type is bool:
                                value = bool(value)
                        except (ValueError, TypeError):
                            continue  # Skip invalid values

                    # Check range
                    if spec.range and isinstance(value, (int, float)):
                        min_val, max_val = spec.range
                        if value < min_val or value > max_val:
                            continue  # Skip out of range values

                validated[key] = value

        return validated

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
