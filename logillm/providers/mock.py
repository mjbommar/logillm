"""Mock provider for testing."""

import asyncio
from typing import Any

from ..core.types import Completion, TokenUsage, Usage
from .base import Provider


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(
        self,
        response_text: str = "Mock response",
        responses: list[str] | None = None,
        latency: float = 0.0,
        error_rate: float = 0.0,
        **kwargs,
    ):
        """Initialize mock provider.

        Args:
            response_text: Default response text
            responses: List of responses to cycle through
            latency: Simulated latency in seconds
            error_rate: Probability of error (0-1)
            **kwargs: Additional arguments for Provider
        """
        # Extract responses before passing to parent
        self.responses = responses
        self.response_index = 0
        self.response_text = response_text
        self.call_count = 0
        self.latency = latency
        self.error_rate = error_rate

        # Remove custom args before passing to parent
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["responses", "response_text", "latency", "error_rate"]
        }

        # Set default model if not provided
        if "model" not in filtered_kwargs:
            filtered_kwargs["model"] = "mock-model"

        super().__init__(provider_name="mock", **filtered_kwargs)

    def set_mock_response(self, response: str | list[str] | Any) -> None:
        """Set the mock response(s) to return.

        Args:
            response: Either a single response string, list of responses, or a Completion object
        """
        if isinstance(response, str):
            self.response_text = response
            self.responses = None
        elif isinstance(response, list):
            self.responses = response
            self.response_index = 0
        else:
            # Handle other types (like Completion objects) by storing them directly
            self._mock_completion = response
            self.responses = None

    async def _complete_impl(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """Generate mock completion."""
        self.call_count += 1

        # Simulate latency
        if self.latency > 0:
            await asyncio.sleep(self.latency)
        else:
            await asyncio.sleep(0.01)  # Default minimal latency

        # Simulate errors
        if self.error_rate > 0:
            import random

            if random.random() < self.error_rate:
                raise Exception(f"Mock error (call #{self.call_count})")

        # Update metrics
        self._metrics["call_count"] = self.call_count
        self._metrics["avg_latency"] = self.latency if self.latency > 0 else 0.01

        # If a mock Completion object was set, return it directly
        if hasattr(self, "_mock_completion") and self._mock_completion:
            return self._mock_completion

        # Use responses list if available
        if self.responses:
            text = self.responses[min(self.response_index, len(self.responses) - 1)]
            self.response_index += 1
        else:
            text = self.response_text

        return Completion(
            text=text,
            usage=Usage(
                tokens=TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                ),
                latency=0.01,
                provider=self.name,
                model=self.model,
            ),
            metadata={"mock": True, "call_number": self.call_count},
            finish_reason="stop",
            model=self.model,
            provider=self.name,
        )

    async def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate mock embeddings."""
        # Simple mock embedding - just return random-looking vectors
        embeddings = []
        for text in texts:
            # Generate deterministic "embedding" based on text
            hash_val = hash(text)
            embedding = [(hash_val >> i) & 0xFF for i in range(0, 64, 8)]
            # Normalize to [-1, 1]
            embedding = [(x - 128) / 128 for x in embedding]
            embeddings.append(embedding)

        return embeddings

    def supports_streaming(self) -> bool:
        """Mock provider supports streaming."""
        return True

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any):
        """Stream mock response token by token."""
        # Get the full response
        completion = await self.complete(messages, **kwargs)

        # Stream it token by token
        tokens = completion.text.split()
        for token in tokens:
            yield token + " "
            await asyncio.sleep(0.01)  # Simulate streaming delay
