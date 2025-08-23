"""Provider implementations for various LLM services.

This package contains provider implementations for different LLM services,
each as an optional dependency:

- OpenAI: Install with `pip install logillm[openai]`
- Anthropic: Install with `pip install logillm[anthropic]`
- Google: Install with `pip install logillm[google]`

Each provider implements the base Provider interface and can be used
interchangeably in LogiLLM modules.
"""

from typing import TYPE_CHECKING

# Always available
from .base import Provider, ProviderError, RateLimitError, TimeoutError
from .mock import MockProvider
from .registry import create_provider, get_provider, list_providers, register_provider

# Optional providers with lazy imports
if TYPE_CHECKING:
    from .anthropic import AnthropicProvider
    from .google import GoogleProvider
    from .openai import OpenAIProvider


def load_openai_provider():
    """Lazy load OpenAI provider."""
    try:
        from .openai import OpenAIProvider

        return OpenAIProvider
    except ImportError as e:
        raise ImportError(
            "OpenAI provider requires the 'openai' package. "
            "Install it with: pip install logillm[openai]"
        ) from e


def load_anthropic_provider():
    """Lazy load Anthropic provider."""
    try:
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    except ImportError as e:
        raise ImportError(
            "Anthropic provider requires the 'anthropic' package. "
            "Install it with: pip install logillm[anthropic]"
        ) from e


def load_google_provider():
    """Lazy load Google provider."""
    try:
        from .google import GoogleProvider

        return GoogleProvider
    except ImportError as e:
        raise ImportError(
            "Google provider requires the 'google-generativeai' package. "
            "Install it with: pip install logillm[google]"
        ) from e


__all__ = [
    # Base classes
    "Provider",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    # Mock provider (always available)
    "MockProvider",
    # Registry functions
    "get_provider",
    "register_provider",
    "list_providers",
    "create_provider",
    # Lazy loaders
    "load_openai_provider",
    "load_anthropic_provider",
    "load_google_provider",
]
