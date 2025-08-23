"""Provider abstractions for LM communication.

This module re-exports the provider functionality from the providers package
for backward compatibility.
"""

# Re-export everything from the new providers package
# Also re-export the lazy loaders
from ..providers import (
    MockProvider,
    Provider,
    ProviderError,
    RateLimitError,
    TimeoutError,
    create_provider,
    get_provider,
    list_providers,
    load_anthropic_provider,
    load_google_provider,
    load_openai_provider,
    register_provider,
)

__all__ = [
    "Provider",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "MockProvider",
    "get_provider",
    "register_provider",
    "list_providers",
    "create_provider",
    "load_openai_provider",
    "load_anthropic_provider",
    "load_google_provider",
]
