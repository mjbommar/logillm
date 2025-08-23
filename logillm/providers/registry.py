"""Provider registry for managing available providers."""

from typing import Any

from .base import Provider, ProviderError

# Global registry of providers
_provider_registry: dict[str, Provider] = {}
_default_provider: str | None = None


def register_provider(
    provider: Provider, name: str | None = None, set_default: bool = False
) -> None:
    """Register a provider in the global registry.

    Args:
        provider: Provider instance to register
        name: Optional name override (defaults to provider.name)
        set_default: Whether to set this as the default provider
    """
    provider_name = name or provider.name
    _provider_registry[provider_name] = provider

    global _default_provider
    if set_default or _default_provider is None:
        _default_provider = provider_name


def get_provider(name: str | None = None) -> Provider:
    """Get a provider from the registry.

    Args:
        name: Provider name (uses default if None)

    Returns:
        Provider instance

    Raises:
        ProviderError: If provider not found
    """
    if name is None:
        if _default_provider is None:
            raise ProviderError(
                "No default provider set. Register a provider with set_default=True "
                "or specify a provider name."
            )
        name = _default_provider

    if name not in _provider_registry:
        raise ProviderError(
            f"Provider '{name}' not found. Available providers: {list(_provider_registry.keys())}"
        )

    return _provider_registry[name]


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(_provider_registry.keys())


def clear_registry() -> None:
    """Clear all registered providers (mainly for testing)."""
    global _default_provider
    _provider_registry.clear()
    _default_provider = None


def create_provider(provider_type: str, **kwargs: Any) -> Provider:
    """Factory function to create providers.

    Args:
        provider_type: Type of provider ("openai", "anthropic", "mock", etc.)
        **kwargs: Provider-specific configuration

    Returns:
        Provider instance

    Raises:
        ImportError: If provider package not installed
        ValueError: If provider type unknown
    """
    provider_type = provider_type.lower()

    if provider_type == "mock":
        from .mock import MockProvider

        return MockProvider(**kwargs)

    elif provider_type == "openai":
        try:
            from .openai import OpenAIProvider

            return OpenAIProvider(**kwargs)
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install logillm[openai]"
            ) from e

    elif provider_type == "anthropic":
        try:
            from .anthropic import AnthropicProvider

            return AnthropicProvider(**kwargs)
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install logillm[anthropic]"
            ) from e

    elif provider_type == "google" or provider_type == "gemini":
        try:
            from .google import GoogleProvider

            return GoogleProvider(**kwargs)
        except ImportError as e:
            raise ImportError(
                "Google provider requires the 'google-generativeai' package. "
                "Install it with: pip install logillm[google]"
            ) from e

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Available types: mock, openai, anthropic, google"
        )
