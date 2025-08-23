"""Google (Gemini) provider implementation for LogiLLM.

This is a placeholder/stub implementation for the Google provider.
To use Google's Gemini models, install the google-generativeai package:
pip install logillm[google]
"""

from typing import Any

from ..core.types import Completion
from .base import Provider


class GoogleProvider(Provider):
    """Google Gemini provider (stub implementation).

    This is a placeholder for the Google provider. The full implementation
    requires the google-generativeai package to be installed.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-pro", **kwargs: Any):
        """Initialize Google provider.

        Args:
            api_key: Google API key
            model: Model name (e.g., "gemini-pro", "gemini-1.5-pro")
            **kwargs: Additional configuration
        """
        super().__init__(provider_name="google", model=model, **kwargs)
        self.api_key = api_key

        # Note: In a full implementation, this would initialize the google-generativeai client
        raise NotImplementedError(
            "Google provider requires the google-generativeai package. "
            "Install it with: pip install logillm[google]"
        )

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """Complete a conversation (not implemented).

        Args:
            messages: Conversation history
            **kwargs: Additional parameters

        Returns:
            Completion result

        Raises:
            NotImplementedError: Always (this is a stub)
        """
        raise NotImplementedError("Google provider not yet implemented")

    def complete_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """Synchronous completion (not implemented).

        Args:
            messages: Conversation history
            **kwargs: Additional parameters

        Returns:
            Completion result

        Raises:
            NotImplementedError: Always (this is a stub)
        """
        raise NotImplementedError("Google provider not yet implemented")


__all__ = ["GoogleProvider"]
