"""Model capabilities registry for LogiLLM providers.

This module tracks which models support various features like vision, audio, and function calling.
Updated for August 2025 models.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelCapabilities:
    """Capabilities of a specific model."""

    vision: bool = False
    audio: bool = False
    function_calling: bool = False
    streaming: bool = True
    max_tokens: int | None = None
    context_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert capabilities to dictionary."""
        return {
            "vision": self.vision,
            "audio": self.audio,
            "function_calling": self.function_calling,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
        }


# Model capabilities registry - August 2025 models
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # OpenAI GPT-4.1 family (August 2025 release)
    "gpt-4.1": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=128000,
        context_window=128000,
    ),
    "gpt-4.1-mini": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=16384,
        context_window=128000,
    ),
    "gpt-4.1-preview": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=128000,
        context_window=128000,
    ),
    # OpenAI GPT-4o family (multimodal)
    "gpt-4o": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=128000,
        context_window=128000,
    ),
    "gpt-4o-mini": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=16384,
        context_window=128000,
    ),
    "gpt-4o-audio-preview": ModelCapabilities(
        vision=True,
        audio=True,  # Audio support
        function_calling=True,
        streaming=True,
        max_tokens=16384,
        context_window=128000,
    ),
    # OpenAI GPT-4 legacy models
    "gpt-4": ModelCapabilities(
        vision=False,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=8192,
        context_window=8192,
    ),
    "gpt-4-turbo": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=128000,
    ),
    "gpt-4-vision-preview": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=128000,
    ),
    # OpenAI GPT-3.5 models (legacy, not recommended)
    "gpt-3.5-turbo": ModelCapabilities(
        vision=False,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=16385,
    ),
    # Anthropic Claude 4 family (May 2025 release)
    "claude-4-opus-20250514": ModelCapabilities(
        vision=True,
        audio=False,  # Anthropic doesn't support audio yet
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=200000,
    ),
    "claude-4-sonnet-20250514": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=200000,
    ),
    "claude-4-haiku-20250514": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=200000,
    ),
    # Anthropic Claude 3 family (legacy)
    "claude-3-opus-20240229": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=200000,
    ),
    "claude-3-sonnet-20240229": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=200000,
    ),
    "claude-3-haiku-20240307": ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=200000,
    ),
    # Google Gemini models
    "gemini-2.5-flash": ModelCapabilities(
        vision=True,
        audio=True,
        function_calling=True,
        streaming=True,
        max_tokens=8192,
        context_window=1048576,  # 1M tokens
    ),
    "gemini-1.5-pro": ModelCapabilities(
        vision=True,
        audio=True,
        function_calling=True,
        streaming=True,
        max_tokens=8192,
        context_window=1048576,
    ),
    "gemini-1.5-flash": ModelCapabilities(
        vision=True,
        audio=True,
        function_calling=True,
        streaming=True,
        max_tokens=8192,
        context_window=1048576,
    ),
}


def get_model_capabilities(model: str) -> ModelCapabilities | None:
    """Get capabilities for a specific model.

    Args:
        model: Model identifier

    Returns:
        ModelCapabilities if found, None otherwise
    """
    return MODEL_CAPABILITIES.get(model)


def supports_vision(model: str) -> bool:
    """Check if a model supports vision/image input.

    Args:
        model: Model identifier

    Returns:
        True if model supports vision, False otherwise
    """
    caps = get_model_capabilities(model)
    return caps.vision if caps else False


def supports_audio(model: str) -> bool:
    """Check if a model supports audio input.

    Args:
        model: Model identifier

    Returns:
        True if model supports audio, False otherwise
    """
    caps = get_model_capabilities(model)
    return caps.audio if caps else False


def supports_function_calling(model: str) -> bool:
    """Check if a model supports function calling.

    Args:
        model: Model identifier

    Returns:
        True if model supports function calling, False otherwise
    """
    caps = get_model_capabilities(model)
    return caps.function_calling if caps else False


def get_max_tokens(model: str) -> int | None:
    """Get maximum token limit for a model.

    Args:
        model: Model identifier

    Returns:
        Maximum tokens if known, None otherwise
    """
    caps = get_model_capabilities(model)
    return caps.max_tokens if caps else None


def get_context_window(model: str) -> int | None:
    """Get context window size for a model.

    Args:
        model: Model identifier

    Returns:
        Context window size if known, None otherwise
    """
    caps = get_model_capabilities(model)
    return caps.context_window if caps else None
