"""
Adapter system for converting between signatures and LLM formats.

Each adapter handles a specific format (chat, markdown, JSON, XML, etc.)
and lives in its own module for better organization.
"""

from .base import AdapterError, BaseAdapter, ParseError
from .chain import AdapterChain
from .chat import ChatAdapter
from .json_adapter import JSONAdapter
from .markdown import MarkdownAdapter
from .text import TextAdapter
from .xml import XMLAdapter

# Alias for consistency
Adapter = BaseAdapter


def create_adapter(format_type: str | object = "chat"):
    """
    Create an adapter by format type.

    Args:
        format_type: Format string or AdapterFormat enum

    Returns:
        Adapter instance
    """
    # Handle AdapterFormat enum
    if hasattr(format_type, "value"):
        format_type = format_type.value

    # Convert to string and lowercase
    format_type = str(format_type).lower()

    adapters = {
        "chat": ChatAdapter,
        "json": JSONAdapter,
        "markdown": MarkdownAdapter,
        "xml": XMLAdapter,
        "text": TextAdapter,
    }

    adapter_class = adapters.get(format_type, ChatAdapter)
    return adapter_class()


__all__ = [
    "Adapter",
    "BaseAdapter",
    "AdapterError",
    "ParseError",
    "ChatAdapter",
    "JSONAdapter",
    "MarkdownAdapter",
    "TextAdapter",
    "XMLAdapter",
    "AdapterChain",
    "create_adapter",
]
