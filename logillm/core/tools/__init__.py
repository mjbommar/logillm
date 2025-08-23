"""Tool system for LogiLLM agents."""

from .base import (
    CALCULATOR_TOOL,
    SEARCH_TOOL,
    Tool,
    ToolResult,
    calculator_tool,
    search_tool,
    tool,
)
from .registry import ToolRegistry, create_default_registry, default_registry

__all__ = [
    # Base classes
    "Tool",
    "ToolResult",
    "tool",
    # Registry
    "ToolRegistry",
    "create_default_registry",
    "default_registry",
    # Common tools
    "search_tool",
    "calculator_tool",
    "SEARCH_TOOL",
    "CALCULATOR_TOOL",
]
