"""Advanced tool registry for managing and discovering tools."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools with categories and discovery."""

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: dict[str, Tool] = {}
        self.categories: dict[str, set[str]] = defaultdict(set)
        self.tags: dict[str, set[str]] = defaultdict(set)
        self._tool_metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool: Tool,
        category: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool with optional categorization.

        Args:
            tool: Tool instance to register
            category: Optional category for the tool
            tags: Optional tags for discovery
            metadata: Optional additional metadata

        Raises:
            ValueError: If tool name already exists
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self.tools[tool.name] = tool

        if category:
            self.categories[category].add(tool.name)

        if tags:
            for tag in tags:
                self.tags[tag].add(tool.name)

        if metadata:
            self._tool_metadata[tool.name] = metadata

        logger.debug(f"Registered tool '{tool.name}' in category '{category}' with tags {tags}")

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of tool to unregister
        """
        if name not in self.tools:
            logger.warning(f"Tool '{name}' not found for unregistration")
            return

        # Remove from main registry
        del self.tools[name]

        # Remove from categories
        for _category, tool_names in self.categories.items():
            tool_names.discard(name)

        # Remove from tags
        for _tag, tool_names in self.tags.items():
            tool_names.discard(name)

        # Remove metadata
        self._tool_metadata.pop(name, None)

        logger.debug(f"Unregistered tool '{name}'")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def list_tools(self, category: str | None = None, tag: str | None = None) -> list[str]:
        """List tool names, optionally filtered by category or tag.

        Args:
            category: Optional category filter
            tag: Optional tag filter

        Returns:
            List of tool names matching filters
        """
        if category and tag:
            # Intersection of category and tag
            category_tools = self.categories.get(category, set())
            tag_tools = self.tags.get(tag, set())
            return list(category_tools & tag_tools)
        elif category:
            return list(self.categories.get(category, set()))
        elif tag:
            return list(self.tags.get(tag, set()))
        else:
            return list(self.tools.keys())

    def list_categories(self) -> list[str]:
        """List all categories.

        Returns:
            List of category names
        """
        return list(self.categories.keys())

    def list_tags(self) -> list[str]:
        """List all tags.

        Returns:
            List of tag names
        """
        return list(self.tags.keys())

    def get_category_tools(self, category: str) -> list[Tool]:
        """Get all tools in a category.

        Args:
            category: Category name

        Returns:
            List of Tool instances in the category
        """
        tool_names = self.categories.get(category, set())
        return [self.tools[name] for name in tool_names if name in self.tools]

    def get_tagged_tools(self, tag: str) -> list[Tool]:
        """Get all tools with a specific tag.

        Args:
            tag: Tag name

        Returns:
            List of Tool instances with the tag
        """
        tool_names = self.tags.get(tag, set())
        return [self.tools[name] for name in tool_names if name in self.tools]

    def search_tools(self, query: str) -> list[Tool]:
        """Search for tools by name or description.

        Args:
            query: Search query string

        Returns:
            List of Tool instances matching the query
        """
        query_lower = query.lower()
        matches = []

        for tool in self.tools.values():
            # Search in name
            if query_lower in tool.name.lower():
                matches.append(tool)
                continue

            # Search in description
            if tool.desc and query_lower in tool.desc.lower():
                matches.append(tool)
                continue

        return matches

    def get_descriptions(self) -> dict[str, str]:
        """Get tool descriptions for prompt generation.

        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {name: tool.desc or "No description" for name, tool in self.tools.items()}

    def format_for_prompt(self, category: str | None = None, tag: str | None = None) -> str:
        """Format tools for inclusion in prompts.

        Args:
            category: Optional category filter
            tag: Optional tag filter

        Returns:
            Formatted string describing available tools
        """
        tool_names = self.list_tools(category=category, tag=tag)
        if not tool_names:
            return "No tools available."

        lines = ["Available tools:"]
        for name in sorted(tool_names):
            tool = self.tools[name]
            lines.append(f"- {name}: {tool.desc or 'No description'}")

        return "\n".join(lines)

    def format_for_llm(
        self, category: str | None = None, tag: str | None = None
    ) -> list[dict[str, Any]]:
        """Format tools for LLM function calling.

        Args:
            category: Optional category filter
            tag: Optional tag filter

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        tool_names = self.list_tools(category=category, tag=tag)
        schemas = []

        for name in tool_names:
            tool = self.tools[name]
            schemas.append(tool.format_for_llm())

        return schemas

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Arguments for the tool

        Returns:
            ToolResult with execution outcome
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                output=None, success=False, error=f"Tool '{tool_name}' not found in registry"
            )

        return await tool.execute(**kwargs)

    def execute_sync(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool synchronously by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Arguments for the tool

        Returns:
            ToolResult with execution outcome
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                output=None, success=False, error=f"Tool '{tool_name}' not found in registry"
            )

        return tool.execute_sync(**kwargs)

    def clear(self, category: str | None = None) -> None:
        """Clear tools from registry.

        Args:
            category: Optional category to clear (clears all if None)
        """
        if category:
            # Clear specific category
            tool_names = list(self.categories.get(category, set()))
            for name in tool_names:
                self.unregister(name)
        else:
            # Clear everything
            self.tools.clear()
            self.categories.clear()
            self.tags.clear()
            self._tool_metadata.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_tools": len(self.tools),
            "categories": len(self.categories),
            "tags": len(self.tags),
            "tools_by_category": {cat: len(tools) for cat, tools in self.categories.items()},
            "tools_by_tag": {tag: len(tools) for tag, tools in self.tags.items()},
        }

    def validate(self) -> list[str]:
        """Validate registry consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check that all categorized tools exist
        for category, tool_names in self.categories.items():
            for name in tool_names:
                if name not in self.tools:
                    errors.append(f"Category '{category}' references non-existent tool '{name}'")

        # Check that all tagged tools exist
        for tag, tool_names in self.tags.items():
            for name in tool_names:
                if name not in self.tools:
                    errors.append(f"Tag '{tag}' references non-existent tool '{name}'")

        # Check that metadata tools exist
        for name in self._tool_metadata:
            if name not in self.tools:
                errors.append(f"Metadata exists for non-existent tool '{name}'")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for serialization.

        Returns:
            Dictionary representation of the registry
        """
        return {
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "categories": {cat: list(tools) for cat, tools in self.categories.items()},
            "tags": {tag: list(tools) for tag, tools in self.tags.items()},
            "metadata": self._tool_metadata,
        }

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self.tools

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self.tools.keys())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self.tools)}, categories={len(self.categories)})"


def create_default_registry() -> ToolRegistry:
    """Create a default tool registry with common tools.

    Returns:
        ToolRegistry with basic tools registered
    """
    from .base import CALCULATOR_TOOL, SEARCH_TOOL

    registry = ToolRegistry()

    # Register common tools
    registry.register(
        CALCULATOR_TOOL,
        category="math",
        tags=["calculation", "arithmetic"],
        metadata={"priority": "high", "safe": True},
    )

    registry.register(
        SEARCH_TOOL,
        category="web",
        tags=["search", "information"],
        metadata={"priority": "medium", "requires_api": True},
    )

    return registry


# Global default registry instance
default_registry = create_default_registry()


__all__ = [
    "ToolRegistry",
    "create_default_registry",
    "default_registry",
]
