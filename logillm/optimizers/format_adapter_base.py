"""
Minimal adapter base classes for FormatOptimizer.

This module contains the minimal Adapter/Formatter/Parser base classes
needed by FormatOptimizer's hybrid adapters. This allows FormatOptimizer
to remain self-contained without depending on legacy adapter code.
"""

from abc import ABC, abstractmethod
from typing import Any


class Formatter(ABC):
    """Base formatter for prompt generation."""

    @abstractmethod
    def format(self, signature, inputs, demos=None) -> str:
        """Format inputs into a prompt string."""
        pass


class Parser(ABC):
    """Base parser for response extraction."""

    @abstractmethod
    def parse(self, completion: str, signature) -> dict[str, Any]:
        """Parse completion into structured output."""
        pass


class Adapter:
    """Base adapter combining formatting and parsing.

    This is not an abstract base class - it provides default implementations
    that can be overridden by subclasses.
    """

    def __init__(self, format_type=None, **kwargs):
        """Initialize adapter with format type."""
        self.format_type = format_type
        self.formatter = self._create_formatter() if hasattr(self, "_create_formatter") else None
        self.parser = self._create_parser() if hasattr(self, "_create_parser") else None

    def format(self, signature, inputs, demos=None):
        """Format inputs into messages."""
        if self.formatter:
            prompt = self.formatter.format(signature, inputs, demos)
            return self._prompt_to_messages(prompt)
        return []

    def parse(self, completion, signature):
        """Parse completion into structured output."""
        if self.parser:
            return self.parser.parse(completion, signature)
        return {}

    def _prompt_to_messages(self, prompt: str) -> list[dict[str, str]]:
        """Convert prompt string to messages format."""
        return [{"role": "user", "content": prompt}]


class HybridMarkdownJSONFormatter(Formatter):
    """Formatter for Markdown structure with JSON output request."""

    def format(self, signature, inputs, demos=None) -> str:
        """Format as Markdown with JSON output instruction."""
        lines = []

        # Title
        lines.append(f"# Task: {signature.instructions or 'Complete the following'}\n")

        # Input fields
        lines.append("## Input Data\n")
        for field_name, value in inputs.items():
            lines.append(f"**{field_name}**: {value}")
        lines.append("")

        # Output instruction
        lines.append("## Expected Output\n")
        lines.append("Please provide your response as a JSON object with these fields:")
        for field_name in signature.output_fields:
            lines.append(f"- `{field_name}`")
        lines.append("\n```json\n{\n  // Your response here\n}\n```")

        return "\n".join(lines)


class HybridMarkdownJSONParser(Parser):
    """Parser for JSON embedded in Markdown."""

    def parse(self, completion: str, signature) -> dict[str, Any]:
        """Extract JSON from Markdown response."""
        import json
        import re

        # Look for JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", completion, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find raw JSON
        json_match = re.search(r"\{.*?\}", completion, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: Create dict with first output field
        if signature.output_fields:
            field_name = list(signature.output_fields.keys())[0]
            return {field_name: completion.strip()}

        return {}
