"""
Markdown adapter for formatted text output.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

import re
from typing import Any, Optional

from ..types import AdapterFormat
from .base import BaseAdapter, ParseError


class MarkdownAdapter(BaseAdapter):
    """
    Adapter for Markdown-formatted output.

    Formats prompts requesting Markdown-structured responses
    and parses Markdown into structured data.
    """

    def __init__(self):
        # Set format type
        self.format_type = AdapterFormat.MARKDOWN

    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """Format prompt requesting Markdown output."""
        lines = []

        # Instructions
        if hasattr(signature, "instructions") and signature.instructions:
            lines.append(f"# Task\n{signature.instructions}\n")

        lines.append("Please provide your response in Markdown format with clear sections.\n")

        # Field descriptions
        if hasattr(signature, "input_fields"):
            lines.append("## Input Fields")
            for name, field in signature.input_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                lines.append(f"- **{name}**: {desc}")
            lines.append("")

        if hasattr(signature, "output_fields"):
            lines.append("## Required Output Sections")
            lines.append("Please structure your response with these sections:")
            for name, field in signature.output_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                lines.append(f"- **{name}**: {desc}")
            lines.append("")

        # Demonstrations
        if demos:
            lines.append("## Examples\n")
            for i, demo in enumerate(demos, 1):
                lines.append(f"### Example {i}\n")
                lines.append("**Input:**")
                for key, value in demo.get("inputs", {}).items():
                    lines.append(f"- {key}: {value}")
                lines.append("\n**Output:**")
                for key, value in demo.get("outputs", {}).items():
                    lines.append(f"\n#### {key}")
                    lines.append(f"{value}")
                lines.append("")

        # Current input
        lines.append("## Your Task\n")
        lines.append("**Input:**")
        for key, value in inputs.items():
            lines.append(f"- {key}: {value}")
        lines.append(
            "\nPlease provide your response with clear section headers for each output field:"
        )

        return "\n".join(lines)

    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """Parse Markdown response into structured output."""
        parsed = {}

        if not hasattr(signature, "output_fields"):
            raise ParseError("MarkdownAdapter", response, "No output fields defined")

        # Try to extract sections based on headers
        lines = response.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            # Check for headers (##, ###, ####, or bold text)
            header_match = re.match(r"^#{1,4}\s+(.+)$", line)
            bold_match = re.match(r"^\*\*(.+?)\*\*:?\s*(.*)$", line)

            if header_match:
                # Save previous section if exists
                if current_section and current_content:
                    content = "\n".join(current_content).strip()
                    self._assign_to_field(parsed, current_section, content, signature)

                # Start new section
                current_section = header_match.group(1).strip()
                current_content = []

            elif bold_match:
                section_name = bold_match.group(1).strip()
                section_value = bold_match.group(2).strip()

                # Check if this matches an output field
                for field_name in signature.output_fields:
                    if field_name.lower() == section_name.lower():
                        if section_value:
                            parsed[field_name] = section_value
                        else:
                            # Start collecting content for this field
                            if current_section and current_content:
                                content = "\n".join(current_content).strip()
                                self._assign_to_field(parsed, current_section, content, signature)
                            current_section = section_name
                            current_content = []
                        break
            else:
                # Accumulate content for current section
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section and current_content:
            content = "\n".join(current_content).strip()
            self._assign_to_field(parsed, current_section, content, signature)

        # If no structured sections found, try to extract based on patterns
        if not parsed:
            for field_name in signature.output_fields:
                # Look for "field_name: value" patterns
                pattern = rf"{field_name}:?\s*(.+?)(?:\n|$)"
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    parsed[field_name] = match.group(1).strip()

        if not parsed:
            raise ParseError("MarkdownAdapter", response, "Could not extract any output fields")

        return parsed

    def _assign_to_field(self, parsed: dict, section: str, content: str, signature):
        """Assign section content to the appropriate field."""
        # Clean up content
        content = content.strip()
        if not content:
            return

        # Remove markdown artifacts
        content = re.sub(r"^[-*]\s+", "", content, flags=re.MULTILINE)
        content = content.strip()

        # Match section name to field name
        section_lower = section.lower()
        for field_name in signature.output_fields:
            if field_name.lower() in section_lower or section_lower in field_name.lower():
                parsed[field_name] = content
                break
