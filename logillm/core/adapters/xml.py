"""
XML adapter for structured markup output.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

from ..types import AdapterFormat
from .base import BaseAdapter, ParseError


class XMLAdapter(BaseAdapter):
    """
    Adapter for XML-formatted output.

    Formats prompts requesting XML-structured responses
    and parses XML into structured data.
    """

    def __init__(self):
        # Set format type
        self.format_type = AdapterFormat.XML

    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """Format prompt requesting XML output."""
        lines = []

        # Instructions
        if hasattr(signature, "instructions") and signature.instructions:
            lines.append(f"Task: {signature.instructions}")
            lines.append("")

        lines.append("Please provide your response in XML format.")
        lines.append("")

        # Field descriptions
        if hasattr(signature, "input_fields"):
            lines.append("Input fields:")
            for name, field in signature.input_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                lines.append(f"- {name}: {desc}")
            lines.append("")

        if hasattr(signature, "output_fields"):
            lines.append("Output fields (provide as XML elements):")
            for name, field in signature.output_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                lines.append(f"- <{name}>: {desc}")
            lines.append("")

        # Demonstrations
        if demos:
            lines.append("Examples:")
            for i, demo in enumerate(demos, 1):
                lines.append(f"\nExample {i}:")
                lines.append("Input:")
                lines.append("<input>")
                for key, value in demo.get("inputs", {}).items():
                    lines.append(f"  <{key}>{self._escape_xml(str(value))}</{key}>")
                lines.append("</input>")
                lines.append("Output:")
                lines.append("<output>")
                for key, value in demo.get("outputs", {}).items():
                    lines.append(f"  <{key}>{self._escape_xml(str(value))}</{key}>")
                lines.append("</output>")
            lines.append("")

        # Current input
        lines.append("Now process this input:")
        lines.append("<input>")
        for key, value in inputs.items():
            lines.append(f"  <{key}>{self._escape_xml(str(value))}</{key}>")
        lines.append("</input>")
        lines.append("")
        lines.append("Provide your output in XML format with a root <output> element:")

        return "\n".join(lines)

    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """Parse XML response into structured output."""
        parsed = {}

        # Extract XML from response
        response = response.strip()

        # Look for XML block markers
        if "```xml" in response:
            start = response.find("```xml") + 6
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        # Try to find root element if not present
        if not response.startswith("<"):
            # Look for first XML tag
            match = re.search(r"<\w+>", response)
            if match:
                response = response[match.start() :]

        # Ensure we have a root element
        if not response.startswith("<output>") and not response.startswith("<?xml"):
            # Check if there are field elements without root
            if hasattr(signature, "output_fields"):
                field_pattern = "|".join(f"<{field}" for field in signature.output_fields)
                if re.search(field_pattern, response):
                    response = f"<output>{response}</output>"

        # Parse XML
        try:
            # Handle XML declaration if present
            if response.startswith("<?xml"):
                # Find where actual content starts
                content_start = response.find("?>")
                if content_start > 0:
                    response = response[content_start + 2 :].strip()

            root = ET.fromstring(response)

            # Extract fields from XML
            if hasattr(signature, "output_fields"):
                for field_name in signature.output_fields:
                    # Try direct child element
                    elem = root.find(field_name)
                    if elem is not None:
                        parsed[field_name] = elem.text or ""
                    else:
                        # Try searching recursively
                        elem = root.find(f".//{field_name}")
                        if elem is not None:
                            parsed[field_name] = elem.text or ""
            else:
                # Extract all child elements
                for child in root:
                    parsed[child.tag] = child.text or ""

        except ET.ParseError as e:
            # Try to repair XML
            repaired = self._repair_xml(response)
            if repaired:
                try:
                    root = ET.fromstring(repaired)
                    if hasattr(signature, "output_fields"):
                        for field_name in signature.output_fields:
                            elem = root.find(f".//{field_name}")
                            if elem is not None:
                                parsed[field_name] = elem.text or ""
                except ET.ParseError:
                    pass

            if not parsed:
                # Fall back to regex extraction
                if hasattr(signature, "output_fields"):
                    for field_name in signature.output_fields:
                        pattern = rf"<{field_name}>(.+?)</{field_name}>"
                        match = re.search(pattern, response, re.DOTALL)
                        if match:
                            parsed[field_name] = match.group(1).strip()

            if not parsed:
                raise ParseError("XMLAdapter", response, f"Invalid XML: {e}") from e

        if not parsed:
            raise ParseError("XMLAdapter", response, "Could not extract any output fields")

        return parsed

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def _repair_xml(self, text: str) -> Optional[str]:
        """Attempt to repair common XML issues."""
        # Close unclosed tags
        import re

        # Find all opening tags
        opening_tags = re.findall(r"<(\w+)>", text)
        # Find all closing tags
        closing_tags = re.findall(r"</(\w+)>", text)

        # Add missing closing tags
        for tag in opening_tags:
            if tag not in closing_tags:
                text += f"</{tag}>"

        # Ensure root element
        if not text.startswith("<output>"):
            text = f"<output>{text}</output>"

        return text
