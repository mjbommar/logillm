"""
JSON adapter for structured output.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

import json
from typing import Any, Optional

from ..types import AdapterFormat
from .base import BaseAdapter, ParseError
from .utils import extract_json_object, repair_json_simple


class JSONAdapter(BaseAdapter):
    """
    Adapter for JSON structured output.

    Formats prompts requesting JSON output and parses
    JSON responses.
    """

    def __init__(self, use_structured_output: bool = True):
        self.use_structured_output = use_structured_output
        # Set format type
        self.format_type = AdapterFormat.JSON

    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """Format prompt requesting JSON output."""
        lines = []

        # Instructions
        if hasattr(signature, "instructions") and signature.instructions:
            lines.append(f"Task: {signature.instructions}")
            lines.append("")

        lines.append("You will receive inputs and must provide outputs in JSON format.")
        lines.append("")

        # Field descriptions
        if hasattr(signature, "input_fields"):
            lines.append("Input fields:")
            for name, field in signature.input_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                lines.append(f"- {name}: {desc}")
            lines.append("")

        if hasattr(signature, "output_fields"):
            lines.append("Output fields to provide in JSON:")
            for name, field in signature.output_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                field_type = getattr(field, "type_", "string")
                lines.append(f"- {name} ({field_type}): {desc}")
            lines.append("")

        # Demonstrations
        if demos:
            lines.append("Examples:")
            for i, demo in enumerate(demos, 1):
                lines.append(f"\nExample {i}:")
                lines.append("Input:")
                lines.append(json.dumps(demo.get("inputs", {}), indent=2))
                lines.append("Output:")
                lines.append(json.dumps(demo.get("outputs", {}), indent=2))
            lines.append("")

        # Current input
        lines.append("Now process this input:")
        lines.append(json.dumps(inputs, indent=2))
        lines.append("")
        lines.append("Provide your output as valid JSON:")

        return "\n".join(lines)

    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """Parse JSON response."""
        # Try to extract JSON from the response
        response = response.strip()

        # Look for JSON block markers
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        # Try to extract and parse JSON using our utilities
        json_str = extract_json_object(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
                elif not isinstance(parsed, dict):
                    raise ParseError("JSONAdapter", response, f"Expected dict, got {type(parsed)}")
            except json.JSONDecodeError:
                pass

        # Try to repair the JSON
        repaired = repair_json_simple(response)
        if repaired:
            if isinstance(repaired, dict):
                return repaired
            else:
                raise ParseError(
                    "JSONAdapter", response, f"Expected dict after repair, got {type(repaired)}"
                )

        raise ParseError("JSONAdapter", response, "Could not parse or repair JSON")
