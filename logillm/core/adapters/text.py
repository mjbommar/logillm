"""
Text adapter for simple text completion.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

import re
from typing import Any, Optional

from ..types import AdapterFormat
from .base import BaseAdapter
from .utils import parse_key_value_pairs


class TextAdapter(BaseAdapter):
    """
    Fallback adapter for plain text.

    Handles simple text completion format with minimal structure.
    """

    def __init__(self):
        # Set format type
        self.format_type = AdapterFormat.CHAT  # TEXT not in enum, use CHAT as fallback

    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """Format as plain text prompt."""
        lines = []

        # Instructions
        if hasattr(signature, "instructions") and signature.instructions:
            lines.append(signature.instructions)
            lines.append("")

        # Demonstrations
        if demos:
            for demo in demos:
                # Input
                for key, value in demo.get("inputs", {}).items():
                    lines.append(f"{key}: {value}")
                lines.append("→")
                # Output
                for key, value in demo.get("outputs", {}).items():
                    lines.append(f"{key}: {value}")
                lines.append("")

        # Current input
        for key, value in inputs.items():
            lines.append(f"{key}: {value}")
        lines.append("→")

        return "\n".join(lines)

    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """Parse plain text response."""
        parsed = {}
        response = response.strip()

        if not hasattr(signature, "output_fields"):
            # If no output fields defined, return whole response
            return {"output": response}

        output_fields = list(signature.output_fields.keys())

        # Single output field - return entire response
        if len(output_fields) == 1:
            parsed[output_fields[0]] = response
            return parsed

        # Multiple fields - try to extract using patterns
        lines = response.split("\n")

        # Try key-value pattern using our utility
        kv_pairs = parse_key_value_pairs(response)
        if kv_pairs:
            for field_name in output_fields:
                for key, value in kv_pairs.items():
                    if field_name.lower() == key.lower() or key.lower() in field_name.lower():
                        parsed[field_name] = value
                        break

        # If not all fields found, try numbered pattern
        if len(parsed) < len(output_fields):
            # Look for numbered items (1. ..., 2. ..., etc.)
            numbered_items = re.findall(r"\d+\.\s+(.+?)(?=\d+\.|$)", response, re.DOTALL)
            for i, item in enumerate(numbered_items):
                if i < len(output_fields) and output_fields[i] not in parsed:
                    parsed[output_fields[i]] = item.strip()

        # If still missing fields, try bullet points
        if len(parsed) < len(output_fields):
            bullet_items = re.findall(r"[-•*]\s+(.+?)(?=[-•*]|$)", response, re.DOTALL)
            field_idx = 0
            for item in bullet_items:
                # Find next unpopulated field
                while field_idx < len(output_fields) and output_fields[field_idx] in parsed:
                    field_idx += 1
                if field_idx < len(output_fields):
                    parsed[output_fields[field_idx]] = item.strip()
                    field_idx += 1

        # Last resort - split by newlines and assign to fields
        if not parsed:
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            for i, field_name in enumerate(output_fields):
                if i < len(non_empty_lines):
                    parsed[field_name] = non_empty_lines[i]

        if not parsed:
            # Return at least something
            parsed[output_fields[0]] = response

        return parsed
