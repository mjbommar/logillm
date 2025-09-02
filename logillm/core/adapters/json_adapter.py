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
                
                # Determine JSON type from python_type
                python_type = getattr(field, "python_type", None) or getattr(field, "annotation", None)
                field_type = "string"  # default
                
                if python_type:
                    from typing import get_origin
                    origin = get_origin(python_type)
                    
                    if origin is list or python_type is list:
                        field_type = "array"
                    elif python_type is int:
                        field_type = "integer"
                    elif python_type is float:
                        field_type = "number"
                    elif python_type is bool:
                        field_type = "boolean"
                    elif python_type is dict or origin is dict:
                        field_type = "object"
                
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
        parsed = None
        if json_str:
            try:
                parsed = json.loads(json_str)
                if not isinstance(parsed, dict):
                    raise ParseError("JSONAdapter", response, f"Expected dict, got {type(parsed)}")
            except json.JSONDecodeError:
                pass

        # Try to repair the JSON if parsing failed
        if parsed is None:
            repaired = repair_json_simple(response)
            if repaired:
                if isinstance(repaired, dict):
                    parsed = repaired
                else:
                    raise ParseError(
                        "JSONAdapter", response, f"Expected dict after repair, got {type(repaired)}"
                    )
            else:
                raise ParseError("JSONAdapter", response, "Could not parse or repair JSON")
        
        # IMPORTANT: Guarantee all output fields are present
        # This is a core promise of LogiLLM signatures
        if signature and hasattr(signature, "output_fields"):
            for field_name, field_spec in signature.output_fields.items():
                if field_name not in parsed:
                    # Provide a default value based on the field type
                    if hasattr(field_spec, "python_type"):
                        field_type = field_spec.python_type
                    elif hasattr(field_spec, "annotation"):
                        field_type = field_spec.annotation
                    else:
                        field_type = str
                    
                    # Generate appropriate default based on type
                    from typing import get_origin
                    origin = get_origin(field_type)
                    
                    if origin is list or field_type is list:
                        parsed[field_name] = []
                    elif field_type is int:
                        parsed[field_name] = 0
                    elif field_type is float:
                        parsed[field_name] = 0.0
                    elif field_type is bool:
                        parsed[field_name] = False
                    elif field_type is dict:
                        parsed[field_name] = {}
                    else:
                        # Default to empty string for string and unknown types
                        parsed[field_name] = ""
        
        return parsed
