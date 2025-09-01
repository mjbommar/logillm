"""
Chat adapter for conversational message format.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

import json
from typing import Any, Optional, Union

from ..types import AdapterFormat
from .base import BaseAdapter, ParseError
from .utils import parse_key_value_pairs


class ChatAdapter(BaseAdapter):
    """
    Adapter for chat/conversational format.

    Formats prompts as conversation messages and parses
    responses from chat completions.
    """

    def __init__(self):
        # Set format type for compatibility
        self.format_type = AdapterFormat.CHAT

    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> Union[str, list[dict[str, Any]]]:
        """Format as chat messages, supporting multimodal content."""
        from ..signatures.types import Audio, History, Image

        # Check if inputs contain multimodal content
        has_multimodal = False
        for value in inputs.values():
            if isinstance(value, (Image, Audio, History)):
                has_multimodal = True
                break
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (Image, Audio)):
                        has_multimodal = True
                        break

        # Build messages list
        messages = []

        # System message with instructions
        if hasattr(signature, "instructions") and signature.instructions:
            system_msg = f"Task: {signature.instructions}\n\n"
        else:
            system_msg = (
                "Transform the inputs to outputs according to the field specifications below.\n\n"
            )

        # Add field descriptions
        if hasattr(signature, "input_fields"):
            system_msg += "Input fields:\n"
            for name, field in signature.input_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                system_msg += f"- {name}: {desc}\n"
            system_msg += "\n"

        if hasattr(signature, "output_fields"):
            system_msg += "Output fields (provide these in your response):\n"
            for name, field in signature.output_fields.items():
                desc = getattr(field, "desc", f"The {name}")
                # Add type hints for numeric fields
                from fractions import Fraction

                # Handle both FieldSpec (python_type) and Pydantic FieldInfo (annotation)
                field_type = getattr(field, "python_type", None) or getattr(
                    field, "annotation", None
                )
                if field_type:
                    if field_type is float:
                        pass  # Don't add confusing hints for float
                    elif field_type is int:
                        desc += " (provide as an integer)"
                    elif field_type is Fraction:
                        desc += " (provide ONLY as a fraction like '1/6' or '3/4', no other text)"
                system_msg += f"- {name}: {desc}\n"

        messages.append({"role": "system", "content": system_msg})

        # Add demonstrations if provided
        if demos:
            for demo in demos:
                # Demo input
                demo_input = "Input:\n"
                for key, value in demo.get("inputs", {}).items():
                    demo_input += f"{key}: {value}\n"
                messages.append({"role": "user", "content": demo_input})

                # Demo output
                demo_output = "Output:\n"
                for key, value in demo.get("outputs", {}).items():
                    demo_output += f"{key}: {value}\n"
                messages.append({"role": "assistant", "content": demo_output})

        # Handle current input
        if has_multimodal:
            # Build multimodal user message
            content_parts = []
            content_parts.append("Input:\n")

            for key, value in inputs.items():
                if isinstance(value, History):
                    # History should be converted to messages
                    if hasattr(value, "to_messages"):
                        return value.to_messages()
                    else:
                        content_parts.append(f"{key}: {str(value)}\n")
                elif isinstance(value, (Image, Audio)):
                    # Add label then the media object
                    content_parts.append(f"{key}: ")
                    content_parts.append(value)
                    content_parts.append("\n")
                elif isinstance(value, list):
                    content_parts.append(f"{key}:\n")
                    for item in value:
                        if isinstance(item, (Image, Audio)):
                            content_parts.append(item)
                        else:
                            content_parts.append(str(item))
                        content_parts.append("\n")
                else:
                    content_parts.append(f"{key}: {value}\n")

            messages.append({"role": "user", "content": content_parts})

            # Return messages directly for multimodal
            return messages
        else:
            # Add current input as string
            user_msg = "Input:\n"
            for key, value in inputs.items():
                user_msg += f"{key}: {value}\n"
            messages.append({"role": "user", "content": user_msg})

        # Convert to string format for non-multimodal
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"[{role}]: {content}\n\n"

        return prompt

    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """Parse chat response into structured output."""
        parsed = {}
        response = response.strip()

        # First, try to parse as JSON if it looks like JSON
        if response.startswith("{") and response.endswith("}"):
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass  # Not valid JSON, continue with other methods

        # Try to parse as key-value pairs using our utility
        kv_pairs = parse_key_value_pairs(response)
        if kv_pairs and hasattr(signature, "output_fields"):
            # Map parsed keys to expected output fields (case-insensitive)
            for field_name in signature.output_fields:
                for key, value in kv_pairs.items():
                    if field_name.lower() == key.lower() and value:  # Only use non-empty values
                        # Check if the field is supposed to be a list
                        field_spec = signature.output_fields.get(field_name)
                        if field_spec and hasattr(field_spec, "python_type"):
                            # Check if it's a list type
                            from typing import get_origin
                            origin = get_origin(field_spec.python_type)
                            if origin is list or field_spec.python_type is list:
                                # Parse the value as a list
                                value = self._parse_list_value(value)
                        parsed[field_name] = value
                        break

        # If no fields parsed, try to extract from free text
        if not parsed and hasattr(signature, "output_fields"):
            # Simple heuristic: look for field names in the text
            import re

            # First pass: Find all field positions
            field_positions = {}
            for field_name in signature.output_fields:
                # Look for patterns like "field_name:" or "- field_name:"
                patterns = [
                    rf"(?:^|\n|\s|-)?\s*{field_name}\s*:\s*",
                    rf"(?:^|\n)\s*{field_name}\s+",
                ]

                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        field_positions[field_name] = match.end()
                        break

            # Sort fields by position
            sorted_fields = sorted(field_positions.items(), key=lambda x: x[1])

            # Extract content between field markers
            for i, (field_name, start_pos) in enumerate(sorted_fields):
                # Find end position (start of next field or end of response)
                if i + 1 < len(sorted_fields):
                    end_pos = (
                        sorted_fields[i + 1][1] - len(sorted_fields[i + 1][0]) - 3
                    )  # Adjust for field name and colon
                    # Find the actual start of the next field
                    next_field = sorted_fields[i + 1][0]
                    next_match = re.search(
                        rf"(?:^|\n|\s|-)?\s*{next_field}\s*:\s*",
                        response[start_pos:],
                        re.IGNORECASE,
                    )
                    if next_match:
                        end_pos = start_pos + next_match.start()
                else:
                    end_pos = len(response)

                content = response[start_pos:end_pos].strip()

                # Clean up the content (remove trailing dashes, etc.)
                content = re.sub(r"[-\s]*$", "", content).strip()

                # Type conversion based on field type
                field_info = signature.output_fields[field_name]
                field_type = None

                # Get the type annotation
                if hasattr(field_info, "annotation"):
                    field_type = field_info.annotation
                elif hasattr(field_info, "python_type"):
                    field_type = field_info.python_type

                if field_type:
                    # Check if it's a list type
                    from typing import get_origin
                    origin = get_origin(field_type)
                    if origin is list or field_type is list:
                        content = self._parse_list_value(content)
                    elif field_type is float:
                        # Try to extract a number
                        number_match = re.search(r"(\d+(?:\.\d+)?)", content)
                        if number_match:
                            try:
                                content = float(number_match.group(1))
                            except ValueError:
                                pass
                        # Fallback: Convert confidence words to numbers
                        elif isinstance(content, str):
                            confidence_map = {
                                "very high": 0.95,
                                "extremely high": 0.99,
                                "high": 0.85,
                                "medium": 0.5,
                                "moderate": 0.5,
                                "low": 0.15,
                                "very low": 0.05,
                                "extremely low": 0.01,
                            }
                            content_lower = content.lower().strip()
                            if content_lower in confidence_map:
                                content = confidence_map[content_lower]
                    elif field_type is int:
                        # Try to extract an integer
                        number_match = re.search(r"(\d+)", content)
                        if number_match:
                            try:
                                content = int(number_match.group(1))
                            except ValueError:
                                pass

                parsed[field_name] = content

            # Fallback: If no fields found, try simpler patterns
            if not parsed:
                for field_name in signature.output_fields:
                    # Try multiple patterns to find the field
                    patterns = [
                        # Field followed by colon and capture until next field or newline
                        rf"{field_name}\s*:\s*([^:\n]+?)(?:\n|$|(?:\s*\w+\s*:))",
                        # Field at start of line
                        rf"^\s*{field_name}\s+([^:\n]+)",
                        # Field as a header (like "Steps to...")
                        rf"{field_name}\s+to\s+([^:\n]+)",
                        # Field followed by "is" (e.g., "The answer is 4")
                        rf"(?:the\s+)?{field_name}\s+is\s+([^,\.\n]+)",
                        # Field with value (e.g., "4 with high confidence")
                        rf"with\s+(?:high\s+|low\s+|medium\s+)?{field_name}",
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            if len(match.groups()) > 0:
                                content = match.group(1).strip()
                            else:
                                # For patterns without capture groups (like "with high confidence")
                                # Extract the confidence level
                                if "confidence" in field_name.lower():
                                    if "high" in response.lower():
                                        content = "high"
                                    elif "low" in response.lower():
                                        content = "low"
                                    elif "medium" in response.lower():
                                        content = "medium"
                                    else:
                                        content = response
                                else:
                                    content = response
                            parsed[field_name] = content
                            break

        # If still no fields parsed and there's only one output field, use the whole response
        if not parsed and hasattr(signature, "output_fields"):
            output_fields = list(signature.output_fields.keys())
            if len(output_fields) == 1:
                field_name = output_fields[0]
                # Check if it's a list field
                field_spec = signature.output_fields.get(field_name)
                value = response
                if field_spec and hasattr(field_spec, "python_type"):
                    # Check if it's a list type
                    from typing import get_origin
                    origin = get_origin(field_spec.python_type)
                    if origin is list or field_spec.python_type is list:
                        value = self._parse_list_value(response)
                # Single output field - assign entire response (or parsed list)
                parsed[field_name] = value
            elif not parsed:
                raise ParseError("ChatAdapter", response, "Could not extract any output fields")

        # Special handling for ChainOfThought: if we found answer but not reasoning,
        # the reasoning is everything before the answer
        if hasattr(signature, "output_fields") and "reasoning" in signature.output_fields:
            if "answer" in parsed and "reasoning" not in parsed:
                # Find where answer starts in the response
                import re

                answer_patterns = [
                    rf"answer\s*:\s*{re.escape(str(parsed['answer']))}",
                    rf"-\s*answer\s*:\s*{re.escape(str(parsed['answer']))}",
                ]
                for pattern in answer_patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        # Everything before the answer is reasoning
                        reasoning_text = response[: match.start()].strip()
                        if reasoning_text:
                            parsed["reasoning"] = reasoning_text
                        break

                # If we still don't have reasoning, use the whole response except answer
                if "reasoning" not in parsed:
                    parsed["reasoning"] = response.replace(
                        str(parsed.get("answer", "")), ""
                    ).strip()

        return parsed

    def _parse_list_value(self, value: str) -> Any:
        """Parse a string value as a list if possible."""
        if not isinstance(value, str):
            return value

        value = value.strip()

        # Handle empty or None-like strings
        if not value or value.lower() in ["none", "n/a", "[]"]:
            return []

        # Try to parse as JSON array
        if value.startswith("[") and value.endswith("]"):
            try:
                import json

                result = json.loads(value)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to parse as bullet list (- item or * item or • item or number. item)
        import re

        bullet_pattern = r"^\s*(?:[-*•]|\d+\.)\s+(.+)$"
        lines = value.split("\n")
        if any(re.match(bullet_pattern, line) for line in lines if line.strip()):
            items = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(bullet_pattern, line)
                if match:
                    items.append(match.group(1).strip())
                elif items:  # continuation of previous item
                    items[-1] += " " + line
            return items if items else []

        # Try to parse as semicolon-separated list (common for complex items)
        if ";" in value and "," not in value:
            items = [item.strip() for item in value.split(";") if item.strip()]
            if len(items) > 1:
                return items

        # Try to parse as comma-separated list
        # Handle formats like "[apple, banana]" or "apple, banana"
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]  # Remove brackets

        # Check if it looks like a single item or multiple
        if "," in value or " and " in value:
            # Split by comma first
            items = []
            if "," in value:
                parts = value.split(",")
                for part in parts:
                    # Handle "and" within comma-separated items
                    if " and " in part and len(parts) == 1:
                        # This is likely a list with "and" as separator
                        items.extend([item.strip().strip("\"'") for item in part.split(" and ")])
                    else:
                        items.append(part.strip().strip("\"'"))
            else:
                # Just "and" separator
                items = [item.strip().strip("\"'") for item in value.split(" and ")]

            # Filter empty items
            items = [item for item in items if item]
            return items if items else []

        # Single item - return as single-item list
        return [value] if value else []
