"""Format adapters for optimizing prompt formats.

Adapters convert between different prompt formats (JSON, Markdown, XML)
to find the most effective format for a given task.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class FormatAdapter(ABC):
    """Base class for format adapters."""

    @abstractmethod
    def format_prompt(self, inputs: dict[str, Any], signature: Any) -> str:
        """Format inputs into a prompt string.

        Args:
            inputs: Input fields and values
            signature: Signature defining expected I/O

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response: str, signature: Any) -> dict[str, Any]:
        """Parse response string into output fields.

        Args:
            response: Raw response string from LLM
            signature: Signature defining expected outputs

        Returns:
            Dictionary of parsed output fields
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get adapter name."""
        pass

    @property
    def format_type(self) -> str:
        """Get format type identifier."""
        return self.name.lower().replace("adapter", "")


class ChatAdapter(FormatAdapter):
    """Standard chat format adapter (default)."""

    def format_prompt(self, inputs: dict[str, Any], signature: Any) -> str:
        """Format as natural language prompt."""
        # Extract field descriptions from signature if available
        input_prompts = []

        if hasattr(signature, "input_fields"):
            for field_name, field_spec in signature.input_fields.items():
                if field_name in inputs:
                    desc = field_spec.desc if hasattr(field_spec, "desc") else field_name
                    value = inputs[field_name]
                    input_prompts.append(f"{desc}: {value}")
        else:
            # Fallback to simple key-value format
            for key, value in inputs.items():
                input_prompts.append(f"{key}: {value}")

        prompt = "\n".join(input_prompts)

        # Add output field hints if available
        if hasattr(signature, "output_fields"):
            output_hints = []
            for field_name, field_spec in signature.output_fields.items():
                desc = field_spec.desc if hasattr(field_spec, "desc") else field_name
                # Add type hints for numeric fields
                if hasattr(field_spec, "annotation"):
                    if field_spec.annotation is float:
                        desc += " (provide as a decimal number between 0.0 and 1.0)"
                    elif field_spec.annotation is int:
                        desc += " (provide as an integer number)"
                output_hints.append(f"- {desc}")

            if output_hints:
                prompt += "\n\nPlease provide:\n" + "\n".join(output_hints)

        return prompt

    def parse_response(self, response: str, signature: Any) -> dict[str, Any]:
        """Parse natural language response."""
        outputs = {}

        if hasattr(signature, "output_fields"):
            # First pass: Find all field positions
            field_positions = {}
            for field_name in signature.output_fields:
                # Look for patterns like "field_name:" or "- field_name:"
                patterns = [
                    rf"(?:^|\n|\s|-)?\s*{field_name}\s*:\s*",
                    rf"(?:^|\n)#+\s*{field_name}\s*:?\s*",
                    rf"(?:^|\n)\*\*{field_name}\*\*\s*:?\s*",
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
                    # Find the actual start of the next field
                    next_field = sorted_fields[i + 1][0]
                    # Look for next field marker
                    next_patterns = [
                        rf"(?:^|\n|\s|-)?\s*{next_field}\s*:\s*",
                        rf"(?:^|\n)#+\s*{next_field}\s*:?\s*",
                        rf"(?:^|\n)\*\*{next_field}\*\*\s*:?\s*",
                    ]
                    end_pos = len(response)
                    for pattern in next_patterns:
                        next_match = re.search(pattern, response[start_pos:], re.IGNORECASE)
                        if next_match:
                            end_pos = start_pos + next_match.start()
                            break
                else:
                    end_pos = len(response)

                content = response[start_pos:end_pos].strip()

                # Clean up the content (remove trailing dashes, etc.)
                content = re.sub(r"[-\s]*$", "", content).strip()

                # Type conversion based on field type
                field_spec = signature.output_fields[field_name]
                if hasattr(field_spec, "annotation"):
                    field_type = field_spec.annotation
                    if field_type is float:
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

                outputs[field_name] = content

            # Fallback: If no fields found, try simpler patterns
            if not outputs:
                for field_name, _field_spec in signature.output_fields.items():
                    # Try multiple patterns to find the field
                    patterns = [
                        # Field followed by colon and capture until next field or newline
                        rf"{field_name}\s*:\s*([^:\n]+?)(?:\n|$|(?:\s*\w+\s*:))",
                        # Field at start of line
                        rf"^\s*{field_name}\s+([^:\n]+)",
                        # Field as a header (like "Steps to...")
                        rf"{field_name}\s+to\s+([^:\n]+)",
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            content = match.group(1).strip()
                            outputs[field_name] = content
                            break

                    # If not found, use full response for single output field
                    if field_name not in outputs and len(signature.output_fields) == 1:
                        outputs[field_name] = response.strip()
        else:
            # Fallback: return entire response
            outputs = {"response": response}

        return outputs

    @property
    def name(self) -> str:
        return "ChatAdapter"


class JSONAdapter(FormatAdapter):
    """JSON format adapter for structured I/O."""

    def format_prompt(self, inputs: dict[str, Any], signature: Any) -> str:
        """Format as JSON prompt."""
        prompt = "Input (JSON format):\n```json\n"
        prompt += json.dumps(inputs, indent=2)
        prompt += "\n```\n\n"

        # Add output schema hint
        if hasattr(signature, "output_fields"):
            output_schema = {}
            for field_name, field_spec in signature.output_fields.items():
                desc = field_spec.desc if hasattr(field_spec, "desc") else field_name
                # Infer type from field spec if available
                field_type = "string"  # default
                if hasattr(field_spec, "annotation"):
                    if field_spec.annotation is float:
                        field_type = "number"
                    elif field_spec.annotation is int:
                        field_type = "integer"
                    elif field_spec.annotation is bool:
                        field_type = "boolean"
                    elif field_spec.annotation is list:
                        field_type = "array"

                output_schema[field_name] = f"<{field_type}> - {desc}"

            prompt += "Expected output format (JSON):\n```json\n"
            prompt += json.dumps(output_schema, indent=2)
            prompt += "\n```\n\nPlease respond with valid JSON matching the schema above."
        else:
            prompt += "Please respond with valid JSON."

        return prompt

    def parse_response(self, response: str, signature: Any) -> dict[str, Any]:
        """Parse JSON response."""
        # Try to extract JSON from response
        # Look for JSON blocks first
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object/array in response
            json_match = re.search(r"(\{.*\}|\[.*\])", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

        try:
            # Parse JSON
            parsed = json.loads(json_str)

            # If parsed is a dict, return it
            if isinstance(parsed, dict):
                return parsed

            # If it's a list and we expect array output, wrap it
            if isinstance(parsed, list):
                if hasattr(signature, "output_fields"):
                    # Find array field
                    for field_name, field_spec in signature.output_fields.items():
                        if hasattr(field_spec, "annotation"):
                            if field_spec.annotation is list:
                                return {field_name: parsed}

                return {"result": parsed}

            # Otherwise wrap primitive
            return {"result": parsed}

        except json.JSONDecodeError:
            # Fallback to text parsing
            return ChatAdapter().parse_response(response, signature)

    @property
    def name(self) -> str:
        return "JSONAdapter"


class MarkdownAdapter(FormatAdapter):
    """Markdown format adapter for readable structured output."""

    def format_prompt(self, inputs: dict[str, Any], signature: Any) -> str:
        """Format as Markdown prompt."""
        prompt = "## Input\n\n"

        if hasattr(signature, "input_fields"):
            for field_name, field_spec in signature.input_fields.items():
                if field_name in inputs:
                    desc = field_spec.desc if hasattr(field_spec, "desc") else field_name
                    value = inputs[field_name]
                    prompt += f"**{desc}:**\n{value}\n\n"
        else:
            for key, value in inputs.items():
                prompt += f"**{key.replace('_', ' ').title()}:**\n{value}\n\n"

        # Add output format hint
        if hasattr(signature, "output_fields"):
            prompt += "## Expected Output\n\n"
            prompt += "Please structure your response with the following sections:\n\n"

            for field_name, field_spec in signature.output_fields.items():
                desc = field_spec.desc if hasattr(field_spec, "desc") else field_name
                prompt += f"### {field_name.replace('_', ' ').title()}\n"
                prompt += f"_{desc}_\n\n"

        return prompt

    def parse_response(self, response: str, signature: Any) -> dict[str, Any]:
        """Parse Markdown response."""
        outputs = {}

        if hasattr(signature, "output_fields"):
            for field_name, _field_spec in signature.output_fields.items():
                # Look for markdown headers
                patterns = [
                    rf"#+ {field_name.replace('_', ' ')}.*?\n(.*?)(?:^#|\Z)",
                    rf"\*\*{field_name.replace('_', ' ')}.*?\*\*:?\s*(.*?)(?:^#|^\*\*|\Z)",
                    rf"{field_name}:?\s*(.*?)(?:^#|^\*\*|\n\n|\Z)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                        # Clean up markdown formatting
                        content = re.sub(r"^[_*]+|[_*]+$", "", content)
                        outputs[field_name] = content
                        break

                # If not found and single field, use whole response
                if field_name not in outputs and len(signature.output_fields) == 1:
                    outputs[field_name] = response.strip()
        else:
            outputs = {"response": response}

        return outputs

    @property
    def name(self) -> str:
        return "MarkdownAdapter"


class XMLAdapter(FormatAdapter):
    """XML format adapter for highly structured data."""

    def format_prompt(self, inputs: dict[str, Any], signature: Any) -> str:
        """Format as XML prompt."""
        prompt = "Input (XML format):\n```xml\n<input>\n"

        for key, value in inputs.items():
            # Escape XML special characters
            escaped_value = (
                str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            prompt += f"  <{key}>{escaped_value}</{key}>\n"

        prompt += "</input>\n```\n\n"

        # Add output schema
        if hasattr(signature, "output_fields"):
            prompt += "Expected output format (XML):\n```xml\n<output>\n"

            for field_name, field_spec in signature.output_fields.items():
                desc = field_spec.desc if hasattr(field_spec, "desc") else field_name
                prompt += f"  <{field_name}><!-- {desc} --></{field_name}>\n"

            prompt += (
                "</output>\n```\n\nPlease respond with valid XML matching the structure above."
            )
        else:
            prompt += "Please respond with valid XML."

        return prompt

    def parse_response(self, response: str, signature: Any) -> dict[str, Any]:
        """Parse XML response."""
        outputs = {}

        # Extract XML from response
        xml_match = re.search(r"```(?:xml)?\s*\n(.*?)\n```", response, re.DOTALL)
        if xml_match:
            xml_str = xml_match.group(1)
        else:
            # Try to find XML in response
            xml_match = re.search(r"(<\w+>.*</\w+>)", response, re.DOTALL)
            if xml_match:
                xml_str = xml_match.group(1)
            else:
                xml_str = response

        # Parse XML fields
        if hasattr(signature, "output_fields"):
            for field_name in signature.output_fields:
                # Look for XML tags
                pattern = rf"<{field_name}>(.*?)</{field_name}>"
                match = re.search(pattern, xml_str, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # Unescape XML entities
                    content = (
                        content.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                    )
                    outputs[field_name] = content

        # If no outputs found, try generic parsing
        if not outputs:
            # Find all XML tags
            tags = re.findall(r"<(\w+)>(.*?)</\1>", xml_str, re.DOTALL)
            for tag_name, content in tags:
                outputs[tag_name] = content.strip()

        # Fallback to text parsing if no XML found
        if not outputs:
            return ChatAdapter().parse_response(response, signature)

        return outputs

    @property
    def name(self) -> str:
        return "XMLAdapter"


# Registry of available adapters
ADAPTER_REGISTRY = {
    "chat": ChatAdapter,
    "json": JSONAdapter,
    "markdown": MarkdownAdapter,
    "xml": XMLAdapter,
}


def get_adapter(name: str) -> FormatAdapter:
    """Get adapter by name.

    Args:
        name: Adapter name (chat, json, markdown, xml)

    Returns:
        Adapter instance

    Raises:
        ValueError: If adapter not found
    """
    adapter_class = ADAPTER_REGISTRY.get(name.lower())
    if not adapter_class:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTER_REGISTRY.keys())}")

    return adapter_class()


def list_adapters() -> list[str]:
    """List available adapter names."""
    return list(ADAPTER_REGISTRY.keys())
