"""
Base adapter class and exceptions.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class AdapterError(Exception):
    """Base exception for adapter errors."""

    pass


class ParseError(AdapterError):
    """Raised when parsing fails."""

    def __init__(self, adapter_name: str, text: str, reason: str):
        self.adapter_name = adapter_name
        self.text = text
        self.reason = reason
        super().__init__(f"{adapter_name} parse failed: {reason}")


class BaseAdapter(ABC):
    """
    Base class for all adapters.

    Each adapter handles conversion between a specific format
    (chat, JSON, XML, Markdown, etc.) and LogiLLM's internal representation.
    """

    format_type = None  # Will be set in subclasses

    @abstractmethod
    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """
        Format the prompt for the LLM.

        Args:
            signature: The signature defining input/output fields
            inputs: Input values to include in the prompt
            demos: Optional demonstrations to include

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """
        Parse the LLM response into structured output.

        Args:
            signature: The signature defining expected output fields
            response: Raw LLM response text

        Returns:
            Dictionary with parsed output fields

        Raises:
            ParseError: If response cannot be parsed
        """
        pass

    def validate_output(self, signature, parsed: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parsed output against signature.

        Args:
            signature: The signature defining expected fields
            parsed: Parsed output dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        expected_fields = set(signature.output_fields.keys())
        parsed_fields = set(parsed.keys())

        missing = expected_fields - parsed_fields
        if missing:
            return False, f"Missing required fields: {missing}"

        return True, None

    # Backward compatibility methods
    def format(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> list[dict[str, str]]:
        """
        Legacy format method for backward compatibility.

        Returns messages in OpenAI format.
        """
        prompt = self.format_prompt(signature, inputs, demos)

        # Convert to message format
        messages = []

        # Add system message if signature has instructions
        if hasattr(signature, "instructions") and signature.instructions:
            messages.append({"role": "system", "content": signature.instructions})

        # Add the formatted prompt as user message
        messages.append({"role": "user", "content": prompt})

        return messages

    def parse(
        self, signature, completion: str, inputs: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Legacy parse method for backward compatibility.
        """
        return self.parse_response(signature, completion)
