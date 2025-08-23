"""
Adapter chain for fallback parsing.

ZERO DEPENDENCIES - Uses only Python standard library.
"""

from typing import Any, Optional

from .base import BaseAdapter, ParseError


class AdapterChain:
    """
    Chain multiple adapters with fallback logic.

    Tries each adapter in sequence until one succeeds.
    """

    def __init__(self, adapters: list[BaseAdapter]):
        """
        Initialize with a list of adapters to try in order.

        Args:
            adapters: List of adapter instances
        """
        if not adapters:
            raise ValueError("AdapterChain requires at least one adapter")
        self.adapters = adapters
        self.primary_adapter = adapters[0]

    def format_prompt(
        self, signature, inputs: dict[str, Any], demos: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """
        Format prompt using the primary adapter.

        Args:
            signature: The signature defining fields
            inputs: Input values
            demos: Optional demonstrations

        Returns:
            Formatted prompt string
        """
        return self.primary_adapter.format_prompt(signature, inputs, demos)

    def parse_response(self, signature, response: str) -> dict[str, Any]:
        """
        Try each adapter until one successfully parses the response.

        Args:
            signature: The signature defining expected fields
            response: Raw LLM response

        Returns:
            Parsed output dictionary

        Raises:
            ParseError: If all adapters fail
        """
        errors = []

        for adapter in self.adapters:
            try:
                parsed = adapter.parse_response(signature, response)

                # Validate if we have a validate method
                if hasattr(adapter, "validate_output"):
                    is_valid, error_msg = adapter.validate_output(signature, parsed)
                    if not is_valid:
                        errors.append(f"{adapter.__class__.__name__}: {error_msg}")
                        continue

                return parsed

            except ParseError as e:
                errors.append(f"{e.adapter_name}: {e.reason}")
            except Exception as e:
                errors.append(f"{adapter.__class__.__name__}: {str(e)}")

        # All adapters failed
        error_summary = "\n".join(errors)
        raise ParseError("AdapterChain", response, f"All adapters failed:\n{error_summary}")

    def set_primary(self, adapter_type: str):
        """
        Set the primary adapter by type name.

        Args:
            adapter_type: Name of adapter type (e.g., 'json', 'chat', 'markdown')
        """
        for i, adapter in enumerate(self.adapters):
            if hasattr(adapter, "format_type"):
                format_value = getattr(adapter.format_type, "value", None)
                if format_value == adapter_type.lower():
                    # Move to front
                    self.adapters.pop(i)
                    self.adapters.insert(0, adapter)
                    self.primary_adapter = adapter
                    return

        raise ValueError(f"No adapter found with type: {adapter_type}")
