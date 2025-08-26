"""String parsing utilities for signatures - handles all parsing cases."""

from __future__ import annotations

import inspect
import typing
from fractions import Fraction
from typing import Any, Callable, get_type_hints

from ...exceptions import SignatureError
from ..types import FieldType
from .base import BaseSignature
from .spec import FieldSpec


def parse_signature_string(
    signature_str: str, custom_types: dict[str, type] | None = None
) -> BaseSignature:
    """Parse signature from string format with full type support.

    Supports:
    - Basic syntax: "input -> output"
    - Typed syntax: "input: str -> output: int"
    - Generic types: "items: list[str] -> summary: str"
    - Optional types: "query: str -> result: Optional[dict]"
    - Union types: "data: str | bytes -> processed: bool"
    - Multiple fields: "a: int, b: float -> sum: float, product: float"
    - Custom types: "image: Image -> caption: str"

    Args:
        signature_str: Signature string to parse
        custom_types: Optional dictionary of custom type mappings

    Returns:
        BaseSignature with parsed fields
    """
    # Validate arrow syntax
    if signature_str.count("->") != 1:
        raise SignatureError(
            f"Invalid signature format: '{signature_str}'. Expected exactly one '->'",
            context={"signature_str": signature_str},
        )

    # Split into input and output parts
    inputs_str, outputs_str = signature_str.split("->")
    inputs_str = inputs_str.strip()
    outputs_str = outputs_str.strip()

    # Parse field definitions with enhanced type support
    input_fields = _parse_field_list(inputs_str, FieldType.INPUT, custom_types)
    output_fields = _parse_field_list(outputs_str, FieldType.OUTPUT, custom_types)

    return BaseSignature(
        input_fields=input_fields,
        output_fields=output_fields,
    )


def _parse_field_list(
    fields_str: str, field_type: FieldType, custom_types: dict[str, type] | None = None
) -> dict[str, FieldSpec]:
    """Parse a comma-separated list of fields with complex type support.

    Handles complex cases like:
    - "a, b, c" (simple names)
    - "a: int, b: str" (typed)
    - "items: list[str], mapping: dict[str, int]" (generics with commas)
    - "result: tuple[int, str, bool]" (nested commas)
    """
    fields = {}
    if not fields_str.strip():
        return fields

    # Use a more sophisticated tokenizer that handles nested brackets
    tokens = _tokenize_field_list(fields_str)

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Check for type annotation
        if ":" in token:
            # Find the first colon not inside brackets
            name, type_str = _split_on_colon(token)
            name = name.strip()
            type_str = type_str.strip()

            # Validate field name
            if not name.isidentifier():
                raise SignatureError(f"Invalid field name: '{name}'", field_name=name)

            # Parse the type expression
            python_type = _parse_type_expression(type_str, custom_types)
        else:
            # No type annotation, use str as default
            name = token
            if not name.isidentifier():
                raise SignatureError(f"Invalid field name: '{name}'", field_name=name)
            python_type = str

        fields[name] = FieldSpec(
            name=name,
            field_type=field_type,
            python_type=python_type,
        )

    return fields


def _tokenize_field_list(field_str: str) -> list[str]:
    """Tokenize a field list, respecting bracket nesting.

    Splits on commas, but not commas inside brackets.
    Example: "a: list[str], b: dict[str, int]" -> ["a: list[str]", "b: dict[str, int]"]
    """
    tokens = []
    current_token = ""
    bracket_depth = 0

    for char in field_str:
        if char == "[":
            bracket_depth += 1
            current_token += char
        elif char == "]":
            bracket_depth -= 1
            current_token += char
        elif char == "," and bracket_depth == 0:
            # This comma is a field separator
            tokens.append(current_token)
            current_token = ""
        else:
            current_token += char

    # Don't forget the last token
    if current_token:
        tokens.append(current_token)

    return tokens


def _split_on_colon(field_def: str) -> tuple[str, str]:
    """Split a field definition on the first colon not inside brackets.

    Example: "items: list[dict[str, int]]" -> ("items", "list[dict[str, int]]")
    """
    bracket_depth = 0

    for i, char in enumerate(field_def):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == ":" and bracket_depth == 0:
            return field_def[:i], field_def[i + 1 :]

    raise ValueError(f"No colon found in field definition: '{field_def}'")


def _parse_type_expression(expr: str, custom_types: dict[str, type] | None = None) -> type:
    """Parse a complex type expression string.

    Supports:
    - Basic types: str, int, float, bool
    - Generic types: list[str], dict[str, int]
    - Optional types: Optional[str], str | None
    - Union types: Union[str, int], str | int
    - Custom types from provided dictionary
    - Multimodal types: Image, Audio, Tool, History
    """
    # Import multimodal types if available
    try:
        from .types import Audio, History, Image, Tool

        multimodal_types = {
            "Image": Image,
            "Audio": Audio,
            "Tool": Tool,
            "History": History,
        }
    except ImportError:
        multimodal_types = {}

    # Prepare namespace with all typing constructs and custom types
    namespace = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "bytes": bytes,
        "bytearray": bytearray,
        "complex": complex,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "fraction": Fraction,
        "Fraction": Fraction,
        "List": list,
        "Dict": dict,
        "Tuple": tuple,
        "Set": set,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "Any": typing.Any,
        "None": type(None),
        **multimodal_types,
    }

    if custom_types:
        namespace.update(custom_types)

    # Clean the expression
    expr = expr.strip()

    # Handle simple types directly
    if expr in namespace:
        return namespace[expr]

    # Handle Union shorthand (|)
    if "|" in expr and not expr.startswith("Union["):
        # Convert "str | int" to "Union[str, int]"
        parts = [part.strip() for part in expr.split("|")]
        expr = f"Union[{', '.join(parts)}]"

    # Try to evaluate the expression safely
    try:
        # Use eval with restricted namespace for safety
        result = eval(expr, {"__builtins__": {}}, namespace)
        return result
    except (SyntaxError, NameError, TypeError):
        # Fall back to string type if parsing fails
        import warnings

        warnings.warn(
            f"Could not parse type expression '{expr}'. Using str as fallback.", stacklevel=2
        )
        return str


def signature_from_function(func: Callable) -> BaseSignature:
    """Create signature from function type annotations."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    input_fields = {}
    output_fields = {}

    # Process parameters as input fields
    for param_name, param in sig.parameters.items():
        if param_name in {"self", "cls"}:
            continue

        python_type = type_hints.get(param_name, str)
        required = param.default == param.empty
        default = None if param.default == param.empty else param.default

        input_fields[param_name] = FieldSpec(
            name=param_name,
            field_type=FieldType.INPUT,
            python_type=python_type,
            required=required,
            default=default,
        )

    # Process return annotation as output field
    return_type = type_hints.get("return", str)
    if return_type is not type(None):
        output_fields["output"] = FieldSpec(
            name="output",
            field_type=FieldType.OUTPUT,
            python_type=return_type,
        )

    return BaseSignature(
        input_fields=input_fields,
        output_fields=output_fields,
        instructions=func.__doc__,
    )


def infer_signature_from_examples(
    examples: list[dict[str, dict[str, Any]]], custom_types: dict[str, type] | None = None
) -> dict[str, FieldSpec]:
    """Infer a signature from example inputs and outputs.

    Args:
        examples: List of examples with "input" and "output" dicts
        custom_types: Optional custom type mappings

    Returns:
        Dictionary mapping field names to FieldSpec objects
    """
    if not examples:
        raise ValueError("Cannot infer signature from empty examples")

    # Collect all field names and their types across examples
    input_types = {}
    output_types = {}

    for example in examples:
        # Process input fields
        if "input" in example:
            for name, value in example["input"].items():
                inferred_type = _infer_type_from_value(value)
                if name in input_types:
                    # Check for type consistency
                    if input_types[name] != inferred_type:
                        # Use Any if types differ
                        input_types[name] = typing.Any
                else:
                    input_types[name] = inferred_type

        # Process output fields
        if "output" in example:
            for name, value in example["output"].items():
                inferred_type = _infer_type_from_value(value)
                if name in output_types:
                    # Check for type consistency
                    if output_types[name] != inferred_type:
                        output_types[name] = typing.Any
                else:
                    output_types[name] = inferred_type

    # Build field specs
    fields = {}
    for name, python_type in input_types.items():
        fields[name] = FieldSpec(
            name=name,
            field_type=FieldType.INPUT,
            python_type=python_type,
            description="Inferred from examples",
        )
    for name, python_type in output_types.items():
        fields[name] = FieldSpec(
            name=name,
            field_type=FieldType.OUTPUT,
            python_type=python_type,
            description="Inferred from examples",
        )

    return fields


def _infer_type_from_value(value: Any) -> type:
    """Infer the type from a runtime value.

    Useful for dynamic signature generation from examples.
    """
    if value is None:
        return type(None)

    value_type = type(value)

    # Handle collections with element types
    if isinstance(value, list) and value:
        element_types = {type(v) for v in value}
        if len(element_types) == 1:
            element_type = element_types.pop()
            return list[element_type]

    if isinstance(value, dict) and value:
        key_types = {type(k) for k in value.keys()}
        val_types = {type(v) for v in value.values()}
        if len(key_types) == 1 and len(val_types) == 1:
            key_type = key_types.pop()
            val_type = val_types.pop()
            return dict[key_type, val_type]

    # Check for multimodal types if available
    try:
        from .types import Audio, History, Image, Tool

        if isinstance(value, Image):
            return Image
        if isinstance(value, Audio):
            return Audio
        if isinstance(value, Tool):
            return Tool
        if isinstance(value, History):
            return History
    except ImportError:
        pass

    return value_type


def signature_to_string(fields: dict[str, FieldSpec]) -> str:
    """Convert a signature fields dict back to string representation.

    Args:
        fields: Dictionary mapping field names to FieldSpec objects

    Returns:
        String representation like "a: int, b: str -> c: float"
    """
    input_parts = []
    output_parts = []

    for name, spec in fields.items():
        # Format type hint
        type_hint = spec.python_type
        if hasattr(type_hint, "__name__"):
            type_str = type_hint.__name__
        else:
            type_str = str(type_hint)
            # Clean up typing module prefixes
            type_str = type_str.replace("typing.", "")

        field_str = f"{name}: {type_str}"

        # Determine if input or output
        if spec.field_type == FieldType.INPUT:
            input_parts.append(field_str)
        else:
            output_parts.append(field_str)

    input_str = ", ".join(input_parts) if input_parts else ""
    output_str = ", ".join(output_parts) if output_parts else ""

    return f"{input_str} -> {output_str}"


__all__ = [
    "parse_signature_string",
    "signature_from_function",
    "infer_signature_from_examples",
    "signature_to_string",
]
