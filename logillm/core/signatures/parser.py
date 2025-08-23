"""String parsing utilities for signatures."""

from __future__ import annotations

import inspect
import re
from fractions import Fraction
from typing import Callable, get_type_hints

from ...exceptions import SignatureError
from ..types import FieldType
from .base import BaseSignature
from .spec import FieldSpec


def parse_signature_string(signature_str: str) -> BaseSignature:
    """Parse signature from string format like 'input1, input2 -> output1, output2'."""
    # Main pattern for "input1, input2 -> output1, output2"
    main_pattern = r"^(.*?)\s*->\s*(.*)$"
    match = re.match(main_pattern, signature_str.strip())

    if not match:
        raise SignatureError(
            f"Invalid signature format: '{signature_str}'. Expected format: 'input1, input2 -> output1, output2'",
            context={"signature_str": signature_str},
        )

    inputs_str, outputs_str = match.groups()

    # Parse field definitions (handle types like "field: int")
    def parse_fields(fields_str: str, field_type: FieldType) -> dict[str, FieldSpec]:
        fields = {}
        if not fields_str.strip():
            return fields

        # Split by comma and parse each field
        for field_def in fields_str.split(","):
            field_def = field_def.strip()
            if not field_def:
                continue

            # Check for type annotation "name: type"
            if ":" in field_def:
                name, type_str = field_def.split(":", 1)
                name = name.strip()
                type_str = type_str.strip()

                # Map string types to Python types
                type_map = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "fraction": Fraction,
                    "Fraction": Fraction,  # Allow both cases
                }
                python_type = type_map.get(type_str, str)
            else:
                name = field_def
                python_type = str

            if not name.isidentifier():
                raise SignatureError(f"Invalid field name: '{name}'", field_name=name)

            fields[name] = FieldSpec(
                name=name,
                field_type=field_type,
                python_type=python_type,
            )

        return fields

    input_fields = parse_fields(inputs_str, FieldType.INPUT)
    output_fields = parse_fields(outputs_str, FieldType.OUTPUT)

    return BaseSignature(
        input_fields=input_fields,
        output_fields=output_fields,
    )


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


__all__ = [
    "parse_signature_string",
    "signature_from_function",
]
