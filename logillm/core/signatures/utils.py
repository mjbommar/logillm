"""Validation helpers and utility functions for signatures."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Union, get_args, get_origin

from ...exceptions import ValidationError
from ..types import FieldType, FieldValue
from .spec import FieldSpec


def coerce_value_to_spec(value: Any, spec: FieldSpec) -> FieldValue:
    """Coerce value to match field specification."""
    if value is None:
        # Handle both FieldSpec and Pydantic FieldInfo
        is_required = (
            spec.required
            if hasattr(spec, "required")
            else (spec.is_required() if hasattr(spec, "is_required") else True)
        )
        if not is_required:
            return None
        raise ValueError("Required field cannot be None")

    # Type coercion based on Python type hints
    # Handle both FieldSpec (has python_type) and Pydantic FieldInfo (has annotation)
    target_type = (
        spec.python_type if hasattr(spec, "python_type") else getattr(spec, "annotation", str)
    )

    # Handle basic types
    if target_type is str and not isinstance(value, str):
        return str(value)
    elif target_type is int and not isinstance(value, int):
        if isinstance(value, str) and value.isdigit():
            return int(value)
        elif isinstance(value, float):
            return int(value)
        raise ValueError(f"Cannot coerce {type(value)} to int")
    elif target_type is float and not isinstance(value, float):
        if isinstance(value, int):
            return float(value)
        elif isinstance(value, str):
            # Try regular float parsing first
            try:
                return float(value)
            except (ValueError, TypeError):
                # If that fails and it contains '/', try fraction
                if "/" in value:
                    try:
                        frac = Fraction(value.strip())
                        return float(frac)
                    except (ValueError, TypeError):
                        pass
                raise ValueError(f"Cannot parse '{value}' as float") from None
        raise ValueError(f"Cannot coerce {type(value)} to float")
    elif target_type is bool and not isinstance(value, bool):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    elif target_type is Fraction:
        if isinstance(value, Fraction):
            return value
        elif isinstance(value, str):
            import re

            # Clean up the string
            cleaned = value.strip()

            # Handle LaTeX fractions like \frac{1}{2} or \(\frac{1}{2}\)
            latex_match = re.search(r"\\frac\{(\d+)\}\{(\d+)\}", cleaned)
            if latex_match:
                numerator = int(latex_match.group(1))
                denominator = int(latex_match.group(2))
                return Fraction(numerator, denominator)

            # Extract fraction from text like "The answer is 1/6"
            fraction_match = re.search(r"(\d+)\s*/\s*(\d+)", cleaned)
            if fraction_match:
                numerator = int(fraction_match.group(1))
                denominator = int(fraction_match.group(2))
                return Fraction(numerator, denominator)

            # Try direct parsing (handles "1/36")
            try:
                return Fraction(cleaned)
            except (ValueError, TypeError):
                # If that fails, try float then convert to Fraction
                try:
                    # Extract decimal from text
                    decimal_match = re.search(r"\d+\.?\d*", cleaned)
                    if decimal_match:
                        float_val = float(decimal_match.group())
                        return Fraction(float_val).limit_denominator(1000)
                    else:
                        float_val = float(cleaned)
                        return Fraction(float_val).limit_denominator(1000)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot parse '{value}' as Fraction") from e
        elif isinstance(value, (int, float)):
            return Fraction(value)
        raise ValueError(f"Cannot coerce {type(value)} to Fraction")

    # Handle generic types (List, Dict, Optional, Union)
    origin = get_origin(target_type)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value)}")
        # Could add element type validation here
        return value
    elif origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")
        return value
    elif origin is Union:
        # Handle Optional and Union types
        args = get_args(target_type)
        if len(args) == 2 and type(None) in args:
            # Optional type
            non_none_type = next(arg for arg in args if arg is not type(None))
            if value is None:
                return None
            return coerce_value_to_spec(value, FieldSpec(spec.name, spec.field_type, non_none_type))

    # Apply constraints if any
    apply_constraints(value, spec)

    return value


def apply_constraints(value: Any, spec: FieldSpec) -> None:
    """Apply field constraints validation."""
    # Handle both FieldSpec (has constraints) and Pydantic FieldInfo (may not have)
    constraints = getattr(spec, "constraints", {})

    if "min_length" in constraints and hasattr(value, "__len__"):
        if len(value) < constraints["min_length"]:
            raise ValueError(f"Length {len(value)} < minimum {constraints['min_length']}")

    if "max_length" in constraints and hasattr(value, "__len__"):
        if len(value) > constraints["max_length"]:
            raise ValueError(f"Length {len(value)} > maximum {constraints['max_length']}")

    if "min_value" in constraints and isinstance(value, (int, float)):
        if value < constraints["min_value"]:
            raise ValueError(f"Value {value} < minimum {constraints['min_value']}")

    if "max_value" in constraints and isinstance(value, (int, float)):
        if value > constraints["max_value"]:
            raise ValueError(f"Value {value} > maximum {constraints['max_value']}")

    if "pattern" in constraints and isinstance(value, str):
        pattern = constraints["pattern"]
        if not re.match(pattern, value):
            raise ValueError(f"Value '{value}' does not match pattern '{pattern}'")

    if "choices" in constraints:
        if value not in constraints["choices"]:
            raise ValueError(f"Value '{value}' not in allowed choices: {constraints['choices']}")


def validate_signature_fields(
    input_fields: dict[str, FieldSpec], output_fields: dict[str, FieldSpec]
) -> list[str]:
    """Get signature validation errors."""
    errors = []

    # Check for duplicate field names between inputs and outputs
    input_names = set(input_fields.keys())
    output_names = set(output_fields.keys())
    duplicates = input_names & output_names
    if duplicates:
        errors.append(f"Duplicate field names in inputs and outputs: {duplicates}")

    # Validate individual field specs
    all_fields = {**input_fields, **output_fields}
    for field_name, _spec in all_fields.items():
        if not field_name.isidentifier():
            errors.append(f"Invalid field name: '{field_name}'")

    if errors:
        raise ValidationError(
            f"Signature validation failed: {'; '.join(errors)}", context={"errors": errors}
        )

    return errors


def make_field_spec_from_dict(name: str, field_data: dict[str, Any]) -> FieldSpec:
    """Helper to reconstruct FieldSpec from dictionary data."""
    # Map string type names back to types
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }
    python_type = type_map.get(field_data["python_type"], str)

    return FieldSpec(
        name=name,
        field_type=FieldType(field_data["field_type"]),
        python_type=python_type,
        description=field_data.get("description"),
        required=field_data.get("required", True),
        default=field_data.get("default"),
        constraints=field_data.get("constraints", {}),
        metadata=field_data.get("metadata", {}),
    )


def infer_type_from_value(value: Any) -> type:
    """Infer Python type from a value."""
    if isinstance(value, bool):
        return bool
    elif isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, list):
        return list
    elif isinstance(value, dict):
        return dict
    elif isinstance(value, Fraction):
        return Fraction
    else:
        return str


__all__ = [
    "coerce_value_to_spec",
    "apply_constraints",
    "validate_signature_fields",
    "make_field_spec_from_dict",
    "infer_type_from_value",
]
