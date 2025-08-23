"""Enhanced signature class with seamless Pydantic/pure Python support."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..types import FieldType
from .base import BaseSignature
from .factory import SignatureBase, SignatureMeta
from .fields import (
    PYDANTIC_AVAILABLE,
    FieldDescriptor,
    get_field_desc,
    get_field_type,
)
from .spec import FieldSpec

# Import Pydantic if available
if PYDANTIC_AVAILABLE:
    from pydantic.fields import FieldInfo


class Signature(SignatureBase, metaclass=SignatureMeta):  # type: ignore[misc]
    """Enhanced signature with transparent Pydantic/pure Python support.

    Can be used in three ways:
    1. String syntax: Signature("input -> output")
    2. Class definition with fields
    3. Dictionary of field definitions

    Works identically whether Pydantic is installed or not.
    """

    @classmethod
    def validate_inputs(cls, **inputs: Any) -> dict[str, Any]:
        """Validate input values against the signature class fields."""
        validated = {}
        for name, field in cls.input_fields.items():
            if name in inputs:
                # For now, just pass through - could add type checking later
                validated[name] = inputs[name]
            elif hasattr(field, "default") and field.default is not None:
                validated[name] = field.default
            elif hasattr(field, "is_required") and field.is_required():
                raise ValueError(f"Required input field '{name}' not provided")
        return validated

    @classmethod
    def validate_outputs(cls, **outputs: Any) -> dict[str, Any]:
        """Validate output values against the signature class fields."""
        validated = {}
        for name, field in cls.output_fields.items():
            if name in outputs:
                # For now, just pass through - could add type checking later
                validated[name] = outputs[name]
            elif hasattr(field, "default") and field.default is not None:
                validated[name] = field.default
        return validated

    @classmethod
    def validate(cls) -> bool:
        """Check if the signature is valid."""
        return bool(cls.input_fields or cls.output_fields)

    @classmethod
    def to_dict(cls) -> dict:
        """Convert signature to dictionary representation."""
        return {
            "instructions": cls.instructions if hasattr(cls, "instructions") else cls.__doc__,
            "input_fields": {name: str(field) for name, field in cls.input_fields.items()},
            "output_fields": {name: str(field) for name, field in cls.output_fields.items()},
        }

    @classmethod
    def with_instructions(cls, instructions: str) -> type[Signature]:
        """Create a new signature class with updated instructions."""
        # Create a copy with new instructions
        if hasattr(cls, "_field_definitions"):
            # Pure Python mode
            fields = cls._field_definitions.copy()
        elif hasattr(cls, "model_fields"):
            # Pydantic mode
            fields = {}
            for name, field in cls.model_fields.items():
                fields[name] = (field.annotation, field)
        else:
            fields = {}

        from .factory import make_signature

        return make_signature(fields, instructions=instructions, signature_name=cls.__name__)  # type: ignore[return-value]

    @classmethod
    def with_updated_field(cls, name: str, **updates) -> type[Signature]:
        """Update a field's metadata."""
        # Create a copy with updated field
        if hasattr(cls, "_field_definitions"):
            # Pure Python mode
            fields = deepcopy(cls._field_definitions)
            if name in fields:
                for key, value in updates.items():
                    setattr(fields[name], key, value)
        elif hasattr(cls, "model_fields"):
            # Pydantic mode
            fields = deepcopy(cls.model_fields)
            if name in fields:
                field = fields[name]
                if field.json_schema_extra is None:
                    field.json_schema_extra = {}
                field.json_schema_extra.update(updates)
        else:
            raise ValueError(f"Field '{name}' not found")

        from .factory import make_signature

        return make_signature(fields, instructions=cls.instructions, signature_name=cls.__name__)  # type: ignore[return-value]

    def to_base_signature(self) -> BaseSignature:
        """Convert to BaseSignature for compatibility."""
        input_specs = {}
        output_specs = {}

        # Get fields based on mode
        if hasattr(self.__class__, "_field_definitions"):
            # Pure Python mode
            field_source = self.__class__._field_definitions
        elif hasattr(self.__class__, "model_fields"):
            # Pydantic mode
            field_source = self.__class__.model_fields
        else:
            field_source = {}

        for name, field_def in field_source.items():
            field_type = get_field_type(field_def)
            desc = get_field_desc(field_def)

            # Get annotation
            if isinstance(field_def, FieldDescriptor):
                annotation = field_def.annotation or str
                required = field_def.is_required()
                default = field_def.default
            elif PYDANTIC_AVAILABLE and isinstance(field_def, FieldInfo):
                annotation = field_def.annotation if hasattr(field_def, "annotation") else str
                required = field_def.is_required() if hasattr(field_def, "is_required") else True
                default = field_def.default if hasattr(field_def, "default") else None
            else:
                annotation = str
                required = True
                default = None

            spec = FieldSpec(
                name=name,
                field_type=FieldType.INPUT if field_type == "input" else FieldType.OUTPUT,
                python_type=annotation,
                description=desc,
                required=required,
                default=default,
            )

            if field_type == "input":
                input_specs[name] = spec
            else:
                output_specs[name] = spec

        return BaseSignature(
            input_fields=input_specs,
            output_fields=output_specs,
            instructions=self.__class__.instructions,
        )


# Keep the old name for compatibility
EnhancedSignature = Signature


def ensure_signature(
    signature: str | type[Signature] | None, instructions: str | None = None
) -> type[Signature] | None:
    """Ensure we have a signature class."""
    if signature is None:
        return None

    if isinstance(signature, str):
        from .factory import make_signature

        return make_signature(signature, instructions)  # type: ignore[return-value]

    if instructions is not None:
        raise ValueError("Cannot specify instructions with an existing signature class")

    return signature


__all__ = [
    "Signature",
    "EnhancedSignature",  # Alias for compatibility
    "ensure_signature",
]
