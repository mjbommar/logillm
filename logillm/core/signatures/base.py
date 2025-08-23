"""Abstract base classes and protocols for the signature system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...exceptions import ValidationError
from ...protocols.runtime import Serializable, Validatable
from ..types import FieldValue, Metadata
from .spec import FieldSpec


class Signature(ABC, Serializable, Validatable):
    """Abstract base class for signature definitions."""

    def __init__(
        self,
        *,
        instructions: str | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        self.instructions = instructions
        self.metadata = metadata or {}
        self._fields: dict[str, FieldSpec] = {}
        self._validated = False

    @property
    @abstractmethod
    def input_fields(self) -> dict[str, FieldSpec]:
        """Get input field specifications."""
        ...

    @property
    @abstractmethod
    def output_fields(self) -> dict[str, FieldSpec]:
        """Get output field specifications."""
        ...

    @property
    def all_fields(self) -> dict[str, FieldSpec]:
        """Get all field specifications."""
        return {**self.input_fields, **self.output_fields}

    def validate_inputs(self, **inputs: Any) -> dict[str, FieldValue]:
        """Validate and coerce input values."""
        return self._validate_fields(inputs, self.input_fields, "input")

    def validate_outputs(self, **outputs: Any) -> dict[str, FieldValue]:
        """Validate and coerce output values."""
        return self._validate_fields(outputs, self.output_fields, "output")

    def _validate_fields(
        self,
        values: dict[str, Any],
        field_specs: dict[str, FieldSpec],
        field_category: str,
    ) -> dict[str, FieldValue]:
        """Internal field validation logic."""
        validated = {}
        errors = []

        # Check for missing required fields
        for field_name, spec in field_specs.items():
            # Handle both FieldSpec (has .required) and Pydantic FieldInfo (has .is_required())
            is_required = (
                spec.required
                if hasattr(spec, "required")
                else (spec.is_required() if hasattr(spec, "is_required") else True)
            )
            if is_required and field_name not in values:
                # Handle default values for both FieldSpec and Pydantic FieldInfo
                default_val = getattr(spec, "default", None)
                if default_val is not None:
                    validated[field_name] = default_val
                else:
                    errors.append(f"Required {field_category} field '{field_name}' is missing")
                continue

            if field_name in values:
                try:
                    validated[field_name] = self._coerce_value(values[field_name], spec)
                except Exception as e:
                    errors.append(f"Invalid {field_category} field '{field_name}': {e}")

        # Check for unexpected fields
        unexpected = set(values.keys()) - set(field_specs.keys())
        if unexpected:
            errors.append(f"Unexpected {field_category} fields: {', '.join(unexpected)}")

        if errors:
            raise ValidationError(
                f"Signature validation failed for {field_category} fields: {'; '.join(errors)}",
                context={"errors": errors, "field_category": field_category},
            )

        return validated

    def _coerce_value(self, value: Any, spec: FieldSpec) -> FieldValue:
        """Coerce value to match field specification."""
        from .utils import coerce_value_to_spec

        return coerce_value_to_spec(value, spec)

    def validate(self) -> bool:
        """Validate the signature definition."""
        try:
            self.validation_errors()
            return True
        except Exception:
            return False

    def validation_errors(self) -> list[str]:
        """Get signature validation errors."""
        from .utils import validate_signature_fields

        return validate_signature_fields(self.input_fields, self.output_fields)

    def to_dict(self) -> dict[str, Any]:
        """Convert signature to dictionary."""
        return {
            "type": self.__class__.__name__,
            "instructions": self.instructions,
            "metadata": self.metadata,
            "input_fields": {
                name: {
                    "field_type": spec.field_type.value,
                    "python_type": spec.python_type.__name__,
                    "description": spec.description,
                    "required": spec.required,
                    "default": spec.default,
                    "constraints": spec.constraints,
                    "metadata": spec.metadata,
                }
                for name, spec in self.input_fields.items()
            },
            "output_fields": {
                name: {
                    "field_type": spec.field_type.value,
                    "python_type": spec.python_type.__name__,
                    "description": spec.description,
                    "required": spec.required,
                    "default": spec.default,
                    "constraints": spec.constraints,
                    "metadata": spec.metadata,
                }
                for name, spec in self.output_fields.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signature:
        """Reconstruct signature from dictionary."""
        # This would need to be implemented by concrete subclasses
        raise NotImplementedError("Subclasses must implement from_dict")

    def __repr__(self) -> str:
        """String representation of signature."""
        inputs = list(self.input_fields.keys())
        outputs = list(self.output_fields.keys())
        return f"{self.__class__.__name__}({', '.join(inputs)} -> {', '.join(outputs)})"


class BaseSignature(Signature):
    """Concrete base signature implementation."""

    def __init__(
        self,
        input_fields: dict[str, FieldSpec] | None = None,
        output_fields: dict[str, FieldSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._input_fields = input_fields or {}
        self._output_fields = output_fields or {}

    @property
    def input_fields(self) -> dict[str, FieldSpec]:
        """Get input field specifications."""
        return self._input_fields

    @property
    def output_fields(self) -> dict[str, FieldSpec]:
        """Get output field specifications."""
        return self._output_fields

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseSignature:
        """Reconstruct BaseSignature from dictionary."""
        from .utils import make_field_spec_from_dict

        input_fields = {
            name: make_field_spec_from_dict(name, spec_data)
            for name, spec_data in data.get("input_fields", {}).items()
        }

        output_fields = {
            name: make_field_spec_from_dict(name, spec_data)
            for name, spec_data in data.get("output_fields", {}).items()
        }

        return cls(
            input_fields=input_fields,
            output_fields=output_fields,
            instructions=data.get("instructions"),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "Signature",
    "BaseSignature",
]
