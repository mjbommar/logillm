"""Field specification for signature system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...exceptions import SignatureError
from ..types import FieldType, Metadata


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single field."""

    name: str
    field_type: FieldType
    python_type: type[Any] = str
    description: str | None = None
    required: bool = True
    default: Any = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    @property
    def annotation(self) -> type[Any]:
        """Alias for python_type for backward compatibility."""
        return self.python_type

    @property
    def desc(self) -> str:
        """Alias for description for backward compatibility."""
        return self.description or self.name

    def __post_init__(self) -> None:
        """Validate field specification after creation."""
        if not self.name.isidentifier():
            raise SignatureError(
                f"Field name '{self.name}' is not a valid Python identifier", field_name=self.name
            )

        if self.default is not None and self.required:
            # If default is provided, field is not required
            object.__setattr__(self, "required", False)


__all__ = [
    "FieldSpec",
]
