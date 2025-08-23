"""Enhanced field definitions for LogiLLM signatures.

Supports both pure Python and Pydantic modes seamlessly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import Field
    from pydantic.fields import FieldInfo

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    Field = None  # type: ignore[assignment]
    FieldInfo = None  # type: ignore[assignment]


# Field argument names specific to LogiLLM
LOGILLM_FIELD_ARG_NAMES = ["desc", "prefix", "format", "parser", "__logillm_field_type"]

# Map Pydantic constraints to human-readable descriptions
PYDANTIC_CONSTRAINT_MAP = {
    "gt": "greater than: ",
    "ge": "greater than or equal to: ",
    "lt": "less than: ",
    "le": "less than or equal to: ",
    "min_length": "minimum length: ",
    "max_length": "maximum length: ",
    "multiple_of": "a multiple of: ",
    "allow_inf_nan": "allow 'inf', '-inf', 'nan' values: ",
}


def infer_prefix(field_name: str) -> str:
    """Convert field name to human-readable prefix.

    Examples:
        "question" -> "Question"
        "user_input" -> "User Input"
        "camelCaseField" -> "Camel Case Field"
        "HTML_content" -> "HTML Content"
    """
    # Convert camelCase to snake_case
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", field_name)
    intermediate = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)

    # Handle numbers
    with_underscores = re.sub(r"([a-zA-Z])(\d)", r"\1_\2", intermediate)
    with_underscores = re.sub(r"(\d)([a-zA-Z])", r"\1_\2", with_underscores)

    # Convert to Title Case
    words = with_underscores.split("_")
    title_words = []
    for word in words:
        if word.isupper() and len(word) > 1:
            # Keep acronyms as-is
            title_words.append(word)
        else:
            title_words.append(word.capitalize())

    return " ".join(title_words)


@dataclass
class FieldDescriptor:
    """Field descriptor for pure Python mode.

    This class acts as a descriptor that stores field metadata
    and can be used in place of Pydantic's Field when Pydantic
    is not available.
    """

    field_type: str  # 'input' or 'output'
    desc: str | None = None
    prefix: str | None = None
    format: str | None = None
    parser: Any | None = None
    default: Any = None
    required: bool = True
    annotation: type | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Store all extra kwargs in metadata."""
        # Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}

    @property
    def json_schema_extra(self) -> dict[str, Any]:
        """Compatibility property for Pydantic-like access."""
        return {
            "__logillm_field_type": self.field_type,
            "desc": self.desc,
            "prefix": self.prefix,
            "format": self.format,
            "parser": self.parser,
            **self.metadata,
        }

    def is_required(self) -> bool:
        """Check if field is required."""
        return self.required and self.default is None


def _move_kwargs_to_json_schema(**kwargs):
    """Move LogiLLM-specific kwargs to json_schema_extra for Pydantic."""
    pydantic_kwargs = {}
    json_schema_extra = {}

    for key, value in kwargs.items():
        if key in LOGILLM_FIELD_ARG_NAMES:
            json_schema_extra[key] = value
        else:
            pydantic_kwargs[key] = value

    # Copy Pydantic description to desc if not provided
    if "description" in kwargs and "desc" not in json_schema_extra:
        json_schema_extra["desc"] = kwargs["description"]

    # Extract and format constraints
    constraints = _extract_pydantic_constraints(**kwargs)
    if constraints:
        json_schema_extra["constraints"] = constraints

    pydantic_kwargs["json_schema_extra"] = json_schema_extra
    return pydantic_kwargs


def _extract_pydantic_constraints(**kwargs):
    """Extract Pydantic constraints to human-readable format."""
    constraints = []
    for key, value in kwargs.items():
        if key in PYDANTIC_CONSTRAINT_MAP:
            constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")
    return ", ".join(constraints) if constraints else None


def InputField(**kwargs):
    """Create an input field for a signature.

    Works seamlessly in both pure Python and Pydantic modes:
    - With Pydantic: Returns a pydantic.Field with json_schema_extra
    - Without Pydantic: Returns a FieldDescriptor

    Args:
        desc: Field description
        prefix: Custom prefix for the field (defaults to inferred)
        format: Expected format of the field
        parser: Custom parser function
        default: Default value for the field
        **kwargs: Additional field arguments

    Returns:
        FieldInfo (Pydantic mode) or FieldDescriptor (pure Python mode)

    Example:
        >>> class MySignature(Signature):
        ...     question: str = InputField(desc="The question to answer")
        ...     context: str = InputField(desc="Relevant context", default="")
    """
    if PYDANTIC_AVAILABLE:
        # Pydantic mode: Use native Field with json_schema_extra
        kwargs["__logillm_field_type"] = "input"
        return Field(**_move_kwargs_to_json_schema(**kwargs))
    else:
        # Pure Python mode: Return a FieldDescriptor
        return FieldDescriptor(
            field_type="input",
            desc=kwargs.get("desc"),
            prefix=kwargs.get("prefix"),
            format=kwargs.get("format"),
            parser=kwargs.get("parser"),
            default=kwargs.get("default"),
            required=kwargs.get("required", True),
            metadata={
                k: v
                for k, v in kwargs.items()
                if k not in ["desc", "prefix", "format", "parser", "default", "required"]
            },
        )


def OutputField(**kwargs):
    """Create an output field for a signature.

    Works seamlessly in both pure Python and Pydantic modes:
    - With Pydantic: Returns a pydantic.Field with json_schema_extra
    - Without Pydantic: Returns a FieldDescriptor

    Args:
        desc: Field description
        prefix: Custom prefix for the field (defaults to inferred)
        format: Expected format of the field
        parser: Custom parser function
        default: Default value for the field
        **kwargs: Additional field arguments

    Returns:
        FieldInfo (Pydantic mode) or FieldDescriptor (pure Python mode)

    Example:
        >>> class MySignature(Signature):
        ...     reasoning: str = OutputField(desc="Step-by-step reasoning")
        ...     answer: float = OutputField(desc="The numerical answer")
    """
    if PYDANTIC_AVAILABLE:
        # Pydantic mode: Use native Field with json_schema_extra
        kwargs["__logillm_field_type"] = "output"
        return Field(**_move_kwargs_to_json_schema(**kwargs))
    else:
        # Pure Python mode: Return a FieldDescriptor
        return FieldDescriptor(
            field_type="output",
            desc=kwargs.get("desc"),
            prefix=kwargs.get("prefix"),
            format=kwargs.get("format"),
            parser=kwargs.get("parser"),
            default=kwargs.get("default"),
            required=kwargs.get("required", True),
            metadata={
                k: v
                for k, v in kwargs.items()
                if k not in ["desc", "prefix", "format", "parser", "default", "required"]
            },
        )


def get_field_type(field: Any) -> str | None:
    """Get the field type (input/output) from a field.

    Works with both Pydantic FieldInfo and FieldDescriptor.
    """
    if isinstance(field, FieldDescriptor):
        return field.field_type
    elif PYDANTIC_AVAILABLE and isinstance(field, FieldInfo):
        extra = field.json_schema_extra or {}
        return extra.get("__logillm_field_type")
    elif isinstance(field, dict):
        return field.get("field_type") or field.get("__logillm_field_type")
    return None


def get_field_desc(field: Any) -> str | None:
    """Get the description from a field.

    Works with both Pydantic FieldInfo and FieldDescriptor.
    """
    if isinstance(field, FieldDescriptor):
        return field.desc
    elif PYDANTIC_AVAILABLE and isinstance(field, FieldInfo):
        extra = field.json_schema_extra or {}
        return extra.get("desc") or field.description
    elif isinstance(field, dict):
        return field.get("desc") or field.get("description")
    return None


def get_field_prefix(field: Any) -> str | None:
    """Get the prefix from a field.

    Works with both Pydantic FieldInfo and FieldDescriptor.
    """
    if isinstance(field, FieldDescriptor):
        return field.prefix
    elif PYDANTIC_AVAILABLE and isinstance(field, FieldInfo):
        extra = field.json_schema_extra or {}
        return extra.get("prefix")
    elif isinstance(field, dict):
        return field.get("prefix")
    return None


def is_field(obj: Any) -> bool:
    """Check if an object is a field definition.

    Returns True for FieldDescriptor, Pydantic FieldInfo, or field-like dicts.
    """
    if isinstance(obj, FieldDescriptor):
        return True
    if PYDANTIC_AVAILABLE and isinstance(obj, FieldInfo):
        return True
    if isinstance(obj, dict) and ("field_type" in obj or "__logillm_field_type" in obj):
        return True
    return False


__all__ = [
    "InputField",
    "OutputField",
    "FieldDescriptor",
    "infer_prefix",
    "get_field_type",
    "get_field_desc",
    "get_field_prefix",
    "is_field",
    "PYDANTIC_AVAILABLE",
]
