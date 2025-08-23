"""Factory functions for creating signatures."""

from __future__ import annotations

import inspect
import typing
from fractions import Fraction

from ...exceptions import SignatureError
from .fields import (
    PYDANTIC_AVAILABLE,
    FieldDescriptor,
    InputField,
    OutputField,
    get_field_type,
    infer_prefix,
    is_field,
)
from .parser import signature_from_function

# Import Pydantic if available
if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, ConfigDict, Field, create_model
    from pydantic.fields import FieldInfo
else:
    BaseModel = object
    FieldInfo = None
    create_model = None
    Field = None
    ConfigDict = None


def _default_instructions(cls) -> str:
    """Generate default instructions from field names."""
    input_names = []
    output_names = []

    # Try to get fields from the class
    if hasattr(cls, "_field_definitions"):
        # Pure Python mode
        for name, field_def in cls._field_definitions.items():
            if get_field_type(field_def) == "input":
                input_names.append(name)
            elif get_field_type(field_def) == "output":
                output_names.append(name)
    elif hasattr(cls, "model_fields"):
        # Pydantic mode
        for name, field in cls.model_fields.items():
            if get_field_type(field) == "input":
                input_names.append(name)
            elif get_field_type(field) == "output":
                output_names.append(name)

    if input_names and output_names:
        inputs_str = ", ".join([f"`{name}`" for name in input_names])
        outputs_str = ", ".join([f"`{name}`" for name in output_names])
        return f"Given the fields {inputs_str}, produce the fields {outputs_str}."
    elif input_names:
        inputs_str = ", ".join([f"`{name}`" for name in input_names])
        return f"Process the fields {inputs_str}."
    elif output_names:
        outputs_str = ", ".join([f"`{name}`" for name in output_names])
        return f"Produce the fields {outputs_str}."
    return "Process the inputs to produce the outputs."


# Create the metaclass based on available mode
if PYDANTIC_AVAILABLE:
    from pydantic._internal._model_construction import ModelMetaclass

    BaseMeta = ModelMetaclass
else:
    BaseMeta = type


class SignatureMeta(BaseMeta):  # type: ignore[misc]
    """Metaclass that handles both Pydantic and pure Python modes transparently."""

    def __call__(cls, *args, **kwargs):
        """Allow Signature(string) to create a new signature class."""
        if cls.__name__ in ("Signature", "EnhancedSignature") and args and isinstance(args[0], str):
            # String signature: "input -> output"
            return make_signature(args[0], *args[1:], **kwargs)
        return super().__call__(*args, **kwargs)

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Create a new signature class, handling both modes."""
        # Check if we're in Pydantic mode
        pydantic_mode = PYDANTIC_AVAILABLE and any(
            issubclass(base, BaseModel)
            for base in bases
            if base is not object and hasattr(base, "__mro__")
        )

        if pydantic_mode:
            # Pydantic mode: Let Pydantic handle most of the work
            cls = mcs._create_pydantic_signature(name, bases, namespace, **kwargs)
        else:
            # Pure Python mode: Handle fields manually
            cls = mcs._create_pure_signature(name, bases, namespace, **kwargs)

        # Set instructions from docstring or generate default
        if not cls.__doc__:
            for base in bases:
                if hasattr(base, "__doc__") and base.__doc__:
                    cls.__doc__ = base.__doc__
                    break
            else:
                cls.__doc__ = _default_instructions(cls)

        return cls

    @classmethod
    def _create_pydantic_signature(mcs, name, bases, namespace, **kwargs):
        """Create a Pydantic-based signature class."""
        # Process annotations for Pydantic
        annotations = namespace.get("__annotations__", {})
        field_definitions = {}

        for field_name, field_value in list(namespace.items()):
            if is_field(field_value):
                field_definitions[field_name] = field_value
                # Ensure annotation exists
                if field_name not in annotations:
                    annotations[field_name] = str

        namespace["__annotations__"] = annotations

        # Create the Pydantic model
        PydanticBase = next((base for base in bases if issubclass(base, BaseModel)), BaseModel)

        # Adjust bases to include Pydantic
        if BaseModel not in bases and not any(
            issubclass(b, BaseModel) for b in bases if b is not object
        ):
            bases = (PydanticBase,) + tuple(b for b in bases if b is not PydanticBase)

        # Let Pydantic's metaclass handle creation
        cls = BaseMeta.__new__(
            SignatureMeta,  # Use our metaclass, not Pydantic's directly
            name,
            bases,
            namespace,
            **kwargs,
        )

        # Enhance Pydantic fields with defaults
        if hasattr(cls, "model_fields"):
            for field_name, field in cls.model_fields.items():  # type: ignore[attr-defined]
                if not isinstance(field, FieldInfo):
                    continue

                # Ensure json_schema_extra exists
                if field.json_schema_extra is None:
                    field.json_schema_extra = {}

                # Add prefix if missing
                if "prefix" not in field.json_schema_extra:
                    field.json_schema_extra["prefix"] = infer_prefix(field_name) + ":"

                # Add description if missing
                if "desc" not in field.json_schema_extra:
                    field.json_schema_extra["desc"] = f"${{{field_name}}}"

        return cls

    @classmethod
    def _create_pure_signature(mcs, name, bases, namespace, **kwargs):
        """Create a pure Python signature class."""
        # Collect field definitions
        field_definitions = {}
        annotations = namespace.get("__annotations__", {})

        for field_name, field_value in list(namespace.items()):
            if isinstance(field_value, FieldDescriptor):
                field_definitions[field_name] = field_value
                # Store the annotation on the descriptor
                if field_name in annotations:
                    field_value.annotation = annotations[field_name]
                else:
                    field_value.annotation = str
                    annotations[field_name] = str

                # Add defaults for prefix and desc
                if field_value.prefix is None:
                    field_value.prefix = infer_prefix(field_name) + ":"
                if field_value.desc is None:
                    field_value.desc = f"${{{field_name}}}"

                # Remove from namespace to avoid conflicts
                del namespace[field_name]

        # Store field definitions on the class
        namespace["_field_definitions"] = field_definitions
        namespace["__annotations__"] = annotations

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)

        return cls

    @property
    def instructions(cls) -> str:
        """Get signature instructions."""
        return inspect.cleandoc(cls.__doc__ or "")

    @instructions.setter
    def instructions(cls, value: str):
        """Set signature instructions."""
        cls.__doc__ = value

    @property
    def input_fields(cls) -> dict:
        """Get input fields."""
        fields = {}

        if hasattr(cls, "_field_definitions"):
            # Pure Python mode
            for name, field_def in cls._field_definitions.items():
                if get_field_type(field_def) == "input":
                    fields[name] = field_def
        elif hasattr(cls, "model_fields"):
            # Pydantic mode
            for name, field in cls.model_fields.items():
                if get_field_type(field) == "input":
                    fields[name] = field

        return fields

    @property
    def output_fields(cls) -> dict:
        """Get output fields."""
        fields = {}

        if hasattr(cls, "_field_definitions"):
            # Pure Python mode
            for name, field_def in cls._field_definitions.items():
                if get_field_type(field_def) == "output":
                    fields[name] = field_def
        elif hasattr(cls, "model_fields"):
            # Pydantic mode
            for name, field in cls.model_fields.items():
                if get_field_type(field) == "output":
                    fields[name] = field

        return fields

    @property
    def fields(cls) -> dict:
        """Get all fields (input then output)."""
        return {**cls.input_fields, **cls.output_fields}

    @property
    def signature(cls) -> str:
        """Get string representation of signature."""
        inputs = ", ".join(cls.input_fields.keys())
        outputs = ", ".join(cls.output_fields.keys())
        return f"{inputs} -> {outputs}"


# Create base Signature class based on available mode
if PYDANTIC_AVAILABLE:

    class SignatureBase(BaseModel):  # type: ignore[misc]
        """Base class for Pydantic mode."""

        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # type: ignore[misc]
else:

    class SignatureBase:
        """Base class for pure Python mode."""

        def __init__(self, **kwargs):
            """Initialize with field values."""
            # Set field values from kwargs
            if hasattr(self.__class__, "_field_definitions"):
                for name, field_def in self.__class__._field_definitions.items():
                    if name in kwargs:
                        setattr(self, name, kwargs[name])
                    elif field_def.default is not None:
                        setattr(self, name, field_def.default)


def make_signature(
    signature: str | dict,
    instructions: str | None = None,
    signature_name: str = "DynamicSignature",
    custom_types: dict[str, type] | None = None,
) -> type:
    """Create a new Signature class.

    Args:
        signature: Either a string like "question -> answer" or a dict of fields
        instructions: Optional instructions/docstring
        signature_name: Name for the generated class
        custom_types: Optional mapping of type names to types

    Returns:
        A new Signature class
    """
    if isinstance(signature, str):
        fields = _parse_signature_string(signature, custom_types)
    else:
        fields = signature

    # Process fields based on mode
    if PYDANTIC_AVAILABLE:
        # Prepare fields for Pydantic
        processed_fields = {}
        for name, value in fields.items():
            if isinstance(value, tuple):
                type_hint, field_info = value
            elif isinstance(value, (FieldInfo, FieldDescriptor)):
                if hasattr(value, "annotation"):
                    type_hint = value.annotation
                else:
                    type_hint = str
                field_info = value
            else:
                type_hint = str
                field_info = value

            if type_hint is None:
                type_hint = str

            processed_fields[name] = (type_hint, field_info)

        # Set default instructions
        if instructions is None:
            temp_cls = type(
                "_Temp",
                (),
                {"_field_definitions": {n: f for n, (_, f) in processed_fields.items()}},
            )
            instructions = _default_instructions(temp_cls)

        # Import here to avoid circular imports
        from .signature import Signature

        # Create Pydantic model
        return create_model(  # type: ignore[call-overload,no-any-return]
            signature_name, __base__=Signature, __doc__=instructions, **processed_fields
        )
    else:
        # Pure Python mode
        namespace = {
            "__doc__": instructions,
            "__annotations__": {},
        }

        for name, value in fields.items():
            if isinstance(value, tuple):
                type_hint, field_def = value
            elif isinstance(value, FieldDescriptor):
                type_hint = value.annotation or str
                field_def = value
            else:
                type_hint = str
                field_def = value

            namespace[name] = field_def
            namespace["__annotations__"][name] = type_hint

        # Set default instructions
        if instructions is None:
            temp_cls = type(
                "_Temp",
                (),
                {
                    "_field_definitions": {
                        n: f for n, f in namespace.items() if isinstance(f, FieldDescriptor)
                    }
                },
            )
            instructions = _default_instructions(temp_cls)
            namespace["__doc__"] = instructions

        # Import here to avoid circular imports
        from .signature import Signature

        return type(signature_name, (Signature,), namespace)


def _parse_signature_string(
    signature_str: str, custom_types: dict[str, type] | None = None
) -> dict:
    """Parse a signature string like 'input1, input2 -> output1, output2'."""
    if signature_str.count("->") != 1:
        raise SignatureError("Invalid signature: must contain exactly one '->'")

    inputs_str, outputs_str = signature_str.split("->")

    # Build type resolution dictionary
    type_names = dict(typing.__dict__)
    type_names.update(
        {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "Fraction": Fraction,
            "fraction": Fraction,
        }
    )
    if custom_types:
        type_names.update(custom_types)

    fields = {}

    # Parse input fields
    for field_def in inputs_str.split(","):
        field_def = field_def.strip()
        if not field_def:
            continue

        name, type_hint = _parse_field_definition(field_def, type_names)
        fields[name] = (type_hint, InputField())

    # Parse output fields
    for field_def in outputs_str.split(","):
        field_def = field_def.strip()
        if not field_def:
            continue

        name, type_hint = _parse_field_definition(field_def, type_names)
        fields[name] = (type_hint, OutputField())

    return fields


def _parse_field_definition(field_def: str, type_names: dict[str, type]) -> tuple[str, type]:
    """Parse a single field definition like 'name: type'."""
    if ":" in field_def:
        name, type_str = field_def.split(":", 1)
        name = name.strip()
        type_str = type_str.strip()

        # Resolve the type
        if type_str in type_names:
            type_hint = type_names[type_str]
        else:
            # Try parsing complex types
            try:
                type_hint = _parse_type_from_string(type_str, type_names)
            except (ValueError, TypeError, AttributeError):
                type_hint = str
    else:
        name = field_def.strip()
        type_hint = str

    if not name.isidentifier():
        raise SignatureError(f"Invalid field name: '{name}'")

    return name, type_hint


def _parse_type_from_string(type_str: str, type_names: dict[str, type]) -> type:
    """Parse a type string (simplified version)."""
    # Handle generic types like list[int], Optional[str]
    if "[" in type_str:
        base_type_str = type_str.split("[")[0]
        if base_type_str in type_names:
            return type_names[base_type_str]

    return type_names.get(type_str, str)


__all__ = [
    "make_signature",
    "signature_from_function",
    "SignatureMeta",
    "SignatureBase",
]
