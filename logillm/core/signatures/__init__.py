"""Clean, modular signature system for LogiLLM.

This module provides a unified signature system that works with or without Pydantic.
The enhanced Signature class is the main class to use.
"""

# Main signature class (from enhanced)
# Base classes (for inheritance/compatibility)
# Types
from ..types import FieldType
from .base import BaseSignature

# Factory functions
from .factory import make_signature

# Field creation
from .fields import FieldDescriptor, InputField, OutputField

# Parsing
from .parser import parse_signature_string, signature_from_function
from .signature import Signature, ensure_signature

# Specs
from .spec import FieldSpec

# Utilities
from .utils import (
    apply_constraints,
    coerce_value_to_spec,
    infer_type_from_value,
    make_field_spec_from_dict,
    validate_signature_fields,
)

# Compatibility aliases
EnhancedSignature = Signature  # The enhanced signature IS the main signature

__all__ = [
    # Main signature class
    "Signature",
    "EnhancedSignature",  # Alias for compatibility
    # Field creation
    "InputField",
    "OutputField",
    "FieldDescriptor",
    # Parsing
    "parse_signature_string",
    "signature_from_function",
    # Factory functions
    "make_signature",
    "ensure_signature",
    # Specs and base classes
    "FieldSpec",
    "FieldType",
    "BaseSignature",
    # Utilities
    "coerce_value_to_spec",
    "apply_constraints",
    "validate_signature_fields",
    "make_field_spec_from_dict",
    "infer_type_from_value",
]
