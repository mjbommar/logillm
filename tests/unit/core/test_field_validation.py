"""Tests for field validation in signatures."""

from typing import Optional

import pytest

from logillm.core.signatures import Signature
from logillm.core.signatures.fields import InputField, OutputField
from logillm.core.signatures.spec import FieldSpec
from logillm.core.types import FieldType


class TestFieldValidation:
    """Test field validation functionality."""

    def test_required_field_validation(self):
        """Test that required fields are validated."""

        class TestSignature(Signature):
            required_input: str = InputField(desc="Required field")
            optional_input: Optional[str] = InputField(default=None, desc="Optional field")

        # Should pass with required field
        result = TestSignature.validate_inputs(required_input="test")
        assert result["required_input"] == "test"

        # Should fail without required field
        with pytest.raises(ValueError) as exc:
            TestSignature.validate_inputs()
        assert "required_input" in str(exc.value)

    def test_type_validation(self):
        """Test that type checking works."""

        class TypedSignature(Signature):
            text_field: str = InputField()
            number_field: int = InputField()
            float_field: float = InputField()

        # Should pass with correct types
        result = TypedSignature.validate_inputs(
            text_field="hello", number_field=42, float_field=3.14
        )
        assert result["text_field"] == "hello"
        assert result["number_field"] == 42
        assert result["float_field"] == 3.14

        # Should coerce compatible types
        result = TypedSignature.validate_inputs(
            text_field="hello",
            number_field="42",  # String that can be int
            float_field=3,  # Int that can be float
        )
        assert result["number_field"] == 42
        assert result["float_field"] == 3.0

    def test_constraint_validation(self):
        """Test that constraints are applied."""
        # Create a FieldSpec with constraints
        spec = FieldSpec(
            name="test_field",
            field_type=FieldType.INPUT,
            python_type=str,
            constraints={"min_length": 3, "max_length": 10},
        )

        # Test with utils directly first
        from logillm.core.signatures.utils import coerce_value_to_spec

        # Should pass with valid length
        coerce_value_to_spec("hello", spec)

        # Should fail with too short
        with pytest.raises(ValueError) as exc:
            coerce_value_to_spec("hi", spec)
        assert "minimum" in str(exc.value)

        # Should fail with too long
        with pytest.raises(ValueError) as exc:
            coerce_value_to_spec("this is way too long", spec)
        assert "maximum" in str(exc.value)

    def test_numeric_constraints(self):
        """Test numeric value constraints."""
        spec = FieldSpec(
            name="number",
            field_type=FieldType.INPUT,
            python_type=int,
            constraints={"min_value": 0, "max_value": 100},
        )

        from logillm.core.signatures.utils import coerce_value_to_spec

        # Should pass within range
        coerce_value_to_spec(50, spec)

        # Should fail below minimum
        with pytest.raises(ValueError) as exc:
            coerce_value_to_spec(-5, spec)
        assert "minimum" in str(exc.value)

        # Should fail above maximum
        with pytest.raises(ValueError) as exc:
            coerce_value_to_spec(150, spec)
        assert "maximum" in str(exc.value)

    def test_choices_constraint(self):
        """Test choices/enum constraint."""
        spec = FieldSpec(
            name="color",
            field_type=FieldType.INPUT,
            python_type=str,
            constraints={"choices": ["red", "green", "blue"]},
        )

        from logillm.core.signatures.utils import coerce_value_to_spec

        # Should pass with valid choice
        coerce_value_to_spec("red", spec)

        # Should fail with invalid choice
        with pytest.raises(ValueError) as exc:
            coerce_value_to_spec("yellow", spec)
        assert "not in allowed choices" in str(exc.value)

    def test_pattern_constraint(self):
        """Test regex pattern constraint."""
        spec = FieldSpec(
            name="email",
            field_type=FieldType.INPUT,
            python_type=str,
            constraints={"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
        )

        from logillm.core.signatures.utils import coerce_value_to_spec

        # Should pass with valid email
        coerce_value_to_spec("test@example.com", spec)

        # Should fail with invalid format
        with pytest.raises(ValueError) as exc:
            coerce_value_to_spec("not_an_email", spec)
        assert "does not match pattern" in str(exc.value)

    def test_output_validation(self):
        """Test that output validation works similarly."""

        class OutputSignature(Signature):
            result: str = OutputField(desc="Result field")
            score: float = OutputField(desc="Score field")

        # Should validate outputs
        result = OutputSignature.validate_outputs(result="success", score=0.95)
        assert result["result"] == "success"
        assert result["score"] == 0.95

        # Should coerce types
        result = OutputSignature.validate_outputs(
            result="success",
            score="0.95",  # String that can be float
        )
        assert result["score"] == 0.95

    def test_default_values(self):
        """Test that default values are used when fields are missing."""

        class DefaultSignature(Signature):
            required: str = InputField(desc="Required")
            with_default: str = InputField(default="default_value", desc="Has default")

        # Should use default when not provided
        result = DefaultSignature.validate_inputs(required="test")
        assert result["required"] == "test"
        assert result["with_default"] == "default_value"

        # Should override default when provided
        result = DefaultSignature.validate_inputs(required="test", with_default="custom")
        assert result["with_default"] == "custom"
