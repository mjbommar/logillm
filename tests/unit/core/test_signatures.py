"""Tests for enhanced Pydantic-based signatures."""

from fractions import Fraction

import pytest

from logillm.core.signatures import (
    InputField,
    OutputField,
    Signature,
    ensure_signature,
    make_signature,
)


def test_string_signature_basic():
    """Test basic string signature parsing."""
    sig = make_signature("question, context -> answer")

    assert "question" in sig.input_fields
    assert "context" in sig.input_fields
    assert "answer" in sig.output_fields
    assert sig.signature == "question, context -> answer"


def test_string_signature_with_types():
    """Test string signature with type annotations."""
    sig = make_signature("question: str, num: int -> answer: str, score: float")

    assert sig.input_fields["question"].annotation == str
    assert sig.input_fields["num"].annotation == int
    assert sig.output_fields["answer"].annotation == str
    assert sig.output_fields["score"].annotation == float


def test_string_signature_with_fraction():
    """Test string signature with Fraction type."""
    sig = make_signature("numerator: int, denominator: int -> result: Fraction")

    assert sig.output_fields["result"].annotation == Fraction


def test_class_based_signature():
    """Test class-based signature definition."""

    class MathSignature(Signature):
        """Solve a mathematical problem."""

        question: str = InputField(desc="The math problem to solve")
        reasoning: str = OutputField(desc="Step-by-step reasoning")
        answer: float = OutputField(desc="The numerical answer")

    assert "question" in MathSignature.input_fields
    assert "reasoning" in MathSignature.output_fields
    assert "answer" in MathSignature.output_fields
    assert MathSignature.instructions == "Solve a mathematical problem."


def test_signature_with_default_instructions():
    """Test signature generates default instructions."""
    sig = make_signature("input1, input2 -> output")

    assert "input1" in sig.instructions
    assert "input2" in sig.instructions
    assert "output" in sig.instructions


def test_signature_with_custom_instructions():
    """Test signature with custom instructions."""
    sig = make_signature("question -> answer", instructions="Answer the question concisely.")

    assert sig.instructions == "Answer the question concisely."


def test_dict_based_signature():
    """Test creating signature from dictionary."""
    sig = make_signature(
        {
            "question": (str, InputField(desc="The question")),
            "context": (list, InputField(desc="Context documents")),
            "answer": (str, OutputField(desc="The answer")),
        }
    )

    assert sig.input_fields["question"].annotation == str
    assert sig.input_fields["context"].annotation == list
    assert sig.output_fields["answer"].annotation == str


def test_signature_field_prefixes():
    """Test that fields get automatic prefixes."""

    class TestSig(Signature):
        user_input: str = InputField()
        camelCaseField: str = InputField()
        response: str = OutputField()

    # Check prefixes are generated
    assert TestSig.input_fields["user_input"].json_schema_extra["prefix"] == "User Input:"
    assert TestSig.input_fields["camelCaseField"].json_schema_extra["prefix"] == "Camel Case Field:"
    assert TestSig.output_fields["response"].json_schema_extra["prefix"] == "Response:"


def test_signature_field_descriptions():
    """Test field descriptions with defaults."""

    class TestSig(Signature):
        field_with_desc: str = InputField(desc="Custom description")
        field_without_desc: str = OutputField()

    assert TestSig.input_fields["field_with_desc"].json_schema_extra["desc"] == "Custom description"
    assert (
        TestSig.output_fields["field_without_desc"].json_schema_extra["desc"]
        == "${field_without_desc}"
    )


def test_signature_inheritance():
    """Test signature class inheritance."""

    class BaseSignature(Signature):
        """Base signature for Q&A."""

        question: str = InputField(desc="The question")
        answer: str = OutputField(desc="The answer")

    class ExtendedSignature(BaseSignature):
        """Extended Q&A with context."""

        context: str = InputField(desc="Additional context")
        confidence: float = OutputField(desc="Answer confidence")

    # Check all fields are present
    assert "question" in ExtendedSignature.input_fields
    assert "context" in ExtendedSignature.input_fields
    assert "answer" in ExtendedSignature.output_fields
    assert "confidence" in ExtendedSignature.output_fields


def test_with_instructions():
    """Test updating signature instructions."""

    class OriginalSig(Signature):
        """Original instructions."""

        input: str = InputField()
        output: str = OutputField()

    UpdatedSig = OriginalSig.with_instructions("New instructions.")

    assert OriginalSig.instructions == "Original instructions."
    assert UpdatedSig.instructions == "New instructions."
    assert "input" in UpdatedSig.input_fields
    assert "output" in UpdatedSig.output_fields


def test_ensure_signature():
    """Test ensure_signature utility."""
    # From string
    sig1 = ensure_signature("input -> output")
    assert sig1.signature == "input -> output"

    # From existing signature
    class ExistingSig(Signature):
        field: str = InputField()
        result: str = OutputField()

    sig2 = ensure_signature(ExistingSig)
    assert sig2 is ExistingSig

    # With instructions (should raise)
    with pytest.raises(ValueError):
        ensure_signature(ExistingSig, instructions="New")


def test_to_base_signature():
    """Test conversion to BaseSignature."""

    class TestSig(Signature):
        """Test signature."""

        question: str = InputField(desc="Question")
        answer: str = OutputField(desc="Answer")

    # Create instance with required fields for Pydantic mode
    # In pure Python mode, fields are optional
    try:
        instance = TestSig(question="test", answer="test")
    except TypeError:
        # Pure Python mode doesn't take field values in __init__
        instance = TestSig()

    base_sig = instance.to_base_signature()

    assert "question" in base_sig.input_fields
    assert "answer" in base_sig.output_fields
    assert base_sig.input_fields["question"].description == "Question"
    assert base_sig.output_fields["answer"].description == "Answer"


def test_custom_types():
    """Test signature with custom types."""

    class CustomType:
        pass

    sig = make_signature("data: CustomType -> result: str", custom_types={"CustomType": CustomType})

    assert sig.input_fields["data"].annotation == CustomType
    assert sig.output_fields["result"].annotation == str


def test_signature_callable_syntax():
    """Test that Signature(string) creates a new signature class."""
    sig = Signature("input1, input2 -> output")

    assert "input1" in sig.input_fields
    assert "input2" in sig.input_fields
    assert "output" in sig.output_fields


def test_signature_equality():
    """Test comparing signatures."""
    sig1 = make_signature("question -> answer", instructions="Answer the question.")
    sig2 = make_signature("question -> answer", instructions="Answer the question.")
    sig3 = make_signature("query -> response", instructions="Answer the question.")

    # Same structure and instructions
    assert sig1.instructions == sig2.instructions
    assert sig1.signature == sig2.signature

    # Different fields
    assert sig1.signature != sig3.signature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
