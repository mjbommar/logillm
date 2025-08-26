"""Tests for enhanced signature features achieving DSPy parity."""

from typing import Optional, Union

from logillm.core.signatures.parser import (
    _infer_type_from_value,
    _parse_type_expression,
    infer_signature_from_examples,
    parse_signature_string,
    signature_to_string,
)
from logillm.core.signatures.types import (
    Audio,
    History,
    Image,
    Tool,
)


class TestMultimodalTypes:
    """Test multimodal type support."""

    def test_image_type(self):
        """Test Image type functionality."""
        # From base64
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        img = Image.from_base64(b64_data, format="png")
        assert img.format == "png"
        assert len(img.data) > 0

        # To data URL
        data_url = img.to_data_url()
        assert data_url.startswith("data:image/png;base64,")

        # String representation
        assert "Image(png" in str(img)

    def test_audio_type(self):
        """Test Audio type functionality."""
        # Create audio from bytes
        audio = Audio(data=b"fake audio data", format="mp3")
        assert audio.format == "mp3"
        assert "Audio(mp3" in str(audio)

        # With duration
        audio_with_duration = Audio(data=b"fake audio", format="wav", duration_seconds=10.5)
        assert "10.5s" in str(audio_with_duration)

    def test_tool_type(self):
        """Test Tool type functionality."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
            function=add,
        )

        assert tool.name == "add"
        assert tool(a=2, b=3) == 5

        schema = tool.to_json_schema()
        assert schema["name"] == "add"
        assert "Tool(add, 2 params)" in str(tool)

    def test_history_type(self):
        """Test History type functionality."""
        history = History(messages=[])

        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")

        assert len(history) == 2
        assert history.get_last_n(1)[0]["role"] == "assistant"
        assert "History(2 messages)" in str(history)


class TestComplexTypeParsing:
    """Test parsing of complex type expressions."""

    def test_basic_types(self):
        """Test basic type parsing."""
        assert _parse_type_expression("str") == str
        assert _parse_type_expression("int") == int
        assert _parse_type_expression("float") == float
        assert _parse_type_expression("bool") == bool

    def test_generic_types(self):
        """Test generic type parsing."""
        # List types
        list_str = _parse_type_expression("list[str]")
        assert list_str == list[str]

        List_int = _parse_type_expression("List[int]")

        assert List_int == list[int]

        # Dict types
        dict_type = _parse_type_expression("dict[str, int]")
        assert dict_type == dict[str, int]

        # Nested generics
        nested = _parse_type_expression("list[dict[str, float]]")
        assert nested == list[dict[str, float]]

    def test_optional_types(self):
        """Test Optional type parsing."""
        from typing import Optional

        opt_str = _parse_type_expression("Optional[str]")
        assert opt_str == Optional[str]

        # Union shorthand
        union_none = _parse_type_expression("str | None")
        assert union_none == Union[str, type(None)]

    def test_union_types(self):
        """Test Union type parsing."""
        from typing import Union

        union_type = _parse_type_expression("Union[str, int]")
        assert union_type == Union[str, int]

        # Pipe syntax
        pipe_union = _parse_type_expression("str | int | float")
        assert pipe_union == Union[str, int, float]

    def test_multimodal_types(self):
        """Test multimodal type parsing."""
        assert _parse_type_expression("Image") == Image
        assert _parse_type_expression("Audio") == Audio
        assert _parse_type_expression("Tool") == Tool
        assert _parse_type_expression("History") == History

    def test_custom_types(self):
        """Test custom type resolution."""

        class CustomType:
            pass

        custom_types = {"CustomType": CustomType}
        parsed = _parse_type_expression("CustomType", custom_types)
        assert parsed == CustomType

        # With generics
        list_custom = _parse_type_expression("list[CustomType]", custom_types)
        assert list_custom == list[CustomType]


class TestEnhancedSignatureParsing:
    """Test enhanced signature string parsing."""

    def test_simple_signature(self):
        """Test basic signature parsing."""
        sig = parse_signature_string("question -> answer")

        assert "question" in sig.input_fields
        assert sig.input_fields["question"].python_type == str
        assert "answer" in sig.output_fields
        assert sig.output_fields["answer"].python_type == str

    def test_typed_signature(self):
        """Test typed signature parsing."""
        sig = parse_signature_string("x: int, y: float -> sum: float")

        assert sig.input_fields["x"].python_type == int
        assert sig.input_fields["y"].python_type == float
        assert sig.output_fields["sum"].python_type == float

    def test_generic_signature(self):
        """Test generic type signature parsing."""
        sig = parse_signature_string(
            "items: list[str], mapping: dict[str, int] -> result: Optional[str]"
        )

        assert sig.input_fields["items"].python_type == list[str]
        assert sig.input_fields["mapping"].python_type == dict[str, int]
        assert sig.output_fields["result"].python_type == Optional[str]

    def test_multimodal_signature(self):
        """Test multimodal type signature parsing."""
        sig = parse_signature_string("image: Image, prompt: str -> caption: str, confidence: float")

        assert sig.input_fields["image"].python_type == Image
        assert sig.input_fields["prompt"].python_type == str
        assert sig.output_fields["caption"].python_type == str
        assert sig.output_fields["confidence"].python_type == float

    def test_complex_nested_types(self):
        """Test complex nested type parsing."""
        sig = parse_signature_string("data: list[dict[str, list[int]]] -> processed: bool")

        assert sig.input_fields["data"].python_type == list[dict[str, list[int]]]
        assert sig.output_fields["processed"].python_type == bool


class TestSignatureInference:
    """Test signature inference from examples."""

    def test_infer_simple_signature(self):
        """Test inferring signature from simple examples."""
        examples = [
            {"input": {"text": "Hello"}, "output": {"length": 5}},
            {"input": {"text": "World"}, "output": {"length": 5}},
        ]

        fields = infer_signature_from_examples(examples)

        assert "text" in fields
        assert fields["text"].python_type == str
        assert "length" in fields
        assert fields["length"].python_type == int

    def test_infer_with_lists(self):
        """Test inferring signature with list types."""
        examples = [
            {"input": {"items": ["a", "b", "c"]}, "output": {"count": 3}},
            {"input": {"items": ["x", "y"]}, "output": {"count": 2}},
        ]

        fields = infer_signature_from_examples(examples)

        assert "items" in fields

        assert fields["items"].python_type == list[str]

    def test_infer_with_mixed_types(self):
        """Test inference when types vary across examples."""
        examples = [
            {"input": {"value": "string"}, "output": {"result": True}},
            {"input": {"value": 123}, "output": {"result": False}},
        ]

        fields = infer_signature_from_examples(examples)

        assert "value" in fields
        from typing import Any

        assert fields["value"].python_type == Any  # Mixed types -> Any


class TestSignatureIntegration:
    """Test integration with Signature class."""

    def test_signature_with_complex_types(self):
        """Test creating Signature with complex types."""
        # This tests that our enhanced parser integrates with the Signature class
        sig_str = "items: list[str], threshold: float -> matches: list[int], score: float"

        # This would be used internally by Signature
        sig = parse_signature_string(sig_str)

        # Verify field structure
        assert len(sig.input_fields) == 2
        assert len(sig.output_fields) == 2
        assert sig.input_fields["items"].python_type == list[str]
        assert sig.input_fields["threshold"].python_type == float
        assert sig.output_fields["matches"].python_type == list[int]
        assert sig.output_fields["score"].python_type == float

    def test_signature_to_string_conversion(self):
        """Test converting signature back to string."""
        from logillm.core.signatures.spec import FieldSpec
        from logillm.core.types import FieldType

        fields = {
            "query": FieldSpec("query", FieldType.INPUT, str),
            "context": FieldSpec("context", FieldType.INPUT, list[str]),
            "answer": FieldSpec("answer", FieldType.OUTPUT, str),
            "confidence": FieldSpec("confidence", FieldType.OUTPUT, float),
        }

        sig_str = signature_to_string(fields)
        # Note: exact format may vary, but should contain all fields
        assert "query" in sig_str
        assert "context" in sig_str
        assert "answer" in sig_str
        assert "->" in sig_str


class TestTypeInference:
    """Test type inference from values."""

    def test_infer_basic_types(self):
        """Test inferring basic types from values."""
        assert _infer_type_from_value("hello") == str
        assert _infer_type_from_value(42) == int
        assert _infer_type_from_value(3.14) == float
        assert _infer_type_from_value(True) == bool

    def test_infer_collection_types(self):
        """Test inferring collection types."""

        # List with uniform types
        assert _infer_type_from_value([1, 2, 3]) == list[int]
        assert _infer_type_from_value(["a", "b"]) == list[str]

        # Dict with uniform types
        assert _infer_type_from_value({"a": 1, "b": 2}) == dict[str, int]

        # Empty collections fall back to base type
        assert _infer_type_from_value([]) == list
        assert _infer_type_from_value({}) == dict

    def test_infer_multimodal_types(self):
        """Test inferring multimodal types."""
        img = Image(data=b"fake", format="png")
        assert _infer_type_from_value(img) == Image

        audio = Audio(data=b"fake", format="mp3")
        assert _infer_type_from_value(audio) == Audio

        tool = Tool("test", "desc", {})
        assert _infer_type_from_value(tool) == Tool

        history = History([])
        assert _infer_type_from_value(history) == History
