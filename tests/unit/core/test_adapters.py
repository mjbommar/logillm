"""Unit tests for adapter system."""

import pytest

from logillm.core.adapters import (
    AdapterChain,
    ChatAdapter,
    JSONAdapter,
    MarkdownAdapter,
    ParseError,
    TextAdapter,
    XMLAdapter,
    create_adapter,
)
from logillm.core.signatures import BaseSignature, FieldSpec, FieldType


@pytest.fixture
def sample_signature():
    """Create a sample signature for testing."""
    return BaseSignature(
        input_fields={
            "question": FieldSpec(
                name="question", field_type=FieldType.INPUT, description="The question to answer"
            ),
            "context": FieldSpec(
                name="context", field_type=FieldType.INPUT, description="Optional context"
            ),
        },
        output_fields={
            "answer": FieldSpec(
                name="answer", field_type=FieldType.OUTPUT, description="The answer"
            ),
            "confidence": FieldSpec(
                name="confidence", field_type=FieldType.OUTPUT, description="Confidence level"
            ),
        },
        instructions="Answer the question based on the context",
    )


@pytest.fixture
def sample_inputs():
    """Sample input data."""
    return {
        "question": "What is 2+2?",
        "context": "Basic arithmetic",
    }


@pytest.fixture
def sample_demos():
    """Sample demonstrations."""
    return [
        {
            "inputs": {"question": "What is 1+1?", "context": "Math"},
            "outputs": {"answer": "2", "confidence": "high"},
        },
        {
            "inputs": {"question": "What is 3+3?", "context": "Math"},
            "outputs": {"answer": "6", "confidence": "high"},
        },
    ]


class TestChatAdapter:
    """Test ChatAdapter functionality."""

    def test_format_prompt(self, sample_signature, sample_inputs):
        """Test formatting a prompt."""
        adapter = ChatAdapter()
        prompt = adapter.format_prompt(sample_signature, sample_inputs)

        assert isinstance(prompt, str)
        assert "question" in prompt.lower()
        assert "2+2" in prompt
        assert "context" in prompt.lower()

    def test_format_prompt_with_demos(self, sample_signature, sample_inputs, sample_demos):
        """Test formatting with demonstrations."""
        adapter = ChatAdapter()
        prompt = adapter.format_prompt(sample_signature, sample_inputs, sample_demos)

        assert isinstance(prompt, str)
        assert "1+1" in prompt  # From demo
        assert "2+2" in prompt  # From input

    def test_parse_response(self, sample_signature):
        """Test parsing a response."""
        adapter = ChatAdapter()

        # Test with key-value format
        response = "answer: 4\nconfidence: high"
        parsed = adapter.parse_response(sample_signature, response)

        assert isinstance(parsed, dict)
        assert "answer" in parsed
        assert parsed["answer"] == "4"
        assert "confidence" in parsed
        assert parsed["confidence"] == "high"

    def test_parse_response_freeform(self, sample_signature):
        """Test parsing freeform text."""
        adapter = ChatAdapter()
        response = "The answer is 4 with high confidence."

        parsed = adapter.parse_response(sample_signature, response)
        assert isinstance(parsed, dict)
        # Should extract something, even if not perfect
        assert len(parsed) > 0


class TestJSONAdapter:
    """Test JSONAdapter functionality."""

    def test_format_prompt(self, sample_signature, sample_inputs):
        """Test JSON prompt formatting."""
        adapter = JSONAdapter()
        prompt = adapter.format_prompt(sample_signature, sample_inputs)

        assert isinstance(prompt, str)
        assert "JSON" in prompt
        assert "question" in prompt
        assert "2+2" in prompt

    def test_parse_json_response(self, sample_signature):
        """Test parsing JSON response."""
        adapter = JSONAdapter()
        response = '{"answer": "4", "confidence": "high"}'

        parsed = adapter.parse_response(sample_signature, response)
        assert parsed == {"answer": "4", "confidence": "high"}

    def test_parse_json_with_markdown(self, sample_signature):
        """Test parsing JSON in markdown code block."""
        adapter = JSONAdapter()
        response = '```json\n{"answer": "4", "confidence": "high"}\n```'

        parsed = adapter.parse_response(sample_signature, response)
        assert parsed == {"answer": "4", "confidence": "high"}

    def test_parse_invalid_json(self, sample_signature):
        """Test parsing invalid JSON raises error."""
        adapter = JSONAdapter()
        response = "not json at all"

        with pytest.raises(ParseError):
            adapter.parse_response(sample_signature, response)


class TestMarkdownAdapter:
    """Test MarkdownAdapter functionality."""

    def test_format_prompt(self, sample_signature, sample_inputs):
        """Test Markdown prompt formatting."""
        adapter = MarkdownAdapter()
        prompt = adapter.format_prompt(sample_signature, sample_inputs)

        assert isinstance(prompt, str)
        assert "#" in prompt  # Headers
        assert "**" in prompt or "##" in prompt  # Bold or headers

    def test_parse_markdown_sections(self, sample_signature):
        """Test parsing markdown with sections."""
        adapter = MarkdownAdapter()
        response = """
## Answer
4

## Confidence
high
"""
        parsed = adapter.parse_response(sample_signature, response)
        assert "answer" in parsed
        assert "confidence" in parsed

    def test_parse_markdown_bold(self, sample_signature):
        """Test parsing markdown with bold labels."""
        adapter = MarkdownAdapter()
        response = """
**answer**: 4
**confidence**: high
"""
        parsed = adapter.parse_response(sample_signature, response)
        assert "answer" in parsed
        assert parsed["answer"] == "4"


class TestXMLAdapter:
    """Test XMLAdapter functionality."""

    def test_format_prompt(self, sample_signature, sample_inputs):
        """Test XML prompt formatting."""
        adapter = XMLAdapter()
        prompt = adapter.format_prompt(sample_signature, sample_inputs)

        assert isinstance(prompt, str)
        assert "<" in prompt and ">" in prompt
        assert "XML" in prompt or "xml" in prompt

    def test_parse_xml_response(self, sample_signature):
        """Test parsing XML response."""
        adapter = XMLAdapter()
        response = """
<output>
    <answer>4</answer>
    <confidence>high</confidence>
</output>
"""
        parsed = adapter.parse_response(sample_signature, response)
        assert parsed["answer"] == "4"
        assert parsed["confidence"] == "high"

    def test_parse_xml_without_root(self, sample_signature):
        """Test parsing XML without root element."""
        adapter = XMLAdapter()
        response = "<answer>4</answer><confidence>high</confidence>"

        parsed = adapter.parse_response(sample_signature, response)
        assert parsed["answer"] == "4"

    def test_parse_invalid_xml(self, sample_signature):
        """Test parsing invalid XML."""
        adapter = XMLAdapter()
        response = "not xml at all"

        with pytest.raises(ParseError):
            adapter.parse_response(sample_signature, response)


class TestTextAdapter:
    """Test TextAdapter functionality."""

    def test_format_prompt(self, sample_signature, sample_inputs):
        """Test text prompt formatting."""
        adapter = TextAdapter()
        prompt = adapter.format_prompt(sample_signature, sample_inputs)

        assert isinstance(prompt, str)
        assert "question: What is 2+2?" in prompt
        assert "→" in prompt  # Arrow separator

    def test_parse_single_field(self):
        """Test parsing with single output field."""
        sig = BaseSignature(
            input_fields={
                "q": FieldSpec(name="q", field_type=FieldType.INPUT, description="Question")
            },
            output_fields={
                "a": FieldSpec(name="a", field_type=FieldType.OUTPUT, description="Answer")
            },
        )

        adapter = TextAdapter()
        response = "The answer is 4"
        parsed = adapter.parse_response(sig, response)

        assert "a" in parsed
        assert parsed["a"] == "The answer is 4"

    def test_parse_key_value(self, sample_signature):
        """Test parsing key-value format."""
        adapter = TextAdapter()
        response = "answer: 4\nconfidence: high"

        parsed = adapter.parse_response(sample_signature, response)
        assert "answer" in parsed
        assert "confidence" in parsed


class TestAdapterChain:
    """Test AdapterChain functionality."""

    def test_chain_fallback(self, sample_signature):
        """Test chain falls back to next adapter on parse error."""
        chain = AdapterChain(
            [
                JSONAdapter(),  # Will fail on non-JSON
                TextAdapter(),  # Should succeed
            ]
        )

        # Include both fields so validation passes
        response = "answer: 4\nconfidence: high"
        parsed = chain.parse_response(sample_signature, response)

        assert "answer" in parsed
        assert parsed["answer"] == "4"

    def test_chain_primary(self, sample_signature, sample_inputs):
        """Test chain uses primary adapter for formatting."""
        chain = AdapterChain([JSONAdapter(), TextAdapter()])

        prompt = chain.format_prompt(sample_signature, sample_inputs)
        assert "JSON" in prompt  # Should use JSONAdapter

    def test_set_primary(self, sample_signature, sample_inputs):
        """Test setting primary adapter."""
        chain = AdapterChain([JSONAdapter(), ChatAdapter(), TextAdapter()])

        chain.set_primary("chat")
        chain.format_prompt(sample_signature, sample_inputs)

        # ChatAdapter should now be primary
        assert chain.primary_adapter == chain.adapters[0]
        assert isinstance(chain.adapters[0], ChatAdapter)


class TestCreateAdapter:
    """Test adapter factory function."""

    def test_create_chat_adapter(self):
        """Test creating chat adapter."""
        adapter = create_adapter("chat")
        assert isinstance(adapter, ChatAdapter)

    def test_create_json_adapter(self):
        """Test creating JSON adapter."""
        adapter = create_adapter("json")
        assert isinstance(adapter, JSONAdapter)

    def test_create_with_enum(self):
        """Test creating with AdapterFormat enum."""
        from logillm.core.types import AdapterFormat

        adapter = create_adapter(AdapterFormat.MARKDOWN)
        assert isinstance(adapter, MarkdownAdapter)

    def test_default_adapter(self):
        """Test default adapter is chat."""
        adapter = create_adapter()
        assert isinstance(adapter, ChatAdapter)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
