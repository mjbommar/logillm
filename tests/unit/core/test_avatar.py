"""Tests for Avatar module."""

import pytest

from logillm.core.avatar import Action, ActionOutput, Avatar, parse_action
from logillm.core.signatures import BaseSignature, FieldSpec
from logillm.core.tools.base import Tool
from logillm.core.types import FieldType
from logillm.providers.mock import MockProvider


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""

    def calculator(expression: str) -> str:
        """Simple calculator that evaluates expressions."""
        try:
            # Simple math evaluation
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def search(query: str) -> str:
        """Mock search tool."""
        return f"Search results for: {query}"

    calc_tool = Tool(func=calculator, name="Calculator", desc="Evaluates mathematical expressions")
    search_tool = Tool(func=search, name="Search", desc="Searches for information")

    return [calc_tool, search_tool]


@pytest.fixture
def simple_signature():
    """Create a simple signature for testing."""
    return BaseSignature(
        input_fields={
            "question": FieldSpec(
                name="question",
                field_type=FieldType.INPUT,
                python_type=str,
                description="Question to answer",
                required=True,
            ),
        },
        output_fields={
            "answer": FieldSpec(
                name="answer",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description="Answer to the question",
                required=True,
            ),
        },
        instructions="Answer the question using available tools",
    )


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    return MockProvider()


class TestAction:
    """Test Action dataclass."""

    def test_action_creation(self):
        """Test creating an Action."""
        action = Action(tool_name="Calculator", tool_input_query="2 + 2")
        assert action.tool_name == "Calculator"
        assert action.tool_input_query == "2 + 2"

    def test_action_str(self):
        """Test Action string representation."""
        action = Action(tool_name="Calculator", tool_input_query="2 + 2")
        assert "Calculator" in str(action)
        assert "2 + 2" in str(action)


class TestActionOutput:
    """Test ActionOutput dataclass."""

    def test_action_output_creation(self):
        """Test creating an ActionOutput."""
        output = ActionOutput(tool_name="Calculator", tool_input_query="2 + 2", tool_output="4")
        assert output.tool_name == "Calculator"
        assert output.tool_input_query == "2 + 2"
        assert output.tool_output == "4"

    def test_action_output_str(self):
        """Test ActionOutput string representation."""
        output = ActionOutput(tool_name="Calculator", tool_input_query="2 + 2", tool_output="4")
        assert "Calculator" in str(output)
        assert "4" in str(output)


class TestParseAction:
    """Test action parsing function."""

    def test_parse_action_with_parentheses(self):
        """Test parsing action with parentheses."""
        action = parse_action("Calculator(2 + 2)")
        assert action.tool_name == "Calculator"
        assert action.tool_input_query == "2 + 2"

    def test_parse_action_without_parentheses(self):
        """Test parsing action without parentheses."""
        action = parse_action("Calculator")
        assert action.tool_name == "Calculator"
        assert action.tool_input_query == ""

    def test_parse_action_finish(self):
        """Test parsing finish action."""
        action = parse_action("Finish(The answer is 42)")
        assert action.tool_name == "Finish"
        assert action.tool_input_query == "The answer is 42"

    def test_parse_action_with_spaces(self):
        """Test parsing action with extra spaces."""
        action = parse_action("  Calculator  ( 2 + 2 )  ")
        assert action.tool_name == "Calculator"
        assert action.tool_input_query == "2 + 2"

    def test_parse_action_no_closing_paren(self):
        """Test parsing action with no closing parenthesis."""
        action = parse_action("Calculator(2 + 2")
        assert action.tool_name == "Calculator"
        assert action.tool_input_query == "2 + 2"


class TestAvatar:
    """Test Avatar module."""

    def test_avatar_initialization(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar initialization."""
        avatar = Avatar(
            signature=simple_signature, tools=mock_tools, max_iters=5, provider=mock_provider
        )

        assert avatar.signature == simple_signature
        assert len(avatar.tools) == 3  # 2 mock tools + finish tool
        assert avatar.max_iters == 5
        assert any(tool.name == "Finish" for tool in avatar.tools)

    def test_avatar_initialization_string_signature(self, mock_tools, mock_provider):
        """Test Avatar initialization with string signature."""
        avatar = Avatar(signature="question -> answer", tools=mock_tools, provider=mock_provider)

        assert avatar.signature is not None
        assert len(avatar.tools) == 3  # 2 mock tools + finish tool

    def test_avatar_parameters(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar parameters for optimization."""
        avatar = Avatar(
            signature=simple_signature, tools=mock_tools, max_iters=5, provider=mock_provider
        )

        assert "tools" in avatar.parameters
        assert "max_iters" in avatar.parameters
        assert avatar.parameters["max_iters"].value == 5
        assert avatar.parameters["max_iters"].learnable is True

    def test_call_tool_calculator(self, simple_signature, mock_tools, mock_provider):
        """Test calling calculator tool."""
        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        result = avatar._call_tool("Calculator", "2 + 3")
        assert result == "5"

    def test_call_tool_search(self, simple_signature, mock_tools, mock_provider):
        """Test calling search tool."""
        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        result = avatar._call_tool("Search", "Python programming")
        assert result == "Search results for: Python programming"

    def test_call_tool_finish(self, simple_signature, mock_tools, mock_provider):
        """Test calling finish tool."""
        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        result = avatar._call_tool("Finish", "Final answer")
        assert result == "Final answer"

    def test_call_tool_not_found(self, simple_signature, mock_tools, mock_provider):
        """Test calling non-existent tool."""
        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        result = avatar._call_tool("NonExistent", "test")
        assert "not found" in result
        assert "Available tools" in result

    def test_call_tool_error(self, simple_signature, mock_provider):
        """Test tool execution error."""

        def error_tool(input_str: str) -> str:
            raise ValueError("Test error")

        error_tool_obj = Tool(func=error_tool, name="ErrorTool", desc="Tool that throws errors")
        avatar = Avatar(signature=simple_signature, tools=[error_tool_obj], provider=mock_provider)

        result = avatar._call_tool("ErrorTool", "test")
        assert "Error executing tool" in result
        assert "Test error" in result

    @pytest.mark.asyncio
    async def test_forward_simple_finish(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar forward with immediate finish."""
        # Set up mock provider to return appropriate responses
        mock_provider.set_mock_response("Finish(The answer is 42)")

        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        result = await avatar(question="What is the answer?")

        assert result.success is True
        assert "answer" in result.outputs
        assert result.outputs["answer"] == "The answer is 42"
        assert hasattr(result, "actions")

    @pytest.mark.asyncio
    async def test_forward_with_tool_use(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar forward with tool usage before finish."""
        # Set up mock provider to return calculator use, then finish
        mock_provider.set_mock_response(["Calculator(2 + 2)", "Finish(The result is 4)"])

        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        result = await avatar(question="What is 2 + 2?")

        assert result.success is True
        assert len(result.actions) == 1  # One non-finish action
        assert result.actions[0].tool_name == "Calculator"
        assert result.actions[0].tool_output == "4"

    @pytest.mark.asyncio
    async def test_forward_max_iters(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar respects max_iters limit."""
        # Set up mock provider to return a calculator action (never finish)
        # Use only 1 iteration to avoid complex signature issues
        mock_provider.set_mock_response("Calculator(1 + 1)")

        avatar = Avatar(
            signature=simple_signature, tools=mock_tools, max_iters=1, provider=mock_provider
        )
        result = await avatar(question="Calculate something")

        assert result.success is True
        assert len(result.actions) == 1  # Should stop at max_iters
        assert result.actions[0].tool_name == "Calculator"
        assert result.actions[0].tool_output == "2"

    @pytest.mark.asyncio
    async def test_forward_with_error(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar handles errors gracefully."""
        # Set up mock provider to return an invalid action that will cause an error
        mock_provider.set_mock_response("InvalidAction(This will cause parse error)")

        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)
        result = await avatar(question="Test question")

        # Should still succeed by breaking out of loop on error
        assert result.success is True
        # Should have gotten an error but still returned something
        assert "answer" in result.outputs or "output" in result.outputs

    @pytest.mark.asyncio
    async def test_forward_verbose(self, simple_signature, mock_tools, mock_provider, capsys):
        """Test Avatar verbose output."""
        # Set up mock provider to return a finish action
        mock_provider.set_mock_response("Finish(Done)")

        avatar = Avatar(
            signature=simple_signature, tools=mock_tools, verbose=True, provider=mock_provider
        )
        await avatar(question="Test question")

        captured = capsys.readouterr()
        assert "Starting Avatar task execution" in captured.out

    def test_update_actor_signature(self, simple_signature, mock_tools, mock_provider):
        """Test signature updates during execution."""
        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        initial_input_fields = len(avatar.actor_signature.input_fields)
        initial_output_fields = len(avatar.actor_signature.output_fields)

        # Update signature for first action
        avatar._update_actor_signature(1)

        # Should add action_1, result_1 inputs and action_2 output
        assert len(avatar.actor_signature.input_fields) == initial_input_fields + 2
        assert len(avatar.actor_signature.output_fields) == initial_output_fields + 1
        assert "action_1" in avatar.actor_signature.input_fields
        assert "result_1" in avatar.actor_signature.input_fields
        assert "action_2" in avatar.actor_signature.output_fields

    def test_update_actor_signature_omit_action(self, simple_signature, mock_tools, mock_provider):
        """Test signature update with omit_action=True."""
        avatar = Avatar(signature=simple_signature, tools=mock_tools, provider=mock_provider)

        initial_input_fields = len(avatar.actor_signature.input_fields)

        # Update signature for final action
        avatar._update_actor_signature(1, omit_action=True)

        # Should add action_1, result_1 inputs and original output fields
        assert len(avatar.actor_signature.input_fields) == initial_input_fields + 2
        assert "answer" in avatar.actor_signature.output_fields  # Original output field

    def test_to_dict_serialization(self, simple_signature, mock_tools, mock_provider):
        """Test Avatar serialization."""
        avatar = Avatar(
            signature=simple_signature, tools=mock_tools, max_iters=5, provider=mock_provider
        )

        data = avatar.to_dict()

        assert data["type"] == "Avatar"
        assert "signature" in data
        assert "parameters" in data
        assert "tools" in data["parameters"]
        assert "max_iters" in data["parameters"]
