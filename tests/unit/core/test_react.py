"""Tests for the enhanced ReAct module."""

from unittest.mock import AsyncMock, Mock

import pytest

from logillm.core.react import ReAct, ReActStep, ReActTrajectory
from logillm.core.tools import Tool, ToolRegistry
from logillm.core.types import Prediction, Usage


class TestReActStep:
    """Test ReActStep dataclass."""

    def test_init(self):
        """Test step initialization."""
        step = ReActStep(
            thought="I need to search",
            tool_name="search",
            tool_args={"query": "test"},
            observation="Found results",
        )

        assert step.thought == "I need to search"
        assert step.tool_name == "search"
        assert step.tool_args == {"query": "test"}
        assert step.observation == "Found results"
        assert step.step_number == 0

    def test_to_dict(self):
        """Test step serialization."""
        step = ReActStep(
            thought="Think step",
            tool_name="tool",
            tool_args={"arg": "value"},
            observation="Observed",
            step_number=1,
        )

        data = step.to_dict()
        expected = {
            "thought": "Think step",
            "tool_name": "tool",
            "tool_args": {"arg": "value"},
            "observation": "Observed",
            "step_number": 1,
        }

        assert data == expected


class TestReActTrajectory:
    """Test ReActTrajectory dataclass."""

    def test_init(self):
        """Test trajectory initialization."""
        traj = ReActTrajectory()

        assert traj.steps == []
        assert traj.final_prediction == {}
        assert traj.success is False

    def test_add_step(self):
        """Test adding steps to trajectory."""
        traj = ReActTrajectory()

        step1 = ReActStep(thought="First thought")
        step2 = ReActStep(thought="Second thought")

        traj.add_step(step1)
        traj.add_step(step2)

        assert len(traj.steps) == 2
        assert traj.steps[0].step_number == 0
        assert traj.steps[1].step_number == 1

    def test_to_dict(self):
        """Test trajectory serialization."""
        traj = ReActTrajectory()
        step = ReActStep(thought="Test thought")
        traj.add_step(step)
        traj.final_prediction = {"answer": "Final answer"}
        traj.success = True

        data = traj.to_dict()

        assert len(data["steps"]) == 1
        assert data["steps"][0]["thought"] == "Test thought"
        assert data["final_prediction"] == {"answer": "Final answer"}
        assert data["success"] is True


class TestReAct:
    """Test the enhanced ReAct module."""

    def test_init_with_string_signature(self):
        """Test initialization with string signature."""

        def search_func(query: str) -> str:
            return f"Results for {query}"

        tools = [Tool(search_func)]
        react = ReAct("question -> answer", tools=tools)

        assert react.original_signature is not None
        assert "search_func" in react.tools
        assert "finish" in react.tools  # Auto-added finish tool
        assert react.max_iters == 10

    def test_init_with_tool_registry(self):
        """Test initialization with tool registry."""

        def calc_func(expr: str) -> float:
            return eval(expr)

        registry = ToolRegistry()
        registry.register(Tool(calc_func))

        react = ReAct("problem -> solution", tools=registry)

        assert "calc_func" in react.tools
        assert "finish" in react.tools

    def test_init_with_no_tools(self):
        """Test initialization without tools."""
        react = ReAct("input -> output")

        # Should only have the finish tool
        assert len(react.tools) == 1
        assert "finish" in react.tools

    def test_get_output_fields(self):
        """Test output field string generation."""
        react = ReAct("question -> answer")
        result = react._get_output_fields()
        assert result == "`answer`"

        # Test with multiple fields - create a mock signature
        react.original_signature = Mock()
        react.original_signature.output_fields = {"answer": "str", "confidence": "float"}
        result = react._get_output_fields()
        assert "`answer`" in result and "`confidence`" in result

    def test_format_trajectory(self):
        """Test trajectory formatting."""
        react = ReAct("test -> result")
        trajectory = ReActTrajectory()

        # Empty trajectory
        formatted = react._format_trajectory(trajectory)
        assert formatted == "No previous steps."

        # With steps
        step1 = ReActStep(
            thought="I should search",
            tool_name="search",
            tool_args={"query": "test"},
            observation="Found something",
        )
        step2 = ReActStep(
            thought="Now I'll finish", tool_name="finish", tool_args={}, observation="Done"
        )

        trajectory.add_step(step1)
        trajectory.add_step(step2)

        formatted = react._format_trajectory(trajectory)

        assert "Step 1:" in formatted
        assert "Step 2:" in formatted
        assert "I should search" in formatted
        assert "search" in formatted
        assert "Found something" in formatted

    def test_truncate_trajectory(self):
        """Test trajectory truncation."""
        react = ReAct("test -> result")
        trajectory = ReActTrajectory()

        # Add multiple steps
        for i in range(10):
            step = ReActStep(thought=f"Thought {i}")
            trajectory.add_step(step)

        # Truncate
        truncated = react._truncate_trajectory(trajectory)

        # Should keep about 70% (7 out of 10)
        assert len(truncated.steps) == 7
        assert truncated.steps[0].step_number == 0  # Renumbered
        assert truncated.steps[-1].step_number == 6

    def test_add_remove_tools(self):
        """Test dynamic tool management."""

        def search_func(query: str) -> str:
            return f"Results for {query}"

        react = ReAct("question -> answer")

        # Add tool
        tool = Tool(search_func)
        react.add_tool(tool)

        assert "search_func" in react.tools
        assert len(react.list_tools()) == 1

        # Remove tool
        react.remove_tool("search_func")

        assert "search_func" not in react.tools
        assert len(react.list_tools()) == 0

        # Can't remove finish tool
        react.remove_tool("finish")
        assert "finish" in react.tools

    @pytest.mark.asyncio
    async def test_forward_basic_flow(self):
        """Test basic ReAct forward flow with mocked components."""

        def add_func(a: int, b: int) -> int:
            return a + b

        tools = [Tool(add_func)]
        react = ReAct("problem -> answer", tools=tools)

        # Mock the internal predict modules
        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Mock reasoning steps - first use tool, then finish
        reasoning_steps = [
            # Step 1: Use add tool
            {
                "next_thought": "I need to add 2 and 3",
                "next_tool_name": "add_func",
                "next_tool_args": '{"a": 2, "b": 3}',
            },
            # Step 2: Finish
            {
                "next_thought": "I have the answer: 5",
                "next_tool_name": "finish",
                "next_tool_args": "{}",
            },
        ]

        react.react_predict.side_effect = [
            Prediction(outputs=step, success=True, usage=Usage()) for step in reasoning_steps
        ]

        # Mock final extraction
        react.extract_predict.return_value = Prediction(
            outputs={"answer": "The sum is 5"}, success=True, usage=Usage()
        )

        # Execute
        result = await react.forward(problem="What is 2 + 3?")

        assert result.success is True
        assert "answer" in result.outputs
        assert result.metadata["iterations_used"] == 2
        assert "add_func" in result.metadata["tools_called"]

        # Verify trajectory
        trajectory = result.metadata["trajectory"]
        assert len(trajectory["steps"]) == 2
        assert trajectory["steps"][0]["tool_name"] == "add_func"
        assert trajectory["steps"][0]["observation"] == "5"  # Tool executed
        assert trajectory["steps"][1]["tool_name"] == "finish"

    @pytest.mark.asyncio
    async def test_forward_finish_immediately(self):
        """Test agent choosing to finish immediately."""
        react = ReAct("question -> answer")

        # Mock the predict modules
        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Agent chooses to finish immediately
        react.react_predict.return_value = Prediction(
            outputs={
                "next_thought": "I can answer this directly",
                "next_tool_name": "finish",
                "next_tool_args": "{}",
            },
            success=True,
            usage=Usage(),
        )

        react.extract_predict.return_value = Prediction(
            outputs={"answer": "Direct answer"}, success=True, usage=Usage()
        )

        result = await react.forward(question="Simple question")

        assert result.success is True
        assert result.metadata["iterations_used"] == 1
        assert result.metadata["tools_called"] == []  # No real tools used

    @pytest.mark.asyncio
    async def test_forward_max_iterations(self):
        """Test hitting max iterations limit."""

        def search_func(query: str) -> str:
            return f"Results for {query}"

        tools = [Tool(search_func)]
        react = ReAct("question -> answer", tools=tools, max_iters=2)

        # Mock continuous search without finish
        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        react.react_predict.return_value = Prediction(
            outputs={
                "next_thought": "I need more information",
                "next_tool_name": "search_func",
                "next_tool_args": '{"query": "test"}',
            },
            success=True,
            usage=Usage(),
        )

        react.extract_predict.return_value = Prediction(
            outputs={"answer": "Extracted from trajectory"}, success=True, usage=Usage()
        )

        result = await react.forward(question="Complex question")

        # Should hit max iterations
        assert result.metadata["iterations_used"] == 2
        assert len(result.metadata["trajectory"]["steps"]) == 2

    @pytest.mark.asyncio
    async def test_forward_tool_error_handling(self):
        """Test tool execution error handling."""

        def error_func(should_fail: bool = True) -> str:
            if should_fail:
                raise ValueError("Tool error")
            return "Success"

        tools = [Tool(error_func)]
        react = ReAct("test -> result", tools=tools)

        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Agent chooses to use failing tool
        react.react_predict.return_value = Prediction(
            outputs={
                "next_thought": "I'll use the error tool",
                "next_tool_name": "error_func",
                "next_tool_args": '{"should_fail": true}',
            },
            success=True,
            usage=Usage(),
        )

        react.extract_predict.return_value = Prediction(
            outputs={"result": "Handled error gracefully"}, success=True, usage=Usage()
        )

        result = await react.forward(test="error test")

        # Should handle error gracefully
        trajectory = result.metadata["trajectory"]
        step = trajectory["steps"][0]
        assert "Tool error" in step["observation"]
        assert result.success is True  # Overall should still succeed

    @pytest.mark.asyncio
    async def test_forward_unknown_tool(self):
        """Test handling of unknown tool selection."""
        react = ReAct("test -> result")

        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Agent chooses unknown tool
        react.react_predict.return_value = Prediction(
            outputs={
                "next_thought": "I'll use a non-existent tool",
                "next_tool_name": "unknown_tool",
                "next_tool_args": '{"param": "value"}',
            },
            success=True,
            usage=Usage(),
        )

        react.extract_predict.return_value = Prediction(
            outputs={"result": "Handled unknown tool"}, success=True, usage=Usage()
        )

        result = await react.forward(test="unknown tool test")

        # Should provide helpful error message
        trajectory = result.metadata["trajectory"]
        step = trajectory["steps"][0]
        assert "Unknown tool 'unknown_tool'" in step["observation"]
        assert "finish" in step["observation"]  # Should list available tools

    @pytest.mark.asyncio
    async def test_forward_extraction_failure(self):
        """Test handling of final extraction failure."""
        react = ReAct("test -> result")

        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Agent finishes successfully
        react.react_predict.return_value = Prediction(
            outputs={
                "next_thought": "I'm done",
                "next_tool_name": "finish",
                "next_tool_args": "{}",
            },
            success=True,
            usage=Usage(),
        )

        # But extraction fails
        react.extract_predict.side_effect = Exception("Extraction failed")

        result = await react.forward(test="extraction failure test")

        # Should fallback gracefully
        assert result.success is False
        assert "I'm done" in str(result.outputs.values())  # Uses last thought

    def test_serialization(self):
        """Test ReAct serialization."""

        def test_func(x: int) -> int:
            return x * 2

        tools = [Tool(test_func)]
        react = ReAct("input -> output", tools=tools, max_iters=5)

        data = react.to_dict()

        assert data["max_iters"] == 5
        assert len(data["tools"]) == 1  # Doesn't include finish tool in serialization
        assert data["tools"][0]["name"] == "test_func"
        assert data["original_signature"] is not None

    @pytest.mark.asyncio
    async def test_forward_with_custom_max_iters(self):
        """Test forward with custom max_iters parameter."""
        react = ReAct("test -> result", max_iters=10)

        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Mock continuous execution without finish
        react.react_predict.return_value = Prediction(
            outputs={
                "next_thought": "Keep going",
                "next_tool_name": "nonexistent",
                "next_tool_args": "{}",
            },
            success=True,
            usage=Usage(),
        )

        react.extract_predict.return_value = Prediction(
            outputs={"result": "Done"}, success=True, usage=Usage()
        )

        # Override max_iters in forward call
        result = await react.forward(test="test", max_iters=3)

        # Should use the overridden value
        assert result.metadata["iterations_used"] == 3

    @pytest.mark.asyncio
    async def test_complex_reasoning_flow(self):
        """Test a complex multi-step reasoning flow."""

        def search_func(query: str) -> str:
            if "capital" in query:
                return "Paris is the capital of France"
            return "No results found"

        from logillm.core.tools.base import calculator_tool as calculator_func

        tools = [Tool(search_func), Tool(calculator_func)]
        react = ReAct("question -> answer", tools=tools)

        # Mock a complex flow: search, then calculate, then finish
        react.react_predict = AsyncMock()
        react.extract_predict = AsyncMock()

        # Define the sequence of reasoning steps
        reasoning_steps = [
            # Step 1: Search
            {
                "next_thought": "I need to search for the capital of France",
                "next_tool_name": "search_func",
                "next_tool_args": '{"query": "capital of France"}',
            },
            # Step 2: Calculate something with the info
            {
                "next_thought": "Now let me calculate the length of the city name",
                "next_tool_name": "calculator_tool",
                "next_tool_args": '{"expression": "len(\\"Paris\\")"}',
            },
            # Step 3: Finish
            {
                "next_thought": "I have the answer now",
                "next_tool_name": "finish",
                "next_tool_args": "{}",
            },
        ]

        # Set up side effects for the reasoning steps
        react.react_predict.side_effect = [
            Prediction(outputs=step, success=True, usage=Usage()) for step in reasoning_steps
        ]

        react.extract_predict.return_value = Prediction(
            outputs={"answer": "Paris is the capital and has 5 letters"},
            success=True,
            usage=Usage(),
        )

        result = await react.forward(
            question="What is the capital of France and how many letters does it have?"
        )

        assert result.success is True
        assert result.metadata["iterations_used"] == 3

        # Check tool usage
        tools_called = result.metadata["tools_called"]
        assert "search_func" in tools_called
        assert "calculator_tool" in tools_called

        # Check trajectory details
        trajectory = result.metadata["trajectory"]
        steps = trajectory["steps"]
        assert len(steps) == 3
        assert "Paris is the capital" in steps[0]["observation"]
        assert "5" in steps[1]["observation"]  # Length of "Paris"
