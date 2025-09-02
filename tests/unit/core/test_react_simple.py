"""Unit tests for the simplified ReAct implementation."""

from unittest.mock import AsyncMock, Mock
import pytest
from logillm.core.react import ReAct
from logillm.core.tools import Tool
from logillm.core.types import Prediction, Usage


class TestReAct:
    """Test the simplified ReAct module."""
    
    def test_init_with_tools(self):
        """Test initialization with tools."""
        def test_func(x: int) -> int:
            return x * 2
        
        tools = [Tool(test_func)]
        react = ReAct("input -> output", tools=tools)
        
        assert "test_func" in react.tools
        assert "finish" in react.tools  # Auto-added
        assert react.max_iters == 10
    
    def test_init_without_tools(self):
        """Test initialization without tools."""
        react = ReAct("input -> output")
        
        # Should only have finish tool
        assert len(react.tools) == 1
        assert "finish" in react.tools
    
    def test_format_trajectory_empty(self):
        """Test trajectory formatting with no steps."""
        react = ReAct("test -> result")
        formatted = react._format_trajectory({})
        assert formatted == "No previous steps."
    
    def test_format_trajectory_with_steps(self):
        """Test trajectory formatting with steps."""
        react = ReAct("test -> result")
        trajectory = {
            "thought_0": "I need to calculate",
            "tool_name_0": "calculator",
            "tool_args_0": {"expr": "2+2"},
            "observation_0": "4",
        }
        
        formatted = react._format_trajectory(trajectory)
        assert "Thought: I need to calculate" in formatted
        assert "Tool: calculator" in formatted
        assert "Observation: 4" in formatted
    
    @pytest.mark.asyncio
    async def test_forward_basic(self):
        """Test basic forward execution."""
        def calc(x: int) -> int:
            return x * 2
        
        tools = [Tool(calc)]
        react = ReAct("problem -> answer", tools=tools)
        
        # Mock the internal modules
        react.react = AsyncMock()
        react.extract = AsyncMock()
        
        # Mock single step that finishes
        react.react.return_value = Prediction(
            outputs={
                "next_thought": "I have the answer",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            success=True,
            usage=Usage(),
        )
        
        react.extract.return_value = Prediction(
            outputs={"answer": "42"},
            success=True,
            usage=Usage(),
        )
        
        result = await react.forward(problem="test")
        
        assert result.success
        assert result.outputs["answer"] == "42"
        assert result.metadata["steps"] == 1
    
    @pytest.mark.asyncio
    async def test_forward_with_tool_execution(self):
        """Test forward with actual tool execution."""
        def double(x: int) -> int:
            return x * 2
        
        tools = [Tool(double, name="double")]
        react = ReAct("input -> output", tools=tools)
        
        react.react = AsyncMock()
        react.extract = AsyncMock()
        
        # First call: use tool
        react.react.side_effect = [
            Prediction(
                outputs={
                    "next_thought": "Double the number",
                    "next_tool_name": "double",
                    "next_tool_args": {"x": 5},
                },
                success=True,
                usage=Usage(),
            ),
            Prediction(
                outputs={
                    "next_thought": "Done",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                },
                success=True,
                usage=Usage(),
            ),
        ]
        
        react.extract.return_value = Prediction(
            outputs={"output": "10"},
            success=True,
            usage=Usage(),
        )
        
        result = await react.forward(input="double 5")
        
        assert result.success
        assert result.metadata["trajectory"]["observation_0"] == "10"
    
    @pytest.mark.asyncio
    async def test_forward_unknown_tool(self):
        """Test handling of unknown tool."""
        react = ReAct("test -> result")
        
        react.react = AsyncMock()
        react.extract = AsyncMock()
        
        # Try to use non-existent tool
        react.react.return_value = Prediction(
            outputs={
                "next_thought": "Use unknown",
                "next_tool_name": "unknown_tool",
                "next_tool_args": {},
            },
            success=True,
            usage=Usage(),
        )
        
        react.extract.return_value = Prediction(
            outputs={"result": "handled"},
            success=True,
            usage=Usage(),
        )
        
        result = await react.forward(test="test")
        
        # Should handle gracefully
        assert "Unknown tool: unknown_tool" in result.metadata["trajectory"]["observation_0"]