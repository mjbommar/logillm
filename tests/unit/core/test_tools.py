"""Tests for the tool system."""

import asyncio

import pytest

from logillm.core.tools import Tool, ToolRegistry, ToolResult, tool
from logillm.core.tools.base import calculator_tool, search_tool


class TestTool:
    """Test the Tool class."""

    def test_init_with_simple_function(self):
        """Test Tool initialization with a simple function."""

        def add(a: int, b: int = 5) -> int:
            """Add two numbers."""
            return a + b

        t = Tool(add)

        assert t.name == "add"
        assert t.desc == "Add two numbers."
        assert "a" in t.args
        assert "b" in t.args
        assert t.args["a"]["type"] == "integer"
        assert t.args["b"]["type"] == "integer"
        assert t.args["b"]["default"] == 5
        assert t.arg_types["a"] is int
        assert t.arg_types["b"] is int

    def test_init_with_custom_metadata(self):
        """Test Tool initialization with custom metadata."""

        def multiply(x: float, y: float) -> float:
            return x * y

        t = Tool(
            multiply,
            name="custom_multiply",
            desc="Custom multiplication tool",
            arg_desc={"x": "First number", "y": "Second number"},
        )

        assert t.name == "custom_multiply"
        assert t.desc == "Custom multiplication tool"
        assert t.args["x"]["description"] == "First number"
        assert t.args["y"]["description"] == "Second number"

    def test_function_parsing_with_type_hints(self):
        """Test parsing function with various type hints."""

        def complex_func(
            text: str,
            count: int = 3,
            active: bool = True,
            items: list = None,
            metadata: dict = None,
        ) -> dict:
            """Complex function with various types."""
            return {"text": text, "count": count, "active": active}

        t = Tool(complex_func)

        assert t.args["text"]["type"] == "string"
        assert t.args["count"]["type"] == "integer"
        assert t.args["count"]["default"] == 3
        assert t.args["active"]["type"] == "boolean"
        assert t.args["active"]["default"] is True
        assert t.args["items"]["type"] == "array"
        assert t.args["metadata"]["type"] == "object"

    def test_function_without_type_hints(self):
        """Test parsing function without type hints."""

        def no_hints(param1, param2="default"):
            """Function without type hints."""
            return f"{param1}-{param2}"

        t = Tool(no_hints)

        # Should default to string for unknown types
        assert t.args["param1"]["type"] == "string"
        assert t.args["param2"]["type"] == "string"
        assert t.args["param2"]["default"] == "default"

    def test_validation_and_parsing(self):
        """Test argument validation and parsing."""

        def typed_func(num: int, text: str = "hello") -> str:
            return f"{text}: {num}"

        t = Tool(typed_func)

        # Valid arguments
        parsed = t._validate_and_parse_args(num=42, text="world")
        assert parsed == {"num": 42, "text": "world"}

        # Missing optional argument (should use default)
        parsed = t._validate_and_parse_args(num=42)
        assert parsed == {"num": 42, "text": "hello"}

        # Type conversion
        parsed = t._validate_and_parse_args(num="123", text="converted")
        assert parsed == {"num": 123, "text": "converted"}

        # Missing required argument
        with pytest.raises(ValueError, match="Required argument 'num'"):
            t._validate_and_parse_args(text="only text")

        # Invalid argument name
        with pytest.raises(ValueError, match="Argument 'invalid'"):
            t._validate_and_parse_args(num=42, invalid="bad")

    def test_sync_execution(self):
        """Test synchronous tool execution."""

        def add_sync(a: int, b: int) -> int:
            return a + b

        t = Tool(add_sync)

        # Test __call__
        result = t(a=3, b=7)
        assert result == 10

        # Test execute_sync
        result = t.execute_sync(a=5, b=3)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == 8
        assert result.error is None

    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test asynchronous tool execution."""

        async def add_async(a: int, b: int) -> int:
            await asyncio.sleep(0.001)  # Simulate async work
            return a + b

        t = Tool(add_async)

        # Test acall
        result = await t.acall(a=3, b=7)
        assert result == 10

        # Test execute
        result = await t.execute(a=5, b=3)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == 8
        assert result.error is None

    @pytest.mark.asyncio
    async def test_mixed_sync_async_execution(self):
        """Test calling sync function from async context."""

        def sync_func(x: int) -> int:
            return x * 2

        t = Tool(sync_func)

        # Sync function should work in async context
        result = await t.acall(x=5)
        assert result == 10

        result = await t.execute(x=7)
        assert result.success is True
        assert result.output == 14

    def test_error_handling(self):
        """Test error handling in tool execution."""

        def error_func(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Intentional error")
            return "success"

        t = Tool(error_func)

        # Successful execution
        result = t.execute_sync(should_fail=False)
        assert result.success is True
        assert result.output == "success"

        # Failed execution
        result = t.execute_sync(should_fail=True)
        assert result.success is False
        assert result.error == "Intentional error"
        assert result.output is None

    def test_format_for_llm(self):
        """Test LLM format generation."""

        def search(query: str, max_results: int = 5) -> list:
            """Search for information."""
            return []

        t = Tool(search)

        schema = t.format_for_llm()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for information."

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["max_results"]["type"] == "integer"
        assert "query" in params["required"]
        assert "max_results" not in params["required"]  # Has default

    def test_string_representations(self):
        """Test string representations."""

        def example_func(param: str) -> str:
            """Example function for testing."""
            return param

        t = Tool(example_func)

        # Test __str__
        str_repr = str(t)
        assert "example_func" in str_repr
        assert "Example function for testing" in str_repr
        assert "param" in str_repr

        # Test __repr__
        repr_str = repr(t)
        assert "Tool(" in repr_str
        assert "example_func" in repr_str
        assert "param" in repr_str

    def test_serialization(self):
        """Test tool serialization."""

        def example(a: int, b: str = "default") -> str:
            """Example function."""
            return f"{b}: {a}"

        t = Tool(example)
        data = t.to_dict()

        assert data["name"] == "example"
        assert data["description"] == "Example function."
        assert "a" in data["args"]
        assert "b" in data["args"]
        assert data["has_kwargs"] is False


class TestToolDecorator:
    """Test the @tool decorator."""

    def test_basic_decorator(self):
        """Test basic tool decoration."""

        @tool()
        def decorated_func(x: int) -> int:
            """A decorated function."""
            return x * 2

        assert isinstance(decorated_func, Tool)
        assert decorated_func.name == "decorated_func"
        assert decorated_func.desc == "A decorated function."

        result = decorated_func(x=5)
        assert result == 10

    def test_decorator_with_params(self):
        """Test decorator with custom parameters."""

        @tool(name="custom_name", desc="Custom description")
        def another_func(value: str) -> str:
            return value.upper()

        assert another_func.name == "custom_name"
        assert another_func.desc == "Custom description"

        result = another_func(value="hello")
        assert result == "HELLO"


class TestBuiltinTools:
    """Test built-in tools."""

    def test_calculator_tool(self):
        """Test the calculator tool."""
        result = calculator_tool("2 + 3")
        assert result == 5

        result = calculator_tool("10 * 4")
        assert result == 40

        result = calculator_tool("16 / 4")
        assert result == 4

        # Test with expressions
        result = calculator_tool("(2 + 3) * 4")
        assert result == 20

        # Test error handling
        with pytest.raises(Exception):  # Should fail on invalid expression
            calculator_tool("invalid expression")

    def test_search_tool(self):
        """Test the search tool."""
        results = search_tool("test query")
        assert isinstance(results, list)
        assert len(results) == 5  # Default max_results
        assert all("test query" in result for result in results)

        # Test with custom max_results
        results = search_tool("another query", max_results=3)
        assert len(results) == 3


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_init(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.get_stats()["total_tools"] == 0

    def test_register_and_get(self):
        """Test tool registration and retrieval."""
        registry = ToolRegistry()

        def test_func(x: int) -> int:
            return x + 1

        tool = Tool(test_func)
        registry.register(tool)

        assert len(registry) == 1
        assert "test_func" in registry
        retrieved = registry.get("test_func")
        assert retrieved is tool

        # Test duplicate registration
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_categorization(self):
        """Test tool categorization."""
        registry = ToolRegistry()

        def math_func(x: int) -> int:
            return x * 2

        def text_func(s: str) -> str:
            return s.upper()

        math_tool = Tool(math_func)
        text_tool = Tool(text_func)

        registry.register(math_tool, category="math", tags=["arithmetic", "calculation"])
        registry.register(text_tool, category="text", tags=["string", "formatting"])

        # Test category filtering
        math_tools = registry.list_tools(category="math")
        assert "math_func" in math_tools
        assert "text_func" not in math_tools

        # Test tag filtering
        string_tools = registry.list_tools(tag="string")
        assert "text_func" in string_tools
        assert "math_func" not in string_tools

        # Test category and tag intersection
        calc_tools = registry.list_tools(category="math", tag="calculation")
        assert "math_func" in calc_tools
        assert "text_func" not in calc_tools

    def test_search(self):
        """Test tool search functionality."""
        registry = ToolRegistry()

        def calculator(expr: str) -> float:
            """Calculate mathematical expressions."""
            return 0.0

        def search_web(query: str) -> list:
            """Search the web for information."""
            return []

        registry.register(Tool(calculator))
        registry.register(Tool(search_web))

        # Search by name
        results = registry.search_tools("calc")
        assert len(results) == 1
        assert results[0].name == "calculator"

        # Search by description
        results = registry.search_tools("mathematical")
        assert len(results) == 1
        assert results[0].name == "calculator"

        # Search with no matches
        results = registry.search_tools("nonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execution(self):
        """Test tool execution through registry."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        registry.register(Tool(add))

        # Test successful execution
        result = await registry.execute("add", a=3, b=7)
        assert result.success is True
        assert result.output == 10

        # Test sync execution
        result = registry.execute_sync("add", a=5, b=2)
        assert result.success is True
        assert result.output == 7

        # Test nonexistent tool
        result = await registry.execute("nonexistent", x=1)
        assert result.success is False
        assert "not found" in result.error

    def test_llm_formatting(self):
        """Test LLM format generation."""
        registry = ToolRegistry()

        def example_tool(param: str) -> str:
            """Example tool for testing."""
            return param

        registry.register(Tool(example_tool), category="test")

        # Test format for all tools
        schemas = registry.format_for_llm()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "example_tool"

        # Test format for specific category
        schemas = registry.format_for_llm(category="test")
        assert len(schemas) == 1

        # Test format for non-existent category
        schemas = registry.format_for_llm(category="nonexistent")
        assert len(schemas) == 0

    def test_prompt_formatting(self):
        """Test prompt format generation."""
        registry = ToolRegistry()

        def tool1(x: int) -> int:
            """First tool."""
            return x

        def tool2(y: str) -> str:
            """Second tool."""
            return y

        registry.register(Tool(tool1))
        registry.register(Tool(tool2))

        prompt = registry.format_for_prompt()
        assert "Available tools:" in prompt
        assert "tool1" in prompt
        assert "tool2" in prompt
        assert "First tool." in prompt
        assert "Second tool." in prompt

    def test_unregister_and_clear(self):
        """Test tool unregistration and clearing."""
        registry = ToolRegistry()

        def func1(x: int) -> int:
            return x

        def func2(y: str) -> str:
            return y

        registry.register(Tool(func1), category="cat1")
        registry.register(Tool(func2), category="cat2")

        assert len(registry) == 2

        # Test unregister
        registry.unregister("func1")
        assert len(registry) == 1
        assert "func1" not in registry
        assert "func2" in registry

        # Test clear specific category
        registry.register(Tool(func1), category="cat1")
        registry.clear(category="cat1")
        assert len(registry) == 1
        assert "func2" in registry

        # Test clear all
        registry.clear()
        assert len(registry) == 0

    def test_validation(self):
        """Test registry validation."""
        registry = ToolRegistry()

        def test_func() -> str:
            return "test"

        tool = Tool(test_func)
        registry.register(tool, category="test", tags=["testing"])

        # Should be valid initially
        errors = registry.validate()
        assert len(errors) == 0

        # Manually corrupt the registry to test validation
        registry.categories["bad_category"].add("nonexistent_tool")
        errors = registry.validate()
        assert len(errors) > 0
        assert any("nonexistent_tool" in error for error in errors)

    def test_stats(self):
        """Test registry statistics."""
        registry = ToolRegistry()

        def func1() -> str:
            return "1"

        def func2() -> str:
            return "2"

        def func3() -> str:
            return "3"

        registry.register(Tool(func1), category="cat1", tags=["tag1"])
        registry.register(Tool(func2), category="cat1", tags=["tag2"])
        registry.register(Tool(func3), category="cat2", tags=["tag1", "tag2"])

        stats = registry.get_stats()
        assert stats["total_tools"] == 3
        assert stats["categories"] == 2
        assert stats["tags"] == 2
        assert stats["tools_by_category"]["cat1"] == 2
        assert stats["tools_by_category"]["cat2"] == 1
        assert stats["tools_by_tag"]["tag1"] == 2
        assert stats["tools_by_tag"]["tag2"] == 2

    def test_serialization(self):
        """Test registry serialization."""
        registry = ToolRegistry()

        def test_func(x: int) -> int:
            return x

        registry.register(Tool(test_func), category="test", tags=["testing"])

        data = registry.to_dict()
        assert "tools" in data
        assert "categories" in data
        assert "tags" in data
        assert "metadata" in data

        assert "test_func" in data["tools"]
        assert "test" in data["categories"]
        assert "testing" in data["tags"]
