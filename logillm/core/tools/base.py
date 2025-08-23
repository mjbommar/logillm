"""Advanced tool system for LogiLLM based on DSPy's implementation."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_origin, get_type_hints

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""

    output: Any
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool:
    """Tool class that wraps callable functions for LLM use.

    Inspired by DSPy's Tool implementation but adapted for LogiLLM's
    zero-dependency architecture and async patterns.
    """

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        desc: str | None = None,
        args: dict[str, Any] | None = None,
        arg_types: dict[str, Any] | None = None,
        arg_desc: dict[str, str] | None = None,
    ):
        """Initialize the Tool.

        Args:
            func: The actual function that is being wrapped by the tool
            name: The name of the tool (optional, inferred from function)
            desc: The description of the tool (optional, inferred from docstring)
            args: The args and their JSON schema (optional, inferred from signature)
            arg_types: The argument types (optional, inferred from type hints)
            arg_desc: Descriptions for each arg (optional)
        """
        self.func = func
        self.name = name
        self.desc = desc
        self.args = args
        self.arg_types = arg_types
        self.arg_desc = arg_desc or {}
        self.has_kwargs = False

        # Parse function to extract metadata
        self._parse_function(func, arg_desc)

    def _parse_function(self, func: Callable, arg_desc: dict[str, str] | None = None):
        """Parse function to extract name, description, and args.

        This automatically infers tool metadata from the function signature
        and type hints using only standard library introspection.
        """
        # Get the actual function to inspect
        annotations_func = (
            func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        )

        # Extract name
        name = getattr(func, "__name__", type(func).__name__)

        # Extract description from docstring
        desc = getattr(func, "__doc__", None) or getattr(annotations_func, "__doc__", "")
        if desc:
            desc = desc.strip()

        args = {}
        arg_types = {}

        # Use inspect.signature to get parameter information
        sig = inspect.signature(annotations_func)

        # Get type hints, defaulting to Any for missing hints
        try:
            available_hints = get_type_hints(annotations_func)
        except (NameError, AttributeError):
            # Handle cases where type hints can't be resolved
            available_hints = {}

        hints = {
            param_name: available_hints.get(param_name, Any) for param_name in sig.parameters.keys()
        }

        default_values = {
            param_name: sig.parameters[param_name].default for param_name in sig.parameters.keys()
        }

        # Process each argument to generate JSON schema
        for param_name, param_type in hints.items():
            if param_name == "return":
                continue

            arg_types[param_name] = param_type

            # Generate JSON schema from Python type
            schema = self._type_to_json_schema(param_type)
            args[param_name] = schema

            # Add default value if present
            if default_values[param_name] is not inspect.Parameter.empty:
                args[param_name]["default"] = default_values[param_name]

            # Add description if provided
            if arg_desc and param_name in arg_desc:
                args[param_name]["description"] = arg_desc[param_name]

        # Set attributes (use provided values or inferred ones)
        self.name = self.name or name
        self.desc = self.desc or desc
        self.args = self.args if self.args is not None else args
        self.arg_types = self.arg_types if self.arg_types is not None else arg_types
        self.has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

    def _type_to_json_schema(self, python_type: type) -> dict[str, Any]:
        """Convert Python type to JSON schema using only standard library.

        This replaces DSPy's Pydantic-based approach with pure Python.
        """
        # Handle common types
        if python_type is str:
            return {"type": "string"}
        elif python_type is int:
            return {"type": "integer"}
        elif python_type is float:
            return {"type": "number"}
        elif python_type is bool:
            return {"type": "boolean"}
        elif python_type is list:
            return {"type": "array"}
        elif python_type is dict:
            return {"type": "object"}
        elif python_type is Any:
            return {"type": "string"}  # Default to string for Any

        # Handle generic types
        origin = get_origin(python_type)
        if origin is list:
            return {"type": "array"}
        elif origin is dict:
            return {"type": "object"}
        elif origin is tuple:
            return {"type": "array"}
        elif origin is set:
            return {"type": "array"}

        # Handle Union types (like str | int)
        if hasattr(python_type, "__args__"):
            # For union types, default to string
            return {"type": "string"}

        # Default fallback
        return {"type": "string"}

    def _validate_and_parse_args(self, **kwargs) -> dict[str, Any]:
        """Validate arguments against the tool schema.

        This replaces DSPy's Pydantic validation with basic type checking
        using only standard library features.
        """
        parsed_kwargs = {}

        # Validate each provided argument
        for k, v in kwargs.items():
            if k not in self.args:
                if self.has_kwargs:
                    parsed_kwargs[k] = v
                    continue
                else:
                    raise ValueError(f"Argument '{k}' is not valid for tool '{self.name}'")

            # Basic type validation and conversion
            expected_type = self.arg_types.get(k, Any)
            if expected_type is not Any:
                try:
                    # Try to convert to expected type
                    if not isinstance(v, expected_type) and expected_type in (
                        str,
                        int,
                        float,
                        bool,
                    ):
                        v = expected_type(v)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Argument '{k}' could not be converted to {expected_type}: {e}"
                    ) from e

            parsed_kwargs[k] = v

        # Check for missing required arguments
        for arg_name, arg_schema in self.args.items():
            if arg_name not in parsed_kwargs and "default" not in arg_schema:
                # This is a required argument that's missing
                raise ValueError(f"Required argument '{arg_name}' missing for tool '{self.name}'")
            elif arg_name not in parsed_kwargs and "default" in arg_schema:
                # Use default value
                parsed_kwargs[arg_name] = arg_schema["default"]

        return parsed_kwargs

    def format(self) -> str:
        """Format tool for display."""
        return str(self)

    def format_for_llm(self) -> dict[str, Any]:
        """Format tool for LLM consumption (OpenAI function call format)."""
        required_args = [name for name, schema in self.args.items() if "default" not in schema]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": {
                    "type": "object",
                    "properties": self.args,
                    "required": required_args,
                },
            },
        }

    def __call__(self, **kwargs) -> Any:
        """Execute the tool synchronously."""
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)

        if asyncio.iscoroutine(result):
            # If function returns a coroutine, run it
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(result)
            except RuntimeError:
                # No event loop running
                return asyncio.run(result)

        return result

    async def acall(self, **kwargs) -> Any:
        """Execute the tool asynchronously."""
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)

        if asyncio.iscoroutine(result):
            return await result
        else:
            # Sync function - just return result
            return result

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        try:
            result = await self.acall(**kwargs)
            return ToolResult(
                output=result, success=True, metadata={"tool": self.name, "args": kwargs}
            )
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            return ToolResult(
                output=None,
                success=False,
                error=str(e),
                metadata={"tool": self.name, "args": kwargs},
            )

    def execute_sync(self, **kwargs) -> ToolResult:
        """Execute the tool synchronously and return a ToolResult."""
        try:
            result = self(**kwargs)
            return ToolResult(
                output=result, success=True, metadata={"tool": self.name, "args": kwargs}
            )
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            return ToolResult(
                output=None,
                success=False,
                error=str(e),
                metadata={"tool": self.name, "args": kwargs},
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.desc,
            "args": self.args,
            "arg_types": {k: str(v) for k, v in self.arg_types.items()},
            "has_kwargs": self.has_kwargs,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name}, desc={self.desc}, args={list(self.args.keys())})"

    def __str__(self) -> str:
        desc_str = f", whose description is <desc>{self.desc}</desc>." if self.desc else "."
        desc_str = desc_str.replace("\n", "  ")
        args_str = f"It takes arguments {self.args}." if self.args else "It takes no arguments."
        return f"{self.name}{desc_str} {args_str}"


def tool(name: str | None = None, desc: str | None = None, **kwargs) -> Callable[[Callable], Tool]:
    """Decorator to create a tool from a function.

    Args:
        name: Tool name (optional, inferred from function)
        desc: Tool description (optional, inferred from docstring)
        **kwargs: Additional arguments for Tool constructor

    Example:
        @tool(name="calculator", desc="Performs basic math")
        def add(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b
    """

    def decorator(func: Callable) -> Tool:
        return Tool(func, name=name, desc=desc, **kwargs)

    return decorator


# Common tool implementations


def search_tool(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search result descriptions
    """
    # Mock implementation - in real use would call actual search API
    return [f"Result {i + 1} for '{query}'" for i in range(max_results)]


def calculator_tool(expression: str) -> float:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Numerical result of the expression
    """
    import ast
    import operator

    # Safe evaluation using AST
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](eval_expr(node.operand))
        elif isinstance(node, ast.Call):
            # Handle function calls - only allow len() for safety
            if isinstance(node.func, ast.Name) and node.func.id == "len":
                if len(node.args) == 1:
                    arg_value = eval_expr(node.args[0])
                    return len(arg_value)
                else:
                    raise ValueError("len() takes exactly one argument")
            else:
                raise ValueError(f"Function '{node.func.id}' not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        else:
            raise ValueError(f"Unsupported operation: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return eval_expr(tree.body)


# Create tool instances for common tools
SEARCH_TOOL = Tool(search_tool)
CALCULATOR_TOOL = Tool(calculator_tool)


__all__ = [
    "Tool",
    "ToolResult",
    "tool",
    "search_tool",
    "calculator_tool",
    "SEARCH_TOOL",
    "CALCULATOR_TOOL",
]
