"""Tool system for agent modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from logillm.core.types import Metadata


class ToolType(Enum):
    """Types of tools available."""

    FUNCTION = "function"
    API = "api"
    RETRIEVER = "retriever"
    CALCULATOR = "calculator"
    SEARCH = "search"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    description: str
    python_type: type
    required: bool = True
    default: Any | None = None
    choices: list[Any] | None = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: Any
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)


class Tool(ABC):
    """Base class for tools that agents can use."""

    def __init__(
        self,
        name: str,
        description: str,
        tool_type: ToolType = ToolType.FUNCTION,
        parameters: list[ToolParameter] | None = None,
        metadata: Metadata | None = None,
    ):
        """Initialize tool.

        Args:
            name: Tool name (used for invocation)
            description: What the tool does
            tool_type: Type of tool
            parameters: Tool parameters
            metadata: Additional metadata
        """
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self.parameters = parameters or []
        self.metadata = metadata or {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        ...

    def validate_args(self, **kwargs: Any) -> dict[str, Any]:
        """Validate arguments against parameter definitions."""
        validated = {}

        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(
                    f"Required parameter '{param.name}' missing for tool '{self.name}'"
                )

            if param.name in kwargs:
                value = kwargs[param.name]

                # Type checking
                if not isinstance(value, param.python_type):
                    try:
                        # Try to coerce
                        value = param.python_type(value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Parameter '{param.name}' expected {param.python_type}, got {type(value)}"
                        ) from e

                # Choice validation
                if param.choices and value not in param.choices:
                    raise ValueError(
                        f"Parameter '{param.name}' must be one of {param.choices}, got {value}"
                    )

                validated[param.name] = value
            elif param.default is not None:
                validated[param.name] = param.default

        return validated

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.python_type.__name__,
                    "required": p.required,
                    "default": p.default,
                    "choices": p.choices,
                }
                for p in self.parameters
            ],
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation."""
        params = ", ".join(p.name for p in self.parameters)
        return f"{self.name}({params}): {self.description}"


class FunctionTool(Tool):
    """Tool that wraps a Python function."""

    def __init__(
        self, func: Callable, name: str | None = None, description: str | None = None, **kwargs
    ):
        """Initialize function tool.

        Args:
            func: Function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            **kwargs: Additional arguments for Tool
        """
        self.func = func

        # Extract metadata from function
        name = name or getattr(func, "__name__", "function")
        description = description or (getattr(func, "__doc__", None) or "").strip().split("\n")[0]

        # Try to extract parameters from function signature
        import inspect

        sig = inspect.signature(func)
        parameters = kwargs.pop("parameters", [])

        if not parameters:
            # Auto-generate from signature
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = (
                    str if param.annotation == inspect.Parameter.empty else param.annotation
                )
                required = param.default == inspect.Parameter.empty
                default = None if required else param.default

                parameters.append(
                    ToolParameter(
                        name=param_name,
                        description=f"Parameter {param_name}",
                        python_type=param_type,
                        required=required,
                        default=default,
                    )
                )

        super().__init__(
            name=name,
            description=description,
            tool_type=ToolType.FUNCTION,
            parameters=parameters,
            **kwargs,
        )

        # Check if function is async
        self.is_async = inspect.iscoroutinefunction(func)

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the function."""
        try:
            validated = self.validate_args(**kwargs)

            if self.is_async:
                result = await self.func(**validated)
            else:
                result = self.func(**validated)

            return ToolResult(
                success=True, output=result, metadata={"tool": self.name, "args": validated}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                metadata={"tool": self.name, "args": kwargs},
            )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_descriptions(self) -> dict[str, str]:
        """Get tool descriptions for prompt."""
        return {name: tool.description for name, tool in self.tools.items()}

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found",
            )

        return await tool.execute(**kwargs)


# Decorator for easy tool creation
def tool(name: str | None = None, description: str | None = None, **kwargs) -> Callable:
    """Decorator to create a tool from a function.

    Example:
        @tool(description="Calculate the sum of two numbers")
        def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func=func, name=name, description=description, **kwargs)

    return decorator


# Built-in tools


class CalculatorTool(Tool):
    """Built-in calculator tool."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions",
            tool_type=ToolType.CALCULATOR,
            parameters=[
                ToolParameter(
                    name="expression",
                    description="Mathematical expression to evaluate",
                    python_type=str,
                    required=True,
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Evaluate expression safely."""
        try:
            validated = self.validate_args(**kwargs)
            expression = validated["expression"]

            # Safe evaluation using ast
            import ast
            import operator

            # Allowed operations
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
                else:
                    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

            tree = ast.parse(expression, mode="eval")
            result = eval_expr(tree.body)

            return ToolResult(success=True, output=result, metadata={"expression": expression})
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to evaluate: {e}",
            )


__all__ = [
    "ToolType",
    "ToolParameter",
    "ToolResult",
    "Tool",
    "FunctionTool",
    "ToolRegistry",
    "tool",
    "CalculatorTool",
]
