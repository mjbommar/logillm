"""Advanced type support for LogiLLM signatures.

Provides multimodal types and enhanced type parsing capabilities
to achieve parity with DSPy and beyond.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass
class Image:
    """Multimodal image type for signature fields.

    Supports loading from:
    - File paths
    - URLs
    - Base64 encoded strings
    - PIL Image objects (if available)
    - Raw bytes
    """

    data: bytes
    format: str = "jpeg"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        """Load image from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        # Infer format from extension
        format_map = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".gif": "gif",
            ".bmp": "bmp",
            ".webp": "webp",
        }
        ext = path.suffix.lower()
        format_ = format_map.get(ext, "jpeg")

        return cls(data=data, format=format_)

    @classmethod
    def from_url(cls, url: str) -> Image:
        """Load image from URL."""
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError(f"Invalid URL: {url}")

        with urlopen(url) as response:
            data = response.read()

        # Try to infer format from URL
        path = Path(parsed.path)
        ext = path.suffix.lower()
        format_map = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".gif": "gif",
            ".bmp": "bmp",
            ".webp": "webp",
        }
        format_ = format_map.get(ext, "jpeg")

        return cls(data=data, format=format_)

    @classmethod
    def from_base64(cls, b64_string: str, format: str = "jpeg") -> Image:
        """Load image from base64 encoded string."""
        # Handle data URLs
        if b64_string.startswith("data:"):
            # Extract base64 part from data URL
            parts = b64_string.split(",", 1)
            if len(parts) == 2:
                b64_string = parts[1]

        data = base64.b64decode(b64_string)
        return cls(data=data, format=format)

    def to_base64(self) -> str:
        """Convert image to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")

    def to_data_url(self) -> str:
        """Convert to data URL for embedding."""
        mime_map = {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp",
        }
        mime = mime_map.get(self.format, "image/jpeg")
        b64 = self.to_base64()
        return f"data:{mime};base64,{b64}"

    def __str__(self) -> str:
        """String representation."""
        size_kb = len(self.data) / 1024
        return f"Image({self.format}, {size_kb:.1f}KB)"


@dataclass
class Audio:
    """Multimodal audio type for signature fields.

    Supports loading from:
    - File paths
    - URLs
    - Base64 encoded strings
    - Raw bytes
    """

    data: bytes
    format: str = "mp3"
    sample_rate: int | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Audio:
        """Load audio from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        # Infer format from extension
        format_map = {
            ".mp3": "mp3",
            ".wav": "wav",
            ".ogg": "ogg",
            ".m4a": "m4a",
            ".flac": "flac",
        }
        ext = path.suffix.lower()
        format_ = format_map.get(ext, "mp3")

        return cls(data=data, format=format_)

    @classmethod
    def from_url(cls, url: str) -> Audio:
        """Load audio from URL."""
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError(f"Invalid URL: {url}")

        with urlopen(url) as response:
            data = response.read()

        # Try to infer format from URL
        path = Path(parsed.path)
        ext = path.suffix.lower()
        format_map = {
            ".mp3": "mp3",
            ".wav": "wav",
            ".ogg": "ogg",
            ".m4a": "m4a",
            ".flac": "flac",
        }
        format_ = format_map.get(ext, "mp3")

        return cls(data=data, format=format_)

    def to_base64(self) -> str:
        """Convert audio to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")

    def __str__(self) -> str:
        """String representation."""
        size_kb = len(self.data) / 1024
        duration = f", {self.duration_seconds:.1f}s" if self.duration_seconds else ""
        return f"Audio({self.format}, {size_kb:.1f}KB{duration})"


@dataclass
class Tool:
    """Function tool type for signature fields.

    Represents a callable function that can be used by the LLM.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable | None = None

    def __call__(self, **kwargs) -> Any:
        """Execute the tool function."""
        if self.function is None:
            raise RuntimeError(f"Tool {self.name} has no implementation")
        return self.function(**kwargs)

    def to_json_schema(self) -> dict:
        """Convert to JSON schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __str__(self) -> str:
        """String representation."""
        param_count = len(self.parameters.get("properties", {}))
        return f"Tool({self.name}, {param_count} params)"


@dataclass
class History:
    """Conversation history type for signature fields.

    Represents a sequence of messages in a conversation.
    """

    messages: list[dict[str, str]]
    metadata: dict[str, Any] | None = None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the history."""
        self.messages.append({"role": role, "content": content})

    def get_last_n(self, n: int) -> list[dict[str, str]]:
        """Get the last n messages."""
        return self.messages[-n:] if n > 0 else []

    def clear(self) -> None:
        """Clear the history."""
        self.messages.clear()

    def __len__(self) -> int:
        """Number of messages."""
        return len(self.messages)

    def __str__(self) -> str:
        """String representation."""
        return f"History({len(self.messages)} messages)"

    def to_messages(self) -> list[dict[str, str]]:
        """Convert history to provider-compatible message format.

        Returns:
            List of messages in standard format with 'role' and 'content' keys.
        """
        # Ensure all messages have the required format
        formatted = []
        for msg in self.messages:
            if isinstance(msg, dict):
                # Ensure it has role and content
                if "role" in msg and "content" in msg:
                    formatted.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # Try to infer format
                    role = msg.get("role", "user")
                    content = msg.get("content", msg.get("text", str(msg)))
                    formatted.append({"role": role, "content": content})
            else:
                # Fallback for non-dict messages
                formatted.append({"role": "user", "content": str(msg)})
        return formatted


# Type alias for complex type parsing
TypeExpression = Union[type, str]


def parse_type_expression(expr: str, custom_types: dict[str, type] | None = None) -> type:
    """Parse a complex type expression string.

    Supports:
    - Basic types: str, int, float, bool
    - Generic types: list[str], dict[str, int]
    - Optional types: Optional[str], str | None
    - Union types: Union[str, int], str | int
    - Custom types from provided dictionary

    Args:
        expr: Type expression string
        custom_types: Optional dictionary of custom type mappings

    Returns:
        Parsed type object
    """
    import typing

    # Prepare namespace with all typing constructs and custom types
    namespace = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "List": list,
        "Dict": dict,
        "Tuple": tuple,
        "Set": set,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "Any": typing.Any,
        "None": type(None),
        # Multimodal types
        "Image": Image,
        "Audio": Audio,
        "Tool": Tool,
        "History": History,
    }

    if custom_types:
        namespace.update(custom_types)

    # Clean the expression
    expr = expr.strip()

    # Handle simple types directly
    if expr in namespace:
        return namespace[expr]

    # Handle Union shorthand (|)
    if "|" in expr and not expr.startswith("Union["):
        # Convert "str | int" to "Union[str, int]"
        parts = [part.strip() for part in expr.split("|")]
        expr = f"Union[{', '.join(parts)}]"

    # Try to evaluate the expression safely
    try:
        # Use eval with restricted namespace for safety
        result = eval(expr, {"__builtins__": {}}, namespace)
        return result
    except (SyntaxError, NameError, TypeError) as e:
        # Fall back to string type if parsing fails
        import warnings

        warnings.warn(
            f"Could not parse type expression '{expr}': {e}. Using str as fallback.", stacklevel=2
        )
        return str


def infer_type_from_value(value: Any) -> type:
    """Infer the type from a runtime value.

    Useful for dynamic signature generation from examples.
    """
    if value is None:
        return type(None)

    value_type = type(value)

    # Handle collections with element types
    if isinstance(value, list) and value:
        element_types = {type(v) for v in value}
        if len(element_types) == 1:
            element_type = element_types.pop()
            return list[element_type]

    if isinstance(value, dict) and value:
        key_types = {type(k) for k in value.keys()}
        val_types = {type(v) for v in value.values()}
        if len(key_types) == 1 and len(val_types) == 1:
            key_type = key_types.pop()
            val_type = val_types.pop()
            return dict[key_type, val_type]

    # Check for multimodal types
    if isinstance(value, Image):
        return Image
    if isinstance(value, Audio):
        return Audio
    if isinstance(value, Tool):
        return Tool
    if isinstance(value, History):
        return History

    return value_type


__all__ = [
    "Image",
    "Audio",
    "Tool",
    "History",
    "parse_type_expression",
    "infer_type_from_value",
]
