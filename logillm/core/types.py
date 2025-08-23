"""Core types, enums, and data structures for LogiLLM."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypeVar, Union, runtime_checkable

# Version check for typing features
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Generic type variables
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# Core type aliases
FieldValue: TypeAlias = Union[str, int, float, bool, list[Any], dict[str, Any], None]
Configuration: TypeAlias = dict[str, Any]
Metadata: TypeAlias = dict[str, Any]


class FieldType(Enum):
    """Types of fields in signatures."""

    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"
    CONTEXT = "context"


class ModuleState(Enum):
    """Lifecycle states of a module."""

    INITIALIZED = "initialized"
    CONFIGURED = "configured"
    COMPILED = "compiled"
    OPTIMIZED = "optimized"
    CACHED = "cached"


class OptimizationStrategy(Enum):
    """Available optimization strategies."""

    BOOTSTRAP = "bootstrap"  # Bootstrap few-shot examples
    INSTRUCTION = "instruction"  # Optimize instructions/prompts
    ENSEMBLE = "ensemble"  # Multi-model ensembles
    EVOLUTION = "evolution"  # Genetic/evolutionary algorithms
    REFLECTION = "reflection"  # Self-reflection based improvement
    HYBRID = "hybrid"  # Combination of strategies
    MULTI_OBJECTIVE = "multi_objective"  # Multi-objective optimization


class AdapterFormat(Enum):
    """Supported adapter formats."""

    CHAT = "chat"  # Conversational messages
    JSON = "json"  # Structured JSON output
    XML = "xml"  # XML markup format
    MARKDOWN = "markdown"  # Markdown formatting
    FUNCTION = "function"  # Function/tool calling format
    COMPLETION = "completion"  # Raw text completion
    STRUCTURED = "structured"  # Provider-specific structured output


class ExecutionMode(Enum):
    """Execution modes for modules."""

    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"
    PARALLEL = "parallel"


class CacheLevel(Enum):
    """Cache levels for performance optimization."""

    NONE = "none"
    MEMORY = "memory"  # In-memory LRU cache
    DISK = "disk"  # Persistent disk cache
    DISTRIBUTED = "distributed"  # Distributed cache (Redis, etc.)
    HYBRID = "hybrid"  # Memory + disk combination


class ValidationLevel(Enum):
    """Validation strictness levels."""

    NONE = "none"
    BASIC = "basic"  # Type checking only
    STRICT = "strict"  # Type + constraint checking
    PARANOID = "paranoid"  # Full validation with detailed errors


class SerializationFormat(Enum):
    """Serialization formats for persistence."""

    JSON = "json"
    PICKLE = "pickle"
    YAML = "yaml"
    TOML = "toml"
    MSGPACK = "msgpack"


class Priority(Enum):
    """Priority levels for execution and optimization."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class TokenUsage:
    """Token usage tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0  # For reasoning models like o1

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two token usage objects."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class Usage:
    """Complete usage tracking including cost."""

    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: float | None = None
    latency: float | None = None  # Response time in seconds
    provider: str | None = None
    model: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __add__(self, other: Usage) -> Usage:
        """Combine usage statistics."""
        return Usage(
            tokens=self.tokens + other.tokens,
            cost=(self.cost or 0) + (other.cost or 0)
            if self.cost is not None or other.cost is not None
            else None,
            latency=max(self.latency or 0, other.latency or 0)
            if self.latency is not None or other.latency is not None
            else None,
            provider=self.provider or other.provider,
            model=self.model or other.model,
            timestamp=max(self.timestamp, other.timestamp),
        )


@dataclass
class Completion:
    """Standardized completion response."""

    text: str
    usage: Usage = field(default_factory=Usage)
    metadata: Metadata = field(default_factory=dict)
    finish_reason: str | None = None
    model: str | None = None
    provider: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if completion finished normally."""
        return self.finish_reason in ("stop", "end_turn", None)


@dataclass
class Prediction:
    """Result of a module execution.

    Contains the outputs from the module, usage statistics, and optionally
    the prompt that was sent to the LLM (when debug mode is enabled).
    """

    outputs: dict[str, FieldValue] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    metadata: Metadata = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    prompt: dict[str, Any] | None = None  # Contains messages, adapter, etc. when debugging

    def __getattr__(self, name: str) -> FieldValue:
        """Allow dot notation access to outputs."""
        # Avoid recursion by using __dict__ directly
        if "outputs" in self.__dict__ and name in self.__dict__["outputs"]:
            return self.__dict__["outputs"][name]
        raise AttributeError(f"Prediction has no output field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow dot notation setting of outputs."""
        # Check if we're initializing the object
        if name in {"outputs", "usage", "metadata", "success", "error", "prompt"}:
            object.__setattr__(self, name, value)
        elif "outputs" in self.__dict__:
            self.__dict__["outputs"][name] = value
        else:
            object.__setattr__(self, name, value)


@dataclass
class TraceStep:
    """Single step in execution trace."""

    module_name: str
    inputs: dict[str, FieldValue]
    outputs: dict[str, FieldValue]
    usage: Usage
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float | None = None  # Execution time in seconds
    metadata: Metadata = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Complete execution trace."""

    steps: list[TraceStep] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)
    success: bool = True
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def duration(self) -> float | None:
        """Total execution duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def add_step(self, step: TraceStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)
        self.total_usage += step.usage
        if not step.success:
            self.success = False


@dataclass(frozen=True)
class CacheKey:
    """Cache key for memoization."""

    signature_hash: str
    inputs_hash: str
    config_hash: str

    def __str__(self) -> str:
        """String representation for storage."""
        return f"{self.signature_hash}:{self.inputs_hash}:{self.config_hash}"


@dataclass
class Example:
    """A training/test example with inputs and expected outputs."""

    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    metadata: Metadata = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of optimization process."""

    optimized_module: Any  # Will be Module type when defined
    improvement: float  # Metric improvement (e.g., 0.1 for 10% improvement)
    iterations: int
    best_score: float
    optimization_time: float
    metadata: Metadata = field(default_factory=dict)


# Runtime protocols for duck typing
@runtime_checkable
class Hashable(Protocol):
    """Protocol for hashable objects."""

    def __hash__(self) -> int: ...


@runtime_checkable
class Comparable(Protocol):
    """Protocol for comparable objects."""

    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...


# Export all public types
__all__ = [
    # Type variables and aliases
    "T",
    "K",
    "V",
    "FieldValue",
    "Configuration",
    "Metadata",
    # Enums
    "FieldType",
    "ModuleState",
    "OptimizationStrategy",
    "AdapterFormat",
    "ExecutionMode",
    "CacheLevel",
    "ValidationLevel",
    "SerializationFormat",
    "Priority",
    # Data classes
    "TokenUsage",
    "Usage",
    "Completion",
    "Prediction",
    "TraceStep",
    "ExecutionTrace",
    "CacheKey",
    "Example",
    "OptimizationResult",
    # Protocols
    "Hashable",
    "Comparable",
]
