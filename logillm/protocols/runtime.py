"""Runtime protocols for LogiLLM components."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from ..core.types import (
    CacheKey,
    Configuration,
    ExecutionTrace,
    SerializationFormat,
)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Serializable:
        """Reconstruct from dictionary."""
        ...

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Serializable:
        """Reconstruct from JSON string."""
        import json

        return cls.from_dict(json.loads(json_str))


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for objects that can be cached."""

    @abstractmethod
    def cache_key(self) -> CacheKey:
        """Generate cache key for this object."""
        ...

    @abstractmethod
    def is_cacheable(self) -> bool:
        """Whether this object should be cached."""
        ...

    def cache_ttl(self) -> int | None:
        """Cache time-to-live in seconds. None means no expiration."""
        return None


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can be validated."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate the object state."""
        ...

    @abstractmethod
    def validation_errors(self) -> list[str]:
        """Get list of validation errors."""
        ...

    def is_valid(self) -> bool:
        """Check if object is valid."""
        return self.validate()


@runtime_checkable
class Optimizable(Protocol):
    """Protocol for objects that can be optimized."""

    @abstractmethod
    def optimize(self, traces: list[ExecutionTrace], **kwargs: Any) -> Optimizable:
        """Create optimized version based on execution traces."""
        ...

    @abstractmethod
    def optimization_parameters(self) -> dict[str, Any]:
        """Get parameters that can be optimized."""
        ...

    def is_optimized(self) -> bool:
        """Check if this object has been optimized."""
        return False


@runtime_checkable
class Traceable(Protocol):
    """Protocol for objects that support execution tracing."""

    @abstractmethod
    def enable_tracing(self) -> None:
        """Enable execution tracing."""
        ...

    @abstractmethod
    def disable_tracing(self) -> None:
        """Disable execution tracing."""
        ...

    @abstractmethod
    def get_trace(self) -> ExecutionTrace | None:
        """Get current execution trace."""
        ...

    def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.get_trace() is not None


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable objects."""

    @abstractmethod
    def configure(self, config: Configuration) -> None:
        """Apply configuration."""
        ...

    @abstractmethod
    def get_config(self) -> Configuration:
        """Get current configuration."""
        ...

    @abstractmethod
    def validate_config(self, config: Configuration) -> bool:
        """Validate configuration."""
        ...

    def config_schema(self) -> dict[str, Any]:
        """Get configuration schema (JSON Schema format)."""
        return {}


@runtime_checkable
class Streamable(Protocol):
    """Protocol for objects that support streaming."""

    @abstractmethod
    async def stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream results asynchronously."""
        ...

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        ...

    def stream_sync(self, **kwargs: Any) -> Iterator[Any]:
        """Synchronous streaming interface."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = self.stream(**kwargs)
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


@runtime_checkable
class Batchable(Protocol):
    """Protocol for objects that support batch processing."""

    @abstractmethod
    async def batch_process(self, items: list[Any], **kwargs: Any) -> list[Any]:
        """Process multiple items in batch."""
        ...

    @abstractmethod
    def optimal_batch_size(self) -> int:
        """Get optimal batch size for this object."""
        ...

    def supports_batching(self) -> bool:
        """Check if batching is supported."""
        return True


@runtime_checkable
class Retryable(Protocol):
    """Protocol for operations that can be retried."""

    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        ...

    @abstractmethod
    def retry_delay(self, attempt: int) -> float:
        """Get delay before retry in seconds."""
        ...

    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return 3


@runtime_checkable
class Metrifiable(Protocol):
    """Protocol for objects that can be measured/evaluated."""

    @abstractmethod
    def compute_metrics(self, reference: Any) -> dict[str, float]:
        """Compute metrics against reference."""
        ...

    @abstractmethod
    def metric_names(self) -> list[str]:
        """Get list of available metric names."""
        ...


@runtime_checkable
class Composable(Protocol):
    """Protocol for objects that can be composed together."""

    @abstractmethod
    def compose(self, other: Composable) -> Composable:
        """Compose with another object."""
        ...

    @abstractmethod
    def is_compatible(self, other: Composable) -> bool:
        """Check compatibility for composition."""
        ...


@runtime_checkable
class Monitorable(Protocol):
    """Protocol for objects that can be monitored."""

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Get current performance metrics."""
        ...

    @abstractmethod
    def get_health_status(self) -> bool:
        """Get health status (True = healthy)."""
        ...

    def get_diagnostic_info(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {"status": "healthy" if self.get_health_status() else "unhealthy"}


@runtime_checkable
class Pluggable(Protocol):
    """Protocol for plugin-based architecture."""

    @abstractmethod
    def register_plugin(self, name: str, plugin: Any) -> None:
        """Register a plugin."""
        ...

    @abstractmethod
    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin."""
        ...

    @abstractmethod
    def get_plugin(self, name: str) -> Any:
        """Get a registered plugin."""
        ...

    def list_plugins(self) -> list[str]:
        """List all registered plugins."""
        return []


@runtime_checkable
class Debuggable(Protocol):
    """Protocol for objects that support debugging."""

    @abstractmethod
    def debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        ...

    @abstractmethod
    def enable_debug_mode(self) -> None:
        """Enable debug mode."""
        ...

    @abstractmethod
    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        ...

    def is_debug_mode_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return False


# Composite protocols for common combinations
@runtime_checkable
class Persistable(Serializable, Cacheable, Protocol):
    """Protocol for objects that can be persisted."""

    @abstractmethod
    def save(self, path: str, format: SerializationFormat = SerializationFormat.JSON) -> None:
        """Save to file."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, format: SerializationFormat | None = None) -> Persistable:
        """Load from file."""
        ...


@runtime_checkable
class ExecutableComponent(Configurable, Traceable, Validatable, Monitorable, Protocol):
    """Protocol for executable components in the framework."""

    pass


# Export all protocols
__all__ = [
    # Basic protocols
    "Serializable",
    "Cacheable",
    "Validatable",
    "Optimizable",
    "Traceable",
    "Configurable",
    "Streamable",
    "Batchable",
    "Retryable",
    "Metrifiable",
    "Composable",
    "Monitorable",
    "Pluggable",
    "Debuggable",
    # Composite protocols
    "Persistable",
    "ExecutableComponent",
]
