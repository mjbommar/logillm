"""Comprehensive callback system for LogiLLM.

This module provides a flexible callback system that supports:
- Multiple callback types for different lifecycle events
- Priority-based execution ordering
- Context passing between callbacks
- Both async and sync callback support
- Thread-safe callback management
- Built-in useful callbacks (logging, metrics, progress)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TypeVar, Union

from .types import Metadata, Prediction, Usage

# Type variable for callbacks
CallbackT = TypeVar("CallbackT", bound="BaseCallback")

# Context variable for tracking active call IDs
ACTIVE_CALL_ID: ContextVar[Optional[str]] = ContextVar("active_call_id", default=None)  # noqa: UP045 # type: ignore[assignment]


class CallbackType(Enum):
    """Types of callbacks available."""

    MODULE_START = "module_start"
    MODULE_END = "module_end"
    OPTIMIZATION_START = "optimization_start"
    OPTIMIZATION_END = "optimization_end"
    ERROR = "error"
    PROVIDER_REQUEST = "provider_request"
    PROVIDER_RESPONSE = "provider_response"
    EVALUATION_START = "evaluation_start"
    EVALUATION_END = "evaluation_end"


class Priority(Enum):
    """Priority levels for callback execution."""

    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5


@dataclass
class CallbackContext:
    """Context object passed between callbacks."""

    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_call_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def add_data(self, key: str, value: Any) -> None:
        """Add data to the context."""
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from the context."""
        return self.data.get(key, default)


class CallbackEvent:
    """Base event object for all callbacks."""

    def __init__(self, context: CallbackContext, timestamp: datetime | None = None):
        """Initialize callback event."""
        self.context = context
        self.timestamp = timestamp or datetime.now()


class ModuleStartEvent(CallbackEvent):
    """Event fired when a module starts execution."""

    def __init__(
        self,
        context: CallbackContext,
        module: Any,
        inputs: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize module start event."""
        super().__init__(context, timestamp)
        self.module = module
        self.inputs = inputs or {}


class ModuleEndEvent(CallbackEvent):
    """Event fired when a module ends execution."""

    def __init__(
        self,
        context: CallbackContext,
        module: Any,
        outputs: dict[str, Any] | None = None,
        prediction: Prediction | None = None,
        success: bool = True,
        duration: float | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize module end event."""
        super().__init__(context, timestamp)
        self.module = module
        self.outputs = outputs or {}
        self.prediction = prediction
        self.success = success
        self.duration = duration


class OptimizationStartEvent(CallbackEvent):
    """Event fired when optimization starts."""

    def __init__(
        self,
        context: CallbackContext,
        optimizer: Any,
        module: Any,
        dataset: list[Any] | None = None,
        config: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize optimization start event."""
        super().__init__(context, timestamp)
        self.optimizer = optimizer
        self.module = module
        self.dataset = dataset or []
        self.config = config or {}


class OptimizationEndEvent(CallbackEvent):
    """Event fired when optimization ends."""

    def __init__(
        self,
        context: CallbackContext,
        optimizer: Any,
        result: Any,
        success: bool = True,
        duration: float | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize optimization end event."""
        super().__init__(context, timestamp)
        self.optimizer = optimizer
        self.result = result
        self.success = success
        self.duration = duration


class ErrorEvent(CallbackEvent):
    """Event fired when an error occurs."""

    def __init__(
        self,
        context: CallbackContext,
        error: Exception,
        module: Any | None = None,
        stage: str = "unknown",
        recoverable: bool = False,
        timestamp: datetime | None = None,
    ):
        """Initialize error event."""
        super().__init__(context, timestamp)
        self.error = error
        self.module = module
        self.stage = stage
        self.recoverable = recoverable


class ProviderRequestEvent(CallbackEvent):
    """Event fired before provider request."""

    def __init__(
        self,
        context: CallbackContext,
        provider: Any,
        messages: list[dict[str, str]],
        parameters: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize provider request event."""
        super().__init__(context, timestamp)
        self.provider = provider
        self.messages = messages
        self.parameters = parameters or {}


class ProviderResponseEvent(CallbackEvent):
    """Event fired after provider response."""

    def __init__(
        self,
        context: CallbackContext,
        provider: Any,
        request_messages: list[dict[str, str]] | None = None,
        response: Any = None,
        usage: Usage | None = None,
        duration: float | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize provider response event."""
        super().__init__(context, timestamp)
        self.provider = provider
        self.request_messages = request_messages or []
        self.response = response
        self.usage = usage
        self.duration = duration


# Union type for all events
CallbackEventType = Union[
    ModuleStartEvent,
    ModuleEndEvent,
    OptimizationStartEvent,
    OptimizationEndEvent,
    ErrorEvent,
    ProviderRequestEvent,
    ProviderResponseEvent,
]


class BaseCallback(Protocol):
    """Protocol defining the callback interface.

    Callbacks can implement any subset of these methods.
    All methods are optional and have default no-op implementations.
    """

    def on_module_start(self, event: ModuleStartEvent) -> None:
        """Called when a module starts execution."""
        ...

    def on_module_end(self, event: ModuleEndEvent) -> None:
        """Called when a module ends execution."""
        ...

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Called when optimization starts."""
        ...

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Called when optimization ends."""
        ...

    def on_error(self, event: ErrorEvent) -> None:
        """Called when an error occurs."""
        ...

    def on_provider_request(self, event: ProviderRequestEvent) -> None:
        """Called before making a provider request."""
        ...

    def on_provider_response(self, event: ProviderResponseEvent) -> None:
        """Called after receiving a provider response."""
        ...


class AbstractCallback:
    """Base class for callbacks with default implementations.

    This is not an abstract base class - it provides default no-op implementations
    for all callback methods. Subclasses can override only the methods they need.
    """

    @property
    def priority(self) -> Priority:
        """Callback execution priority."""
        return Priority.NORMAL

    @property
    def name(self) -> str:
        """Callback name for identification."""
        return self.__class__.__name__

    def on_module_start(self, event: ModuleStartEvent) -> None:
        """Called when a module starts execution."""
        pass

    def on_module_end(self, event: ModuleEndEvent) -> None:
        """Called when a module ends execution."""
        pass

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Called when optimization starts."""
        pass

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Called when optimization ends."""
        pass

    def on_error(self, event: ErrorEvent) -> None:
        """Called when an error occurs."""
        pass

    def on_provider_request(self, event: ProviderRequestEvent) -> None:
        """Called before making a provider request."""
        pass

    def on_provider_response(self, event: ProviderResponseEvent) -> None:
        """Called after receiving a provider response."""
        pass


class LoggingCallback(AbstractCallback):
    """Built-in callback for logging module execution and optimization events."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO):
        """Initialize logging callback.

        Args:
            logger: Logger instance to use. If None, creates one.
            level: Logging level to use.
        """
        self.logger = logger or logging.getLogger("logillm.callbacks")
        self.level = level

    @property
    def priority(self) -> Priority:
        """Logging has low priority to run early."""
        return Priority.LOW

    def on_module_start(self, event: ModuleStartEvent) -> None:
        """Log module start."""
        module_name = getattr(event.module.__class__, "__name__", "Unknown")
        self.logger.log(
            self.level,
            f"Module {module_name} starting execution (call_id: {event.context.call_id})",
            extra={
                "call_id": event.context.call_id,
                "module_name": module_name,
                "inputs": event.inputs,
            },
        )

    def on_module_end(self, event: ModuleEndEvent) -> None:
        """Log module end."""
        module_name = getattr(event.module.__class__, "__name__", "Unknown")
        status = "success" if event.success else "failure"
        duration_str = f" in {event.duration:.3f}s" if event.duration else ""

        self.logger.log(
            self.level,
            f"Module {module_name} finished {status}{duration_str} (call_id: {event.context.call_id})",
            extra={
                "call_id": event.context.call_id,
                "module_name": module_name,
                "success": event.success,
                "duration": event.duration,
                "outputs": event.outputs,
            },
        )

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Log optimization start."""
        optimizer_name = getattr(event.optimizer.__class__, "__name__", "Unknown")
        dataset_size = len(event.dataset)

        self.logger.log(
            self.level,
            f"Optimization starting with {optimizer_name} on {dataset_size} examples (call_id: {event.context.call_id})",
            extra={
                "call_id": event.context.call_id,
                "optimizer_name": optimizer_name,
                "dataset_size": dataset_size,
                "config": event.config,
            },
        )

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Log optimization end."""
        optimizer_name = getattr(event.optimizer.__class__, "__name__", "Unknown")
        status = "success" if event.success else "failure"
        duration_str = f" in {event.duration:.3f}s" if event.duration else ""

        self.logger.log(
            self.level,
            f"Optimization finished {status}{duration_str} with {optimizer_name} (call_id: {event.context.call_id})",
            extra={
                "call_id": event.context.call_id,
                "optimizer_name": optimizer_name,
                "success": event.success,
                "duration": event.duration,
            },
        )

    def on_error(self, event: ErrorEvent) -> None:
        """Log errors."""
        module_name = (
            getattr(event.module.__class__, "__name__", "Unknown") if event.module else None
        )
        error_msg = f"Error in {event.stage}"
        if module_name:
            error_msg += f" of {module_name}"
        error_msg += f": {event.error}"

        self.logger.error(
            error_msg,
            extra={
                "call_id": event.context.call_id,
                "module_name": module_name,
                "stage": event.stage,
                "error": str(event.error),
                "error_type": type(event.error).__name__,
                "recoverable": event.recoverable,
            },
            exc_info=event.error,
        )

    def on_provider_request(self, event: ProviderRequestEvent) -> None:
        """Log provider requests."""
        provider_name = getattr(event.provider, "name", "Unknown")
        model = getattr(event.provider, "model", "Unknown")

        self.logger.log(
            self.level,
            f"Provider request to {provider_name}:{model} (call_id: {event.context.call_id})",
            extra={
                "call_id": event.context.call_id,
                "provider_name": provider_name,
                "model": model,
                "message_count": len(event.messages),
                "parameters": event.parameters,
            },
        )

    def on_provider_response(self, event: ProviderResponseEvent) -> None:
        """Log provider responses."""
        provider_name = getattr(event.provider, "name", "Unknown")
        duration_str = f" in {event.duration:.3f}s" if event.duration else ""

        usage_info = ""
        if event.usage:
            usage_info = f" ({event.usage.tokens.total_tokens} tokens)"

        self.logger.log(
            self.level,
            f"Provider response from {provider_name}{duration_str}{usage_info} (call_id: {event.context.call_id})",
            extra={
                "call_id": event.context.call_id,
                "provider_name": provider_name,
                "duration": event.duration,
                "usage": event.usage.to_dict()  # type: ignore[attr-defined]
                if hasattr(event.usage, "to_dict")
                and callable(getattr(event.usage, "to_dict", None))
                else str(event.usage),
            },
        )


class MetricsCallback(AbstractCallback):
    """Built-in callback for tracking performance metrics."""

    def __init__(self):
        """Initialize metrics callback."""
        self.metrics: dict[str, Any] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    @property
    def priority(self) -> Priority:
        """Metrics collection has normal priority."""
        return Priority.NORMAL

    def on_module_start(self, event: ModuleStartEvent) -> None:
        """Track module starts."""
        with self._lock:
            self.counters["module_starts"] += 1
            event.context.add_data("start_time", time.time())

    def on_module_end(self, event: ModuleEndEvent) -> None:
        """Track module performance."""
        with self._lock:
            self.counters["module_ends"] += 1
            if event.success:
                self.counters["module_successes"] += 1
            else:
                self.counters["module_failures"] += 1

            if event.duration:
                self.metrics["module_durations"].append(event.duration)

            # Track by module type
            module_name = getattr(event.module.__class__, "__name__", "Unknown")
            self.counters[f"module_{module_name}"] += 1

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Track optimization starts."""
        with self._lock:
            self.counters["optimization_starts"] += 1
            event.context.add_data("opt_start_time", time.time())

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Track optimization performance."""
        with self._lock:
            self.counters["optimization_ends"] += 1
            if event.success:
                self.counters["optimization_successes"] += 1
            else:
                self.counters["optimization_failures"] += 1

            if event.duration:
                self.metrics["optimization_durations"].append(event.duration)

    def on_provider_request(self, event: ProviderRequestEvent) -> None:
        """Track provider requests."""
        with self._lock:
            self.counters["provider_requests"] += 1
            provider_name = getattr(event.provider, "name", "Unknown")
            self.counters[f"provider_{provider_name}_requests"] += 1
            event.context.add_data("provider_start_time", time.time())

    def on_provider_response(self, event: ProviderResponseEvent) -> None:
        """Track provider responses."""
        with self._lock:
            self.counters["provider_responses"] += 1

            if event.duration:
                self.metrics["provider_durations"].append(event.duration)

            if event.usage:
                self.metrics["token_usage"].append(event.usage.tokens.total_tokens)
                if event.usage.cost:
                    self.metrics["costs"].append(event.usage.cost)

    def on_error(self, event: ErrorEvent) -> None:
        """Track errors."""
        with self._lock:
            self.counters["errors"] += 1
            error_type = type(event.error).__name__
            self.counters[f"error_{error_type}"] += 1
            if event.stage:
                self.counters[f"error_in_{event.stage}"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            # Calculate summary statistics
            summary = {}

            # Add counters (ensure common counters exist with 0 default)
            common_counters = [
                "module_starts",
                "module_ends",
                "module_successes",
                "module_failures",
                "optimization_starts",
                "optimization_ends",
                "optimization_successes",
                "optimization_failures",
                "provider_requests",
                "provider_responses",
                "errors",
            ]

            for counter in common_counters:
                summary[counter] = self.counters[
                    counter
                ]  # defaultdict will return 0 for missing keys

            # Add all other counters
            for key, value in self.counters.items():
                if key not in summary:
                    summary[key] = value

            # Add averages and sums for numeric metrics
            for key, values in self.metrics.items():
                if values:
                    summary[f"{key}_avg"] = sum(values) / len(values)
                    summary[f"{key}_total"] = sum(values)
                    summary[f"{key}_count"] = len(values)
                    if len(values) > 1:
                        summary[f"{key}_min"] = min(values)
                        summary[f"{key}_max"] = max(values)

            # Calculate success rates
            if summary["module_ends"] > 0:
                summary["module_success_rate"] = (
                    summary["module_successes"] / summary["module_ends"]
                )

            if summary["optimization_ends"] > 0:
                summary["optimization_success_rate"] = (
                    summary["optimization_successes"] / summary["optimization_ends"]
                )

            return summary

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()

    def __getstate__(self) -> dict:
        """Get state for pickling (excludes the Lock)."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling (recreates the Lock)."""
        self.__dict__.update(state)
        self._lock = threading.Lock()


class ProgressCallback(AbstractCallback):
    """Built-in callback for showing progress of long-running operations."""

    def __init__(self, show_details: bool = True, update_interval: float = 1.0):
        """Initialize progress callback.

        Args:
            show_details: Whether to show detailed progress information.
            update_interval: Minimum time between progress updates.
        """
        self.show_details = show_details
        self.update_interval = update_interval
        self._last_update: dict[str, float] = {}
        self._active_operations: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def priority(self) -> Priority:
        """Progress tracking has low priority."""
        return Priority.LOW

    def on_module_start(self, event: ModuleStartEvent) -> None:
        """Track module start for progress."""
        module_name = getattr(event.module.__class__, "__name__", "Unknown")
        if self.show_details:
            print(f"🔄 Starting {module_name}...")

        with self._lock:
            self._active_operations[event.context.call_id] = {
                "type": "module",
                "name": module_name,
                "start_time": time.time(),
            }

    def on_module_end(self, event: ModuleEndEvent) -> None:
        """Track module completion."""
        with self._lock:
            if event.context.call_id in self._active_operations:
                op = self._active_operations.pop(event.context.call_id)
                if self.show_details:
                    status = "✅" if event.success else "❌"
                    duration_str = f" ({event.duration:.2f}s)" if event.duration else ""
                    print(f"{status} Finished {op['name']}{duration_str}")

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Track optimization start."""
        optimizer_name = getattr(event.optimizer.__class__, "__name__", "Unknown")
        dataset_size = len(event.dataset)
        print(f"🚀 Starting optimization with {optimizer_name} on {dataset_size} examples...")

        with self._lock:
            self._active_operations[event.context.call_id] = {
                "type": "optimization",
                "name": optimizer_name,
                "start_time": time.time(),
                "dataset_size": dataset_size,
            }

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Track optimization completion."""
        with self._lock:
            if event.context.call_id in self._active_operations:
                self._active_operations.pop(event.context.call_id)
                status = "🎉" if event.success else "💥"
                duration_str = f" ({event.duration:.1f}s)" if event.duration else ""
                print(f"{status} Optimization finished{duration_str}")

    def on_error(self, event: ErrorEvent) -> None:
        """Show error progress."""
        if self.show_details:
            module_info = (
                f" in {getattr(event.module.__class__, '__name__', 'Unknown')}"
                if event.module
                else ""
            )
            print(f"❌ Error{module_info}: {event.error}")

    def __getstate__(self) -> dict:
        """Get state for pickling (excludes the Lock)."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling (recreates the Lock)."""
        self.__dict__.update(state)
        self._lock = threading.Lock()


@dataclass
class RegisteredCallback:
    """A registered callback with metadata."""

    callback: BaseCallback
    priority: Priority
    name: str
    enabled: bool = True


class CallbackManager:
    """Thread-safe manager for callbacks with priority ordering.

    This is implemented as a singleton to provide global callback management
    while still allowing instance-specific callbacks.
    """

    _instance: CallbackManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> CallbackManager:
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize callback manager."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._callbacks: dict[CallbackType, list[RegisteredCallback]] = defaultdict(list)
        self._global_callbacks: list[RegisteredCallback] = []
        self._callback_lock = threading.RLock()  # Reentrant lock for nested calls
        self._enabled = True
        self._initialized = True

    def register(
        self,
        callback: BaseCallback,
        callback_types: list[CallbackType] | None = None,
        priority: Priority | None = None,
        name: str | None = None,
    ) -> str:
        """Register a callback.

        Args:
            callback: The callback to register.
            callback_types: List of callback types to register for. If None, registers for all types.
            priority: Priority for execution order. If None, uses callback's priority.
            name: Name for the callback. If None, uses callback class name.

        Returns:
            Registration ID for later removal.
        """
        if not self._enabled:
            return ""

        # Determine priority
        if priority is None:
            priority = getattr(callback, "priority", Priority.NORMAL)

        # Determine name
        if name is None:
            name = getattr(callback, "name", callback.__class__.__name__)

        reg_callback = RegisteredCallback(
            callback=callback,
            priority=priority,
            name=name,
        )

        with self._callback_lock:
            if callback_types is None:
                # Register for all callback types
                callback_types = list(CallbackType)

            for cb_type in callback_types:
                self._callbacks[cb_type].append(reg_callback)
                # Sort by priority (highest first)
                self._callbacks[cb_type].sort(key=lambda x: x.priority.value, reverse=True)

        return f"{name}_{id(reg_callback)}"

    def register_global(
        self,
        callback: BaseCallback,
        priority: Priority | None = None,
        name: str | None = None,
    ) -> str:
        """Register a global callback that receives all events.

        Args:
            callback: The callback to register.
            priority: Priority for execution order.
            name: Name for the callback.

        Returns:
            Registration ID for later removal.
        """
        if not self._enabled:
            return ""

        if priority is None:
            priority = getattr(callback, "priority", Priority.NORMAL)

        if name is None:
            name = getattr(callback, "name", callback.__class__.__name__)

        reg_callback = RegisteredCallback(
            callback=callback,
            priority=priority,
            name=name,
        )

        with self._callback_lock:
            self._global_callbacks.append(reg_callback)
            # Sort by priority (highest first)
            self._global_callbacks.sort(key=lambda x: x.priority.value, reverse=True)

        return f"global_{name}_{id(reg_callback)}"

    def __getstate__(self) -> dict:
        """Get state for pickling (excludes the RLock)."""
        state = self.__dict__.copy()
        # Remove the unpicklable RLock
        state.pop("_callback_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling (recreates the RLock)."""
        self.__dict__.update(state)
        # Recreate the RLock
        self._callback_lock = threading.RLock()

    def unregister(self, callback_id: str) -> bool:
        """Unregister a callback by ID.

        Args:
            callback_id: The ID returned by register().

        Returns:
            True if callback was found and removed.
        """
        with self._callback_lock:
            # Check global callbacks first
            if callback_id.startswith("global_"):
                for i, reg_cb in enumerate(self._global_callbacks):
                    if f"global_{reg_cb.name}_{id(reg_cb)}" == callback_id:
                        del self._global_callbacks[i]
                        return True

            # Check specific callback types
            for cb_type in CallbackType:
                for i, reg_cb in enumerate(self._callbacks[cb_type]):
                    if f"{reg_cb.name}_{id(reg_cb)}" == callback_id:
                        del self._callbacks[cb_type][i]
                        return True

        return False

    def unregister_by_name(self, name: str) -> int:
        """Unregister all callbacks with a given name.

        Args:
            name: Name of callbacks to remove.

        Returns:
            Number of callbacks removed.
        """
        removed = 0

        with self._callback_lock:
            # Remove from global callbacks
            self._global_callbacks = [cb for cb in self._global_callbacks if cb.name != name]

            # Remove from specific types
            for cb_type in CallbackType:
                original_len = len(self._callbacks[cb_type])
                self._callbacks[cb_type] = [
                    cb for cb in self._callbacks[cb_type] if cb.name != name
                ]
                removed += original_len - len(self._callbacks[cb_type])

        return removed

    def clear(self) -> None:
        """Clear all registered callbacks."""
        with self._callback_lock:
            # Clear all callbacks
            self._callbacks.clear()
            self._global_callbacks.clear()
            # Reinitialize the defaultdict to ensure proper structure
            self._callbacks = defaultdict(list)

    def enable(self) -> None:
        """Enable callback execution."""
        self._enabled = True

    def disable(self) -> None:
        """Disable callback execution."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if callbacks are enabled."""
        return self._enabled

    async def emit_async(self, event: CallbackEventType) -> None:
        """Emit an event to all registered callbacks (async version).

        Args:
            event: Event to emit.
        """
        if not self._enabled:
            return

        # Determine callback type from event
        callback_type = self._get_callback_type(event)

        with self._callback_lock:
            # Get callbacks for this type
            type_callbacks = self._callbacks.get(callback_type, [])
            # Add global callbacks
            all_callbacks = list(self._global_callbacks) + type_callbacks

            # Filter enabled callbacks and sort by priority
            enabled_callbacks = [cb for cb in all_callbacks if cb.enabled]
            enabled_callbacks.sort(key=lambda x: x.priority.value, reverse=True)

        # Execute callbacks
        for reg_cb in enabled_callbacks:
            try:
                # Get the appropriate method for this event type
                method = self._get_callback_method(reg_cb.callback, event)
                if method:
                    # Check if method is async
                    if asyncio.iscoroutinefunction(method):
                        await method(event)
                    else:
                        method(event)
            except Exception as e:
                # Log callback errors but don't let them break execution
                logging.getLogger("logillm.callbacks").warning(
                    f"Error in callback {reg_cb.name}: {e}",
                    exc_info=True,
                )

    def emit_sync(self, event: CallbackEventType) -> None:
        """Emit an event to all registered callbacks (sync version).

        Args:
            event: Event to emit.
        """
        if not self._enabled:
            return

        # Note: We always execute synchronously to ensure callbacks complete
        # before returning. This is important for testing and predictable behavior.

        # Determine callback type from event
        callback_type = self._get_callback_type(event)

        with self._callback_lock:
            # Get callbacks for this type
            type_callbacks = self._callbacks.get(callback_type, [])
            # Add global callbacks
            all_callbacks = list(self._global_callbacks) + type_callbacks

            # Filter enabled callbacks and sort by priority
            enabled_callbacks = [cb for cb in all_callbacks if cb.enabled]
            enabled_callbacks.sort(key=lambda x: x.priority.value, reverse=True)

        # Execute callbacks synchronously
        for reg_cb in enabled_callbacks:
            try:
                # Get the appropriate method for this event type
                method = self._get_callback_method(reg_cb.callback, event)
                if method:
                    # Only call if it's not async (skip async methods in sync context)
                    if not asyncio.iscoroutinefunction(method):
                        method(event)
            except Exception as e:
                # Log callback errors but don't let them break execution
                logging.getLogger("logillm.callbacks").warning(
                    f"Error in callback {reg_cb.name}: {e}",
                    exc_info=True,
                )

    def _get_callback_type(self, event: CallbackEventType) -> CallbackType:
        """Get callback type from event."""
        if isinstance(event, ModuleStartEvent):
            return CallbackType.MODULE_START
        elif isinstance(event, ModuleEndEvent):
            return CallbackType.MODULE_END
        elif isinstance(event, OptimizationStartEvent):
            return CallbackType.OPTIMIZATION_START
        elif isinstance(event, OptimizationEndEvent):
            return CallbackType.OPTIMIZATION_END
        elif isinstance(event, ErrorEvent):
            return CallbackType.ERROR
        elif isinstance(event, ProviderRequestEvent):
            return CallbackType.PROVIDER_REQUEST
        elif isinstance(event, ProviderResponseEvent):
            return CallbackType.PROVIDER_RESPONSE
        else:
            # Default to module start for unknown events
            return CallbackType.MODULE_START

    def _get_callback_method(
        self, callback: BaseCallback, event: CallbackEventType
    ) -> Callable | None:
        """Get the appropriate callback method for an event."""
        if isinstance(event, ModuleStartEvent):
            return getattr(callback, "on_module_start", None)
        elif isinstance(event, ModuleEndEvent):
            return getattr(callback, "on_module_end", None)
        elif isinstance(event, OptimizationStartEvent):
            return getattr(callback, "on_optimization_start", None)
        elif isinstance(event, OptimizationEndEvent):
            return getattr(callback, "on_optimization_end", None)
        elif isinstance(event, ErrorEvent):
            return getattr(callback, "on_error", None)
        elif isinstance(event, ProviderRequestEvent):
            return getattr(callback, "on_provider_request", None)
        elif isinstance(event, ProviderResponseEvent):
            return getattr(callback, "on_provider_response", None)
        else:
            return None

    def get_registered_callbacks(self) -> dict[str, list[str]]:
        """Get summary of all registered callbacks.

        Returns:
            Dictionary mapping callback types to lists of callback names.
        """
        result = {"global": []}

        with self._callback_lock:
            # Add global callbacks
            result["global"] = [cb.name for cb in self._global_callbacks if cb.enabled]

            # Add type-specific callbacks - always include all types
            for cb_type in CallbackType:
                result[cb_type.value] = [cb.name for cb in self._callbacks[cb_type] if cb.enabled]

        return result


# Singleton instance
callback_manager = CallbackManager()


# Convenience functions
def register_callback(
    callback: BaseCallback,
    callback_types: list[CallbackType] | None = None,
    priority: Priority | None = None,
    name: str | None = None,
) -> str:
    """Register a callback with the global callback manager."""
    return callback_manager.register(callback, callback_types, priority, name)


def register_global_callback(
    callback: BaseCallback,
    priority: Priority | None = None,
    name: str | None = None,
) -> str:
    """Register a global callback with the global callback manager."""
    return callback_manager.register_global(callback, priority, name)


def unregister_callback(callback_id: str) -> bool:
    """Unregister a callback from the global callback manager."""
    return callback_manager.unregister(callback_id)


def clear_callbacks() -> None:
    """Clear all callbacks from the global callback manager."""
    callback_manager.clear()


def enable_callbacks() -> None:
    """Enable callback execution globally."""
    callback_manager.enable()


def disable_callbacks() -> None:
    """Disable callback execution globally."""
    callback_manager.disable()


async def emit_callback_async(event: CallbackEventType) -> None:
    """Emit a callback event asynchronously."""
    await callback_manager.emit_async(event)


def emit_callback_sync(event: CallbackEventType) -> None:
    """Emit a callback event synchronously."""
    callback_manager.emit_sync(event)


# Context manager for call tracking
class CallbackContextManager:
    """Context manager for tracking callback contexts."""

    def __init__(self, context: CallbackContext | None = None):
        """Initialize context manager.

        Args:
            context: Existing context to use, or None to create new one.
        """
        self.context = context or CallbackContext()
        self.token = None

    def __enter__(self) -> CallbackContext:
        """Enter context and set active call ID."""
        self.token = ACTIVE_CALL_ID.set(self.context.call_id)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous call ID."""
        if self.token:
            ACTIVE_CALL_ID.reset(self.token)


def get_current_call_id() -> str | None:
    """Get the current active call ID."""
    return ACTIVE_CALL_ID.get()


__all__ = [
    # Core types
    "CallbackType",
    "Priority",
    "CallbackContext",
    "CallbackEvent",
    # Event types
    "ModuleStartEvent",
    "ModuleEndEvent",
    "OptimizationStartEvent",
    "OptimizationEndEvent",
    "ErrorEvent",
    "ProviderRequestEvent",
    "ProviderResponseEvent",
    "CallbackEventType",
    # Callback interfaces
    "BaseCallback",
    "AbstractCallback",
    # Built-in callbacks
    "LoggingCallback",
    "MetricsCallback",
    "ProgressCallback",
    # Manager
    "CallbackManager",
    "callback_manager",
    # Convenience functions
    "register_callback",
    "register_global_callback",
    "unregister_callback",
    "clear_callbacks",
    "enable_callbacks",
    "disable_callbacks",
    "emit_callback_async",
    "emit_callback_sync",
    # Context management
    "CallbackContextManager",
    "get_current_call_id",
]
