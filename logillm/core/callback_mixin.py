"""Mixin class for adding callback capabilities to LogiLLM components.

This mixin provides a unified way to emit callback events from modules, optimizers,
and providers. It handles both sync and async event emission with minimal overhead.
"""

from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from .callbacks import CallbackContext, CallbackEvent, CallbackManager

# Thread-local context tracking for callback propagation
# Use Optional instead of union with None to avoid forward reference issues
current_callback_context: ContextVar[Optional[CallbackContext]] = ContextVar(  # noqa
    "current_callback_context", default=None
)


def get_current_context() -> CallbackContext | None:
    """Get the current callback context from thread-local storage."""
    return current_callback_context.get()


class CallbackMixin:
    """Mixin to add callback capabilities to any class.

    This mixin provides:
    - Async and sync event emission methods
    - Context creation and management
    - Environment variable based configuration
    - Minimal overhead when callbacks are disabled
    """

    def __init__(self) -> None:
        """Initialize callback mixin state."""
        self._callback_context: CallbackContext | None = None

        # Check if callbacks are enabled via environment variable
        # This allows disabling callbacks globally without code changes
        self._callback_enabled = os.getenv("LOGILLM_CALLBACKS_ENABLED", "1") == "1"

        # Cache the callback manager to avoid repeated imports
        self._callback_manager: CallbackManager | None = None

    def _get_callback_manager(self) -> CallbackManager | None:
        """Get the callback manager singleton lazily.

        Returns None if callbacks are disabled to avoid import overhead.
        """
        if not self._callback_enabled:
            return None

        if self._callback_manager is None:
            # Lazy import to avoid circular dependencies
            from .callbacks import callback_manager

            self._callback_manager = callback_manager

        return self._callback_manager

    async def _emit_async(self, event: CallbackEvent) -> None:
        """Emit event asynchronously if callbacks enabled.

        Args:
            event: The callback event to emit
        """
        manager = self._get_callback_manager()
        if manager and manager.is_enabled():
            try:
                await manager.emit_async(event)
            except Exception:
                # Silently ignore callback errors to not disrupt main execution
                # Could optionally log these errors in debug mode
                pass

    def _emit_sync(self, event: CallbackEvent) -> None:
        """Emit event synchronously if callbacks enabled.

        Args:
            event: The callback event to emit
        """
        manager = self._get_callback_manager()
        if manager and manager.is_enabled():
            try:
                manager.emit_sync(event)
            except Exception:
                # Silently ignore callback errors to not disrupt main execution
                pass

    def _create_context(self, parent: CallbackContext | None = None) -> CallbackContext:
        """Create a new callback context, optionally linked to parent.

        Args:
            parent: Optional parent context for nested calls

        Returns:
            New CallbackContext instance
        """
        # Lazy import to avoid circular dependencies
        from .callbacks import CallbackContext

        context = CallbackContext(
            call_id=str(uuid4()), parent_call_id=parent.call_id if parent else None, metadata={}
        )

        self._callback_context = context
        return context

    def _emit_async_safe(self, event: CallbackEvent) -> None:
        """Emit event asynchronously in a fire-and-forget manner.

        This is useful when you need to emit from a sync context but don't
        want to block. The event is scheduled on the event loop if one exists.

        Args:
            event: The callback event to emit
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the emission as a task
                asyncio.create_task(self._emit_async(event))
            else:
                # No running loop, fall back to sync emission
                self._emit_sync(event)
        except RuntimeError:
            # No event loop, use sync emission
            self._emit_sync(event)

    def _with_callback_context(self, context: CallbackContext):
        """Context manager for setting the current callback context.

        Args:
            context: The context to set as current

        Returns:
            Context manager that sets and resets the context
        """
        from contextlib import contextmanager

        @contextmanager
        def callback_context_manager():
            token = current_callback_context.set(context)
            try:
                yield context
            finally:
                current_callback_context.reset(token)

        return callback_context_manager()

    def _check_callbacks_enabled(self) -> bool:
        """Check if callbacks are currently enabled.

        Returns:
            True if callbacks are enabled, False otherwise
        """
        if not self._callback_enabled:
            return False

        manager = self._get_callback_manager()
        return manager is not None and manager.is_enabled()

    def _set_callback_enabled(self, enabled: bool) -> None:
        """Enable or disable callbacks for this instance.

        Args:
            enabled: Whether callbacks should be enabled
        """
        self._callback_enabled = enabled
