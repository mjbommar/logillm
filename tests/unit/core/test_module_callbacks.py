"""Test that Module class properly emits callback events."""

from unittest.mock import patch

import pytest

from logillm.core.callbacks import (
    ErrorEvent,
    ModuleEndEvent,
    ModuleStartEvent,
    callback_manager,
)
from logillm.core.modules import BaseModule
from logillm.core.types import Prediction, Usage


class TestModuleCallbacks:
    """Test Module callback integration."""

    @pytest.fixture
    def clear_callbacks(self):
        """Clear callback manager state before each test."""
        callback_manager.clear()
        callback_manager.enable()
        yield
        callback_manager.clear()

    @pytest.fixture
    def event_recorder(self):
        """Create a callback that records all events."""
        events = []

        class EventRecorder:
            def on_module_start(self, event: ModuleStartEvent):
                events.append(("module_start", event))

            def on_module_end(self, event: ModuleEndEvent):
                events.append(("module_end", event))

            def on_error(self, event: ErrorEvent):
                events.append(("error", event))

        return EventRecorder(), events

    @pytest.mark.asyncio
    async def test_module_emits_start_and_end_events(self, clear_callbacks, event_recorder):
        """Test that modules emit start and end events."""
        callback, events = event_recorder
        callback_manager.register(callback)

        # Create a simple module
        module = BaseModule()

        # Call the module
        await module(test_input="hello")

        # Check that events were emitted
        assert len(events) == 2
        assert events[0][0] == "module_start"
        assert events[1][0] == "module_end"

        # Verify event contents
        start_event = events[0][1]
        assert start_event.module == module
        assert start_event.inputs == {"test_input": "hello"}

        end_event = events[1][1]
        assert end_event.module == module
        assert end_event.outputs == {"test_input": "hello"}
        assert end_event.success is True
        assert end_event.duration > 0

    @pytest.mark.asyncio
    async def test_module_emits_error_event_on_failure(self, clear_callbacks, event_recorder):
        """Test that modules emit error events on failure."""
        callback, events = event_recorder
        callback_manager.register(callback)

        # Create a module that will fail
        async def failing_forward(**inputs):
            raise ValueError("Test error")

        module = BaseModule(forward_fn=failing_forward)

        # Call the module and expect it to fail
        with pytest.raises(Exception):
            await module(test_input="fail")

        # Check that events were emitted
        assert len(events) >= 2  # start and error (possibly end too)
        assert events[0][0] == "module_start"

        # Find error event
        error_events = [e for e in events if e[0] == "error"]
        assert len(error_events) == 1

        error_event = error_events[0][1]
        assert error_event.module == module
        assert error_event.stage == "forward"
        assert isinstance(error_event.error, ValueError)

    @pytest.mark.asyncio
    async def test_module_callbacks_respect_enabled_flag(self, clear_callbacks, event_recorder):
        """Test that callbacks can be disabled."""
        callback, events = event_recorder
        callback_manager.register(callback)

        # Disable callbacks
        callback_manager.disable()

        # Create and call module
        module = BaseModule()
        await module(test_input="hello")

        # No events should be emitted
        assert len(events) == 0

        # Re-enable callbacks
        callback_manager.enable()
        await module(test_input="world")

        # Now events should be emitted
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_module_callbacks_with_environment_variable(self, event_recorder):
        """Test that callbacks respect environment variable."""
        callback, events = event_recorder

        # Clear and re-initialize with callbacks disabled via env var
        callback_manager.clear()
        callback_manager.enable()
        callback_manager.register(callback)

        with patch.dict("os.environ", {"LOGILLM_CALLBACKS_ENABLED": "0"}):
            # Create a new module with callbacks disabled
            module = BaseModule()
            await module(test_input="hello")

        # No events should be emitted
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_context_propagation_in_nested_calls(self, clear_callbacks, event_recorder):
        """Test that context is propagated in nested module calls."""
        callback, events = event_recorder
        callback_manager.register(callback)

        # Create nested modules
        inner_module = BaseModule()

        async def outer_forward(**inputs):
            # Call inner module
            inner_result = await inner_module(**inputs)
            return Prediction(
                outputs={"wrapped": inner_result.outputs},
                usage=Usage(),
                success=True,
            )

        outer_module = BaseModule(forward_fn=outer_forward)

        # Call outer module
        await outer_module(test_input="nested")

        # Should have events for both modules
        assert len(events) == 4  # outer start, inner start, inner end, outer end

        # Check context relationships
        outer_start = events[0][1]
        inner_start = events[1][1]

        # Inner context should reference outer context as parent
        assert inner_start.context.parent_call_id == outer_start.context.call_id

    @pytest.mark.asyncio
    async def test_module_callbacks_with_custom_forward(self, clear_callbacks, event_recorder):
        """Test callbacks work with custom forward implementation."""
        callback, events = event_recorder
        callback_manager.register(callback)

        # Custom forward that returns specific output
        async def custom_forward(**inputs):
            return Prediction(
                outputs={"processed": f"custom-{inputs.get('input', '')}"},
                usage=Usage(tokens={"input": 10, "output": 5}),
                success=True,
            )

        module = BaseModule(forward_fn=custom_forward)
        await module(input="test")

        # Check events
        assert len(events) == 2

        end_event = events[1][1]
        assert end_event.outputs == {"processed": "custom-test"}
        assert end_event.prediction.usage.tokens == {"input": 10, "output": 5}

    def test_module_sync_call_emits_events(self, clear_callbacks, event_recorder):
        """Test that synchronous calls also emit events."""
        callback, events = event_recorder
        callback_manager.register(callback)

        module = BaseModule()

        # Use synchronous call
        module.call_sync(test_input="sync")

        # Events should still be emitted
        assert len(events) == 2
        assert events[0][0] == "module_start"
        assert events[1][0] == "module_end"

    @pytest.mark.asyncio
    async def test_module_callbacks_performance_overhead(self, clear_callbacks):
        """Test that callbacks have minimal performance overhead when disabled."""
        import time

        module = BaseModule()

        # Warm up
        for _ in range(10):
            await module(test="warmup")

        # Measure with callbacks disabled
        callback_manager.disable()
        start = time.perf_counter()
        for _ in range(100):
            await module(test="disabled")
        time_disabled = time.perf_counter() - start

        # Measure with callbacks enabled but no registered callbacks
        callback_manager.enable()
        start = time.perf_counter()
        for _ in range(100):
            await module(test="enabled")
        time_enabled = time.perf_counter() - start

        # The overhead should be minimal (less than 50% increase)
        overhead_ratio = time_enabled / time_disabled
        assert overhead_ratio < 1.5, f"Callback overhead too high: {overhead_ratio:.2f}x"
