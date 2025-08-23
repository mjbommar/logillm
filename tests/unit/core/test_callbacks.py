"""Unit tests for callback system."""

import logging
import threading
from unittest.mock import Mock, patch

import pytest

from logillm.core.callbacks import (
    AbstractCallback,
    CallbackContext,
    CallbackManager,
    CallbackType,
    ErrorEvent,
    LoggingCallback,
    MetricsCallback,
    ModuleEndEvent,
    ModuleStartEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    Priority,
    ProgressCallback,
    ProviderRequestEvent,
    ProviderResponseEvent,
    RegisteredCallback,
    callback_manager,
    clear_callbacks,
    disable_callbacks,
    emit_callback_async,
    emit_callback_sync,
    enable_callbacks,
    get_current_call_id,
    register_callback,
    register_global_callback,
    unregister_callback,
)
from logillm.core.types import Prediction, Usage


class TestCallbackContext:
    """Test CallbackContext class."""

    def test_callback_context_creation(self):
        """Test creating a callback context."""
        context = CallbackContext()

        assert isinstance(context.call_id, str)
        assert len(context.call_id) == 32  # UUID hex string
        assert context.parent_call_id is None
        assert isinstance(context.metadata, dict)
        assert isinstance(context.data, dict)

    def test_callback_context_with_parent(self):
        """Test creating a context with parent."""
        parent_context = CallbackContext()
        context = CallbackContext(parent_call_id=parent_context.call_id)

        assert context.parent_call_id == parent_context.call_id

    def test_callback_context_data_operations(self):
        """Test adding and getting data from context."""
        context = CallbackContext()

        context.add_data("test_key", "test_value")
        assert context.get_data("test_key") == "test_value"

        assert context.get_data("nonexistent") is None
        assert context.get_data("nonexistent", "default") == "default"


class TestEvents:
    """Test callback event classes."""

    def test_module_start_event(self):
        """Test ModuleStartEvent creation."""
        context = CallbackContext()
        module = Mock()
        inputs = {"input": "test"}

        event = ModuleStartEvent(context=context, module=module, inputs=inputs)

        assert event.context == context
        assert event.module == module
        assert event.inputs == inputs
        assert hasattr(event, "timestamp")

    def test_module_end_event(self):
        """Test ModuleEndEvent creation."""
        context = CallbackContext()
        module = Mock()
        outputs = {"output": "result"}
        prediction = Prediction(outputs=outputs, success=True)

        event = ModuleEndEvent(
            context=context,
            module=module,
            outputs=outputs,
            prediction=prediction,
            success=True,
            duration=1.5,
        )

        assert event.context == context
        assert event.module == module
        assert event.outputs == outputs
        assert event.prediction == prediction
        assert event.success is True
        assert event.duration == 1.5

    def test_optimization_start_event(self):
        """Test OptimizationStartEvent creation."""
        context = CallbackContext()
        optimizer = Mock()
        module = Mock()
        dataset = [{"input": "test1"}, {"input": "test2"}]
        config = {"temperature": 0.7}

        event = OptimizationStartEvent(
            context=context, optimizer=optimizer, module=module, dataset=dataset, config=config
        )

        assert event.context == context
        assert event.optimizer == optimizer
        assert event.module == module
        assert event.dataset == dataset
        assert event.config == config

    def test_optimization_end_event(self):
        """Test OptimizationEndEvent creation."""
        context = CallbackContext()
        optimizer = Mock()
        result = Mock()

        event = OptimizationEndEvent(
            context=context, optimizer=optimizer, result=result, success=True, duration=30.0
        )

        assert event.context == context
        assert event.optimizer == optimizer
        assert event.result == result
        assert event.success is True
        assert event.duration == 30.0

    def test_error_event(self):
        """Test ErrorEvent creation."""
        context = CallbackContext()
        error = ValueError("Test error")
        module = Mock()

        event = ErrorEvent(
            context=context, error=error, module=module, stage="forward", recoverable=True
        )

        assert event.context == context
        assert event.error == error
        assert event.module == module
        assert event.stage == "forward"
        assert event.recoverable is True

    def test_provider_request_event(self):
        """Test ProviderRequestEvent creation."""
        context = CallbackContext()
        provider = Mock()
        messages = [{"role": "user", "content": "test"}]
        parameters = {"temperature": 0.7}

        event = ProviderRequestEvent(
            context=context, provider=provider, messages=messages, parameters=parameters
        )

        assert event.context == context
        assert event.provider == provider
        assert event.messages == messages
        assert event.parameters == parameters

    def test_provider_response_event(self):
        """Test ProviderResponseEvent creation."""
        context = CallbackContext()
        provider = Mock()
        messages = [{"role": "user", "content": "test"}]
        response = Mock()
        usage = Usage()

        event = ProviderResponseEvent(
            context=context,
            provider=provider,
            request_messages=messages,
            response=response,
            usage=usage,
            duration=0.5,
        )

        assert event.context == context
        assert event.provider == provider
        assert event.request_messages == messages
        assert event.response == response
        assert event.usage == usage
        assert event.duration == 0.5


class MockCallback(AbstractCallback):
    """Mock callback for testing."""

    def __init__(self, name: str = "MockCallback", priority: Priority = Priority.NORMAL):
        self._name = name
        self._priority = priority
        self.events_received = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> Priority:
        return self._priority

    def on_module_start(self, event: ModuleStartEvent) -> None:
        self.events_received.append(("module_start", event))

    def on_module_end(self, event: ModuleEndEvent) -> None:
        self.events_received.append(("module_end", event))

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self.events_received.append(("optimization_start", event))

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self.events_received.append(("optimization_end", event))

    def on_error(self, event: ErrorEvent) -> None:
        self.events_received.append(("error", event))

    def on_provider_request(self, event: ProviderRequestEvent) -> None:
        self.events_received.append(("provider_request", event))

    def on_provider_response(self, event: ProviderResponseEvent) -> None:
        self.events_received.append(("provider_response", event))


class AsyncMockCallback(MockCallback):
    """Async mock callback for testing."""

    async def on_module_start(self, event: ModuleStartEvent) -> None:
        self.events_received.append(("async_module_start", event))

    async def on_module_end(self, event: ModuleEndEvent) -> None:
        self.events_received.append(("async_module_end", event))


class TestCallbackManager:
    """Test CallbackManager class."""

    def setup_method(self):
        """Set up for each test."""
        # Clear callbacks before each test
        callback_manager.clear()
        # Ensure callbacks are enabled
        callback_manager.enable()

    def test_singleton_pattern(self):
        """Test that CallbackManager is a singleton."""
        manager1 = CallbackManager()
        manager2 = CallbackManager()

        assert manager1 is manager2
        assert manager1 is callback_manager

    def test_register_callback(self):
        """Test registering a callback."""
        callback = MockCallback()
        callback_id = callback_manager.register(callback)

        assert isinstance(callback_id, str)
        assert callback_id.startswith("MockCallback_")

        registered = callback_manager.get_registered_callbacks()
        assert "global" in registered
        # Should be registered for all callback types
        for cb_type in CallbackType:
            assert cb_type.value in registered
            assert "MockCallback" in registered[cb_type.value]

    def test_register_callback_specific_types(self):
        """Test registering a callback for specific types."""
        callback = MockCallback()
        callback_types = [CallbackType.MODULE_START, CallbackType.MODULE_END]

        callback_id = callback_manager.register(callback, callback_types)

        assert isinstance(callback_id, str)

        registered = callback_manager.get_registered_callbacks()

        # Should only be registered for specified types
        assert "MockCallback" in registered[CallbackType.MODULE_START.value]
        assert "MockCallback" in registered[CallbackType.MODULE_END.value]
        assert "MockCallback" not in registered.get(CallbackType.OPTIMIZATION_START.value, [])

    def test_register_global_callback(self):
        """Test registering a global callback."""
        callback = MockCallback()
        callback_id = callback_manager.register_global(callback)

        assert isinstance(callback_id, str)
        assert callback_id.startswith("global_MockCallback_")

        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" in registered["global"]

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        callback = MockCallback()
        callback_id = callback_manager.register(callback)

        # Verify it's registered
        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" in registered[CallbackType.MODULE_START.value]

        # Unregister it
        result = callback_manager.unregister(callback_id)
        assert result is True

        # Verify it's gone
        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" not in registered.get(CallbackType.MODULE_START.value, [])

    def test_unregister_global_callback(self):
        """Test unregistering a global callback."""
        callback = MockCallback()
        callback_id = callback_manager.register_global(callback)

        # Verify it's registered
        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" in registered["global"]

        # Unregister it
        result = callback_manager.unregister(callback_id)
        assert result is True

        # Verify it's gone
        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" not in registered["global"]

    def test_unregister_by_name(self):
        """Test unregistering callbacks by name."""
        callback1 = MockCallback("TestCallback")
        callback2 = MockCallback("TestCallback")
        callback3 = MockCallback("OtherCallback")

        callback_manager.register(callback1)
        callback_manager.register_global(callback2)
        callback_manager.register(callback3)

        # Unregister by name
        removed = callback_manager.unregister_by_name("TestCallback")
        assert removed > 0

        # Verify only OtherCallback remains
        registered = callback_manager.get_registered_callbacks()
        assert "OtherCallback" in registered[CallbackType.MODULE_START.value]
        assert "TestCallback" not in registered.get("global", [])

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        callback1 = MockCallback("Callback1")
        callback2 = MockCallback("Callback2")

        callback_manager.register(callback1)
        callback_manager.register_global(callback2)

        # Verify callbacks are registered
        registered = callback_manager.get_registered_callbacks()
        assert len(registered["global"]) > 0 or any(len(v) > 0 for v in registered.values())

        # Clear all
        callback_manager.clear()

        # Verify all are gone
        registered = callback_manager.get_registered_callbacks()
        assert len(registered["global"]) == 0
        assert all(len(v) == 0 for k, v in registered.items() if k != "global")

    def test_enable_disable_callbacks(self):
        """Test enabling and disabling callbacks."""
        assert callback_manager.is_enabled() is True

        callback_manager.disable()
        assert callback_manager.is_enabled() is False

        callback_manager.enable()
        assert callback_manager.is_enabled() is True

    def test_emit_sync(self):
        """Test emitting events synchronously."""
        callback = MockCallback()
        callback_manager.register(callback, [CallbackType.MODULE_START])

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={"test": "value"})

        callback_manager.emit_sync(event)

        assert len(callback.events_received) == 1
        assert callback.events_received[0][0] == "module_start"
        assert callback.events_received[0][1] == event

    @pytest.mark.asyncio
    async def test_emit_async(self):
        """Test emitting events asynchronously."""
        callback = MockCallback()
        callback_manager.register(callback, [CallbackType.MODULE_START])

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={"test": "value"})

        await callback_manager.emit_async(event)

        assert len(callback.events_received) == 1
        assert callback.events_received[0][0] == "module_start"

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test async callbacks work properly."""
        callback = AsyncMockCallback()
        callback_manager.register(callback, [CallbackType.MODULE_START])

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={"test": "value"})

        await callback_manager.emit_async(event)

        assert len(callback.events_received) == 1
        assert callback.events_received[0][0] == "async_module_start"

    def test_priority_ordering(self):
        """Test that callbacks are executed in priority order."""
        low_callback = MockCallback("Low", Priority.LOW)
        high_callback = MockCallback("High", Priority.HIGH)
        normal_callback = MockCallback("Normal", Priority.NORMAL)

        callback_manager.register(low_callback, [CallbackType.MODULE_START])
        callback_manager.register(high_callback, [CallbackType.MODULE_START])
        callback_manager.register(normal_callback, [CallbackType.MODULE_START])

        # Track execution order
        execution_order = []

        def track_execution(name):
            def wrapper(original_method):
                def tracked_method(event):
                    execution_order.append(name)
                    return original_method(event)

                return tracked_method

            return wrapper

        low_callback.on_module_start = track_execution("Low")(low_callback.on_module_start)
        high_callback.on_module_start = track_execution("High")(high_callback.on_module_start)
        normal_callback.on_module_start = track_execution("Normal")(normal_callback.on_module_start)

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={})

        callback_manager.emit_sync(event)

        # Should execute in priority order: High -> Normal -> Low
        assert execution_order == ["High", "Normal", "Low"]

    def test_callback_error_handling(self):
        """Test that callback errors don't break execution."""

        class ErrorCallback(MockCallback):
            def on_module_start(self, event):
                raise ValueError("Callback error")

        good_callback = MockCallback("Good")
        error_callback = ErrorCallback("Error")

        callback_manager.register(good_callback, [CallbackType.MODULE_START])
        callback_manager.register(error_callback, [CallbackType.MODULE_START])

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={})

        # Should not raise exception despite error in callback
        with patch("logging.getLogger") as mock_logger:
            callback_manager.emit_sync(event)
            # Should have logged the error
            mock_logger.return_value.warning.assert_called()

        # Good callback should still have been executed
        assert len(good_callback.events_received) == 1

    def test_disabled_callback_manager(self):
        """Test that disabled callback manager doesn't execute callbacks."""
        callback = MockCallback()
        callback_manager.register(callback, [CallbackType.MODULE_START])

        callback_manager.disable()

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={})

        callback_manager.emit_sync(event)

        # Should not have received any events
        assert len(callback.events_received) == 0


class TestBuiltinCallbacks:
    """Test built-in callback implementations."""

    def setup_method(self):
        """Set up for each test."""
        callback_manager.clear()
        # Ensure callbacks are enabled
        callback_manager.enable()

    def test_logging_callback(self):
        """Test LoggingCallback functionality."""
        # Create a mock logger
        mock_logger = Mock()
        callback = LoggingCallback(logger=mock_logger, level=logging.DEBUG)

        # Test module start
        context = CallbackContext()
        module = Mock()
        module.__class__.__name__ = "TestModule"
        event = ModuleStartEvent(context=context, module=module, inputs={"test": "input"})

        callback.on_module_start(event)

        mock_logger.log.assert_called_with(
            logging.DEBUG,
            f"Module TestModule starting execution (call_id: {context.call_id})",
            extra={
                "call_id": context.call_id,
                "module_name": "TestModule",
                "inputs": {"test": "input"},
            },
        )

    def test_metrics_callback(self):
        """Test MetricsCallback functionality."""
        callback = MetricsCallback()

        # Test module events
        context = CallbackContext()
        module = Mock()
        module.__class__.__name__ = "TestModule"

        start_event = ModuleStartEvent(context=context, module=module, inputs={})
        callback.on_module_start(start_event)

        end_event = ModuleEndEvent(
            context=context, module=module, outputs={}, success=True, duration=1.5
        )
        callback.on_module_end(end_event)

        # Check metrics
        metrics = callback.get_metrics()
        assert metrics["module_starts"] == 1
        assert metrics["module_ends"] == 1
        assert metrics["module_successes"] == 1
        assert metrics["module_failures"] == 0
        assert metrics["module_success_rate"] == 1.0
        assert metrics["module_durations_avg"] == 1.5
        assert metrics["module_TestModule"] == 1

    def test_metrics_callback_reset(self):
        """Test MetricsCallback reset functionality."""
        callback = MetricsCallback()

        # Add some metrics
        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={})
        callback.on_module_start(event)

        metrics = callback.get_metrics()
        assert metrics["module_starts"] == 1

        # Reset metrics
        callback.reset_metrics()

        metrics = callback.get_metrics()
        assert metrics["module_starts"] == 0

    def test_progress_callback(self):
        """Test ProgressCallback functionality."""
        with patch("builtins.print") as mock_print:
            callback = ProgressCallback(show_details=True)

            # Test module start
            context = CallbackContext()
            module = Mock()
            module.__class__.__name__ = "TestModule"

            start_event = ModuleStartEvent(context=context, module=module, inputs={})
            callback.on_module_start(start_event)

            mock_print.assert_called_with("🔄 Starting TestModule...")

            # Test module end
            end_event = ModuleEndEvent(
                context=context, module=module, outputs={}, success=True, duration=1.5
            )
            callback.on_module_end(end_event)

            mock_print.assert_called_with("✅ Finished TestModule (1.50s)")


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up for each test."""
        clear_callbacks()
        # Ensure callbacks are enabled
        enable_callbacks()

    def test_register_callback_function(self):
        """Test register_callback function."""
        callback = MockCallback()
        callback_id = register_callback(callback)

        assert isinstance(callback_id, str)

        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" in registered[CallbackType.MODULE_START.value]

    def test_register_global_callback_function(self):
        """Test register_global_callback function."""
        callback = MockCallback()
        callback_id = register_global_callback(callback)

        assert isinstance(callback_id, str)

        registered = callback_manager.get_registered_callbacks()
        assert "MockCallback" in registered["global"]

    def test_unregister_callback_function(self):
        """Test unregister_callback function."""
        callback = MockCallback()
        callback_id = register_callback(callback)

        result = unregister_callback(callback_id)
        assert result is True

    def test_clear_callbacks_function(self):
        """Test clear_callbacks function."""
        callback = MockCallback()
        register_callback(callback)

        clear_callbacks()

        registered = callback_manager.get_registered_callbacks()
        assert all(len(v) == 0 for v in registered.values())

    def test_enable_disable_callbacks_functions(self):
        """Test enable_callbacks and disable_callbacks functions."""
        enable_callbacks()
        assert callback_manager.is_enabled() is True

        disable_callbacks()
        assert callback_manager.is_enabled() is False

    def test_emit_callback_sync_function(self):
        """Test emit_callback_sync function."""
        callback = MockCallback()
        register_callback(callback, [CallbackType.MODULE_START])

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={})

        emit_callback_sync(event)

        assert len(callback.events_received) == 1

    @pytest.mark.asyncio
    async def test_emit_callback_async_function(self):
        """Test emit_callback_async function."""
        callback = MockCallback()
        register_callback(callback, [CallbackType.MODULE_START])

        context = CallbackContext()
        event = ModuleStartEvent(context=context, module=Mock(), inputs={})

        await emit_callback_async(event)

        assert len(callback.events_received) == 1


class TestContextManagement:
    """Test callback context management."""

    def test_get_current_call_id_no_context(self):
        """Test getting call ID when no context is active."""
        call_id = get_current_call_id()
        assert call_id is None

    def test_context_manager(self):
        """Test CallbackContextManager context manager."""
        context = CallbackContext()

        # Should be no active context initially
        assert get_current_call_id() is None

        # Context manager should set active context
        from logillm.core.callbacks import CallbackContextManager

        with CallbackContextManager(context):
            assert get_current_call_id() == context.call_id

        # Should restore previous context (None) after exiting
        assert get_current_call_id() is None


class TestThreadSafety:
    """Test thread safety of callback manager."""

    def setup_method(self):
        """Set up for each test."""
        clear_callbacks()
        # Ensure callbacks are enabled
        enable_callbacks()

    def test_concurrent_registration(self):
        """Test that concurrent callback registration is thread-safe."""
        results = []

        def register_callbacks(thread_id):
            for i in range(10):
                callback = MockCallback(f"Thread{thread_id}Callback{i}")
                callback_id = register_callback(callback)
                results.append(callback_id)

        # Start multiple threads registering callbacks
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_callbacks, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have registered all callbacks successfully
        assert len(results) == 50  # 5 threads * 10 callbacks each
        assert len(set(results)) == 50  # All IDs should be unique

        registered = callback_manager.get_registered_callbacks()
        # Should have all callbacks registered
        total_registered = sum(len(v) for v in registered.values() if isinstance(v, list))
        # Each callback is registered for all callback types
        expected_total = 50 * len(CallbackType)
        assert total_registered == expected_total

    def test_concurrent_emission(self):
        """Test that concurrent event emission is thread-safe."""
        callback = MockCallback()
        register_callback(callback)

        def emit_events(thread_id):
            for i in range(10):
                context = CallbackContext()
                event = ModuleStartEvent(
                    context=context, module=Mock(), inputs={"thread_id": thread_id, "event_id": i}
                )
                emit_callback_sync(event)

        # Start multiple threads emitting events
        threads = []
        for i in range(3):
            thread = threading.Thread(target=emit_events, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have received all events
        assert len(callback.events_received) == 30  # 3 threads * 10 events each


class TestRegisteredCallback:
    """Test RegisteredCallback dataclass."""

    def test_registered_callback_creation(self):
        """Test creating a RegisteredCallback."""
        callback = MockCallback()
        reg_callback = RegisteredCallback(
            callback=callback, priority=Priority.HIGH, name="TestCallback", enabled=True
        )

        assert reg_callback.callback == callback
        assert reg_callback.priority == Priority.HIGH
        assert reg_callback.name == "TestCallback"
        assert reg_callback.enabled is True

    def test_registered_callback_defaults(self):
        """Test RegisteredCallback with default values."""
        callback = MockCallback()
        reg_callback = RegisteredCallback(callback=callback, priority=Priority.NORMAL, name="Test")

        assert reg_callback.enabled is True  # Default value
