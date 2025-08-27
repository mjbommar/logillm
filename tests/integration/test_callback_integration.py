"""Integration tests for callback system end-to-end flow."""

import json
import tempfile
from pathlib import Path

import pytest

from logillm.core.callbacks import (
    AbstractCallback,
    CallbackManager,
    EvaluationEndEvent,
    EvaluationStartEvent,
    ModuleEndEvent,
    ModuleStartEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ProviderRequestEvent,
    ProviderResponseEvent,
)
from logillm.core.predict import Predict
from logillm.providers import register_provider
from logillm.providers.mock import MockProvider


class EventCollector(AbstractCallback):
    """Callback that collects all events for testing."""

    def __init__(self):
        self.events = []

    async def on_module_start(self, event: ModuleStartEvent):
        self.events.append(("module_start", event))

    async def on_module_end(self, event: ModuleEndEvent):
        self.events.append(("module_end", event))

    async def on_optimization_start(self, event: OptimizationStartEvent):
        self.events.append(("optimization_start", event))

    async def on_optimization_end(self, event: OptimizationEndEvent):
        self.events.append(("optimization_end", event))

    async def on_provider_request(self, event: ProviderRequestEvent):
        self.events.append(("provider_request", event))

    async def on_provider_response(self, event: ProviderResponseEvent):
        self.events.append(("provider_response", event))

    async def on_evaluation_start(self, event: EvaluationStartEvent):
        self.events.append(("evaluation_start", event))

    async def on_evaluation_end(self, event: EvaluationEndEvent):
        self.events.append(("evaluation_end", event))


class TestCallbackIntegration:
    """Test integrated callback functionality across the system."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Clear manager before each test
        manager = CallbackManager()
        manager.clear()
        manager.enable()

        # Register mock provider
        provider = MockProvider(response_text="Test answer")
        register_provider(provider, set_default=True)

        yield

        # Clear after test
        manager.clear()

    @pytest.mark.asyncio
    async def test_predict_module_with_callbacks(self):
        """Test that Predict module emits proper callback events."""
        collector = EventCollector()
        manager = CallbackManager()
        manager.register(collector)

        # Create Predict module with simple signature
        module = Predict("question -> answer")

        # Execute module
        result = await module(question="What is 2+2?")

        # Verify events were collected
        assert len(collector.events) >= 4

        # Should have module start/end and provider request/response
        event_types = [evt[0] for evt in collector.events]
        assert "module_start" in event_types
        assert "module_end" in event_types
        assert "provider_request" in event_types
        assert "provider_response" in event_types

        # Verify event order
        start_idx = event_types.index("module_start")
        end_idx = event_types.index("module_end")
        assert start_idx < end_idx

        # Verify result
        assert result.outputs["answer"] == "Test answer"

    @pytest.mark.asyncio
    async def test_nested_module_context_propagation(self):
        """Test that nested modules maintain proper context relationships."""
        collector = EventCollector()
        manager = CallbackManager()
        manager.register(collector)

        # Create nested modules
        inner = Predict("text -> summary")
        outer = Predict("question -> answer")

        # Override outer's forward to call inner
        async def nested_forward(**inputs):
            # First call inner module
            summary = await inner(text=inputs.get("question", ""))
            # Then call the provider for the answer
            from logillm.core.types import Prediction
            return Prediction(outputs={"answer": f"Based on summary: {summary.outputs['summary']}"})

        # Patch the forward method
        outer.forward = nested_forward

        # Execute outer module
        await outer(question="What is AI?")

        # Should have events from both modules
        module_events = [evt for evt in collector.events if evt[0].startswith("module_")]
        assert len(module_events) >= 4  # 2 starts + 2 ends

        # Verify context parent-child relationship
        contexts = [evt[1].context for evt in module_events]
        assert len({ctx.call_id for ctx in contexts}) >= 2  # At least 2 unique contexts

    @pytest.mark.asyncio
    async def test_callback_disable_enable(self):
        """Test that callbacks can be disabled and re-enabled."""
        collector = EventCollector()
        manager = CallbackManager()
        manager.register(collector)

        module = Predict("question -> answer")

        # Execute with callbacks enabled
        await module(question="Test 1")
        events_with_callbacks = len(collector.events)

        # Clear and disable callbacks
        collector.events.clear()
        manager.disable()

        # Execute with callbacks disabled
        await module(question="Test 2")
        events_without_callbacks = len(collector.events)

        # Re-enable callbacks
        collector.events.clear()
        manager.enable()

        # Execute with callbacks re-enabled
        await module(question="Test 3")
        events_reenabled = len(collector.events)

        # Verify
        assert events_with_callbacks > 0
        assert events_without_callbacks == 0
        assert events_reenabled > 0

    @pytest.mark.asyncio
    async def test_callback_with_error_handling(self):
        """Test that callbacks properly handle errors."""
        collector = EventCollector()
        manager = CallbackManager()
        manager.register(collector)

        # Create provider that will fail
        error_provider = MockProvider(error_rate=1.0)
        register_provider(error_provider, set_default=True)

        module = Predict("question -> answer")

        # Execute and expect failure
        with pytest.raises(Exception, match="Mock error"):
            await module(question="Will fail")

        # Should still have start events
        event_types = [evt[0] for evt in collector.events]
        assert "module_start" in event_types
        assert "provider_request" in event_types
        # May or may not have end events depending on error handling

    @pytest.mark.asyncio
    async def test_callback_json_logging(self):
        """Test JSON logging callback functionality."""
        # Create a custom JSON logger
        class JSONLogger(AbstractCallback):
            def __init__(self, filepath):
                self.filepath = filepath
                self.logs = []

            async def on_module_end(self, event: ModuleEndEvent):
                log_entry = {
                    "timestamp": event.timestamp.isoformat(),
                    "module": event.module.__class__.__name__,
                    "success": event.success,
                    "duration": event.duration,
                    "outputs": event.outputs,
                }
                self.logs.append(log_entry)

                # Write to file
                with open(self.filepath, "w") as f:
                    json.dump(self.logs, f, indent=2)

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Register JSON logger
            logger = JSONLogger(str(tmp_path))
            manager = CallbackManager()
            manager.register(logger)

            # Execute modules
            module = Predict("question -> answer")
            await module(question="Test 1")
            await module(question="Test 2")

            # Read and verify JSON log
            with open(tmp_path) as f:
                logs = json.load(f)

            assert len(logs) == 2
            assert all("timestamp" in log for log in logs)
            assert all("module" in log for log in logs)
            assert all(log["module"] == "Predict" for log in logs)

        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_multiple_callbacks_priority(self):
        """Test that multiple callbacks execute in priority order."""
        execution_order = []

        class HighPriorityCallback(AbstractCallback):
            @property
            def priority(self):
                from logillm.core.callbacks import Priority
                return Priority.HIGH

            async def on_module_start(self, event):
                execution_order.append("high")

        class NormalPriorityCallback(AbstractCallback):
            async def on_module_start(self, event):
                execution_order.append("normal")

        class LowPriorityCallback(AbstractCallback):
            @property
            def priority(self):
                from logillm.core.callbacks import Priority
                return Priority.LOW

            async def on_module_start(self, event):
                execution_order.append("low")

        # Register in mixed order
        manager = CallbackManager()
        manager.register(NormalPriorityCallback())
        manager.register(LowPriorityCallback())
        manager.register(HighPriorityCallback())

        # Execute module
        module = Predict("question -> answer")
        await module(question="Test")

        # Verify execution order (high -> normal -> low)
        assert execution_order[0] == "high"
        assert execution_order[1] == "normal"
        assert execution_order[2] == "low"

    @pytest.mark.asyncio
    async def test_callback_performance_overhead(self):
        """Test that callbacks have minimal performance overhead."""
        import time

        module = Predict("question -> answer")
        manager = CallbackManager()

        # Time without callbacks
        manager.disable()
        start = time.time()
        for _ in range(10):
            await module(question="Test")
        time_without = time.time() - start

        # Time with lightweight callback
        manager.enable()

        class LightweightCallback(AbstractCallback):
            async def on_module_end(self, event):
                pass  # Do nothing

        manager.register(LightweightCallback())

        start = time.time()
        for _ in range(10):
            await module(question="Test")
        time_with = time.time() - start

        # Callbacks should add less than 20% overhead for lightweight operations
        overhead = (time_with - time_without) / time_without
        assert overhead < 0.2, f"Callback overhead too high: {overhead:.2%}"
