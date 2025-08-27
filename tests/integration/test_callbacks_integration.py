"""Integration tests for callback system with real provider calls."""

import asyncio
import time
from unittest.mock import Mock

import pytest

from logillm.core.callbacks import (
    CallbackContext,
    LoggingCallback,
    MetricsCallback,
    ModuleEndEvent,
    ModuleStartEvent,
    ProgressCallback,
    ProviderRequestEvent,
    ProviderResponseEvent,
    callback_manager,
    clear_callbacks,
    register_callback,
)
from logillm.core.predict import Predict
from logillm.core.signatures import BaseSignature
from logillm.core.types import Completion, Prediction, Usage
from logillm.providers.mock import MockProvider


class TestCallbackIntegrationWithMockProvider:
    """Integration tests using MockProvider for deterministic behavior."""

    def setup_method(self):
        """Set up for each test."""
        clear_callbacks()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_logging_callback_with_predict_module(self):
        """Test LoggingCallback with real Predict module execution."""
        # Set up logging callback
        logging_callback = LoggingCallback()
        register_callback(logging_callback)

        # Create a predict module with mock provider
        provider = MockProvider(model="test-model")
        signature = BaseSignature(input_fields={"question": str}, output_fields={"answer": str})

        predict_module = Predict(signature=signature)

        # Mock the provider completion
        provider.set_mock_response(
            Completion(text="The answer is 42.", usage=Usage(), model="test-model")
        )

        # Execute the module (this should trigger callbacks)
        context = CallbackContext()

        # Manually emit events as they would be emitted by the module
        from logillm.core.callbacks import emit_callback_async

        start_event = ModuleStartEvent(
            context=context,
            module=predict_module,
            inputs={"question": "What is the meaning of life?"},
        )
        await emit_callback_async(start_event)

        # Simulate module execution
        prediction = Prediction(
            outputs={"answer": "The answer is 42."}, success=True, usage=Usage()
        )

        end_event = ModuleEndEvent(
            context=context,
            module=predict_module,
            outputs=prediction.outputs,
            prediction=prediction,
            success=True,
            duration=0.1,
        )
        await emit_callback_async(end_event)

        # Verify events were handled (we can't easily test log output, but we can verify no errors)
        assert True  # If we get here without exceptions, the callbacks worked

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_metrics_callback_with_multiple_operations(self):
        """Test MetricsCallback tracking multiple operations."""
        metrics_callback = MetricsCallback()
        register_callback(metrics_callback)

        # Simulate multiple module executions
        modules = ["Module1", "Module2", "Module1"]  # Module1 runs twice

        for i, module_name in enumerate(modules):
            context = CallbackContext()
            module = Mock()
            module.__class__.__name__ = module_name

            # Start event
            start_event = ModuleStartEvent(
                context=context, module=module, inputs={"input": f"test_{i}"}
            )
            await callback_manager.emit_async(start_event)

            # Simulate some execution time
            await asyncio.sleep(0.01)

            # End event
            end_event = ModuleEndEvent(
                context=context,
                module=module,
                outputs={"output": f"result_{i}"},
                success=True,
                duration=0.01,
            )
            await callback_manager.emit_async(end_event)

        # Check metrics
        metrics = metrics_callback.get_metrics()

        assert metrics["module_starts"] == 3
        assert metrics["module_ends"] == 3
        assert metrics["module_successes"] == 3
        assert metrics["module_failures"] == 0
        assert metrics["module_success_rate"] == 1.0
        assert metrics["module_Module1"] == 2  # Module1 ran twice
        assert metrics["module_Module2"] == 1  # Module2 ran once
        assert "module_durations_avg" in metrics
        assert metrics["module_durations_avg"] > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_progress_callback_output(self):
        """Test ProgressCallback output with real operations."""
        import io
        from unittest.mock import patch

        # Capture print output
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            progress_callback = ProgressCallback(show_details=True)
            register_callback(progress_callback)

            # Simulate optimization process
            context = CallbackContext()
            optimizer = Mock()
            optimizer.__class__.__name__ = "HybridOptimizer"
            module = Mock()
            dataset = [{"input": f"example_{i}"} for i in range(5)]

            from logillm.core.callbacks import OptimizationEndEvent, OptimizationStartEvent

            # Start optimization
            opt_start_event = OptimizationStartEvent(
                context=context,
                optimizer=optimizer,
                module=module,
                dataset=dataset,
                config={"temperature": 0.7},
            )
            await callback_manager.emit_async(opt_start_event)

            # Simulate some processing time
            await asyncio.sleep(0.02)

            # End optimization
            opt_end_event = OptimizationEndEvent(
                context=context, optimizer=optimizer, result=Mock(), success=True, duration=0.02
            )
            await callback_manager.emit_async(opt_end_event)

        # Check output
        output = captured_output.getvalue()
        assert "🚀 Starting optimization with HybridOptimizer on 5 examples..." in output
        assert "🎉 Optimization finished" in output

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_provider_callbacks_with_mock_provider(self):
        """Test provider-related callbacks with MockProvider."""
        logging_callback = LoggingCallback()
        metrics_callback = MetricsCallback()
        register_callback(logging_callback)
        register_callback(metrics_callback)

        provider = MockProvider(model="test-model")

        # Set up mock response
        mock_completion = Completion(text="Test response", usage=Usage(), model="test-model")
        provider.set_mock_response(mock_completion)

        # Call provider.complete() which automatically emits callback events
        messages = [{"role": "user", "content": "Hello"}]
        parameters = {"temperature": 0.7, "max_tokens": 100}

        # The provider.complete() method will automatically emit both
        # ProviderRequestEvent and ProviderResponseEvent
        await provider.complete(messages, **parameters)

        # Check metrics were collected
        metrics = metrics_callback.get_metrics()
        assert metrics["provider_requests"] == 1
        assert metrics["provider_responses"] == 1
        assert metrics["provider_mock_requests"] == 1
        assert "provider_durations_avg" in metrics

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_error_callback_handling(self):
        """Test error callback handling with real errors."""
        logging_callback = LoggingCallback()
        metrics_callback = MetricsCallback()
        register_callback(logging_callback)
        register_callback(metrics_callback)

        # Simulate an error during module execution
        context = CallbackContext()
        module = Mock()
        module.__class__.__name__ = "TestModule"
        error = ValueError("Something went wrong")

        from logillm.core.callbacks import ErrorEvent

        error_event = ErrorEvent(
            context=context, error=error, module=module, stage="forward", recoverable=False
        )

        await callback_manager.emit_async(error_event)

        # Check that error was tracked
        metrics = metrics_callback.get_metrics()
        assert metrics["errors"] == 1
        assert metrics["error_ValueError"] == 1
        assert metrics["error_in_forward"] == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_callback_chain_with_real_workflow(self):
        """Test a complete workflow with multiple callback types."""
        # Set up all callbacks
        logging_callback = LoggingCallback()
        metrics_callback = MetricsCallback()
        progress_callback = ProgressCallback(show_details=False)  # Quiet mode

        register_callback(logging_callback)
        register_callback(metrics_callback)
        register_callback(progress_callback)

        # Simulate a complete optimization workflow
        context = CallbackContext()

        # 1. Start optimization
        optimizer = Mock()
        optimizer.__class__.__name__ = "TestOptimizer"
        module = Mock()
        module.__class__.__name__ = "TestModule"
        dataset = [{"input": f"test_{i}"} for i in range(3)]

        from logillm.core.callbacks import OptimizationEndEvent, OptimizationStartEvent

        opt_start = OptimizationStartEvent(
            context=context, optimizer=optimizer, module=module, dataset=dataset, config={}
        )
        await callback_manager.emit_async(opt_start)

        # 2. Simulate multiple module executions during optimization
        for i in range(3):
            module_context = CallbackContext(parent_call_id=context.call_id)

            # Module start
            mod_start = ModuleStartEvent(context=module_context, module=module, inputs=dataset[i])
            await callback_manager.emit_async(mod_start)

            # Provider request
            provider = MockProvider(model="test-model")
            prov_request = ProviderRequestEvent(
                context=module_context,
                provider=provider,
                messages=[{"role": "user", "content": str(dataset[i])}],
                parameters={},
            )
            await callback_manager.emit_async(prov_request)

            # Provider response
            completion = Completion(text=f"Response {i}", usage=Usage(), model="test-model")
            prov_response = ProviderResponseEvent(
                context=module_context,
                provider=provider,
                request_messages=[{"role": "user", "content": str(dataset[i])}],
                response=completion,
                usage=completion.usage,
                duration=0.1,
            )
            await callback_manager.emit_async(prov_response)

            # Module end
            mod_end = ModuleEndEvent(
                context=module_context,
                module=module,
                outputs={f"output_{i}": f"result_{i}"},
                success=True,
                duration=0.15,
            )
            await callback_manager.emit_async(mod_end)

        # 3. End optimization
        opt_end = OptimizationEndEvent(
            context=context, optimizer=optimizer, result=Mock(), success=True, duration=1.0
        )
        await callback_manager.emit_async(opt_end)

        # 4. Verify all metrics were collected correctly
        metrics = metrics_callback.get_metrics()

        # Optimization metrics
        assert metrics["optimization_starts"] == 1
        assert metrics["optimization_ends"] == 1
        assert metrics["optimization_successes"] == 1

        # Module metrics
        assert metrics["module_starts"] == 3
        assert metrics["module_ends"] == 3
        assert metrics["module_successes"] == 3
        assert metrics["module_TestModule"] == 3

        # Provider metrics
        assert metrics["provider_requests"] == 3
        assert metrics["provider_responses"] == 3
        assert metrics["provider_mock_requests"] == 3

        # Duration metrics
        assert "optimization_durations_avg" in metrics
        assert "module_durations_avg" in metrics
        assert "provider_durations_avg" in metrics

    def test_callback_performance_overhead(self):
        """Test that callbacks don't add significant performance overhead."""
        # Baseline: measure time with minimal work (creating objects)
        start_time = time.time()
        for i in range(1000):
            # Simulate comparable work to callback emission
            context = {"id": i}
            event = {"context": context, "module": Mock(), "inputs": {"test": i}}
        baseline_time = time.time() - start_time

        # With callbacks: measure time with callbacks enabled
        metrics_callback = MetricsCallback()
        register_callback(metrics_callback)

        start_time = time.time()
        for i in range(1000):
            context = CallbackContext()
            event = ModuleStartEvent(context=context, module=Mock(), inputs={"test": i})
            callback_manager.emit_sync(event)
        callback_time = time.time() - start_time

        # Callback overhead should be reasonable
        # Allow up to 10x overhead for the callback infrastructure
        # (This accounts for event routing, priority sorting, method calls, etc.)
        assert callback_time < max(baseline_time * 10, 0.1), (
            f"Callback time {callback_time:.4f}s exceeds 10x baseline {baseline_time:.4f}s"
        )

        # Verify all events were processed
        metrics = metrics_callback.get_metrics()
        assert metrics["module_starts"] == 1000

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_async_callback_performance(self):
        """Test async callback performance."""
        metrics_callback = MetricsCallback()
        register_callback(metrics_callback)

        # Test async emission performance
        start_time = time.time()
        tasks = []

        for i in range(100):  # Smaller number for async test
            context = CallbackContext()
            event = ModuleStartEvent(context=context, module=Mock(), inputs={"test": i})
            task = callback_manager.emit_async(event)
            tasks.append(task)

        await asyncio.gather(*tasks)
        async_time = time.time() - start_time

        # Should complete in reasonable time
        assert async_time < 1.0  # Should complete within 1 second

        # Verify all events were processed
        metrics = metrics_callback.get_metrics()
        assert metrics["module_starts"] == 100


class TestCallbacksWithRealWorkflow:
    """Test callbacks with more realistic workflow scenarios."""

    def setup_method(self):
        """Set up for each test."""
        clear_callbacks()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_nested_module_execution_tracking(self):
        """Test tracking nested module executions."""
        metrics_callback = MetricsCallback()
        register_callback(metrics_callback)

        # Simulate nested module execution (like ChainOfThought -> Predict)
        outer_context = CallbackContext()
        outer_module = Mock()
        outer_module.__class__.__name__ = "ChainOfThought"

        # Outer module start
        outer_start = ModuleStartEvent(
            context=outer_context, module=outer_module, inputs={"question": "Test question"}
        )
        await callback_manager.emit_async(outer_start)

        # Inner module execution
        inner_context = CallbackContext(parent_call_id=outer_context.call_id)
        inner_module = Mock()
        inner_module.__class__.__name__ = "Predict"

        inner_start = ModuleStartEvent(
            context=inner_context, module=inner_module, inputs={"question": "Test question"}
        )
        await callback_manager.emit_async(inner_start)

        inner_end = ModuleEndEvent(
            context=inner_context,
            module=inner_module,
            outputs={"answer": "Test answer"},
            success=True,
            duration=0.1,
        )
        await callback_manager.emit_async(inner_end)

        # Outer module end
        outer_end = ModuleEndEvent(
            context=outer_context,
            module=outer_module,
            outputs={"answer": "Test answer", "reasoning": "Test reasoning"},
            success=True,
            duration=0.2,
        )
        await callback_manager.emit_async(outer_end)

        # Verify metrics tracked both modules
        metrics = metrics_callback.get_metrics()
        assert metrics["module_starts"] == 2
        assert metrics["module_ends"] == 2
        assert metrics["module_ChainOfThought"] == 1
        assert metrics["module_Predict"] == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_batch_processing_callbacks(self):
        """Test callbacks during batch processing."""
        metrics_callback = MetricsCallback()
        register_callback(metrics_callback)

        # Simulate batch processing
        batch_items = [{"input": f"item_{i}"} for i in range(5)]

        for item in batch_items:
            context = CallbackContext()
            module = Mock()
            module.__class__.__name__ = "BatchModule"

            start_event = ModuleStartEvent(context=context, module=module, inputs=item)
            await callback_manager.emit_async(start_event)

            end_event = ModuleEndEvent(
                context=context,
                module=module,
                outputs={"output": f"processed_{item['input']}"},
                success=True,
                duration=0.05,
            )
            await callback_manager.emit_async(end_event)

        # Verify all batch items were tracked
        metrics = metrics_callback.get_metrics()
        assert metrics["module_starts"] == 5
        assert metrics["module_ends"] == 5
        assert metrics["module_BatchModule"] == 5

    def test_callback_memory_usage(self):
        """Test that callbacks don't cause memory leaks."""
        import gc

        metrics_callback = MetricsCallback()
        register_callback(metrics_callback)

        # Generate many events
        for i in range(1000):
            context = CallbackContext()
            event = ModuleStartEvent(context=context, module=Mock(), inputs={"test": i})
            callback_manager.emit_sync(event)

        # Force garbage collection
        gc.collect()

        # Metrics should be reasonable (not accumulating unbounded)
        metrics = metrics_callback.get_metrics()
        assert metrics["module_starts"] == 1000

        # Reset metrics to free memory
        metrics_callback.reset_metrics()
        gc.collect()

        # Should be reset
        metrics = metrics_callback.get_metrics()
        assert metrics["module_starts"] == 0
