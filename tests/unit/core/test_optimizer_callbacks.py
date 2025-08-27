"""Tests for optimizer callback functionality."""

import asyncio
from typing import Any

import pytest

from logillm.core.callbacks import (
    AbstractCallback,
    CallbackManager,
    EvaluationEndEvent,
    EvaluationStartEvent,
    HyperparameterUpdateEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
)
from logillm.core.modules import Module
from logillm.core.optimizers import Optimizer
from logillm.core.types import ModuleState, OptimizationResult, Prediction


class MockModule(Module):
    """Mock module for testing."""

    def __init__(self):
        super().__init__(signature=None)
        self.state = ModuleState.INITIALIZED
        self.call_count = 0
        self.name = "mock"

    async def forward(self, **inputs: Any) -> Prediction:
        """Mock forward pass."""
        self.call_count += 1
        return Prediction(outputs={"result": "mock_output"})

    def forward_sync(self, **inputs: Any) -> Prediction:
        """Mock sync forward pass."""
        return asyncio.run(self.forward(**inputs))


class MockOptimizer(Optimizer[MockModule]):
    """Mock optimizer for testing callbacks."""

    def __init__(self, metric=None):
        from logillm.core.types import OptimizationStrategy

        # Create a simple metric if none provided
        if metric is None:
            class SimpleMetric:
                def __call__(self, pred, label):
                    return 1.0
                @property
                def name(self):
                    return "simple_metric"
                def is_better(self, score1, score2):
                    return score1 > score2
            metric = SimpleMetric()

        super().__init__(
            strategy=OptimizationStrategy.BOOTSTRAP,
            metric=metric
        )
        self.optimize_call_count = 0
        self.evaluate_call_count = 0

    async def optimize(
        self,
        module: MockModule,
        dataset: list[dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Mock optimization."""
        import time
        start_time = time.time()
        self.optimize_call_count += 1

        # Create context and emit start event
        context = self._create_context(None)

        if self._check_callbacks_enabled():
            from logillm.core.callbacks import OptimizationStartEvent
            await self._emit_async(OptimizationStartEvent(
                context=context,
                optimizer=self,
                module=module,
                dataset=dataset,
                config=self.config.metadata if hasattr(self.config, 'metadata') else {}
            ))

        # Just simulate optimization work
        await asyncio.sleep(0.01)

        # Simulate hyperparameter updates
        if self._check_callbacks_enabled():
            await self._emit_async(HyperparameterUpdateEvent(
                context=self._create_context(None),
                optimizer=self,
                module=module,
                parameters={"temperature": 0.8, "top_p": 0.95},
                iteration=1
            ))

        # Simulate evaluation
        score = await self.evaluate(module, dataset[:2])  # Eval on subset

        result = OptimizationResult(
            optimized_module=module,
            improvement=0.1,
            iterations=1,
            best_score=score,
            optimization_time=time.time() - start_time,
            metadata={"final_params": {"temperature": 0.8}}
        )

        # Emit end event
        if self._check_callbacks_enabled():
            from logillm.core.callbacks import OptimizationEndEvent
            await self._emit_async(OptimizationEndEvent(
                context=context,
                optimizer=self,
                result=result,
                success=True,
                duration=result.optimization_time
            ))

        return result

    async def evaluate(
        self,
        module: MockModule,
        dataset: list[dict[str, Any]],
        **kwargs: Any
    ) -> float:
        """Mock evaluation."""
        import time
        start_time = time.time()
        self.evaluate_call_count += 1

        # Create context and emit start event
        context = self._create_context(None)

        if self._check_callbacks_enabled():
            from logillm.core.callbacks import EvaluationStartEvent
            await self._emit_async(EvaluationStartEvent(
                context=context,
                optimizer=self,
                module=module,
                dataset=dataset
            ))

        # Simulate evaluation work
        await asyncio.sleep(0.01)

        # Calculate mock score
        score = 0.85  # Fixed score for testing

        # Emit end event
        if self._check_callbacks_enabled():
            from logillm.core.callbacks import EvaluationEndEvent
            await self._emit_async(EvaluationEndEvent(
                context=context,
                optimizer=self,
                module=module,
                score=score,
                duration=time.time() - start_time
            ))

        return score


class TestOptimizerCallbacks:
    """Test optimizer callback functionality."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Clear manager before each test
        manager = CallbackManager()
        manager.clear()
        manager.enable()  # Ensure callbacks are enabled
        yield
        # Clear after test
        manager.clear()

    @pytest.mark.asyncio
    async def test_optimizer_emits_optimization_start_event(self):
        """Test that optimizer emits optimization start event."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_optimization_start(self, event: OptimizationStartEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        await optimizer.optimize(module, dataset)

        assert len(events) == 1
        assert isinstance(events[0], OptimizationStartEvent)
        assert events[0].optimizer == optimizer
        assert events[0].module == module
        assert events[0].dataset == dataset

    @pytest.mark.asyncio
    async def test_optimizer_emits_optimization_end_event(self):
        """Test that optimizer emits optimization end event."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_optimization_end(self, event: OptimizationEndEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        result = await optimizer.optimize(module, dataset)

        assert len(events) == 1
        assert isinstance(events[0], OptimizationEndEvent)
        assert events[0].optimizer == optimizer
        assert events[0].result == result
        assert events[0].duration > 0

    @pytest.mark.asyncio
    async def test_optimizer_emits_evaluation_events(self):
        """Test that optimizer emits evaluation start and end events."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_evaluation_start(self, event: EvaluationStartEvent):
                events.append(("eval_start", event))

            async def on_evaluation_end(self, event: EvaluationEndEvent):
                events.append(("eval_end", event))

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        await optimizer.evaluate(module, dataset)

        assert len(events) == 2
        assert events[0][0] == "eval_start"
        assert events[1][0] == "eval_end"
        assert isinstance(events[0][1], EvaluationStartEvent)
        assert isinstance(events[1][1], EvaluationEndEvent)
        assert events[1][1].score == 0.85

    @pytest.mark.asyncio
    async def test_optimizer_emits_hyperparameter_update_event(self):
        """Test that optimizer emits hyperparameter update events."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_hyperparameter_update(self, event: HyperparameterUpdateEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        await optimizer.optimize(module, dataset)

        assert len(events) == 1
        assert isinstance(events[0], HyperparameterUpdateEvent)
        assert events[0].optimizer == optimizer
        assert events[0].module == module
        assert events[0].parameters == {"temperature": 0.8, "top_p": 0.95}
        assert events[0].iteration == 1

    @pytest.mark.asyncio
    async def test_optimizer_callbacks_with_score_tracking(self):
        """Test that optimizer tracks scores across evaluations."""
        scores = []

        class TestHandler(AbstractCallback):
            async def on_evaluation_end(self, event: EvaluationEndEvent):
                scores.append(event.score)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        # Run multiple evaluations
        for _ in range(3):
            await optimizer.evaluate(module, dataset)

        assert len(scores) == 3
        assert all(score == 0.85 for score in scores)

    @pytest.mark.asyncio
    async def test_optimizer_callbacks_disabled(self):
        """Test that callbacks can be disabled."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_optimization_start(self, event: OptimizationStartEvent):
                events.append(event)
            async def on_optimization_end(self, event: OptimizationEndEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())
        manager.disable()  # Disable callbacks

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        await optimizer.optimize(module, dataset)

        assert len(events) == 0  # No events should be emitted

    @pytest.mark.asyncio
    async def test_optimizer_context_propagation(self):
        """Test that optimizer maintains context across events."""
        start_contexts = []
        end_contexts = []

        class TestHandler(AbstractCallback):
            async def on_optimization_start(self, event: OptimizationStartEvent):
                start_contexts.append(event.context)
            async def on_optimization_end(self, event: OptimizationEndEvent):
                end_contexts.append(event.context)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        await optimizer.optimize(module, dataset)

        assert len(start_contexts) == 1
        assert len(end_contexts) == 1
        assert start_contexts[0].call_id == end_contexts[0].call_id  # Same context

    @pytest.mark.asyncio
    async def test_optimizer_nested_evaluation_contexts(self):
        """Test that nested evaluations have parent-child context relationships."""
        contexts = []

        class TestHandler(AbstractCallback):
            async def on_evaluation_start(self, event: EvaluationStartEvent):
                contexts.append(event.context)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        # Optimization calls evaluation internally
        await optimizer.optimize(module, dataset)

        # Should have one evaluation context from the internal call
        assert len(contexts) == 1

    @pytest.mark.asyncio
    async def test_optimizer_multiple_iterations_tracking(self):
        """Test tracking multiple optimization iterations."""
        iterations = []

        class TestHandler(AbstractCallback):
            async def on_hyperparameter_update(self, event: HyperparameterUpdateEvent):
                iterations.append(event.iteration)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        # Run multiple optimizations
        for _ in range(3):
            await optimizer.optimize(module, dataset)

        assert len(iterations) == 3
        assert all(i == 1 for i in iterations)  # MockOptimizer always reports iteration 1

    @pytest.mark.asyncio
    async def test_optimizer_metadata_in_events(self):
        """Test that optimizer events include relevant metadata."""
        end_events = []

        class TestHandler(AbstractCallback):
            async def on_optimization_end(self, event: OptimizationEndEvent):
                end_events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        await optimizer.optimize(module, dataset)

        assert len(end_events) == 1
        event = end_events[0]
        assert event.result.best_score == 0.85
        assert event.result.iterations == 1
        assert event.result.metadata["final_params"]["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_optimizer_error_handling(self):
        """Test that optimizer handles errors gracefully with callbacks."""

        class ErrorOptimizer(MockOptimizer):
            async def optimize(self, module, dataset, **kwargs):
                # Emit start event
                if self._check_callbacks_enabled():
                    await self._emit_async(OptimizationStartEvent(
                        context=self._create_context(None),
                        optimizer=self,
                        module=module,
                        dataset=dataset,
                        config=kwargs
                    ))
                # Then fail
                raise ValueError("Optimization failed")

        events = []

        class TestHandler(AbstractCallback):
            async def on_optimization_start(self, event: OptimizationStartEvent):
                events.append(("start", event))
            async def on_optimization_end(self, event: OptimizationEndEvent):
                events.append(("end", event))

        manager = CallbackManager()
        manager.register(TestHandler())

        optimizer = ErrorOptimizer()
        module = MockModule()
        dataset = [{"input": "test", "label": "expected"}]

        with pytest.raises(ValueError, match="Optimization failed"):
            await optimizer.optimize(module, dataset)

        # Should have start event but no end event
        assert len(events) == 1
        assert events[0][0] == "start"

    @pytest.mark.asyncio
    async def test_optimizer_callback_performance(self):
        """Test that callbacks have acceptable performance overhead."""
        import time

        # Time without callbacks
        manager = CallbackManager()
        manager.disable()

        optimizer = MockOptimizer()
        module = MockModule()
        dataset = [{"input": f"test{i}", "label": f"expected{i}"} for i in range(10)]

        start = time.time()
        for _ in range(10):
            await optimizer.evaluate(module, dataset)
        time_without = time.time() - start

        # Time with callbacks
        manager.enable()

        class TestHandler(AbstractCallback):
            async def on_evaluation_start(self, event: EvaluationStartEvent):
                pass
            async def on_evaluation_end(self, event: EvaluationEndEvent):
                pass

        manager.register(TestHandler())

        start = time.time()
        for _ in range(10):
            await optimizer.evaluate(module, dataset)
        time_with = time.time() - start

        # Callbacks should add less than 50% overhead
        overhead = time_with / time_without
        assert overhead < 1.5, f"Callback overhead too high: {overhead:.2f}x"
