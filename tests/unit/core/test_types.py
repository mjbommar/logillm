"""Unit tests for core types."""

import pytest

from logillm.core.types import (
    AdapterFormat,
    Comparable,
    Completion,
    Configuration,
    ExecutionTrace,
    FieldType,
    Hashable,
    Metadata,
    ModuleState,
    OptimizationResult,
    Prediction,
    TokenUsage,
    TraceStep,
    Usage,
)


class TestEnums:
    """Test enum types."""

    def test_field_type_enum(self):
        """Test FieldType enum values."""
        assert FieldType.INPUT.value == "input"
        assert FieldType.OUTPUT.value == "output"
        assert FieldType.INTERMEDIATE.value == "intermediate"
        assert FieldType.CONTEXT.value == "context"

    def test_adapter_format_enum(self):
        """Test AdapterFormat enum values."""
        assert AdapterFormat.CHAT.value == "chat"
        assert AdapterFormat.JSON.value == "json"
        assert AdapterFormat.XML.value == "xml"
        assert AdapterFormat.MARKDOWN.value == "markdown"
        assert AdapterFormat.FUNCTION.value == "function"
        assert AdapterFormat.COMPLETION.value == "completion"
        assert AdapterFormat.STRUCTURED.value == "structured"

    def test_module_state_enum(self):
        """Test ModuleState enum values."""
        assert ModuleState.INITIALIZED.value == "initialized"
        assert ModuleState.CONFIGURED.value == "configured"
        assert ModuleState.COMPILED.value == "compiled"
        assert ModuleState.OPTIMIZED.value == "optimized"
        assert ModuleState.CACHED.value == "cached"


class TestTypeAliases:
    """Test type aliases."""

    def test_configuration_type(self):
        """Test Configuration type alias."""
        config: Configuration = {"temperature": 0.7, "max_tokens": 100}
        assert isinstance(config, dict)

    def test_metadata_type(self):
        """Test Metadata type alias."""
        metadata: Metadata = {"source": "test", "version": "1.0"}
        assert isinstance(metadata, dict)


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage."""
        usage = TokenUsage(input_tokens=10, output_tokens=20, cached_tokens=5, reasoning_tokens=3)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.cached_tokens == 5
        assert usage.reasoning_tokens == 3
        assert usage.total_tokens == 33  # 10 + 20 + 3 (property)

    def test_token_usage_defaults(self):
        """Test TokenUsage with default values."""
        usage = TokenUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_add(self):
        """Test adding two TokenUsage instances."""
        usage1 = TokenUsage(10, 20, 30)
        usage2 = TokenUsage(5, 15, 20)

        combined = usage1 + usage2

        assert combined.input_tokens == 15
        assert combined.output_tokens == 35
        assert combined.total_tokens == 50


class TestUsage:
    """Test Usage dataclass."""

    def test_usage_creation(self):
        """Test creating Usage."""
        tokens = TokenUsage(10, 20, 30)
        usage = Usage(tokens=tokens, latency=100.5, cost=0.001)

        assert usage.tokens == tokens
        assert usage.latency == 100.5
        assert usage.cost == 0.001

    def test_usage_defaults(self):
        """Test Usage with default values."""
        usage = Usage()

        assert usage.tokens.input_tokens == 0
        assert usage.tokens.output_tokens == 0
        assert usage.latency is None
        assert usage.cost is None

    def test_usage_add(self):
        """Test adding two Usage instances."""
        usage1 = Usage(tokens=TokenUsage(10, 20, 30), latency=100, cost=0.001)
        usage2 = Usage(tokens=TokenUsage(5, 15, 20), latency=50, cost=0.0005)

        combined = usage1 + usage2

        assert combined.tokens.input_tokens == 15
        assert combined.tokens.output_tokens == 35
        assert combined.latency == 100  # max of the two
        assert combined.cost == 0.0015


class TestCompletion:
    """Test Completion dataclass."""

    def test_completion_creation(self):
        """Test creating Completion."""
        completion = Completion(
            text="Generated text",
            usage=Usage(tokens=TokenUsage(5, 10, 15)),
            metadata={"model": "test"},
        )

        assert completion.text == "Generated text"
        assert completion.usage.tokens.total_tokens == 15
        assert completion.metadata == {"model": "test"}

    def test_completion_defaults(self):
        """Test Completion with defaults."""
        completion = Completion(text="Test")

        assert completion.text == "Test"
        assert completion.usage.tokens.total_tokens == 0
        assert completion.metadata == {}


class TestPrediction:
    """Test Prediction dataclass."""

    def test_prediction_creation(self):
        """Test creating Prediction."""
        prediction = Prediction(
            outputs={"answer": "42"},
            success=True,
            error=None,
            usage=Usage(tokens=TokenUsage(5, 10, 15)),
            metadata={"confidence": 0.9},
        )

        assert prediction.outputs == {"answer": "42"}
        assert prediction.success is True
        assert prediction.error is None
        assert prediction.usage.tokens.total_tokens == 15
        assert prediction.metadata == {"confidence": 0.9}

    def test_prediction_failure(self):
        """Test failed prediction."""
        prediction = Prediction(outputs={}, success=False, error="Something went wrong")

        assert prediction.success is False
        assert prediction.error == "Something went wrong"
        assert prediction.outputs == {}

    def test_prediction_attribute_access(self):
        """Test dot notation access for outputs."""
        prediction = Prediction(outputs={"key1": "value1", "key2": "value2"}, success=True)

        assert prediction.key1 == "value1"
        assert prediction.key2 == "value2"

        # Test missing attribute raises error
        with pytest.raises(AttributeError):
            _ = prediction.missing


class TestTraceStep:
    """Test TraceStep dataclass."""

    def test_trace_step_creation(self):
        """Test creating TraceStep."""
        step = TraceStep(
            module_name="process",
            inputs={"x": 1},
            outputs={"y": 2},
            usage=Usage(),
            success=True,
            duration=50.5,
            metadata={"step": 1},
        )

        assert step.module_name == "process"
        assert step.inputs == {"x": 1}
        assert step.outputs == {"y": 2}
        assert step.duration == 50.5
        assert step.success is True
        assert step.metadata == {"step": 1}

    def test_trace_step_defaults(self):
        """Test TraceStep with defaults."""
        step = TraceStep(module_name="test", inputs={}, outputs={}, usage=Usage(), success=True)

        assert step.module_name == "test"
        assert step.inputs == {}
        assert step.outputs == {}
        assert step.success is True
        assert step.metadata == {}


class TestExecutionTrace:
    """Test ExecutionTrace dataclass."""

    def test_execution_trace_creation(self):
        """Test creating ExecutionTrace."""
        steps = [
            TraceStep(
                module_name="step1", inputs={}, outputs={}, usage=Usage(), success=True, duration=10
            ),
            TraceStep(
                module_name="step2", inputs={}, outputs={}, usage=Usage(), success=True, duration=20
            ),
        ]

        trace = ExecutionTrace(steps=steps, total_usage=Usage(tokens=TokenUsage(5, 10, 15)))

        assert len(trace.steps) == 2
        assert trace.total_usage.tokens.total_tokens == 15

    def test_execution_trace_defaults(self):
        """Test ExecutionTrace with defaults."""
        trace = ExecutionTrace()

        assert trace.steps == []
        assert trace.total_usage.tokens.total_tokens == 0


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating OptimizationResult."""
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        result = OptimizationResult(
            optimized_module=mock_module,
            improvement=0.15,
            iterations=10,
            best_score=0.95,
            optimization_time=120.5,
            metadata={"strategy": "hybrid"},
        )

        assert result.optimized_module == mock_module
        assert result.improvement == 0.15
        assert result.iterations == 10
        assert result.best_score == 0.95
        assert result.optimization_time == 120.5
        assert result.metadata == {"strategy": "hybrid"}

    def test_optimization_result_defaults(self):
        """Test OptimizationResult with defaults."""
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        result = OptimizationResult(
            optimized_module=mock_module,
            improvement=0.1,
            iterations=5,
            best_score=0.8,
            optimization_time=60.0,
        )

        assert result.metadata == {}


class TestProtocols:
    """Test protocol types (runtime checkable)."""

    def test_hashable_protocol(self):
        """Test Hashable protocol."""
        # String is hashable
        assert isinstance("test", Hashable)
        assert isinstance(42, Hashable)
        assert isinstance((1, 2), Hashable)

        # List is not hashable
        assert not isinstance([1, 2], Hashable)
        assert not isinstance({"key": "value"}, Hashable)

    def test_comparable_protocol(self):
        """Test Comparable protocol."""
        # Numbers are comparable
        assert isinstance(42, Comparable)
        assert isinstance(3.14, Comparable)
        assert isinstance("string", Comparable)
