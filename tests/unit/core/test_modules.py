"""Unit tests for module system."""

import pytest

from logillm.core.modules import Module, Parameter
from logillm.core.signatures import BaseSignature
from logillm.core.types import ExecutionTrace, ModuleState, Prediction, Usage
from logillm.exceptions import ConfigurationError


class TestParameter:
    """Test Parameter class."""

    def test_parameter_creation(self):
        """Test creating a parameter."""
        param = Parameter(value={"temperature": 0.7}, learnable=True, metadata={"type": "config"})

        assert param.value == {"temperature": 0.7}
        assert param.learnable is True
        assert param.metadata == {"type": "config"}

    def test_parameter_optimize_default(self):
        """Test default optimize method returns deep copy."""
        param = Parameter(value=[1, 2, 3])
        traces = []

        optimized = param.optimize(traces)

        assert optimized.value == param.value
        assert optimized is not param  # Should be a new object
        assert optimized.value is not param.value  # Should be deep copied

    def test_parameter_reset(self):
        """Test reset method (default does nothing)."""
        param = Parameter(value=42)
        param.reset()
        assert param.value == 42  # Should remain unchanged


class ConcreteModule(Module):
    """Concrete module implementation for testing."""

    async def forward(self, **inputs):
        """Simple forward implementation."""
        return Prediction(outputs={"result": "test"}, success=True, usage=Usage())


class TestModule:
    """Test Module abstract base class."""

    def test_module_initialization_with_string_signature(self):
        """Test module init with string signature."""
        module = ConcreteModule(signature="input -> output")

        assert module.signature is not None
        assert module.config == {}
        assert module.metadata == {}
        assert module.state == ModuleState.INITIALIZED
        assert module.parameters == {}
        assert module.trace is None
        assert module._tracing_enabled is False
        assert module._debug_mode is False

    def test_module_initialization_with_signature_object(self):
        """Test module init with Signature object."""
        sig = BaseSignature(input_fields={}, output_fields={})
        module = ConcreteModule(signature=sig)

        assert module.signature == sig

    def test_module_initialization_with_config(self):
        """Test module init with custom config."""
        config = {"temperature": 0.5, "max_tokens": 100}
        module = ConcreteModule(config=config)

        assert module.config == config

    def test_module_initialization_with_metadata(self):
        """Test module init with metadata."""
        metadata = {"version": "1.0", "author": "test"}
        module = ConcreteModule(metadata=metadata)

        assert module.metadata == metadata

    def test_module_initialization_with_none_signature(self):
        """Test module init with None signature."""
        module = ConcreteModule(signature=None)
        assert module.signature is None

    def test_module_initialization_with_invalid_signature(self):
        """Test module init with invalid signature type."""
        with pytest.raises(ConfigurationError) as exc_info:
            ConcreteModule(signature=123)  # Invalid type

        assert "Invalid signature type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_module_call_delegates_to_forward(self):
        """Test that __call__ delegates to forward."""
        module = ConcreteModule()

        result = await module(test="input")

        assert isinstance(result, Prediction)
        assert result.success is True
        assert result.outputs == {"result": "test"}

    def test_module_call_sync(self):
        """Test call_sync method."""
        module = ConcreteModule()

        # Test that call_sync works and returns a Prediction
        result = module.call_sync(test="input")

        assert isinstance(result, Prediction)
        assert result.success is True
        assert result.outputs == {"result": "test"}

    def test_module_enable_tracing(self):
        """Test enabling tracing."""
        module = ConcreteModule()

        module.enable_tracing()

        assert module._tracing_enabled is True
        assert module.trace is not None  # Should create trace

    def test_module_disable_tracing(self):
        """Test disabling tracing."""
        module = ConcreteModule()
        module._tracing_enabled = True

        module.disable_tracing()

        assert module._tracing_enabled is False

    def test_module_get_trace(self):
        """Test getting execution trace."""
        module = ConcreteModule()
        trace = ExecutionTrace(steps=[], total_usage=Usage())
        module.trace = trace

        retrieved_trace = module.get_trace()

        assert retrieved_trace == trace

    def test_module_clear_trace(self):
        """Test clearing execution trace."""
        module = ConcreteModule()
        module._tracing_enabled = True
        module.trace = ExecutionTrace(steps=[], total_usage=Usage())

        module.clear_trace()

        # When tracing is enabled, it creates a new empty trace
        assert module.trace is not None
        assert len(module.trace.steps) == 0

    def test_module_enable_debug_mode(self):
        """Test enabling/disabling debug mode."""
        module = ConcreteModule()

        module.enable_debug_mode()
        assert module._debug_mode is True

        module.disable_debug_mode()
        assert module._debug_mode is False

    @pytest.mark.asyncio
    async def test_module_batch_process(self):
        """Test batch_process method processes multiple inputs."""
        module = ConcreteModule()

        inputs_list = [{"input": "test1"}, {"input": "test2"}, {"input": "test3"}]

        results = await module.batch_process(inputs_list)

        assert len(results) == 3
        assert all(isinstance(r, Prediction) for r in results)
        assert all(r.success for r in results)

    def test_module_compile_returns_copy(self):
        """Test compile method returns deep copy."""
        module = ConcreteModule(config={"test": "value"})
        module.parameters["param1"] = Parameter(value=42)

        compiled = module.compile()

        assert compiled is not module
        assert compiled.config == module.config
        assert compiled.config is not module.config  # Deep copied
        assert compiled.parameters["param1"].value == 42

    def test_parameter_reset(self):
        """Test parameter reset method."""
        # Module doesn't have reset, but Parameter does
        param = Parameter(value=10)
        param.reset()  # Default implementation does nothing
        assert param.value == 10

    def test_module_to_dict(self):
        """Test serialization to dict."""
        module = ConcreteModule(
            signature="input -> output", config={"temperature": 0.7}, metadata={"version": "1.0"}
        )
        module.parameters["demo"] = Parameter(value=[1, 2, 3])

        module_dict = module.to_dict()

        assert "config" in module_dict
        assert module_dict["config"] == {"temperature": 0.7}
        assert "metadata" in module_dict
        assert module_dict["metadata"] == {"version": "1.0"}
        assert "parameters" in module_dict

    def test_module_from_dict_not_implemented(self):
        """Test from_dict raises NotImplementedError."""
        module_dict = {"signature": None, "config": {"max_tokens": 100}, "metadata": {"test": True}}

        # Base Module.from_dict raises NotImplementedError
        with pytest.raises(NotImplementedError):
            Module.from_dict(module_dict)

    def test_module_setup_called_on_init(self):
        """Test setup method is called during initialization."""

        class SetupTestModule(Module):
            def __init__(self, *args, **kwargs):
                self.setup_called = False
                super().__init__(*args, **kwargs)

            def setup(self):
                self.setup_called = True

            async def forward(self, **inputs):
                return Prediction(outputs={}, success=True)

        module = SetupTestModule()
        assert module.setup_called is True

    def test_module_signature_type_class_handling(self):
        """Test module handles signature type classes."""
        # Module accepts signature classes with proper interface
        from logillm.core.signatures import BaseSignature

        # Create a valid signature
        sig = BaseSignature(input_fields={}, output_fields={})

        module = ConcreteModule(signature=sig)

        assert module.signature == sig

    def test_module_state_transitions(self):
        """Test module state transitions."""
        module = ConcreteModule()

        assert module.state == ModuleState.INITIALIZED

        module.state = ModuleState.CONFIGURED
        assert module.state == ModuleState.CONFIGURED

        module.state = ModuleState.OPTIMIZED
        assert module.state == ModuleState.OPTIMIZED
