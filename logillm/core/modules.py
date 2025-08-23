"""Module system for execution strategies."""

from __future__ import annotations

import asyncio
import copy
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from ..exceptions import ConfigurationError, ModuleError
from ..protocols.runtime import Batchable, ExecutableComponent, Optimizable, Serializable
from .signatures import BaseSignature, Signature, parse_signature_string
from .types import (
    Configuration,
    ExecutionTrace,
    Metadata,
    ModuleState,
    Prediction,
    TraceStep,
    Usage,
)

M = TypeVar("M", bound="Module")


@dataclass
class Parameter:
    """Learnable/optimizable parameter in a module."""

    value: Any
    learnable: bool = True
    metadata: Metadata = field(default_factory=dict)

    def optimize(self, traces: list[ExecutionTrace]) -> Parameter:
        """Create optimized version based on traces."""
        # Default implementation - subclasses should override
        return copy.deepcopy(self)

    def reset(self) -> None:
        """Reset parameter to initial state."""
        # Subclasses can implement specific reset logic
        pass


class Module(ABC, ExecutableComponent, Optimizable, Batchable, Serializable):
    """Abstract base class for all modules."""

    def __init__(
        self,
        signature: Signature | str | None = None,
        *,
        config: Configuration | None = None,
        metadata: Metadata | None = None,
        debug: bool | None = None,
    ) -> None:
        self.signature = self._resolve_signature(signature)
        self.config = config or {}
        self.metadata = metadata or {}
        self.state = ModuleState.INITIALIZED
        self.parameters: dict[str, Parameter] = {}
        self.trace: ExecutionTrace | None = None
        self._tracing_enabled = False

        # Debug mode: explicit parameter > environment variable > default (False)
        if debug is not None:
            self._debug_mode = debug
        elif os.environ.get("LOGILLM_DEBUG"):
            self._debug_mode = True
        else:
            self._debug_mode = False

        # Initialize module-specific setup
        self.setup()

    def _resolve_signature(
        self, signature: Signature | str | type | None
    ) -> Signature | BaseSignature | None:
        """Resolve signature from various input formats."""
        if signature is None:
            return None
        elif isinstance(signature, str):
            return parse_signature_string(signature)
        elif isinstance(signature, (Signature, BaseSignature)):
            return signature
        elif (
            isinstance(signature, type)
            and hasattr(signature, "input_fields")
            and hasattr(signature, "output_fields")
        ):
            # Handle signature classes - they have the same interface as instances
            # Signature classes created with our metaclass have all the needed methods
            return signature  # type: ignore[return-value]
        else:
            raise ConfigurationError(
                f"Invalid signature type: {type(signature)}", context={"signature": signature}
            )

    def setup(self) -> None:
        """Override for module-specific initialization."""
        pass

    @abstractmethod
    async def forward(self, **inputs: Any) -> Prediction:
        """Execute the module with given inputs."""
        ...

    async def __call__(self, **inputs: Any) -> Prediction:
        """Call interface - wraps forward with tracing and validation."""
        start_time = time.time()

        try:
            # Validate inputs if signature is available
            if self.signature:
                try:
                    validated_inputs = self.signature.validate_inputs(**inputs)
                except Exception:
                    # If validation fails, use original inputs
                    # This is important for dynamic signatures
                    validated_inputs = inputs
            else:
                validated_inputs = inputs

            # Execute forward pass
            prediction = await self.forward(**validated_inputs)

            # Validate outputs if signature is available
            if self.signature and prediction.outputs:
                try:
                    validated_outputs = self.signature.validate_outputs(**prediction.outputs)
                    prediction.outputs = validated_outputs
                except Exception:
                    # If validation fails, keep original outputs
                    # This is important for dynamic signatures
                    pass

            # Add trace step if tracing enabled
            if self._tracing_enabled and self.trace:
                duration = time.time() - start_time
                step = TraceStep(
                    module_name=self.__class__.__name__,
                    inputs=validated_inputs,
                    outputs=prediction.outputs,
                    usage=prediction.usage,
                    success=prediction.success,
                    duration=duration,
                    metadata={"module_metadata": self.metadata},
                )
                self.trace.add_step(step)

            return prediction

        except Exception as e:
            # Add failed trace step
            if self._tracing_enabled and self.trace:
                duration = time.time() - start_time
                step = TraceStep(
                    module_name=self.__class__.__name__,
                    inputs=inputs,
                    outputs={},
                    usage=Usage(),
                    success=False,
                    duration=duration,
                    metadata={"error": str(e)},
                )
                self.trace.add_step(step)

            # Wrap in ModuleError with context
            raise ModuleError(
                f"Module execution failed: {e}",
                module_name=self.__class__.__name__,
                module_type=type(self).__name__,
                execution_stage="forward",
                context={"inputs": inputs, "original_error": str(e)},
            ) from e

    def call_sync(self, **inputs: Any) -> Prediction:
        """Synchronous call interface."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to use a different approach
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self(**inputs))
                    return future.result()
            else:
                return loop.run_until_complete(self(**inputs))
        except RuntimeError:
            # No event loop
            return asyncio.run(self(**inputs))

    def compile(self, optimizer: Any | None = None) -> Module:
        """Compile module for optimization."""
        if optimizer:
            optimized = optimizer.optimize(self, [], {})  # Would need proper interface
            optimized.state = ModuleState.COMPILED
            return optimized
        else:
            compiled_module = copy.deepcopy(self)
            compiled_module.state = ModuleState.COMPILED
            return compiled_module

    def optimize(self, traces: list[ExecutionTrace], **kwargs: Any) -> Module:
        """Create optimized version based on execution traces."""
        optimized = copy.deepcopy(self)

        # Optimize parameters
        for name, param in optimized.parameters.items():
            if param.learnable:
                optimized.parameters[name] = param.optimize(traces)

        optimized.state = ModuleState.OPTIMIZED
        return optimized

    def optimization_parameters(self) -> dict[str, Any]:
        """Get parameters that can be optimized."""
        return {name: param.value for name, param in self.parameters.items() if param.learnable}

    async def batch_process(self, items: list[dict[str, Any]], **kwargs: Any) -> list[Prediction]:
        """Process multiple items in batch."""
        batch_size = kwargs.get("batch_size", self.optimal_batch_size())
        results = []

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self(**item) for item in batch], return_exceptions=True
            )

            # Convert exceptions to failed predictions
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    processed_results.append(Prediction(success=False, error=str(result)))
                else:
                    processed_results.append(result)

            results.extend(processed_results)

        return results

    def optimal_batch_size(self) -> int:
        """Get optimal batch size for this module."""
        return 10  # Default batch size

    # Configuration management
    def configure(self, config: Configuration) -> None:
        """Apply configuration."""
        if not self.validate_config(config):
            raise ConfigurationError(
                "Invalid configuration provided",
                context={"config": config, "module": self.__class__.__name__},
            )
        self.config.update(config)

    def get_config(self) -> Configuration:
        """Get current configuration."""
        return self.config.copy()

    def validate_config(self, config: Configuration) -> bool:
        """Validate configuration."""
        # Default implementation - always valid
        # Subclasses should override with specific validation
        return True

    # Tracing support
    def enable_tracing(self) -> None:
        """Enable execution tracing."""
        self._tracing_enabled = True
        if self.trace is None:
            self.trace = ExecutionTrace()

    def disable_tracing(self) -> None:
        """Disable execution tracing."""
        self._tracing_enabled = False

    def get_trace(self) -> ExecutionTrace | None:
        """Get current execution trace."""
        return self.trace

    def clear_trace(self) -> None:
        """Clear execution trace."""
        self.trace = ExecutionTrace() if self._tracing_enabled else None

    # Validation
    def validate(self) -> bool:
        """Validate module configuration and state."""
        try:
            errors = self.validation_errors()
            return len(errors) == 0
        except Exception:
            return False

    def validation_errors(self) -> list[str]:
        """Get validation errors."""
        errors = []

        # Validate signature if present
        if self.signature and not self.signature.validate():
            errors.append("Invalid signature")

        # Validate configuration
        if not self.validate_config(self.config):
            errors.append("Invalid configuration")

        return errors

    # Monitoring
    def get_metrics(self) -> dict[str, float]:
        """Get current performance metrics."""
        metrics = {}

        if self.trace:
            metrics["total_steps"] = len(self.trace.steps)
            metrics["success_rate"] = (
                sum(1 for s in self.trace.steps if s.success) / len(self.trace.steps)
                if self.trace.steps
                else 0
            )
            metrics["avg_duration"] = (
                sum(s.duration or 0 for s in self.trace.steps) / len(self.trace.steps)
                if self.trace.steps
                else 0
            )
            metrics["total_tokens"] = self.trace.total_usage.tokens.total_tokens

        return metrics

    def get_health_status(self) -> bool:
        """Get health status."""
        return self.state in {ModuleState.INITIALIZED, ModuleState.COMPILED, ModuleState.OPTIMIZED}

    # Debug support
    def enable_debug_mode(self) -> None:
        """Enable debug mode to capture prompts in predictions."""
        self._debug_mode = True

    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self._debug_mode = False

    def is_debugging(self) -> bool:
        """Check if debug mode is enabled."""
        return self._debug_mode

    def debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        return {
            "class": self.__class__.__name__,
            "state": self.state.value,
            "signature": repr(self.signature) if self.signature else None,
            "config": self.config,
            "parameters": {name: param.value for name, param in self.parameters.items()},
            "tracing_enabled": self._tracing_enabled,
            "debug_mode": self._debug_mode,
            "metrics": self.get_metrics(),
        }

    # Serialization
    def to_dict(self) -> dict[str, Any]:
        """Convert module to dictionary."""
        return {
            "type": self.__class__.__name__,
            "signature": self.signature.to_dict() if self.signature else None,
            "config": self.config,
            "metadata": self.metadata,
            "state": self.state.value,
            "parameters": {
                name: {
                    "value": param.value,
                    "learnable": param.learnable,
                    "metadata": param.metadata,
                }
                for name, param in self.parameters.items()
            },
        }

    def deepcopy(self) -> Module:
        """Create a deep copy of the module."""
        import copy

        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Module:
        """Reconstruct module from dictionary."""
        # This would need to be implemented by concrete subclasses
        raise NotImplementedError("Subclasses must implement from_dict")

    def __repr__(self) -> str:
        """String representation."""
        sig_str = repr(self.signature) if self.signature else "no signature"
        return f"{self.__class__.__name__}({sig_str}, state={self.state.value})"


class BaseModule(Module):
    """Concrete base module implementation."""

    def __init__(
        self,
        forward_fn: Callable[..., Awaitable[Prediction]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._forward_fn = forward_fn

    async def forward(self, **inputs: Any) -> Prediction:
        """Execute the forward function."""
        if self._forward_fn:
            return await self._forward_fn(**inputs)
        else:
            # Default implementation - just return inputs as outputs
            return Prediction(
                outputs=inputs,
                usage=Usage(),
                success=True,
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModule:
        """Reconstruct BaseModule from dictionary."""
        # Reconstruct signature
        signature = None
        if data.get("signature"):
            signature = BaseSignature.from_dict(data["signature"])

        # Reconstruct parameters
        module = cls(
            signature=signature,
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )

        # Restore state
        module.state = ModuleState(data.get("state", "initialized"))

        # Restore parameters
        for name, param_data in data.get("parameters", {}).items():
            module.parameters[name] = Parameter(
                value=param_data["value"],
                learnable=param_data.get("learnable", True),
                metadata=param_data.get("metadata", {}),
            )

        return module


def module_decorator(
    signature: Signature | str | None = None,
    **config: Any,
) -> Callable[[Callable], BaseModule]:
    """Decorator to create modules from functions."""

    def decorator(func: Callable) -> BaseModule:
        # Create async wrapper if function is sync
        if not asyncio.iscoroutinefunction(func):

            async def async_wrapper(**inputs: Any) -> Prediction:
                result = func(**inputs)
                if isinstance(result, Prediction):
                    return result
                elif isinstance(result, dict):
                    return Prediction(outputs=result)
                else:
                    return Prediction(outputs={"output": result})

            forward_fn = async_wrapper
        else:

            async def async_wrapper(**inputs: Any) -> Prediction:
                result = await func(**inputs)
                if isinstance(result, Prediction):
                    return result
                elif isinstance(result, dict):
                    return Prediction(outputs=result)
                else:
                    return Prediction(outputs={"output": result})

            forward_fn = async_wrapper

        # Infer signature from function if not provided
        if signature is None:
            from .signatures import signature_from_function

            inferred_signature = signature_from_function(func)
        else:
            inferred_signature = signature

        return BaseModule(
            forward_fn=forward_fn,
            signature=inferred_signature,
            config=config,
        )

    return decorator


# Convenience aliases
module = module_decorator


__all__ = [
    "Parameter",
    "Module",
    "BaseModule",
    "module_decorator",
    "module",
]
