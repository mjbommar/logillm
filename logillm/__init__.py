"""
LogiLLM: A generic, high-performance, low-dependency LLM programming framework.

Inspired by DSPy but designed for better performance, fewer dependencies,
and cleaner abstractions.

Core Architecture:
    Application Layer (User Programs)
        |
    Module Layer (Execution Strategies)
        |
    Signature Layer (I/O Specifications)
        |
    Adapter Layer (Format Conversion)
        |
    Provider Layer (LM Communication)
        |
    Core Layer (Base Abstractions)
"""

__version__ = "0.1.0"

# Core exports
from .core.modules import (
    BaseModule,
    Module,
    Parameter,
    module,
    module_decorator,
)
from .core.signatures import (
    BaseSignature,
    FieldSpec,
    Signature,
    parse_signature_string,
    signature_from_function,
)
from .core.types import (
    AdapterFormat,
    CacheLevel,
    Completion,
    ExecutionMode,
    ExecutionTrace,
    FieldType,
    ModuleState,
    OptimizationStrategy,
    Prediction,
    SerializationFormat,
    TokenUsage,
    TraceStep,
    Usage,
    ValidationLevel,
)

# TODO: Implement these modules
# from .core.adapters import (
#     Adapter,
#     Formatter,
#     Parser,
# )
# from .core.providers import (
#     Provider,
#     Completion,
#     Usage,
# )
# from .core.optimizers import (
#     Optimizer,
#     Metric,
#     Trace,
# )
# from .fields import (
#     Field,
#     InputField,
#     OutputField,
#     BaseField,
# )
from .exceptions import (
    AdapterError,
    CacheError,
    ConfigurationError,
    LogiLLMError,
    ModuleError,
    OptimizationError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    SerializationError,
    SignatureError,
    TimeoutError,
    ValidationError,
)

# Utility exports - TODO: Implement
# from .utils.decorators import (
#     cached,
#     traced,
#     validated,
#     retry,
#     async_cached,
# )
from .protocols.runtime import (
    Batchable,
    Cacheable,
    Configurable,
    Optimizable,
    Serializable,
    Streamable,
    Traceable,
    Validatable,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "FieldType",
    "ModuleState",
    "OptimizationStrategy",
    "AdapterFormat",
    "ExecutionMode",
    "CacheLevel",
    "ValidationLevel",
    "SerializationFormat",
    "Completion",
    "Usage",
    "Prediction",
    "TokenUsage",
    "ExecutionTrace",
    "TraceStep",
    # Core abstractions
    "Signature",
    "BaseSignature",
    "FieldSpec",
    "parse_signature_string",
    "signature_from_function",
    "Module",
    "BaseModule",
    "Parameter",
    "module",
    "module_decorator",
    # Exceptions
    "LogiLLMError",
    "ValidationError",
    "AdapterError",
    "ProviderError",
    "OptimizationError",
    "ConfigurationError",
    "SerializationError",
    "CacheError",
    "ModuleError",
    "SignatureError",
    "TimeoutError",
    "RateLimitError",
    "QuotaExceededError",
    # Protocols
    "Serializable",
    "Cacheable",
    "Validatable",
    "Optimizable",
    "Traceable",
    "Configurable",
    "Streamable",
    "Batchable",
]
