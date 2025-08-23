"""Exception hierarchy for LogiLLM."""

from .base import (
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

__all__ = [
    "LogiLLMError",
    "ValidationError",
    "AdapterError",
    "ProviderError",
    "OptimizationError",
    "SerializationError",
    "CacheError",
    "ConfigurationError",
    "ModuleError",
    "SignatureError",
    "TimeoutError",
    "RateLimitError",
    "QuotaExceededError",
]
