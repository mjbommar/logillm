"""Base exception hierarchy for LogiLLM."""

from __future__ import annotations

from typing import Any


class LogiLLMError(Exception):
    """Base exception for all LogiLLM errors."""

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
        recoverable: bool = True,
    ) -> None:
        """Initialize exception with enhanced error information.

        Args:
            message: Human-readable error message
            context: Additional context about the error
            suggestions: List of suggested fixes
            recoverable: Whether the error might be recoverable
        """
        super().__init__(message)
        self.context = context or {}
        self.suggestions = suggestions or []
        self.recoverable = recoverable

    def __str__(self) -> str:
        """Enhanced string representation with context and suggestions."""
        parts = [super().__str__()]

        if self.context:
            parts.append(f"Context: {self.context}")

        if self.suggestions:
            parts.append(f"Suggestions: {'; '.join(self.suggestions)}")

        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": str(self.args[0]) if self.args else "",
            "context": self.context,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
        }


class ValidationError(LogiLLMError):
    """Error in data validation."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        expected_type: type | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = value
        if expected_type:
            context["expected_type"] = expected_type.__name__

        super().__init__(message, context=context, **kwargs)
        self.field = field
        self.value = value
        self.expected_type = expected_type


class ConfigurationError(LogiLLMError):
    """Error in configuration or setup."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        config_value: Any = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = config_value

        super().__init__(message, context=context, **kwargs)
        self.config_key = config_key
        self.config_value = config_value


class AdapterError(LogiLLMError):
    """Error in adapter operations (formatting/parsing)."""

    def __init__(
        self,
        message: str,
        *,
        adapter_type: str | None = None,
        input_data: Any = None,
        parsing_stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if adapter_type:
            context["adapter_type"] = adapter_type
        if input_data is not None:
            context["input_data"] = str(input_data)[:200]  # Truncate long data
        if parsing_stage:
            context["parsing_stage"] = parsing_stage

        super().__init__(message, context=context, **kwargs)
        self.adapter_type = adapter_type
        self.input_data = input_data
        self.parsing_stage = parsing_stage


class ProviderError(LogiLLMError):
    """Error in provider communication."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        response_data: Any = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if provider:
            context["provider"] = provider
        if model:
            context["model"] = model
        if status_code:
            context["status_code"] = status_code
        if response_data:
            context["response_data"] = str(response_data)[:200]

        super().__init__(message, context=context, **kwargs)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.response_data = response_data


class OptimizationError(LogiLLMError):
    """Error in optimization process."""

    def __init__(
        self,
        message: str,
        *,
        optimizer_type: str | None = None,
        iteration: int | None = None,
        score: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if optimizer_type:
            context["optimizer_type"] = optimizer_type
        if iteration is not None:
            context["iteration"] = iteration
        if score is not None:
            context["score"] = score

        super().__init__(message, context=context, **kwargs)
        self.optimizer_type = optimizer_type
        self.iteration = iteration
        self.score = score


class SerializationError(LogiLLMError):
    """Error in serialization/deserialization."""

    def __init__(
        self,
        message: str,
        *,
        format: str | None = None,
        object_type: str | None = None,
        operation: str | None = None,  # "serialize" or "deserialize"
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if format:
            context["format"] = format
        if object_type:
            context["object_type"] = object_type
        if operation:
            context["operation"] = operation

        super().__init__(message, context=context, **kwargs)
        self.format = format
        self.object_type = object_type
        self.operation = operation


class CacheError(LogiLLMError):
    """Error in cache operations."""

    def __init__(
        self,
        message: str,
        *,
        cache_type: str | None = None,
        key: str | None = None,
        operation: str | None = None,  # "get", "set", "delete"
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if cache_type:
            context["cache_type"] = cache_type
        if key:
            context["key"] = key
        if operation:
            context["operation"] = operation

        super().__init__(message, context=context, **kwargs)
        self.cache_type = cache_type
        self.key = key
        self.operation = operation


class ModuleError(LogiLLMError):
    """Error in module execution."""

    def __init__(
        self,
        message: str,
        *,
        module_name: str | None = None,
        module_type: str | None = None,
        execution_stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if module_name:
            context["module_name"] = module_name
        if module_type:
            context["module_type"] = module_type
        if execution_stage:
            context["execution_stage"] = execution_stage

        super().__init__(message, context=context, **kwargs)
        self.module_name = module_name
        self.module_type = module_type
        self.execution_stage = execution_stage


class SignatureError(LogiLLMError):
    """Error in signature definition or validation."""

    def __init__(
        self,
        message: str,
        *,
        signature_name: str | None = None,
        field_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if signature_name:
            context["signature_name"] = signature_name
        if field_name:
            context["field_name"] = field_name

        super().__init__(message, context=context, **kwargs)
        self.signature_name = signature_name
        self.field_name = field_name


class TimeoutError(LogiLLMError):
    """Timeout during operation."""

    def __init__(
        self,
        message: str,
        *,
        timeout_duration: float | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if timeout_duration:
            context["timeout_duration"] = timeout_duration
        if operation:
            context["operation"] = operation

        super().__init__(message, context=context, recoverable=True, **kwargs)
        self.timeout_duration = timeout_duration
        self.operation = operation


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if retry_after:
            context["retry_after"] = retry_after

        suggestions = kwargs.pop("suggestions", [])
        if retry_after:
            suggestions.append(f"Retry after {retry_after} seconds")
        else:
            suggestions.append("Implement exponential backoff")

        super().__init__(
            message, context=context, suggestions=suggestions, recoverable=True, **kwargs
        )
        self.retry_after = retry_after


class QuotaExceededError(ProviderError):
    """Quota/usage limit exceeded."""

    def __init__(
        self,
        message: str,
        *,
        quota_type: str | None = None,  # "tokens", "requests", "cost"
        limit: int | float | None = None,
        used: int | float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if quota_type:
            context["quota_type"] = quota_type
        if limit is not None:
            context["limit"] = limit
        if used is not None:
            context["used"] = used

        suggestions = kwargs.pop("suggestions", [])
        suggestions.extend(
            [
                "Check usage limits in provider dashboard",
                "Consider upgrading plan or using different provider",
            ]
        )

        super().__init__(
            message, context=context, suggestions=suggestions, recoverable=False, **kwargs
        )
        self.quota_type = quota_type
        self.limit = limit
        self.used = used


# Exception mapping for common HTTP status codes
HTTP_STATUS_EXCEPTIONS = {
    400: ValidationError,
    401: ProviderError,  # Unauthorized
    403: ProviderError,  # Forbidden
    404: ProviderError,  # Not Found
    429: RateLimitError,
    500: ProviderError,  # Internal Server Error
    502: ProviderError,  # Bad Gateway
    503: ProviderError,  # Service Unavailable
    504: TimeoutError,  # Gateway Timeout
}


def create_exception_from_status(
    status_code: int,
    message: str,
    **kwargs: Any,
) -> LogiLLMError:
    """Create appropriate exception based on HTTP status code."""
    exception_class = HTTP_STATUS_EXCEPTIONS.get(status_code, ProviderError)
    return exception_class(message, status_code=status_code, **kwargs)


__all__ = [
    "LogiLLMError",
    "ValidationError",
    "ConfigurationError",
    "AdapterError",
    "ProviderError",
    "OptimizationError",
    "SerializationError",
    "CacheError",
    "ModuleError",
    "SignatureError",
    "TimeoutError",
    "RateLimitError",
    "QuotaExceededError",
    "HTTP_STATUS_EXCEPTIONS",
    "create_exception_from_status",
]
