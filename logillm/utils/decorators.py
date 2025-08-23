"""Decorators for LogiLLM framework."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def cached(
    ttl: int | None = 3600,
    key_func: Callable[..., str] | None = None,
) -> Callable[[F], F]:
    """Cache decorator for functions.

    Args:
        ttl: Time-to-live in seconds (None = no expiration)
        key_func: Custom function to generate cache key
    """

    def decorator(func: F) -> F:
        cache: dict[str, tuple[Any, float | None]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "args": args,
                    "kwargs": kwargs,
                }
                cache_key = hashlib.sha256(
                    json.dumps(key_data, sort_keys=True, default=str).encode()
                ).hexdigest()

            # Check cache
            if cache_key in cache:
                value, expiry = cache[cache_key]
                if expiry is None or time.time() < expiry:
                    return value
                else:
                    # Expired, remove from cache
                    del cache[cache_key]

            # Call function and cache result
            result = func(*args, **kwargs)
            expiry = time.time() + ttl if ttl else None
            cache[cache_key] = (result, expiry)

            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}

        return wrapper

    return decorator


def async_cached(
    ttl: int | None = 3600,
    key_func: Callable[..., str] | None = None,
) -> Callable[[F], F]:
    """Async cache decorator for coroutine functions.

    Args:
        ttl: Time-to-live in seconds (None = no expiration)
        key_func: Custom function to generate cache key
    """

    def decorator(func: F) -> F:
        cache: dict[str, tuple[Any, float | None]] = {}

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "args": args,
                    "kwargs": kwargs,
                }
                cache_key = hashlib.sha256(
                    json.dumps(key_data, sort_keys=True, default=str).encode()
                ).hexdigest()

            # Check cache
            if cache_key in cache:
                value, expiry = cache[cache_key]
                if expiry is None or time.time() < expiry:
                    return value
                else:
                    # Expired, remove from cache
                    del cache[cache_key]

            # Call async function and cache result
            result = await func(*args, **kwargs)
            expiry = time.time() + ttl if ttl else None
            cache[cache_key] = (result, expiry)

            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}

        return wrapper

    return decorator


def traced(
    capture_args: bool = True,
    capture_result: bool = True,
    capture_time: bool = True,
) -> Callable[[F], F]:
    """Trace decorator to capture function calls.

    Args:
        capture_args: Whether to capture arguments
        capture_result: Whether to capture return value
        capture_time: Whether to capture execution time
    """

    def decorator(func: F) -> F:
        traces: list[dict[str, Any]] = []

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace: dict[str, Any] = {
                "function": getattr(func, "__name__", "unknown"),
                "timestamp": time.time(),
            }

            if capture_args:
                trace["args"] = args
                trace["kwargs"] = kwargs

            start_time = time.time() if capture_time else 0.0

            try:
                result = func(*args, **kwargs)

                if capture_result:
                    trace["result"] = result

                if capture_time and start_time:
                    trace["duration"] = time.time() - start_time

                trace["success"] = True

                return result

            except Exception as e:
                trace["error"] = str(e)
                trace["success"] = False

                if capture_time and start_time:
                    trace["duration"] = time.time() - start_time

                raise

            finally:
                traces.append(trace)

        # Add trace access methods
        wrapper.get_traces = lambda: traces.copy()
        wrapper.clear_traces = lambda: traces.clear()
        wrapper.trace_count = lambda: len(traces)

        return wrapper

    return decorator


def validated(
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    strict: bool = False,
) -> Callable[[F], F]:
    """Validation decorator for function inputs/outputs.

    Args:
        input_schema: JSON schema for input validation
        output_schema: JSON schema for output validation
        strict: Whether to raise on validation failure
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate inputs
            if input_schema:
                # Simple validation - could use jsonschema library
                if strict and not _validate_schema({"args": args, "kwargs": kwargs}, input_schema):
                    raise ValueError(
                        f"Input validation failed for {getattr(func, '__name__', 'unknown')}"
                    )

            # Call function
            result = func(*args, **kwargs)

            # Validate output
            if output_schema:
                if strict and not _validate_schema(result, output_schema):
                    raise ValueError(
                        f"Output validation failed for {getattr(func, '__name__', 'unknown')}"
                    )

            return result

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise

                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def timed(
    log_func: Callable[[str], None] | None = print,
) -> Callable[[F], F]:
    """Time execution decorator.

    Args:
        log_func: Function to call with timing message
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            if log_func:
                log_func(f"{getattr(func, '__name__', 'unknown')} took {duration:.3f} seconds")

            return result

        return wrapper

    return decorator


def singleton(cls: type) -> type:
    """Singleton decorator for classes."""
    instances = {}

    @functools.wraps(cls)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def deprecated(
    message: str = "This function is deprecated",
    version: str | None = None,
) -> Callable[[F], F]:
    """Mark function as deprecated.

    Args:
        message: Deprecation message
        version: Version when deprecated
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import warnings

            full_message = f"{message}"
            if version:
                full_message += f" (deprecated since v{version})"

            warnings.warn(full_message, category=DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def synchronized(lock: asyncio.Lock | None = None) -> Callable[[F], F]:
    """Synchronization decorator for async functions.

    Args:
        lock: Lock to use (creates new if None)
    """
    _lock = lock or asyncio.Lock()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with _lock:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limited(
    calls: int = 10,
    period: float = 1.0,
) -> Callable[[F], F]:
    """Rate limiting decorator.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """

    def decorator(func: F) -> F:
        call_times: list[float] = []

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.time()

            # Remove old calls outside the period
            call_times[:] = [t for t in call_times if now - t < period]

            # Check rate limit
            if len(call_times) >= calls:
                sleep_time = period - (now - call_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    call_times[:] = [t for t in call_times if now - t < period]

            # Record this call
            call_times.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Helper functions


def _validate_schema(data: Any, schema: dict[str, Any]) -> bool:
    """Simple schema validation (placeholder)."""
    # This is a simplified validation
    # In production, use jsonschema library
    return True


__all__ = [
    "cached",
    "async_cached",
    "traced",
    "validated",
    "retry",
    "async_retry",
    "timed",
    "singleton",
    "deprecated",
    "synchronized",
    "rate_limited",
]
