"""Test utilities for handling flaky LLM tests.

Provides retry decorators and utilities for dealing with non-deterministic LLM behavior.
"""

import asyncio
import functools
import logging
from typing import Callable, TypeVar

import pytest

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Retry a test function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(f"Test {func.__name__} failed on attempt {attempt}: {e}")
                        import time

                        time.sleep(delay)
                    else:
                        logger.error(f"Test {func.__name__} failed after {max_attempts} attempts")

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def async_retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Async version of retry_on_failure for async test functions.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(f"Test {func.__name__} failed on attempt {attempt}: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Test {func.__name__} failed after {max_attempts} attempts")

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def mark_flaky(test_func):
    """Mark a test as flaky and add retry logic.

    Automatically adds the flaky marker and retry decorator.
    """
    # Add pytest marker
    test_func = pytest.mark.flaky(test_func)

    # Add retry logic based on function type
    if asyncio.iscoroutinefunction(test_func):
        test_func = async_retry_on_failure(max_attempts=3, delay=2.0)(test_func)
    else:
        test_func = retry_on_failure(max_attempts=3, delay=2.0)(test_func)

    return test_func


def assert_llm_output_contains(
    output: str,
    expected_patterns: list[str],
    any_match: bool = False,
    case_sensitive: bool = False,
) -> None:
    """Assert that LLM output contains expected patterns.

    More flexible than exact matching for non-deterministic outputs.

    Args:
        output: The LLM output to check
        expected_patterns: List of patterns to look for
        any_match: If True, only one pattern needs to match. If False, all must match.
        case_sensitive: Whether to match case-sensitively
    """
    if not case_sensitive:
        output = output.lower()
        expected_patterns = [p.lower() for p in expected_patterns]

    matches = [pattern in output for pattern in expected_patterns]

    if any_match:
        assert any(matches), (
            f"Expected at least one of {expected_patterns} in output:\n{output[:200]}"
        )
    else:
        for pattern, matched in zip(expected_patterns, matches):
            assert matched, f"Expected '{pattern}' in output:\n{output[:200]}"


def assert_llm_output_quality(
    output: str,
    min_length: int | None = None,
    max_length: int | None = None,
    min_sentences: int | None = None,
    required_format: str | None = None,
) -> None:
    """Assert quality metrics for LLM output.

    Args:
        output: The LLM output to check
        min_length: Minimum character count
        max_length: Maximum character count
        min_sentences: Minimum number of sentences
        required_format: Expected format (json, list, etc.)
    """
    if min_length is not None:
        assert len(output) >= min_length, f"Output too short: {len(output)} < {min_length} chars"

    if max_length is not None:
        assert len(output) <= max_length, f"Output too long: {len(output)} > {max_length} chars"

    if min_sentences is not None:
        # Count sentences (rough heuristic)
        sentence_count = output.count(".") + output.count("!") + output.count("?")
        assert sentence_count >= min_sentences, (
            f"Too few sentences: {sentence_count} < {min_sentences}"
        )

    if required_format == "json":
        import json

        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Output is not valid JSON: {e}\n{output[:200]}")
    elif required_format == "list":
        assert any(marker in output for marker in ["-", "*", "•", "1.", "2."]), (
            f"Output doesn't appear to be a list:\n{output[:200]}"
        )
