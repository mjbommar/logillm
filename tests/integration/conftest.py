"""Integration test configuration and fixtures.

All fixtures here should use real API calls only.
Includes timeout handling for slow tests and API key validation.
"""

import pytest

from logillm.providers import create_provider, register_provider

# Timeout handling is now in root conftest.py


@pytest.fixture
def openai_provider():
    """Create OpenAI provider for integration tests."""
    return create_provider("openai", model="gpt-4.1")


@pytest.fixture
def openai_provider_registered(openai_provider):
    """Register OpenAI provider as default for tests."""
    register_provider(openai_provider, set_default=True)
    return openai_provider


@pytest.fixture
def simple_qa_dataset():
    """Simple Q&A dataset for testing."""
    return [
        {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        {"inputs": {"question": "What color is the sky?"}, "outputs": {"answer": "blue"}},
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
    ]


@pytest.fixture
def math_dataset():
    """Math dataset for testing."""
    return [
        {"inputs": {"x": 2, "y": 3}, "outputs": {"result": 5}},
        {"inputs": {"x": 5, "y": 7}, "outputs": {"result": 12}},
        {"inputs": {"x": 10, "y": 15}, "outputs": {"result": 25}},
    ]


@pytest.fixture(autouse=True)
def check_api_keys(skip_if_no_openai_key):
    """Check that required API keys are available."""
    # Delegates to root conftest fixture
    pass


@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Reset provider registry before each test."""
    from logillm.providers.registry import clear_registry

    clear_registry()
    yield
    clear_registry()


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Include slow tests (tests that take more than 30 seconds)",
    )


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip slow tests unless --slow is passed
    if "slow" in item.keywords and not item.config.getoption("--slow"):
        pytest.skip("Slow test - use --slow flag to run")
