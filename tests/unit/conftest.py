"""Unit test configuration and fixtures.

All fixtures here should use mock objects only.
"""

import pytest

from logillm.core.providers import MockProvider, register_provider
from tests.unit.fixtures.mock_components import MockDataset, MockMetric, MockModule


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MockProvider(response_text="Mock response for testing.")
    register_provider(provider, "test_provider", set_default=True)
    return provider


@pytest.fixture
def mock_module():
    """Create a mock module for testing."""
    return MockModule(behavior="linear", seed=42)


@pytest.fixture
def mock_metric():
    """Create a mock metric for testing."""
    return MockMetric(target_value=0.8)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return MockDataset(size=10, task_type="general")


@pytest.fixture
def sample_dataset():
    """Create a simple test dataset."""
    return [
        {"inputs": {"x": 1}, "outputs": {"y": 2}},
        {"inputs": {"x": 2}, "outputs": {"y": 4}},
        {"inputs": {"x": 3}, "outputs": {"y": 6}},
        {"inputs": {"x": 4}, "outputs": {"y": 8}},
    ]


@pytest.fixture
def validation_dataset():
    """Create a validation dataset."""
    return [
        {"inputs": {"x": 5}, "outputs": {"y": 10}},
        {"inputs": {"x": 6}, "outputs": {"y": 12}},
    ]


@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Reset provider registry before each test."""
    from logillm.providers.registry import clear_registry

    clear_registry()
    yield
    clear_registry()
