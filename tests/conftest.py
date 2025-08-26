"""Root conftest for all tests.

Provides shared configuration and fixtures for both unit and integration tests.
"""

import os
import sys
import logging
from pathlib import Path

import pytest

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Test configuration based on environment
TEST_ENV = os.getenv("TEST_ENV", "local")  # local, ci, full

# Configuration presets
TEST_CONFIGS = {
    "local": {
        "timeout": 30,
        "max_retries": 2,
        "skip_slow": True,
        "skip_flaky": False,
    },
    "ci": {
        "timeout": 60,
        "max_retries": 3,
        "skip_slow": False,
        "skip_flaky": False,
    },
    "full": {
        "timeout": 300,
        "max_retries": 5,
        "skip_slow": False,
        "skip_flaky": False,
    },
}

CURRENT_CONFIG = TEST_CONFIGS[TEST_ENV]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests with mocks")
    config.addinivalue_line("markers", "integration: Slow integration tests with real APIs")
    config.addinivalue_line("markers", "openai: Tests requiring OpenAI API key")
    config.addinivalue_line("markers", "anthropic: Tests requiring Anthropic API key")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Slow tests that should be run separately")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location and configuration."""
    skip_slow = pytest.mark.skip(reason=f"Skipping slow tests in {TEST_ENV} mode")
    skip_flaky = pytest.mark.skip(reason=f"Skipping flaky tests in {TEST_ENV} mode")
    
    for item in items:
        # Auto-mark tests based on their directory
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add timeout to integration tests if not already specified
        if "integration" in [m.name for m in item.iter_markers()]:
            if not any(m.name == "timeout" for m in item.iter_markers()):
                item.add_marker(pytest.mark.timeout(CURRENT_CONFIG["timeout"]))
        
        # Skip tests based on configuration
        if CURRENT_CONFIG["skip_slow"] and "slow" in [m.name for m in item.iter_markers()]:
            item.add_marker(skip_slow)
        
        if CURRENT_CONFIG["skip_flaky"] and "flaky" in [m.name for m in item.iter_markers()]:
            item.add_marker(skip_flaky)


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture(scope="session")
def api_keys_available():
    """Check if API keys are available."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@pytest.fixture
def skip_if_no_openai_key(api_keys_available):
    """Skip test if OpenAI API key is not available."""
    if not api_keys_available["openai"]:
        pytest.skip("OpenAI API key not available")


@pytest.fixture
def skip_if_no_anthropic_key(api_keys_available):
    """Skip test if Anthropic API key is not available."""
    if not api_keys_available["anthropic"]:
        pytest.skip("Anthropic API key not available")


@pytest.fixture(autouse=True)
def clean_global_state():
    """Automatically clean global state before and after each test.

    This fixture runs for every test to ensure proper isolation by:
    - Clearing the global assertion context
    - Clearing the provider registry
    - Yielding control to the test
    - Cleaning up again after the test
    """
    # Import here to avoid circular imports
    from logillm.core.assertions import get_global_context
    from logillm.providers.registry import clear_registry

    # Clean before test
    get_global_context().assertions.clear()
    get_global_context().results.clear()
    clear_registry()

    # Run the test
    yield

    # Clean after test
    get_global_context().assertions.clear()
    get_global_context().results.clear()
    clear_registry()


@pytest.fixture
def isolated_module():
    """Create a module and restore its original methods after test.

    Use this fixture when testing with assert_module_output to ensure
    the module's methods are restored after the test.

    Example:
        def test_something(isolated_module):
            module = isolated_module(MyModule())
            assert_module_output(module, assertions)
            # module.forward is now wrapped
            result = await module(...)
            # module.forward will be restored after test
    """
    modules_to_restore = []

    def track_module(module):
        """Track a module's original forward method."""
        if hasattr(module, "forward"):
            original_forward = module.forward
            modules_to_restore.append((module, original_forward))
        return module

    yield track_module

    # Restore all tracked modules
    for module, original_forward in modules_to_restore:
        module.forward = original_forward
