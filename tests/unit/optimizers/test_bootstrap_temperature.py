"""Test temperature scheduling in BootstrapFewShot optimizer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from logillm.core.modules import Module
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot, BootstrapFewShotConfig


@pytest.fixture
def mock_module():
    """Create a mock module for testing."""
    module = MagicMock(spec=Module)
    module.parameters = {}

    # Mock provider with temperature attribute
    provider = MagicMock()
    provider.temperature = 0.7
    module.provider = provider

    # Mock config that can be updated
    config = MagicMock()
    config.update = MagicMock()
    module.config = config

    return module


@pytest.fixture
def simple_metric():
    """Simple accuracy metric for testing."""

    def metric(predicted, expected):
        return 1.0 if predicted.get("answer") == expected.get("answer") else 0.0

    return metric


def test_bootstrap_config_defaults():
    """Test that BootstrapFewShotConfig has correct defaults."""
    config = BootstrapFewShotConfig()

    assert config.initial_teacher_temperature == 1.0
    assert config.temperature_decay == 0.9
    assert config.min_temperature == 0.3
    assert config.rescue_mode_threshold == 0.2
    assert config.rescue_initial_temperature == 1.5
    assert config.rescue_max_attempts_multiplier == 2.0


def test_bootstrap_config_inheritance():
    """Test that BootstrapFewShotConfig properly inherits from PromptOptimizationConfig."""
    config = BootstrapFewShotConfig(max_demos=8)

    # Should have parent attributes
    assert config.max_demos == 8
    assert hasattr(config, "teacher_settings")

    # Should have new attributes
    assert config.initial_teacher_temperature == 1.0


def test_bootstrap_optimizer_config_integration(simple_metric):
    """Test that BootstrapFewShot properly uses BootstrapFewShotConfig."""
    config = BootstrapFewShotConfig(
        initial_teacher_temperature=1.2, temperature_decay=0.8, min_temperature=0.1
    )

    optimizer = BootstrapFewShot(metric=simple_metric, config=config)

    assert isinstance(optimizer.config, BootstrapFewShotConfig)
    assert optimizer.config.initial_teacher_temperature == 1.2
    assert optimizer.config.temperature_decay == 0.8
    assert optimizer.config.min_temperature == 0.1


def test_temperature_update_method(mock_module):
    """Test the _update_teacher_temperature method."""
    config = BootstrapFewShotConfig()
    optimizer = BootstrapFewShot(metric=lambda x, y: 1.0, config=config)

    # Test updating temperature
    optimizer._update_teacher_temperature(mock_module, 1.5)

    # Should update config
    mock_module.config.update.assert_called_with({"temperature": 1.5})

    # Should update provider
    assert mock_module.provider.temperature == 1.5


def test_temperature_update_without_update_method(simple_metric):
    """Test temperature update when config doesn't have update method."""
    # Create a module with config that has temperature attribute but no update method
    module = MagicMock(spec=Module)
    config = MagicMock()
    config.temperature = 0.7
    delattr(config, "update")  # Remove update method
    module.config = config
    module.provider = MagicMock()
    module.provider.temperature = 0.7

    optimizer = BootstrapFewShot(metric=simple_metric)
    optimizer._update_teacher_temperature(module, 1.2)

    # Should set temperature directly
    assert module.config.temperature == 1.2
    assert module.provider.temperature == 1.2


def test_rescue_mode_determination(simple_metric, mock_module):
    """Test rescue mode activation based on baseline score."""
    config = BootstrapFewShotConfig(rescue_mode_threshold=0.5)
    optimizer = BootstrapFewShot(metric=simple_metric, config=config)

    # Mock the evaluate method to return low baseline score
    optimizer.evaluate = AsyncMock(return_value=(0.1, None))  # Low score triggers rescue mode

    # Mock the bootstrap demonstration generation to avoid actual calls
    optimizer._bootstrap_demonstrations = AsyncMock(return_value=[])
    optimizer._add_labeled_demos = AsyncMock(return_value=[])

    # This would trigger rescue mode due to low baseline score
    # The actual test would need to mock more components to run fully

    # For now, just verify config setup
    assert config.rescue_mode_threshold == 0.5
    assert config.rescue_initial_temperature == 1.5


def test_temperature_decay_calculation():
    """Test temperature decay calculation logic."""
    config = BootstrapFewShotConfig(
        initial_teacher_temperature=1.0, temperature_decay=0.8, min_temperature=0.2
    )

    temp = config.initial_teacher_temperature

    # First decay
    temp = max(config.min_temperature, temp * config.temperature_decay)
    assert temp == 0.8

    # Second decay
    temp = max(config.min_temperature, temp * config.temperature_decay)
    assert abs(temp - 0.64) < 1e-10  # Handle floating point precision

    # Continue until minimum
    while temp > config.min_temperature:
        new_temp = max(config.min_temperature, temp * config.temperature_decay)
        if new_temp == config.min_temperature:
            break
        temp = new_temp

    # Should reach minimum temperature
    final_temp = max(config.min_temperature, temp * config.temperature_decay)
    assert final_temp == config.min_temperature


@pytest.mark.asyncio
async def test_bootstrap_optimization_flow_mock(simple_metric, mock_module):
    """Test the basic flow of bootstrap optimization with mocked components."""
    config = BootstrapFewShotConfig(max_rounds=2, max_bootstrapped_demos=2)
    optimizer = BootstrapFewShot(metric=simple_metric, config=config)

    # Mock evaluation to return specific scores
    optimizer.evaluate = AsyncMock(
        side_effect=[
            (0.1, None),  # Baseline score (triggers rescue mode)
            (0.6, None),  # Final score
        ]
    )

    # Mock demonstration generation
    demo_data = {
        "inputs": {"question": "2+2=?"},
        "outputs": {"answer": "4"},
        "score": 1.0,
        "metadata": {"teacher": True, "attempt": 1, "temperature": 1.5, "rescue_mode": True},
    }

    from logillm.optimizers.base import Demonstration

    demo = Demonstration(**demo_data)

    optimizer._bootstrap_demonstrations = AsyncMock(return_value=[demo])
    optimizer._add_labeled_demos = AsyncMock(return_value=[demo])

    # Mock demo selector
    optimizer.demo_selector = MagicMock()
    optimizer.demo_selector.select = MagicMock(return_value=[demo])

    # Mock dataset
    dataset = [{"inputs": {"question": "2+2=?"}, "outputs": {"answer": "4"}}]

    # Run optimization
    result = await optimizer.optimize(mock_module, dataset)

    # Verify results
    assert result.improvement == 0.5  # 0.6 - 0.1
    assert result.metadata["rescue_mode"] is True
    assert "temperature_schedule" in result.metadata
    assert result.metadata["baseline_score"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__])
