"""Tests for LabeledFewShot optimizer."""

import pytest

from logillm.optimizers.demo_selectors import BestDemoSelector, RandomDemoSelector
from logillm.optimizers.labeled_fewshot import LabeledFewShot
from tests.unit.fixtures.mock_components import MockMetric, MockModule


class TestLabeledFewShot:
    """Test suite for LabeledFewShot optimizer."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset."""
        return [
            {"inputs": {"x": 1}, "outputs": {"y": 2}},
            {"inputs": {"x": 2}, "outputs": {"y": 4}},
            {"inputs": {"x": 3}, "outputs": {"y": 6}},
            {"inputs": {"x": 4}, "outputs": {"y": 8}},
            {"inputs": {"x": 5}, "outputs": {"y": 10}},
        ]

    @pytest.fixture
    def validation_set(self):
        """Create a validation set."""
        return [
            {"inputs": {"x": 6}, "outputs": {"y": 12}},
            {"inputs": {"x": 7}, "outputs": {"y": 14}},
        ]

    @pytest.mark.asyncio
    async def test_basic_functionality(self, dataset, validation_set):
        """Test basic LabeledFewShot functionality."""
        # Create module and optimizer
        module = MockModule(behavior="linear")
        metric = MockMetric()
        optimizer = LabeledFewShot(metric=metric, max_demos=3)

        # Optimize
        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Check that demonstrations were added
        assert "demonstrations" in result.optimized_module.parameters
        demos = result.optimized_module.parameters["demonstrations"].value
        assert len(demos) == 3

        # Check that each demo has the right structure
        for demo in demos:
            assert "inputs" in demo
            assert "outputs" in demo
            assert "score" in demo

    @pytest.mark.asyncio
    async def test_demo_selection(self, dataset):
        """Test that demo selection works correctly."""
        module = MockModule()
        metric = MockMetric()

        # Test with BestDemoSelector
        optimizer = LabeledFewShot(metric=metric, max_demos=2, demo_selector=BestDemoSelector())

        result = await optimizer.optimize(module=module, dataset=dataset)
        demos = result.optimized_module.parameters["demonstrations"].value
        assert len(demos) == 2

        # Test with RandomDemoSelector
        optimizer = LabeledFewShot(
            metric=metric, max_demos=2, demo_selector=RandomDemoSelector(seed=42)
        )

        result = await optimizer.optimize(module=module, dataset=dataset)
        demos = result.optimized_module.parameters["demonstrations"].value
        assert len(demos) == 2

    @pytest.mark.asyncio
    async def test_no_learning(self, dataset):
        """Test that LabeledFewShot doesn't actually learn/optimize."""
        module = MockModule()
        metric = MockMetric()
        optimizer = LabeledFewShot(metric=metric, max_demos=3)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should only have 1 iteration (no actual optimization)
        assert result.iterations == 1

        # Demonstrations should all have score 1.0 (labeled data)
        demos = result.optimized_module.parameters["demonstrations"].value
        for demo in demos:
            assert demo["score"] == 1.0

    @pytest.mark.asyncio
    async def test_metadata(self, dataset):
        """Test that metadata is correctly set."""
        module = MockModule()
        metric = MockMetric()
        optimizer = LabeledFewShot(metric=metric, max_demos=2)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check parameter metadata
        demo_param = result.optimized_module.parameters["demonstrations"]
        assert demo_param.metadata["type"] == "demonstrations"
        assert demo_param.metadata["source"] == "labeled"

        # Check result metadata
        assert "num_demos" in result.metadata
        assert result.metadata["num_demos"] == 2
        assert "demo_scores" in result.metadata
        assert len(result.metadata["demo_scores"]) == 2

    @pytest.mark.asyncio
    async def test_empty_dataset(self):
        """Test handling of empty dataset."""
        module = MockModule()
        metric = MockMetric()
        optimizer = LabeledFewShot(metric=metric, max_demos=3)

        result = await optimizer.optimize(module=module, dataset=[])

        # Should still work but with no demos
        demos = result.optimized_module.parameters["demonstrations"].value
        assert len(demos) == 0

    @pytest.mark.asyncio
    async def test_small_dataset(self, dataset):
        """Test when dataset is smaller than max_demos."""
        module = MockModule()
        metric = MockMetric()
        optimizer = LabeledFewShot(metric=metric, max_demos=10)  # More than dataset size

        small_dataset = dataset[:2]
        result = await optimizer.optimize(module=module, dataset=small_dataset)

        # Should use all available data
        demos = result.optimized_module.parameters["demonstrations"].value
        assert len(demos) == 2

    @pytest.mark.asyncio
    async def test_improvement_calculation(self, dataset, validation_set):
        """Test that improvement is calculated correctly."""
        module = MockModule(behavior="quadratic")
        metric = MockMetric()
        optimizer = LabeledFewShot(metric=metric, max_demos=3)

        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Improvement should be calculated
        assert "baseline_score" in result.metadata
        assert result.improvement == result.best_score - result.metadata["baseline_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
