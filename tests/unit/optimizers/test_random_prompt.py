"""Tests for RandomPromptOptimizer."""

import pytest

from logillm.optimizers.random_prompt import RandomPromptOptimizer
from tests.unit.fixtures.mock_components import MockMetric, MockModule


class TestRandomPromptOptimizer:
    """Test suite for RandomPromptOptimizer."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset."""
        return [
            {"inputs": {"text": "Hello"}, "outputs": {"response": "Hi"}},
            {"inputs": {"text": "How are you?"}, "outputs": {"response": "I'm fine"}},
            {"inputs": {"text": "What's 2+2?"}, "outputs": {"response": "4"}},
            {"inputs": {"text": "Goodbye"}, "outputs": {"response": "Bye"}},
        ]

    @pytest.fixture
    def validation_set(self):
        """Create a validation set."""
        return [
            {"inputs": {"text": "Good morning"}, "outputs": {"response": "Morning"}},
            {"inputs": {"text": "Thank you"}, "outputs": {"response": "You're welcome"}},
        ]

    @pytest.mark.asyncio
    async def test_basic_functionality(self, dataset, validation_set):
        """Test basic RandomPromptOptimizer functionality."""
        # Create module and optimizer
        module = MockModule(behavior="linear")
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(
            metric=metric,
            num_candidates=5,
            seed=42,  # For reproducibility
        )

        # Optimize
        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Check that instruction was optimized
        assert "instruction" in result.optimized_module.parameters
        instruction = result.optimized_module.parameters["instruction"].value
        assert isinstance(instruction, str)
        assert len(instruction) > 0

        # Check metadata
        assert "best_prompt" in result.metadata
        assert "all_scores" in result.metadata
        assert len(result.metadata["all_scores"]) == 5

    @pytest.mark.asyncio
    async def test_prompt_generation(self):
        """Test that different prompts are generated."""
        MockModule()
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(metric=metric, num_candidates=10, seed=42)

        # Generate multiple prompts
        prompts = set()
        for _ in range(10):
            prompt = optimizer._default_prompt_generator("analyze the data")
            prompts.add(prompt)

        # Should generate different prompts
        assert len(prompts) > 1  # At least some variation

        # All should contain the task
        for prompt in prompts:
            assert "analyze the data" in prompt.lower()

    @pytest.mark.asyncio
    async def test_with_demos(self, dataset):
        """Test optimization with demo selection."""
        module = MockModule()
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(
            metric=metric, num_candidates=5, include_demos=True, seed=42
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should have both instruction and demonstrations
        assert "instruction" in result.optimized_module.parameters
        assert "demonstrations" in result.optimized_module.parameters

        # Check metadata
        assert result.metadata["included_demos"] is True
        assert result.metadata["num_demo_variations"] > 0

    @pytest.mark.asyncio
    async def test_custom_generator(self, dataset):
        """Test with custom prompt generator."""

        def custom_generator(base: str) -> str:
            return f"CUSTOM: {base or 'do the task'}"

        module = MockModule()
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(
            metric=metric, num_candidates=3, prompt_generator=custom_generator
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should use custom generator
        assert result.metadata["best_prompt"].startswith("CUSTOM:")

    @pytest.mark.asyncio
    async def test_improvement_calculation(self, dataset, validation_set):
        """Test that improvement is calculated correctly."""
        module = MockModule(behavior="quadratic")
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(metric=metric, num_candidates=5, seed=42)

        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Should calculate baseline and improvement
        assert "baseline_score" in result.metadata
        assert result.improvement == result.best_score - result.metadata["baseline_score"]

    @pytest.mark.asyncio
    async def test_reproducibility(self, dataset):
        """Test that results are reproducible with same seed."""
        module = MockModule()
        metric = MockMetric()

        # Run twice with same seed
        optimizer1 = RandomPromptOptimizer(metric=metric, num_candidates=5, seed=123)
        result1 = await optimizer1.optimize(module=module, dataset=dataset)

        optimizer2 = RandomPromptOptimizer(metric=metric, num_candidates=5, seed=123)
        result2 = await optimizer2.optimize(module=module, dataset=dataset)

        # Should get same best prompt
        assert result1.metadata["best_prompt"] == result2.metadata["best_prompt"]

    @pytest.mark.asyncio
    async def test_empty_dataset(self):
        """Test handling of empty dataset."""
        module = MockModule()
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(metric=metric, num_candidates=3)

        # Should still work with empty dataset
        result = await optimizer.optimize(module=module, dataset=[])

        # Should have instruction parameter
        assert "instruction" in result.optimized_module.parameters

    @pytest.mark.asyncio
    async def test_selection_best(self, dataset):
        """Test that best scoring prompt is selected."""
        module = MockModule(behavior="linear")  # Higher temperature = better
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(metric=metric, num_candidates=10, seed=42)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Best score should be max of all scores
        all_scores = result.metadata["all_scores"]
        assert result.best_score == max(all_scores)

    @pytest.mark.asyncio
    async def test_demo_variations(self, dataset):
        """Test demo variation generation."""
        MockModule()
        metric = MockMetric()
        optimizer = RandomPromptOptimizer(
            metric=metric, num_candidates=4, include_demos=True, seed=42
        )

        # Test demo generation
        demo_variations = optimizer._generate_demo_variations(dataset, n_demos=2)

        # Should generate variations
        assert len(demo_variations) > 0

        # Each variation should have correct size
        for variation in demo_variations:
            assert len(variation) == 2

        # Variations should be different (with high probability)
        if len(demo_variations) > 1:
            # Convert to sets of tuples for comparison
            var_sets = [{tuple(d["inputs"].items()) for d in var} for var in demo_variations]
            # At least some should be different
            assert len({tuple(s) for s in var_sets}) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
