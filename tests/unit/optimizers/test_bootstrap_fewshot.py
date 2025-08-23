"""Tests for BootstrapFewShot optimizer."""

import pytest

from logillm.exceptions import OptimizationError
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot
from logillm.optimizers.demo_selectors import BestDemoSelector, RandomDemoSelector
from tests.unit.fixtures.mock_components import MockMetric, MockModule


class TestBootstrapFewShot:
    """Test suite for BootstrapFewShot optimizer."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset."""
        return [
            {"inputs": {"x": 1}, "outputs": {"y": 2}},
            {"inputs": {"x": 2}, "outputs": {"y": 4}},
            {"inputs": {"x": 3}, "outputs": {"y": 6}},
            {"inputs": {"x": 4}, "outputs": {"y": 8}},
            {"inputs": {"x": 5}, "outputs": {"y": 10}},
            {"inputs": {"x": 6}, "outputs": {"y": 12}},
        ]

    @pytest.fixture
    def validation_set(self):
        """Create a validation set."""
        return [
            {"inputs": {"x": 7}, "outputs": {"y": 14}},
            {"inputs": {"x": 8}, "outputs": {"y": 16}},
        ]

    @pytest.mark.asyncio
    async def test_basic_functionality(self, dataset, validation_set):
        """Test basic BootstrapFewShot functionality."""
        # Create module and optimizer
        module = MockModule(behavior="linear")
        metric = MockMetric(target_value=0.8)
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3, metric_threshold=0.5)

        # Optimize
        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Check that demonstrations were bootstrapped
        assert "demonstrations" in result.optimized_module.parameters
        demos = result.optimized_module.parameters["demonstrations"].value
        assert len(demos) > 0
        assert len(demos) <= 3

        # Check metadata
        assert "num_demos" in result.metadata
        assert "demo_scores" in result.metadata
        assert "num_bootstrapped" in result.metadata
        assert result.metadata["num_demos"] == len(demos)

    @pytest.mark.asyncio
    async def test_teacher_settings(self, dataset):
        """Test that teacher settings are applied."""
        module = MockModule(behavior="linear")
        metric = MockMetric()

        # Custom teacher settings
        teacher_settings = {"temperature": 1.5, "top_p": 0.95}
        optimizer = BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=2, teacher_settings=teacher_settings
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check that teacher settings were used
        assert result.metadata["teacher_settings"] == teacher_settings

    @pytest.mark.asyncio
    async def test_metric_threshold(self, dataset):
        """Test metric threshold filtering."""
        module = MockModule(behavior="quadratic")  # Variable scores

        # Strict metric that only passes high scores
        class StrictMetric:
            def __init__(self, threshold=0.8):
                self.threshold = threshold
                self.name = "strict_metric"

            def __call__(self, prediction, target):
                # Extract score from prediction
                if isinstance(prediction, dict):
                    score = prediction.get("score", 0.5)
                else:
                    score = 0.5
                return score

        metric = StrictMetric()
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=3,
            metric_threshold=0.7,  # High threshold
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # All demo scores should be above threshold
        demo_scores = result.metadata["demo_scores"]
        assert all(score >= 0.7 for score in demo_scores)

    @pytest.mark.asyncio
    async def test_multiple_rounds(self, dataset):
        """Test multiple bootstrapping rounds."""
        module = MockModule(behavior="random", seed=42)
        metric = MockMetric()
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_rounds=3,  # Multiple rounds
        )

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should have collected demos over multiple rounds
        assert result.iterations == 3
        assert result.metadata["num_demos"] > 0

    @pytest.mark.asyncio
    async def test_labeled_demo_fallback(self, dataset):
        """Test fallback to labeled demos when bootstrapping fails."""

        # Module that always fails
        class FailingModule(MockModule):
            async def forward(self, **inputs):
                raise RuntimeError("Always fails")

        module = FailingModule()
        metric = MockMetric()
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3, max_labeled_demos=3)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should have fallen back to labeled demos
        assert result.metadata["num_labeled"] > 0
        assert result.metadata["num_bootstrapped"] == 0

    @pytest.mark.asyncio
    async def test_demo_selection(self, dataset):
        """Test different demo selection strategies."""
        module = MockModule(behavior="linear")
        metric = MockMetric()

        # Test with BestDemoSelector
        optimizer_best = BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=2, demo_selector=BestDemoSelector()
        )

        result_best = await optimizer_best.optimize(module=module, dataset=dataset)

        # Test with RandomDemoSelector
        optimizer_random = BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=2, demo_selector=RandomDemoSelector(seed=42)
        )

        result_random = await optimizer_random.optimize(module=module, dataset=dataset)

        # Both should work
        assert result_best.metadata["num_demos"] == 2
        assert result_random.metadata["num_demos"] == 2

    @pytest.mark.asyncio
    async def test_custom_teacher(self, dataset):
        """Test with custom teacher module."""
        student = MockModule(behavior="linear")
        teacher = MockModule(behavior="quadratic")  # Different behavior
        metric = MockMetric()

        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)

        result = await optimizer.optimize(
            module=student,
            dataset=dataset,
            teacher=teacher,  # Provide custom teacher
        )

        # Should have demonstrations
        assert result.metadata["num_demos"] > 0

    @pytest.mark.asyncio
    async def test_empty_dataset(self):
        """Test handling of empty dataset."""
        module = MockModule()
        metric = MockMetric()
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)

        with pytest.raises(OptimizationError):
            # Should fail with empty dataset
            await optimizer.optimize(module=module, dataset=[])

    @pytest.mark.asyncio
    async def test_improvement_calculation(self, dataset, validation_set):
        """Test improvement calculation."""
        module = MockModule(behavior="linear")
        metric = MockMetric()
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)

        result = await optimizer.optimize(
            module=module, dataset=dataset, validation_set=validation_set
        )

        # Should calculate baseline and improvement
        assert "baseline_score" in result.metadata
        assert result.improvement == result.best_score - result.metadata["baseline_score"]

    @pytest.mark.asyncio
    async def test_bootstrapping_success_rate(self, dataset):
        """Test tracking of bootstrapping success rate."""
        module = MockModule(behavior="quadratic", seed=42)
        metric = MockMetric(target_value=0.7)
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3, metric_threshold=0.6)

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Check that we track attempts
        assert "total_attempts" in result.metadata
        assert result.metadata["total_attempts"] >= result.metadata["num_demos"]

        # Check average demo score
        assert "avg_demo_score" in result.metadata
        avg_score = result.metadata["avg_demo_score"]
        assert 0 <= avg_score <= 1

    @pytest.mark.asyncio
    async def test_wraparound_dataset(self):
        """Test that we wrap around dataset if needed."""
        # Small dataset
        small_dataset = [
            {"inputs": {"x": 1}, "outputs": {"y": 2}},
            {"inputs": {"x": 2}, "outputs": {"y": 4}},
        ]

        module = MockModule(behavior="random", seed=42)
        metric = MockMetric()
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=5,  # More than dataset size
            metric_threshold=0.3,  # Low threshold to get more demos
        )

        result = await optimizer.optimize(module=module, dataset=small_dataset)

        # Should have wrapped around and tried examples multiple times
        assert result.metadata["total_attempts"] > len(small_dataset)

    @pytest.mark.asyncio
    async def test_diversity_scoring_enabled(self, dataset):
        """Test that diversity scoring is enabled by default and affects selection."""
        module = MockModule(behavior="linear")
        metric = MockMetric()

        # Create optimizer with diversity scoring enabled (default)
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=3,
            metric_threshold=0.1,  # Low threshold to get many demos
        )

        # Ensure diversity scoring is enabled by default
        assert optimizer.config.use_diversity_scoring is True
        assert optimizer.config.diversity_weight == 0.3

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should have demonstrations
        assert result.metadata["num_demos"] > 0
        assert "diversity_scoring" in result.metadata["config"]
        assert result.metadata["config"]["diversity_scoring"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_diversity_scoring_disabled(self, dataset):
        """Test that diversity scoring can be disabled."""
        from logillm.core.types import OptimizationStrategy
        from logillm.optimizers.bootstrap_fewshot import BootstrapFewShotConfig

        # Create config with diversity scoring disabled
        config = BootstrapFewShotConfig(
            strategy=OptimizationStrategy.BOOTSTRAP,
            max_bootstrapped_demos=3,
            use_diversity_scoring=False,
        )

        module = MockModule(behavior="linear")
        metric = MockMetric()
        optimizer = BootstrapFewShot(metric=metric, config=config, metric_threshold=0.1)

        # Ensure diversity scoring is disabled
        assert optimizer.config.use_diversity_scoring is False

        result = await optimizer.optimize(module=module, dataset=dataset)

        # Should have demonstrations but without diversity scoring
        assert result.metadata["num_demos"] > 0
        assert result.metadata["config"]["diversity_scoring"]["enabled"] is False

    def test_text_similarity(self):
        """Test text similarity calculation."""
        MockModule()
        metric = MockMetric()
        optimizer = BootstrapFewShot(metric=metric)

        # Test identical texts
        assert optimizer._text_similarity("hello world", "hello world") == 1.0

        # Test completely different texts
        assert optimizer._text_similarity("hello world", "foo bar") == 0.0

        # Test partial overlap
        similarity = optimizer._text_similarity("hello world", "hello foo")
        assert 0.0 < similarity < 1.0

        # Test empty strings
        assert optimizer._text_similarity("", "hello") == 0.0
        assert optimizer._text_similarity("", "") == 0.0

    def test_diversity_score_calculation(self):
        """Test diversity score calculation logic."""
        from logillm.optimizers.base import Demonstration

        MockModule()
        metric = MockMetric()
        optimizer = BootstrapFewShot(metric=metric)

        # Create test demonstrations
        demo1 = Demonstration(
            inputs={"question": "What is 2+2?"}, outputs={"answer": "4"}, score=0.9
        )

        demo2 = Demonstration(
            inputs={"question": "What is 3+3?"},  # Similar question
            outputs={"answer": "6"},
            score=0.8,
        )

        demo3 = Demonstration(
            inputs={"question": "What is the capital of France?"},  # Different question
            outputs={"answer": "Paris"},
            score=0.7,
        )

        # First demo - should return its score (no selected demos)
        score1 = optimizer._calculate_diversity_score(demo1, [], alpha=0.5)
        assert score1 == 0.9

        # Second demo compared to first - should be lower due to similarity
        score2 = optimizer._calculate_diversity_score(demo2, [demo1], alpha=0.5)
        assert score2 < 0.8  # Should be penalized for similarity

        # Third demo compared to first - should be higher due to diversity
        score3 = optimizer._calculate_diversity_score(demo3, [demo1], alpha=0.5)
        assert score3 > score2  # Should be higher due to greater diversity

    def test_diverse_demo_selection(self):
        """Test that diverse demo selection actually selects diverse demos."""
        from logillm.optimizers.base import Demonstration

        MockModule()
        metric = MockMetric()

        # Create config with diversity scoring enabled
        from logillm.core.types import OptimizationStrategy
        from logillm.optimizers.bootstrap_fewshot import BootstrapFewShotConfig

        config = BootstrapFewShotConfig(
            strategy=OptimizationStrategy.BOOTSTRAP,
            use_diversity_scoring=True,
            diversity_weight=0.5,  # Balanced between accuracy and diversity
        )

        optimizer = BootstrapFewShot(metric=metric, config=config)

        # Create demonstrations with varying similarity and scores
        demos = [
            # High-scoring but similar math questions
            Demonstration(inputs={"question": "What is 2+2?"}, outputs={"answer": "4"}, score=0.95),
            Demonstration(inputs={"question": "What is 3+3?"}, outputs={"answer": "6"}, score=0.94),
            Demonstration(inputs={"question": "What is 4+4?"}, outputs={"answer": "8"}, score=0.93),
            # Lower-scoring but diverse questions
            Demonstration(
                inputs={"question": "What is the capital of France?"},
                outputs={"answer": "Paris"},
                score=0.85,
            ),
            Demonstration(
                inputs={"question": "Who wrote Romeo and Juliet?"},
                outputs={"answer": "Shakespeare"},
                score=0.80,
            ),
        ]

        # Select 3 demonstrations
        selected = optimizer._select_demonstrations(demos, 3)

        # Should have selected 3 demos
        assert len(selected) == 3

        # Should include some diverse demos, not just the top 3 math questions
        selected_questions = [d.inputs["question"] for d in selected]

        # Count how many are math questions vs other types
        math_questions = sum(
            1 for q in selected_questions if any(op in q for op in ["+", "-", "*", "/"])
        )
        other_questions = len(selected) - math_questions

        # With diversity scoring, we should have some non-math questions
        # (Without diversity, we'd get all 3 math questions)
        assert other_questions > 0, f"Expected some diverse questions, got: {selected_questions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
