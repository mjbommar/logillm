"""Integration tests for MIPROv2 optimizer.

Tests the full MIPROv2 optimization pipeline with real LLM calls.
"""

import pytest

from logillm.core.predict import Predict
from logillm.optimizers.miprov2 import MIPROv2Config, MIPROv2Optimizer
from logillm.providers import create_provider, register_provider


@pytest.mark.integration
@pytest.mark.openai
class TestMIPROv2Integration:
    """Integration tests for MIPROv2 optimizer."""

    @pytest.fixture(autouse=True)
    def setup(self, openai_provider_registered):
        """Setup for each test."""
        self.results = {}

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # Increased timeout for optimization
    async def test_miprov2_basic_optimization(self):
        """Test basic MIPROv2 optimization with real LLM."""
        # Simple classification task
        module = Predict("text -> sentiment")

        training_data = [
            {"inputs": {"text": "I love this product!"}, "outputs": {"sentiment": "positive"}},
            {"inputs": {"text": "This is terrible."}, "outputs": {"sentiment": "negative"}},
            {"inputs": {"text": "Amazing quality!"}, "outputs": {"sentiment": "positive"}},
            {"inputs": {"text": "Waste of money."}, "outputs": {"sentiment": "negative"}},
        ]

        def sentiment_metric(prediction, expected):
            """Check if sentiment matches."""
            pred_sentiment = prediction.get("sentiment", "").lower()
            exp_sentiment = expected.get("sentiment", "").lower()
            return 1.0 if pred_sentiment == exp_sentiment else 0.0

        # Ultra-light mode for faster test
        config = MIPROv2Config(
            mode="light",
            num_candidates=1,  # Reduced from 3
            num_trials=2,      # Reduced from 3
            max_bootstrapped_demos=1,  # Reduced from 2
            max_labeled_demos=1,       # Reduced from 2
        )

        optimizer = MIPROv2Optimizer(metric=sentiment_metric, config=config)

        # Run optimization
        result = await optimizer.optimize(module, training_data)

        # Store results
        self.results = {
            "best_score": result.best_score,
            "iterations": result.iterations,
            "optimization_time": result.optimization_time,
        }

        # Test the optimized module
        test_cases = [
            {"text": "Excellent service!", "expected": "positive"},
            {"text": "Very disappointing.", "expected": "negative"},
        ]

        correct = 0
        for test in test_cases:
            pred = await result.optimized_module.forward(text=test["text"])
            if pred.outputs.get("sentiment", "").lower() == test["expected"]:
                correct += 1

        test_accuracy = correct / len(test_cases)
        self.results["test_accuracy"] = test_accuracy

        # Assertions
        assert result.best_score >= 0, "Score should be non-negative"
        assert result.iterations > 0, "Should have completed iterations"
        assert result.optimization_time > 0, "Should have taken time"

        # Log results
        print("\nMIPROv2 Optimization Results:")
        print(f"  Best score: {result.best_score:.2%}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Time: {result.optimization_time:.1f}s")
        print(f"  Test accuracy: {test_accuracy:.2%}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # Increased timeout for optimization
    async def test_miprov2_with_instruction_proposals(self):
        """Test MIPROv2 with instruction proposal generation."""
        # Math reasoning task
        module = Predict("problem: str -> solution: str, answer: float")

        training_data = [
            {
                "inputs": {"problem": "If apples cost $2 each and I buy 3, how much do I pay?"},
                "outputs": {"solution": "3 apples × $2 = $6", "answer": 6.0},
            },
            {
                "inputs": {
                    "problem": "A store offers 20% off on a $50 item. What's the final price?"
                },
                "outputs": {"solution": "20% of $50 = $10, so $50 - $10 = $40", "answer": 40.0},
            },
        ]

        def math_metric(prediction, expected):
            """Check if answer is close."""
            try:
                pred_answer = float(prediction.get("answer", 0))
                exp_answer = float(expected.get("answer", 0))
                return 1.0 if abs(pred_answer - exp_answer) < 0.5 else 0.0
            except (ValueError, TypeError):
                return 0.0

        # Medium mode with instruction proposals
        config = MIPROv2Config(
            mode="medium",
            num_instructions=3,  # Generate 3 instruction variants
            max_bootstrapped_demos=1,
            num_trials=2,  # Quick test
        )

        optimizer = MIPROv2Optimizer(metric=math_metric, config=config)

        # Run optimization
        result = await optimizer.optimize(module, training_data)

        # Test the optimized module
        test_problem = "If I have $100 and spend 30%, how much is left?"
        test_result = await result.optimized_module.forward(problem=test_problem)

        print("\nMIPROv2 Instruction Optimization Results:")
        print(f"  Best score: {result.best_score:.2%}")
        print(f"  Test problem: {test_problem}")
        print(f"  Solution: {test_result.outputs.get('solution', 'No solution')}")
        print(f"  Answer: {test_result.outputs.get('answer', 'No answer')}")

        assert result.best_score >= 0, "Score should be non-negative"
        assert result.iterations > 0, "Should have completed iterations"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_miprov2_with_mock_provider(self):
        """Test MIPROv2 runs correctly with mock provider (smoke test)."""
        # Setup mock provider
        provider = create_provider("mock")
        register_provider(provider, set_default=True)

        module = Predict("input -> output")
        training_data = [
            {"inputs": {"input": "test1"}, "outputs": {"output": "result1"}},
            {"inputs": {"input": "test2"}, "outputs": {"output": "result2"}},
        ]

        def simple_metric(pred, expected):
            return 1.0 if pred.get("output") == expected.get("output") else 0.0

        config = MIPROv2Config(
            mode="light",
            num_candidates=2,
            num_trials=1,
            max_bootstrapped_demos=1,
        )

        optimizer = MIPROv2Optimizer(metric=simple_metric, config=config)
        result = await optimizer.optimize(module, training_data)

        # Should complete without errors
        assert result is not None
        assert hasattr(result, "optimized_module")
        assert hasattr(result, "best_score")

        print("\nMIPROv2 Mock Test:")
        print("  Completed successfully: ✓")
        print(f"  Score: {result.best_score:.2%}")
        print(f"  Time: {result.optimization_time:.2f}s")
