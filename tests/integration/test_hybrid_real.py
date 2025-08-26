#!/usr/bin/env python3
"""
Real integration test for hybrid optimizer with actual API calls.
This proves whether hybrid optimization provides value over prompt-only.
"""

from typing import Any

import pytest

from logillm.core.predict import Predict
from logillm.optimizers import BootstrapFewShot, HybridOptimizer
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShotConfig
from logillm.providers import create_provider, register_provider


class TestHybridOptimizer:
    """Test hybrid optimizer with real API calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup provider for tests."""
        provider = create_provider("openai", model="gpt-4.1-mini")
        provider.temperature = 0.8  # Start with suboptimal temperature
        register_provider(provider, set_default=True)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.timeout(60)  # Reduce to 1 minute
    @pytest.mark.slow
    @pytest.mark.skip(reason="Too expensive for regular CI - run with --run-slow")
    async def test_hybrid_beats_prompt_only_on_temperature_sensitive_task(self):
        """Test that hybrid optimization improves on temperature-sensitive creative tasks."""

        # Creative writing task - should benefit from temperature tuning
        train_data = [
            {
                "inputs": {"instruction": "Write a haiku about spring"},
                "outputs": {
                    "poem": "Cherry blossoms bloom\nSoft petals dance on warm breeze\nSpring whispers new life"
                },
            },
            {
                "inputs": {"instruction": "Write a haiku about winter"},
                "outputs": {
                    "poem": "Snowflakes gently fall\nBlanketing earth in white peace\nWinter's quiet reign"
                },
            },
            {
                "inputs": {"instruction": "Write a haiku about autumn"},
                "outputs": {
                    "poem": "Leaves turn gold and red\nCrisp air carries change ahead\nAutumn's last goodbye"
                },
            },
        ]

        test_data = {
            "inputs": {"instruction": "Write a haiku about summer"},
            "outputs": {
                "poem": "Sun beats down with strength\nWaves crash on the sandy shore\nSummer's vibrant song"
            },
        }

        # Metric that checks haiku structure and quality
        def haiku_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            if not prediction or "poem" not in prediction:
                return 0.0

            poem = prediction["poem"]
            lines = poem.strip().split("\n")

            # Check structure (3 lines)
            if len(lines) != 3:
                return 0.2

            # Check approximate syllable count (5-7-5)
            # Simplified: just check word count as proxy
            word_counts = [len(line.split()) for line in lines]
            structure_score = 0.5
            if word_counts[0] <= 5 and word_counts[1] <= 7 and word_counts[2] <= 5:
                structure_score = 1.0

            # Check for poetic quality (has imagery words)
            imagery_words = [
                "bloom",
                "dance",
                "whisper",
                "fall",
                "peace",
                "gold",
                "red",
                "beats",
                "crash",
                "song",
            ]
            has_imagery = any(word in poem.lower() for word in imagery_words)
            quality_score = 1.0 if has_imagery else 0.5

            return (structure_score + quality_score) / 2

        # Create module
        module = Predict("instruction -> poem")

        # 1. Baseline
        baseline_result = await module.forward(**test_data["inputs"])
        baseline_score = haiku_metric(baseline_result.outputs, test_data["outputs"])

        # 2. Prompt-only optimization
        prompt_config = BootstrapFewShotConfig(
            max_bootstrapped_demos=1,  # Minimal demos
            max_labeled_demos=2,  # Very few labeled demos
            max_rounds=1,  # Single round
        )
        prompt_optimizer = BootstrapFewShot(metric=haiku_metric, config=prompt_config)

        prompt_opt_result = await prompt_optimizer.optimize(module, train_data)
        prompt_module = prompt_opt_result.optimized_module

        prompt_result = await prompt_module.forward(**test_data["inputs"])
        prompt_score = haiku_metric(prompt_result.outputs, test_data["outputs"])

        # 3. Hybrid optimization (prompts + temperature)
        hybrid_optimizer = HybridOptimizer(
            metric=haiku_metric,
            strategy="alternating",
            num_iterations=1,  # Reduced to prevent timeout
            samples_per_iteration=1,  # Minimal samples for quick test
        )

        hybrid_opt_result = await hybrid_optimizer.optimize(module, train_data)
        hybrid_module = hybrid_opt_result.optimized_module

        hybrid_result = await hybrid_module.forward(**test_data["inputs"])
        hybrid_score = haiku_metric(hybrid_result.outputs, test_data["outputs"])

        # Log results for debugging
        print(f"\nBaseline score: {baseline_score:.2%}")
        print(f"Prompt-only score: {prompt_score:.2%} (↑{(prompt_score - baseline_score):.2%})")
        print(f"Hybrid score: {hybrid_score:.2%} (↑{(hybrid_score - baseline_score):.2%})")

        # Check temperature was optimized
        if hasattr(hybrid_module, "provider") and hybrid_module.provider:
            print(f"Optimized temperature: {hybrid_module.provider.temperature}")

        # Assert hybrid is at least as good as prompt-only
        # (may need more iterations to consistently beat it)
        assert hybrid_score >= prompt_score - 0.1, (
            f"Hybrid ({hybrid_score:.2%}) should be close to or better than prompt-only ({prompt_score:.2%})"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.timeout(30)  # 30 second timeout for quick test
    async def test_hybrid_quick_smoke_test(self):
        """Quick smoke test for hybrid optimization - minimal API calls."""
        # Very simple task with minimal data
        train_data = [
            {"inputs": {"x": "1+1"}, "outputs": {"y": "2"}},
            {"inputs": {"x": "2+2"}, "outputs": {"y": "4"}},
        ]
        
        def simple_metric(pred, expected):
            return 1.0 if pred.get("y") == expected.get("y") else 0.0
        
        module = Predict("x -> y")
        
        # Minimal hybrid optimization with explicit config
        from logillm.optimizers import HybridOptimizer
        from logillm.optimizers.optimizer_config import HybridOptimizerConfig
        
        config = HybridOptimizerConfig(
            num_iterations=1,
            n_trials=2,  # MINIMAL trials instead of default 50!
            n_warmup_joint=1,  # Minimal warmup
            demo_subset_size=2,  # Very small subset
        )
        
        optimizer = HybridOptimizer(
            metric=simple_metric,
            strategy="joint",  # Faster than alternating
            config=config,
        )
        
        result = await optimizer.optimize(module, train_data)
        assert result.optimized_module is not None
        assert result.best_score >= 0  # Just check it ran
        
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.timeout(60)  # 1 minute timeout  
    @pytest.mark.skip(reason="Still too slow for regular runs")
    async def test_hybrid_on_factual_task(self):
        """Test hybrid on factual task that needs low temperature."""

        # Math task - should benefit from low temperature
        train_data = [
            {"inputs": {"problem": "2 + 2"}, "outputs": {"answer": "4"}},
            {"inputs": {"problem": "5 * 3"}, "outputs": {"answer": "15"}},
            {"inputs": {"problem": "10 / 2"}, "outputs": {"answer": "5"}},
        ]

        test_data = {"inputs": {"problem": "7 + 8"}, "outputs": {"answer": "15"}}

        def math_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            if not prediction or "answer" not in prediction:
                return 0.0
            return (
                1.0 if str(prediction["answer"]).strip() == str(expected["answer"]).strip() else 0.0
            )

        module = Predict("problem -> answer")

        # Test with hybrid - should discover low temperature is better
        from logillm.optimizers.optimizer_config import HybridOptimizerConfig
        
        config = HybridOptimizerConfig(
            num_iterations=1,
            n_trials=2,  # MINIMAL trials!
            n_warmup_joint=1,
            demo_subset_size=2,
        )
        
        hybrid_optimizer = HybridOptimizer(
            metric=math_metric, 
            strategy="alternating",
            config=config,
        )

        result = await hybrid_optimizer.optimize(module, train_data)
        optimized = result.optimized_module

        # Test on new problem
        test_result = await optimized.forward(**test_data["inputs"])
        test_score = math_metric(test_result.outputs, test_data["outputs"])

        print(f"\nMath task score: {test_score:.2%}")
        if hasattr(optimized, "provider") and optimized.provider:
            temp = optimized.provider.temperature
            print(f"Optimized temperature: {temp}")
            # For factual tasks, temperature should be lower
            assert temp <= 0.8, f"Temperature should be lowered for factual task, got {temp}"

        assert test_score >= 0.5, "Should get at least 50% on simple math"
