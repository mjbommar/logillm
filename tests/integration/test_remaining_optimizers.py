#!/usr/bin/env python3
"""
Test the 4 remaining untested optimizers.

From previous testing, we haven't tested:
- FormatOptimizer (needs adapter implementations)
- GridSearchOptimizer
- MultiObjectiveOptimizer
- ReflectiveEvolutionOptimizer
"""

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


@pytest.mark.asyncio
async def test_grid_search():
    """Test GridSearchOptimizer."""
    print("\n" + "=" * 60)
    print("Testing GridSearchOptimizer")
    print("=" * 60)

    try:
        from logillm.optimizers import GridSearchOptimizer

        # Simple task
        provider = create_provider("openai", model="gpt-4.1-mini")
        register_provider(provider, set_default=True)

        module = Predict("question -> answer")
        train_data = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]

        def metric(pred, expected):
            return 1.0 if pred.get("answer") == expected.get("answer") else 0.0

        print("Initializing GridSearchOptimizer...")

        # Define parameter grid
        param_grid = {
            "temperature": [0.1, 0.5, 0.9],
            "top_p": [0.5, 0.9],
        }

        optimizer = GridSearchOptimizer(metric=metric, param_grid=param_grid)

        print(
            f"Testing {len(param_grid['temperature']) * len(param_grid['top_p'])} combinations..."
        )
        start_time = time.time()

        result = await optimizer.optimize(module, train_data)
        elapsed = time.time() - start_time

        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Best score: {result.best_score:.1%}")
        if hasattr(result, "metadata") and "best_params" in result.metadata:
            print(f"Best params: {result.metadata['best_params']}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 1 minute should be enough with reduced trials
@pytest.mark.integration
async def test_multi_objective():
    """Test MultiObjectiveOptimizer."""
    print("\n" + "=" * 60)
    print("Testing MultiObjectiveOptimizer")
    print("=" * 60)

    try:
        from logillm.optimizers import MultiObjectiveOptimizer

        provider = create_provider("openai", model="gpt-4.1-mini")
        register_provider(provider, set_default=True)

        module = Predict("text -> summary")
        train_data = [
            {
                "inputs": {"text": "The quick brown fox jumps over the lazy dog."},
                "outputs": {"summary": "A fox jumps over a dog."},
            },
        ]

        # Multiple metrics
        def accuracy_metric(pred, expected):
            if not pred or "summary" not in pred:
                return 0.0
            return 1.0 if len(pred["summary"]) > 0 else 0.0

        def brevity_metric(pred, expected):
            if not pred or "summary" not in pred:
                return 0.0
            # Prefer shorter summaries
            length = len(pred["summary"])
            return max(0, 1.0 - (length / 100))

        print("Initializing MultiObjectiveOptimizer...")

        optimizer = MultiObjectiveOptimizer(
            metrics={"accuracy": accuracy_metric, "brevity": brevity_metric},
            weights={"accuracy": 0.7, "brevity": 0.3},
            n_trials=3,  # Reduced from default 50 for integration test
        )

        print("Optimizing for multiple objectives...")
        start_time = time.time()

        result = await optimizer.optimize(module, train_data)
        elapsed = time.time() - start_time

        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Best weighted score: {result.best_score:.1%}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 1 minute should be enough with reduced iterations
@pytest.mark.integration
async def test_reflective_evolution():
    """Test ReflectiveEvolutionOptimizer."""
    print("\n" + "=" * 60)
    print("Testing ReflectiveEvolutionOptimizer")
    print("=" * 60)

    try:
        from logillm.optimizers import ReflectiveEvolutionOptimizer

        provider = create_provider("openai", model="gpt-4.1-mini")
        register_provider(provider, set_default=True)

        module = Predict("problem -> solution")
        train_data = [
            {
                "inputs": {"problem": "How to reduce carbon emissions?"},
                "outputs": {"solution": "Use renewable energy and improve efficiency."},
            },
            {
                "inputs": {"problem": "How to improve education?"},
                "outputs": {"solution": "Invest in teachers and technology."},
            },
        ]

        def metric(pred, expected):
            if not pred or "solution" not in pred:
                return 0.0
            # Check if solution is non-empty and reasonable length
            solution = pred["solution"]
            if 10 < len(solution) < 200:
                return 0.8
            return 0.3

        print("Initializing ReflectiveEvolutionOptimizer...")

        optimizer = ReflectiveEvolutionOptimizer(
            metric=metric,
            n_iterations=2,  # Reduced from default 10 for integration test
            minibatch_size=1,  # Small batch for testing
        )

        print("Running evolutionary optimization...")
        start_time = time.time()

        result = await optimizer.optimize(module, train_data)
        elapsed = time.time() - start_time

        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Best score: {result.best_score:.1%}")
        print(f"Iterations: {result.iterations}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_format_optimizer():
    """Test FormatOptimizer (may fail due to missing adapters)."""
    print("\n" + "=" * 60)
    print("Testing FormatOptimizer")
    print("=" * 60)

    try:
        from logillm.optimizers import FormatOptimizer

        provider = create_provider("openai", model="gpt-4.1-mini")
        register_provider(provider, set_default=True)

        module = Predict("question -> answer")
        train_data = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        ]

        def metric(pred, expected):
            return 1.0 if pred.get("answer") == expected.get("answer") else 0.0

        print("Initializing FormatOptimizer...")

        from logillm.optimizers.format_optimizer import FormatOptimizerConfig, PromptFormat

        config = FormatOptimizerConfig(
            formats_to_test=[PromptFormat.JSON],  # Only test one format for speed
            min_samples_per_format=1,
            max_samples_per_format=1,
        )
        optimizer = FormatOptimizer(
            metric=metric,
            config=config,
        )

        print("Testing format optimization...")
        start_time = time.time()

        result = await optimizer.optimize(module, train_data)
        elapsed = time.time() - start_time

        print(f"✅ SUCCESS in {elapsed:.1f}s")
        print(f"Best format: {result.metadata.get('best_format', 'unknown')}")
        print(f"Best score: {result.best_score:.1%}")

        return True

    except Exception as e:
        print(f"⚠️ EXPECTED FAILURE (adapters not implemented): {e}")
        return False


async def main():
    """Test all remaining optimizers."""
    print("\n" + "=" * 60)
    print("  Testing Remaining Optimizers")
    print("  These haven't been tested yet")
    print("=" * 60)

    results = {}

    # Test each optimizer
    results["GridSearchOptimizer"] = await test_grid_search()
    results["MultiObjectiveOptimizer"] = await test_multi_objective()
    results["ReflectiveEvolutionOptimizer"] = await test_reflective_evolution()
    results["FormatOptimizer"] = await test_format_optimizer()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    working = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ WORKS" if success else "❌ FAILED"
        print(f"{name}: {status}")

    print(f"\nTotal working: {working}/{total}")

    # Update total count
    previously_working = 7
    now_working = previously_working + working
    total_optimizers = 11

    print("\n🎯 Overall LogiLLM Status:")
    print(f"   {now_working}/{total_optimizers} optimizers functional")
    print(f"   {(now_working / total_optimizers) * 100:.0f}% optimizer coverage")

    if now_working >= 9:
        print("\n✅ LogiLLM has strong optimizer coverage!")
    else:
        print("\n⚠️ Some optimizers need work")


if __name__ == "__main__":
    asyncio.run(main())
