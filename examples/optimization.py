#!/usr/bin/env python3
"""LogiLLM Optimization Example.

This example demonstrates LogiLLM's hybrid optimization capabilities:
1. Setting up training data for optimization
2. Optimizing both prompts and hyperparameters simultaneously
3. Comparing performance before and after optimization
4. Using the optimized module in production

LogiLLM can automatically find the best prompt structure, examples, and
model parameters (temperature, top_p) to maximize performance on your task.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.optimizers import AccuracyMetric
from logillm.core.predict import Predict
from logillm.optimizers import HybridOptimizer
from logillm.providers import create_provider, register_provider


async def main():
    """Demonstrate hybrid optimization."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== LogiLLM Hybrid Optimization ===")

    try:
        # Step 1: Set up provider (required for optimization)
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Step 2: Create a module to optimize
        # We'll optimize a customer support ticket classifier
        classifier = Predict("ticket -> category: str, urgency: str", provider=provider)

        print("📊 Setting up training data...")

        # Step 3: Prepare training data
        # The optimizer needs examples to learn from
        training_data = [
            {
                "inputs": {"ticket": "My account was charged twice for the same order"},
                "outputs": {"category": "billing", "urgency": "high"},
            },
            {
                "inputs": {"ticket": "How do I reset my password?"},
                "outputs": {"category": "account", "urgency": "low"},
            },
            {
                "inputs": {"ticket": "The app crashes when I try to upload photos"},
                "outputs": {"category": "technical", "urgency": "high"},
            },
            {
                "inputs": {"ticket": "Can you add a dark mode to the app?"},
                "outputs": {"category": "feature", "urgency": "low"},
            },
            {
                "inputs": {"ticket": "I can't access my account after the update"},
                "outputs": {"category": "technical", "urgency": "high"},
            },
            {
                "inputs": {"ticket": "Where can I find my purchase history?"},
                "outputs": {"category": "account", "urgency": "low"},
            },
            {
                "inputs": {"ticket": "I was charged but didn't receive my order"},
                "outputs": {"category": "billing", "urgency": "high"},
            },
            {
                "inputs": {"ticket": "The new feature is really great!"},
                "outputs": {"category": "feedback", "urgency": "low"},
            },
        ]

        # Step 4: Test baseline performance
        print("\n🧪 Testing baseline performance...")

        test_examples = training_data[:3]  # Use first 3 for quick baseline test
        correct = 0
        total = len(test_examples)

        for example in test_examples:
            result = await classifier(**example["inputs"])
            predicted_category = result.outputs.get("category")
            expected_category = example["outputs"]["category"]

            if predicted_category == expected_category:
                correct += 1

            print(f"  Ticket: '{example['inputs']['ticket'][:40]}...'")
            print(
                f"  Predicted: {predicted_category}, Expected: {expected_category} {'✓' if predicted_category == expected_category else '✗'}"
            )

        baseline_accuracy = correct / total
        print(f"\n📈 Baseline accuracy: {baseline_accuracy:.2%}")

        # Step 5: Set up optimization
        print("\n🚀 Starting optimization...")
        print("This will:")
        print("• Find the best prompt structure")
        print("• Add helpful examples to the prompt")
        print("• Tune temperature and top_p parameters")
        print("• Test different approaches and pick the best")

        # Define what "good" means for this task
        accuracy_metric = AccuracyMetric(key="category")  # Focus on category classification

        # Create hybrid optimizer - this is LogiLLM's killer feature!
        optimizer = HybridOptimizer(
            metric=accuracy_metric,
            strategy="alternating",  # Alternate between prompts and parameters
            verbose=True,  # Show real-time progress
            n_trials=5,  # Fewer trials for faster demo (default is 25)
        )

        # Step 6: Run optimization
        optimization_result = await optimizer.optimize(
            module=classifier,
            dataset=training_data,
            param_space={
                "temperature": (0.0, 1.0),  # Find best temperature
                "top_p": (0.7, 1.0),  # Find best top_p
            },
        )

        # Step 7: Show optimization results
        print("\n🎯 Optimization Results:")
        print(f"Best Score: {optimization_result.best_score:.2%}")
        print(f"Improvement: {optimization_result.improvement:.1%} better than baseline")

        best_config = optimization_result.metadata.get("best_config", {})
        if best_config:
            print(f"Best Parameters: {best_config}")

        optimized_module = optimization_result.optimized_module
        demo_count = (
            len(optimized_module.demo_manager.demos) if optimized_module.demo_manager else 0
        )
        print(f"Examples Added: {demo_count} helpful examples")

        # Step 8: Test the optimized module
        print("\n🧪 Testing optimized performance...")

        test_ticket = (
            "The website is down and I can't place my order for an important event tomorrow"
        )

        # Compare original vs optimized
        original_result = await classifier(ticket=test_ticket)
        optimized_result = await optimized_module(ticket=test_ticket)

        print(f"Test Ticket: '{test_ticket}'")
        print("\nOriginal Model:")
        print(f"  Category: {original_result.outputs.get('category')}")
        print(f"  Urgency: {original_result.outputs.get('urgency')}")

        print("\nOptimized Model:")
        print(f"  Category: {optimized_result.outputs.get('category')}")
        print(f"  Urgency: {optimized_result.outputs.get('urgency')}")

        # Step 9: Use in production
        print("\n🏭 Ready for Production!")
        print("You can now use `optimized_module` in your application:")
        print("• It has better prompts")
        print("• It has optimized parameters")
        print("• It has learned from your training data")

        # Show how to use it
        production_tickets = [
            "I love the new update!",
            "My payment failed three times",
            "How do I cancel my subscription?",
        ]

        print("\n📝 Processing production tickets...")
        for ticket in production_tickets:
            result = await optimized_module(ticket=ticket)
            print(
                f"'{ticket}' → {result.outputs.get('category')} ({result.outputs.get('urgency')})"
            )

        print("\n✅ Optimization complete! Your model is now production-ready.")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
