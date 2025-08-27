#!/usr/bin/env python3
"""Simple demonstration of instruction optimization that actually works.

This example shows how LogiLLM can automatically improve the instructions
given to an LLM to get better performance on a specific task.
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import InstructionOptimizer
from logillm.providers import create_provider, register_provider


class NumberClassification(Signature):
    """Classify numbers into categories."""

    number: int = InputField(desc="A number to classify")
    category: str = OutputField(desc="Category: even, odd, prime, or composite")


async def main():
    """Show real instruction optimization."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== Instruction Optimization Demo ===\n")

    # Use smaller model for testing
    provider = create_provider("openai", model="gpt-4.1-mini")
    register_provider(provider, set_default=True)

    # Create module with POOR initial instructions
    classifier = Predict(NumberClassification)
    classifier.signature.instructions = "Just guess."  # Deliberately bad instruction

    # Training data
    train_data = [
        {"inputs": {"number": 2}, "outputs": {"category": "prime"}},
        {"inputs": {"number": 3}, "outputs": {"category": "prime"}},
        {"inputs": {"number": 4}, "outputs": {"category": "composite"}},
        {"inputs": {"number": 5}, "outputs": {"category": "prime"}},
        {"inputs": {"number": 6}, "outputs": {"category": "composite"}},
        {"inputs": {"number": 7}, "outputs": {"category": "prime"}},
        {"inputs": {"number": 8}, "outputs": {"category": "composite"}},
        {"inputs": {"number": 9}, "outputs": {"category": "composite"}},
    ]

    # Test numbers
    test_numbers = [11, 12, 13, 14, 15]
    expected = ["prime", "composite", "prime", "composite", "composite"]

    # Test with bad instructions
    print("📊 BASELINE with bad instruction: 'Just guess.'\n")
    baseline_correct = 0

    for num, exp in zip(test_numbers, expected):
        result = await classifier(number=num)
        category = result.outputs.get("category", "").lower()
        correct = exp in category
        baseline_correct += correct
        symbol = "✓" if correct else "✗"
        print(f"{symbol} {num} → {category} (expected: {exp})")

    print(f"\nBaseline accuracy: {baseline_correct}/{len(test_numbers)}")

    # Define accuracy metric
    def category_metric(pred, ref):
        """Check if category matches."""
        pred_cat = pred.get("category", "").lower()
        ref_cat = ref.get("category", "").lower()
        return 1.0 if ref_cat in pred_cat else 0.0

    # Optimize instructions
    print("\n🚀 OPTIMIZING instructions...\n")

    optimizer = InstructionOptimizer(
        metric=category_metric,
        num_candidates=5,  # Try 5 different instructions
        max_iterations=2,  # 2 rounds of improvement
    )

    result = await optimizer.optimize(module=classifier, dataset=train_data)

    optimized = result.optimized_module

    # Show the new instruction
    print("📝 Original instruction: 'Just guess.'")
    print(f"📝 Optimized instruction: '{optimized.signature.instructions}'")

    # Test with optimized instructions
    print("\n📊 OPTIMIZED performance:\n")
    optimized_correct = 0

    for num, exp in zip(test_numbers, expected):
        result = await optimized(number=num)
        category = result.outputs.get("category", "").lower()
        correct = exp in category
        optimized_correct += correct
        symbol = "✓" if correct else "✗"
        print(f"{symbol} {num} → {category} (expected: {exp})")

    print(f"\nOptimized accuracy: {optimized_correct}/{len(test_numbers)}")

    # Show improvement
    print("\n" + "=" * 50)
    print("📈 RESULTS:")
    print(f"  Before: {baseline_correct}/{len(test_numbers)} correct")
    print(f"  After:  {optimized_correct}/{len(test_numbers)} correct")
    print(f"  Improvement: {optimized_correct - baseline_correct} more correct")

    print("\n✨ The optimizer discovered that clear, specific instructions")
    print("   dramatically improve performance compared to vague ones.")


if __name__ == "__main__":
    asyncio.run(main())
