#!/usr/bin/env python3
"""SIMBA: Stochastic Introspective Mini-Batch Ascent.

SIMBA optimizes prompts through intelligent demo selection and introspective
rule generation. It uses mini-batches to efficiently explore the space of
possible demonstrations and learn what makes examples effective.

Key features:
- Mini-batch sampling for efficient exploration
- Introspective rules that explain why demos work
- Parallel evaluation for speed
- Automatic demo appending based on performance
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import SIMBA
from logillm.providers import create_provider, register_provider


class SimpleClassification(Signature):
    """Classify text into a category."""

    text: str = InputField(desc="Text to classify")
    category: str = OutputField(desc="Category: news, sports, tech, or entertainment")
    confidence: float = OutputField(desc="Confidence score from 0.0 to 1.0")


async def main():
    """Demonstrate SIMBA's introspective optimization."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== SIMBA: Introspective Demo Optimization ===\n")

    # Use smaller model for realistic baseline
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    classifier = Predict(SimpleClassification)

    # Training data with clear categories
    train_data = [
        {
            "inputs": {
                "text": "The new iPhone launches next week with improved cameras and battery life."
            },
            "outputs": {
                "category": "tech",
                "confidence": "0.9",
            },
        },
        {
            "inputs": {
                "text": "The Lakers won their game last night with a buzzer-beater three-pointer."
            },
            "outputs": {
                "category": "sports",
                "confidence": "0.95",
            },
        },
        {
            "inputs": {
                "text": "Breaking: Congress passes new climate legislation after months of debate."
            },
            "outputs": {
                "category": "news",
                "confidence": "0.85",
            },
        },
        {
            "inputs": {"text": "Marvel announces three new movies for their upcoming phase."},
            "outputs": {
                "category": "entertainment",
                "confidence": "0.9",
            },
        },
        {
            "inputs": {
                "text": "Stock market reaches all-time high amid positive earnings reports."
            },
            "outputs": {
                "category": "news",
                "confidence": "0.8",
            },
        },
        {
            "inputs": {
                "text": "New AI model achieves breakthrough in natural language understanding."
            },
            "outputs": {
                "category": "tech",
                "confidence": "0.95",
            },
        },
    ]

    # Test cases
    test_cases = [
        "Microsoft unveils new Surface laptop with AI capabilities",
        "The championship game ended in overtime with a dramatic finish",
        "Scientists discover new treatment for rare disease",
        "Netflix releases trailer for highly anticipated new series",
    ]

    # Expected categories for scoring
    expected = ["tech", "sports", "news", "entertainment"]

    # Baseline test
    print("📊 BASELINE (no optimization):")
    print("-" * 40)
    baseline_correct = 0

    for text, expected_cat in zip(test_cases, expected):
        result = await classifier(text=text)
        category = result.outputs.get("category", "unknown").lower()
        confidence = Extractors.percentage(
            result.outputs.get("confidence"), as_decimal=True, default=0.5
        )

        # Check if correct
        if expected_cat in category:
            baseline_correct += 1
            symbol = "✓"
        else:
            symbol = "✗"

        print(f"{symbol} Text: {text[:40]}...")
        print(f"  → {category} (confidence: {confidence:.2f})")

    print(f"\nBaseline accuracy: {baseline_correct}/{len(test_cases)}")

    # Define classification metric
    def classification_metric(pred, ref):
        """Score based on category match and confidence."""
        pred_cat = pred.get("category", "").lower()
        ref_cat = ref.get("category", "").lower()
        confidence = Extractors.percentage(pred.get("confidence"), as_decimal=True, default=0.5)

        # Exact match with confidence weighting
        if pred_cat == ref_cat:
            return confidence
        # Partial credit if category appears in response
        elif ref_cat in pred_cat:
            return confidence * 0.5
        else:
            return 0.0

    # Optimize with SIMBA
    print("\n🚀 OPTIMIZING with SIMBA...")

    optimizer = SIMBA(
        metric=classification_metric,
        bsize=3,  # Small batch for demo
        num_candidates=4,  # Number of instruction candidates
        max_steps=2,  # Quick optimization
        max_demos=3,  # Max examples to include
    )

    result = await optimizer.optimize(module=classifier, dataset=train_data)

    optimized_classifier = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (with SIMBA):")
    print("-" * 40)
    optimized_correct = 0

    for text, expected_cat in zip(test_cases, expected):
        result = await optimized_classifier(text=text)
        category = result.outputs.get("category", "unknown").lower()
        confidence = Extractors.percentage(
            result.outputs.get("confidence"), as_decimal=True, default=0.5
        )

        # Check if correct
        if expected_cat in category:
            optimized_correct += 1
            symbol = "✓"
        else:
            symbol = "✗"

        print(f"{symbol} Text: {text[:40]}...")
        print(f"  → {category} (confidence: {confidence:.2f})")

    improvement = optimized_correct - baseline_correct

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline:  {baseline_correct}/{len(test_cases)} correct")
    print(f"  Optimized: {optimized_correct}/{len(test_cases)} correct")
    print(f"  Improvement: {improvement:+d} more correct")

    # Show selected demos
    if optimized_classifier.demo_manager and optimized_classifier.demo_manager.demos:
        print(f"\n📚 SIMBA selected {len(optimized_classifier.demo_manager.demos)} demos:")
        for i, demo in enumerate(optimized_classifier.demo_manager.demos[:3], 1):
            text = demo.inputs.get("text", "")[:40]
            cat = demo.outputs.get("category", "")
            print(f"  {i}. '{text}...' → {cat}")

    print("\n✨ Key insight: SIMBA learns WHY certain examples work better,")
    print("   creating introspective rules that guide future selections.")


if __name__ == "__main__":
    asyncio.run(main())
