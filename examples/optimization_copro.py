#!/usr/bin/env python3
"""COPRO: Collaborative Prompt Optimization.

COPRO automatically generates and refines instructions by exploring
variations and learning what works. It uses breadth-first search with
temperature-based creativity to discover effective prompts.

Key features:
- Breadth-first instruction exploration
- Temperature-controlled creativity
- Iterative refinement based on performance
- Duplicate detection to avoid redundancy
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import COPRO
from logillm.providers import create_provider, register_provider


class SentimentAnalysis(Signature):
    """Analyze sentiment with reasoning."""

    text: str = InputField(desc="Text to analyze")
    sentiment: str = OutputField(desc="Sentiment: positive, negative, or neutral")
    confidence: float = OutputField(desc="Confidence score 0-1")
    reasoning: str = OutputField(desc="Why this sentiment was chosen")


async def main():
    """Demonstrate COPRO's instruction optimization."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== COPRO: Collaborative Instruction Optimization ===\n")

    # Use smaller model for realistic baseline
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    analyzer = Predict(SentimentAnalysis)

    # Training data with nuanced sentiments
    train_data = [
        {
            "inputs": {"text": "This product exceeded my expectations in every way!"},
            "outputs": {
                "sentiment": "positive",
                "confidence": "0.95",
                "reasoning": "Strong positive language with enthusiasm",
            },
        },
        {
            "inputs": {"text": "It's okay, nothing special but does the job."},
            "outputs": {
                "sentiment": "neutral",
                "confidence": "0.8",
                "reasoning": "Balanced view without strong emotion",
            },
        },
        {
            "inputs": {"text": "Terrible experience, would not recommend to anyone."},
            "outputs": {
                "sentiment": "negative",
                "confidence": "0.9",
                "reasoning": "Clear negative sentiment with warning to others",
            },
        },
        {
            "inputs": {"text": "Not bad, but I've seen better for the price."},
            "outputs": {
                "sentiment": "negative",
                "confidence": "0.6",
                "reasoning": "Mild criticism comparing to alternatives",
            },
        },
        {
            "inputs": {"text": "Absolutely love it! Worth every penny!"},
            "outputs": {
                "sentiment": "positive",
                "confidence": "0.98",
                "reasoning": "Multiple strong positive indicators",
            },
        },
    ]

    # Test cases with tricky sentiments
    test_cases = [
        "The product works, I guess. Could be worse.",  # Ambiguous
        "Amazing quality but the price is ridiculous!",  # Mixed
        "Fine for what it is.",  # Neutral
        "Surprisingly good, actually impressed!",  # Positive
    ]

    # Baseline test
    print("📊 BASELINE (default instructions):")
    print("-" * 40)
    baseline_scores = []

    for text in test_cases:
        result = await analyzer(text=text)
        sentiment = result.outputs.get("sentiment", "unknown")
        confidence = Extractors.percentage(
            result.outputs.get("confidence"), as_decimal=True, default=0.5
        )
        baseline_scores.append(confidence)
        print(f"'{text[:40]}...'")
        print(f"  → {sentiment} (conf: {confidence:.2f})")

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"\nBaseline confidence: {baseline_avg:.2f}")

    # Define metric focusing on confidence
    def confidence_metric(pred, ref):
        """Higher confidence with correct sentiment is better."""
        pred_sentiment = pred.get("sentiment", "").lower()
        ref_sentiment = ref.get("sentiment", "").lower()
        confidence = Extractors.percentage(pred.get("confidence"), as_decimal=True, default=0.5)

        # Reward high confidence when correct
        if pred_sentiment == ref_sentiment:
            return confidence
        else:
            return 1.0 - confidence  # Penalize confident wrong answers

    # Optimize with COPRO
    print("\n🚀 OPTIMIZING instructions with COPRO...")

    optimizer = COPRO(
        metric=confidence_metric,
        breadth=5,  # Try 5 variations per round
        depth=2,  # 2 rounds of refinement
        init_temperature=1.2,  # Start creative
    )

    result = await optimizer.optimize(module=analyzer, dataset=train_data)

    optimized_analyzer = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (with COPRO instructions):")
    print("-" * 40)
    optimized_scores = []

    for text in test_cases:
        result = await optimized_analyzer(text=text)
        sentiment = result.outputs.get("sentiment", "unknown")
        confidence = Extractors.percentage(
            result.outputs.get("confidence"), as_decimal=True, default=0.5
        )
        reasoning = result.outputs.get("reasoning", "")[:50]
        optimized_scores.append(confidence)
        print(f"'{text[:40]}...'")
        print(f"  → {sentiment} (conf: {confidence:.2f})")
        if reasoning:
            print(f"  Reason: {reasoning}...")

    optimized_avg = sum(optimized_scores) / len(optimized_scores)
    improvement = ((optimized_avg - baseline_avg) / baseline_avg) * 100

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline confidence:  {baseline_avg:.2f}")
    print(f"  Optimized confidence: {optimized_avg:.2f}")
    print(f"  Improvement:          {improvement:+.1f}%")

    # Show optimized instruction
    if hasattr(optimized_analyzer, "signature") and hasattr(
        optimized_analyzer.signature, "__doc__"
    ):
        print("\n📝 COPRO discovered instruction:")
        print(f'  "{optimized_analyzer.signature.__doc__}"')

    # Show COPRO stats if available
    if hasattr(result, "metadata") and "candidates_evaluated" in result.metadata:
        print("\n🔍 COPRO exploration:")
        print(f"  Instructions tested: {result.metadata['candidates_evaluated']}")
        print(f"  Rounds completed:    {result.metadata.get('rounds', 'N/A')}")

    print("\n✨ Key insight: COPRO automatically discovers better ways to")
    print("   phrase instructions, often finding non-obvious improvements")
    print("   that human prompt engineers might miss.")


if __name__ == "__main__":
    asyncio.run(main())
