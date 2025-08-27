#!/usr/bin/env python3
"""FormatOptimizer: Automatic Format Discovery.

FormatOptimizer discovers which prompt format (Markdown, JSON, XML, etc.)
works best for your specific task. Different models and tasks perform
better with different formats, and this optimizer finds the optimal one.

Key features:
- Tests multiple format adapters automatically
- Considers both accuracy and parsing reliability
- Tracks format preferences per model
- Discovers hybrid formats for complex tasks
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import FormatOptimizer
from logillm.providers import create_provider, register_provider


class DataExtraction(Signature):
    """Extract structured data from text."""

    text: str = InputField(desc="Unstructured text to parse")
    dates: list[str] = OutputField(desc="Dates mentioned")
    amounts: list[float] = OutputField(desc="Monetary amounts")
    entities: list[str] = OutputField(desc="Named entities")
    summary: str = OutputField(desc="Brief summary")


async def main():
    """Demonstrate format discovery for structured extraction."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== FormatOptimizer: Automatic Format Discovery ===\n")

    # Use smaller model to show format impact
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    extractor = Predict(DataExtraction)

    # Training data with complex structured outputs
    train_data = [
        {
            "inputs": {
                "text": "On March 15, 2024, ABC Corp paid $1,250.50 to XYZ Ltd for consulting services."
            },
            "outputs": {
                "dates": ["March 15, 2024"],
                "amounts": [1250.50],
                "entities": ["ABC Corp", "XYZ Ltd"],
                "summary": "Payment for consulting services",
            },
        },
        {
            "inputs": {
                "text": "The contract worth $5,000,000 was signed on January 1st by Microsoft and OpenAI."
            },
            "outputs": {
                "dates": ["January 1st"],
                "amounts": [5000000],
                "entities": ["Microsoft", "OpenAI"],
                "summary": "Contract signing between tech companies",
            },
        },
        {
            "inputs": {
                "text": "Invoice #12345 dated 2024-06-30 shows a total of $789.99 from Amazon."
            },
            "outputs": {
                "dates": ["2024-06-30"],
                "amounts": [789.99],
                "entities": ["Amazon"],
                "summary": "Invoice from Amazon",
            },
        },
    ]

    # Test cases
    test_cases = [
        "Apple announced $10 billion revenue on October 30, 2024.",
        "The $2,500 payment to Google was processed on Dec 25.",
        "Tesla and SpaceX signed a $50M deal yesterday.",
    ]

    # Baseline test
    print("📊 BASELINE (default format):")
    print("-" * 40)
    baseline_scores = []

    for text in test_cases:
        result = await extractor(text=text)
        dates = result.outputs.get("dates") or []
        amounts = result.outputs.get("amounts") or []
        entities = result.outputs.get("entities") or []

        # Score based on extraction completeness
        score = (len(dates) + len(amounts) + len(entities)) / 3.0
        baseline_scores.append(score)
        print(f"Text: {text[:40]}...")
        print(f"  Extracted: {len(dates)} dates, {len(amounts)} amounts, {len(entities)} entities")

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"\nBaseline score: {baseline_avg:.2f}")

    # Define extraction quality metric
    def extraction_metric(pred, ref):
        """Score based on extraction accuracy."""
        score = 0.0

        # Check dates
        pred_dates = pred.get("dates") or []
        ref_dates = ref.get("dates") or []
        if pred_dates and ref_dates:
            score += (
                min(len(pred_dates), len(ref_dates)) / max(len(pred_dates), len(ref_dates)) * 0.25
            )

        # Check amounts
        pred_amounts = pred.get("amounts") or []
        ref_amounts = ref.get("amounts") or []
        if pred_amounts and ref_amounts:
            score += (
                min(len(pred_amounts), len(ref_amounts))
                / max(len(pred_amounts), len(ref_amounts))
                * 0.25
            )

        # Check entities
        pred_entities = pred.get("entities") or []
        ref_entities = ref.get("entities") or []
        if pred_entities and ref_entities:
            score += (
                min(len(pred_entities), len(ref_entities))
                / max(len(pred_entities), len(ref_entities))
                * 0.25
            )

        # Check summary exists
        if pred.get("summary"):
            score += 0.25

        return score

    # Optimize format
    print("\n🚀 DISCOVERING optimal format...")

    optimizer = FormatOptimizer(
        metric=extraction_metric,
        track_by_model=True,  # Track which formats work best per model
    )

    result = await optimizer.optimize(module=extractor, dataset=train_data)

    optimized_extractor = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (with best format):")
    print("-" * 40)
    optimized_scores = []

    for text in test_cases:
        result = await optimized_extractor(text=text)
        dates = result.outputs.get("dates") or []
        amounts = result.outputs.get("amounts") or []
        entities = result.outputs.get("entities") or []
        summary = result.outputs.get("summary", "")

        score = (len(dates) + len(amounts) + len(entities)) / 3.0
        optimized_scores.append(score)
        print(f"Text: {text[:40]}...")
        print(f"  Extracted: {len(dates)} dates, {len(amounts)} amounts, {len(entities)} entities")
        if summary:
            print(f"  Summary: {summary[:50]}...")

    optimized_avg = sum(optimized_scores) / len(optimized_scores)
    improvement = ((optimized_avg - baseline_avg) / max(0.1, baseline_avg)) * 100

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline score:  {baseline_avg:.2f}")
    print(f"  Optimized score: {optimized_avg:.2f}")
    print(f"  Improvement:     {improvement:+.1f}%")

    # Show discovered format
    if hasattr(result, "metadata") and "best_format" in result.metadata:
        print("\n📝 Discovered optimal format:")
        print(f"  Format: {result.metadata['best_format']}")
        if "format_scores" in result.metadata:
            print("\n  Format comparison:")
            for fmt, score in result.metadata["format_scores"].items():
                print(f"    {fmt:<12}: {score:.3f}")

    print("\n✨ Key insight: Different formats (JSON, Markdown, XML) have")
    print("   dramatic impacts on extraction accuracy. FormatOptimizer")
    print("   automatically discovers which works best for your task.")


if __name__ == "__main__":
    asyncio.run(main())
