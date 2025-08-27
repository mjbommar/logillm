#!/usr/bin/env python3
"""Few-Shot Learning with LogiLLM.

This comprehensive example demonstrates how few-shot learning improves model performance
using both string and class-based signatures. We use a smaller model (gpt-4.1-nano) to
show realistic improvement from baseline performance.

Key demonstrations:
1. String signature for simple classification
2. Class-based signature for structured outputs
3. Real performance improvement without tricks
4. Automatic example selection via bootstrap learning
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


class TechnicalTermClassifier(Signature):
    """Classify technical terms into specific categories."""

    term: str = InputField(desc="Technical term to classify")
    category: str = OutputField(desc="Category: algorithm, datastructure, pattern, or protocol")


async def demo_string_signature():
    """Demonstrate few-shot learning with simple string signatures."""
    print("\n📝 PART 1: String Signature Demo")
    print("=" * 45)

    # Simple string signature - great for quick prototyping
    classifier = Predict("language -> paradigm")

    # Test cases
    test_langs = ["python", "haskell", "java", "prolog"]
    expected = ["multiparadigm", "functional", "objectoriented", "logic"]

    # Baseline performance
    print("\nWithout examples:")
    baseline_correct = 0
    for lang, exp in zip(test_langs, expected):
        result = await classifier(language=lang)
        pred = result.outputs["paradigm"].replace(" ", "").replace("-", "").lower()
        exp_normalized = exp.replace(" ", "").replace("-", "").lower()
        correct = pred == exp_normalized
        baseline_correct += correct
        print(f"  {lang:8} → {result.outputs['paradigm']:20} {'✓' if correct else '✗'}")

    # Training data
    training = [
        {"inputs": {"language": "c"}, "outputs": {"paradigm": "procedural"}},
        {"inputs": {"language": "lisp"}, "outputs": {"paradigm": "functional"}},
        {"inputs": {"language": "smalltalk"}, "outputs": {"paradigm": "objectoriented"}},
        {"inputs": {"language": "erlang"}, "outputs": {"paradigm": "functional"}},
    ]

    # Optimize with few-shot
    optimizer = BootstrapFewShot(
        metric=lambda p, r: 1.0 if p.get("paradigm") == r.get("paradigm") else 0.0, max_demos=2
    )
    result = await optimizer.optimize(classifier, dataset=training)
    improved = result.optimized_module

    # Test with examples
    print("\nWith examples:")
    improved_correct = 0
    for lang, exp in zip(test_langs, expected):
        result = await improved(language=lang)
        pred = result.outputs["paradigm"].replace(" ", "").replace("-", "").lower()
        exp_normalized = exp.replace(" ", "").replace("-", "").lower()
        correct = pred == exp_normalized
        improved_correct += correct
        print(f"  {lang:8} → {result.outputs['paradigm']:20} {'✓' if correct else '✗'}")

    print(
        f"\nImprovement: {baseline_correct}/{len(test_langs)} → {improved_correct}/{len(test_langs)} "
        f"({(improved_correct - baseline_correct) / len(test_langs):+.0%})"
    )


async def demo_class_signature():
    """Demonstrate few-shot learning with class-based signatures."""
    print("\n📚 PART 2: Class-Based Signature Demo")
    print("=" * 45)

    # Create classifier with structured signature
    classifier = Predict(TechnicalTermClassifier)

    # Test cases - exact answers required
    test_cases = [
        ("quicksort", "algorithm"),
        ("hashmap", "datastructure"),
        ("singleton", "pattern"),
        ("http", "protocol"),
        ("bfs", "algorithm"),
        ("linkedlist", "datastructure"),
    ]

    # Baseline performance
    print("\nWithout examples:")

    baseline_correct = 0
    for term, expected in test_cases:
        result = await classifier(term=term)
        predicted = result.outputs.get("category", "").lower().strip()

        is_correct = predicted == expected
        if is_correct:
            baseline_correct += 1

        symbol = "✓" if is_correct else "✗"
        print(f"  {symbol} '{term}' → {predicted:15} (expected: {expected})")

    baseline_accuracy = baseline_correct / len(test_cases)
    print(f"\nBaseline: {baseline_accuracy:.1%} ({baseline_correct}/{len(test_cases)})")

    # Training data
    training_data = [
        {"inputs": {"term": "dijkstra"}, "outputs": {"category": "algorithm"}},
        {"inputs": {"term": "stack"}, "outputs": {"category": "datastructure"}},
        {"inputs": {"term": "observer"}, "outputs": {"category": "pattern"}},
        {"inputs": {"term": "tcp"}, "outputs": {"category": "protocol"}},
        {"inputs": {"term": "mergesort"}, "outputs": {"category": "algorithm"}},
        {"inputs": {"term": "queue"}, "outputs": {"category": "datastructure"}},
        {"inputs": {"term": "factory"}, "outputs": {"category": "pattern"}},
        {"inputs": {"term": "smtp"}, "outputs": {"category": "protocol"}},
    ]

    # Optimize with few-shot learning
    print("\n🎓 Training with few-shot examples...")

    def normalize_category(text):
        """Normalize category text for matching."""
        # Remove spaces, hyphens, and convert to lowercase
        return text.lower().replace(" ", "").replace("-", "")

    def exact_match_metric(pred, ref):
        """Simple exact match metric with normalization."""
        pred_norm = normalize_category(pred.get("category", ""))
        ref_norm = normalize_category(ref.get("category", ""))
        return 1.0 if pred_norm == ref_norm else 0.0

    optimizer = BootstrapFewShot(
        metric=exact_match_metric,
        max_demos=3,  # Use only 3 examples
        max_rounds=1,
    )

    result = await optimizer.optimize(module=classifier, dataset=training_data)
    improved_classifier = result.optimized_module

    # Test with examples
    print("\nWith examples:")

    improved_correct = 0
    for term, expected in test_cases:
        result = await improved_classifier(term=term)
        predicted = result.outputs.get("category", "").lower().strip()

        # Normalize for comparison
        pred_norm = predicted.replace(" ", "").replace("-", "")
        exp_norm = expected.replace(" ", "").replace("-", "")

        is_correct = pred_norm == exp_norm
        if is_correct:
            improved_correct += 1

        symbol = "✓" if is_correct else "✗"
        print(f"  {symbol} '{term}' → {predicted:15} (expected: {expected})")

    improved_accuracy = improved_correct / len(test_cases)
    improvement = improved_accuracy - baseline_accuracy

    # Show results
    print(
        f"\nImprovement: {baseline_correct}/{len(test_cases)} → {improved_correct}/{len(test_cases)} "
        f"({improvement:+.1%})"
    )

    # Show which examples were selected
    if improved_classifier.demo_manager.demos:
        print(f"\n📚 Selected Examples ({len(improved_classifier.demo_manager.demos)}):")
        for i, demo in enumerate(improved_classifier.demo_manager.demos, 1):
            term = demo.inputs.get("term", "")
            category = demo.outputs.get("category", "")
            print(f"  {i}. '{term}' → {category}")


async def main():
    """Run comprehensive few-shot learning demonstration."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=" * 50)
    print("🚀 FEW-SHOT LEARNING WITH LOGILLM")
    print("=" * 50)

    # Use smaller model for realistic baseline
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    # Demo 1: Simple string signatures
    await demo_string_signature()

    # Demo 2: Structured class signatures
    await demo_class_signature()

    # Summary
    print("\n" + "=" * 50)
    print("✨ KEY TAKEAWAYS")
    print("=" * 50)
    print("""
1. Few-shot learning dramatically improves accuracy on specific tasks
2. String signatures are great for quick prototyping
3. Class signatures provide better structure and validation
4. Even 2-3 examples can provide significant improvement
5. Bootstrap learning automatically selects the best examples
6. Using smaller models (gpt-4.1-nano) shows more realistic improvements
""")


if __name__ == "__main__":
    asyncio.run(main())
