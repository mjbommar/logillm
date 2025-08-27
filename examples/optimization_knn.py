#!/usr/bin/env python3
"""KNNFewShot: Dynamic Similarity-Based Example Selection.

KNNFewShot selects the most relevant examples for each query using
semantic similarity. Unlike static few-shot approaches, it dynamically
chooses different examples based on what's most similar to the input.

Key features:
- Dynamic per-query example selection
- Semantic similarity using embeddings
- Falls back to bootstrap if performance is poor
- Adapts examples to each specific input
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import KNNFewShot
from logillm.providers import create_provider, register_provider


class TechnicalSupport(Signature):
    """Provide technical support answers."""

    question: str = InputField(desc="User's technical question")
    category: str = OutputField(desc="Problem category")
    solution: str = OutputField(desc="Step-by-step solution")
    difficulty: str = OutputField(desc="Difficulty: easy, medium, hard")


async def main():
    """Demonstrate KNN's dynamic example selection."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== KNNFewShot: Dynamic Example Selection ===\n")

    # Use smaller model to show improvement
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    support = Predict(TechnicalSupport)

    # Diverse training data covering different categories
    train_data = [
        # Network issues
        {
            "inputs": {"question": "My WiFi keeps disconnecting every few minutes"},
            "outputs": {
                "category": "network",
                "solution": "1. Restart router\n2. Update network drivers\n3. Check for interference",
                "difficulty": "easy",
            },
        },
        {
            "inputs": {"question": "Can't access company VPN from home"},
            "outputs": {
                "category": "network",
                "solution": "1. Verify credentials\n2. Check firewall settings\n3. Try different network",
                "difficulty": "medium",
            },
        },
        # Software issues
        {
            "inputs": {"question": "Excel crashes when opening large files"},
            "outputs": {
                "category": "software",
                "solution": "1. Update Excel\n2. Increase memory allocation\n3. Split large files",
                "difficulty": "medium",
            },
        },
        {
            "inputs": {"question": "Python script gives ImportError"},
            "outputs": {
                "category": "software",
                "solution": "1. Check module installation\n2. Verify Python path\n3. Install missing packages",
                "difficulty": "easy",
            },
        },
        # Hardware issues
        {
            "inputs": {"question": "Computer won't turn on at all"},
            "outputs": {
                "category": "hardware",
                "solution": "1. Check power cable\n2. Test power supply\n3. Remove and reseat RAM",
                "difficulty": "hard",
            },
        },
        {
            "inputs": {"question": "External monitor not detected"},
            "outputs": {
                "category": "hardware",
                "solution": "1. Check cable connection\n2. Update display drivers\n3. Try different port",
                "difficulty": "easy",
            },
        },
        # Security issues
        {
            "inputs": {"question": "Suspicious emails asking for password"},
            "outputs": {
                "category": "security",
                "solution": "1. Don't click links\n2. Report as phishing\n3. Change password if clicked",
                "difficulty": "easy",
            },
        },
        {
            "inputs": {"question": "Ransomware encrypted my files"},
            "outputs": {
                "category": "security",
                "solution": "1. Disconnect from network\n2. Don't pay ransom\n3. Contact IT security team",
                "difficulty": "hard",
            },
        },
    ]

    # Test with questions similar to different categories
    test_cases = [
        "WiFi signal is weak in my office",  # Should match network examples
        "JavaScript code throws undefined error",  # Should match software examples
        "Laptop battery drains too quickly",  # Should match hardware examples
        "Got a suspicious login attempt alert",  # Should match security examples
    ]

    # Baseline test
    print("📊 BASELINE (no optimization):")
    print("-" * 40)
    baseline_categories = []

    for question in test_cases:
        result = await support(question=question)
        category = result.outputs.get("category", "unknown")
        difficulty = result.outputs.get("difficulty", "unknown")
        baseline_categories.append(category)
        print(f"Q: {question[:40]}...")
        print(f"  Category: {category:<10} Difficulty: {difficulty}")

    # Simple accuracy metric
    def support_metric(pred, ref):
        """Score based on category match and solution quality."""
        pred_cat = pred.get("category", "").lower()
        ref_cat = ref.get("category", "").lower()
        solution = pred.get("solution", "")

        score = 0.0
        if pred_cat == ref_cat:
            score += 0.5
        if len(solution) > 20:  # Has substantial solution
            score += 0.3
        if "\n" in solution:  # Multi-step solution
            score += 0.2

        return score

    # Optimize with KNN
    print("\n🚀 OPTIMIZING with KNN dynamic selection...")

    optimizer = KNNFewShot(
        k=2,  # Use 2 nearest neighbors
        trainset=train_data,
        metric=support_metric,
        bootstrap_fallback=True,  # Fall back if performance is poor
    )

    result = await optimizer.optimize(module=support, dataset=train_data)

    optimized_support = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (with KNN selection):")
    print("-" * 40)
    optimized_categories = []

    for _, question in enumerate(test_cases):
        result = await optimized_support(question=question)
        category = result.outputs.get("category", "unknown")
        difficulty = result.outputs.get("difficulty", "unknown")
        solution = result.outputs.get("solution", "")
        optimized_categories.append(category)

        print(f"Q: {question[:40]}...")
        print(f"  Category: {category:<10} Difficulty: {difficulty}")

        # Show which examples KNN selected for this query
        if hasattr(optimized_support, "demo_manager"):
            # In KNN, demos are selected per query, so this might vary
            print(f"  Solution preview: {solution[:60]}...")

    # Compare category accuracy
    expected_categories = ["network", "software", "hardware", "security"]
    baseline_correct = sum(
        1 for b, e in zip(baseline_categories, expected_categories) if b.lower() == e
    )
    optimized_correct = sum(
        1 for o, e in zip(optimized_categories, expected_categories) if o.lower() == e
    )

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline correct:  {baseline_correct}/{len(test_cases)}")
    print(f"  Optimized correct: {optimized_correct}/{len(test_cases)}")
    print(f"  Improvement:       {(optimized_correct - baseline_correct)} more correct")

    print("\n✨ Key insight: KNN dynamically selects the most relevant")
    print("   examples for each query, unlike static few-shot which")
    print("   uses the same examples for everything.")


if __name__ == "__main__":
    asyncio.run(main())
