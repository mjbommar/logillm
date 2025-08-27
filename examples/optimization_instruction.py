#!/usr/bin/env python3
"""InstructionOptimizer: Task-Aware Instruction Generation.

InstructionOptimizer automatically generates and refines task-specific
instructions by analyzing successful patterns in training data and
evolving instructions based on performance feedback.

Key features:
- Automatic instruction generation from examples
- Task-aware instruction refinement
- Pattern analysis of successful executions
- Instruction evolution based on failure analysis
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import InstructionOptimizer
from logillm.providers import create_provider, register_provider


class NewsHeadline(Signature):
    """Generate engaging news headlines."""

    article_summary: str = InputField(desc="Brief summary of the news article")
    target_tone: str = InputField(desc="Desired tone: serious, casual, dramatic")
    headline: str = OutputField(desc="Compelling news headline")
    engagement_score: float = OutputField(desc="Predicted engagement 0-10")
    word_count: int = OutputField(desc="Number of words in headline")


async def main():
    """Demonstrate task-aware instruction optimization."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== InstructionOptimizer: Task-Aware Generation ===\n")

    # Use smaller model for demonstration
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    headline_generator = Predict(NewsHeadline)

    # Training data with effective headlines
    train_data = [
        {
            "inputs": {
                "article_summary": "Scientists discover new exoplanet with conditions similar to Earth",
                "target_tone": "serious",
            },
            "outputs": {
                "headline": "Earth-Like Exoplanet Discovered in Habitable Zone of Distant Star",
                "engagement_score": "8.5",
                "word_count": "10",
            },
        },
        {
            "inputs": {
                "article_summary": "Local bakery wins national competition with unique sourdough recipe",
                "target_tone": "casual",
            },
            "outputs": {
                "headline": "Small Town Bakery's Secret Recipe Takes Home National Prize",
                "engagement_score": "7.8",
                "word_count": "9",
            },
        },
        {
            "inputs": {
                "article_summary": "Major earthquake hits coastal region, evacuation underway",
                "target_tone": "dramatic",
            },
            "outputs": {
                "headline": "Massive 7.2 Earthquake Triggers Emergency Evacuations Along Coast",
                "engagement_score": "9.2",
                "word_count": "8",
            },
        },
        {
            "inputs": {
                "article_summary": "Tech company announces breakthrough in quantum computing",
                "target_tone": "serious",
            },
            "outputs": {
                "headline": "Quantum Computing Milestone Achieved by Leading Tech Firm",
                "engagement_score": "8.0",
                "word_count": "8",
            },
        },
        {
            "inputs": {
                "article_summary": "Celebrity chef opens new restaurant in downtown district",
                "target_tone": "casual",
            },
            "outputs": {
                "headline": "Famous Chef Brings Culinary Magic to Downtown Scene",
                "engagement_score": "7.5",
                "word_count": "8",
            },
        },
    ]

    # Test cases for different tones and topics
    test_cases = [
        {
            "article_summary": "New study reveals surprising health benefits of dark chocolate",
            "target_tone": "casual",
        },
        {
            "article_summary": "Government announces major infrastructure investment plan",
            "target_tone": "serious",
        },
        {
            "article_summary": "Rare wildlife species spotted in urban park after decades",
            "target_tone": "dramatic",
        },
    ]

    # Baseline test with default instructions
    print("📊 BASELINE (generic instructions):")
    print("-" * 40)
    baseline_scores = []
    baseline_lengths = []

    for test in test_cases:
        result = await headline_generator(**test)
        headline = result.outputs.get("headline", "")
        engagement = Extractors.number(result.outputs.get("engagement_score"), default=5.0)
        word_count = Extractors.number(
            result.outputs.get("word_count"), default=len(headline.split())
        )

        baseline_scores.append(engagement)
        baseline_lengths.append(word_count)

        print(f"Summary: {test['article_summary'][:50]}...")
        print(f"Tone: {test['target_tone']}")
        print(f"  → '{headline}'")
        print(f"  Engagement: {engagement:.1f}, Words: {int(word_count)}")

    baseline_avg_engagement = sum(baseline_scores) / len(baseline_scores)
    baseline_avg_length = sum(baseline_lengths) / len(baseline_lengths)
    print(
        f"\nBaseline avg: engagement {baseline_avg_engagement:.1f}, length {baseline_avg_length:.0f} words"
    )

    # Define metric that balances engagement and conciseness
    def headline_metric(pred, ref=None):
        """Reward high engagement with appropriate length."""
        engagement = Extractors.number(pred.get("engagement_score"), default=5.0)
        word_count = Extractors.number(pred.get("word_count"), default=10.0)
        headline = pred.get("headline", "")

        # Normalize engagement (0-10 scale)
        engagement_score = engagement / 10.0

        # Optimal headline length is 6-12 words
        if 6 <= word_count <= 12:
            length_penalty = 0.0
        elif word_count < 6:
            length_penalty = (6 - word_count) * 0.05  # Too short
        else:
            length_penalty = (word_count - 12) * 0.03  # Too long

        # Bonus for using powerful words
        power_words = [
            "breakthrough",
            "shocking",
            "amazing",
            "revolutionary",
            "exclusive",
            "urgent",
        ]
        power_bonus = sum(0.05 for word in power_words if word.lower() in headline.lower())

        return max(0.0, engagement_score - length_penalty + power_bonus)

    # Optimize instructions for this specific task
    print("\n🎯 OPTIMIZING task-specific instructions...")

    optimizer = InstructionOptimizer(
        metric=headline_metric,
        instruction_generations=5,  # Try 5 instruction variations
        evolution_rounds=2,  # 2 rounds of refinement
        analyze_patterns=True,  # Analyze successful patterns
        use_failure_analysis=True,  # Learn from failures
    )

    result = await optimizer.optimize(
        module=headline_generator,
        dataset=train_data,
    )

    optimized_generator = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (task-aware instructions):")
    print("-" * 40)
    optimized_scores = []
    optimized_lengths = []

    for test in test_cases:
        result = await optimized_generator(**test)
        headline = result.outputs.get("headline", "")
        engagement = Extractors.number(result.outputs.get("engagement_score"), default=5.0)
        word_count = Extractors.number(
            result.outputs.get("word_count"), default=len(headline.split())
        )

        optimized_scores.append(engagement)
        optimized_lengths.append(word_count)

        print(f"Summary: {test['article_summary'][:50]}...")
        print(f"Tone: {test['target_tone']}")
        print(f"  → '{headline}'")
        print(f"  Engagement: {engagement:.1f}, Words: {int(word_count)}")

    optimized_avg_engagement = sum(optimized_scores) / len(optimized_scores)
    optimized_avg_length = sum(optimized_lengths) / len(optimized_lengths)

    engagement_improvement = (
        (optimized_avg_engagement - baseline_avg_engagement) / baseline_avg_engagement
    ) * 100
    length_change = optimized_avg_length - baseline_avg_length

    # Results
    print("\n" + "=" * 50)
    print("📈 RESULTS:")
    print(f"  Baseline engagement:  {baseline_avg_engagement:.1f}")
    print(f"  Optimized engagement: {optimized_avg_engagement:.1f}")
    print(f"  Engagement improvement: {engagement_improvement:+.1f}%")
    print(f"  Average length change:  {length_change:+.1f} words")

    # Show discovered instruction patterns
    if hasattr(result, "metadata"):
        print("\n🎯 Instruction optimization insights:")
        if "best_instruction" in result.metadata:
            instruction = result.metadata["best_instruction"][:100]
            print(f"  Best instruction: '{instruction}...'")
        if "patterns_found" in result.metadata:
            print(f"  Success patterns:  {len(result.metadata['patterns_found'])}")
        if "instructions_tested" in result.metadata:
            print(f"  Instructions tested: {result.metadata['instructions_tested']}")

    # Show evolved instruction
    if hasattr(optimized_generator, "signature") and hasattr(
        optimized_generator.signature, "__doc__"
    ):
        evolved_instruction = optimized_generator.signature.__doc__
        if evolved_instruction and evolved_instruction != "Generate engaging news headlines.":
            print("\n📝 Evolved instruction:")
            print(f'  "{evolved_instruction}"')

    # Show pattern analysis if available
    if hasattr(result, "metadata") and "successful_patterns" in result.metadata:
        patterns = result.metadata["successful_patterns"][:3]  # Show top 3
        if patterns:
            print("\n📊 Discovered success patterns:")
            for i, pattern in enumerate(patterns, 1):
                print(f"  {i}. {pattern}")

    print("\n✨ Key insight: Task-aware instruction optimization learns")
    print("   what makes headlines effective for specific topics and tones,")
    print("   automatically improving instructions based on real patterns.")


if __name__ == "__main__":
    asyncio.run(main())
