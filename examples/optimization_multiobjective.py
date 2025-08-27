#!/usr/bin/env python3
"""MultiObjective: Optimize Multiple Goals Simultaneously.

MultiObjectiveOptimizer balances multiple objectives like accuracy,
speed, and cost. Instead of optimizing for just one metric, it finds
solutions that represent different trade-offs between competing goals.

Key features:
- Pareto frontier optimization for non-dominated solutions
- Configurable weights for different objectives
- Constraint-based optimization
- Trade-off analysis and reporting
"""

import asyncio
import os
import time

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import MultiObjectiveOptimizer
from logillm.providers import create_provider, register_provider


class ProductDescription(Signature):
    """Generate product descriptions for e-commerce."""

    product_name: str = InputField(desc="Name of the product")
    key_features: str = InputField(desc="Main product features")
    target_audience: str = InputField(desc="Target customer group")
    description: str = OutputField(desc="Compelling product description")
    selling_points: str = OutputField(desc="Key selling points")
    urgency_score: float = OutputField(desc="How urgently worded 0-10")


async def main():
    """Demonstrate multi-objective optimization across competing goals."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== MultiObjective: Balancing Competing Goals ===\n")

    # Use smaller model for cost optimization
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    writer = Predict(ProductDescription)

    # Training data for product descriptions
    train_data = [
        {
            "inputs": {
                "product_name": "Wireless Bluetooth Headphones",
                "key_features": "Noise cancelling, 30hr battery, quick charge",
                "target_audience": "professionals and commuters",
            },
            "outputs": {
                "description": "Experience crystal-clear audio anywhere with these premium wireless headphones featuring advanced noise cancellation and marathon battery life.",
                "selling_points": "All-day comfort, superior sound quality, quick 15-minute charge gives 3 hours playback",
                "urgency_score": "6",
            },
        },
        {
            "inputs": {
                "product_name": "Organic Green Tea",
                "key_features": "Premium leaves, antioxidant-rich, sustainably sourced",
                "target_audience": "health-conscious consumers",
            },
            "outputs": {
                "description": "Savor the pure taste of nature with our hand-picked organic green tea, carefully crafted for the ultimate wellness experience.",
                "selling_points": "Certified organic, rich in antioxidants, ethically sourced from mountain gardens",
                "urgency_score": "3",
            },
        },
        {
            "inputs": {
                "product_name": "Gaming Mechanical Keyboard",
                "key_features": "RGB lighting, tactile switches, programmable keys",
                "target_audience": "gamers and enthusiasts",
            },
            "outputs": {
                "description": "Dominate every game with this high-performance mechanical keyboard featuring lightning-fast response and stunning RGB effects.",
                "selling_points": "Tournament-grade switches, customizable lighting, macro programming support",
                "urgency_score": "8",
            },
        },
    ]

    # Test cases
    test_cases = [
        {
            "product_name": "Smart Fitness Watch",
            "key_features": "Heart rate monitor, GPS, waterproof",
            "target_audience": "fitness enthusiasts",
        },
        {
            "product_name": "Artisan Coffee Beans",
            "key_features": "Single origin, small batch, dark roast",
            "target_audience": "coffee connoisseurs",
        },
    ]

    print("📊 SINGLE-OBJECTIVE baselines:")
    print("-" * 40)

    # Test baseline performance
    baseline_times = []
    baseline_urgency = []
    baseline_quality = []

    for test in test_cases:
        start_time = time.time()
        result = await writer(**test)
        end_time = time.time()

        latency = end_time - start_time
        description = result.outputs.get("description", "")
        urgency = Extractors.number(result.outputs.get("urgency_score"), default=5.0)
        quality_score = min(10.0, len(description.split()) / 3)  # Simple quality metric

        baseline_times.append(latency)
        baseline_urgency.append(urgency)
        baseline_quality.append(quality_score)

        print(f"Product: {test['product_name']}")
        print(f"  Quality: {quality_score:.1f} | Urgency: {urgency:.1f} | Time: {latency:.2f}s")

    avg_quality = sum(baseline_quality) / len(baseline_quality)
    avg_urgency = sum(baseline_urgency) / len(baseline_urgency)
    avg_time = sum(baseline_times) / len(baseline_times)

    print("\nBaseline averages:")
    print(f"  Quality: {avg_quality:.2f} | Urgency: {avg_urgency:.2f} | Time: {avg_time:.2f}s")

    # Define multiple competing objectives
    def quality_metric(pred, ref=None):
        """Higher quality descriptions (length, detail)."""
        description = pred.get("description", "")
        selling_points = pred.get("selling_points", "")

        # Reward longer, more detailed descriptions
        desc_score = min(1.0, len(description.split()) / 25)
        points_score = min(1.0, len(selling_points.split()) / 15)

        return (desc_score + points_score) / 2

    def speed_metric(pred, ref=None):
        """Shorter descriptions are faster to generate (proxy for speed)."""
        description = pred.get("description", "")
        # Inverse relationship: shorter = faster
        return max(0.1, 1.0 - (len(description.split()) / 50))

    def urgency_metric(pred, ref=None):
        """Higher urgency scores for marketing effectiveness."""
        urgency = Extractors.number(pred.get("urgency_score"), default=5.0)
        return urgency / 10.0

    # Multi-objective optimization
    print("\n🎯 MULTI-OBJECTIVE optimization...")

    optimizer = MultiObjectiveOptimizer(
        metrics={
            "quality": quality_metric,  # Want high quality
            "speed": speed_metric,  # Want fast generation
            "urgency": urgency_metric,  # Want compelling copy
        },
        weights={
            "quality": 0.5,  # Quality is most important
            "speed": 0.2,  # Speed matters for cost
            "urgency": 0.3,  # Urgency drives conversions
        },
        strategy="weighted",  # Try: weighted, pareto, constraint
        maintain_pareto=True,
        n_trials=15,  # Keep reasonable for demo
    )

    result = await optimizer.optimize(
        module=writer,
        dataset=train_data,
        param_space={
            "temperature": (0.3, 1.2),  # Creativity vs consistency
            "max_tokens": (50, 200),  # Length vs speed trade-off
        },
    )

    optimized_writer = result.optimized_module

    # Test optimized version
    print("\n📊 MULTI-OBJECTIVE optimized:")
    print("-" * 40)

    opt_times = []
    opt_urgency = []
    opt_quality = []

    for test in test_cases:
        start_time = time.time()
        result = await optimized_writer(**test)
        end_time = time.time()

        latency = end_time - start_time
        description = result.outputs.get("description", "")
        urgency = Extractors.number(result.outputs.get("urgency_score"), default=5.0)
        quality_score = min(10.0, len(description.split()) / 3)

        opt_times.append(latency)
        opt_urgency.append(urgency)
        opt_quality.append(quality_score)

        print(f"Product: {test['product_name']}")
        print(f"  Quality: {quality_score:.1f} | Urgency: {urgency:.1f} | Time: {latency:.2f}s")
        print(f"  Description: {description[:60]}...")

    opt_avg_quality = sum(opt_quality) / len(opt_quality)
    opt_avg_urgency = sum(opt_urgency) / len(opt_urgency)
    opt_avg_time = sum(opt_times) / len(opt_times)

    # Compare results
    print("\n" + "=" * 50)
    print("📈 MULTI-OBJECTIVE RESULTS:")
    print(
        f"Quality:  {avg_quality:.2f} → {opt_avg_quality:.2f} ({((opt_avg_quality - avg_quality) / avg_quality) * 100:+.1f}%)"
    )
    print(
        f"Urgency:  {avg_urgency:.2f} → {opt_avg_urgency:.2f} ({((opt_avg_urgency - avg_urgency) / avg_urgency) * 100:+.1f}%)"
    )
    print(
        f"Speed:    {avg_time:.2f}s → {opt_avg_time:.2f}s ({((opt_avg_time - avg_time) / avg_time) * 100:+.1f}%)"
    )

    # Show discovered trade-offs
    if hasattr(result, "metadata"):
        print("\n🎯 Optimization insights:")
        if "pareto_solutions" in result.metadata:
            print(f"  Pareto solutions found: {len(result.metadata['pareto_solutions'])}")
        if "trials_completed" in result.metadata:
            print(f"  Trials completed:       {result.metadata['trials_completed']}")

    # Show final hyperparameters
    if hasattr(optimized_writer, "config") and optimized_writer.config:
        print("\n⚙️ Optimal trade-off hyperparameters:")
        if "temperature" in optimized_writer.config:
            print(f"  Temperature:  {optimized_writer.config['temperature']:.2f}")
        if "max_tokens" in optimized_writer.config:
            print(f"  Max tokens:   {optimized_writer.config['max_tokens']}")

    print("\n✨ Key insight: Multi-objective optimization finds solutions")
    print("   that balance competing goals rather than maximizing just one,")
    print("   leading to more practical real-world performance.")


if __name__ == "__main__":
    asyncio.run(main())
