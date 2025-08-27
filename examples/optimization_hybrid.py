#!/usr/bin/env python3
"""HybridOptimizer: LogiLLM's Killer Feature.

This is LogiLLM's competitive advantage - simultaneous optimization of
prompts AND hyperparameters. DSPy fundamentally cannot do this.

The HybridOptimizer finds the optimal combination of:
- Prompt instructions and few-shot examples
- Model hyperparameters (temperature, top_p, etc.)
- Even format selection (Markdown, JSON, XML)

Key insight: The best prompts depend on hyperparameters and vice versa.
A creative temperature needs different prompting than a deterministic one.
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import HybridOptimizer
from logillm.providers import create_provider, register_provider


class CreativeWriting(Signature):
    """Generate creative text with specific constraints."""

    topic: str = InputField(desc="Topic to write about")
    style: str = InputField(desc="Writing style (formal, casual, poetic)")
    text: str = OutputField(desc="Generated creative text")
    creativity_score: float = OutputField(desc="Self-assessed creativity score from 0.0 to 1.0")


async def main():
    """Demonstrate simultaneous prompt + hyperparameter optimization."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== HybridOptimizer: LogiLLM's Killer Feature ===\n")

    # Use smaller model for demonstration
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    writer = Predict(CreativeWriting)

    # Training data with varied styles
    train_data = [
        {
            "inputs": {"topic": "sunrise", "style": "poetic"},
            "outputs": {
                "text": "Golden fingers reach across the awakening sky, painting dreams in amber hues",
                "creativity_score": "0.8",
            },
        },
        {
            "inputs": {"topic": "coffee", "style": "casual"},
            "outputs": {
                "text": "That first sip hits different when you really need it, you know?",
                "creativity_score": "0.6",
            },
        },
        {
            "inputs": {"topic": "technology", "style": "formal"},
            "outputs": {
                "text": "The rapid advancement of artificial intelligence presents both opportunities and challenges for modern society.",
                "creativity_score": "0.4",
            },
        },
        {
            "inputs": {"topic": "ocean", "style": "poetic"},
            "outputs": {
                "text": "Endless whispers of salt and time, where horizons dissolve into infinity",
                "creativity_score": "0.9",
            },
        },
    ]

    # Test cases
    test_cases = [
        {"topic": "mountain", "style": "poetic"},
        {"topic": "pizza", "style": "casual"},
        {"topic": "climate", "style": "formal"},
    ]

    # Baseline test
    print("📊 BASELINE (no optimization):")
    print("-" * 40)
    baseline_scores = []
    for test in test_cases:
        result = await writer(**test)
        score = Extractors.number(result.outputs.get("creativity_score"), default=0.5)
        baseline_scores.append(score)
        print(f"Topic: {test['topic']:<10} Style: {test['style']:<8} → Score: {score:.1f}")

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"\nBaseline avg: {baseline_avg:.2f}")

    # Define metric that values creativity
    def creativity_metric(pred, ref=None):
        """Higher creativity scores are better."""
        score = Extractors.number(pred.get("creativity_score"), default=0.5)
        return min(1.0, max(0.0, score))  # Already in 0-1 range

    # KILLER FEATURE: Simultaneous optimization
    print("\n🚀 OPTIMIZING prompts + hyperparameters simultaneously...")

    optimizer = HybridOptimizer(
        metric=creativity_metric,
        strategy="alternating",  # Try: alternating, joint, sequential
        optimize_format=False,  # Could also optimize format!
    )

    # Define hyperparameter search space
    param_space = {
        "temperature": (0.3, 1.5),  # Creative tasks need higher temp
        "top_p": (0.5, 1.0),  # Nucleus sampling threshold
    }

    result = await optimizer.optimize(module=writer, dataset=train_data, param_space=param_space)

    optimized_writer = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (prompts + hyperparameters):")
    print("-" * 40)
    optimized_scores = []
    for test in test_cases:
        result = await optimized_writer(**test)
        score = Extractors.number(result.outputs.get("creativity_score"), default=0.5)
        optimized_scores.append(score)
        print(f"Topic: {test['topic']:<10} Style: {test['style']:<8} → Score: {score:.1f}")

    optimized_avg = sum(optimized_scores) / len(optimized_scores)
    improvement = ((optimized_avg - baseline_avg) / baseline_avg) * 100

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline average:  {baseline_avg:.2f}")
    print(f"  Optimized average: {optimized_avg:.2f}")
    print(f"  Improvement:       {improvement:+.1f}%")

    # Show discovered hyperparameters
    if hasattr(optimized_writer, "config") and optimized_writer.config:
        print("\n🎯 Discovered optimal hyperparameters:")
        if "temperature" in optimized_writer.config:
            print(f"  Temperature: {optimized_writer.config['temperature']:.2f}")
        if "top_p" in optimized_writer.config:
            print(f"  Top-p:       {optimized_writer.config['top_p']:.2f}")

    # Show selected demos
    if optimized_writer.demo_manager and optimized_writer.demo_manager.demos:
        print(f"\n📚 Selected examples ({len(optimized_writer.demo_manager.demos)}):")
        for i, demo in enumerate(optimized_writer.demo_manager.demos[:2], 1):
            topic = demo.inputs.get("topic", "")
            style = demo.inputs.get("style", "")
            print(f"  {i}. {style} writing about '{topic}'")

    print("\n✨ Key insight: DSPy cannot do this! LogiLLM uniquely optimizes")
    print("   both prompts AND hyperparameters together, finding combinations")
    print("   that neither approach alone would discover.")


if __name__ == "__main__":
    asyncio.run(main())
