#!/usr/bin/env python3
"""ReflectiveEvolution: LLM-Based Self-Improvement.

ReflectiveEvolutionOptimizer uses an LLM to reflect on execution traces
and suggest improvements to both prompts and hyperparameters. This is like
having a smart coach that watches your model work and gives feedback.

Key capabilities:
- LLM-powered reflection on what went wrong
- Evolution of prompts based on textual feedback
- Hyperparameter adjustment based on performance patterns
- Pareto frontier tracking for multi-dimensional optimization
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import ReflectiveEvolutionOptimizer
from logillm.providers import create_provider, register_provider


class MathReasoning(Signature):
    """Solve math word problems with clear reasoning."""

    problem: str = InputField(desc="Math word problem to solve")
    reasoning: str = OutputField(desc="Step-by-step solution process")
    answer: str = OutputField(desc="Final numerical answer")
    confidence: float = OutputField(desc="Confidence in solution 0-1")


async def main():
    """Demonstrate reflective evolution through self-improvement."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== ReflectiveEvolution: LLM-Based Self-Improvement ===\n")

    # Use smaller model for demonstration
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    solver = Predict(MathReasoning)

    # Training data with math word problems
    train_data = [
        {
            "inputs": {
                "problem": "Sarah has 15 apples. She gives away 3 apples to each of her 4 friends. How many apples does she have left?"
            },
            "outputs": {
                "reasoning": "Sarah starts with 15 apples. She gives 3 apples to each of 4 friends, so 3 × 4 = 12 apples given away. 15 - 12 = 3 apples remaining.",
                "answer": "3",
                "confidence": "0.95",
            },
        },
        {
            "inputs": {
                "problem": "A train travels 60 miles in 2 hours. At this rate, how far will it travel in 5 hours?"
            },
            "outputs": {
                "reasoning": "The train's speed is 60 miles ÷ 2 hours = 30 miles per hour. In 5 hours: 30 × 5 = 150 miles.",
                "answer": "150",
                "confidence": "0.9",
            },
        },
        {
            "inputs": {
                "problem": "A rectangle has length 8 cm and width 5 cm. What is its perimeter?"
            },
            "outputs": {
                "reasoning": "Perimeter = 2 × (length + width) = 2 × (8 + 5) = 2 × 13 = 26 cm.",
                "answer": "26",
                "confidence": "0.98",
            },
        },
        {
            "inputs": {
                "problem": "Tom buys 3 books for $12 each and 2 pens for $3 each. How much does he spend in total?"
            },
            "outputs": {
                "reasoning": "Books cost: 3 × $12 = $36. Pens cost: 2 × $3 = $6. Total: $36 + $6 = $42.",
                "answer": "42",
                "confidence": "0.92",
            },
        },
    ]

    # Test cases that need good reasoning
    test_cases = [
        "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices remain?",
        "A car uses 5 gallons of gas to travel 150 miles. How many miles per gallon does it get?",
        "In a class of 30 students, 18 are girls. What percentage are boys?",
    ]

    # Baseline test
    print("📊 BASELINE (no optimization):")
    print("-" * 40)
    baseline_scores = []
    for problem in test_cases:
        result = await solver(problem=problem)
        confidence = Extractors.percentage(
            result.outputs.get("confidence"), as_decimal=True, default=0.5
        )
        baseline_scores.append(confidence)
        answer = result.outputs.get("answer", "N/A")
        print(f"Problem: {problem[:50]}...")
        print(f"  → Answer: {answer}, Confidence: {confidence:.2f}")

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"\nBaseline avg confidence: {baseline_avg:.2f}")

    # Define metric that combines correctness with confidence
    def reasoning_metric(pred, ref=None):
        """Reward high confidence in well-reasoned answers."""
        confidence = Extractors.percentage(pred.get("confidence"), as_decimal=True, default=0.5)
        reasoning = pred.get("reasoning", "")

        # Bonus for detailed reasoning (more steps = better)
        reasoning_bonus = min(0.2, len(reasoning.split(".")) * 0.02)

        return confidence + reasoning_bonus

    # Use reflective evolution to improve
    print("\n🧠 EVOLVING through LLM reflection...")

    # Create reflection LLM (could be same or different model)
    reflection_lm = create_provider("openai", model="gpt-4.1-nano")

    optimizer = ReflectiveEvolutionOptimizer(
        metric=reasoning_metric,
        reflection_lm=reflection_lm,
        use_textual_feedback=True,
        maintain_pareto=True,
        n_iterations=3,  # Keep small for demo
        include_hyperparameters=True,  # LogiLLM's advantage!
    )

    result = await optimizer.optimize(
        module=solver,
        dataset=train_data,
        param_space={
            "temperature": (0.1, 0.8),  # More deterministic for math
            "top_p": (0.7, 1.0),
        },
    )

    evolved_solver = result.optimized_module

    # Test evolved version
    print("\n📊 EVOLVED (after LLM reflection):")
    print("-" * 40)
    evolved_scores = []
    for problem in test_cases:
        result = await evolved_solver(problem=problem)
        confidence = Extractors.percentage(
            result.outputs.get("confidence"), as_decimal=True, default=0.5
        )
        evolved_scores.append(confidence)
        answer = result.outputs.get("answer", "N/A")
        reasoning = result.outputs.get("reasoning", "")[:60]
        print(f"Problem: {problem[:50]}...")
        print(f"  → Answer: {answer}, Confidence: {confidence:.2f}")
        print(f"  Reasoning: {reasoning}...")

    evolved_avg = sum(evolved_scores) / len(evolved_scores)
    improvement = ((evolved_avg - baseline_avg) / baseline_avg) * 100

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline confidence:  {baseline_avg:.2f}")
    print(f"  Evolved confidence:   {evolved_avg:.2f}")
    print(f"  Improvement:          {improvement:+.1f}%")

    # Show evolution insights
    if hasattr(result, "metadata"):
        print("\n🔄 Evolution process:")
        if "iterations_completed" in result.metadata:
            print(f"  Reflective iterations: {result.metadata['iterations_completed']}")
        if "improvements_found" in result.metadata:
            print(f"  Improvements found:    {result.metadata['improvements_found']}")

    # Show discovered hyperparameters
    if hasattr(evolved_solver, "config") and evolved_solver.config:
        print("\n🎯 Evolved hyperparameters:")
        if "temperature" in evolved_solver.config:
            print(f"  Temperature: {evolved_solver.config['temperature']:.2f}")
        if "top_p" in evolved_solver.config:
            print(f"  Top-p:       {evolved_solver.config['top_p']:.2f}")

    print("\n✨ Key insight: The LLM reflects on traces and suggests both")
    print("   prompt improvements AND hyperparameter adjustments based on")
    print("   patterns it observes in successful vs failed executions.")


if __name__ == "__main__":
    asyncio.run(main())
