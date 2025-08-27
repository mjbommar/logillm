#!/usr/bin/env python3
"""Few-Shot Learning for Math Word Problems.

This example demonstrates how few-shot learning improves math problem solving
by teaching the model to show step-by-step reasoning. Unlike classification
tasks, this shows how few-shot can teach structured problem-solving approaches.

Key features:
- Multiple output fields (reasoning + answer)
- Numerical extraction and validation
- Teaching reasoning patterns, not just answers
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


class MathProblem(Signature):
    """Solve math word problems step by step."""

    problem: str = InputField(desc="Math word problem to solve")
    reasoning: str = OutputField(desc="Step-by-step solution")
    answer: str = OutputField(desc="Final numerical answer only")


async def main():
    """Demonstrate few-shot learning on math problems."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== Few-Shot Learning for Math Problems ===\n")

    # Use smaller model for realistic baseline
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    solver = Predict(MathProblem)

    # Test problems - require multi-step reasoning
    test_problems = [
        ("If you have 23 apples and buy 17 more, how many do you have?", 40),
        ("A shirt costs $25. With a 20% discount, what's the price?", 20),
        ("You read 30 pages per day. How many pages in 5 days?", 150),
        ("A recipe needs 3 eggs. How many eggs for 4 batches?", 12),
    ]

    # Test WITHOUT examples
    print("📊 WITHOUT examples:")
    print("-" * 40)
    baseline_correct = 0

    for problem, expected in test_problems:
        result = await solver(problem=problem)
        # Use the new Extractors.number with a clear invalid default
        answer = Extractors.number(result.outputs.get("answer"), default=-999)
        correct = abs(answer - expected) < 0.01 if answer != -999 else False
        baseline_correct += correct
        print(f"  {'✓' if correct else '✗'} Got {answer}, expected {expected}")
        print(f"    Problem: {problem[:50]}...")

    baseline_acc = baseline_correct / len(test_problems)
    print(f"\nBaseline: {baseline_acc:.0%} ({baseline_correct}/{len(test_problems)})")

    # Training examples - teach step-by-step format
    training = [
        {
            "inputs": {"problem": "You have 15 marbles and get 8 more. How many total?"},
            "outputs": {
                "reasoning": "Start: 15 marbles. Add: 8 marbles. Total: 15 + 8 = 23",
                "answer": "23",
            },
        },
        {
            "inputs": {"problem": "A pizza has 8 slices. You eat 3. How many left?"},
            "outputs": {
                "reasoning": "Start: 8 slices. Eat: 3 slices. Left: 8 - 3 = 5",
                "answer": "5",
            },
        },
        {
            "inputs": {"problem": "Books cost $7 each. How much for 6 books?"},
            "outputs": {
                "reasoning": "Price per book: $7. Number: 6. Total: 7 × 6 = 42",
                "answer": "42",
            },
        },
        {
            "inputs": {"problem": "You save $10 per week. How much in 4 weeks?"},
            "outputs": {"reasoning": "Per week: $10. Weeks: 4. Total: 10 × 4 = 40", "answer": "40"},
        },
    ]

    # Optimize with few-shot
    print("\n🎓 Training with examples...")

    def math_metric(pred, ref):
        """Check if numerical answers match."""
        pred_num = Extractors.number(pred.get("answer"), default=-999)
        ref_num = Extractors.number(ref.get("answer"), default=-999)
        if pred_num == -999 or ref_num == -999:
            return 0.0
        return 1.0 if abs(pred_num - ref_num) < 0.01 else 0.0

    optimizer = BootstrapFewShot(metric=math_metric, max_demos=2, max_rounds=1)
    result = await optimizer.optimize(solver, dataset=training)
    improved_solver = result.optimized_module

    # Test WITH examples
    print("\n📊 WITH examples:")
    print("-" * 40)
    improved_correct = 0

    for problem, expected in test_problems:
        result = await improved_solver(problem=problem)
        answer = Extractors.number(result.outputs.get("answer"), default=-999)
        reasoning = result.outputs.get("reasoning") or ""
        correct = abs(answer - expected) < 0.01 if answer != -999 else False
        improved_correct += correct
        print(f"  {'✓' if correct else '✗'} Got {answer}, expected {expected}")
        if reasoning:
            print(f"    Reasoning: {reasoning[:80]}...")

    improved_acc = improved_correct / len(test_problems)
    improvement = improved_acc - baseline_acc

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Without examples: {baseline_acc:6.0%} ({baseline_correct}/{len(test_problems)})")
    print(f"  With examples:    {improved_acc:6.0%} ({improved_correct}/{len(test_problems)})")
    print(f"  Improvement:      {improvement:+6.0%}")

    # Show selected examples
    if improved_solver.demo_manager.demos:
        print(f"\n📚 Examples used ({len(improved_solver.demo_manager.demos)}):")
        for i, demo in enumerate(improved_solver.demo_manager.demos[:2], 1):
            prob = demo.inputs.get("problem", "")[:40]
            ans = demo.outputs.get("answer", "")
            print(f"  {i}. '{prob}...' → {ans}")

    print("\n✨ Key insight: Teaching step-by-step reasoning dramatically improves math accuracy!")


if __name__ == "__main__":
    asyncio.run(main())
