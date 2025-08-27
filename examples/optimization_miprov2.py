#!/usr/bin/env python3
"""MIPROv2: Multi-Instruction Proposal Optimizer.

MIPROv2 is DSPy's flagship optimizer, combining bootstrap few-shot,
instruction proposals, and Bayesian optimization. It explores a large
space of instructions and examples to find optimal combinations.

Key features:
- Three modes: light, medium, heavy (increasing compute)
- Grounded instruction proposals based on data
- Bayesian optimization over instruction/demo combinations
- State-of-the-art performance on many tasks
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import ChainOfThought
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import MIPROv2Optimizer
from logillm.providers import create_provider, register_provider


class LogicalReasoning(Signature):
    """Solve logical reasoning problems."""

    premise: str = InputField(desc="Given facts or conditions")
    question: str = InputField(desc="Question to answer")
    answer: str = OutputField(desc="Logical answer")
    valid: bool = OutputField(desc="Is the reasoning valid?")


async def main():
    """Demonstrate MIPROv2's sophisticated optimization."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== MIPROv2: Multi-Instruction Proposal Optimization ===\n")

    # Use smaller model to show improvement
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    # Use ChainOfThought for reasoning tasks
    reasoner = ChainOfThought(LogicalReasoning)

    # Training data with logical puzzles
    train_data = [
        {
            "inputs": {
                "premise": "All birds can fly. Penguins are birds.",
                "question": "Can penguins fly?",
            },
            "outputs": {
                "answer": "According to the premises, yes (though factually incorrect)",
                "valid": "True",
            },
        },
        {
            "inputs": {
                "premise": "If it rains, the ground is wet. The ground is wet.",
                "question": "Did it rain?",
            },
            "outputs": {
                "answer": "Not necessarily - other things can make ground wet",
                "valid": "False",
            },
        },
        {
            "inputs": {
                "premise": "All cats are mammals. Some mammals are dogs.",
                "question": "Are some cats dogs?",
            },
            "outputs": {"answer": "No, cats and dogs are different species", "valid": "False"},
        },
        {
            "inputs": {
                "premise": "John is taller than Mary. Mary is taller than Bob.",
                "question": "Is John taller than Bob?",
            },
            "outputs": {"answer": "Yes, by transitivity", "valid": "True"},
        },
    ]

    # Test cases
    test_cases = [
        {
            "premise": "All roses are flowers. Some flowers are red.",
            "question": "Are all roses red?",
        },
        {
            "premise": "If you study, you pass. Tom passed.",
            "question": "Did Tom study?",
        },
        {
            "premise": "No fish can walk. Salmon are fish.",
            "question": "Can salmon walk?",
        },
    ]

    # Baseline test
    print("📊 BASELINE (no optimization):")
    print("-" * 40)
    baseline_valid = 0

    for test in test_cases:
        result = await reasoner(**test)
        answer = result.outputs.get("answer", "Unknown")
        valid = Extractors.boolean(result.outputs.get("valid"), default=False)
        baseline_valid += int(valid)
        print(f"Q: {test['question'][:40]}...")
        print(f"  Answer: {answer[:60]}...")
        print(f"  Valid reasoning: {valid}")

    print(f"\nBaseline valid: {baseline_valid}/{len(test_cases)}")

    # Define reasoning quality metric
    def reasoning_metric(pred, ref):
        """Score based on answer quality and validity."""
        answer = pred.get("answer", "")
        pred_valid = Extractors.boolean(pred.get("valid"), default=False)
        ref_valid = Extractors.boolean(ref.get("valid"), default=False)

        score = 0.0
        if len(answer) > 10:  # Has substantial answer
            score += 0.3
        if "because" in answer.lower() or "therefore" in answer.lower():
            score += 0.2  # Shows reasoning
        if pred_valid == ref_valid:  # Correct validity assessment
            score += 0.5

        return score

    # Optimize with MIPROv2
    print("\n🚀 OPTIMIZING with MIPROv2...")
    print("Mode: light (fast demonstration)")

    optimizer = MIPROv2Optimizer(
        metric=reasoning_metric,
        mode="light",  # light, medium, or heavy
    )

    result = await optimizer.optimize(module=reasoner, dataset=train_data, valset=train_data)

    optimized_reasoner = result.optimized_module

    # Test optimized version
    print("\n📊 OPTIMIZED (with MIPROv2):")
    print("-" * 40)
    optimized_valid = 0

    for test in test_cases:
        result = await optimized_reasoner(**test)
        answer = result.outputs.get("answer", "Unknown")
        reasoning = result.outputs.get("reasoning", "")  # ChainOfThought adds this
        valid = Extractors.boolean(result.outputs.get("valid"), default=False)
        optimized_valid += int(valid)
        print(f"Q: {test['question'][:40]}...")
        print(f"  Answer: {answer[:60]}...")
        if reasoning:
            print(f"  Reasoning: {reasoning[:60]}...")
        print(f"  Valid reasoning: {valid}")

    improvement = optimized_valid - baseline_valid

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline valid:  {baseline_valid}/{len(test_cases)}")
    print(f"  Optimized valid: {optimized_valid}/{len(test_cases)}")
    print(f"  Improvement:     {improvement:+d} more correct")

    # Show optimization details
    if hasattr(result, "metadata"):
        meta = result.metadata
        print("\n🔍 MIPROv2 exploration:")
        if "proposals_generated" in meta:
            print(f"  Instructions proposed: {meta['proposals_generated']}")
        if "trials_run" in meta:
            print(f"  Trials evaluated:      {meta['trials_run']}")
        if "mode" in meta:
            print(f"  Optimization mode:     {meta['mode']}")

    # Show selected instruction
    if hasattr(optimized_reasoner, "signature") and hasattr(
        optimized_reasoner.signature, "__doc__"
    ):
        print("\n📝 MIPROv2 selected instruction:")
        doc = optimized_reasoner.signature.__doc__
        if doc:
            print(f'  "{doc[:100]}..."')

    print("\n✨ Key insight: MIPROv2 uses sophisticated Bayesian optimization")
    print("   to explore a large space of instructions and demonstrations,")
    print("   finding combinations that simpler methods would miss.")


if __name__ == "__main__":
    asyncio.run(main())
