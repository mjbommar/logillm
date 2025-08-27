#!/usr/bin/env python3
"""Minimal math solving example.
For detailed version: examples/few_shot_math.py
"""

import asyncio

from logillm.core.extractors import Extractors
from logillm.core.optimizers import Metric
from logillm.core.predict import ChainOfThought
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider

# Setup
provider = create_provider("openai", model="gpt-4.1")
register_provider(provider, set_default=True)


# Metric using the new Extractors
class MathAccuracy(Metric):
    def __call__(self, pred, ref, **kwargs):
        pred_num = Extractors.number(pred.get("answer"), default=-999)
        ref_num = Extractors.number(ref.get("answer"), default=-999)
        if pred_num == -999 or ref_num == -999:
            return 0.0
        return 1.0 if abs(pred_num - ref_num) < 0.01 else 0.0

    def name(self):
        return "math_accuracy"


# Data
train = [
    {
        "inputs": {"q": "If Sara has 23 apples and buys 17 more, how many does she have?"},
        "outputs": {"answer": "40"},
    },
    {
        "inputs": {"q": "A store has 156 items. They sell 89. How many are left?"},
        "outputs": {"answer": "67"},
    },
    {
        "inputs": {"q": "Tom runs 8 miles a day for 12 days. Total miles?"},
        "outputs": {"answer": "96"},
    },
]

test = [
    {"q": "Jane has 145 marbles and gives away 67. How many left?", "a": "78"},
    {"q": "A bus has 8 rows of 6 seats. How many seats total?", "a": "48"},
]


async def main():
    # Baseline
    solver = ChainOfThought("q -> reasoning, answer")

    print("Before optimization:")
    for t in test:
        result = await solver(q=t["q"])
        ans = Extractors.number(result.outputs.get("answer"), default=-999)
        ans_str = str(int(ans)) if ans != -999 else "?"
        print(f"  {ans_str} (expected {t['a']})")

    # Optimize
    optimizer = BootstrapFewShot(metric=MathAccuracy(), max_bootstrapped_demos=2)
    optimized = (await optimizer.optimize(solver, dataset=train)).optimized_module

    print("\nAfter optimization:")
    for t in test:
        result = await optimized(q=t["q"])
        ans = Extractors.number(result.outputs.get("answer"), default=-999)
        ans_str = str(int(ans)) if ans != -999 else "?"
        print(f"  {ans_str} (expected {t['a']})")


asyncio.run(main())
