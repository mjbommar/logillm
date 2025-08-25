#!/usr/bin/env python3
"""Minimal math solving example.
For detailed version: examples/few_shot_math.py
"""

import asyncio
import re
from logillm.core.predict import Predict, ChainOfThought
from logillm.optimizers import BootstrapFewShot
from logillm.core.optimizers import Metric
from logillm.providers import create_provider, register_provider

# Setup
provider = create_provider("openai", model="gpt-4.1")
register_provider(provider, set_default=True)

# Metric
class MathAccuracy(Metric):
    def __call__(self, pred, ref, **kwargs):
        extract = lambda x: float(re.findall(r'-?\d+\.?\d*', str(x.get("answer", "")))[-1] or -1)
        return 1.0 if abs(extract(pred) - extract(ref)) < 0.01 else 0.0
    def name(self): return "math_accuracy"

# Data
train = [
    {"inputs": {"q": "If Sara has 23 apples and buys 17 more, how many does she have?"}, 
     "outputs": {"answer": "40"}},
    {"inputs": {"q": "A store has 156 items. They sell 89. How many are left?"}, 
     "outputs": {"answer": "67"}},
    {"inputs": {"q": "Tom runs 8 miles a day for 12 days. Total miles?"}, 
     "outputs": {"answer": "96"}},
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
        ans = re.findall(r'-?\d+\.?\d*', result.outputs.get("answer", ""))[-1] if re.findall(r'-?\d+\.?\d*', result.outputs.get("answer", "")) else "?"
        print(f"  {ans} (expected {t['a']})")
    
    # Optimize
    optimizer = BootstrapFewShot(metric=MathAccuracy(), max_bootstrapped_demos=2)
    optimized = (await optimizer.optimize(solver, dataset=train)).optimized_module
    
    print("\nAfter optimization:")
    for t in test:
        result = await optimized(q=t["q"]) 
        ans = re.findall(r'-?\d+\.?\d*', result.outputs.get("answer", ""))[-1] if re.findall(r'-?\d+\.?\d*', result.outputs.get("answer", "")) else "?"
        print(f"  {ans} (expected {t['a']})")

asyncio.run(main())