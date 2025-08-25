#!/usr/bin/env python3
"""Minimal few-shot learning demonstration.
For detailed tutorial: examples/few_shot.py
"""

import asyncio
from logillm.core.predict import Predict
from logillm.optimizers import BootstrapFewShot
from logillm.core.optimizers import AccuracyMetric
from logillm.providers import create_provider, register_provider

# Training data
data = [
    {"inputs": {"text": "I hate this!"}, "outputs": {"sentiment": "negative"}},
    {"inputs": {"text": "This is amazing!"}, "outputs": {"sentiment": "positive"}},
    {"inputs": {"text": "Terrible service"}, "outputs": {"sentiment": "negative"}},
    {"inputs": {"text": "Love it!"}, "outputs": {"sentiment": "positive"}},
    {"inputs": {"text": "Not good"}, "outputs": {"sentiment": "negative"}},
    {"inputs": {"text": "Excellent!"}, "outputs": {"sentiment": "positive"}},
]

async def main():
    # Setup
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Create classifier
    classifier = Predict("text -> sentiment")
    
    # Test before optimization
    print("Before optimization:")
    result = await classifier(text="This is awful")
    print(f"  'This is awful' -> {result.outputs['sentiment']}")
    
    # Optimize with few-shot learning
    optimizer = BootstrapFewShot(
        metric=AccuracyMetric(key="sentiment"),
        max_bootstrapped_demos=3
    )
    optimized = (await optimizer.optimize(classifier, dataset=data)).optimized_module
    
    # Test after optimization
    print("\nAfter optimization (with examples):")
    result = await optimized(text="This is awful")
    print(f"  'This is awful' -> {result.outputs['sentiment']}")
    
    # Show what examples were selected
    if hasattr(optimized, 'demo_manager'):
        print(f"\nUsing {len(optimized.demo_manager.demos)} few-shot examples")

asyncio.run(main())