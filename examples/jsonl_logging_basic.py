#!/usr/bin/env python3
"""Basic JSONL logging example - simple and easy to understand.

This example demonstrates the simplest way to use JSONL logging
to track optimization progress and results.
"""

import asyncio
import json
from pathlib import Path

from logillm.core.jsonl_logger import OptimizationLogger
from logillm.core.predict import Predict
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot
from logillm.providers import register_provider
from logillm.providers.mock import MockProvider


async def main():
    """Run basic JSONL logging demonstration."""

    print("=== Basic JSONL Logging Example ===\n")

    # Setup mock provider for demonstration
    provider = MockProvider(responses=["Paris", "London", "Berlin", "Rome", "Madrid"])
    register_provider(provider, set_default=True)

    # Create a simple QA module
    qa_module = Predict("question -> answer")

    # Training data
    dataset = [
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {"inputs": {"question": "What is the capital of UK?"}, "outputs": {"answer": "London"}},
        {
            "inputs": {"question": "What is the capital of Germany?"},
            "outputs": {"answer": "Berlin"},
        },
    ]

    # Validation data
    validation = [
        {"inputs": {"question": "What is the capital of Italy?"}, "outputs": {"answer": "Rome"}}
    ]

    # Simple metric
    def accuracy(pred, ref):
        """Check if answer matches."""
        pred_text = pred.get("answer", "").lower()
        ref_text = ref.get("answer", "").lower()
        return 1.0 if ref_text in pred_text else 0.0

    # Create optimizer
    optimizer = BootstrapFewShot(metric=accuracy, max_demos=2, max_rounds=2)

    # Setup JSONL logger
    log_path = Path("optimization_log.jsonl")
    logger = OptimizationLogger(filepath=str(log_path))

    print(f"Starting optimization with logging to {log_path}\n")

    # Run optimization
    result = await logger.log_optimization(
        optimizer=optimizer, module=qa_module, dataset=dataset, validation_set=validation
    )

    print("✅ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")
    print(f"   Time: {result.optimization_time:.2f}s")

    # Read and display key events from the log
    print("\n📋 Key Events from JSONL Log:")
    print("-" * 40)

    with open(log_path) as f:
        events = [json.loads(line) for line in f]

    for event in events:
        event_type = event.get("event_type")

        if event_type == "optimization_start":
            print(f"Started: {event['timestamp']}")
            print(f"Dataset size: {event['dataset_size']}")

        elif event_type == "evaluation_end":
            print(f"Evaluation score: {event['score']:.2%}")

        elif event_type == "optimization_end":
            print(f"Finished: {event['timestamp']}")
            print(f"Final score: {event['best_score']:.2%}")
            print(f"Total time: {event['optimization_time']:.2f}s")

    print(f"\n✨ Full log saved to '{log_path}'")


if __name__ == "__main__":
    asyncio.run(main())
