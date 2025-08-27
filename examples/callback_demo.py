#!/usr/bin/env python3
"""Demonstration of LogiLLM's callback system for monitoring and debugging."""

import asyncio

from logillm.core.callbacks import AbstractCallback, CallbackManager
from logillm.core.jsonl_callback import register_jsonl_logger
from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


class SimpleMonitor(AbstractCallback):
    """Simple callback that counts module executions."""

    def __init__(self):
        self.call_count = 0
        self.total_duration = 0.0

    async def on_module_end(self, event):
        self.call_count += 1
        if event.duration:
            self.total_duration += event.duration
        print(f"  ✓ Module {event.module.__class__.__name__} completed in {event.duration:.2f}s")


async def main():
    """Demonstrate callback system features."""
    print("=== LogiLLM Callback System Demo ===\n")

    # Setup provider
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)

    # Get callback manager
    manager = CallbackManager()

    # 1. Simple monitoring callback
    print("1. Simple Monitoring Callback")
    print("-" * 30)

    monitor = SimpleMonitor()
    manager.register(monitor)

    # Create and use module
    qa = Predict("question -> answer")
    result = await qa(question="What is the capital of France?")
    print(f"Answer: {result.outputs['answer']}")
    print(f"Stats: {monitor.call_count} calls, {monitor.total_duration:.2f}s total\n")

    # 2. JSONL logging
    print("2. JSONL Logging")
    print("-" * 30)

    # Clear previous callbacks
    manager.clear()

    # Register JSONL logger
    callback_id = register_jsonl_logger(
        "demo_execution.jsonl", include_module_events=True, include_provider_events=True
    )
    print(f"Logging to: demo_execution.jsonl (ID: {callback_id})")

    # Run multiple queries
    questions = ["What is AI?", "What is machine learning?", "What is deep learning?"]

    for q in questions:
        result = await qa(question=q)
        print(f"Q: {q}")
        print(f"A: {result.outputs['answer'][:50]}...\n")

    # 3. Disable callbacks for performance
    print("3. Performance Mode (Callbacks Disabled)")
    print("-" * 30)

    manager.disable()
    print("Callbacks disabled for maximum performance")

    result = await qa(question="What is Python?")
    print(f"Answer: {result.outputs['answer'][:50]}...")

    # Re-enable
    manager.enable()
    print("\nCallbacks re-enabled")

    # 4. Custom event filtering
    print("\n4. Custom Callback with Priority")
    print("-" * 30)

    from logillm.core.callbacks import Priority

    class PriorityMonitor(AbstractCallback):
        @property
        def priority(self):
            return Priority.HIGH

        async def on_module_start(self, event):
            print(f"  [HIGH PRIORITY] Starting {event.module.__class__.__name__}")

    manager.clear()
    manager.register(PriorityMonitor())
    manager.register(SimpleMonitor())  # Normal priority

    await qa(question="Test priority order")

    print("\n✅ Demo complete!")
    print(f"   Total executions: {monitor.call_count}")
    print(f"   Total time: {monitor.total_duration:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
