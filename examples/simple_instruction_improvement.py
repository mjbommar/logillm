#!/usr/bin/env python3
"""Simple example showing how instruction changes affect performance.

This demonstrates the impact of good vs bad instructions on the same task.
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


async def main():
    """Show how instruction changes affect task performance."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== Impact of Instructions on LLM Performance ===\n")

    # Use smaller model
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    # Test data: extract the main topic from sentences
    test_sentences = [
        "The weather in Paris is beautiful today.",
        "Python programming is becoming more popular.",
        "The stock market closed higher yesterday.",
        "Scientists discovered a new species of butterfly.",
    ]

    expected_topics = ["weather", "programming", "finance", "science"]

    # Test 1: Vague instruction
    print("📊 Test 1: VAGUE instruction")
    print("Instruction: 'Process this text.'\n")

    vague_module = Predict("text -> topic")
    vague_module.signature.instructions = "Process this text."

    vague_correct = 0
    for sentence, expected in zip(test_sentences, expected_topics):
        result = await vague_module(text=sentence)
        topic = result.outputs.get("topic", "").lower()
        is_correct = expected in topic or topic in expected
        vague_correct += is_correct
        symbol = "✓" if is_correct else "✗"
        print(f"{symbol} '{sentence[:40]}...'")
        print(f"   → Topic: {topic}")

    print(f"\nVague instruction accuracy: {vague_correct}/{len(test_sentences)}")

    # Test 2: Clear instruction
    print("\n" + "=" * 50)
    print("\n📊 Test 2: CLEAR instruction")
    print(
        "Instruction: 'Extract the main topic category from the text. Choose from: weather, programming, finance, science, sports, or other.'\n"
    )

    clear_module = Predict("text -> topic")
    clear_module.signature.instructions = (
        "Extract the main topic category from the text. "
        "Choose from: weather, programming, finance, science, sports, or other."
    )

    clear_correct = 0
    for sentence, expected in zip(test_sentences, expected_topics):
        result = await clear_module(text=sentence)
        topic = result.outputs.get("topic", "").lower()
        is_correct = expected in topic or topic in expected
        clear_correct += is_correct
        symbol = "✓" if is_correct else "✗"
        print(f"{symbol} '{sentence[:40]}...'")
        print(f"   → Topic: {topic}")

    print(f"\nClear instruction accuracy: {clear_correct}/{len(test_sentences)}")

    # Test 3: Step-by-step instruction
    print("\n" + "=" * 50)
    print("\n📊 Test 3: STEP-BY-STEP instruction")
    print(
        "Instruction: 'Follow these steps: 1) Read the text carefully. 2) Identify the main subject being discussed. 3) Categorize it as: weather, programming, finance, science, sports, or other. 4) Return only the category name.'\n"
    )

    stepwise_module = Predict("text -> topic")
    stepwise_module.signature.instructions = (
        "Follow these steps: "
        "1) Read the text carefully. "
        "2) Identify the main subject being discussed. "
        "3) Categorize it as: weather, programming, finance, science, sports, or other. "
        "4) Return only the category name."
    )

    stepwise_correct = 0
    for sentence, expected in zip(test_sentences, expected_topics):
        result = await stepwise_module(text=sentence)
        topic = result.outputs.get("topic", "").lower()
        is_correct = expected in topic or topic in expected
        stepwise_correct += is_correct
        symbol = "✓" if is_correct else "✗"
        print(f"{symbol} '{sentence[:40]}...'")
        print(f"   → Topic: {topic}")

    print(f"\nStep-by-step instruction accuracy: {stepwise_correct}/{len(test_sentences)}")

    # Summary
    print("\n" + "=" * 50)
    print("📈 RESULTS SUMMARY:\n")
    print(
        f"  Vague instruction:      {vague_correct}/{len(test_sentences)} correct ({vague_correct * 25}%)"
    )
    print(
        f"  Clear instruction:      {clear_correct}/{len(test_sentences)} correct ({clear_correct * 25}%)"
    )
    print(
        f"  Step-by-step:          {stepwise_correct}/{len(test_sentences)} correct ({stepwise_correct * 25}%)"
    )

    improvement = max(clear_correct, stepwise_correct) - vague_correct
    if improvement > 0:
        print(f"\n✨ Better instructions improved accuracy by {improvement} examples!")

    print("\n💡 KEY INSIGHT: The same model performs very differently based on")
    print("   how clearly and specifically we phrase our instructions.")
    print("\n   Optimizers like COPRO, MIPROv2, and InstructionOptimizer")
    print("   automatically discover better instructions through trial and error,")
    print("   often finding phrasings that humans wouldn't think of.")


if __name__ == "__main__":
    asyncio.run(main())
