#!/usr/bin/env python3
"""
Example: Debugging prompts in LogiLLM

This example shows how to view the actual prompts being sent to the LLM,
which is essential for debugging and understanding how LogiLLM works.
"""
import asyncio

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


async def main():
    # Setup provider
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)

    print("LogiLLM Prompt Debugging Example")
    print("=" * 60)

    # Method 1: Enable debug mode in constructor
    print("\nMethod 1: Enable debug in constructor")
    print("-" * 40)

    predictor = Predict("text -> sentiment: str, confidence: float", debug=True)
    result = await predictor(text="I love this framework!")

    print(f"Result: sentiment={result.sentiment}, confidence={result.confidence}")

    # Access the prompt that was sent
    if result.prompt:
        print("\nPrompt information:")
        print(f"  Adapter: {result.prompt['adapter']}")
        print(f"  Provider: {result.prompt['provider']}")
        print(f"  Model: {result.prompt['model']}")
        print(f"  Number of messages: {len(result.prompt['messages'])}")

        # Show the actual prompt
        print("\nActual prompt sent to LLM:")
        for i, msg in enumerate(result.prompt['messages'], 1):
            print(f"\nMessage {i}:")
            print(f"  Role: {msg.get('role')}")
            print(f"  Content: {msg.get('content')[:200]}...")  # First 200 chars

    # Method 2: Toggle debug mode with methods
    print("\n" + "=" * 60)
    print("Method 2: Toggle debug mode with methods")
    print("-" * 40)

    predictor2 = Predict("question -> answer")

    # Initially off
    result = await predictor2(question="What is 2+2?")
    print(f"Debug off - Prompt captured: {result.prompt is not None}")

    # Turn on debug
    predictor2.enable_debug_mode()
    result = await predictor2(question="What is 5+3?")
    print(f"Debug on - Prompt captured: {result.prompt is not None}")

    # Turn off debug
    predictor2.disable_debug_mode()
    result = await predictor2(question="What is 10-4?")
    print(f"Debug off again - Prompt captured: {result.prompt is not None}")

    # Method 3: Use environment variable
    print("\n" + "=" * 60)
    print("Method 3: Environment variable")
    print("-" * 40)
    print("\nSet LOGILLM_DEBUG=1 to enable debug globally:")
    print("  export LOGILLM_DEBUG=1")
    print("  python your_app.py")
    print("\nThis enables debug for all modules by default.")

    # Demonstration with few-shot examples
    print("\n" + "=" * 60)
    print("Bonus: Debug with few-shot examples")
    print("-" * 40)

    qa = Predict("question -> answer", debug=True)

    # Add some demonstrations
    qa.add_demo({"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}})
    qa.add_demo({"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}})

    result = await qa(question="What is the capital of Spain?")
    print(f"Answer: {result.answer}")

    if result.prompt:
        print("\nWith demonstrations:")
        print(f"  Number of demos included: {result.prompt['demos_count']}")

        # The prompt will now include the demonstrations
        content = result.prompt['messages'][0]['content']
        if "2+2" in content and "Paris" in content:
            print("  ✓ Demonstrations are included in the prompt")

    print("\n" + "=" * 60)
    print("Summary:")
    print("- Use debug=True in constructor for specific modules")
    print("- Use enable_debug_mode()/disable_debug_mode() to toggle")
    print("- Use LOGILLM_DEBUG=1 environment variable for global debug")
    print("- Access prompts via result.prompt when debugging")
    print("- Prompts include messages, adapter info, and demo count")


if __name__ == "__main__":
    asyncio.run(main())
