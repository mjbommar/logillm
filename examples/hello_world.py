#!/usr/bin/env python3
"""Hello World example for LogiLLM.

This example demonstrates the basic concepts of LogiLLM:
1. Setting up a provider (OpenAI in this case)
2. Creating simple predictive modules
3. Getting structured outputs from language models

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


async def main():
    """Demonstrate basic LogiLLM usage."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== LogiLLM Hello World ===")

    try:
        # Step 1: Set up your LLM provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Step 2: Create a simple question-answering module
        # The signature "question -> answer" tells LogiLLM:
        # - Input: a field called "question"
        # - Output: a field called "answer"
        qa = Predict("question -> answer")

        # Step 3: Use the module
        result = await qa(question="What is the capital of France?")
        print("Q: What is the capital of France?")
        print(f"A: {result.outputs.get('answer')}")
        print(f"Tokens: {result.usage.tokens.total_tokens}")

        print("\n" + "=" * 50 + "\n")

        # Step 4: Structured outputs with type hints
        # LogiLLM can automatically extract structured data
        analyzer = Predict("text -> sentiment: str, confidence: float")

        result = await analyzer(text="This product is amazing! I love it!")
        print("Text: 'This product is amazing! I love it!'")
        print(f"Sentiment: {result.outputs.get('sentiment')}")
        print(f"Confidence: {result.outputs.get('confidence')}")

        # Try with negative sentiment
        result = await analyzer(text="This is terrible and broken.")
        print("\nText: 'This is terrible and broken.'")
        print(f"Sentiment: {result.outputs.get('sentiment')}")
        print(f"Confidence: {result.outputs.get('confidence')}")

        print("\n✅ Success! LogiLLM converted your intentions into working LLM calls.")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
