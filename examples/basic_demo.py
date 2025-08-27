#!/usr/bin/env python3
"""Basic LogiLLM features demonstration.
For the complete tutorial, see: docs/getting-started/quickstart.md
"""

import asyncio

from logillm.core.predict import ChainOfThought, Predict
from logillm.providers import create_provider, register_provider


async def main():
    # Configure once
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)

    # 1. Simple Q&A
    qa = Predict("question -> answer")
    result = await qa(question="What is the capital of France?")
    print(f"Q&A: {result.outputs['answer']}")

    # 2. Multiple outputs
    calc = Predict("problem -> answer, explanation")
    result = await calc(problem="What is 15% of 80?")
    print(f"\nMath: {result.outputs['answer']}")
    print(f"Why: {result.outputs['explanation']}")

    # 3. Typed extraction
    extract = Predict("text -> name: str, age: int, city: str")
    result = await extract(text="John Smith, 25, from New York")
    print(f"\nExtracted: {result.outputs}")

    # 4. Chain of thought reasoning
    cot = ChainOfThought("question -> answer")
    result = await cot(question="Why do objects fall?")
    print(f"\nReasoning: {result.outputs.get('reasoning', '')[:50]}...")
    print(f"Answer: {result.outputs['answer']}")


asyncio.run(main())
