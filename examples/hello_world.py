#!/usr/bin/env python3
"""Minimal LogiLLM - matches docs/getting-started/quickstart.md Step 1"""

import asyncio

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


async def main():
    # Setup
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)

    # Use
    qa = Predict("question -> answer")
    result = await qa(question="What is 2 + 2?")
    print(result.answer)


asyncio.run(main())
