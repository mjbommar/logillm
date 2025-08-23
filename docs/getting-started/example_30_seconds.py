#!/usr/bin/env python3
"""
30 Seconds to Your First REAL LogiLLM App
Uses actual OpenAI GPT-4.1 - NO MOCKS!
AS CLAUDE.md DEMANDS!
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider

async def main():
    # Use REAL OpenAI GPT-4.1 as CLAUDE.md line 7 demands
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Say what you want
    calculator = Predict("question -> answer")
    
    # Get it with REAL LLM
    print("Asking GPT-4.1: What is 6 times 7?")
    result = await calculator(question="What is 6 times 7?")
    print(f"Answer: {result.answer}")
    
    print("\n✅ Success! You just used a REAL LLM (GPT-4.1) with LogiLLM.")
    print("📖 This was a REAL API call with GPT-4.1, not a mock!")

if __name__ == "__main__":
    asyncio.run(main())