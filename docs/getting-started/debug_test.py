#!/usr/bin/env python3
"""
Debug the tutorial to see what's happening with multiple outputs.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Test multiple outputs
    calculator = Predict("math_problem -> answer, explanation")
    result = await calculator(math_problem="What is 15% of 80?")
    
    print(f"Result type: {type(result)}")
    print(f"Result dir: {[x for x in dir(result) if not x.startswith('_')]}")
    
    # Try to access fields
    print(f"\nTrying to access 'answer':")
    if hasattr(result, 'answer'):
        print(f"  answer = {result.answer}")
    else:
        print("  No 'answer' attribute")
    
    print(f"\nTrying to access 'explanation':")  
    if hasattr(result, 'explanation'):
        print(f"  explanation = {result.explanation}")
    else:
        print("  No 'explanation' attribute")
    
    # Check outputs
    print(f"\nChecking outputs:")
    if hasattr(result, 'outputs'):
        print(f"  outputs = {result.outputs}")
    
    # Check get_output
    print(f"\nChecking get_output method:")
    if hasattr(result, 'get_output'):
        try:
            answer = result.get_output('answer')
            print(f"  get_output('answer') = {answer}")
        except:
            print("  get_output('answer') failed")
        
        try:
            explanation = result.get_output('explanation')
            print(f"  get_output('explanation') = {explanation}")
        except:
            print("  get_output('explanation') failed")

if __name__ == "__main__":
    asyncio.run(main())