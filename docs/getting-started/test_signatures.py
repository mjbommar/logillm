#!/usr/bin/env python3
"""
Test different signature formats to understand how they work.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.core.signatures import Signature, InputField, OutputField, parse_signature_string
from logillm.providers import create_provider, register_provider

async def test_string_signatures():
    """Test string signature parsing."""
    print("\n" + "="*60)
    print("TESTING STRING SIGNATURES")
    print("="*60)
    
    # Test parsing
    sig1 = parse_signature_string("math_problem -> answer, explanation")
    print(f"\nParsed 'math_problem -> answer, explanation':")
    print(f"  Input fields: {list(sig1.input_fields.keys())}")
    print(f"  Output fields: {list(sig1.output_fields.keys())}")
    
    sig2 = parse_signature_string("text -> sentiment: str, confidence: float, keywords: list[str]")
    print(f"\nParsed 'text -> sentiment: str, confidence: float, keywords: list[str]':")
    print(f"  Input fields: {list(sig2.input_fields.keys())}")
    print(f"  Output fields: {list(sig2.output_fields.keys())}")
    for name, field in sig2.output_fields.items():
        print(f"    {name}: type={field.python_type}")

async def test_actual_predictions():
    """Test actual predictions with different signatures."""
    print("\n" + "="*60)
    print("TESTING ACTUAL PREDICTIONS")
    print("="*60)
    
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Test 1: Multiple simple outputs
    print("\nTest 1: Multiple simple outputs")
    calc = Predict("problem -> answer, explanation")
    result = await calc(problem="What is 15% of 80?")
    print(f"  Result outputs: {result.outputs}")
    print(f"  Has answer: {hasattr(result, 'answer')}")
    print(f"  Has explanation: {hasattr(result, 'explanation')}")
    
    # Test 2: Typed outputs
    print("\nTest 2: Typed outputs")
    typed = Predict("number -> value: float, is_even: bool")
    result2 = await typed(number="42")
    print(f"  Result outputs: {result2.outputs}")
    print(f"  Value type: {type(result2.value) if hasattr(result2, 'value') else 'N/A'}")
    
    # Test 3: Class-based for comparison
    print("\nTest 3: Class-based signature")
    
    class TestSig(Signature):
        problem: str = InputField()
        answer: str = OutputField()
        explanation: str = OutputField()
    
    class_predict = Predict(signature=TestSig)
    result3 = await class_predict(problem="What is 15% of 80?")
    print(f"  Result outputs: {result3.outputs}")
    print(f"  Has answer: {hasattr(result3, 'answer')}")
    print(f"  Has explanation: {hasattr(result3, 'explanation')}")

async def main():
    await test_string_signatures()
    await test_actual_predictions()

if __name__ == "__main__":
    asyncio.run(main())