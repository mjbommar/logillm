#!/usr/bin/env python3
"""Chain of Thought Reasoning with LogiLLM.

This example demonstrates the ACTUAL ChainOfThought module in LogiLLM:
1. Compare Predict vs ChainOfThought for the same problem
2. See how ChainOfThought automatically adds reasoning fields
3. Complex multi-step reasoning problems
4. Debug mode showing full prompts

ChainOfThought automatically adds a "reasoning" field to any signature,
helping models break down complex problems into steps.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.predict import ChainOfThought, Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers import create_provider, register_provider


class SimpleMath(Signature):
    """Solve a mathematical problem."""

    problem: str = InputField(desc="Math problem to solve")
    answer: str = OutputField(desc="Final numerical answer")


class ComplexAnalysis(Signature):
    """Analyze data and provide recommendations."""

    topic: str = InputField(desc="Topic to analyze")
    data: str = InputField(desc="Data and context")

    conclusion: str = OutputField(desc="Final conclusion")
    confidence: float = OutputField(desc="Confidence level (0.0 to 1.0)")


async def main():
    """Demonstrate the real ChainOfThought module."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== REAL Chain of Thought vs Regular Predict ===")

    try:
        # Set up provider
        provider = create_provider("openai", model="gpt-4.1-mini")
        register_provider(provider, set_default=True)

        # 1. Compare Predict vs ChainOfThought on same problem
        print("\n1. Predict vs ChainOfThought Comparison")
        print("-" * 40)

        problem = "A store has 45 apples. They sell 18 in the morning and 12 in the afternoon. How many are left?"

        # Regular Predict - just gets the answer
        print("Using regular Predict:")
        regular = Predict(SimpleMath)
        result = await regular(problem=problem)
        print(f"  Answer: {result.outputs.get('answer')}")
        print(f"  Output fields: {list(result.outputs.keys())}")

        # ChainOfThought - automatically adds reasoning field!
        print("\nUsing ChainOfThought:")
        cot = ChainOfThought(SimpleMath)  # Same signature!
        result = await cot(problem=problem)
        print(f"  Answer: {result.outputs.get('answer')}")
        print(f"  Reasoning: {result.outputs.get('reasoning')}")  # Auto-added!
        print(f"  Output fields: {list(result.outputs.keys())}")

        print("\n" + "=" * 60)

        # 2. Complex problem with ChainOfThought
        print("\n2. Complex Problem with ChainOfThought")
        print("-" * 40)

        complex_problem = """
        A car rental costs $25/day plus $0.15/mile.
        Sarah rents for 3 days, drives 420 miles, and has a 10% discount.
        What's her total cost?
        """

        cot_math = ChainOfThought("problem: str -> answer: float")
        result = await cot_math(problem=complex_problem)

        print(f"Problem: {complex_problem.strip()}")
        print("\nReasoning Process:")
        print(result.outputs.get("reasoning"))
        print(f"\nFinal Answer: ${result.outputs.get('answer'):.2f}")

        print("\n" + "=" * 60)

        # 3. Show that ChainOfThought works with any signature
        print("\n3. ChainOfThought with Complex Signature")
        print("-" * 40)

        market_data = """
        TechCorp: Stock $150, 52-week high $180, low $90
        P/E: 25, Revenue growth: 15% YoY, Debt-to-equity: 0.3
        Recent: Launched AI product, facing increased competition
        """

        # ChainOfThought automatically adds reasoning to ComplexAnalysis
        cot_analyst = ChainOfThought(ComplexAnalysis)
        result = await cot_analyst(topic="TechCorp Stock", data=market_data)

        print("Analysis of TechCorp Stock:")
        print("\nReasoning (auto-added by ChainOfThought):")
        print(result.outputs.get("reasoning"))
        print(f"\nConclusion: {result.outputs.get('conclusion')}")
        print(f"Confidence: {result.outputs.get('confidence')}")

        print("\n" + "=" * 60)

        # 4. Debug mode - see the actual prompts
        print("\n4. Debug Mode - See Full Prompts")
        print("-" * 40)

        # Enable debug to see what ChainOfThought actually sends
        debug_cot = ChainOfThought("question: str -> answer: str", debug=True)

        result = await debug_cot(question="What is the capital of France?")

        print("Question: What is the capital of France?")
        print(f"Answer: {result.outputs.get('answer')}")
        print(f"Reasoning: {result.outputs.get('reasoning')}")

        if result.prompt:
            print("\n🔍 Debug Info - Full Prompt:")
            print("-" * 40)
            messages = result.prompt.get("messages", [])
            for i, msg in enumerate(messages):
                print(f"\nMessage {i + 1} [{msg.get('role', 'unknown')}]:")
                content = msg.get("content", "")
                if len(content) > 500:
                    print(content[:500] + "...")
                else:
                    print(content)

            print("\n📊 Metadata:")
            print(f"  Model: {result.prompt.get('model', 'unknown')}")
            print(f"  Adapter: {result.prompt.get('adapter', 'unknown')}")
            print(f"  Demo count: {result.prompt.get('demos_count', 0)}")

        print("\n" + "=" * 60)

        # 5. Show signature inspection
        print("\n5. Understanding ChainOfThought Modifications")
        print("-" * 40)

        original_sig = SimpleMath
        cot_module = ChainOfThought(original_sig)

        print("Original SimpleMath signature fields:")
        print(f"  Inputs: {list(original_sig.input_fields.keys())}")
        print(f"  Outputs: {list(original_sig.output_fields.keys())}")

        print("\nChainOfThought-modified signature fields:")
        print(f"  Inputs: {list(cot_module.signature.input_fields.keys())}")
        print(f"  Outputs: {list(cot_module.signature.output_fields.keys())}")
        print("\n✅ Notice: ChainOfThought automatically added 'reasoning' field!")

        print("\n" + "=" * 60)
        print("\n✨ Key Takeaways:")
        print("• ChainOfThought automatically adds a 'reasoning' field to ANY signature")
        print("• This helps LLMs show their work and improve accuracy")
        print("• You can use it with string signatures or class-based signatures")
        print("• Debug mode shows the full prompts sent to the LLM")
        print("• The reasoning field appears BEFORE the original output fields")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
