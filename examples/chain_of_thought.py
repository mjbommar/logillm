#!/usr/bin/env python3
"""Chain of Thought Reasoning with LogiLLM.

This example demonstrates how LogiLLM handles complex reasoning tasks:
1. Basic prediction vs. chain of thought
2. Step-by-step mathematical reasoning
3. Complex analysis with intermediate steps
4. Debugging reasoning with trace outputs

Chain of thought prompting helps language models break down complex problems
into smaller, manageable steps, leading to more accurate results.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers import create_provider, register_provider


class MathProblem(Signature):
    """Solve a mathematical word problem step by step."""

    problem: str = InputField(desc="Mathematical word problem to solve")

    reasoning: str = OutputField(desc="Step-by-step reasoning process")
    answer: str = OutputField(desc="Final numerical answer")
    confidence: float = OutputField(desc="Confidence in answer (0.0 to 1.0)")


class MarketAnalysis(Signature):
    """Analyze market conditions with detailed reasoning."""

    company: str = InputField(desc="Company name")
    data: str = InputField(desc="Market data and context")

    analysis: str = OutputField(desc="Detailed step-by-step market analysis")
    recommendation: str = OutputField(desc="Investment recommendation")
    risk_factors: str = OutputField(desc="Key risk factors identified")
    confidence: float = OutputField(desc="Confidence in recommendation (0.0 to 1.0)")


async def main():
    """Demonstrate chain of thought reasoning."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== Chain of Thought Reasoning ===")

    try:
        # Set up provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # 1. Compare basic prediction vs chain of thought
        print("1. Basic vs Chain of Thought")
        print("-" * 30)

        problem = "A store has 45 apples. They sell 18 apples in the morning and 12 apples in the afternoon. How many apples are left?"

        # Basic prediction - often gets simple problems right but lacks explanation
        basic = Predict("problem -> answer")
        result = await basic(problem=problem)
        print(f"Basic Prediction: {result.outputs.get('answer')}")

        # Chain of thought - shows the reasoning process
        cot = Predict(signature=MathProblem)
        result = await cot(problem=problem)
        print("\nChain of Thought:")
        print(f"Reasoning: {result.outputs.get('reasoning')}")
        print(f"Answer: {result.outputs.get('answer')}")
        print(f"Confidence: {result.outputs.get('confidence')}")

        print("\n" + "=" * 60 + "\n")

        # 2. More complex mathematical reasoning
        print("2. Complex Problem Solving")
        print("-" * 27)

        complex_problem = """
        A car rental company charges $25 per day plus $0.15 per mile.
        Sarah rents a car for 3 days and drives 420 miles.
        If she has a 10% discount coupon, how much will she pay in total?
        """

        result = await cot(problem=complex_problem)
        print(f"Problem: {complex_problem.strip()}")
        print(f"\nReasoning: {result.outputs.get('reasoning')}")
        print(f"Final Answer: {result.outputs.get('answer')}")

        print("\n" + "=" * 60 + "\n")

        # 3. Complex analysis requiring multiple reasoning steps
        print("3. Multi-Step Analysis")
        print("-" * 22)

        analyst = Predict(signature=MarketAnalysis)

        market_data = """
        TechCorp Stock Analysis:
        - Current price: $150
        - 52-week high: $180
        - 52-week low: $90
        - P/E ratio: 25
        - Revenue growth: 15% YoY
        - Profit margin: 12%
        - Debt-to-equity: 0.3
        - Recent news: Launched new AI product, increased competition from rivals
        """

        result = await analyst(company="TechCorp", data=market_data)
        print("Company: TechCorp")
        print("\nDetailed Analysis:")
        print(result.outputs.get("analysis"))
        print(f"\nRecommendation: {result.outputs.get('recommendation')}")
        print(f"Risk Factors: {result.outputs.get('risk_factors')}")
        print(f"Confidence: {result.outputs.get('confidence')}")

        print("\n" + "=" * 60 + "\n")

        # 4. Enable debug mode to see the actual prompts
        print("4. Debug Mode - See the Prompts")
        print("-" * 32)

        debug_solver = Predict(signature=MathProblem)
        debug_solver.enable_debug_mode()

        simple_problem = "If a pizza is cut into 8 slices and John eats 3 slices, what fraction of the pizza is left?"
        result = await debug_solver(problem=simple_problem)

        print(f"Problem: {simple_problem}")
        print(f"Answer: {result.outputs.get('answer')}")

        # Show what prompt was actually sent
        print("\n🔍 Debug Info:")
        print(f"Messages sent to LLM: {len(result.prompt['messages'])}")
        print(f"Model used: {result.prompt['model']}")
        print(f"First message preview: {result.prompt['messages'][0]['content'][:100]}...")

        print("\n✅ Chain of thought helps with:")
        print("• Complex mathematical problems")
        print("• Multi-step reasoning tasks")
        print("• Analysis requiring explanation")
        print("• Building confidence in answers")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
