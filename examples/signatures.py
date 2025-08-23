#!/usr/bin/env python3
"""LogiLLM Signature Examples.

This example demonstrates different ways to define signatures in LogiLLM:
1. Simple string signatures
2. String signatures with type hints
3. Class-based signatures with detailed field specifications
4. Complex nested outputs

Signatures define what inputs your module expects and what outputs it should produce.
LogiLLM uses this specification to construct appropriate prompts and parse responses.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers import create_provider, register_provider


class CustomerFeedback(Signature):
    """Analyze customer feedback for actionable insights."""

    feedback: str = InputField(desc="Raw customer feedback text")

    sentiment: str = OutputField(desc="Overall sentiment: positive, negative, or neutral")
    issues: list[str] = OutputField(desc="List of specific issues mentioned")
    suggestions: list[str] = OutputField(desc="Actionable improvement suggestions")
    priority: int = OutputField(desc="Priority level from 1 (low) to 5 (urgent)")
    category: str = OutputField(desc="Feedback category: product, service, billing, etc.")


class CodeReview(Signature):
    """Review code for quality and suggest improvements."""

    code: str = InputField(desc="Code to review")
    language: str = InputField(desc="Programming language")

    score: int = OutputField(desc="Quality score from 1-10")
    issues: list[str] = OutputField(desc="List of issues found")
    suggestions: list[str] = OutputField(desc="Specific improvement suggestions")
    security_concerns: list[str] = OutputField(desc="Security issues if any")


async def main():
    """Demonstrate different signature formats."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== LogiLLM Signature Examples ===")

    try:
        # Set up provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # 1. Simple string signature - great for quick prototyping
        print("1. Simple String Signature")
        print("-" * 30)

        classifier = Predict("text -> category")
        result = await classifier(text="I need help with my billing account")
        print("Text: 'I need help with my billing account'")
        print(f"Category: {result.outputs.get('category')}")

        print("\n" + "=" * 60 + "\n")

        # 2. String signature with type hints - adds structure
        print("2. String Signature with Types")
        print("-" * 32)

        sentiment = Predict("review -> rating: int, summary: str")
        result = await sentiment(review="The product works well but shipping was slow")
        print("Review: 'The product works well but shipping was slow'")
        print(f"Rating: {result.outputs.get('rating')}")
        print(f"Summary: {result.outputs.get('summary')}")

        print("\n" + "=" * 60 + "\n")

        # 3. Class-based signature with detailed specifications
        print("3. Class-Based Signature (Detailed)")
        print("-" * 35)

        feedback_analyzer = Predict(signature=CustomerFeedback)

        sample_feedback = """
        The app crashes every time I try to upload a photo. This has been happening
        for weeks and makes the app basically unusable. Also, customer support
        hasn't responded to my three emails. Very frustrated!
        """

        result = await feedback_analyzer(feedback=sample_feedback)
        print(f"Sentiment: {result.outputs.get('sentiment')}")
        print(f"Issues: {result.outputs.get('issues')}")
        print(f"Priority: {result.outputs.get('priority')}")
        print(f"Category: {result.outputs.get('category')}")
        print(f"Suggestions: {result.outputs.get('suggestions')}")

        print("\n" + "=" * 60 + "\n")

        # 4. Technical analysis with complex outputs
        print("4. Complex Technical Analysis")
        print("-" * 30)

        reviewer = Predict(signature=CodeReview)

        sample_code = """
def process_data(data):
    result = []
    for item in data:
        if item != None:
            result.append(item.upper())
    return result
        """

        result = await reviewer(code=sample_code, language="python")
        print(f"Code Quality Score: {result.outputs.get('score')}/10")
        print(f"Issues Found: {result.outputs.get('issues')}")
        print(f"Suggestions: {result.outputs.get('suggestions')}")
        if result.outputs.get("security_concerns"):
            print(f"Security: {result.outputs.get('security_concerns')}")

        print("\n✅ Different signature styles give you different levels of control!")
        print("• String signatures: Quick and simple")
        print("• Type hints: Add structure to outputs")
        print("• Class signatures: Full control with documentation")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
