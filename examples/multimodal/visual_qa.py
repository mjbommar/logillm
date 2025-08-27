#!/usr/bin/env python3
"""Example of visual question answering with LogiLLM multimodal support.

This example demonstrates:
- Asking questions about images
- Using confidence scores in outputs
- Working with both OpenAI and Anthropic providers

Requirements:
- OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables
- pip install logillm[openai] or logillm[anthropic]
"""

import asyncio
import sys
from pathlib import Path

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import Image


class VisualQASignature(Signature):
    """Answer questions about an image with confidence.

    Analyze the image carefully and answer the question accurately.
    """

    question: str = InputField(desc="Question about the image")
    image: Image = InputField(desc="The image to analyze")
    answer: str = OutputField(desc="Detailed answer to the question")
    confidence: float = OutputField(desc="Confidence score from 0 to 1")


async def analyze_image_openai(image_path: str, question: str):
    """Analyze image with OpenAI GPT-4.1."""
    from logillm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4.1")
    predictor = Predict(signature=VisualQASignature, provider=provider)

    image = Image.from_path(image_path)
    result = await predictor.forward(question=question, image=image)

    return result


async def analyze_image_anthropic(image_path: str, question: str):
    """Analyze image with Anthropic Claude 4."""
    from logillm.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-4-sonnet-20250514")
    predictor = Predict(signature=VisualQASignature, provider=provider)

    image = Image.from_path(image_path)
    result = await predictor.forward(question=question, image=image)

    return result


async def main():
    """Run visual QA example."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python visual_qa.py <provider> [image_path] [question]")
        print("Providers: openai, anthropic")
        sys.exit(1)

    provider_name = sys.argv[1].lower()

    # Get image path
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
    else:
        image_path = "tests/resources/test_image.png"

    # Get question
    if len(sys.argv) > 3:
        question = " ".join(sys.argv[3:])
    else:
        question = "What type of document is this and what are its key details?"

    # Validate image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Provider: {provider_name}")
    print(f"Image: {image_path}")
    print(f"Question: {question}")
    print("-" * 50)

    # Analyze based on provider
    try:
        if provider_name == "openai":
            result = await analyze_image_openai(image_path, question)
        elif provider_name == "anthropic":
            result = await analyze_image_anthropic(image_path, question)
        else:
            print(f"Unknown provider: {provider_name}")
            print("Use 'openai' or 'anthropic'")
            sys.exit(1)

        print(f"\nAnswer: {result.answer}")
        print(f"\nConfidence: {result.confidence:.2%}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
