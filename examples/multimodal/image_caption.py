#!/usr/bin/env python3
"""Example of image captioning with LogiLLM multimodal support.

This example demonstrates:
- Loading images with the Image type
- Using signatures with multimodal inputs
- Generating captions with GPT-4.1

Requirements:
- OPENAI_API_KEY environment variable
- pip install logillm[openai]
"""

import asyncio
from pathlib import Path

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import Image
from logillm.providers.openai import OpenAIProvider


class ImageCaptionSignature(Signature):
    """Generate a descriptive caption for an image.

    You are an expert at describing images concisely and accurately.
    """

    image: Image = InputField(desc="The image to caption")
    caption: str = OutputField(desc="A descriptive caption for the image")


async def main():
    """Run image captioning example."""
    # Set up the provider with GPT-4.1 (August 2025 model with vision support)
    provider = OpenAIProvider(model="gpt-4.1")

    # Create the predictor
    predictor = Predict(signature=ImageCaptionSignature, provider=provider)

    # Load an image - replace with your own image path
    image_path = Path("tests/resources/test_image.png")
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid image path")
        return

    print(f"Loading image from: {image_path}")
    image = Image.from_path(str(image_path))

    # Generate caption
    print("Generating caption...")
    result = await predictor.forward(image=image)

    print(f"\nCaption: {result.caption}")

    # You can also access the usage information
    if hasattr(result, "_prediction") and result._prediction.usage:
        usage = result._prediction.usage
        print("\nToken usage:")
        print(f"  Input: {usage.tokens.input_tokens}")
        print(f"  Output: {usage.tokens.output_tokens}")
        print(f"  Total: {usage.tokens.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
