#!/usr/bin/env python3
"""Example of comparing multiple images with LogiLLM multimodal support.

This example demonstrates:
- Loading and comparing multiple images
- Using GPT-4.1 for visual comparison
- Analyzing differences and similarities

Requirements:
- OPENAI_API_KEY environment variable
- pip install logillm[openai]
"""

import asyncio
import sys
from pathlib import Path

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import Image
from logillm.providers.openai import OpenAIProvider


class ImageComparisonSignature(Signature):
    """Compare two images and identify differences and similarities.

    Analyze both images carefully and provide detailed comparison.
    """

    image1: Image = InputField(desc="The first image to compare")
    image2: Image = InputField(desc="The second image to compare")
    similarities: str = OutputField(desc="Key similarities between the images")
    differences: str = OutputField(desc="Key differences between the images")
    summary: str = OutputField(desc="Brief comparison summary")


async def main():
    """Run image comparison example."""
    # Get image paths from arguments or use defaults
    if len(sys.argv) > 2:
        image1_path = Path(sys.argv[1])
        image2_path = Path(sys.argv[2])
    else:
        # Use the same test image twice for demo (you'd use different images)
        image1_path = Path("tests/resources/test_image.png")
        image2_path = Path("tests/resources/test_image.png")
        print("Using default test images. Pass two image paths as arguments for custom comparison.")

    # Validate both images exist
    for img_path in [image1_path, image2_path]:
        if not img_path.exists():
            print(f"Error: Image not found at {img_path}")
            sys.exit(1)

    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print("-" * 50)

    # Set up provider with GPT-4.1 (August 2025 model with vision support)
    provider = OpenAIProvider(model="gpt-4.1")

    # Create the predictor
    predictor = Predict(signature=ImageComparisonSignature, provider=provider)

    # Load both images
    print("Loading images...")
    image1 = Image.from_path(str(image1_path))
    image2 = Image.from_path(str(image2_path))

    # Compare images
    print("Comparing images...")
    result = await predictor.forward(image1=image1, image2=image2)

    # Display results
    print(f"\nSimilarities:\n{result.similarities}")
    print(f"\nDifferences:\n{result.differences}")
    print(f"\nSummary:\n{result.summary}")

    # Show usage info if available
    if hasattr(result, "_prediction") and result._prediction.usage:
        usage = result._prediction.usage
        print("\nToken usage:")
        print(f"  Input: {usage.tokens.input_tokens}")
        print(f"  Output: {usage.tokens.output_tokens}")
        print(f"  Total: {usage.tokens.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
