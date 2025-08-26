"""Integration tests for OpenAI multimodal support.

These tests use REAL OpenAI API calls to test multimodal functionality.
Requires OPENAI_API_KEY environment variable.
"""

import asyncio
import os
from pathlib import Path

import pytest

from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import Image
from logillm.core.predict import Predict
from logillm.providers.base import ProviderError
from logillm.providers.openai import OpenAIProvider

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


class ImageCaptionSignature(Signature):
    """Generate a caption for an image."""

    image: Image = InputField(desc="The image to caption")
    caption: str = OutputField(desc="A descriptive caption for the image")


class ImageAnalysisSignature(Signature):
    """Analyze an image in detail."""

    question: str = InputField(desc="Question about the image")
    image: Image = InputField(desc="The image to analyze")
    answer: str = OutputField(desc="Answer to the question")
    confidence: float = OutputField(desc="Confidence score between 0 and 1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_caption_with_gpt41():
    """Test image captioning with gpt-4.1 using real API."""
    # Load test image
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists(), f"Test image not found at {image_path}"

    # Create provider with gpt-4.1
    provider = OpenAIProvider(model="gpt-4.1")

    # Create predictor
    predictor = Predict(
        signature=ImageCaptionSignature,
        provider=provider
    )

    # Load image
    image = Image.from_path(str(image_path))

    # Generate caption
    result = await predictor.forward(image=image)

    # Verify we got a caption
    assert result.caption
    assert isinstance(result.caption, str)
    assert len(result.caption) > 10  # Should be a meaningful caption

    print(f"Generated caption: {result.caption}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_question_answering():
    """Test visual question answering with gpt-4.1."""
    # Load test image
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    # Create provider
    provider = OpenAIProvider(model="gpt-4.1")

    # Create predictor
    predictor = Predict(
        signature=ImageAnalysisSignature,
        provider=provider
    )

    # Load image and ask question
    image = Image.from_path(str(image_path))

    result = await predictor.forward(
        question="What type of document is shown in this image?",
        image=image
    )

    # Check results
    assert result.answer
    assert isinstance(result.answer, str)
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_vision_model_error():
    """Test that non-vision models raise appropriate error with images."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    # Create provider with non-vision model
    provider = OpenAIProvider(model="gpt-3.5-turbo")

    # Load image
    image = Image.from_path(str(image_path))

    # Try to send image to non-vision model
    messages = [{"role": "user", "content": [image]}]

    # Should raise error about model not supporting vision
    with pytest.raises(ProviderError, match="doesn't support vision"):
        await provider.complete(messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_direct_api_call_with_image():
    """Test direct API call with image using provider."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    # Create provider
    provider = OpenAIProvider(model="gpt-4.1")

    # Load image
    image = Image.from_path(str(image_path))

    # Create messages with multimodal content
    messages = [
        {"role": "system", "content": "You are a helpful assistant that describes images."},
        {"role": "user", "content": [
            "Please describe this image in one sentence:",
            image
        ]}
    ]

    # Make API call
    completion = await provider.complete(messages)

    # Check response
    assert completion.text
    assert len(completion.text) > 10
    assert completion.usage.tokens.input_tokens > 0
    assert completion.usage.tokens.output_tokens > 0

    print(f"Direct API response: {completion.text}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_images():
    """Test sending multiple images in one request."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    provider = OpenAIProvider(model="gpt-4.1")

    # Load same image twice (as if they were different)
    image1 = Image.from_path(str(image_path))
    image2 = Image.from_path(str(image_path))

    # Create message with multiple images
    messages = [{
        "role": "user",
        "content": [
            "Are these two images the same or different?",
            image1,
            image2,
            "Please answer with just 'same' or 'different'."
        ]
    }]

    completion = await provider.complete(messages)

    assert completion.text
    assert "same" in completion.text.lower() or "different" in completion.text.lower()

    print(f"Multiple images response: {completion.text}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_size_validation():
    """Test that oversized images are rejected."""
    # Create a fake large image (over 20MB when base64 encoded)
    large_image_data = b"x" * (21 * 1024 * 1024)  # 21MB of data
    large_image = Image(data=large_image_data, format="png")

    provider = OpenAIProvider(model="gpt-4.1")

    messages = [{"role": "user", "content": [large_image]}]

    # Should raise error about size
    with pytest.raises(ProviderError, match="exceeds.*limit"):
        await provider.complete(messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mixed_text_and_image():
    """Test mixed text and image content in same message."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    provider = OpenAIProvider(model="gpt-4.1")
    image = Image.from_path(str(image_path))

    # Mix text and image
    messages = [{
        "role": "user",
        "content": [
            "I have an important document here.",
            image,
            "Can you tell me what type of document this is and what organization it's from?"
        ]
    }]

    completion = await provider.complete(messages)

    assert completion.text
    assert len(completion.text) > 20  # Should be a detailed response

    print(f"Mixed content response: {completion.text}")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_image_caption_with_gpt41())
    asyncio.run(test_image_question_answering())
    asyncio.run(test_direct_api_call_with_image())
    asyncio.run(test_multiple_images())
    asyncio.run(test_mixed_text_and_image())
    print("All integration tests completed!")
