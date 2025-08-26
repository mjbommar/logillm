"""Integration tests for Anthropic multimodal support.

These tests use REAL Anthropic API calls to test multimodal functionality.
Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import os
from pathlib import Path

import pytest

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import Audio, Image
from logillm.providers.anthropic import AnthropicProvider

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)


class ImageDescriptionSignature(Signature):
    """Describe an image."""

    image: Image = InputField(desc="The image to describe")
    description: str = OutputField(desc="A detailed description of the image")


class ImageComparisonSignature(Signature):
    """Compare two images."""

    instruction: str = InputField(desc="What to compare")
    image1: Image = InputField(desc="First image")
    image2: Image = InputField(desc="Second image")
    comparison: str = OutputField(desc="Comparison results")
    similarity: float = OutputField(desc="Similarity score from 0 to 1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_image_description():
    """Test image description with Claude 4 Opus using real API."""
    # Load test image
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists(), f"Test image not found at {image_path}"

    # Create provider with Claude 4 Opus
    provider = AnthropicProvider(model="claude-4-opus-20250514")

    # Create predictor
    predictor = Predict(
        signature=ImageDescriptionSignature,
        provider=provider
    )

    # Load image
    image = Image.from_path(str(image_path))

    # Generate description
    result = await predictor.forward(image=image)

    # Verify we got a description
    assert result.description
    assert isinstance(result.description, str)
    assert len(result.description) > 20  # Should be a detailed description

    print(f"Claude description: {result.description}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_direct_api_call():
    """Test direct API call with image using Anthropic provider."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    # Create provider with Claude 4 Sonnet (more cost-effective)
    provider = AnthropicProvider(model="claude-4-sonnet-20250514")

    # Load image
    image = Image.from_path(str(image_path))

    # Create messages with multimodal content
    messages = [
        {"role": "user", "content": [
            "What document is shown in this image? Please be specific.",
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

    print(f"Claude response: {completion.text}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_mixed_text_and_image():
    """Test mixed text and image content with Claude."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    provider = AnthropicProvider(model="claude-4-sonnet-20250514")
    image = Image.from_path(str(image_path))

    # Mix text before and after image
    messages = [
        {"role": "user", "content": [
            "I need help understanding this document.",
            image,
            "What organization submitted this form and why?"
        ]}
    ]

    completion = await provider.complete(messages)

    assert completion.text
    assert len(completion.text) > 30

    # Should mention OpenAI since it's their form
    assert "OpenAI" in completion.text or "openai" in completion.text.lower()

    print(f"Mixed content response: {completion.text}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_audio_not_supported():
    """Test that Audio raises NotImplementedError for Anthropic."""
    # Create fake audio
    audio = Audio(data=b"fake audio data", format="wav")

    provider = AnthropicProvider(model="claude-4-sonnet-20250514")

    messages = [{"role": "user", "content": [audio]}]

    # Should raise NotImplementedError for audio
    with pytest.raises(NotImplementedError, match="doesn't support audio"):
        await provider.complete(messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_multiple_images():
    """Test sending multiple images to Claude."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    provider = AnthropicProvider(model="claude-4-sonnet-20250514")

    # Load same image twice (simulating different images)
    image1 = Image.from_path(str(image_path))
    image2 = Image.from_path(str(image_path))

    messages = [
        {"role": "user", "content": [
            "Compare these two images:",
            image1,
            "and",
            image2,
            "Are they the same document?"
        ]}
    ]

    completion = await provider.complete(messages)

    assert completion.text
    # Should recognize they're the same
    assert "same" in completion.text.lower() or "identical" in completion.text.lower()

    print(f"Multiple images response: {completion.text}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_with_system_prompt():
    """Test image with system prompt."""
    image_path = Path("tests/resources/test_image.png")
    assert image_path.exists()

    provider = AnthropicProvider(model="claude-4-sonnet-20250514")
    image = Image.from_path(str(image_path))

    messages = [
        {"role": "system", "content": "You are a tax document expert. Be very precise and technical."},
        {"role": "user", "content": [
            "Analyze this tax form:",
            image
        ]}
    ]

    completion = await provider.complete(messages)

    assert completion.text
    # Should use technical tax terminology
    assert "501(c)(3)" in completion.text or "tax" in completion.text.lower() or "IRS" in completion.text

    print(f"System prompt response: {completion.text}")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_claude_image_description())
    asyncio.run(test_claude_direct_api_call())
    asyncio.run(test_claude_mixed_text_and_image())
    asyncio.run(test_claude_multiple_images())
    asyncio.run(test_claude_with_system_prompt())
    print("All Anthropic integration tests completed!")
