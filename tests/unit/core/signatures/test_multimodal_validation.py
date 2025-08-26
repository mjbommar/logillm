"""Unit tests for multimodal field validation in signatures."""

import pytest

from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import Audio, Image


class ImageSignature(Signature):
    """Test signature with image field."""

    image: Image = InputField(desc="Input image")
    caption: str = OutputField(desc="Image caption")


class AudioSignature(Signature):
    """Test signature with audio field."""

    audio: Audio = InputField(desc="Input audio")
    transcript: str = OutputField(desc="Audio transcript")


def test_validate_image_field():
    """Test validation of Image fields."""
    # Valid image
    image = Image(data=b"fake image data", format="jpeg")
    validated = ImageSignature.validate_inputs(image=image)
    assert validated["image"] == image


def test_validate_image_field_wrong_type():
    """Test validation fails with wrong type for Image field."""
    with pytest.raises(ValueError, match="Field 'image' expects Image type"):
        ImageSignature.validate_inputs(image="not an image")


def test_validate_image_field_no_data():
    """Test validation fails when Image has no data."""
    image = Image(data=None, format="jpeg")
    with pytest.raises(ValueError, match="Image field 'image' has no data"):
        ImageSignature.validate_inputs(image=image)


def test_validate_image_field_no_format():
    """Test validation fails when Image has no format."""
    image = Image(data=b"data", format=None)
    with pytest.raises(ValueError, match="Image field 'image' has no format"):
        ImageSignature.validate_inputs(image=image)


def test_validate_audio_field():
    """Test validation of Audio fields."""
    # Valid audio
    audio = Audio(data=b"fake audio data", format="mp3")
    validated = AudioSignature.validate_inputs(audio=audio)
    assert validated["audio"] == audio


def test_validate_audio_field_wrong_type():
    """Test validation fails with wrong type for Audio field."""
    with pytest.raises(ValueError, match="Field 'audio' expects Audio type"):
        AudioSignature.validate_inputs(audio="not audio")


def test_validate_audio_field_no_data():
    """Test validation fails when Audio has no data."""
    audio = Audio(data=None, format="mp3")
    with pytest.raises(ValueError, match="Audio field 'audio' has no data"):
        AudioSignature.validate_inputs(audio=audio)


def test_validate_audio_field_no_format():
    """Test validation fails when Audio has no format."""
    audio = Audio(data=b"data", format=None)
    with pytest.raises(ValueError, match="Audio field 'audio' has no format"):
        AudioSignature.validate_inputs(audio=audio)


def test_validate_audio_invalid_format():
    """Test validation fails with invalid audio format."""
    audio = Audio(data=b"data", format="invalid")
    with pytest.raises(ValueError, match="Audio field 'audio' has invalid format"):
        AudioSignature.validate_inputs(audio=audio)


def test_validate_audio_valid_formats():
    """Test all valid audio formats are accepted."""
    valid_formats = ["mp3", "wav", "flac", "ogg", "m4a", "webm"]
    for fmt in valid_formats:
        audio = Audio(data=b"data", format=fmt)
        validated = AudioSignature.validate_inputs(audio=audio)
        assert validated["audio"] == audio
