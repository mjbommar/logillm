"""Unit tests for multimodal message conversion."""

import base64
import pytest
from unittest.mock import MagicMock, patch

from logillm.core.signatures.types import Image, Audio
from logillm.providers.base import ProviderError


class TestOpenAIMultimodalConversion:
    """Test OpenAI provider multimodal message conversion."""
    
    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        # Patch HAS_OPENAI and clients before import
        with patch('logillm.providers.openai.HAS_OPENAI', True):
            with patch('logillm.providers.openai.OpenAIClient'):
                with patch('logillm.providers.openai.AsyncOpenAIClient'):
                    from logillm.providers.openai import OpenAIProvider
                    provider = OpenAIProvider(
                        api_key="test-key",
                        model="gpt-4o"
                    )
                    return provider
    
    def test_string_content_unchanged(self, openai_provider):
        """Test that string content passes through unchanged."""
        messages = [
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert result == messages
        assert isinstance(result[0]["content"], str)
    
    def test_image_object_conversion(self, openai_provider):
        """Test Image object converts to OpenAI format."""
        # Create a small test image (1x1 pixel PNG)
        image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        image = Image(data=image_data, format="png")
        
        messages = [{"role": "user", "content": image}]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        
        content_part = result[0]["content"][0]
        assert content_part["type"] == "image_url"
        assert "image_url" in content_part
        assert content_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert content_part["image_url"]["detail"] == "auto"
    
    def test_audio_object_conversion(self, openai_provider):
        """Test Audio object converts to OpenAI format."""
        # Create test audio data
        audio_data = b"fake audio data"
        audio = Audio(data=audio_data, format="wav")
        
        # Use audio-supporting model
        openai_provider.model = "gpt-4o-audio-preview"
        messages = [{"role": "user", "content": audio}]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        
        content_part = result[0]["content"][0]
        assert content_part["type"] == "input_audio"
        assert "input_audio" in content_part
        assert content_part["input_audio"]["format"] == "wav"
        assert content_part["input_audio"]["data"] == base64.b64encode(audio_data).decode("utf-8")
    
    def test_audio_unsupported_model_error(self, openai_provider):
        """Test that audio with unsupported model raises error."""
        audio = Audio(data=b"audio", format="wav")
        messages = [{"role": "user", "content": audio}]
        
        # gpt-4o doesn't support audio input
        openai_provider.model = "gpt-4"
        
        with pytest.raises(ProviderError, match="doesn't support audio"):
            openai_provider._prepare_multimodal_messages(messages)
    
    def test_mixed_content_list(self, openai_provider):
        """Test list with mixed content types."""
        image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        image = Image(data=image_data, format="png")
        
        messages = [{
            "role": "user",
            "content": [
                "Look at this image:",
                image,
                "What do you see?"
            ]
        }]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 3
        
        # Check each part
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Look at this image:"
        
        assert result[0]["content"][1]["type"] == "image_url"
        assert "image_url" in result[0]["content"][1]
        
        assert result[0]["content"][2]["type"] == "text"
        assert result[0]["content"][2]["text"] == "What do you see?"
    
    def test_image_size_limit_error(self, openai_provider):
        """Test that images over 20MB raise error."""
        # Create a large fake image (over 20MB when base64 encoded)
        large_data = b"x" * (21 * 1024 * 1024)  # 21MB
        large_image = Image(data=large_data, format="jpeg")
        
        messages = [{"role": "user", "content": large_image}]
        
        with pytest.raises(ProviderError, match="exceeds OpenAI limit"):
            openai_provider._prepare_multimodal_messages(messages)
    
    def test_multiple_images_in_list(self, openai_provider):
        """Test multiple images in content list."""
        img1 = Image(data=b"img1", format="png")
        img2 = Image(data=b"img2", format="jpeg")
        
        messages = [{
            "role": "user",
            "content": [
                "Compare these images:",
                img1,
                img2
            ]
        }]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][1]["type"] == "image_url"
        assert result[0]["content"][2]["type"] == "image_url"
        assert result[0]["content"][1]["image_url"]["url"].startswith("data:image/png")
        assert result[0]["content"][2]["image_url"]["url"].startswith("data:image/jpeg")
    
    def test_already_formatted_content_passthrough(self, openai_provider):
        """Test that already formatted content passes through."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}
            ]
        }]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        # Should pass through unchanged
        assert result[0]["content"][1] == messages[0]["content"][1]
    
    def test_history_object_raises_error(self, openai_provider):
        """Test that History objects raise appropriate error."""
        from logillm.core.signatures.types import History
        
        history = History(messages=[])
        messages = [{"role": "user", "content": history}]
        
        with pytest.raises(ProviderError, match="History objects should be converted"):
            openai_provider._prepare_multimodal_messages(messages)
    
    def test_fallback_to_string(self, openai_provider):
        """Test fallback conversion to string for unknown types."""
        custom_obj = {"some": "object"}
        messages = [{"role": "user", "content": custom_obj}]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert result[0]["content"] == str(custom_obj)
    
    def test_empty_messages(self, openai_provider):
        """Test empty message list."""
        result = openai_provider._prepare_multimodal_messages([])
        assert result == []
    
    def test_preserve_system_messages(self, openai_provider):
        """Test that system messages are preserved correctly."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = openai_provider._prepare_multimodal_messages(messages)
        
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
    
    def test_validate_content_types_vision_error(self, openai_provider):
        """Test that non-vision models raise error with images."""
        # Use a non-vision model
        openai_provider.model = "gpt-3.5-turbo"
        
        image = Image(data=b"img", format="png")
        messages = [{"role": "user", "content": [image]}]
        
        with pytest.raises(ProviderError, match="doesn't support vision"):
            openai_provider.validate_content_types(messages)
    
    def test_validate_content_types_audio_error(self, openai_provider):
        """Test that non-audio models raise error with audio."""
        # Use a non-audio model
        openai_provider.model = "gpt-4o"
        
        audio = Audio(data=b"audio", format="wav")
        messages = [{"role": "user", "content": [audio]}]
        
        with pytest.raises(ProviderError, match="doesn't support audio"):
            openai_provider.validate_content_types(messages)
    
    def test_validate_content_types_vision_allowed(self, openai_provider):
        """Test that vision models allow images."""
        # gpt-4o supports vision
        openai_provider.model = "gpt-4o"
        
        image = Image(data=b"img", format="png")
        messages = [{"role": "user", "content": [image]}]
        
        # Should not raise
        openai_provider.validate_content_types(messages)
    
    def test_validate_content_types_audio_allowed(self, openai_provider):
        """Test that audio models allow audio."""
        openai_provider.model = "gpt-4o-audio-preview"
        
        audio = Audio(data=b"audio", format="wav")
        messages = [{"role": "user", "content": [audio]}]
        
        # Should not raise
        openai_provider.validate_content_types(messages)
    
    def test_validate_content_types_text_always_allowed(self, openai_provider):
        """Test that text content is always allowed."""
        openai_provider.model = "gpt-3.5-turbo"  # Model without multimodal
        
        messages = [{"role": "user", "content": "Just text"}]
        
        # Should not raise
        openai_provider.validate_content_types(messages)
    
    def test_validate_content_types_dict_format(self, openai_provider):
        """Test validation works with dict format (already converted)."""
        openai_provider.model = "gpt-3.5-turbo"
        
        # Image already converted to dict format
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}
            ]
        }]
        
        with pytest.raises(ProviderError, match="doesn't support vision"):
            openai_provider.validate_content_types(messages)


class TestAnthropicMultimodalConversion:
    """Test Anthropic provider multimodal message conversion."""
    
    @pytest.fixture
    def anthropic_provider(self):
        """Create Anthropic provider instance."""
        from logillm.providers.anthropic import AnthropicProvider
        
        # Mock the client initialization
        with patch('logillm.providers.anthropic.Anthropic'):
            with patch('logillm.providers.anthropic.AsyncAnthropic'):
                provider = AnthropicProvider(
                    api_key="test-key",
                    model="claude-3-opus-20240229"
                )
                return provider
    
    def test_string_content_unchanged(self, anthropic_provider):
        """Test that string content passes through unchanged."""
        content = "Hello, world!"
        result = anthropic_provider._convert_multimodal_content(content)
        assert result == content
    
    def test_image_object_conversion(self, anthropic_provider):
        """Test Image object converts to Anthropic format."""
        image_data = b"fake image data"
        image = Image(data=image_data, format="jpeg")
        
        result = anthropic_provider._convert_multimodal_content(image)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == base64.b64encode(image_data).decode("utf-8")
    
    def test_audio_raises_not_implemented(self, anthropic_provider):
        """Test that Audio objects raise NotImplementedError."""
        audio = Audio(data=b"audio", format="wav")
        
        with pytest.raises(NotImplementedError, match="doesn't support audio"):
            anthropic_provider._convert_multimodal_content(audio)
    
    def test_mixed_content_list(self, anthropic_provider):
        """Test list with mixed content types."""
        image = Image(data=b"img", format="png")
        
        content = [
            "Check this image:",
            image,
            "What is it?"
        ]
        
        result = anthropic_provider._convert_multimodal_content(content)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Check this image:"
        
        assert result[1]["type"] == "image"
        assert result[1]["source"]["media_type"] == "image/png"
        
        assert result[2]["type"] == "text"
        assert result[2]["text"] == "What is it?"
    
    def test_audio_in_list_raises_error(self, anthropic_provider):
        """Test that Audio in list raises NotImplementedError."""
        audio = Audio(data=b"audio", format="wav")
        content = ["Listen to this:", audio]
        
        with pytest.raises(NotImplementedError, match="doesn't support audio"):
            anthropic_provider._convert_multimodal_content(content)
    
    def test_history_raises_error(self, anthropic_provider):
        """Test that History objects raise appropriate error."""
        from logillm.core.signatures.types import History
        
        history = History(messages=[])
        
        with pytest.raises(ProviderError, match="History objects should be converted"):
            anthropic_provider._convert_multimodal_content(history)
    
    def test_convert_messages_with_images(self, anthropic_provider):
        """Test _convert_messages with multimodal content."""
        image = Image(data=b"img", format="jpeg")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": image},
            {"role": "assistant", "content": "I see the image."}
        ]
        
        system_prompt, anthropic_messages = anthropic_provider._convert_messages(messages)
        
        assert system_prompt == "You are helpful."
        assert len(anthropic_messages) == 2
        
        # Check user message with image
        assert anthropic_messages[0]["role"] == "user"
        assert isinstance(anthropic_messages[0]["content"], list)
        assert anthropic_messages[0]["content"][0]["type"] == "image"
        
        # Check assistant message
        assert anthropic_messages[1]["role"] == "assistant"
        assert anthropic_messages[1]["content"] == "I see the image."
    
    def test_multiple_images_in_message(self, anthropic_provider):
        """Test multiple images in single message."""
        img1 = Image(data=b"img1", format="png")
        img2 = Image(data=b"img2", format="jpeg")
        
        content = ["Compare:", img1, "and", img2]
        
        result = anthropic_provider._convert_multimodal_content(content)
        
        assert len(result) == 4
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"
        assert result[1]["source"]["media_type"] == "image/png"
        assert result[2]["type"] == "text"
        assert result[3]["type"] == "image"
        assert result[3]["source"]["media_type"] == "image/jpeg"
    
    def test_already_formatted_content_passthrough(self, anthropic_provider):
        """Test that already formatted content passes through."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xyz"}}
        ]
        
        result = anthropic_provider._convert_multimodal_content(content)
        
        # Should pass through unchanged
        assert result == content