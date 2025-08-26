"""Unit tests for multimodal adapter support."""

import pytest
from unittest.mock import Mock, MagicMock
from logillm.core.format_adapters import ChatAdapter, JSONAdapter
from logillm.core.signatures.types import Image, Audio, History


class TestChatAdapterMultimodal:
    """Test ChatAdapter multimodal content handling."""
    
    @pytest.fixture
    def adapter(self):
        """Create ChatAdapter instance."""
        return ChatAdapter()
    
    @pytest.fixture
    def mock_signature(self):
        """Create mock signature with input/output fields."""
        sig = Mock()
        sig.input_fields = {
            "question": Mock(desc="Question about the image"),
            "image": Mock(desc="Image to analyze")
        }
        sig.output_fields = {
            "answer": Mock(desc="Your analysis"),
            "confidence": Mock(desc="Confidence score", annotation=float)
        }
        return sig
    
    def test_text_only_returns_string(self, adapter, mock_signature):
        """Test that text-only inputs return string format."""
        inputs = {"question": "What is the capital of France?"}
        
        result = adapter.format_prompt(inputs, mock_signature)
        
        assert isinstance(result, str)
        assert "What is the capital of France?" in result
        assert "Please provide:" in result
        assert "Confidence score" in result
    
    def test_image_input_returns_messages(self, adapter, mock_signature):
        """Test that image inputs return message list format."""
        image = Image(data=b"fake image data", format="jpeg")
        inputs = {"question": "What's in this image?", "image": image}
        
        result = adapter.format_prompt(inputs, mock_signature)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        
        # Check content includes both text and image
        content = result[0]["content"]
        assert "What's in this image?" in str(content)
        assert image in content
    
    def test_audio_input_returns_messages(self, adapter, mock_signature):
        """Test that audio inputs return message list format."""
        audio = Audio(data=b"fake audio data", format="wav")
        inputs = {"question": "Transcribe this", "audio": audio}
        
        # Update signature for audio field
        mock_signature.input_fields["audio"] = Mock(desc="Audio to transcribe")
        
        result = adapter.format_prompt(inputs, mock_signature)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert audio in result[0]["content"]
    
    def test_mixed_multimodal_content(self, adapter, mock_signature):
        """Test mixed text, image, and regular content."""
        image = Image(data=b"img", format="png")
        inputs = {
            "question": "Analyze this",
            "image": image,
            "context": "Additional context here"
        }
        mock_signature.input_fields["context"] = Mock(desc="Context information")
        
        result = adapter.format_prompt(inputs, mock_signature)
        
        assert isinstance(result, list)
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        
        # Check all parts are present
        assert any("Analyze this" in str(part) for part in content)
        assert image in content
        assert any("Additional context here" in str(part) for part in content)
    
    def test_list_with_images(self, adapter):
        """Test list containing multiple images."""
        sig = Mock()
        sig.input_fields = {"images": Mock(desc="Images to compare")}
        sig.output_fields = {}
        
        img1 = Image(data=b"img1", format="png")
        img2 = Image(data=b"img2", format="jpeg")
        inputs = {"images": [img1, img2]}
        
        result = adapter.format_prompt(inputs, sig)
        
        assert isinstance(result, list)
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert img1 in content
        assert img2 in content
    
    def test_history_object_to_messages(self, adapter):
        """Test History object conversion to messages."""
        # Create a mock History that has to_messages method
        history = MagicMock()
        history.__class__ = History
        history.to_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        sig = Mock()
        sig.input_fields = {"conversation": Mock(desc="Previous conversation")}
        sig.output_fields = {}
        
        inputs = {"conversation": history}
        
        result = adapter.format_prompt(inputs, sig)
        
        # Should return the History's messages directly
        assert result == history.to_messages.return_value
        history.to_messages.assert_called_once()
    
    def test_history_without_to_messages(self, adapter):
        """Test History fallback when to_messages not available."""
        # Create a mock History without to_messages
        history = MagicMock()
        history.__class__ = History
        del history.to_messages
        history.__str__.return_value = "History content"
        
        sig = Mock()
        sig.input_fields = {"conversation": Mock(desc="Previous conversation")}
        sig.output_fields = {}
        
        inputs = {"conversation": history}
        
        result = adapter.format_prompt(inputs, sig)
        
        assert isinstance(result, list)
        assert result[0]["role"] == "user"
        assert "History content" in str(result[0]["content"])
    
    def test_no_signature_fields(self, adapter):
        """Test adapter works without signature field definitions."""
        sig = Mock()
        delattr(sig, "input_fields")
        delattr(sig, "output_fields")
        
        image = Image(data=b"img", format="png")
        inputs = {"prompt": "Test", "image": image}
        
        result = adapter.format_prompt(inputs, sig)
        
        assert isinstance(result, list)
        assert result[0]["role"] == "user"
        assert image in result[0]["content"]
    
    def test_output_hints_preserved(self, adapter, mock_signature):
        """Test output hints are included in multimodal format."""
        image = Image(data=b"img", format="png")
        inputs = {"question": "Analyze", "image": image}
        
        result = adapter.format_prompt(inputs, mock_signature)
        
        assert isinstance(result, list)
        content = result[0]["content"]
        
        # Output hints should be included
        assert any("Please provide:" in str(part) for part in content)
        assert any("Confidence score" in str(part) for part in content)
        assert any("0.0 and 1.0" in str(part) for part in content)


class TestJSONAdapterConsistency:
    """Test JSONAdapter maintains consistent interface."""
    
    def test_json_adapter_returns_string(self):
        """Test JSONAdapter still returns string format."""
        adapter = JSONAdapter()
        sig = Mock()
        sig.output_fields = {"result": Mock(desc="Result", annotation=str)}
        
        inputs = {"data": "test"}
        result = adapter.format_prompt(inputs, sig)
        
        # Should still return string (JSON doesn't need multimodal format)
        assert isinstance(result, str)
        assert "```json" in result
        assert '"data": "test"' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])