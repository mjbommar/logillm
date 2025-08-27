"""Unit tests for model capabilities registry."""

from logillm.providers.capabilities import (
    ModelCapabilities,
    get_context_window,
    get_max_tokens,
    get_model_capabilities,
    supports_audio,
    supports_function_calling,
    supports_vision,
)


def test_model_capabilities_dataclass():
    """Test ModelCapabilities dataclass."""
    caps = ModelCapabilities(
        vision=True,
        audio=False,
        function_calling=True,
        streaming=True,
        max_tokens=4096,
        context_window=128000,
    )

    assert caps.vision is True
    assert caps.audio is False
    assert caps.function_calling is True
    assert caps.streaming is True
    assert caps.max_tokens == 4096
    assert caps.context_window == 128000

    # Test to_dict
    d = caps.to_dict()
    assert d["vision"] is True
    assert d["audio"] is False
    assert d["max_tokens"] == 4096


def test_gpt_4_1_capabilities():
    """Test GPT-4.1 model capabilities."""
    caps = get_model_capabilities("gpt-4.1")
    assert caps is not None
    assert caps.vision is True  # GPT-4.1 supports vision
    assert caps.audio is False
    assert caps.function_calling is True
    assert caps.streaming is True
    assert caps.max_tokens == 128000
    assert caps.context_window == 128000


def test_claude_4_capabilities():
    """Test Claude 4 model capabilities."""
    caps = get_model_capabilities("claude-4-opus-20250514")
    assert caps is not None
    assert caps.vision is True
    assert caps.audio is False  # Anthropic doesn't support audio
    assert caps.function_calling is True
    assert caps.streaming is True
    assert caps.context_window == 200000


def test_gpt_4o_audio_capabilities():
    """Test GPT-4o audio model capabilities."""
    caps = get_model_capabilities("gpt-4o-audio-preview")
    assert caps is not None
    assert caps.vision is True
    assert caps.audio is True  # This model supports audio
    assert caps.function_calling is True


def test_gemini_capabilities():
    """Test Gemini model capabilities."""
    caps = get_model_capabilities("gemini-2.5-flash")
    assert caps is not None
    assert caps.vision is True
    assert caps.audio is True  # Gemini supports audio
    assert caps.function_calling is True
    assert caps.context_window == 1048576  # 1M tokens


def test_unknown_model():
    """Test unknown model returns None."""
    caps = get_model_capabilities("unknown-model")
    assert caps is None


def test_supports_vision():
    """Test supports_vision helper function."""
    assert supports_vision("gpt-4.1") is True
    assert supports_vision("claude-4-opus-20250514") is True
    assert supports_vision("gpt-4") is False  # Legacy GPT-4 doesn't support vision
    assert supports_vision("gpt-3.5-turbo") is False
    assert supports_vision("unknown-model") is False


def test_supports_audio():
    """Test supports_audio helper function."""
    assert supports_audio("gpt-4o-audio-preview") is True
    assert supports_audio("gemini-2.5-flash") is True
    assert supports_audio("gpt-4.1") is False
    assert supports_audio("claude-4-opus-20250514") is False
    assert supports_audio("unknown-model") is False


def test_supports_function_calling():
    """Test supports_function_calling helper function."""
    assert supports_function_calling("gpt-4.1") is True
    assert supports_function_calling("claude-4-opus-20250514") is True
    assert supports_function_calling("gpt-3.5-turbo") is True
    assert supports_function_calling("unknown-model") is False


def test_get_max_tokens():
    """Test get_max_tokens helper function."""
    assert get_max_tokens("gpt-4.1") == 128000
    assert get_max_tokens("claude-4-opus-20250514") == 4096
    assert get_max_tokens("gemini-2.5-flash") == 8192
    assert get_max_tokens("unknown-model") is None


def test_get_context_window():
    """Test get_context_window helper function."""
    assert get_context_window("gpt-4.1") == 128000
    assert get_context_window("claude-4-opus-20250514") == 200000
    assert get_context_window("gemini-2.5-flash") == 1048576
    assert get_context_window("gpt-4") == 8192
    assert get_context_window("unknown-model") is None
