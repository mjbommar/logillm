"""Unit tests for History type."""

from logillm.core.signatures.types import History


def test_history_to_messages():
    """Test History.to_messages() method."""
    # Test with properly formatted messages
    history = History(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )

    messages = history.to_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi there!"}


def test_history_to_messages_missing_role():
    """Test to_messages() with messages missing 'role'."""
    history = History(
        messages=[
            {"content": "Hello"},  # Missing role
            {"text": "Hi there!"},  # Missing both role and content
        ]
    )

    messages = history.to_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello"}  # Default role added
    assert messages[1] == {"role": "user", "content": "Hi there!"}  # Text converted to content


def test_history_to_messages_non_dict():
    """Test to_messages() with non-dict messages."""
    history = History(
        messages=[
            "Simple string message",
            123,  # Non-string
        ]
    )

    messages = history.to_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Simple string message"}
    assert messages[1] == {"role": "user", "content": "123"}


def test_history_add_message():
    """Test adding messages to history."""
    history = History(messages=[])

    history.add_message("user", "Question?")
    history.add_message("assistant", "Answer!")

    assert len(history) == 2
    messages = history.to_messages()
    assert messages[0] == {"role": "user", "content": "Question?"}
    assert messages[1] == {"role": "assistant", "content": "Answer!"}


def test_history_get_last_n():
    """Test getting last n messages."""
    history = History(
        messages=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
        ]
    )

    last_two = history.get_last_n(2)
    assert len(last_two) == 2
    assert last_two[0]["content"] == "Third"
    assert last_two[1]["content"] == "Fourth"

    # Test edge cases
    assert history.get_last_n(0) == []
    assert len(history.get_last_n(10)) == 4  # More than available


def test_history_clear():
    """Test clearing history."""
    history = History(
        messages=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )

    assert len(history) == 2
    history.clear()
    assert len(history) == 0
    assert history.messages == []


def test_history_metadata():
    """Test history with metadata."""
    metadata = {"session_id": "123", "user_id": "abc"}
    history = History(messages=[], metadata=metadata)

    assert history.metadata == metadata
    assert history.metadata["session_id"] == "123"
