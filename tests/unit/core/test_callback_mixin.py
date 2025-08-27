"""Unit tests for CallbackMixin functionality."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from logillm.core.callback_mixin import CallbackMixin, get_current_context


class TestCallbackMixin:
    """Test CallbackMixin functionality."""

    def test_mixin_initialization(self):
        """Test that CallbackMixin initializes properly."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()
        assert obj._callback_context is None
        assert obj._callback_manager is None
        assert isinstance(obj._callback_enabled, bool)

    def test_callback_enabled_from_environment(self):
        """Test that callbacks can be controlled via environment variable."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        # Test with callbacks enabled (default)
        with patch.dict("os.environ", {"LOGILLM_CALLBACKS_ENABLED": "1"}):
            obj = TestClass()
            assert obj._callback_enabled is True

        # Test with callbacks disabled
        with patch.dict("os.environ", {"LOGILLM_CALLBACKS_ENABLED": "0"}):
            obj = TestClass()
            assert obj._callback_enabled is False

    def test_get_callback_manager_lazy_import(self):
        """Test that callback manager is lazily imported."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()

        # Initially should be None
        assert obj._callback_manager is None

        # When disabled, should return None without importing
        obj._callback_enabled = False
        manager = obj._get_callback_manager()
        assert manager is None
        assert obj._callback_manager is None

    def test_create_context(self):
        """Test context creation."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()

        # Create context without parent
        context = obj._create_context()
        assert context.call_id is not None
        assert context.parent_call_id is None
        assert obj._callback_context == context

        # Create context with parent
        parent_context = MagicMock()
        parent_context.call_id = "parent-123"
        child_context = obj._create_context(parent=parent_context)
        assert child_context.call_id is not None
        assert child_context.parent_call_id == "parent-123"

    def test_check_callbacks_enabled(self):
        """Test checking if callbacks are enabled."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()

        # When instance disabled
        obj._callback_enabled = False
        assert obj._check_callbacks_enabled() is False

        # When instance enabled but manager is None
        obj._callback_enabled = True
        with patch.object(obj, "_get_callback_manager", return_value=None):
            assert obj._check_callbacks_enabled() is False

    def test_set_callback_enabled(self):
        """Test enabling/disabling callbacks."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()

        obj._set_callback_enabled(True)
        assert obj._callback_enabled is True

        obj._set_callback_enabled(False)
        assert obj._callback_enabled is False

    @pytest.mark.asyncio
    async def test_emit_async_safe_with_running_loop(self):
        """Test safe async emission with running event loop."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()
        event = MagicMock()

        # Mock the emit_async method to return a coroutine
        async def mock_emit():
            pass

        obj._emit_async = MagicMock(return_value=mock_emit())

        # When event loop is running
        obj._emit_async_safe(event)

        # Give the task a chance to execute
        await asyncio.sleep(0.01)

    def test_emit_async_safe_without_loop(self):
        """Test safe async emission without event loop falls back to sync."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()
        event = MagicMock()

        # Mock the sync emission
        obj._emit_sync = MagicMock()

        # Mock to simulate no running loop
        with patch("asyncio.get_event_loop", side_effect=RuntimeError("No event loop")):
            obj._emit_async_safe(event)
            obj._emit_sync.assert_called_once_with(event)

    def test_context_manager(self):
        """Test callback context manager."""

        class TestClass(CallbackMixin):
            def __init__(self):
                CallbackMixin.__init__(self)

        obj = TestClass()

        # Create a mock context
        from logillm.core.callbacks import CallbackContext

        context = CallbackContext(call_id="test-123")

        # Initially no context
        assert get_current_context() is None

        # Use context manager
        with obj._with_callback_context(context):
            # Context should be set
            current = get_current_context()
            assert current == context

        # Context should be reset
        assert get_current_context() is None


class TestModuleIntegration:
    """Test that Module class properly integrates CallbackMixin."""

    def test_module_has_callback_mixin(self):
        """Test that Module inherits from CallbackMixin."""
        from logillm.core.modules import Module

        assert issubclass(Module, CallbackMixin)

    def test_module_initializes_mixin(self):
        """Test that Module properly initializes CallbackMixin."""
        from logillm.core.modules import BaseModule

        module = BaseModule()

        # Check that mixin attributes are present
        assert hasattr(module, "_callback_context")
        assert hasattr(module, "_callback_enabled")
        assert hasattr(module, "_callback_manager")

        # Check that mixin methods are available
        assert hasattr(module, "_emit_async")
        assert hasattr(module, "_emit_sync")
        assert hasattr(module, "_create_context")
