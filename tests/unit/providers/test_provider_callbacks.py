"""Tests for provider callback functionality."""


import pytest

from logillm.core.callbacks import (
    AbstractCallback,
    CallbackManager,
    ProviderErrorEvent,
    ProviderRequestEvent,
    ProviderResponseEvent,
)
from logillm.providers.mock import MockProvider


class TestProviderCallbacks:
    """Test provider callback functionality."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Clear manager before each test
        manager = CallbackManager()
        manager.clear()
        manager.enable()  # Ensure callbacks are enabled
        yield
        # Clear after test
        manager.clear()

    @pytest.mark.asyncio
    async def test_provider_emits_request_event(self):
        """Test that provider emits request event."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        await provider.complete(messages)

        assert len(events) == 1
        assert isinstance(events[0], ProviderRequestEvent)
        assert events[0].provider == provider
        assert events[0].messages == messages

    @pytest.mark.asyncio
    async def test_provider_emits_response_event(self):
        """Test that provider emits response event."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_provider_response(self, event: ProviderResponseEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        result = await provider.complete(messages)

        assert len(events) == 1
        assert isinstance(events[0], ProviderResponseEvent)
        assert events[0].provider == provider
        assert events[0].request_messages == messages
        assert events[0].response == result
        assert events[0].duration > 0

    @pytest.mark.asyncio
    async def test_provider_emits_error_event(self):
        """Test that provider emits error event on failure."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_provider_error(self, event: ProviderErrorEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        # Create provider that will error
        provider = MockProvider(error_rate=1.0)  # 100% error rate
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception, match="Mock error"):
            await provider.complete(messages)

        assert len(events) == 1
        assert isinstance(events[0], ProviderErrorEvent)
        assert events[0].provider == provider
        assert events[0].messages == messages
        assert "Mock error" in str(events[0].error)

    @pytest.mark.asyncio
    async def test_provider_callbacks_with_usage_tracking(self):
        """Test that provider events include usage information."""
        response_events = []

        class TestHandler(AbstractCallback):
            async def on_provider_response(self, event: ProviderResponseEvent):
                response_events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        await provider.complete(messages)

        assert len(response_events) == 1
        event = response_events[0]
        assert event.usage is not None
        assert event.usage.tokens.input_tokens == 10  # MockProvider returns fixed values
        assert event.usage.tokens.output_tokens == 5

    @pytest.mark.asyncio
    async def test_provider_callbacks_disabled(self):
        """Test that callbacks can be disabled."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                events.append(event)
            async def on_provider_response(self, event: ProviderResponseEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())
        manager.disable()  # Disable callbacks

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        await provider.complete(messages)

        assert len(events) == 0  # No events should be emitted

    @pytest.mark.asyncio
    async def test_provider_context_propagation(self):
        """Test that provider maintains context across events."""
        request_contexts = []
        response_contexts = []

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                request_contexts.append(event.context)
            async def on_provider_response(self, event: ProviderResponseEvent):
                response_contexts.append(event.context)

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        await provider.complete(messages)

        assert len(request_contexts) == 1
        assert len(response_contexts) == 1
        assert request_contexts[0].call_id == response_contexts[0].call_id  # Same context

    @pytest.mark.asyncio
    async def test_provider_multiple_calls_different_contexts(self):
        """Test that each provider call has its own context."""
        contexts = []

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                contexts.append(event.context)

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        # Make multiple calls
        await provider.complete(messages)
        await provider.complete(messages)
        await provider.complete(messages)

        assert len(contexts) == 3
        # Each context should be unique
        context_ids = [ctx.call_id for ctx in contexts]
        assert len(set(context_ids)) == 3

    @pytest.mark.asyncio
    async def test_provider_streaming_callbacks(self):
        """Test that streaming operations emit appropriate events."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                events.append(("request", event))
            async def on_provider_response(self, event: ProviderResponseEvent):
                events.append(("response", event))

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test streaming response")
        messages = [{"role": "user", "content": "Hello"}]

        # Stream the response
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks).strip() == "Test streaming response"

        # Should have emitted request and response events
        assert len(events) >= 2
        assert events[0][0] == "request"
        assert events[1][0] == "response"

    def test_provider_sync_complete_callbacks(self):
        """Test that sync complete method emits callbacks."""
        events = []

        class TestHandler(AbstractCallback):
            async def on_provider_response(self, event: ProviderResponseEvent):
                events.append(event)

        manager = CallbackManager()
        manager.register(TestHandler())

        provider = MockProvider(response_text="Test response")
        messages = [{"role": "user", "content": "Hello"}]

        # Use sync method
        result = provider.complete_sync(messages)

        assert result.text == "Test response"
        assert len(events) == 1
        assert isinstance(events[0], ProviderResponseEvent)

    @pytest.mark.asyncio
    async def test_provider_with_retry_callbacks(self):
        """Test that retries emit multiple callback events."""
        attempt_count = []

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                attempt_count.append(1)

        manager = CallbackManager()
        manager.register(TestHandler())

        # Provider that will fail twice then succeed
        # Use a counter to control failures
        class RetryProvider(MockProvider):
            def __init__(self):
                super().__init__(response_text="Success")
                self.attempt = 0

            async def _complete_impl(self, messages, **kwargs):
                self.attempt += 1
                if self.attempt < 3:
                    raise Exception(f"Mock error (attempt {self.attempt})")
                return await super()._complete_impl(messages, **kwargs)

        provider = RetryProvider()
        messages = [{"role": "user", "content": "Hello"}]

        # This should retry twice then succeed
        result = await provider.complete_with_retry(messages)

        # Should have 3 attempts (2 failures + 1 success)
        assert len(attempt_count) == 3
        assert result.text == "Success"

    @pytest.mark.asyncio
    async def test_provider_callback_performance(self):
        """Test that callbacks have acceptable performance overhead."""
        import time

        # Time without callbacks
        manager = CallbackManager()
        manager.disable()

        provider = MockProvider(response_text="Test", latency=0.001)
        messages = [{"role": "user", "content": "Hello"}]

        start = time.time()
        for _ in range(100):
            await provider.complete(messages)
        time_without = time.time() - start

        # Time with callbacks
        manager.enable()

        class TestHandler(AbstractCallback):
            async def on_provider_request(self, event: ProviderRequestEvent):
                pass
            async def on_provider_response(self, event: ProviderResponseEvent):
                pass

        manager.register(TestHandler())

        start = time.time()
        for _ in range(100):
            await provider.complete(messages)
        time_with = time.time() - start

        # Callbacks should add less than 50% overhead
        overhead = time_with / time_without
        assert overhead < 1.5, f"Callback overhead too high: {overhead:.2f}x"
