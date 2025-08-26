"""Integration tests for usage tracking system with real provider interactions."""

import tempfile
import time
from pathlib import Path

import pytest

from logillm.core.callbacks import CallbackContext, ProviderResponseEvent, callback_manager
from logillm.core.types import TokenUsage, Usage
from logillm.core.usage_tracker import (
    ExportFormat,
    TimeWindow,
    UsageTracker,
    UsageTrackingCallback,
    get_global_tracker,
    track_usage,
)
from logillm.providers.mock import MockProvider


@pytest.mark.integration
class TestUsageTrackerIntegration:
    """Integration tests for usage tracker with mock provider."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider for testing."""
        return MockProvider(responses=["This is a test response from the mock provider."])

    @pytest.fixture
    def tracker(self):
        """Create usage tracker for testing."""
        return UsageTracker(session_id="integration_test_session")

    def test_track_mock_provider_usage(self, tracker, mock_provider):
        """Test tracking usage from mock provider synchronously."""
        # Track usage manually
        messages = [{"role": "user", "content": "Hello, world!"}]
        completion = mock_provider.complete_sync(messages)

        # The mock provider should provide usage information
        assert completion.usage is not None
        assert completion.usage.tokens.total_tokens > 0

        # Track the usage
        record = tracker.track_usage_from_response(completion.usage, call_id="test_call")

        assert record is not None
        assert record.provider == "mock"
        assert record.model == "mock-model"
        assert record.tokens.total_tokens > 0
        assert record.cost == 0.0  # Mock provider is free
        assert record.call_id == "test_call"

        # Verify tracker state
        assert len(tracker) == 1
        stats = tracker.get_stats()
        assert stats.total_requests == 1
        assert stats.total_cost == 0.0  # Mock is free

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_track_mock_provider_usage_async(self, tracker, mock_provider):
        """Test tracking usage from mock provider asynchronously."""
        messages = [{"role": "user", "content": "Hello, async world!"}]
        completion = await mock_provider.complete(messages)

        # Track the usage
        record = tracker.track_usage_from_response(completion.usage, call_id="async_test_call")

        assert record is not None
        assert record.provider == "mock"
        assert record.call_id == "async_test_call"

    def test_usage_tracking_callback_integration(self, tracker, mock_provider):
        """Test integration with callback system."""
        # Register usage tracking callback
        callback = UsageTrackingCallback(tracker)
        callback_id = callback_manager.register_global(callback)

        try:
            # Create provider response event
            context = CallbackContext(call_id="callback_test")
            tokens = TokenUsage(input_tokens=100, output_tokens=50)
            usage = Usage(
                tokens=tokens,
                provider="mock",
                model="mock-model",
                cost=0.0,
                latency=0.5,
            )

            event = ProviderResponseEvent(
                context=context,
                provider=mock_provider,
                usage=usage,
            )

            # Emit event synchronously
            callback_manager.emit_sync(event)

            # Should have tracked usage automatically
            assert len(tracker) == 1

            history = tracker.get_history()
            record = history[0]
            assert record.provider == "mock"
            assert record.model == "mock-model"
            assert record.call_id == "callback_test"
            assert record.latency == 0.5

        finally:
            callback_manager.unregister(callback_id)

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_usage_tracking_callback_async_integration(self, tracker, mock_provider):
        """Test async integration with callback system."""
        callback = UsageTrackingCallback(tracker)
        callback_id = callback_manager.register_global(callback)

        try:
            context = CallbackContext(call_id="async_callback_test")
            tokens = TokenUsage(input_tokens=200, output_tokens=100)
            usage = Usage(
                tokens=tokens,
                provider="mock",
                model="mock-model",
                latency=1.2,
            )

            event = ProviderResponseEvent(
                context=context,
                provider=mock_provider,
                usage=usage,
            )

            # Emit event asynchronously
            await callback_manager.emit_async(event)

            # Should have tracked usage automatically
            assert len(tracker) == 1

            record = tracker.get_history()[0]
            assert record.call_id == "async_callback_test"
            assert record.latency == 1.2

        finally:
            callback_manager.unregister(callback_id)

    def test_context_manager_integration(self, tracker, mock_provider):
        """Test track_usage context manager with real provider."""
        with track_usage("mock", "mock-model", tracker=tracker, call_id="context_test") as ctx:
            # Simulate making a request
            start_time = time.time()
            messages = [{"role": "user", "content": "Context manager test"}]
            completion = mock_provider.complete_sync(messages)
            latency = time.time() - start_time

            # Set context values
            ctx["tokens"] = completion.usage.tokens
            ctx["latency"] = latency

        # Should have tracked usage
        assert len(tracker) == 1

        record = tracker.get_history()[0]
        assert record.provider == "mock"
        assert record.model == "mock-model"
        assert record.call_id == "context_test"
        assert record.latency > 0
        assert record.tokens.total_tokens > 0

    def test_batch_usage_tracking(self, tracker, mock_provider):
        """Test tracking usage from multiple requests."""
        # Simulate multiple requests
        messages_list = [[{"role": "user", "content": f"Request {i}"}] for i in range(5)]

        for i, messages in enumerate(messages_list):
            completion = mock_provider.complete_sync(messages)
            tracker.track_usage_from_response(completion.usage, call_id=f"batch_request_{i}")

        # Verify all requests were tracked
        assert len(tracker) == 5

        stats = tracker.get_stats()
        assert stats.total_requests == 5
        assert stats.total_tokens > 0
        assert stats.total_cost == 0.0  # Mock is free

        # Check individual records
        history = tracker.get_history()
        for i, record in enumerate(reversed(history)):  # History is reverse chronological
            assert record.call_id == f"batch_request_{i}"
            assert record.provider == "mock"

    def test_export_import_round_trip(self, tracker, mock_provider):
        """Test exporting and importing usage data."""
        # Generate some usage data
        for i in range(3):
            messages = [{"role": "user", "content": f"Export test {i}"}]
            completion = mock_provider.complete_sync(messages)
            tracker.track_usage_from_response(completion.usage, call_id=f"export_test_{i}")

        original_stats = tracker.get_stats()

        # Export to JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            tracker.export_usage(json_path, ExportFormat.JSON)

            # Create new tracker and import
            new_tracker = UsageTracker(session_id="import_test_session")
            count = new_tracker.import_usage(json_path)

            assert count == 3
            assert len(new_tracker) == 3

            # Compare stats
            imported_stats = new_tracker.get_stats()
            assert imported_stats.total_requests == original_stats.total_requests
            assert imported_stats.total_tokens == original_stats.total_tokens
            assert imported_stats.total_cost == original_stats.total_cost

            # Verify individual records
            imported_history = new_tracker.get_history()
            for record in imported_history:
                assert record.provider == "mock"
                assert record.model == "mock-model"
                assert record.call_id.startswith("export_test_")

        finally:
            json_path.unlink(missing_ok=True)

    def test_export_csv_with_real_data(self, tracker, mock_provider):
        """Test CSV export with real provider data."""
        # Generate usage data with different characteristics
        test_cases = [
            {"content": "Short message", "expected_tokens_min": 10},
            {
                "content": "This is a much longer message that should result in more tokens being used by the provider",
                "expected_tokens_min": 10,
            },  # Mock provider returns fixed tokens
            {"content": "Another test case with different content", "expected_tokens_min": 10},
        ]

        for i, case in enumerate(test_cases):
            messages = [{"role": "user", "content": case["content"]}]
            completion = mock_provider.complete_sync(messages)

            # Verify we got realistic token counts (MockProvider returns fixed 15 tokens total)
            assert completion.usage.tokens.total_tokens >= case["expected_tokens_min"]

            tracker.track_usage_from_response(completion.usage, call_id=f"csv_test_{i}")

        # Export to CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            tracker.export_usage(csv_path, ExportFormat.CSV)

            # Verify CSV contents
            import csv

            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Should have header + 3 data rows
            assert len(rows) == 4

            # Check header
            expected_header = [
                "timestamp",
                "provider",
                "model",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
                "cost",
                "latency",
                "session_id",
                "call_id",
            ]
            assert rows[0] == expected_header

            # Check data rows
            for i in range(1, 4):
                row = rows[i]
                assert row[1] == "mock"  # provider
                assert row[2] == "mock-model"  # model
                assert int(row[6]) > 0  # total_tokens
                assert float(row[7]) == 0.0  # cost (mock is free)
                assert row[10] == f"csv_test_{i - 1}"  # call_id

        finally:
            csv_path.unlink(missing_ok=True)

    def test_time_window_filtering(self, tracker, mock_provider):
        """Test usage statistics with time window filtering."""
        # Generate some recent usage
        for i in range(3):
            messages = [{"role": "user", "content": f"Recent message {i}"}]
            completion = mock_provider.complete_sync(messages)
            tracker.track_usage_from_response(completion.usage, call_id=f"recent_call_{i}")

        # Get stats for different time windows
        all_stats = tracker.get_stats()
        day_stats = tracker.get_stats(TimeWindow.DAY)
        hour_stats = tracker.get_stats(TimeWindow.HOUR)

        # All stats should include all records
        assert all_stats.total_requests == 3

        # Recent stats should include all recent records
        assert day_stats.total_requests == 3
        assert hour_stats.total_requests == 3

    def test_concurrent_tracking(self, tracker, mock_provider):
        """Test concurrent usage tracking with multiple threads."""
        import threading

        results = []
        errors = []

        def worker_thread(thread_id: int):
            try:
                for i in range(5):
                    messages = [{"role": "user", "content": f"Thread {thread_id} message {i}"}]
                    completion = mock_provider.complete_sync(messages)
                    record = tracker.track_usage_from_response(
                        completion.usage, call_id=f"thread_{thread_id}_call_{i}"
                    )
                    results.append(record)
                    time.sleep(0.01)  # Small delay to simulate real usage
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent tracking errors: {errors}"
        assert len(results) == 15  # 3 threads × 5 calls each
        assert len(tracker) == 15

        # Verify thread safety
        stats = tracker.get_stats()
        assert stats.total_requests == 15

        # Check that all call IDs are unique and present
        history = tracker.get_history()
        call_ids = [record.call_id for record in history]
        assert len(set(call_ids)) == 15  # All unique

        # Verify we have records from all threads
        for thread_id in range(3):
            thread_calls = [cid for cid in call_ids if cid.startswith(f"thread_{thread_id}_")]
            assert len(thread_calls) == 5

    def test_usage_limits_and_alerts(self, mock_provider):
        """Test usage limits and alert system."""
        alerts = []

        def alert_callback(alert_type: str, message: str, current_value: float, limit: float):
            alerts.append(
                {
                    "type": alert_type,
                    "message": message,
                    "current": current_value,
                    "limit": limit,
                }
            )

        # Create tracker with very low token limit (MockProvider returns 15 tokens per call)
        tracker = UsageTracker(daily_token_limit=50)  # Lower than 5 * 15 = 75 total tokens
        tracker.add_alert_callback(alert_callback)

        # Generate usage that exceeds limit (MockProvider gives 15 tokens each time)
        for i in range(8):  # 8 * 15 = 120 tokens > 50 limit
            messages = [{"role": "user", "content": f"Message {i}"}]
            completion = mock_provider.complete_sync(messages)
            tracker.track_usage_from_response(completion.usage, call_id=f"limit_test_{i}")

            # Check if we've hit the limit
            stats = tracker.get_stats(TimeWindow.DAY)
            if stats.total_tokens > 50:
                break

        # Should have triggered token limit alert
        assert len(alerts) > 0
        token_alerts = [a for a in alerts if a["type"] == "daily_token_limit"]
        assert len(token_alerts) > 0

        alert = token_alerts[0]
        assert alert["current"] > alert["limit"]
        assert "token limit exceeded" in alert["message"].lower()

    def test_global_tracker_integration(self, mock_provider):
        """Test integration with global tracker."""
        # Get global tracker
        global_tracker = get_global_tracker()
        original_count = len(global_tracker)

        # Use context manager without specifying tracker (should use global)
        with track_usage("mock", "mock-model", call_id="global_test") as ctx:
            messages = [{"role": "user", "content": "Global tracker test"}]
            completion = mock_provider.complete_sync(messages)
            ctx["tokens"] = completion.usage.tokens

        # Should have tracked usage in global tracker
        assert len(global_tracker) == original_count + 1

        # Verify the record
        history = global_tracker.get_history()
        latest_record = history[0]  # Most recent first
        assert latest_record.call_id == "global_test"
        assert latest_record.provider == "mock"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_streaming_usage_tracking(self, tracker):
        """Test usage tracking with streaming responses (simulated)."""
        # Note: MockProvider doesn't support real streaming, so we simulate it
        mock_provider = MockProvider(
            responses=["Stream chunk 1", "Stream chunk 2", "Stream chunk 3"]
        )

        # Simulate streaming by making multiple calls
        messages = [{"role": "user", "content": "Streaming test"}]
        total_tokens = TokenUsage()

        for i in range(3):  # Simulate 3 chunks
            completion = await mock_provider.complete(messages)
            # In real streaming, we'd accumulate tokens across chunks
            total_tokens = total_tokens + completion.usage.tokens

            # Track partial usage (in real streaming, this might be done differently)
            if i == 2:  # Only track on final chunk
                record = tracker.track_usage_from_response(
                    completion.usage, call_id="streaming_test"
                )

        # Should have one final usage record
        assert len(tracker) == 1
        record = tracker.get_history()[0]
        assert record.call_id == "streaming_test"
        assert record.tokens.total_tokens > 0

    def test_usage_tracker_memory_efficiency(self, mock_provider):
        """Test that usage tracker respects memory limits."""
        # Create tracker with small history limit
        tracker = UsageTracker(max_history=5)

        # Generate more records than limit
        for i in range(10):
            messages = [{"role": "user", "content": f"Memory test {i}"}]
            completion = mock_provider.complete_sync(messages)
            tracker.track_usage_from_response(completion.usage, call_id=f"memory_test_{i}")

        # Should only keep the last 5 records
        assert len(tracker) == 5

        # Verify we kept the most recent records
        history = tracker.get_history()
        call_ids = [record.call_id for record in history]

        # Should have records 5-9 (most recent)
        for i in range(5, 10):
            assert f"memory_test_{i}" in call_ids

        # Should not have records 0-4 (oldest)
        for i in range(5):
            assert f"memory_test_{i}" not in call_ids
