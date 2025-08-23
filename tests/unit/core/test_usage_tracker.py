"""Unit tests for usage tracking system."""

import csv
import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from logillm.core.callbacks import CallbackContext, ProviderResponseEvent
from logillm.core.types import TokenUsage, Usage
from logillm.core.usage_tracker import (
    DEFAULT_PRICING,
    ExportFormat,
    PricingInfo,
    TimeWindow,
    UsageRecord,
    UsageStats,
    UsageTracker,
    UsageTrackingCallback,
    get_global_tracker,
    set_global_tracker,
    track_usage,
)


class TestPricingInfo:
    """Test PricingInfo class."""

    def test_init(self):
        """Test PricingInfo initialization."""
        pricing = PricingInfo("openai", "gpt-4", 0.03, 0.06)
        assert pricing.provider == "openai"
        assert pricing.model == "gpt-4"
        assert pricing.input_price_per_1k == 0.03
        assert pricing.output_price_per_1k == 0.06
        assert pricing.reasoning_price_per_1k == 0.0

    def test_init_with_reasoning_price(self):
        """Test PricingInfo initialization with reasoning price."""
        pricing = PricingInfo("openai", "o1", 0.015, 0.06, 0.06)
        assert pricing.reasoning_price_per_1k == 0.06

    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        pricing = PricingInfo("openai", "gpt-4", 0.03, 0.06)
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)

        cost = pricing.calculate_cost(tokens)
        expected = (1000 / 1000.0) * 0.03 + (500 / 1000.0) * 0.06
        assert cost == pytest.approx(expected)

    def test_calculate_cost_with_reasoning(self):
        """Test cost calculation with reasoning tokens."""
        pricing = PricingInfo("openai", "o1", 0.015, 0.06, 0.06)
        tokens = TokenUsage(input_tokens=1000, output_tokens=500, reasoning_tokens=2000)

        cost = pricing.calculate_cost(tokens)
        expected = (1000 / 1000.0) * 0.015 + (500 / 1000.0) * 0.06 + (2000 / 1000.0) * 0.06
        assert cost == pytest.approx(expected)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        pricing = PricingInfo("openai", "gpt-4", 0.03, 0.06)
        tokens = TokenUsage()

        cost = pricing.calculate_cost(tokens)
        assert cost == 0.0


class TestUsageRecord:
    """Test UsageRecord class."""

    def test_init(self):
        """Test UsageRecord initialization."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        record = UsageRecord(
            provider="openai",
            model="gpt-4",
            tokens=tokens,
            cost=0.012,
        )

        assert record.provider == "openai"
        assert record.model == "gpt-4"
        assert record.tokens == tokens
        assert record.cost == 0.012
        assert isinstance(record.timestamp, datetime)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        record = UsageRecord(
            provider="openai",
            model="gpt-4",
            tokens=tokens,
            cost=0.012,
            session_id="test_session",
            call_id="test_call",
            latency=1.5,
        )

        data = record.to_dict()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"
        assert data["cost"] == 0.012
        assert data["session_id"] == "test_session"
        assert data["call_id"] == "test_call"
        assert data["latency"] == 1.5
        assert isinstance(data["timestamp"], str)  # Should be ISO format

    def test_from_dict(self):
        """Test creation from dictionary."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        original = UsageRecord(
            provider="openai",
            model="gpt-4",
            tokens=tokens,
            cost=0.012,
        )

        data = original.to_dict()
        restored = UsageRecord.from_dict(data)

        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.tokens.input_tokens == original.tokens.input_tokens
        assert restored.tokens.output_tokens == original.tokens.output_tokens
        assert restored.cost == original.cost


class TestUsageStats:
    """Test UsageStats class."""

    def test_init(self):
        """Test UsageStats initialization."""
        stats = UsageStats()
        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_reasoning_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.avg_latency == 0.0

    def test_total_tokens(self):
        """Test total tokens property."""
        stats = UsageStats(
            total_input_tokens=1000,
            total_output_tokens=500,
            total_reasoning_tokens=200,
        )
        assert stats.total_tokens == 1700

    def test_duration_hours(self):
        """Test duration calculation."""
        start = datetime.now()
        end = start + timedelta(hours=2, minutes=30)
        stats = UsageStats(start_time=start, end_time=end)

        assert stats.duration_hours == pytest.approx(2.5)

    def test_cost_per_request(self):
        """Test cost per request calculation."""
        stats = UsageStats(total_cost=10.0, total_requests=5)
        assert stats.cost_per_request == 2.0

        # Zero requests
        stats_zero = UsageStats(total_cost=10.0, total_requests=0)
        assert stats_zero.cost_per_request == 0.0

    def test_tokens_per_request(self):
        """Test tokens per request calculation."""
        stats = UsageStats(
            total_input_tokens=1000,
            total_output_tokens=500,
            total_requests=5,
        )
        assert stats.tokens_per_request == 300.0


class TestUsageTracker:
    """Test UsageTracker class."""

    def test_init(self):
        """Test UsageTracker initialization."""
        tracker = UsageTracker()
        assert len(tracker) == 0
        assert not bool(tracker)
        assert tracker._session_id.startswith("session_")

    def test_init_with_params(self):
        """Test UsageTracker initialization with parameters."""
        custom_pricing = {("custom", "model"): PricingInfo("custom", "model", 0.01, 0.02)}

        tracker = UsageTracker(
            max_history=5000,
            session_id="test_session",
            custom_pricing=custom_pricing,
            daily_cost_limit=100.0,
            daily_token_limit=1000000,
        )

        assert tracker._session_id == "test_session"
        assert tracker._daily_cost_limit == 100.0
        assert tracker._daily_token_limit == 1000000
        assert ("custom", "model") in tracker._pricing

    def test_add_pricing(self):
        """Test adding custom pricing."""
        tracker = UsageTracker()
        pricing = PricingInfo("custom", "model", 0.01, 0.02)

        tracker.add_pricing("custom", "model", pricing)
        assert tracker.get_pricing("custom", "model") == pricing

    def test_get_pricing(self):
        """Test getting pricing information."""
        tracker = UsageTracker()

        # Should get default pricing
        pricing = tracker.get_pricing("openai", "gpt-4")
        assert pricing is not None
        assert pricing.provider == "openai"
        assert pricing.model == "gpt-4"

        # Should return None for unknown provider/model
        unknown = tracker.get_pricing("unknown", "model")
        assert unknown is None

    def test_track_usage(self):
        """Test basic usage tracking."""
        tracker = UsageTracker()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)

        record = tracker.track_usage("openai", "gpt-4", tokens, call_id="test_call")

        assert record.provider == "openai"
        assert record.model == "gpt-4"
        assert record.tokens == tokens
        assert record.call_id == "test_call"
        assert record.cost > 0  # Should calculate cost from default pricing

        assert len(tracker) == 1
        assert bool(tracker)

    def test_track_usage_unknown_pricing(self):
        """Test tracking usage with unknown pricing."""
        tracker = UsageTracker()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)

        record = tracker.track_usage("unknown", "model", tokens)

        assert record.cost == 0.0  # No pricing info, so cost is 0

    def test_track_usage_from_response(self):
        """Test tracking usage from Usage object."""
        tracker = UsageTracker()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        usage = Usage(
            tokens=tokens,
            provider="openai",
            model="gpt-4",
            latency=1.5,
        )

        record = tracker.track_usage_from_response(usage, call_id="test_call")

        assert record is not None
        assert record.provider == "openai"
        assert record.model == "gpt-4"
        assert record.tokens == tokens
        assert record.latency == 1.5
        assert record.call_id == "test_call"

    def test_track_usage_from_response_missing_info(self):
        """Test tracking usage from incomplete Usage object."""
        tracker = UsageTracker()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        usage = Usage(tokens=tokens)  # Missing provider/model

        record = tracker.track_usage_from_response(usage)
        assert record is None

    def test_get_stats_empty(self):
        """Test getting stats with no usage."""
        tracker = UsageTracker()
        stats = tracker.get_stats()

        assert stats.total_requests == 0
        assert stats.total_cost == 0.0

    def test_get_stats_with_data(self):
        """Test getting stats with usage data."""
        tracker = UsageTracker()

        # Add some usage records
        tokens1 = TokenUsage(input_tokens=1000, output_tokens=500)
        tokens2 = TokenUsage(input_tokens=800, output_tokens=400)

        tracker.track_usage("openai", "gpt-4", tokens1, latency=1.5)
        tracker.track_usage("anthropic", "claude-3", tokens2, latency=2.0)

        stats = tracker.get_stats()

        assert stats.total_requests == 2
        assert stats.total_input_tokens == 1800
        assert stats.total_output_tokens == 900
        assert stats.total_cost > 0
        assert stats.avg_latency == 1.75
        assert len(stats.providers) == 2
        assert stats.providers["openai"] == 1
        assert stats.providers["anthropic"] == 1

    def test_get_stats_with_time_window(self):
        """Test getting stats with time window."""
        tracker = UsageTracker()

        # Add usage record now
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.track_usage("openai", "gpt-4", tokens)

        # Should have data for recent time windows
        day_stats = tracker.get_stats(TimeWindow.DAY)
        assert day_stats.total_requests == 1

        hour_stats = tracker.get_stats(TimeWindow.HOUR)
        assert hour_stats.total_requests == 1

    def test_get_history(self):
        """Test getting usage history."""
        tracker = UsageTracker()

        # Add some records
        for i in range(5):
            tokens = TokenUsage(input_tokens=100 * i, output_tokens=50 * i)
            tracker.track_usage("openai", "gpt-4", tokens)

        # Get all history (reverse chronological order)
        history = tracker.get_history()
        assert len(history) == 5
        assert history[0].tokens.input_tokens == 400  # Most recent first

        # Get limited history
        limited = tracker.get_history(limit=3)
        assert len(limited) == 3

    def test_clear_history(self):
        """Test clearing usage history."""
        tracker = UsageTracker()

        # Add some records
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.track_usage("openai", "gpt-4", tokens)
        assert len(tracker) == 1

        # Clear history
        tracker.clear_history()
        assert len(tracker) == 0
        assert not bool(tracker)

    def test_export_json(self):
        """Test exporting usage to JSON."""
        tracker = UsageTracker(session_id="test_session")

        # Add some usage
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.track_usage("openai", "gpt-4", tokens, call_id="test_call")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            tracker.export_usage(filepath, ExportFormat.JSON)

            # Verify file contents
            with open(filepath) as f:
                data = json.load(f)

            assert data["session_id"] == "test_session"
            assert data["total_records"] == 1
            assert len(data["records"]) == 1
            assert "stats" in data

            record = data["records"][0]
            assert record["provider"] == "openai"
            assert record["model"] == "gpt-4"
            assert record["call_id"] == "test_call"

        finally:
            filepath.unlink(missing_ok=True)

    def test_export_csv(self):
        """Test exporting usage to CSV."""
        tracker = UsageTracker()

        # Add some usage
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.track_usage("openai", "gpt-4", tokens, call_id="test_call", latency=1.5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = Path(f.name)

        try:
            tracker.export_usage(filepath, ExportFormat.CSV)

            # Verify file contents
            with open(filepath, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2  # Header + 1 data row

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

            # Check data row
            data_row = rows[1]
            assert data_row[1] == "openai"  # provider
            assert data_row[2] == "gpt-4"  # model
            assert data_row[3] == "1000"  # input_tokens
            assert data_row[4] == "500"  # output_tokens
            assert data_row[8] == "1.5"  # latency
            assert data_row[10] == "test_call"  # call_id

        finally:
            filepath.unlink(missing_ok=True)

    def test_export_csv_empty(self):
        """Test exporting empty usage to CSV."""
        tracker = UsageTracker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = Path(f.name)

        try:
            tracker.export_usage(filepath, ExportFormat.CSV)

            # Verify file has header only
            with open(filepath, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 1  # Header only

        finally:
            filepath.unlink(missing_ok=True)

    def test_import_usage(self):
        """Test importing usage from JSON."""
        # Create a tracker with some data
        original_tracker = UsageTracker(session_id="original_session")
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        original_tracker.track_usage("openai", "gpt-4", tokens, call_id="test_call")

        # Export to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            original_tracker.export_usage(filepath, ExportFormat.JSON)

            # Create new tracker and import
            new_tracker = UsageTracker()
            assert len(new_tracker) == 0

            count = new_tracker.import_usage(filepath)
            assert count == 1
            assert len(new_tracker) == 1

            # Verify imported data
            history = new_tracker.get_history()
            record = history[0]
            assert record.provider == "openai"
            assert record.model == "gpt-4"
            assert record.call_id == "test_call"
            assert record.tokens.input_tokens == 1000

        finally:
            filepath.unlink(missing_ok=True)

    def test_thread_safety(self):
        """Test thread safety of usage tracking."""
        tracker = UsageTracker()
        results = []
        errors = []

        def track_usage_worker(thread_id: int):
            try:
                for i in range(10):
                    tokens = TokenUsage(input_tokens=100, output_tokens=50)
                    record = tracker.track_usage(
                        "openai", "gpt-4", tokens, call_id=f"thread_{thread_id}_call_{i}"
                    )
                    results.append(record)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=track_usage_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Threading errors occurred: {errors}"
        assert len(results) == 50  # 5 threads * 10 records each
        assert len(tracker) == 50

        # Verify all records are tracked
        stats = tracker.get_stats()
        assert stats.total_requests == 50

    def test_circular_buffer(self):
        """Test circular buffer behavior with max_history."""
        tracker = UsageTracker(max_history=3)

        # Add more records than max_history
        for i in range(5):
            tokens = TokenUsage(input_tokens=100 + i, output_tokens=50 + i)
            tracker.track_usage("openai", "gpt-4", tokens, call_id=f"call_{i}")

        # Should only keep last 3 records
        assert len(tracker) == 3

        history = tracker.get_history()
        # Should have the most recent records (call_2, call_3, call_4)
        call_ids = [record.call_id for record in history]
        assert "call_4" in call_ids  # Most recent
        assert "call_3" in call_ids
        assert "call_2" in call_ids
        assert "call_0" not in call_ids  # Should be evicted
        assert "call_1" not in call_ids  # Should be evicted

    def test_session_context_manager(self):
        """Test session context manager."""
        tracker = UsageTracker(session_id="original_session")

        # Track usage in different session
        with tracker.session("temp_session"):
            tokens = TokenUsage(input_tokens=1000, output_tokens=500)
            record = tracker.track_usage("openai", "gpt-4", tokens)
            assert record.session_id == "temp_session"

        # Should restore original session
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        record = tracker.track_usage("openai", "gpt-4", tokens)
        assert record.session_id == "original_session"

    def test_alert_callbacks(self):
        """Test usage limit alerts."""
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

        # Create tracker with low limits
        tracker = UsageTracker(
            daily_cost_limit=0.01,  # Very low limit
            daily_token_limit=100,  # Very low limit
        )
        tracker.add_alert_callback(alert_callback)

        # Track usage that exceeds limits
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.track_usage("openai", "gpt-4", tokens)  # Should exceed both limits

        # Should have triggered both alerts
        assert len(alerts) == 2

        alert_types = [alert["type"] for alert in alerts]
        assert "daily_cost_limit" in alert_types
        assert "daily_token_limit" in alert_types


class TestUsageTrackingCallback:
    """Test UsageTrackingCallback integration."""

    def test_init(self):
        """Test callback initialization."""
        tracker = UsageTracker()
        callback = UsageTrackingCallback(tracker)

        assert callback.usage_tracker is tracker
        assert callback.name == "UsageTrackingCallback"

    def test_on_provider_response(self):
        """Test handling provider response events."""
        tracker = UsageTracker()
        callback = UsageTrackingCallback(tracker)

        # Create mock event
        context = CallbackContext(call_id="test_call")
        provider = MagicMock()
        provider.name = "openai"

        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        usage = Usage(
            tokens=tokens,
            provider="openai",
            model="gpt-4",
            latency=1.5,
        )

        event = ProviderResponseEvent(
            context=context,
            provider=provider,
            usage=usage,
        )

        # Should have no usage initially
        assert len(tracker) == 0

        # Handle event
        callback.on_provider_response(event)

        # Should have tracked usage
        assert len(tracker) == 1

        history = tracker.get_history()
        record = history[0]
        assert record.provider == "openai"
        assert record.model == "gpt-4"
        assert record.call_id == "test_call"
        assert record.latency == 1.5

    def test_on_provider_response_no_usage(self):
        """Test handling provider response without usage."""
        tracker = UsageTracker()
        callback = UsageTrackingCallback(tracker)

        context = CallbackContext()
        provider = MagicMock()

        event = ProviderResponseEvent(
            context=context,
            provider=provider,
            usage=None,  # No usage
        )

        # Should not track anything
        callback.on_provider_response(event)
        assert len(tracker) == 0


class TestGlobalTracker:
    """Test global tracker functions."""

    def test_get_global_tracker(self):
        """Test getting global tracker."""
        tracker1 = get_global_tracker()
        tracker2 = get_global_tracker()

        # Should be the same instance
        assert tracker1 is tracker2

    def test_set_global_tracker(self):
        """Test setting global tracker."""
        custom_tracker = UsageTracker(session_id="custom_session")
        set_global_tracker(custom_tracker)

        global_tracker = get_global_tracker()
        assert global_tracker is custom_tracker
        assert global_tracker._session_id == "custom_session"


class TestTrackUsageContext:
    """Test track_usage context manager."""

    def test_basic_usage(self):
        """Test basic usage tracking context."""
        tracker = UsageTracker()

        with track_usage("openai", "gpt-4", tracker=tracker, call_id="test_call") as ctx:
            # Simulate provider response
            ctx["tokens"] = TokenUsage(input_tokens=1000, output_tokens=500)
            ctx["latency"] = 1.5

        # Should have tracked usage
        assert len(tracker) == 1

        history = tracker.get_history()
        record = history[0]
        assert record.provider == "openai"
        assert record.model == "gpt-4"
        assert record.call_id == "test_call"
        assert record.latency == 1.5

    def test_context_auto_provider_model(self):
        """Test context with provider/model set in context."""
        tracker = UsageTracker()

        with track_usage(tracker=tracker) as ctx:
            ctx["provider"] = "anthropic"
            ctx["model"] = "claude-3"
            ctx["tokens"] = TokenUsage(input_tokens=800, output_tokens=400)

        # Should have tracked usage
        assert len(tracker) == 1

        history = tracker.get_history()
        record = history[0]
        assert record.provider == "anthropic"
        assert record.model == "claude-3"

    def test_context_missing_info(self):
        """Test context with missing required information."""
        tracker = UsageTracker()

        with track_usage(tracker=tracker):
            # Don't set required fields
            pass

        # Should not have tracked anything
        assert len(tracker) == 0

    def test_context_with_exception(self):
        """Test context manager with exception."""
        tracker = UsageTracker()

        try:
            with track_usage("openai", "gpt-4", tracker=tracker) as ctx:
                ctx["tokens"] = TokenUsage(input_tokens=1000, output_tokens=500)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still track usage even with exception
        assert len(tracker) == 1

    def test_context_global_tracker(self):
        """Test context manager with global tracker."""
        # Set up global tracker
        global_tracker = UsageTracker(session_id="global_test")
        set_global_tracker(global_tracker)

        with track_usage("openai", "gpt-4") as ctx:  # No tracker specified
            ctx["tokens"] = TokenUsage(input_tokens=1000, output_tokens=500)

        # Should use global tracker
        assert len(global_tracker) == 1


class TestDefaultPricing:
    """Test default pricing data."""

    def test_default_pricing_exists(self):
        """Test that default pricing exists for major providers."""
        assert len(DEFAULT_PRICING) > 0

        # Check major providers
        assert ("openai", "gpt-4") in DEFAULT_PRICING
        assert ("openai", "gpt-4.1-mini") in DEFAULT_PRICING
        assert ("anthropic", "claude-3-opus") in DEFAULT_PRICING
        assert ("google", "gemini-pro") in DEFAULT_PRICING
        assert ("mock", "mock-model") in DEFAULT_PRICING

    def test_pricing_info_validity(self):
        """Test that all pricing info is valid."""
        for (provider, model), pricing in DEFAULT_PRICING.items():
            assert isinstance(pricing, PricingInfo)
            assert pricing.provider == provider
            assert pricing.model == model
            assert pricing.input_price_per_1k >= 0
            assert pricing.output_price_per_1k >= 0
            assert pricing.reasoning_price_per_1k >= 0

    def test_mock_pricing_free(self):
        """Test that mock provider pricing is free."""
        mock_pricing = DEFAULT_PRICING[("mock", "mock-model")]
        assert mock_pricing.input_price_per_1k == 0.0
        assert mock_pricing.output_price_per_1k == 0.0
        assert mock_pricing.reasoning_price_per_1k == 0.0
