"""Comprehensive usage tracking system for LogiLLM.

This module provides a zero-dependency usage tracking system that:
- Tracks token usage per provider and model
- Calculates costs based on provider pricing
- Supports both input and output tokens
- Aggregates usage across sessions
- Exports usage reports
- Thread-safe for production use
- Memory-efficient with circular buffers
"""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .callbacks import AbstractCallback, ProviderResponseEvent
from .types import TokenUsage, Usage


class ExportFormat(Enum):
    """Supported export formats for usage reports."""

    JSON = "json"
    CSV = "csv"


class TimeWindow(Enum):
    """Time windows for usage aggregation."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass(frozen=True)
class PricingInfo:
    """Pricing information for a provider/model combination."""

    provider: str
    model: str
    input_price_per_1k: float  # Price per 1K input tokens
    output_price_per_1k: float  # Price per 1K output tokens
    reasoning_price_per_1k: float = 0.0  # For models with reasoning tokens (like o1)

    def calculate_cost(self, tokens: TokenUsage) -> float:
        """Calculate cost for given token usage.

        Args:
            tokens: Token usage to calculate cost for

        Returns:
            Total cost in dollars
        """
        input_cost = (tokens.input_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (tokens.output_tokens / 1000.0) * self.output_price_per_1k
        reasoning_cost = (tokens.reasoning_tokens / 1000.0) * self.reasoning_price_per_1k
        return input_cost + output_cost + reasoning_cost


@dataclass
class UsageRecord:
    """Single usage event record."""

    provider: str
    model: str
    tokens: TokenUsage
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str | None = None
    call_id: str | None = None
    latency: float | None = None  # Response time in seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        """Create from dictionary."""
        # Handle timestamp
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        # Handle tokens field
        if isinstance(data["tokens"], dict):
            data["tokens"] = TokenUsage(**data["tokens"])

        return cls(**data)


@dataclass
class UsageStats:
    """Aggregated usage statistics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    providers: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    models: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hourly_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all types."""
        return self.total_input_tokens + self.total_output_tokens + self.total_reasoning_tokens

    @property
    def duration_hours(self) -> float:
        """Duration of tracked period in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600.0

    @property
    def cost_per_request(self) -> float:
        """Average cost per request."""
        return self.total_cost / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def tokens_per_request(self) -> float:
        """Average tokens per request."""
        return self.total_tokens / self.total_requests if self.total_requests > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat()
        return result


# Default pricing information for major providers
DEFAULT_PRICING: dict[tuple[str, str], PricingInfo] = {
    # OpenAI
    ("openai", "gpt-4"): PricingInfo("openai", "gpt-4", 0.03, 0.06),
    ("openai", "gpt-4-turbo"): PricingInfo("openai", "gpt-4-turbo", 0.01, 0.03),
    ("openai", "gpt-4o"): PricingInfo("openai", "gpt-4o", 0.005, 0.015),
    ("openai", "gpt-4.1-mini"): PricingInfo("openai", "gpt-4.1-mini", 0.00015, 0.0006),
    ("openai", "gpt-4.1"): PricingInfo("openai", "gpt-4.1", 0.0015, 0.002),
    ("openai", "o1"): PricingInfo("openai", "o1", 0.015, 0.06, 0.06),
    ("openai", "o1-mini"): PricingInfo("openai", "o1-mini", 0.003, 0.012, 0.012),
    # Anthropic
    ("anthropic", "claude-3-5-sonnet"): PricingInfo("anthropic", "claude-3-5-sonnet", 0.003, 0.015),
    ("anthropic", "claude-3-opus"): PricingInfo("anthropic", "claude-3-opus", 0.015, 0.075),
    ("anthropic", "claude-3-sonnet"): PricingInfo("anthropic", "claude-3-sonnet", 0.003, 0.015),
    ("anthropic", "claude-3-haiku"): PricingInfo("anthropic", "claude-3-haiku", 0.00025, 0.00125),
    # Google
    ("google", "gemini-pro"): PricingInfo("google", "gemini-pro", 0.00025, 0.0005),
    ("google", "gemini-1.5-pro"): PricingInfo("google", "gemini-1.5-pro", 0.0035, 0.0105),
    ("google", "gemini-1.5-flash"): PricingInfo("google", "gemini-1.5-flash", 0.000075, 0.0003),
    # Mock provider (free for testing)
    ("mock", "mock-model"): PricingInfo("mock", "mock-model", 0.0, 0.0),
}


class UsageTracker:
    """Thread-safe usage tracker with cost calculation and reporting.

    Features:
    - Thread-safe tracking for concurrent requests
    - Automatic cost calculation with provider-specific pricing
    - Memory-efficient circular buffer for history (prevents unlimited growth)
    - Session-based tracking
    - Time-based aggregation (hourly, daily, monthly)
    - Export to JSON/CSV formats
    - Usage limits and alerts
    - Integration with callback system
    """

    def __init__(
        self,
        *,
        max_history: int = 10000,
        session_id: str | None = None,
        custom_pricing: dict[tuple[str, str], PricingInfo] | None = None,
        daily_cost_limit: float | None = None,
        daily_token_limit: int | None = None,
    ):
        """Initialize usage tracker.

        Args:
            max_history: Maximum number of records to keep in memory (uses circular buffer)
            session_id: Optional session identifier
            custom_pricing: Custom pricing information (overrides defaults)
            daily_cost_limit: Optional daily cost limit for alerts
            daily_token_limit: Optional daily token limit for alerts
        """
        self._history: deque[UsageRecord] = deque(maxlen=max_history)
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._session_id = session_id or f"session_{int(time.time())}"
        self._pricing = DEFAULT_PRICING.copy()

        # Update with custom pricing
        if custom_pricing:
            self._pricing.update(custom_pricing)

        # Limits and alerts
        self._daily_cost_limit = daily_cost_limit
        self._daily_token_limit = daily_token_limit
        self._alert_callbacks: list[Callable] = []

        # Statistics cache (invalidated on new records)
        self._stats_cache: dict[str, UsageStats] = {}
        self._cache_dirty = True

    def add_pricing(self, provider: str, model: str, pricing: PricingInfo) -> None:
        """Add or update pricing information.

        Args:
            provider: Provider name
            model: Model name
            pricing: Pricing information
        """
        with self._lock:
            self._pricing[(provider, model)] = pricing

    def get_pricing(self, provider: str, model: str) -> PricingInfo | None:
        """Get pricing information for provider/model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Pricing information or None if not found
        """
        return self._pricing.get((provider, model))

    def track_usage(
        self,
        provider: str,
        model: str,
        tokens: TokenUsage,
        *,
        call_id: str | None = None,
        latency: float | None = None,
    ) -> UsageRecord:
        """Track usage for a single request.

        Args:
            provider: Provider name
            model: Model name
            tokens: Token usage
            call_id: Optional call identifier
            latency: Optional response time in seconds

        Returns:
            Created usage record
        """
        # Calculate cost
        pricing = self._pricing.get((provider, model))
        cost = pricing.calculate_cost(tokens) if pricing else 0.0

        # Create record
        record = UsageRecord(
            provider=provider,
            model=model,
            tokens=tokens,
            cost=cost,
            session_id=self._session_id,
            call_id=call_id,
            latency=latency,
        )

        with self._lock:
            self._history.append(record)
            self._cache_dirty = True

            # Check limits and trigger alerts
            self._check_limits()

        return record

    def __getstate__(self) -> dict:
        """Get state for pickling (excludes the RLock)."""
        state = self.__dict__.copy()
        # Remove the unpicklable RLock
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling (recreates the RLock)."""
        self.__dict__.update(state)
        # Recreate the RLock
        self._lock = threading.RLock()

    def track_usage_from_response(
        self, usage: Usage, call_id: str | None = None
    ) -> UsageRecord | None:
        """Track usage from a Usage object (typically from provider response).

        Args:
            usage: Usage object from provider response
            call_id: Optional call identifier

        Returns:
            Created usage record or None if provider/model not specified
        """
        if not usage.provider or not usage.model:
            return None

        return self.track_usage(
            provider=usage.provider,
            model=usage.model,
            tokens=usage.tokens,
            call_id=call_id,
            latency=usage.latency,
        )

    def get_stats(self, time_window: TimeWindow | None = None) -> UsageStats:
        """Get usage statistics.

        Args:
            time_window: Optional time window to filter by

        Returns:
            Aggregated usage statistics
        """
        cache_key = f"stats_{time_window.value if time_window else 'all'}"

        with self._lock:
            # Return cached stats if available and cache is clean
            if not self._cache_dirty and cache_key in self._stats_cache:
                return self._stats_cache[cache_key]

            # Calculate time cutoff
            now = datetime.now()
            cutoff = None
            if time_window == TimeWindow.HOUR:
                cutoff = now - timedelta(hours=1)
            elif time_window == TimeWindow.DAY:
                cutoff = now - timedelta(days=1)
            elif time_window == TimeWindow.WEEK:
                cutoff = now - timedelta(weeks=1)
            elif time_window == TimeWindow.MONTH:
                cutoff = now - timedelta(days=30)

            # Filter records
            records = [r for r in self._history if cutoff is None or r.timestamp >= cutoff]

            if not records:
                stats = UsageStats()
                self._stats_cache[cache_key] = stats
                return stats

            # Calculate statistics
            stats = UsageStats()
            stats.total_requests = len(records)
            stats.start_time = records[0].timestamp
            stats.end_time = records[-1].timestamp

            total_latency = 0.0
            latency_count = 0

            for record in records:
                stats.total_input_tokens += record.tokens.input_tokens
                stats.total_output_tokens += record.tokens.output_tokens
                stats.total_reasoning_tokens += record.tokens.reasoning_tokens
                stats.total_cost += record.cost

                stats.providers[record.provider] += 1
                stats.models[f"{record.provider}:{record.model}"] += 1

                # Hourly usage
                hour_key = record.timestamp.strftime("%Y-%m-%d %H:00")
                stats.hourly_usage[hour_key] += 1

                # Average latency
                if record.latency is not None:
                    total_latency += record.latency
                    latency_count += 1

            if latency_count > 0:
                stats.avg_latency = total_latency / latency_count

            # Cache results
            self._stats_cache[cache_key] = stats
            if time_window is None:  # Only mark clean if we calculated all stats
                self._cache_dirty = False

            return stats

    def get_history(self, limit: int | None = None) -> list[UsageRecord]:
        """Get usage history records.

        Args:
            limit: Maximum number of records to return (most recent first)

        Returns:
            List of usage records in reverse chronological order
        """
        with self._lock:
            records = list(reversed(self._history))
            return records[:limit] if limit else records

    def clear_history(self) -> None:
        """Clear all usage history."""
        with self._lock:
            self._history.clear()
            self._stats_cache.clear()
            self._cache_dirty = True

    def export_usage(
        self,
        filepath: Path | str,
        format: ExportFormat = ExportFormat.JSON,
        time_window: TimeWindow | None = None,
    ) -> None:
        """Export usage data to file.

        Args:
            filepath: Output file path
            format: Export format (JSON or CSV)
            time_window: Optional time window to filter by
        """
        filepath = Path(filepath)

        # Get filtered records
        now = datetime.now()
        cutoff = None
        if time_window == TimeWindow.HOUR:
            cutoff = now - timedelta(hours=1)
        elif time_window == TimeWindow.DAY:
            cutoff = now - timedelta(days=1)
        elif time_window == TimeWindow.WEEK:
            cutoff = now - timedelta(weeks=1)
        elif time_window == TimeWindow.MONTH:
            cutoff = now - timedelta(days=30)

        with self._lock:
            records = [r for r in self._history if cutoff is None or r.timestamp >= cutoff]

        # Export based on format
        if format == ExportFormat.JSON:
            self._export_json(filepath, records)
        elif format == ExportFormat.CSV:
            self._export_csv(filepath, records)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, filepath: Path, records: list[UsageRecord]) -> None:
        """Export records to JSON file."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_records": len(records),
            "session_id": self._session_id,
            "records": [record.to_dict() for record in records],
            "stats": self.get_stats().to_dict(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_csv(self, filepath: Path, records: list[UsageRecord]) -> None:
        """Export records to CSV file."""
        if not records:
            # Create empty CSV with headers
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
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
                )
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
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
            )

            # Write records
            for record in records:
                writer.writerow(
                    [
                        record.timestamp.isoformat(),
                        record.provider,
                        record.model,
                        record.tokens.input_tokens,
                        record.tokens.output_tokens,
                        record.tokens.reasoning_tokens,
                        record.tokens.total_tokens,
                        record.cost,
                        record.latency,
                        record.session_id,
                        record.call_id,
                    ]
                )

    def import_usage(self, filepath: Path | str) -> int:
        """Import usage data from JSON file.

        Args:
            filepath: Input file path

        Returns:
            Number of records imported
        """
        filepath = Path(filepath)

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        records = [UsageRecord.from_dict(record_data) for record_data in data["records"]]

        with self._lock:
            for record in records:
                self._history.append(record)
            self._cache_dirty = True

        return len(records)

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for usage limit alerts.

        Args:
            callback: Function to call when limits are exceeded.
                     Signature: callback(alert_type: str, message: str, current_value: float, limit: float)
        """
        self._alert_callbacks.append(callback)

    def _check_limits(self) -> None:
        """Check usage limits and trigger alerts if exceeded."""
        if not self._alert_callbacks:
            return

        # Check daily limits
        daily_stats = self.get_stats(TimeWindow.DAY)

        if self._daily_cost_limit and daily_stats.total_cost > self._daily_cost_limit:
            for callback in self._alert_callbacks:
                callback(
                    "daily_cost_limit",
                    f"Daily cost limit exceeded: ${daily_stats.total_cost:.4f} > ${self._daily_cost_limit}",
                    daily_stats.total_cost,
                    self._daily_cost_limit,
                )

        if self._daily_token_limit and daily_stats.total_tokens > self._daily_token_limit:
            for callback in self._alert_callbacks:
                callback(
                    "daily_token_limit",
                    f"Daily token limit exceeded: {daily_stats.total_tokens} > {self._daily_token_limit}",
                    daily_stats.total_tokens,
                    self._daily_token_limit,
                )

    @contextmanager
    def session(self, session_id: str) -> Iterator[UsageTracker]:
        """Context manager for temporary session tracking.

        Args:
            session_id: Session identifier

        Yields:
            Self with session_id set
        """
        old_session = self._session_id
        self._session_id = session_id
        try:
            yield self
        finally:
            self._session_id = old_session

    def __len__(self) -> int:
        """Return number of tracked records."""
        return len(self._history)

    def __bool__(self) -> bool:
        """Return True if any usage has been tracked."""
        return len(self._history) > 0


class UsageTrackingCallback(AbstractCallback):
    """Callback that automatically tracks usage from provider responses.

    This callback integrates with the LogiLLM callback system to automatically
    track usage whenever providers return response data.
    """

    def __init__(self, usage_tracker: UsageTracker):
        """Initialize usage tracking callback.

        Args:
            usage_tracker: UsageTracker instance to use
        """
        self.usage_tracker = usage_tracker

    @property
    def name(self) -> str:
        """Callback name."""
        return "UsageTrackingCallback"

    def on_provider_response(self, event: ProviderResponseEvent) -> None:
        """Track usage from provider response.

        Args:
            event: Provider response event
        """
        if event.usage:
            self.usage_tracker.track_usage_from_response(
                event.usage,
                call_id=event.context.call_id,
            )


# Global usage tracker instance
_global_tracker: UsageTracker | None = None
_global_lock = threading.Lock()


def get_global_tracker() -> UsageTracker:
    """Get or create the global usage tracker instance.

    Returns:
        Global UsageTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        with _global_lock:
            if _global_tracker is None:
                _global_tracker = UsageTracker()

    return _global_tracker  # type: ignore[return-value]


def set_global_tracker(tracker: UsageTracker) -> None:
    """Set the global usage tracker instance.

    Args:
        tracker: UsageTracker instance to use globally
    """
    global _global_tracker
    with _global_lock:
        _global_tracker = tracker


@contextmanager
def track_usage(
    provider: str | None = None,
    model: str | None = None,
    *,
    tracker: UsageTracker | None = None,
    call_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Context manager for automatic usage tracking.

    This context manager can be used to automatically track usage
    for code blocks that make LLM requests.

    Args:
        provider: Provider name (will be set automatically if not provided)
        model: Model name (will be set automatically if not provided)
        tracker: UsageTracker to use (defaults to global tracker)
        call_id: Optional call identifier

    Yields:
        Dictionary that can be used to store usage information

    Example:
        with track_usage("openai", "gpt-4") as usage_context:
            result = await provider.complete(messages)
            usage_context["tokens"] = result.usage.tokens
            usage_context["latency"] = time.time() - start_time
    """
    if tracker is None:
        tracker = get_global_tracker()

    usage_context: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "tokens": None,
        "latency": None,
    }

    start_time = time.time()

    try:
        yield usage_context
    finally:
        # Track usage if we have the necessary information
        if (
            usage_context.get("tokens")
            and usage_context.get("provider")
            and usage_context.get("model")
        ):
            # Calculate latency if not provided
            if usage_context.get("latency") is None:
                usage_context["latency"] = time.time() - start_time

            tracker.track_usage(
                provider=usage_context["provider"],
                model=usage_context["model"],
                tokens=usage_context["tokens"],
                call_id=call_id,
                latency=usage_context["latency"],
            )


__all__ = [
    # Core classes
    "UsageTracker",
    "UsageRecord",
    "UsageStats",
    "PricingInfo",
    # Enums
    "ExportFormat",
    "TimeWindow",
    # Callback integration
    "UsageTrackingCallback",
    # Convenience functions
    "get_global_tracker",
    "set_global_tracker",
    "track_usage",
    # Default data
    "DEFAULT_PRICING",
]
