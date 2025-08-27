"""Tests for JSONL callback functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from logillm.core.callbacks import CallbackManager
from logillm.core.jsonl_callback import (
    JSONLCallback,
    OptimizationJSONLCallback,
    register_jsonl_logger,
)
from logillm.core.predict import Predict
from logillm.providers import register_provider
from logillm.providers.mock import MockProvider


class TestJSONLCallback:
    """Test JSONL callback functionality."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Clear manager before each test
        manager = CallbackManager()
        manager.clear()
        manager.enable()

        # Register mock provider
        provider = MockProvider(response_text="Test answer")
        register_provider(provider, set_default=True)

        yield

        # Clear after test
        manager.clear()

    @pytest.mark.asyncio
    async def test_jsonl_module_logging(self):
        """Test that module events are logged to JSONL."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Register JSONL callback
            callback = JSONLCallback(
                tmp_path,
                include_module_events=True,
                include_provider_events=False
            )
            manager = CallbackManager()
            manager.register(callback)

            # Execute module
            module = Predict("question -> answer")
            await module(question="What is AI?")

            # Read and verify JSONL file
            with open(tmp_path) as f:
                lines = f.readlines()

            assert len(lines) >= 2  # At least start and end events

            # Parse events
            events = [json.loads(line) for line in lines]

            # Check module start event
            start_events = [e for e in events if e["event_type"] == "module_start"]
            assert len(start_events) >= 1
            assert start_events[0]["module_name"] == "Predict"
            assert "inputs" in start_events[0]

            # Check module end event
            end_events = [e for e in events if e["event_type"] == "module_end"]
            assert len(end_events) >= 1
            assert end_events[0]["module_name"] == "Predict"
            assert end_events[0]["success"]
            assert "outputs" in end_events[0]
            assert "duration" in end_events[0]

        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_jsonl_provider_logging(self):
        """Test that provider events are logged when enabled."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Register JSONL callback with provider events
            callback = JSONLCallback(
                tmp_path,
                include_module_events=False,
                include_provider_events=True
            )
            manager = CallbackManager()
            manager.register(callback)

            # Execute module (which calls provider)
            module = Predict("question -> answer")
            await module(question="Test")

            # Read and verify JSONL file
            with open(tmp_path) as f:
                lines = f.readlines()

            assert len(lines) >= 2  # Request and response

            events = [json.loads(line) for line in lines]

            # Check provider request
            request_events = [e for e in events if e["event_type"] == "provider_request"]
            assert len(request_events) >= 1
            assert request_events[0]["provider_name"] == "mock"

            # Check provider response
            response_events = [e for e in events if e["event_type"] == "provider_response"]
            assert len(response_events) >= 1
            assert "duration" in response_events[0]

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_jsonl_append_mode(self):
        """Test that append mode works correctly."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # First execution
            callback = JSONLCallback(tmp_path, append_mode=True)
            manager = CallbackManager()
            callback_id = manager.register(callback)

            module = Predict("question -> answer")
            await module(question="First")

            # Count lines
            with open(tmp_path) as f:
                first_count = len(f.readlines())

            # Unregister first callback
            manager.unregister(callback_id)

            # Second execution with new callback instance
            callback2 = JSONLCallback(tmp_path, append_mode=True)
            manager.register(callback2)

            await module(question="Second")

            # Count lines again
            with open(tmp_path) as f:
                second_count = len(f.readlines())

            # Should have more lines (appended)
            assert second_count > first_count

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_jsonl_overwrite_mode(self):
        """Test that overwrite mode clears existing file."""
        # Create temporary file with initial content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write('{"initial": "content"}\n')

        try:
            # Create callback with append_mode=False (overwrite)
            callback = JSONLCallback(tmp_path, append_mode=False)
            manager = CallbackManager()
            manager.register(callback)

            module = Predict("question -> answer")
            await module(question="Test")

            # Read file
            with open(tmp_path) as f:
                lines = f.readlines()

            # Should not have initial content
            events = [json.loads(line) for line in lines]
            assert not any("initial" in e for e in events)

            # Should have new events
            assert all("event_type" in e for e in events)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_optimization_jsonl_callback(self):
        """Test the specialized OptimizationJSONLCallback."""
        from logillm.core.types import OptimizationResult

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Use OptimizationJSONLCallback
            callback = OptimizationJSONLCallback(tmp_path)
            manager = CallbackManager()
            manager.register(callback)

            # Emit mock optimization events
            from logillm.core.callbacks import (
                CallbackContext,
                OptimizationEndEvent,
                OptimizationStartEvent,
            )

            context = CallbackContext()

            # Mock optimizer and module
            class MockOptimizer:
                pass

            class MockModule:
                pass

            optimizer = MockOptimizer()
            module = MockModule()

            # Emit start event
            await callback.on_optimization_start(OptimizationStartEvent(
                context=context,
                optimizer=optimizer,
                module=module,
                dataset=[{"input": "test"}]
            ))

            # Emit end event
            result = OptimizationResult(
                optimized_module=module,
                improvement=0.1,
                iterations=10,
                best_score=0.9,
                optimization_time=1.5
            )

            await callback.on_optimization_end(OptimizationEndEvent(
                context=context,
                optimizer=optimizer,
                result=result,
                success=True,
                duration=1.5
            ))

            # Read and verify
            with open(tmp_path) as f:
                lines = f.readlines()

            events = [json.loads(line) for line in lines]

            # Should have optimization events only
            assert len(events) == 2
            assert events[0]["event_type"] == "optimization_start"
            assert events[1]["event_type"] == "optimization_end"
            assert events[1]["result"]["best_score"] == 0.9

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_register_jsonl_logger(self):
        """Test the convenience registration function."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Use convenience function
            callback_id = register_jsonl_logger(
                tmp_path,
                include_module_events=True
            )

            # Execute module
            module = Predict("question -> answer")
            await module(question="Test")

            # Verify file was written
            with open(tmp_path) as f:
                lines = f.readlines()

            assert len(lines) > 0

            # Unregister
            manager = CallbackManager()
            assert manager.unregister(callback_id)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
