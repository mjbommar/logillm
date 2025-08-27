"""JSONL logging callback implementation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .callbacks import (
    AbstractCallback,
    EvaluationEndEvent,
    EvaluationStartEvent,
    ModuleEndEvent,
    ModuleStartEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    Priority,
    ProviderRequestEvent,
    ProviderResponseEvent,
)


class JSONLCallback(AbstractCallback):
    """Callback that logs events to JSONL (JSON Lines) format.

    This replaces the wrapper-based OptimizationLogger with a proper
    callback implementation that integrates cleanly with the callback system.
    """

    def __init__(
        self,
        filepath: str | Path,
        include_module_events: bool = True,
        include_optimization_events: bool = True,
        include_provider_events: bool = False,
        include_metadata: bool = True,
        append_mode: bool = True,
    ):
        """Initialize JSONL callback logger.

        Args:
            filepath: Path to JSONL file to write
            include_module_events: Whether to log module start/end events
            include_optimization_events: Whether to log optimization events
            include_provider_events: Whether to log provider request/response
            include_metadata: Whether to include full metadata in logs
            append_mode: Whether to append to existing file or overwrite
        """
        self.filepath = Path(filepath)
        self.include_module_events = include_module_events
        self.include_optimization_events = include_optimization_events
        self.include_provider_events = include_provider_events
        self.include_metadata = include_metadata
        self.append_mode = append_mode

        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Clear file if not in append mode
        if not append_mode and self.filepath.exists():
            self.filepath.unlink()

    @property
    def priority(self) -> Priority:
        """Low priority to run early and capture all events."""
        return Priority.LOW

    def _write_log(self, log_entry: dict[str, Any]) -> None:
        """Write a single log entry to the JSONL file."""
        # Add timestamp if not present
        if "timestamp" not in log_entry:
            log_entry["timestamp"] = datetime.now().isoformat()

        # Write to file
        with open(self.filepath, "a") as f:
            json.dump(log_entry, f, default=str, ensure_ascii=False)
            f.write("\n")

    async def on_module_start(self, event: ModuleStartEvent) -> None:
        """Log module start event."""
        if not self.include_module_events:
            return

        log_entry = {
            "event_type": "module_start",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "module_name": event.module.__class__.__name__,
            "inputs": event.inputs,
        }

        if self.include_metadata and hasattr(event.module, "metadata"):
            log_entry["module_metadata"] = event.module.metadata

        self._write_log(log_entry)

    async def on_module_end(self, event: ModuleEndEvent) -> None:
        """Log module end event."""
        if not self.include_module_events:
            return

        log_entry = {
            "event_type": "module_end",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "module_name": event.module.__class__.__name__,
            "outputs": event.outputs,
            "success": event.success,
            "duration": event.duration,
        }

        # Add usage information if available
        if event.prediction and event.prediction.usage:
            usage = event.prediction.usage
            log_entry["usage"] = {
                "tokens": {
                    "input": usage.tokens.input_tokens if usage.tokens else 0,
                    "output": usage.tokens.output_tokens if usage.tokens else 0,
                    "total": usage.tokens.total_tokens if usage.tokens else 0,
                },
                "cost": usage.cost,
                "latency": usage.latency,
            }

        self._write_log(log_entry)

    async def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Log optimization start event."""
        if not self.include_optimization_events:
            return

        log_entry = {
            "event_type": "optimization_start",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "optimizer_name": event.optimizer.__class__.__name__,
            "module_name": event.module.__class__.__name__,
            "dataset_size": len(event.dataset) if event.dataset else 0,
        }

        if self.include_metadata and event.config:
            log_entry["config"] = event.config

        self._write_log(log_entry)

    async def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Log optimization end event."""
        if not self.include_optimization_events:
            return

        log_entry = {
            "event_type": "optimization_end",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "optimizer_name": event.optimizer.__class__.__name__,
            "success": event.success,
            "duration": event.duration,
        }

        if event.result:
            log_entry["result"] = {
                "best_score": event.result.best_score,
                "improvement": event.result.improvement,
                "iterations": event.result.iterations,
                "optimization_time": event.result.optimization_time,
            }

            if self.include_metadata and event.result.metadata:
                log_entry["result"]["metadata"] = event.result.metadata

        self._write_log(log_entry)

    async def on_evaluation_start(self, event: EvaluationStartEvent) -> None:
        """Log evaluation start event."""
        if not self.include_optimization_events:
            return

        log_entry = {
            "event_type": "evaluation_start",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "optimizer_name": event.optimizer.__class__.__name__,
            "module_name": event.module.__class__.__name__,
            "dataset_size": len(event.dataset) if event.dataset else 0,
        }

        self._write_log(log_entry)

    async def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        """Log evaluation end event."""
        if not self.include_optimization_events:
            return

        log_entry = {
            "event_type": "evaluation_end",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "optimizer_name": event.optimizer.__class__.__name__,
            "module_name": event.module.__class__.__name__,
            "score": event.score,
            "duration": event.duration,
        }

        self._write_log(log_entry)

    async def on_provider_request(self, event: ProviderRequestEvent) -> None:
        """Log provider request event."""
        if not self.include_provider_events:
            return

        log_entry = {
            "event_type": "provider_request",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "provider_name": getattr(event.provider, "name", "unknown"),
            "model": getattr(event.provider, "model", "unknown"),
            "message_count": len(event.messages),
        }

        if self.include_metadata:
            log_entry["messages"] = event.messages
            log_entry["parameters"] = event.parameters

        self._write_log(log_entry)

    async def on_provider_response(self, event: ProviderResponseEvent) -> None:
        """Log provider response event."""
        if not self.include_provider_events:
            return

        log_entry = {
            "event_type": "provider_response",
            "timestamp": event.timestamp.isoformat(),
            "call_id": event.context.call_id,
            "provider_name": getattr(event.provider, "name", "unknown"),
            "model": getattr(event.provider, "model", "unknown"),
            "duration": event.duration,
        }

        if event.usage:
            log_entry["usage"] = {
                "input_tokens": event.usage.tokens.input_tokens if event.usage.tokens else 0,
                "output_tokens": event.usage.tokens.output_tokens if event.usage.tokens else 0,
                "total_tokens": event.usage.tokens.total_tokens if event.usage.tokens else 0,
                "cost": event.usage.cost,
            }

        if self.include_metadata and event.response:
            # Only include text, not full response object
            if hasattr(event.response, "text"):
                log_entry["response_text"] = event.response.text[:500]  # Truncate long responses

        self._write_log(log_entry)


class OptimizationJSONLCallback(JSONLCallback):
    """Specialized JSONL callback for optimization workflows.

    This is a convenience class that focuses on optimization events only,
    replacing the old OptimizationLogger wrapper.
    """

    def __init__(self, filepath: str | Path, **kwargs):
        """Initialize optimization-focused JSONL logger.

        Args:
            filepath: Path to JSONL file
            **kwargs: Additional arguments for JSONLCallback
        """
        super().__init__(
            filepath=filepath,
            include_module_events=False,  # Focus on optimization
            include_optimization_events=True,
            include_provider_events=False,
            **kwargs,
        )


def register_jsonl_logger(
    filepath: str | Path, callback_type: type[JSONLCallback] = JSONLCallback, **kwargs
) -> str:
    """Convenience function to register a JSONL logger callback.

    Args:
        filepath: Path to JSONL file
        callback_type: Type of JSONL callback to use
        **kwargs: Additional arguments for the callback

    Returns:
        Callback ID that can be used to unregister
    """
    from .callbacks import callback_manager

    callback = callback_type(filepath, **kwargs)
    return callback_manager.register(callback)
