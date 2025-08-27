"""JSONL optimization logger for LogiLLM.

This module provides both legacy wrapper-based logging and modern callback-based logging.
The OptimizationLogger class maintains backward compatibility by wrapping optimizers,
while the new callback-based approach is available via jsonl_callback.JSONLCallback.

For new code, prefer using the callback-based approach:
    from logillm.core.jsonl_callback import OptimizationJSONLCallback
    from logillm.core.callbacks import CallbackManager

    callback = OptimizationJSONLCallback("optimization.jsonl")
    CallbackManager().register(callback)

Each line in the JSONL file is a JSON object representing an optimization event:
- optimization_start: Initial configuration
- evaluation_start/end: Module state and scores at each evaluation
- optimization_end: Final results and configuration
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .modules import Module
from .optimizers import Optimizer


class OptimizationLogger:
    """Logger that wraps optimizers to capture optimization history."""

    def __init__(self, filepath: str = "optimization_log.jsonl"):
        """Initialize the optimization logger.

        Args:
            filepath: Path to JSONL output file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.start_time = None

    def _write_event(self, event: dict[str, Any]) -> None:
        """Write event to JSONL file."""
        event["timestamp"] = datetime.now().isoformat()
        event["elapsed"] = time.time() - self.start_time if self.start_time else 0

        with open(self.filepath, "a", encoding="utf-8") as f:
            json.dump(event, f, default=str)
            f.write("\n")

        self.events.append(event)

    def _extract_module_state(self, module: Module) -> dict[str, Any]:
        """Extract current state from a module."""
        state = {
            "type": module.__class__.__name__,
            "config": module.config.copy() if hasattr(module, "config") else {},
        }

        # Extract signature info
        if hasattr(module, "signature"):
            sig = module.signature

            # Format signature properly
            if hasattr(sig, "to_dict"):
                # Class-based signature - format it nicely
                sig_dict = sig.to_dict()
                input_fields = list(sig_dict.get("input_fields", {}).keys())
                output_fields = list(sig_dict.get("output_fields", {}).keys())

                # Format as ClassName(input1, input2 -> output1, output2)
                field_str = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
                sig_name = sig.__class__.__name__ if hasattr(sig, "__class__") else "Signature"
                state["signature"] = f"{sig_name}({field_str})"

                # Get instructions from the dict
                state["instructions"] = sig_dict.get("instructions")
            else:
                # String-based signature or BaseSignature
                state["signature"] = str(sig)
                if hasattr(sig, "instructions"):
                    state["instructions"] = sig.instructions

        # Extract demos
        if hasattr(module, "demo_manager") and module.demo_manager:
            demo_manager = module.demo_manager
            if hasattr(demo_manager, "demos"):
                demos = getattr(demo_manager, "demos", [])
                state["num_demos"] = len(demos)
                if demos:
                    # Include first demo as example
                    first_demo = demos[0]
                    state["demo_example"] = {
                        "inputs": first_demo.inputs,
                        "outputs": first_demo.outputs,
                    }

        # Extract hyperparameters from config
        if "config" in state:
            config = state["config"]
            state["hyperparameters"] = {
                "temperature": config.get("temperature"),
                "top_p": config.get("top_p"),
                "max_tokens": config.get("max_tokens"),
            }

        # Check provider config - use module's provider or get default
        provider = None
        if hasattr(module, "provider"):
            provider = module.provider
            if provider is None:
                # Module has no explicit provider, get the default
                from ..providers.registry import get_provider

                try:
                    provider = get_provider()
                except:
                    provider = None

        if provider:
            if hasattr(provider, "config") and isinstance(provider.config, dict):
                state["hyperparameters"].update(
                    {
                        k: v
                        for k, v in provider.config.items()
                        if k
                        in [
                            "temperature",
                            "top_p",
                            "max_tokens",
                            "frequency_penalty",
                            "presence_penalty",
                        ]
                    }
                )
            if hasattr(provider, "model"):
                state["model"] = provider.model

        return state

    async def log_optimization(
        self, optimizer: Optimizer, module: Module, dataset: list[dict[str, Any]], **kwargs
    ):
        """Wrap and log an optimization run.

        Args:
            optimizer: The optimizer to use
            module: Module to optimize
            dataset: Training dataset
            **kwargs: Additional optimizer arguments

        Returns:
            OptimizationResult from the optimizer
        """
        self.start_time = time.time()

        # Log start
        self._write_event(
            {
                "event_type": "optimization_start",
                "optimizer": optimizer.__class__.__name__,
                "dataset_size": len(dataset),
                "initial_module": self._extract_module_state(module),
                "optimizer_config": {
                    "strategy": getattr(optimizer, "strategy", None),
                    "metric": getattr(optimizer.metric, "name", lambda: "unknown")()
                    if hasattr(optimizer, "metric")
                    else None,
                },
            }
        )

        # Monkey-patch the evaluate method to capture intermediate states
        original_evaluate = optimizer.evaluate
        scores = []
        iteration = 0

        async def logged_evaluate(module, dataset_subset, *args, **eval_kwargs):
            nonlocal iteration
            iteration += 1

            # Enable debug mode on module to capture prompts
            original_debug = getattr(module, "_debug_mode", False)
            if hasattr(module, "_debug_mode"):
                module._debug_mode = True

            # Log module state before evaluation
            self._write_event(
                {
                    "event_type": "evaluation_start",
                    "iteration": iteration,
                    "module_state": self._extract_module_state(module),
                }
            )

            # Call original evaluate
            score, traces = await original_evaluate(module, dataset_subset, *args, **eval_kwargs)
            scores.append(score)

            # Extract prompts if module captured them during debug mode
            prompts_captured = []

            # Check if module has last captured prompts (would need to be stored)
            # For now, we note that prompts were captured if debug mode was on
            if getattr(module, "_debug_mode", False):
                # Module was in debug mode, prompts should be in prediction metadata
                # Note: actual prompt capture would require modifying the module to store them
                prompts_captured = [
                    {
                        "note": "Debug mode was enabled - prompts captured in predictions",
                        "debug_mode": True,
                    }
                ]

            # Log evaluation result with prompts
            self._write_event(
                {
                    "event_type": "evaluation_end",
                    "iteration": iteration,
                    "score": score,
                    "num_traces": len(traces) if traces else 0,
                    "prompts_sampled": prompts_captured if prompts_captured else None,
                }
            )

            # Restore original debug mode
            if hasattr(module, "_debug_mode"):
                module._debug_mode = original_debug

            return score, traces

        # Store original evaluate method
        original_evaluate = optimizer.evaluate
        # Temporarily replace evaluate method
        optimizer.evaluate = logged_evaluate

        try:
            # Run optimization
            result = await optimizer.optimize(module, dataset, **kwargs)

            # Log completion
            self._write_event(
                {
                    "event_type": "optimization_end",
                    "success": True,
                    "final_module": self._extract_module_state(result.optimized_module),
                    "best_score": result.best_score,
                    "improvement": result.improvement,
                    "iterations": result.iterations,
                    "optimization_time": result.optimization_time,
                    "score_progression": scores,
                    "metadata": result.metadata if hasattr(result, "metadata") else {},
                }
            )

            return result

        except Exception as e:
            # Log error
            self._write_event(
                {
                    "event_type": "optimization_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "scores_before_error": scores,
                }
            )
            raise

        finally:
            # Restore original evaluate method
            optimizer.evaluate = original_evaluate

    def get_summary(self) -> dict[str, Any]:
        """Get summary of logged optimization."""
        if not self.events:
            return {}

        summary = {
            "total_events": len(self.events),
            "duration": time.time() - self.start_time if self.start_time else 0,
            "iterations": 0,
            "scores": [],
        }

        for event in self.events:
            if event.get("event_type") == "evaluation_end":
                summary["iterations"] += 1
                if "score" in event:
                    summary["scores"].append(event["score"])

        if summary["scores"]:
            summary["best_score"] = max(summary["scores"])
            summary["improvement"] = (
                summary["best_score"] - summary["scores"][0] if len(summary["scores"]) > 1 else 0
            )

        return summary
