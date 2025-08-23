"""Enhanced ReAct module based on DSPy's implementation with LogiLLM patterns."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import ModuleError
from .modules import Module
from .predict import Predict
from .signatures import BaseSignature, Signature
from .tools import Tool, ToolRegistry
from .types import Configuration, Metadata, Prediction, Usage

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """A single step in ReAct reasoning."""

    thought: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    step_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "observation": self.observation,
            "step_number": self.step_number,
        }


@dataclass
class ReActTrajectory:
    """Complete trajectory of ReAct execution."""

    steps: list[ReActStep] = field(default_factory=list)
    final_prediction: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    total_usage: Usage = field(default_factory=Usage)

    def add_step(self, step: ReActStep) -> None:
        """Add a step to the trajectory."""
        step.step_number = len(self.steps)
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_prediction": self.final_prediction,
            "success": self.success,
            "total_usage": self.total_usage.to_dict()
            if hasattr(self.total_usage, "to_dict")
            else {},
        }


class ReAct(Module):
    """ReAct module for reasoning and acting with tools.

    Based on DSPy's ReAct implementation but adapted for LogiLLM's
    architecture and zero-dependency approach.
    """

    def __init__(
        self,
        signature: Signature | str,
        tools: list[Tool] | ToolRegistry | None = None,
        max_iters: int = 10,
        *,
        config: Configuration | None = None,
        metadata: Metadata | None = None,
    ):
        """Initialize ReAct module.

        Args:
            signature: Input/output specification for the task
            tools: List of tools or tool registry to use
            max_iters: Maximum number of reasoning iterations
            config: Additional configuration
            metadata: Module metadata
        """
        super().__init__(signature=signature, config=config, metadata=metadata)

        self.original_signature = self._resolve_signature(signature)
        self.max_iters = max_iters

        # Set up tools
        if isinstance(tools, ToolRegistry):
            self.tools = {tool.name: tool for tool in tools.tools.values()}
        elif tools:
            self.tools = {tool.name: tool for tool in tools}
        else:
            self.tools = {}

        # Add the finish tool
        from .tools.base import Tool

        def finish_func() -> str:
            """Signal task completion."""
            return "Task completed."

        finish_tool = Tool(finish_func, name="finish")
        finish_tool.desc = f"Signal task completion when all information needed for {self._get_output_fields()} is available."
        self.tools["finish"] = finish_tool

        # Create ReAct-specific signature for reasoning steps
        self._create_react_signature()

        # Create modules for reasoning and extraction
        self.react_predict = Predict(self.react_signature, config=config)
        self.extract_predict = Predict(self.fallback_signature, config=config)

    def _get_output_fields(self) -> str:
        """Get output field names as a string."""
        if not self.original_signature or not hasattr(self.original_signature, "output_fields"):
            return "the outputs"

        fields = list(self.original_signature.output_fields.keys())
        if len(fields) == 1:
            return f"`{fields[0]}`"
        elif len(fields) == 2:
            return f"`{fields[0]}` and `{fields[1]}`"
        else:
            return ", ".join(f"`{f}`" for f in fields[:-1]) + f", and `{fields[-1]}`"

    def _create_react_signature(self) -> None:
        """Create the internal ReAct signature for reasoning."""
        # Get input and output field descriptions
        inputs = (
            ", ".join([f"`{k}`" for k in self.original_signature.input_fields.keys()])
            if self.original_signature
            else "the input"
        )
        outputs = self._get_output_fields()

        # Build instruction text
        instructions = []
        if self.original_signature and self.original_signature.instructions:
            instructions.append(f"{self.original_signature.instructions}\n")

        instructions.extend(
            [
                f"You are an agent. You receive {inputs} as input and need to produce {outputs}.",
                f"Your goal is to use tools to collect necessary information for producing {outputs}.\n",
                "To do this, you will interleave thoughts, tool calls, and observations.",
                "After each tool call, you receive an observation that gets added to your trajectory.\n",
                "When thinking, reason about the current situation and plan future steps.",
                "When selecting a tool and arguments, choose from:\n",
            ]
        )

        # Add tool descriptions
        for idx, (_, tool) in enumerate(self.tools.items(), 1):
            instructions.append(f"({idx}) {tool}")

        instructions.append("\nWhen providing tool arguments, use JSON format.")

        instruction_text = "\n".join(instructions)

        # Create signature for reasoning steps
        from .signatures.spec import FieldSpec
        from .types import FieldType

        react_input_fields = {
            "trajectory": FieldSpec(
                name="trajectory",
                field_type=FieldType.INPUT,
                python_type=str,
                description="Current trajectory of thoughts and observations",
                required=True,
            )
        }

        react_output_fields = {
            "next_thought": FieldSpec(
                name="next_thought",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description="What should I do next? Let me think step by step.",
                required=True,
            ),
            "next_tool_name": FieldSpec(
                name="next_tool_name",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description=f"Tool name from: {', '.join(self.tools.keys())}",
                required=True,
            ),
            "next_tool_args": FieldSpec(
                name="next_tool_args",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description="Tool arguments in JSON format",
                required=True,
            ),
        }

        self.react_signature = BaseSignature(
            input_fields=react_input_fields,
            output_fields=react_output_fields,
            instructions=instruction_text,
        )

        # Create fallback signature for final extraction
        fallback_inputs = (
            {**self.original_signature.input_fields} if self.original_signature else {}
        )
        fallback_outputs = (
            {**self.original_signature.output_fields}
            if self.original_signature
            else {"answer": "string"}
        )

        self.fallback_signature = BaseSignature(
            input_fields=fallback_inputs,
            output_fields=fallback_outputs,
            instructions=self.original_signature.instructions
            if self.original_signature
            else "Extract the final answer.",
        )
        self.fallback_signature.input_fields["trajectory"] = "string"  # Add trajectory input

    def _format_trajectory(self, trajectory: ReActTrajectory) -> str:
        """Format trajectory for prompt."""
        if not trajectory.steps:
            return "No previous steps."

        lines = []
        for step in trajectory.steps:
            lines.append(f"\nStep {step.step_number + 1}:")
            lines.append(f"Thought: {step.thought}")
            if step.tool_name:
                lines.append(f"Tool: {step.tool_name}")
                if step.tool_args:
                    lines.append(f"Args: {json.dumps(step.tool_args)}")
            if step.observation:
                lines.append(f"Observation: {step.observation}")

        return "\n".join(lines)

    async def _call_with_trajectory_truncation(
        self, predict_module: Predict, trajectory: ReActTrajectory, **input_args
    ) -> Prediction:
        """Call predict module with automatic trajectory truncation on context overflow."""
        for attempt in range(3):  # Max 3 truncation attempts
            try:
                trajectory_text = self._format_trajectory(trajectory)
                return await predict_module(**input_args, trajectory=trajectory_text)
            except Exception as e:
                # Check if this is a context window error
                error_msg = str(e).lower()
                if "context" in error_msg or "token" in error_msg or "length" in error_msg:
                    if len(trajectory.steps) < 2:
                        # Can't truncate further
                        raise ModuleError(
                            "Context window exceeded with minimal trajectory",
                            module_name="ReAct",
                            execution_stage="prediction",
                            context={"trajectory_steps": len(trajectory.steps)},
                        ) from e

                    # Truncate trajectory by removing oldest steps
                    logger.warning(
                        f"Truncating trajectory due to context overflow (attempt {attempt + 1})"
                    )
                    trajectory = self._truncate_trajectory(trajectory)
                else:
                    # Not a context error, re-raise
                    raise

        raise ModuleError(
            "Failed to execute after maximum trajectory truncation attempts",
            module_name="ReAct",
            execution_stage="prediction",
        )

    def _truncate_trajectory(self, trajectory: ReActTrajectory) -> ReActTrajectory:
        """Truncate trajectory by removing oldest steps."""
        # Keep last 70% of steps
        keep_count = max(1, int(len(trajectory.steps) * 0.7))
        new_trajectory = ReActTrajectory(
            steps=trajectory.steps[-keep_count:],
            final_prediction=trajectory.final_prediction,
            success=trajectory.success,
            total_usage=trajectory.total_usage,
        )

        # Update step numbers
        for i, step in enumerate(new_trajectory.steps):
            step.step_number = i

        return new_trajectory

    async def forward(self, **inputs: Any) -> Prediction:
        """Execute ReAct reasoning loop.

        Args:
            **inputs: Input values matching the original signature

        Returns:
            Prediction with final outputs and trajectory metadata
        """
        trajectory = ReActTrajectory()
        max_iters = inputs.pop("max_iters", self.max_iters)

        # Store original inputs for context
        original_inputs = inputs.copy()

        # Main reasoning loop
        for iteration in range(max_iters):
            try:
                # Get next reasoning step
                step_prediction = await self._call_with_trajectory_truncation(
                    self.react_predict, trajectory, **original_inputs
                )

                # Accumulate usage
                if step_prediction.usage:
                    trajectory.total_usage = self._add_usage(
                        trajectory.total_usage, step_prediction.usage
                    )

            except Exception as e:
                logger.error(f"ReAct reasoning failed at iteration {iteration}: {e}")
                raise ModuleError(
                    f"ReAct reasoning failed at iteration {iteration}",
                    module_name="ReAct",
                    execution_stage="reasoning",
                    context={
                        "iteration": iteration,
                        "trajectory_steps": len(trajectory.steps),
                        "original_error": str(e),
                    },
                ) from e

            # Extract reasoning outputs
            outputs = step_prediction.outputs
            thought = outputs.get("next_thought", "")
            tool_name = outputs.get("next_tool_name", "").strip()
            tool_args = outputs.get("next_tool_args", {})

            # Clean up tool name (but preserve original case for matching)
            tool_name_clean = tool_name.replace("##", "").replace("\n", " ").strip()

            # Find case-insensitive match in available tools
            actual_tool_name = None
            for available_name in self.tools.keys():
                if available_name.lower() == tool_name_clean.lower():
                    actual_tool_name = available_name
                    break

            tool_name = actual_tool_name or tool_name_clean

            # Parse tool args if string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args) if tool_args.strip() else {}
                except json.JSONDecodeError:
                    # Fallback: try to guess the right parameter name from the tool
                    if tool_name and tool_name in self.tools:
                        tool_params = list(self.tools[tool_name].args.keys())
                        if tool_params:
                            # Use the first parameter name
                            tool_args = {tool_params[0]: tool_args}
                        else:
                            tool_args = {"input": tool_args}
                    else:
                        tool_args = {"input": tool_args}

            # Create step
            step = ReActStep(
                thought=thought, tool_name=tool_name if tool_name else None, tool_args=tool_args
            )

            # Check if finishing
            if tool_name == "finish":
                step.observation = "Task completed."
                trajectory.add_step(step)
                break

            # Execute tool if specified
            if tool_name and tool_name in self.tools:
                try:
                    tool = self.tools[tool_name]
                    result = await tool.execute(**tool_args)

                    if result.success:
                        step.observation = str(result.output)
                    else:
                        step.observation = f"Tool error: {result.error}"

                except Exception as e:
                    step.observation = f"Tool execution failed: {str(e)}"
                    logger.error(f"Tool '{tool_name}' execution failed: {e}")

            elif tool_name:
                # Unknown tool
                available_tools = ", ".join(self.tools.keys())
                step.observation = f"Unknown tool '{tool_name}'. Available tools: {available_tools}"

            # Add step to trajectory
            trajectory.add_step(step)

        # Extract final answer using fallback module
        try:
            final_prediction = await self._call_with_trajectory_truncation(
                self.extract_predict, trajectory, **original_inputs
            )

            # Accumulate usage
            if final_prediction.usage:
                trajectory.total_usage = self._add_usage(
                    trajectory.total_usage, final_prediction.usage
                )

            trajectory.final_prediction = final_prediction.outputs
            trajectory.success = final_prediction.success

        except Exception as e:
            logger.error(f"Final extraction failed: {e}")
            # Fallback to last thought if extraction fails
            if trajectory.steps:
                last_thought = trajectory.steps[-1].thought
                output_fields = (
                    list(self.original_signature.output_fields.keys())
                    if self.original_signature
                    else ["answer"]
                )
                trajectory.final_prediction = dict.fromkeys(output_fields, last_thought)
            else:
                trajectory.final_prediction = {"answer": "No solution found"}

            trajectory.success = False

        # Create final prediction
        return Prediction(
            outputs=trajectory.final_prediction,
            usage=trajectory.total_usage,
            success=trajectory.success,
            metadata={
                "trajectory": trajectory.to_dict(),
                "iterations_used": len(trajectory.steps),
                "tools_called": list(
                    {
                        s.tool_name
                        for s in trajectory.steps
                        if s.tool_name and s.tool_name != "finish"
                    }
                ),
                "react_type": "full_reasoning_loop",
            },
        )

    def _add_usage(self, total: Usage, new: Usage) -> Usage:
        """Add usage statistics."""
        # Simple implementation - in practice would need proper Usage addition
        return new  # For now, just return the latest

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the available tools."""
        self.tools[tool.name] = tool
        # Recreate signature to include new tool
        self._create_react_signature()
        self.react_predict = Predict(self.react_signature, config=self.config)

    def remove_tool(self, name: str) -> None:
        """Remove a tool by name."""
        if name in self.tools and name != "finish":
            del self.tools[name]
            # Recreate signature
            self._create_react_signature()
            self.react_predict = Predict(self.react_signature, config=self.config)

    def list_tools(self) -> list[str]:
        """Get list of available tool names."""
        return [name for name in self.tools.keys() if name != "finish"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "max_iters": self.max_iters,
                "tools": [tool.to_dict() for name, tool in self.tools.items() if name != "finish"],
                "original_signature": self.original_signature.to_dict()
                if self.original_signature
                else None,
            }
        )
        return base_dict


__all__ = [
    "ReAct",
    "ReActStep",
    "ReActTrajectory",
]
