"""Minimal ReAct - DSPy-level simplicity for LogiLLM."""

from __future__ import annotations

import json
import traceback
from typing import Any, Literal

from ..exceptions import ModuleError
from .modules import Module
from .predict import ChainOfThought, Predict
from .signatures import BaseSignature
from .tools import Tool
from .types import Configuration, Metadata, Prediction, Usage


class ReAct(Module):
    """Minimal ReAct implementation - true DSPy simplicity.
    
    ~230 lines, matching DSPy's elegance.
    """
    
    def __init__(
        self,
        signature: str,
        tools: list[Tool] | dict[str, Tool] | None = None,
        max_iters: int = 10,
        *,
        config: Configuration | None = None,
        metadata: Metadata | None = None,
    ):
        super().__init__(signature=signature, config=config, metadata=metadata)
        
        self.original_signature = self._resolve_signature(signature)
        self.max_iters = max_iters
        
        # Set up tools
        if isinstance(tools, dict):
            self.tools = tools
        elif tools:
            self.tools = {tool.name: tool for tool in tools}
        else:
            self.tools = {}
        
        # Add finish tool
        self.tools["finish"] = Tool(lambda: "Done", name="finish")
        
        # Create signatures and modules
        self._setup()
    
    def _setup(self) -> None:
        """Create minimal signatures for ReAct."""
        tool_names = tuple(self.tools.keys())
        
        # Simple instructions
        instructions = "Interleave Thought, Tool, and Observation steps.\n"
        for tool in self.tools.values():
            instructions += f"- {tool}\n"
        
        # ReAct signature with Literal type for tools
        from .signatures.spec import FieldSpec
        from .types import FieldType
        
        # Try Literal types, fall back to str
        try:
            from typing import Literal as LiteralType
            tool_type = LiteralType[tool_names] if tool_names else str
        except:
            tool_type = str
        
        self.react_signature = BaseSignature(
            input_fields={
                "trajectory": FieldSpec(
                    name="trajectory",
                    field_type=FieldType.INPUT,
                    python_type=str,
                    description="Previous steps",
                    required=True,
                ),
            },
            output_fields={
                "next_thought": FieldSpec(
                    name="next_thought",
                    field_type=FieldType.OUTPUT,
                    python_type=str,
                    description="Think step by step",
                    required=True,
                ),
                "next_tool_name": FieldSpec(
                    name="next_tool_name",
                    field_type=FieldType.OUTPUT,
                    python_type=tool_type,
                    description=f"Tool: {', '.join(tool_names)}",
                    required=True,
                ),
                "next_tool_args": FieldSpec(
                    name="next_tool_args",
                    field_type=FieldType.OUTPUT,
                    python_type=dict,
                    description="Tool arguments",
                    required=True,
                ),
            },
            instructions=instructions,
        )
        
        # Extract signature
        self.extract_signature = BaseSignature(
            input_fields={
                **self.original_signature.input_fields,
                "trajectory": "string",
            },
            output_fields=self.original_signature.output_fields,
            instructions="Extract the final answer from the trajectory.",
        )
        
        # Create modules
        provider = self.config.get('provider') if self.config else None
        clean_config = {k: v for k, v in (self.config or {}).items() if k != 'provider'}
        
        self.react = Predict(self.react_signature, provider=provider, config=clean_config)
        self.extract = ChainOfThought(self.extract_signature, provider=provider, config=clean_config)
    
    def _format_trajectory(self, trajectory: dict) -> str:
        """Format trajectory as simple string."""
        if not trajectory:
            return "No previous steps."
        
        lines = []
        idx = 0
        while f"thought_{idx}" in trajectory:
            lines.append(f"Thought: {trajectory[f'thought_{idx}']}")
            if f"tool_name_{idx}" in trajectory:
                lines.append(f"Tool: {trajectory[f'tool_name_{idx}']}")
                lines.append(f"Args: {trajectory.get(f'tool_args_{idx}', {})}")
            if f"observation_{idx}" in trajectory:
                lines.append(f"Observation: {trajectory[f'observation_{idx}']}")
            idx += 1
        
        return "\n".join(lines)
    
    async def forward(self, **inputs: Any) -> Prediction:
        """Execute ReAct loop."""
        trajectory = {}
        max_iters = inputs.pop("max_iters", self.max_iters)
        
        for idx in range(max_iters):
            # Get next step
            trajectory_str = self._format_trajectory(trajectory)
            pred = await self.react(trajectory=trajectory_str, **inputs)
            
            # Extract outputs
            thought = pred.outputs.get("next_thought", "")
            tool_name = pred.outputs.get("next_tool_name", "").strip()
            tool_args = pred.outputs.get("next_tool_args", {})
            
            # Parse args if string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args) if tool_args else {}
                except:
                    tool_args = {}
            
            # Store in trajectory
            trajectory[f"thought_{idx}"] = thought
            trajectory[f"tool_name_{idx}"] = tool_name
            trajectory[f"tool_args_{idx}"] = tool_args
            
            # Check finish
            if tool_name == "finish":
                trajectory[f"observation_{idx}"] = "Task completed."
                break
            
            # Execute tool
            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name].execute(**tool_args)
                    trajectory[f"observation_{idx}"] = str(result.output) if result.success else f"Error: {result.error}"
                except Exception as e:
                    trajectory[f"observation_{idx}"] = f"Tool error: {e}"
            elif tool_name:
                trajectory[f"observation_{idx}"] = f"Unknown tool: {tool_name}"
                break
        
        # Extract final answer
        try:
            trajectory_str = self._format_trajectory(trajectory)
            final = await self.extract(trajectory=trajectory_str, **inputs)
            return Prediction(
                outputs=final.outputs,
                usage=final.usage,
                success=final.success,
                metadata={
                    "trajectory": trajectory,
                    "steps": idx + 1,
                },
            )
        except Exception as e:
            # Fallback to last thought
            return Prediction(
                outputs={"answer": trajectory.get(f"thought_{idx}", "No answer")},
                success=False,
                metadata={"trajectory": trajectory, "error": str(e)},
            )


__all__ = ["ReAct"]