"""Avatar module - Tool-using agent for LogiLLM."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

from ..exceptions import ConfigurationError
from .modules import Module, Parameter
from .predict import Predict
from .signatures import BaseSignature, FieldSpec, parse_signature_string
from .tools.base import Tool
from .types import FieldType, Prediction

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """An action to be taken by the Avatar."""

    tool_name: str
    tool_input_query: str

    def __str__(self) -> str:
        return f"Action(tool={self.tool_name}, input={self.tool_input_query})"


@dataclass
class ActionOutput:
    """Result of executing an action."""

    tool_name: str
    tool_input_query: str
    tool_output: str

    def __str__(self) -> str:
        return f"ActionOutput(tool={self.tool_name}, output={self.tool_output[:100]}...)"


class ActorSignature(BaseSignature):
    """Base signature for the Actor module that makes decisions."""

    def __init__(self):
        super().__init__(
            input_fields={
                "goal": FieldSpec(
                    name="goal",
                    field_type=FieldType.INPUT,
                    python_type=str,
                    description="Task to be accomplished",
                    required=True,
                ),
                "tools": FieldSpec(
                    name="tools",
                    field_type=FieldType.INPUT,
                    python_type=list,
                    description="List of available tools",
                    required=True,
                ),
            },
            output_fields={
                "action_1": FieldSpec(
                    name="action_1",
                    field_type=FieldType.OUTPUT,
                    python_type=str,
                    description="First action to take: tool_name(tool_input_query)",
                    required=True,
                ),
            },
            instructions=(
                "You will be given a Goal and a list of Tools. Your task is to decide which tool to use "
                "and what input to provide. Output the action in the format: tool_name(tool_input_query). "
                "You can use the 'Finish' tool to complete the task with your final answer. "
                "Choose the most appropriate tool for making progress toward the goal."
            ),
        )


def get_number_with_suffix(number: int) -> str:
    """Get ordinal suffix for a number."""
    if number == 1:
        return "1st"
    elif number == 2:
        return "2nd"
    elif number == 3:
        return "3rd"
    else:
        return f"{number}th"


def parse_action(action_str: str) -> Action:
    """Parse action string into Action object.

    Expected formats:
    - tool_name(input_query)
    - Finish(final_answer)
    """
    action_str = action_str.strip()

    # Find the opening parenthesis
    paren_pos = action_str.find("(")
    if paren_pos == -1:
        # No parentheses found, treat the whole string as tool name with empty input
        return Action(tool_name=action_str, tool_input_query="")

    tool_name = action_str[:paren_pos].strip()

    # Extract everything between the first ( and last )
    if action_str.endswith(")"):
        tool_input = action_str[paren_pos + 1 : -1].strip()
    else:
        # No closing parenthesis, take everything after the opening one
        tool_input = action_str[paren_pos + 1 :].strip()

    return Action(tool_name=tool_name, tool_input_query=tool_input)


class Avatar(Module):
    """Avatar module for tool-using agents.

    The Avatar executes a loop where it:
    1. Decides what action to take using tools
    2. Executes the chosen tool
    3. Adds the result to its context
    4. Repeats until the 'Finish' tool is chosen

    This is LogiLLM's implementation of DSPy's Avatar module,
    adapted for our zero-dependency architecture.
    """

    def __init__(
        self,
        signature: BaseSignature | str,
        tools: list[Tool],
        max_iters: int = 3,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """Initialize Avatar module.

        Args:
            signature: The task signature defining inputs and outputs
            tools: List of tools available to the agent
            max_iters: Maximum number of action iterations
            verbose: Whether to print debug information
            **kwargs: Additional arguments for Module
        """
        # Resolve signature
        if isinstance(signature, str):
            try:
                signature = parse_signature_string(signature)
            except Exception as e:
                raise ConfigurationError(f"Invalid signature string: {e}") from e

        # Extract provider if provided
        provider = kwargs.pop("provider", None)

        super().__init__(signature=signature, **kwargs)

        self.input_fields = signature.input_fields if signature else {}
        self.output_fields = signature.output_fields if signature else {}
        self.max_iters = max_iters
        self.verbose = verbose

        # Create finish tool
        def finish_tool_fn(final_answer: str) -> str:
            """Return the final answer and finish the task."""
            return final_answer

        self.finish_tool = Tool(
            func=finish_tool_fn,
            name="Finish",
            desc="Returns the final output and finishes the task",
        )

        # Store tools (including finish tool)
        self.tools = tools + [self.finish_tool]

        # Create base actor signature and predictor
        self.actor_signature = ActorSignature()

        # Add task input fields to actor signature
        for field_name, field_spec in self.input_fields.items():
            self.actor_signature.input_fields[field_name] = field_spec

        # Create actor with provider if available
        actor_kwargs = {}
        if provider:
            actor_kwargs["provider"] = provider
        self.actor = Predict(signature=self.actor_signature, **actor_kwargs)
        self.actor_clone = copy.deepcopy(self.actor)

        # Parameters for optimization
        self.parameters["tools"] = Parameter(
            value=self.tools, learnable=False, metadata={"type": "tools"}
        )
        self.parameters["max_iters"] = Parameter(
            value=max_iters, learnable=True, metadata={"type": "hyperparameter"}
        )

    def _update_actor_signature(self, idx: int, omit_action: bool = False) -> None:
        """Update the actor signature to include previous actions and results."""
        # Add the previous action as an input field
        action_field_name = f"action_{idx}"
        self.actor_signature.input_fields[action_field_name] = FieldSpec(
            name=action_field_name,
            field_type=FieldType.INPUT,
            python_type=str,
            description=f"{get_number_with_suffix(idx)} action taken",
            required=True,
        )

        # Add the result as an input field
        result_field_name = f"result_{idx}"
        self.actor_signature.input_fields[result_field_name] = FieldSpec(
            name=result_field_name,
            field_type=FieldType.INPUT,
            python_type=str,
            description=f"Result of the {get_number_with_suffix(idx)} action",
            required=True,
        )

        if omit_action:
            # Add final output fields
            for field_name, field_spec in self.output_fields.items():
                self.actor_signature.output_fields[field_name] = field_spec
        else:
            # Add next action as output field
            next_action_field_name = f"action_{idx + 1}"
            self.actor_signature.output_fields[next_action_field_name] = FieldSpec(
                name=next_action_field_name,
                field_type=FieldType.OUTPUT,
                python_type=str,
                description=f"{get_number_with_suffix(idx + 1)} action to take",
                required=True,
            )

        # Update the actor's signature
        self.actor.signature = self.actor_signature

    def _call_tool(self, tool_name: str, tool_input_query: str) -> str:
        """Execute a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    if tool.name == "Finish":
                        # For finish tool, just return the input
                        return tool_input_query
                    else:
                        # Execute the tool - try to determine the correct parameter name
                        # Get the first parameter name from the tool's args
                        if tool.args:
                            # Use the first argument name
                            param_name = list(tool.args.keys())[0]
                            result = tool(**{param_name: tool_input_query})
                        else:
                            # Fallback to calling with no specific parameter name
                            result = tool(tool_input_query)
                        return str(result)
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' execution failed: {e}")
                    return f"Error executing tool '{tool_name}': {e}"

        # Tool not found
        available_tools = [t.name for t in self.tools]
        return f"Tool '{tool_name}' not found. Available tools: {available_tools}"

    async def forward(self, **inputs: Any) -> Prediction:
        """Execute the Avatar's tool-using loop."""
        if self.verbose:
            print("Starting Avatar task execution...")

        # Prepare initial arguments for the actor
        args = {
            "goal": self.signature.instructions if self.signature else "Complete the task",
            "tools": [f"{tool.name}: {tool.desc}" for tool in self.tools],
        }

        # Add task-specific inputs
        for field_name in self.input_fields.keys():
            if field_name in inputs:
                args[field_name] = inputs[field_name]

        # Track action history
        action_results: list[ActionOutput] = []
        idx = 1
        tool_name = None
        max_iters = inputs.get("max_iters", self.max_iters)

        # Track if max_iters was originally set
        has_max_iters = max_iters is not None

        # Main execution loop
        # Add hard limit to prevent infinite loops even if max_iters is not set
        hard_limit = 100
        while tool_name != "Finish" and (max_iters > 0 if has_max_iters else hard_limit > 0):
            try:
                # Get next action from actor
                actor_output = await self.actor(**args)

                # Extract action
                action_field = f"action_{idx}"
                if action_field in actor_output.outputs:
                    action_str = actor_output.outputs[action_field]
                elif "output" in actor_output.outputs:
                    # Common fallback field from mock providers
                    action_str = actor_output.outputs["output"]
                elif actor_output.outputs:
                    # Fallback to first output field
                    action_str = list(actor_output.outputs.values())[0]
                else:
                    raise ValueError("No output from actor")

                # Parse the action
                action = parse_action(action_str)
                tool_name = action.tool_name
                tool_input_query = action.tool_input_query

                if self.verbose:
                    print(f"Action {idx}: {tool_name}({tool_input_query})")

                if tool_name != "Finish":
                    # Execute the tool
                    tool_output = self._call_tool(tool_name, tool_input_query)
                    action_results.append(
                        ActionOutput(
                            tool_name=tool_name,
                            tool_input_query=tool_input_query,
                            tool_output=tool_output,
                        )
                    )

                    # Update signature for next iteration
                    self._update_actor_signature(idx)

                    # Add action and result to args
                    args[f"action_{idx}"] = action_str
                    args[f"result_{idx}"] = tool_output
                else:
                    # Finish action - extract final output directly
                    # We don't need to call the actor again
                    tool_name = "Finish"
                    break

                idx += 1
                if max_iters:
                    max_iters -= 1
                else:
                    hard_limit -= 1

            except Exception as e:
                logger.error(f"Error in Avatar execution loop: {e}")
                # On error, break without updating signature or args
                # This prevents cascading errors in final output retrieval
                break

        # Get final outputs
        final_outputs = {}

        if tool_name == "Finish":
            # Use the Finish tool's input as the final output
            if self.output_fields:
                # If we have a single output field, use the finish content directly
                if len(self.output_fields) == 1:
                    output_field = list(self.output_fields.keys())[0]
                    final_outputs[output_field] = tool_input_query
                else:
                    # For multiple output fields, try to parse or distribute the content
                    # For now, just put it in the first field
                    first_field = list(self.output_fields.keys())[0]
                    final_outputs[first_field] = tool_input_query
            else:
                # No output fields defined, use a default
                final_outputs = {"result": tool_input_query}
        else:
            # No Finish action was called, need to get outputs from actor
            # This handles max_iters exhaustion or error case
            try:
                # Only update signature if we had at least one successful action
                if idx > 1:
                    self._update_actor_signature(idx - 1, omit_action=True)
                final_prediction = await self.actor(**args)

                # Extract outputs based on original signature
                for field_name in self.output_fields.keys():
                    if field_name in final_prediction.outputs:
                        final_outputs[field_name] = final_prediction.outputs[field_name]

                # If no specific outputs, use the last action result
                if not final_outputs and action_results:
                    if self.output_fields:
                        first_output_field = list(self.output_fields.keys())[0]
                        final_outputs = {first_output_field: action_results[-1].tool_output}
                    else:
                        final_outputs = {"output": action_results[-1].tool_output}
                elif not final_outputs:
                    # Map to the first output field if available
                    if self.output_fields:
                        first_output_field = list(self.output_fields.keys())[0]
                        final_outputs = {first_output_field: "Task completed"}
                    else:
                        final_outputs = {"output": "Task completed"}

            except Exception as e:
                logger.error(f"Error getting final outputs: {e}")
                # Map to the first output field if available
                if self.output_fields:
                    first_output_field = list(self.output_fields.keys())[0]
                    final_outputs = {first_output_field: f"Error completing task: {e}"}
                else:
                    final_outputs = {"output": f"Error completing task: {e}"}

        # Reset actor signature for next use
        self.actor = copy.deepcopy(self.actor_clone)

        # Create final prediction
        prediction = Prediction(
            outputs=final_outputs,
            metadata={
                "actions": [
                    {
                        "tool_name": action.tool_name,
                        "tool_input_query": action.tool_input_query,
                        "tool_output": action.tool_output,
                    }
                    for action in action_results
                ],
                "total_actions": len(action_results),
                "action_results": action_results,  # Store full action results in metadata
                **self.metadata,
            },
            success=True,
        )

        # Store actions as a direct attribute to avoid __setattr__ issues
        # Use object.__setattr__ to bypass the custom __setattr__ that would put it in outputs
        object.__setattr__(prediction, "actions", action_results)

        return prediction


__all__ = [
    "Avatar",
    "Action",
    "ActionOutput",
    "ActorSignature",
    "parse_action",
]
