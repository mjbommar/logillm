"""Utility functions for SIMBA optimizer."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from ..core.modules import Module
from ..core.signatures import BaseSignature
from ..core.signatures.spec import FieldSpec
from ..core.types import FieldType, Prediction

logger = logging.getLogger(__name__)


class MockLM:
    """Mock language model for temperature variations."""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.kwargs = {"temperature": temperature}

    def copy(self, **kwargs: Any) -> MockLM:
        """Create copy with updated parameters."""
        new_temp = kwargs.get("temperature", self.temperature)
        return MockLM(temperature=new_temp)


async def prepare_models_for_resampling(program: Module, n: int) -> list[MockLM]:
    """Prepare different language models with varying temperatures.

    Args:
        program: Base program to extract LM settings from
        n: Number of models to create

    Returns:
        List of mock LM objects with different temperatures
    """
    # Extract base temperature from program if available
    base_temp = 0.7
    if hasattr(program, "metadata") and "temperature" in program.metadata:
        base_temp = program.metadata["temperature"]

    # Generate temperature variations
    temps = [base_temp] + [0.5 + i * (0.5 / n) for i in range(n)]
    temps = list(dict.fromkeys(temps))[:n]  # Remove duplicates and limit to n

    return [MockLM(temperature=t) for t in temps]


def wrap_program(
    program: Module, metric: Callable[[Any, Any], float]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a program for evaluation with metric.

    Args:
        program: Module to wrap
        metric: Metric function

    Returns:
        Callable that takes an example and returns evaluation results
    """

    def wrapped_program(example: dict[str, Any]) -> dict[str, Any]:
        """Execute program and return evaluation results."""
        prediction = None
        trace = []
        score = 0.0

        try:
            # Enable tracing for this execution
            program.enable_tracing()
            program.clear_trace()

            # Extract inputs from example
            inputs = example.get("inputs", {})
            if not inputs:
                # If no explicit inputs, use the whole example except outputs
                inputs = {k: v for k, v in example.items() if k != "outputs"}

            # Run the program synchronously (SIMBA expects sync execution)
            prediction = program.call_sync(**inputs)

            # Extract trace
            execution_trace = program.get_trace()
            if execution_trace and execution_trace.steps:
                # Convert LogiLLM trace to DSPy-style trace
                for step in execution_trace.steps:
                    # Create mock predictor object
                    mock_predictor = type("MockPredictor", (), {"__module__": step.module_name})()
                    trace.append((mock_predictor, step.inputs, step.outputs))

            # Compute score using metric
            expected = example.get("outputs", {})
            if prediction and prediction.success:
                score = metric(prediction.outputs, expected)

        except Exception as e:
            logger.error(f"Program execution failed: {e}")
            prediction = Prediction(outputs={}, success=False, error=str(e))
            score = 0.0

        finally:
            program.disable_tracing()

        return {
            "prediction": prediction.outputs if prediction else {},
            "trace": trace,
            "score": score,
            "example": example,
        }

    return wrapped_program


def inspect_modules(program: Module) -> str:
    """Generate a textual description of program modules.

    Args:
        program: Program to inspect

    Returns:
        Formatted string describing the modules
    """
    separator = "-" * 80
    output = [separator]

    # Get module information
    module_name = program.__class__.__name__
    output.append(f"Module {module_name}")

    # Add signature information if available
    if hasattr(program, "signature") and program.signature:
        output.append("\n\tInput Fields:")
        if hasattr(program.signature, "input_fields"):
            for field_name, field in program.signature.input_fields.items():
                desc = getattr(field, "desc", "No description")
                output.append(f"\t\t{field_name}: {desc}")

        output.append("\tOutput Fields:")
        if hasattr(program.signature, "output_fields"):
            for field_name, field in program.signature.output_fields.items():
                desc = getattr(field, "desc", "No description")
                output.append(f"\t\t{field_name}: {desc}")

        instructions = ""
        if hasattr(program.signature, "instructions"):
            instructions = program.signature.instructions
        elif hasattr(program, "parameters") and "instruction" in program.parameters:
            instructions = program.parameters["instruction"].value

        if instructions:
            formatted_instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())
            output.append(f"\tInstructions: {formatted_instructions}")

    # Add parameter information
    if hasattr(program, "parameters") and program.parameters:
        output.append("\n\tParameters:")
        for name, param in program.parameters.items():
            param_type = type(param.value).__name__
            output.append(f"\t\t{name}: {param_type} (learnable={param.learnable})")

    output.append(separator)
    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(obj: Any) -> Any:
    """Recursively mask non-serializable objects.

    Args:
        obj: Object to mask

    Returns:
        Serializable version of the object
    """
    # If the object is already serializable, return it
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    # If it's a dictionary, apply recursively to its values
    if isinstance(obj, dict):
        return {k: recursive_mask(v) for k, v in obj.items()}
    # If it's a list, apply recursively
    elif isinstance(obj, list):
        return [recursive_mask(v) for v in obj]
    # If it's a tuple, apply recursively
    elif isinstance(obj, tuple):
        return tuple(recursive_mask(v) for v in obj)
    # For LogiLLM objects, try to extract useful information
    elif hasattr(obj, "to_dict"):
        try:
            return recursive_mask(obj.to_dict())
        except Exception:
            pass
    elif hasattr(obj, "__dict__"):
        try:
            obj_dict = obj.__dict__
            # If the __dict__ is empty or only contains non-serializable items,
            # return a placeholder instead
            if not obj_dict:
                return f"<non-serializable: {type(obj).__name__}>"
            masked_dict = recursive_mask(obj_dict)
            # Check if the masked dict is substantially different (i.e., had non-serializable items)
            return masked_dict
        except Exception:
            pass

    # Check for special types that should be converted to placeholders
    if callable(obj):
        return f"<non-serializable: {type(obj).__name__}>"

    # Otherwise, replace with a placeholder string
    else:
        return f"<non-serializable: {type(obj).__name__}>"


def create_offer_feedback_signature():
    """Create OfferFeedback signature with proper field specifications."""
    instructions = """
    You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
    build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
    similar inputs in the future.

    The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

    In your advice:
    - Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
    - Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
    - Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
    - Ensure each unique module name appears exactly once as a key in the advice dictionary.
    """

    input_fields = {
        "program_code": FieldSpec(
            name="program_code",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The code of the program that we are analyzing",
        ),
        "modules_defn": FieldSpec(
            name="modules_defn",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The definition of each module in the program, including its I/O",
        ),
        "program_inputs": FieldSpec(
            name="program_inputs",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The inputs to the program that we are analyzing",
        ),
        "oracle_metadata": FieldSpec(
            name="oracle_metadata",
            field_type=FieldType.INPUT,
            python_type=str,
            description="Any (hidden) metadata about the training set instance we're analyzing",
        ),
        "worse_program_trajectory": FieldSpec(
            name="worse_program_trajectory",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The trajectory of the program's execution, showing each module's I/O",
        ),
        "worse_program_outputs": FieldSpec(
            name="worse_program_outputs",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The outputs of the program that we are analyzing",
        ),
        "worse_reward_value": FieldSpec(
            name="worse_reward_value",
            field_type=FieldType.INPUT,
            python_type=float,
            description="The reward value assigned to the program's outputs",
        ),
        "better_program_trajectory": FieldSpec(
            name="better_program_trajectory",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The trajectory of the program's execution, showing each module's I/O",
        ),
        "better_program_outputs": FieldSpec(
            name="better_program_outputs",
            field_type=FieldType.INPUT,
            python_type=str,
            description="The outputs of the program that we are analyzing",
        ),
        "better_reward_value": FieldSpec(
            name="better_reward_value",
            field_type=FieldType.INPUT,
            python_type=float,
            description="The reward value assigned to the program's outputs",
        ),
        "module_names": FieldSpec(
            name="module_names",
            field_type=FieldType.INPUT,
            python_type=list,
            description="The names of the modules in the program, for which we seek advice",
        ),
    }

    output_fields = {
        "discussion": FieldSpec(
            name="discussion",
            field_type=FieldType.OUTPUT,
            python_type=str,
            description="Discussing blame of where each module went wrong, if it did",
        ),
        "module_advice": FieldSpec(
            name="module_advice",
            field_type=FieldType.OUTPUT,
            python_type=dict,
            description="For each module, describe very concretely: If the module receives ${description of input or patterns "
            "therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. "
            "Basically, your advice be such that if the module has access to your tip, it would be much more likely to act "
            "like the successful trajectory rather than the lower-scoring trajectory.",
        ),
    }

    return BaseSignature(
        input_fields=input_fields, output_fields=output_fields, instructions=instructions
    )


# Create the signature instance
OfferFeedback = create_offer_feedback_signature()


__all__ = [
    "prepare_models_for_resampling",
    "wrap_program",
    "inspect_modules",
    "recursive_mask",
    "OfferFeedback",
    "MockLM",
]
