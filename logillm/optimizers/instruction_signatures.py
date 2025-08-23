"""Instruction generation signatures for COPRO optimizer."""

from ..core.signatures import InputField, OutputField, Signature


class BasicGenerateInstruction(Signature):
    """Generate improved instructions for LLM tasks.

    You are an instruction optimizer for large language models. I will give you
    the initial instructions for a task. Your task is to propose an improved
    instruction that will lead a good language model to perform the task better.
    Don't be afraid to be creative.
    """

    basic_instruction: str = InputField(desc="The initial instructions before optimization")
    proposed_instruction: str = OutputField(desc="The improved instructions for the language model")
    proposed_prefix: str = OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task"
    )


class GenerateInstructionGivenAttempts(Signature):
    """Generate instructions based on previous attempts and scores.

    You are an instruction optimizer for large language models. I will give some
    task instructions I've tried, along with their corresponding validation scores.
    The instructions are arranged in increasing order based on their scores, where
    higher scores indicate better quality.

    Your task is to propose a new instruction that will lead a good language model
    to perform the task even better. Don't be afraid to be creative.
    """

    attempted_instructions: list[str] = InputField(
        desc="Previous instruction attempts with their scores"
    )
    proposed_instruction: str = OutputField(desc="The improved instructions for the language model")
    proposed_prefix: str = OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task"
    )


__all__ = [
    "BasicGenerateInstruction",
    "GenerateInstructionGivenAttempts",
]
