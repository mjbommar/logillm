"""Base classes for prompt optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.optimizers import OptimizationConfig, Optimizer

if TYPE_CHECKING:
    from ..core.types import Example


@dataclass
class PromptOptimizationConfig(OptimizationConfig):
    """Configuration specific to prompt optimization."""

    max_demos: int = 4
    max_instructions: int = 1
    teacher_settings: dict[str, Any] = field(default_factory=dict)
    demo_selection_strategy: str = "best"  # best, random, diverse
    instruction_generation_strategy: str = "template"  # template, learned
    use_validation_for_selection: bool = True
    min_demo_score: float = 0.5  # Minimum score for a demo to be considered


@dataclass
class Demonstration:
    """A single demonstration example."""

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_example(cls, example: dict[str, Any] | Example) -> Demonstration:
        """Create demonstration from example."""
        if isinstance(example, dict):
            return cls(
                inputs=example.get("inputs", {}),
                outputs=example.get("outputs", {}),
                score=example.get("score", 0.0),
                metadata=example.get("metadata", {}),
            )
        else:
            # Handle object with attributes
            return cls(
                inputs=getattr(example, "inputs", {}),
                outputs=getattr(example, "outputs", {}),
                score=getattr(example, "score", 0.0),
                metadata=getattr(example, "metadata", {}),
            )


class DemoSelector(ABC):
    """Abstract base class for demo selection strategies."""

    @abstractmethod
    def select(self, candidates: list[Demonstration], n: int, **kwargs: Any) -> list[Demonstration]:
        """Select n demonstrations from candidates."""
        ...


class PromptOptimizer(Optimizer):
    """Base class for prompt-based optimizers."""

    def __init__(self, *args, **kwargs):
        """Initialize prompt optimizer."""
        # Ensure we have a PromptOptimizationConfig
        if "config" in kwargs and not isinstance(kwargs["config"], PromptOptimizationConfig):
            # Convert to PromptOptimizationConfig
            base_config = kwargs["config"]
            kwargs["config"] = PromptOptimizationConfig(**base_config.__dict__)
        super().__init__(*args, **kwargs)

    def _extract_demonstrations(self, module) -> list[Demonstration]:
        """Extract demonstrations from a module."""
        demos = []

        # Check module parameters
        if hasattr(module, "parameters"):
            demo_param = module.parameters.get("demonstrations")
            if demo_param and demo_param.value:
                for demo_dict in demo_param.value:
                    demos.append(Demonstration.from_example(demo_dict))

        return demos

    def _extract_instruction(self, module) -> str:
        """Extract instruction from a module."""
        # Check parameters first
        if hasattr(module, "parameters"):
            inst_param = module.parameters.get("instruction")
            if inst_param:
                return inst_param.value

        # Check signature
        if hasattr(module, "signature") and module.signature:
            return getattr(module.signature, "instructions", "")

        return ""


__all__ = [
    "PromptOptimizationConfig",
    "Demonstration",
    "DemoSelector",
    "PromptOptimizer",
]
