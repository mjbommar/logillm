"""Base classes for instruction proposers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ProposalStrategy(Enum):
    """Strategy for instruction proposal."""

    PROGRAM_AWARE = "program_aware"  # Analyze code structure
    DATA_AWARE = "data_aware"  # Summarize dataset
    TIP_AWARE = "tip_aware"  # Include prompting tips
    FEWSHOT_AWARE = "fewshot_aware"  # Use bootstrapped examples
    ALL = "all"  # Use all strategies


@dataclass
class InstructionProposal:
    """A proposed instruction with metadata."""

    instruction: str
    strategy: ProposalStrategy
    score: Optional[float] = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InstructionProposer(ABC):
    """Base class for instruction proposers."""

    @abstractmethod
    async def propose(
        self,
        module: Any,
        dataset: list[dict[str, Any]],
        demonstrations: list[dict[str, Any]] | None = None,
        num_proposals: int = 5,
        strategy: ProposalStrategy = ProposalStrategy.ALL,
    ) -> list[InstructionProposal]:
        """Propose instructions for a module.

        Args:
            module: The module to optimize
            dataset: Training dataset
            demonstrations: Optional bootstrapped demonstrations
            num_proposals: Number of proposals to generate
            strategy: Proposal strategy to use

        Returns:
            List of instruction proposals
        """
        pass
