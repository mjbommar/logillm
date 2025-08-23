"""GroundedProposer for instruction generation based on DSPy's implementation."""

import logging
import random
from typing import Any

from ...core.predict import Predict
from ...providers import get_provider
from .base import InstructionProposal, InstructionProposer, ProposalStrategy

logger = logging.getLogger(__name__)


# Prompting tips from research papers
PROMPTING_TIPS = [
    "Think step-by-step through the problem.",
    "Break down complex problems into simpler sub-problems.",
    "Consider multiple perspectives before answering.",
    "Verify your reasoning before providing the final answer.",
    "Use examples to clarify your thinking.",
    "Be concise but thorough in your explanation.",
    "Focus on accuracy over speed.",
    "Double-check your work for errors.",
    "Consider edge cases and exceptions.",
    "Explain your reasoning clearly.",
    "Use structured thinking to organize your thoughts.",
    "Validate assumptions before proceeding.",
    "Consider alternative approaches.",
    "Be systematic in your analysis.",
    "Prioritize clarity and correctness.",
]


class GroundedProposer(InstructionProposer):
    """Grounded proposer that generates instructions based on multiple strategies.

    Based on DSPy's MIPROv2 GroundedProposer implementation.
    Uses program structure, data characteristics, prompting tips, and demonstrations
    to generate diverse instruction proposals.
    """

    def __init__(self, provider=None):
        """Initialize the proposer.

        Args:
            provider: LLM provider to use for generation (defaults to system default)
        """
        self.provider = provider or get_provider()

        # Create instruction generator module
        self.instruction_generator = Predict(
            "context, strategy, examples -> instruction", provider=self.provider
        )

    async def propose(
        self,
        module: Any,
        dataset: list[dict[str, Any]],
        demonstrations: list[dict[str, Any]] | None = None,
        num_proposals: int = 5,
        strategy: ProposalStrategy = ProposalStrategy.ALL,
    ) -> list[InstructionProposal]:
        """Propose instructions using grounded strategies.

        Args:
            module: The module to optimize
            dataset: Training dataset
            demonstrations: Optional bootstrapped demonstrations
            num_proposals: Number of proposals to generate
            strategy: Proposal strategy to use

        Returns:
            List of instruction proposals
        """
        proposals = []

        # Determine which strategies to use
        strategies = self._get_strategies(strategy)

        # Generate proposals per strategy
        proposals_per_strategy = max(1, num_proposals // len(strategies))

        for strat in strategies:
            if strat == ProposalStrategy.PROGRAM_AWARE:
                props = await self._program_aware_proposals(module, dataset, proposals_per_strategy)
                proposals.extend(props)

            elif strat == ProposalStrategy.DATA_AWARE:
                props = await self._data_aware_proposals(dataset, proposals_per_strategy)
                proposals.extend(props)

            elif strat == ProposalStrategy.TIP_AWARE:
                props = await self._tip_aware_proposals(module, dataset, proposals_per_strategy)
                proposals.extend(props)

            elif strat == ProposalStrategy.FEWSHOT_AWARE:
                if demonstrations:
                    props = await self._fewshot_aware_proposals(
                        demonstrations, proposals_per_strategy
                    )
                    proposals.extend(props)

        # Ensure we have exactly num_proposals
        if len(proposals) > num_proposals:
            proposals = proposals[:num_proposals]
        elif len(proposals) < num_proposals:
            # Generate additional proposals using random strategy
            remaining = num_proposals - len(proposals)
            extra = await self._mixed_strategy_proposals(module, dataset, demonstrations, remaining)
            proposals.extend(extra)

        return proposals

    def _get_strategies(self, strategy: ProposalStrategy) -> list[ProposalStrategy]:
        """Get list of strategies to use."""
        if strategy == ProposalStrategy.ALL:
            return [
                ProposalStrategy.PROGRAM_AWARE,
                ProposalStrategy.DATA_AWARE,
                ProposalStrategy.TIP_AWARE,
                ProposalStrategy.FEWSHOT_AWARE,
            ]
        else:
            return [strategy]

    async def _program_aware_proposals(
        self, module: Any, dataset: list[dict[str, Any]], num_proposals: int
    ) -> list[InstructionProposal]:
        """Generate proposals based on program structure."""
        proposals = []

        # Analyze module structure
        module_info = self._analyze_module(module)

        context = f"""
        Module Type: {module_info["type"]}
        Signature: {module_info["signature"]}
        Input Fields: {module_info["input_fields"]}
        Output Fields: {module_info["output_fields"]}

        Task: Generate an instruction that helps this module perform better.
        Focus on the module's structure and expected behavior.
        """

        for _ in range(num_proposals):
            result = await self.instruction_generator.forward(
                context=context,
                strategy="program_aware",
                examples=str(dataset[:3]) if dataset else "No examples available",
            )

            if result.success:
                instruction = result.outputs.get("instruction", "")
                proposals.append(
                    InstructionProposal(
                        instruction=instruction,
                        strategy=ProposalStrategy.PROGRAM_AWARE,
                        metadata={"module_info": module_info},
                    )
                )

        return proposals

    async def _data_aware_proposals(
        self, dataset: list[dict[str, Any]], num_proposals: int
    ) -> list[InstructionProposal]:
        """Generate proposals based on dataset characteristics."""
        proposals = []

        # Analyze dataset
        data_summary = self._summarize_dataset(dataset)

        context = f"""
        Dataset Summary:
        - Number of examples: {data_summary["num_examples"]}
        - Input pattern: {data_summary["input_pattern"]}
        - Output pattern: {data_summary["output_pattern"]}
        - Task type: {data_summary["task_type"]}

        Task: Generate an instruction that helps solve this type of problem.
        Focus on the patterns and characteristics in the data.
        """

        for _ in range(num_proposals):
            result = await self.instruction_generator.forward(
                context=context,
                strategy="data_aware",
                examples=str(dataset[:3]) if dataset else "No examples",
            )

            if result.success:
                instruction = result.outputs.get("instruction", "")
                proposals.append(
                    InstructionProposal(
                        instruction=instruction,
                        strategy=ProposalStrategy.DATA_AWARE,
                        metadata={"data_summary": data_summary},
                    )
                )

        return proposals

    async def _tip_aware_proposals(
        self, module: Any, dataset: list[dict[str, Any]], num_proposals: int
    ) -> list[InstructionProposal]:
        """Generate proposals using prompting tips from research."""
        proposals = []

        for _ in range(num_proposals):
            # Select random tips
            selected_tips = random.sample(PROMPTING_TIPS, min(3, len(PROMPTING_TIPS)))

            context = f"""
            Research has shown these prompting strategies work well:
            {chr(10).join(f"- {tip}" for tip in selected_tips)}

            Task: Create an instruction that incorporates these strategies.
            Make it natural and specific to the task at hand.
            """

            result = await self.instruction_generator.forward(
                context=context,
                strategy="tip_aware",
                examples=str(dataset[:2]) if dataset else "General task",
            )

            if result.success:
                instruction = result.outputs.get("instruction", "")
                proposals.append(
                    InstructionProposal(
                        instruction=instruction,
                        strategy=ProposalStrategy.TIP_AWARE,
                        metadata={"tips_used": selected_tips},
                    )
                )

        return proposals

    async def _fewshot_aware_proposals(
        self, demonstrations: list[dict[str, Any]], num_proposals: int
    ) -> list[InstructionProposal]:
        """Generate proposals based on successful demonstrations."""
        proposals = []

        # Analyze demonstrations for patterns
        demo_patterns = self._analyze_demonstrations(demonstrations)

        context = f"""
        Successful demonstrations show these patterns:
        - Common reasoning: {demo_patterns["reasoning_pattern"]}
        - Output format: {demo_patterns["output_format"]}
        - Key insights: {demo_patterns["insights"]}

        Task: Generate an instruction that captures what makes these examples successful.
        """

        for _ in range(num_proposals):
            result = await self.instruction_generator.forward(
                context=context,
                strategy="fewshot_aware",
                examples=str(demonstrations[:2]) if demonstrations else "No demos",
            )

            if result.success:
                instruction = result.outputs.get("instruction", "")
                proposals.append(
                    InstructionProposal(
                        instruction=instruction,
                        strategy=ProposalStrategy.FEWSHOT_AWARE,
                        metadata={"demo_patterns": demo_patterns},
                    )
                )

        return proposals

    async def _mixed_strategy_proposals(
        self,
        module: Any,
        dataset: list[dict[str, Any]],
        demonstrations: list[dict[str, Any]],
        num_proposals: int,
    ) -> list[InstructionProposal]:
        """Generate proposals using mixed strategies."""
        proposals = []

        for _ in range(num_proposals):
            # Randomly select a strategy
            strategy = random.choice(
                [
                    ProposalStrategy.PROGRAM_AWARE,
                    ProposalStrategy.DATA_AWARE,
                    ProposalStrategy.TIP_AWARE,
                ]
            )

            if strategy == ProposalStrategy.PROGRAM_AWARE:
                props = await self._program_aware_proposals(module, dataset, 1)
            elif strategy == ProposalStrategy.DATA_AWARE:
                props = await self._data_aware_proposals(dataset, 1)
            else:
                props = await self._tip_aware_proposals(module, dataset, 1)

            proposals.extend(props)

        return proposals

    def _analyze_module(self, module: Any) -> dict[str, Any]:
        """Analyze module structure."""
        info = {
            "type": type(module).__name__,
            "signature": "",
            "input_fields": [],
            "output_fields": [],
        }

        # Extract signature information
        if hasattr(module, "signature"):
            sig = module.signature
            if hasattr(sig, "signature_str"):
                info["signature"] = sig.signature_str
            if hasattr(sig, "input_fields"):
                info["input_fields"] = list(sig.input_fields.keys())
            if hasattr(sig, "output_fields"):
                info["output_fields"] = list(sig.output_fields.keys())

        return info

    def _summarize_dataset(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize dataset characteristics."""
        if not dataset:
            return {
                "num_examples": 0,
                "input_pattern": "unknown",
                "output_pattern": "unknown",
                "task_type": "unknown",
            }

        summary = {
            "num_examples": len(dataset),
            "input_pattern": "",
            "output_pattern": "",
            "task_type": "general",
        }

        # Analyze first few examples
        if dataset and "inputs" in dataset[0]:
            input_keys = list(dataset[0]["inputs"].keys())
            summary["input_pattern"] = f"Fields: {', '.join(input_keys)}"

        if dataset and "outputs" in dataset[0]:
            output_keys = list(dataset[0]["outputs"].keys())
            summary["output_pattern"] = f"Fields: {', '.join(output_keys)}"

            # Guess task type
            if "answer" in output_keys:
                summary["task_type"] = "question_answering"
            elif "label" in output_keys or "category" in output_keys:
                summary["task_type"] = "classification"
            elif "code" in output_keys:
                summary["task_type"] = "code_generation"

        return summary

    def _analyze_demonstrations(self, demonstrations: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze patterns in demonstrations."""
        if not demonstrations:
            return {
                "reasoning_pattern": "none",
                "output_format": "none",
                "insights": "none",
            }

        patterns = {
            "reasoning_pattern": "step-by-step",
            "output_format": "structured",
            "insights": [],
        }

        # Look for common patterns
        for demo in demonstrations[:5]:  # Analyze first 5
            if "reasoning" in demo.get("outputs", {}):
                if "step" in str(demo["outputs"]["reasoning"]).lower():
                    patterns["reasoning_pattern"] = "step-by-step reasoning"

            if "outputs" in demo:
                if isinstance(demo["outputs"], dict):
                    patterns["output_format"] = f"Dict with keys: {list(demo['outputs'].keys())}"

        # Extract insights
        patterns["insights"] = "Successful examples show clear reasoning and structured outputs"

        return patterns
