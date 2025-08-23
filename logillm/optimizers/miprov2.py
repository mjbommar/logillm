"""MIPROv2 Optimizer - DSPy's flagship optimizer.

Implementation based on reference/dspy/notes.md lines 586-610.
Combines bootstrap few-shot, instruction proposals, and Bayesian optimization.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..core.modules import Module
from ..core.optimizers import Metric
from ..core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace
from ..core.types import OptimizationResult, OptimizationStrategy
from .base import Demonstration, PromptOptimizationConfig, PromptOptimizer
from .bootstrap_fewshot import BootstrapFewShot
from .proposers import GroundedProposer, InstructionProposal, ProposalStrategy
from .search_strategies import SimpleBayesianStrategy, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class MIPROv2Config(PromptOptimizationConfig):
    """Configuration for MIPROv2 optimizer.

    Based on reference/dspy/notes.md:602-605 auto settings.
    """

    # Mode presets (light/medium/heavy)
    mode: str = "medium"

    # Core parameters (set by mode)
    num_candidates: int | None = None  # N in DSPy
    validation_size: int | None = None  # val_size in DSPy

    # Bootstrap parameters
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    bootstrap_rounds: int = 1

    # Instruction proposal parameters
    num_instructions: int = 5
    proposal_strategy: ProposalStrategy = ProposalStrategy.ALL

    # Optimization parameters
    num_trials: int | None = None
    minibatch_size: int = 25
    full_eval_frequency: int = 5  # Full evaluation every N trials

    # Search strategy
    use_bayesian: bool = True
    exploration_weight: float = 1.5  # UCB exploration weight

    def __post_init__(self):
        """Apply mode presets."""
        # Auto settings from reference/dspy/notes.md:602-605
        if self.mode == "light":
            self.num_candidates = self.num_candidates or 6
            self.validation_size = self.validation_size or 100
            self.num_trials = self.num_trials or 20
        elif self.mode == "medium":
            self.num_candidates = self.num_candidates or 12
            self.validation_size = self.validation_size or 300
            self.num_trials = self.num_trials or 30
        elif self.mode == "heavy":
            self.num_candidates = self.num_candidates or 18
            self.validation_size = self.validation_size or 1000
            self.num_trials = self.num_trials or 50
        else:
            # Custom mode - use provided values or defaults
            self.num_candidates = self.num_candidates or 12
            self.validation_size = self.validation_size or 300
            self.num_trials = self.num_trials or 30


class MIPROv2Optimizer(PromptOptimizer):
    """MIPROv2 - Multi-Instruction Proposal and Optimization v2.

    DSPy's most advanced optimizer combining:
    1. Bootstrap few-shot examples (reference/dspy/notes.md:587-590)
    2. Instruction proposals via GroundedProposer (594-596)
    3. Bayesian optimization over combinations (597-601)
    4. Auto settings for different compute budgets (602-605)

    This is our implementation using SimpleBayesianStrategy instead of Optuna
    to maintain zero dependencies.
    """

    def __init__(
        self,
        metric: Metric,
        mode: str = "medium",
        config: Optional[MIPROv2Config] = None,
        proposer: Optional[GroundedProposer] = None,
        **kwargs: Any,
    ):
        """Initialize MIPROv2 optimizer.

        Args:
            metric: Evaluation metric
            mode: Optimization mode ('light', 'medium', 'heavy')
            config: Full configuration (overrides mode)
            proposer: Instruction proposer (defaults to GroundedProposer)
        """
        # Create config with mode
        if config is None:
            config = MIPROv2Config(mode=mode, **kwargs)

        super().__init__(strategy=OptimizationStrategy.BOOTSTRAP, metric=metric, config=config)

        self.config: MIPROv2Config = config
        self.proposer = proposer or GroundedProposer()

        # Initialize components
        self.bootstrapper = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            max_labeled_demos=config.max_labeled_demos,
            max_rounds=config.bootstrap_rounds,
        )

        # Tracking
        self.instruction_candidates: list[InstructionProposal] = []
        self.demo_candidates: list[list[Demonstration]] = []
        self.trial_history: list[dict[str, Any]] = []
        self.best_config: dict[str, Any] = None
        self.best_score: float = -float("inf")

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        valset: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize module using MIPROv2 algorithm.

        Algorithm from reference/dspy/notes.md:586-605:
        1. Bootstrap N candidate sets of demonstrations
        2. Propose M instruction variations
        3. Use Bayesian optimization to find best combination
        4. Periodic full evaluations for validation

        Args:
            module: Module to optimize
            dataset: Training dataset
            valset: Validation set (created from dataset if not provided)

        Returns:
            Optimized module with best instructions and demonstrations
        """
        start_time = time.time()
        logger.info(f"Starting MIPROv2 optimization in {self.config.mode} mode")

        # Step 1: Bootstrap demonstrations (reference/dspy/notes.md:587-590)
        logger.info(f"Step 1: Bootstrapping {self.config.num_candidates} demo sets")
        self.demo_candidates = await self._bootstrap_demonstrations(module, dataset)

        # Step 2: Propose instructions (reference/dspy/notes.md:591-596)
        logger.info(f"Step 2: Proposing {self.config.num_instructions} instructions")
        self.instruction_candidates = await self._propose_instructions(
            module, dataset, self.demo_candidates
        )

        # Step 3: Create validation set if needed
        if valset is None:
            val_size = min(self.config.validation_size, len(dataset) // 3)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            val_indices = indices[:val_size]
            valset = [dataset[i] for i in val_indices]
            logger.info(f"Created validation set with {len(valset)} examples")

        # Step 4: Bayesian optimization (reference/dspy/notes.md:597-601)
        logger.info(f"Step 3: Running Bayesian optimization for {self.config.num_trials} trials")
        optimized_module = await self._bayesian_optimize(module, valset)

        # Create result
        improvement = self.best_score
        if hasattr(module, "_baseline_score"):
            improvement = self.best_score - module._baseline_score

        metadata = {
            "mode": self.config.mode,
            "best_config": self.best_config,
            "num_instructions": len(self.instruction_candidates),
            "num_demo_sets": len(self.demo_candidates),
            "trial_history": self.trial_history[-10:],  # Last 10 trials
        }

        # Add validation_size if valset was used
        if valset is not None:
            metadata["validation_size"] = len(valset)

        return OptimizationResult(
            optimized_module=optimized_module,
            best_score=self.best_score,
            improvement=improvement,
            iterations=self.config.num_trials,
            optimization_time=time.time() - start_time,
            metadata=metadata,
        )

    async def _bootstrap_demonstrations(
        self, module: Module, dataset: list[dict[str, Any]]
    ) -> list[list[Demonstration]]:
        """Bootstrap N candidate sets of demonstrations.

        Reference: dspy/notes.md:587-590
        """
        demo_sets = []

        for i in range(self.config.num_candidates):
            logger.debug(f"Bootstrapping demo set {i + 1}/{self.config.num_candidates}")

            # Use different random seeds for diversity
            teacher_settings = {
                "temperature": 0.7 + (i * 0.1),  # Vary temperature
                "seed": i * 42,
            }

            # Bootstrap with variation
            result = await self.bootstrapper.optimize(
                module=module,
                dataset=dataset,
                teacher_settings=teacher_settings,
            )

            # Extract demonstrations from optimized module
            if hasattr(result.optimized_module, "demo_manager"):
                demos = result.optimized_module.demo_manager.get_best(
                    n=self.config.max_bootstrapped_demos
                )
                demo_sets.append(
                    [
                        Demonstration(
                            inputs=d.inputs,
                            outputs=d.outputs,
                            score=d.score,
                        )
                        for d in demos
                    ]
                )
            else:
                demo_sets.append([])  # Empty set if no demos

        logger.info(f"Bootstrapped {len(demo_sets)} demonstration sets")
        return demo_sets

    async def _propose_instructions(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        demo_candidates: list[list[Demonstration]],
    ) -> list[InstructionProposal]:
        """Propose instruction variations using GroundedProposer.

        Reference: dspy/notes.md:591-596
        """
        # Convert demonstrations for proposer
        all_demos = []
        for demo_set in demo_candidates:
            for demo in demo_set[:3]:  # Use first 3 from each set
                all_demos.append(
                    {
                        "inputs": demo.inputs,
                        "outputs": demo.outputs,
                    }
                )

        # Generate proposals
        proposals = await self.proposer.propose(
            module=module,
            dataset=dataset,
            demonstrations=all_demos,
            num_proposals=self.config.num_instructions,
            strategy=self.config.proposal_strategy,
        )

        logger.info(f"Generated {len(proposals)} instruction proposals")
        for i, prop in enumerate(proposals):
            logger.debug(f"Proposal {i + 1} ({prop.strategy.value}): {prop.instruction[:100]}...")

        return proposals

    async def _bayesian_optimize(self, module: Module, valset: list[dict[str, Any]]) -> Module:
        """Optimize instruction/demo combination using Bayesian optimization.

        Reference: dspy/notes.md:597-601
        Uses our SimpleBayesianStrategy instead of Optuna.
        """
        # Create search space
        search_space = self._create_search_space()

        # Initialize Bayesian strategy
        strategy = SimpleBayesianStrategy(
            config=StrategyConfig(
                n_warmup=min(10, self.config.num_trials // 3),
                exploration_weight=self.config.exploration_weight,
            )
        )
        strategy.initialize(search_space)

        # Optimization loop
        for trial in range(self.config.num_trials):
            # Get next configuration
            config = strategy.suggest_next()

            # Apply configuration to module
            trial_module = await self._apply_configuration(module, config)

            # Evaluate (minibatch or full)
            if trial % self.config.full_eval_frequency == 0:
                # Full evaluation
                score = await self._evaluate_module(trial_module, valset)
                eval_type = "full"
            else:
                # Minibatch evaluation
                minibatch = random.sample(valset, min(self.config.minibatch_size, len(valset)))
                score = await self._evaluate_module(trial_module, minibatch)
                eval_type = "minibatch"

            # Update strategy
            strategy.update(config, score)

            # Track history
            self.trial_history.append(
                {
                    "trial": trial,
                    "config": config,
                    "score": score,
                    "eval_type": eval_type,
                }
            )

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                logger.info(
                    f"Trial {trial + 1}/{self.config.num_trials}: "
                    f"New best score {score:.3f} ({eval_type} eval)"
                )
            else:
                logger.debug(
                    f"Trial {trial + 1}/{self.config.num_trials}: "
                    f"Score {score:.3f} ({eval_type} eval)"
                )

        # Apply best configuration to final module
        optimized_module = await self._apply_configuration(module, self.best_config)

        return optimized_module

    def _create_search_space(self) -> SearchSpace:
        """Create search space for instruction/demo combinations."""
        param_specs = {}

        # Instruction selection (categorical)
        if self.instruction_candidates:
            param_specs["instruction_idx"] = ParamSpec(
                name="instruction_idx",
                param_type=ParamType.CATEGORICAL,
                domain=ParamDomain.BEHAVIOR,
                description="Instruction index",
                default=0,
                choices=list(range(len(self.instruction_candidates))),
            )

        # Demo set selection (categorical)
        if self.demo_candidates:
            param_specs["demo_set_idx"] = ParamSpec(
                name="demo_set_idx",
                param_type=ParamType.CATEGORICAL,
                domain=ParamDomain.BEHAVIOR,
                description="Demonstration set index",
                default=0,
                choices=list(range(len(self.demo_candidates))),
            )

        # Number of demos to use (integer)
        param_specs["num_demos"] = ParamSpec(
            name="num_demos",
            param_type=ParamType.INT,
            domain=ParamDomain.BEHAVIOR,
            description="Number of demonstrations",
            default=self.config.max_bootstrapped_demos,
            range=(0, self.config.max_bootstrapped_demos),
            step=1,
        )

        return SearchSpace(param_specs)

    async def _apply_configuration(self, module: Module, config: dict[str, Any]) -> Module:
        """Apply a configuration to create a trial module."""
        # Deep copy module
        trial_module = module.deepcopy()

        # Apply instruction
        if "instruction_idx" in config and self.instruction_candidates:
            idx = config["instruction_idx"]
            instruction = self.instruction_candidates[idx].instruction

            # Set instruction on signature
            if hasattr(trial_module, "signature"):
                trial_module.signature.instructions = instruction

        # Apply demonstrations
        if "demo_set_idx" in config and self.demo_candidates:
            demo_idx = config["demo_set_idx"]
            num_demos = config.get("num_demos", self.config.max_bootstrapped_demos)

            demo_set = self.demo_candidates[demo_idx][:num_demos]

            # Add demos to module
            if hasattr(trial_module, "demo_manager"):
                demo_manager = trial_module.demo_manager
                demo_manager.clear()  # type: ignore[attr-defined]
                for demo in demo_set:
                    demo_manager.add(  # type: ignore[attr-defined]
                        {
                            "inputs": demo.inputs,
                            "outputs": demo.outputs,
                        }
                    )

        return trial_module

    async def _evaluate_module(self, module: Module, dataset: list[dict[str, Any]]) -> float:
        """Evaluate module on dataset."""
        scores = []

        for example in dataset:
            try:
                # Run module
                prediction = await module.forward(**example["inputs"])

                if prediction.success:
                    # Evaluate with metric
                    score = self.metric(prediction.outputs, example["outputs"])
                    scores.append(score)
                else:
                    scores.append(0.0)
            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0
