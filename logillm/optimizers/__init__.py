"""Optimizers for LogiLLM framework."""

from ..core.optimizers import (
    BootstrapOptimizer,
    Metric,
    OptimizationConfig,
    Optimizer,
    RandomSearchOptimizer,
    Trace,
)
from .base import (
    Demonstration,
    DemoSelector,
    PromptOptimizationConfig,
    PromptOptimizer,
)
from .bootstrap_fewshot import BootstrapFewShot
from .copro import COPRO, COPROConfig, COPROStats, InstructionCandidate
from .demo_selectors import (
    BestDemoSelector,
    DiverseDemoSelector,
    RandomDemoSelector,
    StratifiedDemoSelector,
    create_demo_selector,
)
from .format_optimizer import FormatOptimizer
from .hybrid_optimizer import HybridOptimizer
from .hyperparameter import (
    AdaptiveOptimizer,
    GridSearchOptimizer,
    HyperparameterOptimizer,
)
from .instruction_optimizer import InstructionOptimizer
from .knn_fewshot import KNNFewShot, KNNFewShotConfig
from .labeled_fewshot import LabeledFewShot
from .miprov2 import MIPROv2Optimizer
from .multi_objective import MultiObjectiveOptimizer
from .random_prompt import RandomPromptOptimizer
from .reflective_evolution import ReflectiveEvolutionOptimizer
from .search_strategies import (
    AcquisitionType,
    GridSearchStrategy,
    LatinHypercubeStrategy,
    RandomSearchStrategy,
    SearchStrategy,
    SimpleBayesianStrategy,
    StrategyConfig,
    create_strategy,
)
from .signature_optimizer import SignatureOptimizer
from .simba import SIMBA, SIMBAConfig

# Aliases for backward compatibility
MultiObjective = MultiObjectiveOptimizer

__all__ = [
    # From core.optimizers
    "Optimizer",
    "BootstrapOptimizer",
    "RandomSearchOptimizer",
    "Metric",
    "OptimizationConfig",
    "Trace",
    # From hyperparameter
    "HyperparameterOptimizer",
    "GridSearchOptimizer",
    "AdaptiveOptimizer",
    # From search_strategies
    "SearchStrategy",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "SimpleBayesianStrategy",
    "LatinHypercubeStrategy",
    "StrategyConfig",
    "AcquisitionType",
    "create_strategy",
    # From base
    "PromptOptimizationConfig",
    "Demonstration",
    "DemoSelector",
    "PromptOptimizer",
    # From demo_selectors
    "BestDemoSelector",
    "RandomDemoSelector",
    "DiverseDemoSelector",
    "StratifiedDemoSelector",
    "create_demo_selector",
    # From bootstrap_fewshot
    "BootstrapFewShot",
    # From copro
    "COPRO",
    "COPROConfig",
    "COPROStats",
    "InstructionCandidate",
    # From instruction_optimizer
    "InstructionOptimizer",
    # From knn_fewshot
    "KNNFewShot",
    "KNNFewShotConfig",
    # From labeled_fewshot
    "LabeledFewShot",
    # From miprov2
    "MIPROv2Optimizer",
    # From random_prompt
    "RandomPromptOptimizer",
    # From hybrid_optimizer
    "HybridOptimizer",
    # From reflective_evolution
    "ReflectiveEvolutionOptimizer",
    # From multi_objective
    "MultiObjectiveOptimizer",
    "MultiObjective",
    # From format_optimizer
    "FormatOptimizer",
    # From simba
    "SIMBA",
    "SIMBAConfig",
    # From signature_optimizer
    "SignatureOptimizer",
]
