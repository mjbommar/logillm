"""Configuration classes for optimizers."""

from dataclasses import dataclass, field


@dataclass
class HybridOptimizerConfig:
    """Configuration for HybridOptimizer."""

    # Strategy configuration
    num_iterations: int = 3
    n_trials: int = 50
    balance_weight: float = 0.5
    convergence_threshold: float = 0.001

    # Joint optimization
    n_warmup_joint: int = 10
    demo_subset_size: int = 20  # For bootstrap efficiency
    max_demo_subset: bool = True  # Whether to limit demo generation dataset

    # Instruction styles for joint optimization
    instruction_styles: list[str] = field(
        default_factory=lambda: ["concise", "detailed", "step_by_step", "formal"]
    )

    # Search space ranges
    num_demos_range: tuple[int, int] = (0, 8)

    # Logging
    enable_logging: bool = True
    log_level: str = "WARNING"


@dataclass
class ReflectiveEvolutionConfig:
    """Configuration for ReflectiveEvolutionOptimizer."""

    # Core settings
    n_iterations: int = 10
    minibatch_size: int = 5
    pareto_size_limit: int = 10

    # Reflection settings
    max_traces_for_reflection: int = 3  # Limit context size
    demo_generation_subset: int = 20  # For efficiency

    # Heuristic thresholds
    poor_performance_threshold: float = 0.3
    moderate_performance_threshold: float = 0.5
    high_variance_threshold: float = 0.1

    # Temperature adjustments
    temp_adjustment_poor: float = -0.2
    temp_adjustment_moderate: float = -0.1
    temp_adjustment_variance: float = -0.15
    top_p_adjustment_variance: float = -0.1

    # Demo adjustments
    demo_increment_poor: int = 2

    # Candidate generation
    n_candidate_random: int = 50  # For acquisition function
    n_candidate_local: int = 50
    top_k_for_local_search: int = 5
    mutation_noise: float = 0.2

    # Merging
    max_candidates_to_merge: int = 3
    max_demos_per_candidate: int = 2
    max_total_demos_merged: int = 6

    # Temperature bounds
    temp_min: float = 0.0
    temp_max: float = 2.0
    top_p_min: float = 0.1
    top_p_max: float = 1.0

    # Pareto frontier
    exploration_base_weight: float = 0.1  # Base weight for non-dominated solutions

    # Logging
    enable_logging: bool = True
    log_level: str = "DEBUG"


@dataclass
class MultiObjectiveConfig:
    """Configuration for MultiObjectiveOptimizer."""

    # Core settings
    n_trials: int = 50
    pareto_size: int = 20

    # Evaluation settings
    latency_sample_size: int = 10  # Examples to test for latency
    cost_sample_size: int = 10  # Examples to estimate cost
    consistency_repeats: int = 3  # Times to run for consistency check

    # Default penalties
    latency_failure_penalty: float = 10.0  # Seconds
    cost_failure_tokens: int = 100

    # Cost estimation
    cost_per_1k_tokens: float = 0.01  # Default pricing

    # Search space defaults
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.1, 1.0)
    max_tokens_range: tuple[int, int] = (50, 500)
    max_tokens_default: int = 150
    num_demos_range: tuple[int, int] = (0, 8)
    num_demos_default: int = 3

    # Search strategy
    n_warmup_bayesian: int = 10

    # Logging
    enable_logging: bool = True
    log_level: str = "DEBUG"


def get_default_config(optimizer_type: str) -> object:
    """Get default configuration for an optimizer type."""
    configs = {
        "hybrid": HybridOptimizerConfig,
        "reflective": ReflectiveEvolutionConfig,
        "multi_objective": MultiObjectiveConfig,
    }

    config_class = configs.get(optimizer_type)
    if config_class:
        return config_class()

    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


__all__ = [
    "HybridOptimizerConfig",
    "ReflectiveEvolutionConfig",
    "MultiObjectiveConfig",
    "get_default_config",
]
