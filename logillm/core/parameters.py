"""Parameter specification and management system for LogiLLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from .types import Configuration, Metadata

T = TypeVar("T")


class ParamType(Enum):
    """Types of parameters supported."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    CATEGORICAL = "categorical"
    LIST = "list"
    DICT = "dict"


class ParamDomain(Enum):
    """Domain of parameter influence."""

    GENERATION = "generation"  # Affects text generation (temperature, top_p)
    BEHAVIOR = "behavior"  # Affects model behavior (reasoning_effort)
    EFFICIENCY = "efficiency"  # Affects performance (max_tokens, timeout)
    QUALITY = "quality"  # Affects output quality (best_of, n)
    SAFETY = "safety"  # Affects safety (moderation, filters)


class ParamPreset(Enum):
    """Common parameter presets."""

    CREATIVE = "creative"  # High temperature, high top_p
    BALANCED = "balanced"  # Medium settings
    PRECISE = "precise"  # Low temperature, deterministic
    FAST = "fast"  # Optimize for speed
    QUALITY = "quality"  # Optimize for quality
    CHEAP = "cheap"  # Optimize for cost


@dataclass
class ParamConstraint:
    """Constraint on a parameter value."""

    type: str  # "range", "choices", "pattern", "custom"
    value: Any  # The constraint value
    validator: Callable[[Any], bool] | None = None

    def validate(self, value: Any) -> bool:
        """Check if value satisfies constraint."""
        if self.type == "range":
            min_val, max_val = self.value
            return min_val <= value <= max_val
        elif self.type == "choices":
            return value in self.value
        elif self.type == "pattern":
            import re

            return bool(re.match(self.value, str(value)))
        elif self.type == "custom" and self.validator:
            return self.validator(value)
        return True


@dataclass
class ParamSpec:
    """Specification for a parameter."""

    name: str
    param_type: ParamType
    domain: ParamDomain
    description: str
    default: Any
    constraints: list[ParamConstraint] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    # For numeric types
    range: tuple[float, float] | None = None
    step: float | None = None

    # For categorical types
    choices: list[Any] | None = None

    # For structured types
    schema: dict[str, Any] | None = None

    # Dependencies on other parameters
    depends_on: dict[str, Any] | None = None

    # Model/provider specific
    supported_models: list[str] | None = None
    provider_specific: bool = False

    def validate(self, value: Any) -> bool:
        """Validate a value against this spec."""
        # Type checking
        if self.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.range and not (self.range[0] <= value <= self.range[1]):
                return False

        elif self.param_type == ParamType.INT:
            if not isinstance(value, int):
                return False
            if self.range and not (self.range[0] <= value <= self.range[1]):
                return False

        elif self.param_type == ParamType.BOOL:
            if not isinstance(value, bool):
                return False

        elif self.param_type == ParamType.CATEGORICAL:
            if self.choices and value not in self.choices:
                return False

        # Check additional constraints
        for constraint in self.constraints:
            if not constraint.validate(value):
                return False

        return True

    def sample(self, rng=None) -> Any:
        """Sample a random valid value."""
        import random

        rng = rng or random.Random()

        if self.param_type == ParamType.FLOAT and self.range:
            return rng.uniform(self.range[0], self.range[1])
        elif self.param_type == ParamType.INT and self.range:
            return rng.randint(int(self.range[0]), int(self.range[1]))
        elif self.param_type == ParamType.BOOL:
            return rng.choice([True, False])
        elif self.param_type == ParamType.CATEGORICAL and self.choices:
            return rng.choice(self.choices)
        else:
            return self.default


@dataclass
class ParamGroup:
    """Group of related parameters."""

    name: str
    description: str
    params: list[ParamSpec]
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "params": [p.name for p in self.params],
            "metadata": self.metadata,
        }


class ParameterProvider(ABC):
    """Protocol for parameter-aware providers."""

    @abstractmethod
    def get_param_specs(self) -> dict[str, ParamSpec]:
        """Get specifications for all supported parameters."""
        ...

    @abstractmethod
    def get_param_groups(self) -> list[ParamGroup]:
        """Get parameter groups for organization."""
        ...

    @abstractmethod
    def get_param_presets(self) -> dict[ParamPreset, Configuration]:
        """Get preset configurations."""
        ...

    @abstractmethod
    def validate_params(self, params: Configuration) -> tuple[bool, list[str]]:
        """Validate parameter configuration.

        Returns:
            (is_valid, list_of_errors)
        """
        ...

    def get_param_value(self, name: str) -> Any:
        """Get current value of a parameter."""
        return self.config.get(name)

    def set_param_value(self, name: str, value: Any) -> None:
        """Set value of a parameter."""
        spec = self.get_param_specs().get(name)
        if spec and not spec.validate(value):
            raise ValueError(f"Invalid value {value} for parameter {name}")
        self.config[name] = value

    def apply_preset(self, preset: ParamPreset) -> None:
        """Apply a parameter preset."""
        presets = self.get_param_presets()
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        self.config.update(presets[preset])


class SearchSpace:
    """Defines the hyperparameter search space."""

    def __init__(self, param_specs: dict[str, ParamSpec]):
        """Initialize search space.

        Args:
            param_specs: Parameter specifications to search over
        """
        self.param_specs = param_specs
        self.fixed_params: dict[str, Any] = {}
        self.conditional_params: dict[str, Callable[[dict], bool]] = {}

    def fix_param(self, name: str, value: Any) -> None:
        """Fix a parameter to a specific value."""
        if name not in self.param_specs:
            raise ValueError(f"Unknown parameter: {name}")
        self.fixed_params[name] = value

    def add_conditional(self, name: str, condition: Callable[[dict], bool]) -> None:
        """Add conditional inclusion of a parameter.

        Args:
            name: Parameter name
            condition: Function that takes current config and returns whether to include
        """
        self.conditional_params[name] = condition

    def sample(self, rng=None) -> Configuration:
        """Sample a configuration from the search space."""
        config = {}

        # Add fixed parameters
        config.update(self.fixed_params)

        # Sample other parameters
        for name, spec in self.param_specs.items():
            if name in self.fixed_params:
                continue

            # Check conditional inclusion
            if name in self.conditional_params:
                if not self.conditional_params[name](config):
                    continue

            # Check dependencies
            if spec.depends_on:
                skip = False
                for dep_name, dep_value in spec.depends_on.items():
                    if config.get(dep_name) != dep_value:
                        skip = True
                        break
                if skip:
                    continue

            config[name] = spec.sample(rng)

        return config

    def to_optuna_search_space(self, trial):
        """Convert to Optuna search space.

        Args:
            trial: Optuna trial object

        Returns:
            Configuration sampled from trial
        """
        config = {}

        # Add fixed parameters
        config.update(self.fixed_params)

        for name, spec in self.param_specs.items():
            if name in self.fixed_params:
                continue

            # Check conditional inclusion
            if name in self.conditional_params:
                if not self.conditional_params[name](config):
                    continue

            # Sample based on type
            if spec.param_type == ParamType.FLOAT and spec.range:
                if spec.step:
                    value = trial.suggest_float(name, spec.range[0], spec.range[1], step=spec.step)
                else:
                    value = trial.suggest_float(name, spec.range[0], spec.range[1])
            elif spec.param_type == ParamType.INT and spec.range:
                if spec.step:
                    value = trial.suggest_int(
                        name, int(spec.range[0]), int(spec.range[1]), step=int(spec.step)
                    )
                else:
                    value = trial.suggest_int(name, int(spec.range[0]), int(spec.range[1]))
            elif spec.param_type == ParamType.BOOL:
                value = trial.suggest_categorical(name, [True, False])
            elif spec.param_type == ParamType.CATEGORICAL and spec.choices:
                value = trial.suggest_categorical(name, spec.choices)
            else:
                value = spec.default

            config[name] = value

        return config


@dataclass
class ParameterTrace:
    """Tracks parameter values used during execution."""

    module_name: str
    parameters: Configuration
    score: float
    timestamp: float
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module": self.module_name,
            "parameters": self.parameters,
            "score": self.score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ParameterHistory:
    """Tracks optimization history of parameters."""

    def __init__(self):
        """Initialize parameter history."""
        self.traces: list[ParameterTrace] = []
        self.best_config: Configuration | None = None
        self.best_score: float = float("-inf")

    def add_trace(self, trace: ParameterTrace) -> None:
        """Add a parameter trace."""
        self.traces.append(trace)
        if trace.score > self.best_score:
            self.best_score = trace.score
            self.best_config = trace.parameters.copy()

    def get_trajectory(self, param_name: str) -> list[tuple[float, Any]]:
        """Get optimization trajectory for a parameter.

        Returns:
            List of (timestamp, value) tuples
        """
        trajectory = []
        for trace in self.traces:
            if param_name in trace.parameters:
                trajectory.append((trace.timestamp, trace.parameters[param_name]))
        return trajectory

    def get_correlation(self, param_name: str) -> float:
        """Get correlation between parameter and score.

        Returns:
            Correlation coefficient
        """
        if not self.traces:
            return 0.0

        values = []
        scores = []
        for trace in self.traces:
            if param_name in trace.parameters:
                # Try to convert to float for correlation
                try:
                    val = float(trace.parameters[param_name])
                    values.append(val)
                    scores.append(trace.score)
                except (TypeError, ValueError):
                    # Skip non-numeric values
                    continue

        if len(values) < 2:
            return 0.0

        # Calculate Pearson correlation coefficient manually
        try:
            n = len(values)

            # Calculate means
            mean_x = sum(values) / n
            mean_y = sum(scores) / n

            # Calculate correlation components
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, scores))

            # Calculate standard deviations
            sum_sq_diff_x = sum((x - mean_x) ** 2 for x in values)
            sum_sq_diff_y = sum((y - mean_y) ** 2 for y in scores)

            denominator = (sum_sq_diff_x * sum_sq_diff_y) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator
        except Exception:
            return 0.0


# Standard parameter specifications for common LLM parameters

STANDARD_PARAM_SPECS = {
    "temperature": ParamSpec(
        name="temperature",
        param_type=ParamType.FLOAT,
        domain=ParamDomain.GENERATION,
        description="Controls randomness in generation",
        default=0.7,
        range=(0.0, 2.0),
        step=0.1,
    ),
    "top_p": ParamSpec(
        name="top_p",
        param_type=ParamType.FLOAT,
        domain=ParamDomain.GENERATION,
        description="Nucleus sampling threshold",
        default=1.0,
        range=(0.0, 1.0),
        step=0.05,
    ),
    "top_k": ParamSpec(
        name="top_k",
        param_type=ParamType.INT,
        domain=ParamDomain.GENERATION,
        description="Top-k sampling",
        default=0,
        range=(0, 100),
        step=10,
    ),
    "max_tokens": ParamSpec(
        name="max_tokens",
        param_type=ParamType.INT,
        domain=ParamDomain.EFFICIENCY,
        description="Maximum tokens to generate",
        default=1000,
        range=(1, 32000),
        step=100,
    ),
    "presence_penalty": ParamSpec(
        name="presence_penalty",
        param_type=ParamType.FLOAT,
        domain=ParamDomain.GENERATION,
        description="Penalty for token presence",
        default=0.0,
        range=(-2.0, 2.0),
        step=0.1,
        provider_specific=True,
    ),
    "frequency_penalty": ParamSpec(
        name="frequency_penalty",
        param_type=ParamType.FLOAT,
        domain=ParamDomain.GENERATION,
        description="Penalty for token frequency",
        default=0.0,
        range=(-2.0, 2.0),
        step=0.1,
        provider_specific=True,
    ),
}


# Standard presets
STANDARD_PRESETS = {
    ParamPreset.CREATIVE: {
        "temperature": 0.9,
        "top_p": 0.95,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    },
    ParamPreset.BALANCED: {
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    ParamPreset.PRECISE: {
        "temperature": 0.1,
        "top_p": 0.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
}


__all__ = [
    "ParamType",
    "ParamDomain",
    "ParamPreset",
    "ParamConstraint",
    "ParamSpec",
    "ParamGroup",
    "ParameterProvider",
    "SearchSpace",
    "ParameterTrace",
    "ParameterHistory",
    "STANDARD_PARAM_SPECS",
    "STANDARD_PRESETS",
]
