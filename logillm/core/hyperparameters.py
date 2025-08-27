"""Protected hyperparameter configuration system with validation.

This module provides a protective wrapper for hyperparameter configuration
using dunder methods to ensure consistent and safe parameter handling across
all optimizers and modules.
"""

from typing import Any, Optional, Union

# Define hyperparameter metadata with valid ranges and defaults
HYPERPARAMETER_DEFINITIONS: dict[str, dict[str, Any]] = {
    # Temperature controls randomness (0=deterministic, higher=more random)
    "temperature": {
        "default": 0.7,
        "min": 0.0,
        "max": 2.0,  # OpenAI supports up to 2.0, Anthropic up to 1.0
        "type": float,
        "description": "Controls randomness in generation",
    },
    # Top-p controls nucleus sampling
    "top_p": {
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "type": float,
        "description": "Nucleus sampling threshold",
    },
    # Top-k for limiting token choices (Anthropic-specific)
    "top_k": {
        "default": None,
        "min": 1,
        "max": None,
        "type": int,
        "description": "Limit to top K token choices",
    },
    # Maximum tokens to generate
    "max_tokens": {
        "default": 150,
        "min": 1,
        "max": 32768,  # Common max across providers
        "type": int,
        "description": "Maximum tokens to generate",
    },
    # Frequency penalty (OpenAI-specific)
    "frequency_penalty": {
        "default": 0.0,
        "min": -2.0,
        "max": 2.0,
        "type": float,
        "description": "Penalize frequent tokens",
    },
    # Presence penalty (OpenAI-specific)
    "presence_penalty": {
        "default": 0.0,
        "min": -2.0,
        "max": 2.0,
        "type": float,
        "description": "Penalize tokens based on presence",
    },
    # Seed for reproducibility
    "seed": {
        "default": None,
        "min": 0,
        "max": None,
        "type": int,
        "description": "Seed for reproducible generation",
    },
    # Number of completions to generate
    "n": {
        "default": 1,
        "min": 1,
        "max": 100,
        "type": int,
        "description": "Number of completions to generate",
    },
    # Stop sequences
    "stop": {
        "default": None,
        "min": None,
        "max": None,
        "type": (str, list),
        "description": "Stop sequences for generation",
    },
}


class HyperparameterConfig:
    """Protected configuration wrapper with validation and dunder methods.

    This class ensures safe and consistent hyperparameter handling by:
    1. Validating parameter values against defined ranges
    2. Providing both dict-like and attribute access
    3. Tracking parameter changes for optimization
    4. Supporting provider-specific parameter mapping
    """

    def __init__(self, initial: Optional[dict[str, Any]] = None, strict: bool = False):
        """Initialize hyperparameter configuration.

        Args:
            initial: Initial configuration values
            strict: Whether to enforce strict validation (raise on invalid params)
        """
        self._params: dict[str, Any] = {}
        self._strict = strict
        self._change_history: list[dict[str, Any]] = []  # Track changes for optimization debugging

        # Initialize with defaults
        for param, spec in HYPERPARAMETER_DEFINITIONS.items():
            if spec["default"] is not None:
                self._params[param] = spec["default"]

        # Apply initial values if provided
        if initial:
            for key, value in initial.items():
                self[key] = value

    def _validate_value(self, key: str, value: Any) -> tuple[bool, Any]:
        """Validate and potentially coerce a hyperparameter value.

        Args:
            key: Parameter name
            value: Value to validate

        Returns:
            Tuple of (is_valid, coerced_value)
        """
        if key not in HYPERPARAMETER_DEFINITIONS:
            # Unknown parameter - allow in non-strict mode
            return not self._strict, value

        spec = HYPERPARAMETER_DEFINITIONS[key]

        # Skip validation for None values (using defaults)
        if value is None:
            return True, spec["default"]

        # Type checking and coercion
        expected_type = spec["type"]
        if expected_type and not isinstance(value, expected_type):
            # Try to coerce
            try:
                if expected_type is float:
                    value = float(value)
                elif expected_type is int:
                    value = int(value)
                elif expected_type is str:
                    value = str(value)
            except (ValueError, TypeError):
                return False, spec["default"]

        # Range validation
        if spec.get("min") is not None and value < spec["min"]:
            value = spec["min"]
        if spec.get("max") is not None and value > spec["max"]:
            value = spec["max"]

        return True, value

    # Dict-like interface with dunder methods
    def __getitem__(self, key: str) -> Any:
        """Get parameter value using dict-like access."""
        return self._params.get(key, HYPERPARAMETER_DEFINITIONS.get(key, {}).get("default"))

    def __setitem__(self, key: str, value: Any) -> None:
        """Set parameter value with validation."""
        valid, coerced = self._validate_value(key, value)
        if valid:
            old_value = self._params.get(key)
            self._params[key] = coerced
            self._change_history.append(
                {"param": key, "old": old_value, "new": coerced, "timestamp": _get_timestamp()}
            )
        elif self._strict:
            raise ValueError(f"Invalid value {value} for parameter {key}")

    def __delitem__(self, key: str) -> None:
        """Remove a parameter (reset to default)."""
        if key in self._params:
            del self._params[key]

    def __contains__(self, key: str) -> bool:
        """Check if parameter is set."""
        return key in self._params

    def __len__(self) -> int:
        """Return number of set parameters."""
        return len(self._params)

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self._params)

    # Attribute-like interface
    def __getattr__(self, name: str) -> Any:
        """Get parameter value using attribute access."""
        if name.startswith("_"):
            # Internal attributes
            return object.__getattribute__(self, name)
        return self.__getitem__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set parameter value using attribute access."""
        if name.startswith("_"):
            # Internal attributes
            object.__setattr__(self, name, value)
        else:
            self.__setitem__(name, value)

    # Dictionary compatibility methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter with default fallback."""
        return self._params.get(key, default)

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple parameters at once."""
        for key, value in updates.items():
            self[key] = value

    def keys(self):
        """Return parameter names."""
        return self._params.keys()

    def values(self):
        """Return parameter values."""
        return self._params.values()

    def items(self):
        """Return parameter items."""
        return self._params.items()

    def copy(self) -> dict[str, Any]:
        """Return a copy of parameters as dict."""
        return self._params.copy()

    # Advanced methods for optimization
    def adjust(self, param: str, delta: float) -> None:
        """Adjust a parameter by a delta value (for optimization).

        Args:
            param: Parameter name
            delta: Amount to adjust by
        """
        current = self.get(param)
        if current is not None:
            self[param] = current + delta

    def interpolate(
        self, other: "HyperparameterConfig", weight: float = 0.5
    ) -> "HyperparameterConfig":
        """Create interpolated configuration between this and another.

        Args:
            other: Other configuration
            weight: Weight for other config (0=this, 1=other)

        Returns:
            New interpolated configuration
        """
        result = HyperparameterConfig()

        # Get all parameters from both configs
        all_params = set(self._params.keys()) | set(other._params.keys())

        for param in all_params:
            val1 = self.get(param)
            val2 = other.get(param)

            if val1 is None:
                result[param] = val2
            elif val2 is None:
                result[param] = val1
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric interpolation
                result[param] = val1 * (1 - weight) + val2 * weight
            else:
                # Non-numeric - use weighted selection
                result[param] = val2 if weight > 0.5 else val1

        return result

    def to_provider_params(self, provider_type: str) -> dict[str, Any]:
        """Convert to provider-specific parameter names.

        Args:
            provider_type: Provider type (openai, anthropic, etc.)

        Returns:
            Parameters with provider-specific naming
        """
        # Map common parameters to provider-specific names
        mapping = {
            "openai": {
                "max_tokens": "max_completion_tokens",  # For newer models
            },
            "anthropic": {
                "stop": "stop_sequences",
            },
        }

        result = self._params.copy()

        if provider_type in mapping:
            for old_name, new_name in mapping[provider_type].items():
                if old_name in result:
                    result[new_name] = result.pop(old_name)

        return result

    def get_changes(self) -> list[dict[str, Any]]:
        """Get history of parameter changes."""
        return self._change_history.copy()

    def __repr__(self) -> str:
        """String representation."""
        params = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"HyperparameterConfig({params})"


def _get_timestamp() -> str:
    """Get current timestamp for change tracking."""
    from datetime import datetime

    return datetime.now().isoformat()


def ensure_hyperparameter_config(
    config: Union[dict, HyperparameterConfig, None],
) -> HyperparameterConfig:
    """Ensure configuration is a HyperparameterConfig instance.

    Args:
        config: Configuration (dict, HyperparameterConfig, or None)

    Returns:
        HyperparameterConfig instance
    """
    if isinstance(config, HyperparameterConfig):
        return config
    elif isinstance(config, dict):
        return HyperparameterConfig(config)
    else:
        return HyperparameterConfig()


def merge_configs(*configs: Union[dict, HyperparameterConfig]) -> HyperparameterConfig:
    """Merge multiple configurations with later ones taking precedence.

    Args:
        *configs: Configurations to merge

    Returns:
        Merged HyperparameterConfig
    """
    result = HyperparameterConfig()

    for config in configs:
        if config:
            if isinstance(config, dict):
                result.update(config)
            elif isinstance(config, HyperparameterConfig):
                result.update(config.copy())

    return result


__all__ = [
    "HyperparameterConfig",
    "HYPERPARAMETER_DEFINITIONS",
    "ensure_hyperparameter_config",
    "merge_configs",
]
