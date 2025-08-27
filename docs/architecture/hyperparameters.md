# Hyperparameter Configuration System

## Overview

LogiLLM provides a robust hyperparameter configuration system that ensures safe, consistent parameter handling across all modules and optimizers. This system addresses the brittleness commonly found in LLM frameworks where direct dictionary manipulation can lead to runtime errors.

## Key Components

### HyperparameterConfig Class

The `HyperparameterConfig` class is the foundation of our parameter management system, providing:

- **Automatic validation and clamping** to valid ranges
- **Type coercion** for flexible input handling
- **Both dict-like and attribute access** patterns
- **Change tracking** for optimization debugging
- **Provider-specific parameter conversion**

```python
from logillm.core.hyperparameters import HyperparameterConfig

# Create a config with initial values
config = HyperparameterConfig({
    "temperature": 0.8,
    "top_p": 0.95
})

# Dict-like access
config["temperature"] = 1.2

# Attribute access
config.temperature = 1.2

# Automatic validation (values are clamped to valid ranges)
config["temperature"] = 3.0  # Automatically clamped to 2.0
config["top_p"] = -0.5      # Automatically clamped to 0.0

# Type coercion
config["max_tokens"] = "150"  # Automatically converted to int(150)
```

## Supported Hyperparameters

Based on analysis of OpenAI and Anthropic SDKs, LogiLLM supports:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temperature` | 0.0 - 2.0 | 0.7 | Controls randomness in generation |
| `top_p` | 0.0 - 1.0 | 1.0 | Nucleus sampling threshold |
| `top_k` | 1+ | None | Limits token choices (Anthropic) |
| `max_tokens` | 1 - 32768 | 150 | Maximum tokens to generate |
| `frequency_penalty` | -2.0 - 2.0 | 0.0 | Penalizes frequent tokens (OpenAI) |
| `presence_penalty` | -2.0 - 2.0 | 0.0 | Penalizes present tokens (OpenAI) |
| `seed` | 0+ | None | For reproducible generation |
| `n` | 1 - 100 | 1 | Number of completions |
| `stop` | string/list | None | Stop sequences |

## Configuration Utilities

LogiLLM provides utility functions for consistent configuration handling:

```python
from logillm.core.config_utils import (
    set_hyperparameter,
    get_hyperparameter,
    ensure_config,
    update_config
)

# Ensure a module has proper configuration
ensure_config(module)  # Creates HyperparameterConfig if needed

# Set a single hyperparameter
set_hyperparameter(module, "temperature", 0.9)

# Get a hyperparameter with default fallback
temp = get_hyperparameter(module, "temperature", default=0.7)

# Update multiple parameters at once
update_config(module, {
    "temperature": 1.0,
    "top_p": 0.9,
    "max_tokens": 200
})
```

## Optimizer Integration

All LogiLLM optimizers use the hyperparameter system for safe parameter updates:

```python
# ReflectiveEvolution optimizer automatically handles hyperparameters
optimizer = ReflectiveEvolutionOptimizer(
    metric=accuracy_metric,
    include_hyperparameters=True  # Enable hyperparameter optimization
)

# HybridOptimizer optimizes both prompts and hyperparameters
optimizer = HybridOptimizer(
    metric=accuracy_metric,
    strategy="alternating"
)
```

### Protected Updates in Optimizers

Optimizers use the configuration utilities to safely update parameters:

```python
# Example from ReflectiveEvolutionOptimizer
async def _apply_improvements(self, module, improvements, dataset):
    improved = copy.deepcopy(module)
    
    # Ensure module has proper config
    ensure_config(improved)
    
    # Apply hyperparameter improvements safely
    if improvements.get("temperature") is not None:
        current_temp = get_hyperparameter(improved, "temperature", 0.7)
        new_temp = max(0.0, min(2.0, current_temp + improvements["temperature"]))
        set_hyperparameter(improved, "temperature", new_temp)
```

## Provider-Specific Conversion

The system automatically converts parameters to provider-specific names:

```python
config = HyperparameterConfig({
    "temperature": 0.7,
    "max_tokens": 100,
    "stop": ["\n\n"]
})

# Convert for OpenAI (uses max_completion_tokens for newer models)
openai_params = config.to_provider_params("openai")
# Result: {"temperature": 0.7, "max_completion_tokens": 100, "stop": ["\n\n"]}

# Convert for Anthropic (uses stop_sequences)
anthropic_params = config.to_provider_params("anthropic")
# Result: {"temperature": 0.7, "max_tokens": 100, "stop_sequences": ["\n\n"]}
```

## Advanced Features

### Change Tracking

Track parameter changes for debugging and optimization analysis:

```python
config = HyperparameterConfig()
config["temperature"] = 0.8
config["temperature"] = 0.9

# Get history of changes
changes = config.get_changes()
# [{"param": "temperature", "old": 0.7, "new": 0.8, "timestamp": "..."},
#  {"param": "temperature", "old": 0.8, "new": 0.9, "timestamp": "..."}]
```

### Parameter Interpolation

Create interpolated configurations for optimization:

```python
config1 = HyperparameterConfig({"temperature": 0.5, "top_p": 0.8})
config2 = HyperparameterConfig({"temperature": 1.5, "top_p": 1.0})

# Interpolate halfway between configs
interpolated = config1.interpolate(config2, weight=0.5)
# Result: temperature=1.0, top_p=0.9
```

### Parameter Adjustment

Incrementally adjust parameters during optimization:

```python
config = HyperparameterConfig({"temperature": 0.7})
config.adjust("temperature", 0.2)  # Adds 0.2 to current value
# Result: temperature=0.9
```

## Benefits

1. **Safety**: Automatic validation prevents invalid parameter values
2. **Consistency**: Unified interface across all modules and optimizers
3. **Flexibility**: Supports multiple access patterns (dict, attribute)
4. **Debugging**: Change tracking helps understand optimization behavior
5. **Compatibility**: Automatic conversion for different LLM providers
6. **Type Safety**: Automatic type coercion with validation

## Migration Guide

If you have existing code using direct dictionary manipulation:

```python
# Old approach (brittle)
module.config = {}
module.config["temperature"] = 0.8  # May fail if config isn't a dict

# New approach (safe)
from logillm.core.config_utils import ensure_config, set_hyperparameter

ensure_config(module)  # Ensures proper HyperparameterConfig
set_hyperparameter(module, "temperature", 0.8)  # Safe, validated update
```

## Best Practices

1. **Always use config utilities** instead of direct manipulation
2. **Call ensure_config()** before working with a module's configuration
3. **Use get_hyperparameter()** with defaults for safe parameter access
4. **Let the system handle validation** - don't manually clamp values
5. **Track changes** when debugging optimization behavior

## Implementation Details

The hyperparameter system uses Python's dunder methods (`__getitem__`, `__setitem__`, `__getattr__`, `__setattr__`) to intercept all configuration access and ensure validation. This provides a transparent protective layer that works with existing code patterns while preventing common errors.

See also:
- [Optimization Overview](../optimization/overview.md) for how hyperparameters integrate with optimizers
- [Module Documentation](../core-concepts/modules.md) for module configuration patterns
- [Provider Documentation](../core-concepts/providers.md) for provider-specific parameters