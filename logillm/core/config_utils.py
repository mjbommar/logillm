"""Utilities for consistent configuration handling across LogiLLM.

This module addresses the systemic brittleness in parameter handling by providing
consistent functions for getting and setting configuration values, regardless of
whether the config is a dict, dataclass, or other object.
"""

from typing import Any


def set_config_value(obj: Any, key: str, value: Any) -> None:
    """Set a configuration value on an object, handling different config types.

    Args:
        obj: Object with a config (Module, Provider, etc.)
        key: Configuration key (e.g., "temperature", "top_p")
        value: Value to set
    """
    if not hasattr(obj, "config"):
        return

    config = obj.config

    if isinstance(config, dict):
        # Config is a dict - use dictionary access
        config[key] = value
    elif hasattr(config, "__setattr__"):
        # Config is an object - try to set attribute
        try:
            setattr(config, key, value)
        except (AttributeError, TypeError):
            # If setting attribute fails, try dict-like access
            if hasattr(config, "__setitem__"):
                config[key] = value


def get_config_value(obj: Any, key: str, default: Any = None) -> Any:
    """Get a configuration value from an object, handling different config types.

    Args:
        obj: Object with a config (Module, Provider, etc.)
        key: Configuration key (e.g., "temperature", "top_p")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    if not hasattr(obj, "config"):
        return default

    config = obj.config

    if isinstance(config, dict):
        # Config is a dict - use get method
        return config.get(key, default)
    elif hasattr(config, key):
        # Config is an object with the attribute
        return getattr(config, key, default)
    elif hasattr(config, "__getitem__"):
        # Config supports dict-like access
        try:
            return config[key]
        except (KeyError, TypeError):
            return default
    else:
        return default


def update_config(obj: Any, updates: dict[str, Any]) -> None:
    """Update multiple configuration values on an object.

    Args:
        obj: Object with a config (Module, Provider, etc.)
        updates: Dictionary of configuration updates
    """
    if not hasattr(obj, "config"):
        return

    config = obj.config

    if isinstance(config, dict):
        # Config is a dict - use update method
        config.update(updates)
    else:
        # Config is an object - set each attribute
        for key, value in updates.items():
            set_config_value(obj, key, value)


def copy_config(obj: Any) -> dict[str, Any]:
    """Copy configuration from an object to a dict.

    Args:
        obj: Object with a config

    Returns:
        Dictionary copy of configuration
    """
    if not hasattr(obj, "config"):
        return {}

    config = obj.config

    if isinstance(config, dict):
        return config.copy()
    elif hasattr(config, "__dict__"):
        # Config is an object - copy its __dict__
        return config.__dict__.copy()
    elif hasattr(config, "to_dict"):
        # Config has a to_dict method
        return config.to_dict()
    else:
        # Try to extract what we can
        result = {}
        for attr in dir(config):
            if not attr.startswith("_"):
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        result[attr] = value
                except Exception:
                    pass
        return result


# Common hyperparameter keys for reference
HYPERPARAMETER_KEYS = [
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "stop",
    "top_k",
]


def set_hyperparameter(obj: Any, param: str, value: Any) -> None:
    """Set a hyperparameter on both the object and its provider if present.

    Args:
        obj: Module or other object
        param: Hyperparameter name
        value: Value to set
    """
    # Set on the object's config
    set_config_value(obj, param, value)

    # Also set on provider if present
    if hasattr(obj, "provider") and obj.provider:
        set_config_value(obj.provider, param, value)


def get_hyperparameter(obj: Any, param: str, default: Any = None) -> Any:
    """Get a hyperparameter from object or its provider.

    Args:
        obj: Module or other object
        param: Hyperparameter name
        default: Default value

    Returns:
        Hyperparameter value or default
    """
    # Try object's config first
    value = get_config_value(obj, param, None)
    if value is not None:
        return value

    # Try provider's config
    if hasattr(obj, "provider") and obj.provider:
        value = get_config_value(obj.provider, param, None)
        if value is not None:
            return value

    return default


__all__ = [
    "set_config_value",
    "get_config_value",
    "update_config",
    "copy_config",
    "set_hyperparameter",
    "get_hyperparameter",
    "HYPERPARAMETER_KEYS",
]
