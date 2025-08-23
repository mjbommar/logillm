"""Core persistence functionality for LogiLLM modules.

This module provides robust save/load functionality for LogiLLM modules,
handling complex objects, provider configurations, and version compatibility.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from ..core.demos import Demo, DemoManager
from ..core.modules import Module
from ..core.signatures import BaseSignature
from ..core.types import SerializationFormat
from ..exceptions import LogiLLMError
from ..providers import ProviderError, create_provider, get_provider, register_provider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Module)

LOGILLM_VERSION = "0.1.0"


class PersistenceError(LogiLLMError):
    """Errors related to module persistence."""

    pass


class ModuleSaver:
    """Handles saving LogiLLM modules with full state preservation."""

    @classmethod
    def save_module(
        cls,
        module: Module,
        path: str,
        *,
        include_provider: bool = True,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> None:
        """Save a LogiLLM module to disk.

        Args:
            module: The module to save
            path: File path to save to
            include_provider: Whether to save provider configuration
            format: Serialization format (JSON, YAML, etc.)
        """
        if format != SerializationFormat.JSON:
            raise PersistenceError(f"Format {format} not yet supported")

        # Build comprehensive save data
        save_data = {
            "logillm_version": LOGILLM_VERSION,
            "save_timestamp": datetime.now().isoformat(),
            "module_type": module.__class__.__name__,
            "module_class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            # Core module data
            "signature": module.signature.to_dict() if module.signature else None,
            "config": module.config.copy(),
            "metadata": module.metadata.copy(),
            "state": module.state.value,
        }

        # Handle demo_manager with full fidelity
        if hasattr(module, "demo_manager") and module.demo_manager:
            save_data["demo_manager"] = cls._serialize_demo_manager(module.demo_manager)

        # Handle provider configuration (without secrets)
        if include_provider and hasattr(module, "provider") and module.provider:
            save_data["provider_config"] = cls._serialize_provider(module.provider)

        # Handle adapter configuration
        if hasattr(module, "adapter") and module.adapter:
            save_data["adapter_config"] = {
                "name": module.adapter.__class__.__name__,
                "config": getattr(module.adapter, "config", {}),
            }

        # Save to file
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(save_data, f, indent=2, default=cls._json_serializer)

        logger.info(f"Module {module.__class__.__name__} saved to {path}")

    @staticmethod
    def _serialize_demo_manager(demo_manager: DemoManager) -> dict[str, Any]:
        """Serialize demo manager with full state."""
        return {
            "demos": [demo.to_dict() for demo in demo_manager.demos],
            "teacher_demos": [demo.to_dict() for demo in demo_manager.teacher_demos],
            "max_demos": demo_manager.max_demos,
            "selection_strategy": demo_manager.selection_strategy,
        }

    @staticmethod
    def _serialize_provider(provider) -> dict[str, Any]:
        """Serialize provider configuration (without API keys)."""
        return {
            "name": provider.name,
            "model": provider.model,
            "config": {
                k: v
                for k, v in provider.config.items()
                if not k.lower().endswith("key") and not k.lower().endswith("token")
            },
            "provider_class": f"{provider.__class__.__module__}.{provider.__class__.__qualname__}",
        }

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "isoformat"):  # datetime
            return obj.isoformat()
        return str(obj)


class ModuleLoader:
    """Handles loading LogiLLM modules with full state restoration."""

    @classmethod
    def load_module(
        cls, path: str, *, setup_provider: bool = False, strict_version: bool = False
    ) -> Module:
        """Load a LogiLLM module from disk.

        Args:
            path: File path to load from
            setup_provider: Whether to automatically set up the saved provider
            strict_version: Whether to enforce exact version compatibility

        Returns:
            The reconstructed module
        """
        file_path = Path(path)
        if not file_path.exists():
            raise PersistenceError(f"File not found: {path}")

        with open(file_path) as f:
            save_data = json.load(f)

        # Version compatibility check
        saved_version = save_data.get("logillm_version", "unknown")
        if strict_version and saved_version != LOGILLM_VERSION:
            raise PersistenceError(
                f"Version mismatch: saved with {saved_version}, current is {LOGILLM_VERSION}"
            )
        elif saved_version != LOGILLM_VERSION:
            warnings.warn(
                f"Loading module saved with LogiLLM {saved_version}, current version is {LOGILLM_VERSION}. "
                f"This may cause compatibility issues.",
                stacklevel=2,
            )

        # Determine module type and create instance
        module_type = save_data["module_type"]
        module = cls._create_module_instance(module_type, save_data)

        # Restore demo manager
        if "demo_manager" in save_data:
            module.demo_manager = cls._deserialize_demo_manager(save_data["demo_manager"])

        # Handle provider setup
        if "provider_config" in save_data:
            provider_config = save_data["provider_config"]
            if setup_provider:
                cls._setup_provider(provider_config)
            else:
                # Just warn about provider mismatch
                try:
                    current_provider = get_provider()
                except ProviderError:
                    current_provider = None
                if current_provider and current_provider.name != provider_config["name"]:
                    warnings.warn(
                        f"Module was optimized with {provider_config['name']} ({provider_config['model']}), "
                        f"but current provider is {current_provider.name} ({current_provider.model}). "
                        f"Performance may differ.",
                        stacklevel=2,
                    )

        logger.info(f"Module {module_type} loaded from {path}")
        return module

    @staticmethod
    def _create_module_instance(module_type: str, save_data: dict[str, Any]) -> Module:
        """Create module instance from saved data."""
        # Reconstruct signature
        signature = None
        if save_data.get("signature"):
            signature = BaseSignature.from_dict(save_data["signature"])

        # For now, handle common module types
        if module_type == "Predict":
            from ..core.predict import Predict

            # Handle optional signature - Predict can work without one
            if signature is not None:
                # Cast BaseSignature to Signature for type compatibility
                module = Predict(signature=signature)  # type: ignore[arg-type]
            else:
                # Create a generic predict module if no signature saved
                module = Predict("input -> output")
        else:
            raise PersistenceError(f"Unsupported module type: {module_type}")

        # Restore configuration
        module.config.update(save_data.get("config", {}))
        module.metadata.update(save_data.get("metadata", {}))

        return module

    @staticmethod
    def _deserialize_demo_manager(demo_data: dict[str, Any]) -> DemoManager:
        """Deserialize demo manager from saved data."""
        demo_manager = DemoManager(
            max_demos=demo_data.get("max_demos", 5),
            selection_strategy=demo_data.get("selection_strategy", "best"),
        )

        # Restore regular demos
        for demo_dict in demo_data.get("demos", []):
            demo = Demo.from_dict(demo_dict)
            demo_manager.demos.append(demo)

        # Restore teacher demos
        for demo_dict in demo_data.get("teacher_demos", []):
            demo = Demo.from_dict(demo_dict)
            demo_manager.teacher_demos.append(demo)

        return demo_manager

    @staticmethod
    def _setup_provider(provider_config: dict[str, Any]) -> None:
        """Set up provider from saved configuration."""
        try:
            provider = create_provider(
                provider_config["name"],
                model=provider_config["model"],
                **provider_config.get("config", {}),
            )
            register_provider(provider, set_default=True)
            logger.info(
                f"Provider {provider_config['name']} ({provider_config['model']}) set up automatically"
            )
        except Exception as e:
            warnings.warn(f"Failed to set up saved provider: {e}", stacklevel=2)


# Add methods directly to Module class
def _save_method(self: Module, path: str, **kwargs) -> None:
    """Save this module to disk."""
    ModuleSaver.save_module(self, path, **kwargs)


def _load_classmethod(cls: type[T], path: str, **kwargs) -> T:
    """Load a module from disk."""
    return ModuleLoader.load_module(path, **kwargs)  # type: ignore


# Monkey patch the methods onto Module
Module.save = _save_method  # type: ignore[attr-defined]
Module.load = classmethod(_load_classmethod)  # type: ignore[attr-defined]
