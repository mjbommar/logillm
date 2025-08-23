"""Type definitions for module extensions and dynamic attributes.

This module provides type information for dynamic attributes that are added
to modules at runtime, helping type checkers understand these patterns.
"""

from typing import Any, Protocol, runtime_checkable

from .adapters import Adapter
from .demos import DemoManager


@runtime_checkable
class AdapterModule(Protocol):
    """Protocol for modules with adapter support."""

    adapter: Adapter


@runtime_checkable
class DemoModule(Protocol):
    """Protocol for modules with demonstration support."""

    demo_manager: DemoManager


@runtime_checkable
class ProviderModule(Protocol):
    """Protocol for modules with provider support."""

    provider: Any  # Provider type from providers module


@runtime_checkable
class ExtendedModule(AdapterModule, DemoModule, ProviderModule, Protocol):
    """Protocol for modules with all extensions."""

    pass
