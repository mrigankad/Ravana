"""
Plugin and extension system for Face Swap SDK.

Allows third-party packages to register custom pipeline components
that the SDK can discover and use at runtime.
"""

from .registry import (
    PluginRegistry,
    PluginInfo,
    get_registry,
    register_plugin,
)

__all__ = [
    "PluginRegistry",
    "PluginInfo",
    "get_registry",
    "register_plugin",
]
