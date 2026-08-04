"""Resolve resource and runtime paths for source vs frozen (PyInstaller) builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def bundle_dir() -> Path:
    """PyInstaller extract dir, or repo root when running from source."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def app_dir() -> Path:
    """Directory containing the executable (or repo root in source)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return bundle_dir().joinpath(*parts)


def ensure_runtime_cwd() -> Path:
    """
    Make relative paths like ./models resolve next to the .exe when frozen.
    Returns the directory used as cwd.
    """
    target = app_dir()
    if is_frozen():
        os.chdir(target)
    return target


def models_dir() -> Path:
    """Weights live next to the exe when frozen, else ./models under the repo."""
    return app_dir() / "models"
