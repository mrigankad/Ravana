"""
Native C/C++ API bindings for Face Swap SDK.

As per PRD Section 4.1:
  - C++ or C API suitable for integration into native apps.
  - Python ctypes wrapper for testing and hybrid deployments.

Build the native library:
    cd face_swap/native
    cmake -B build -S .
    cmake --build build

Then use from Python:
    from face_swap.native import NativeFaceSwap
    fs = NativeFaceSwap("./build/face_swap.dll")
"""

from .bindings import NativeFaceSwap

__all__ = ["NativeFaceSwap"]
