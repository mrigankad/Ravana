"""
Example 5: Custom Plugin Registration

Demonstrates how third-party developers can create and register
custom pipeline components using the plugin system.

Usage:
    python examples/05_custom_plugin.py
"""

import numpy as np
from face_swap.plugins import register_plugin, get_registry, PluginInfo


# ── Step 1: Define your custom detector ──────────────────────────────────

class MyCustomDetector:
    """
    Example custom face detector that wraps a different model.

    In practice, this could wrap:
      - A YOLO-based face detector
      - A custom-trained detector
      - A cloud API-backed detector
    """

    PLUGIN_CATEGORY = "detector"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Example custom face detector"

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None

    def load_model(self) -> None:
        """Load the underlying detection model."""
        print(f"  Loading custom detector on {self.device}...")
        # self._model = load_my_custom_model(self.device)
        print("  ✅ Model loaded!")

    def detect(self, frame: np.ndarray):
        """
        Detect faces in a frame.

        Returns a list of detections matching the FaceBBox interface.
        """
        h, w = frame.shape[:2]
        print(f"  Running detection on {w}×{h} frame...")

        # Placeholder: return a dummy detection
        from face_swap.core.types import FaceBBox
        return [
            FaceBBox(
                x1=w * 0.3, y1=h * 0.2,
                x2=w * 0.7, y2=h * 0.8,
                confidence=0.99,
            )
        ]


# ── Step 2: Register using the decorator ────────────────────────────────

@register_plugin(
    name="my_enhanced_detector",
    category="detector",
    version="2.0.0",
    priority=20,  # Higher priority = preferred over defaults
    description="Enhanced custom detector with better accuracy",
)
class MyEnhancedDetector(MyCustomDetector):
    """An even better detector registered via decorator."""

    def detect(self, frame: np.ndarray):
        print("  Running ENHANCED detection...")
        return super().detect(frame)


# ── Step 3: Manual registration ──────────────────────────────────────────

def register_manual():
    """Register a plugin manually (alternative to decorator)."""
    registry = get_registry()

    info = PluginInfo(
        name="my_custom_detector",
        version="1.0.0",
        category="detector",
        cls=MyCustomDetector,
        description="A manually registered custom detector",
        author="Example Author",
        priority=15,
    )
    registry.register(info)
    print("  ✅ Manual registration complete.")


def main():
    print("=" * 50)
    print("  Custom Plugin Registration Example")
    print("=" * 50)
    print()

    # Register the manual plugin
    print("Step 1: Registering plugins...")
    register_manual()
    print()

    # List all registered plugins
    print("Step 2: Listing all registered plugins...")
    registry = get_registry()
    all_plugins = registry.list_plugins()

    print(f"  Total plugins: {len(all_plugins)}")
    print()

    for plugin in all_plugins:
        print(f"  [{plugin.category}] {plugin.name} v{plugin.version}")
        if plugin.description:
            print(f"    → {plugin.description}")
    print()

    # Get preferred detector
    print("Step 3: Finding preferred detector...")
    preferred = registry.get_preferred("detector")
    if preferred:
        print(f"  Preferred: {preferred.__name__}")
    print()

    # Use a specific plugin
    print("Step 4: Using a specific detector...")
    DetCls = registry.get("detector", "my_custom_detector")
    if DetCls:
        detector = DetCls(device="cpu")
        detector.load_model()

        # Create a dummy frame and detect
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        print(f"  Detected {len(faces)} face(s)")
    print()

    print("✅ Plugin example complete!")
    print()
    print("Tip: Third-party packages can auto-register via entry_points:")
    print('  setup(entry_points={"face_swap.plugins": ["my_det = my_pkg:MyCls"]})')


if __name__ == "__main__":
    main()
