# Installation

## Requirements

- **Python** 3.9 or later (3.11–3.13 recommended)
- **GPU** (recommended):
  - NVIDIA + CUDA → `pip install -e ".[gpu]"`
  - AMD on Windows → `pip install -e ".[directml]"` (DirectML)
- **CPU**: Works for images; use `quality="fast_cpu"` for video

ONNX restore/swap models (GFPGAN, GPEN, CodeFormer, HyperSwap, XSeg) download automatically on first use, or prefetch:

```bash
python -m demos.cli models list --presets
python -m demos.cli models download --preset seamless
```

The pip `.[enhancement]` extras (basicsr/GFPGAN wheels) are optional and often fail on Python 3.13 — prefer the built-in ONNX path.

## Quick Install

```bash
# Clone the repository
git clone https://github.com/mrigankad/Ravana.git
cd Ravana

# Install with pip
pip install -e .
```

## Full Install (with all optional dependencies)

```bash
# Core + training + optimization + docs
pip install -e ".[all]"

# Or install specific extras:
pip install -e ".[gpu]"          # onnxruntime-gpu (NVIDIA)
pip install -e ".[directml]"     # onnxruntime-directml (AMD Windows)
pip install -e ".[training]"     # Model training extras
pip install -e ".[enhancement]"  # Optional PyTorch GFPGAN wheels (not required)
pip install -e ".[tensorrt]"     # TensorRT optimization
pip install -e ".[dev]"          # pytest, linters, type checkers
```

## Docker

```bash
# Build the GPU-enabled Docker image
docker build -t face-swap .

# Run
docker run --gpus all -v $(pwd)/data:/data face-swap \
    --mode image --source /data/source.jpg \
    --target /data/target.jpg --output /data/output.jpg
```

## Verify Installation

```bash
python scripts/first_run_check.py
# Prefetch seamless weights:
python scripts/first_run_check.py --download
```

```python
import face_swap
from face_swap.core.providers import resolve_ort_providers

print(f"Ravana v{face_swap.__version__}")
print("ORT providers (auto):", resolve_ort_providers("auto"))
```

## Native C++ Library (Optional)

```bash
cd face_swap/native
cmake -B build -S .
cmake --build build --config Release
```

## Platform Support

| Platform | GPU Acceleration | Status |
|----------|-----------------|--------|
| Windows  | CUDA / DirectML | ✅ Supported |
| Linux    | CUDA            | ✅ Supported |
| macOS    | CPU (MPS experimental) | ✅ Supported |
| Android  | TFLite          | ✅ Export ready |
| iOS      | CoreML + ANE    | ✅ Export ready |
