# Installation

## Requirements

- **Python** 3.9 or later
- **GPU** (recommended): NVIDIA GPU with CUDA 11.8+ (RTX 30/40 series for real-time)
- **CPU**: Works on CPU but limited to images and low-resolution video

## Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/ravana.git
cd ravana

# Install with pip
pip install -e .
```

## Full Install (with all optional dependencies)

```bash
# Core + training + enhancement + optimization
pip install -e ".[all]"

# Or install specific extras:
pip install -e ".[training]"     # Model training (PyTorch)
pip install -e ".[enhancement]"  # GFPGAN, RealESRGAN
pip install -e ".[tensorrt]"     # TensorRT optimization
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

```python
import face_swap
print(f"Ravana v{face_swap.__version__}")
print(f"TensorRT available: {face_swap.TensorRTExporter is not None}")
print(f"Training available: {face_swap.FaceSwapTrainer is not None}")
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
| Windows  | CUDA            | ✅ Full support |
| Linux    | CUDA            | ✅ Full support |
| macOS    | Metal (MPS)     | ✅ Supported |
| Android  | TFLite          | ✅ Export ready |
| iOS      | CoreML + ANE    | ✅ Export ready |
