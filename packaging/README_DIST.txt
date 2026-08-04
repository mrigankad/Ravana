# Ravana desktop (Windows)

## Run
1. Double-click `Ravana.exe`
2. On first launch, accept **Download models** (or use Help → Download models…)
3. Drop a source face and a target image/video, then **Swap**

## Models
Weights are **not** bundled (keeps the zip smaller). They download into a `models\`
folder next to this executable.

You can also copy an existing `models\` folder here before launching.

## Notes
- Bundle size is large because of PyTorch + ONNX Runtime + InsightFace.
- AMD GPUs: DirectML is used when available (`Device` → Auto / DirectML).
- Source install / rebuild: see the Ravana repo `scripts/build_exe.ps1`.
