"""Hardware doctor: explain why Ravana is (or isn't) using your GPU.

Every check returns a :class:`Check` with a status and a one-line fix, so the
CLI (`ravana doctor`), the GUI and bug reports all say the same thing.
No check downloads anything or needs model weights.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

OK, WARN, FAIL = "ok", "warn", "fail"

# onnxruntime ships as mutually exclusive wheels; two installed at once is the
# most common reason a GPU build silently runs on CPU.
ORT_DISTS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-silicon",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    device: str = "cpu"

    @property
    def status(self) -> str:
        states = {c.status for c in self.checks}
        return FAIL if FAIL in states else WARN if WARN in states else OK

    def to_dict(self) -> dict:
        return {"status": self.status, "device": self.device, "checks": [asdict(c) for c in self.checks]}


def _installed(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_python() -> Check:
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro} ({platform.machine()})"
    if v < (3, 9):
        return Check("python", FAIL, detail, "Install Python 3.9 to 3.12.")
    if v >= (3, 13):
        return Check("python", WARN, detail, "Some GPU wheels lag behind 3.13. Use 3.10 to 3.12 if installs fail.")
    return Check("python", OK, detail)


def check_ort_packages(installed: Callable[[str], Optional[str]] = _installed) -> Check:
    found = {d: v for d in ORT_DISTS if (v := installed(d))}
    if not found:
        return Check("onnxruntime", FAIL, "onnxruntime is not installed",
                     'pip install onnxruntime  (or "ravana-sdk[gpu]" / "ravana-sdk[directml]")')
    detail = ", ".join(f"{d} {v}" for d, v in found.items())
    if len(found) > 1:
        keep = next((d for d in ("onnxruntime-gpu", "onnxruntime-directml") if d in found), "onnxruntime")
        drop = " ".join(d for d in found if d != keep)
        return Check("onnxruntime", FAIL, f"Several builds installed: {detail}",
                     f"pip uninstall -y {drop} {keep} && pip install {keep}")
    return Check("onnxruntime", OK, detail)


def check_providers(avail: Optional[List[str]] = None) -> Check:
    if avail is None:
        from .providers import available_providers

        avail = available_providers()
    gpu = [p for p in avail if p not in ("CPUExecutionProvider", "AzureExecutionProvider")]
    if gpu:
        return Check("providers", OK, ", ".join(avail))
    return Check("providers", WARN, "CPU only: " + ", ".join(avail),
                 'NVIDIA: pip install "ravana-sdk[gpu]". AMD or Intel on Windows: pip install "ravana-sdk[directml]".')


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def check_nvidia(avail: List[str], smi: Callable[[], Optional[str]] = _nvidia_smi) -> Optional[Check]:
    info = smi()
    if info is None:
        if "CUDAExecutionProvider" in avail:
            return Check("nvidia", WARN, "CUDA provider present but nvidia-smi not found",
                         "Install or update the NVIDIA driver.")
        return None
    if "CUDAExecutionProvider" not in avail:
        return Check("nvidia", WARN, f"NVIDIA GPU found ({info}) but onnxruntime has no CUDA",
                     'pip uninstall -y onnxruntime && pip install "ravana-sdk[gpu]"')
    return Check("nvidia", OK, info)


def check_cuda_libs(avail: List[str]) -> Optional[Check]:
    """CUDA provider listed doesn't mean cuDNN/cuBLAS load. Try to preload them."""
    if "CUDAExecutionProvider" not in avail:
        return None
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        return Check("cuda-libs", OK, "CUDA and cuDNN libraries load")
    except Exception as e:  # pragma: no cover - depends on the machine
        return Check("cuda-libs", FAIL, f"CUDA libraries failed to load: {e}",
                     'pip install "nvidia-cudnn-cu12" "nvidia-cublas-cu12", or match CUDA 12 + cuDNN 9 to your onnxruntime-gpu.')


def check_ffmpeg(which: Callable[[str], Optional[str]] = shutil.which) -> Check:
    if which("ffmpeg"):
        return Check("ffmpeg", OK, "ffmpeg on PATH (video audio is kept)")
    return Check("ffmpeg", WARN, "ffmpeg not found: video swaps lose their audio",
                 "Install ffmpeg (winget install Gyan.FFmpeg, brew install ffmpeg, apt install ffmpeg).")


def check_models(models_dir: Path) -> Check:
    onnx = list(models_dir.glob("**/*.onnx")) if models_dir.exists() else []
    if not onnx:
        return Check("models", WARN, f"No models in {models_dir}", "ravana models download --preset seamless")
    size = sum(p.stat().st_size for p in onnx) / 1e9
    detail = f"{len(onnx)} models, {size:.1f} GB in {models_dir}"
    return Check("models", OK, detail)


def check_disk(models_dir: Path, need_gb: float = 2.0) -> Check:
    probe = models_dir if models_dir.exists() else Path.cwd()
    free = shutil.disk_usage(probe).free / 1e9
    if free < need_gb:
        return Check("disk", WARN, f"{free:.1f} GB free", f"Free at least {need_gb:.0f} GB for model downloads.")
    return Check("disk", OK, f"{free:.0f} GB free")


def run(models_dir: str = "./models") -> Report:
    from .providers import available_providers, resolve_ort_providers

    avail = available_providers()
    rep = Report(device=resolve_ort_providers("auto")[0].replace("ExecutionProvider", ""))
    for c in (
        check_python(),
        check_ort_packages(),
        check_providers(avail),
        check_nvidia(avail),
        check_cuda_libs(avail),
        check_ffmpeg(),
        check_models(Path(models_dir)),
        check_disk(Path(models_dir)),
    ):
        if c is not None:
            rep.checks.append(c)
    return rep


def format_report(rep: Report) -> str:
    mark = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}
    lines = [f"Ravana doctor: device auto -> {rep.device}", ""]
    for c in rep.checks:
        lines.append(f"{mark[c.status]}  {c.name:<12} {c.detail}")
        if c.fix and c.status != OK:
            lines.append(f"      {'':<12} fix: {c.fix}")
    lines += ["", {OK: "All good.", WARN: "Works, with the warnings above.",
                   FAIL: "Fix the failures above, then rerun `ravana doctor`."}[rep.status]]
    return "\n".join(lines)
