"""
Model versioning and management.

As per PRD Section 8.3:
- Pre-trained model weights should be versioned and downloadable separately.
- The SDK must expose a mechanism to load different model versions
  (e.g., fast vs. high-quality) and roll back to previous versions
  if regressions are detected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("face_swap.models")

ProgressCallback = Callable[[int, Optional[int]], None]


@dataclass
class ModelInfo:
    """Metadata for a single model weight file."""

    name: str
    version: str
    path: str
    format: str  # "onnx", "pth", "pt"
    resolution: int = 128
    description: str = ""
    sha256: str = ""
    download_url: str = ""
    download_urls: List[str] = field(default_factory=list)
    min_bytes: int = 1_000_000
    license: str = ""

    @property
    def is_downloaded(self) -> bool:
        return (
            os.path.isfile(self.path)
            and os.path.getsize(self.path) >= max(1, self.min_bytes)
        )

    @property
    def mirrors(self) -> List[str]:
        urls: List[str] = []
        for u in list(self.download_urls) + ([self.download_url] if self.download_url else []):
            if u and u not in urls:
                urls.append(u)
        return urls


@dataclass
class ModelRegistry:
    """Registry of all known models with their versions."""

    models: Dict[str, List[ModelInfo]] = field(default_factory=dict)

    def register(self, model: ModelInfo) -> None:
        """Register a model (or a new version of an existing model)."""
        key = model.name
        if key not in self.models:
            self.models[key] = []
        for existing in self.models[key]:
            if existing.version == model.version:
                return
        self.models[key].append(model)
        self.models[key].sort(key=lambda m: m.version, reverse=True)

    def get_latest(self, name: str) -> Optional[ModelInfo]:
        versions = self.models.get(name, [])
        return versions[0] if versions else None

    def get_version(self, name: str, version: str) -> Optional[ModelInfo]:
        for m in self.models.get(name, []):
            if m.version == version:
                return m
        return None

    def list_versions(self, name: str) -> List[str]:
        return [m.version for m in self.models.get(name, [])]

    def list_models(self) -> List[str]:
        return list(self.models.keys())


# Presets used by CLI / ensure_preset
MODEL_PRESETS: Dict[str, List[str]] = {
    "core": ["inswapper"],
    "seamless": ["hyperswap", "gfpgan", "xseg"],
    "enhance": ["gfpgan", "gpen", "codeformer", "restoreformer"],
    "all": ["inswapper", "hyperswap", "gfpgan", "gpen", "codeformer", "restoreformer", "xseg"],
}


def download_with_progress(
    url: str,
    dest: str,
    *,
    label: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    show_progress: bool = True,
) -> None:
    """
    Download ``url`` to ``dest`` with optional progress reporting.

    ``progress(received_bytes, total_bytes_or_None)`` is called as data arrives.
    When ``show_progress`` is True and no callback is given, prints to stderr.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".partial"
    name = label or os.path.basename(dest)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Ravana-FaceSwap/0.3"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        total_i: Optional[int] = int(total) if total and total.isdigit() else None
        received = 0
        last_pct = -1

        def _default_progress(n: int, t: Optional[int]) -> None:
            nonlocal last_pct
            if t and t > 0:
                pct = int(100 * n / t)
                if pct != last_pct and (pct % 5 == 0 or n >= t):
                    last_pct = pct
                    mb = n / (1024 * 1024)
                    tot_mb = t / (1024 * 1024)
                    sys.stderr.write(f"\r{name}: {mb:.1f}/{tot_mb:.1f} MB ({pct}%)")
                    sys.stderr.flush()
                    if n >= t:
                        sys.stderr.write("\n")
            else:
                if n == 0 or n % (8 << 20) < (1 << 20):
                    sys.stderr.write(f"\r{name}: {n / (1024 * 1024):.1f} MB")
                    sys.stderr.flush()

        cb = progress
        if cb is None and show_progress:
            cb = _default_progress

        try:
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    received += len(chunk)
                    if cb is not None:
                        cb(received, total_i)
        except Exception:
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    os.replace(tmp, dest)


def ensure_downloaded(
    path: str,
    *,
    urls: Sequence[str],
    min_bytes: int = 1_000_000,
    sha256: Optional[str] = None,
    label: Optional[str] = None,
    show_progress: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> str:
    """
    Ensure a weight file exists at ``path``, downloading from ``urls`` if needed.

    Used by swap/enhance/mask modules so each does not reimplement urlretrieve.
    """
    if os.path.isfile(path) and os.path.getsize(path) >= max(1, min_bytes):
        if sha256 and not _verify_sha256(path, sha256):
            logger.warning("SHA-256 mismatch for %s — re-downloading.", path)
            os.remove(path)
        else:
            return path

    if not urls:
        raise FileNotFoundError(
            f"Model not found locally and no download URL: {path}"
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_err: Optional[Exception] = None
    display = label or os.path.basename(path)

    for url in urls:
        try:
            logger.info("Downloading %s from %s …", display, url)
            if show_progress and progress is None:
                print(f"Downloading {display} …", flush=True)
            download_with_progress(
                url,
                path,
                label=display,
                progress=progress,
                show_progress=show_progress,
            )
            if not (os.path.isfile(path) and os.path.getsize(path) >= max(1, min_bytes)):
                raise RuntimeError(f"Downloaded file too small: {path}")
            if sha256 and not _verify_sha256(path, sha256):
                logger.warning(
                    "SHA-256 mismatch for %s (expected %s); keeping file.",
                    path,
                    sha256,
                )
            return path
        except Exception as e:
            last_err = e
            logger.warning("Download failed from %s: %s", url, e)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            partial = path + ".partial"
            if os.path.isfile(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass

    raise RuntimeError(f"Could not download {display}: {last_err}")


def _verify_sha256(path: str, expected: str) -> bool:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest().lower() == expected.lower()


class ModelManager:
    """
    High-level manager for model discovery, download, verification,
    and version rollback.

    Usage:
        >>> mgr = ModelManager("./models")
        >>> mgr.ensure_model("inswapper")
        >>> mgr.ensure_preset("seamless")
        >>> for row in mgr.status():
        ...     print(row["name"], row["present"])
    """

    MANIFEST_FILE = "manifest.json"

    _DEFAULT_MODELS: List[ModelInfo] = [
        ModelInfo(
            name="inswapper",
            version="v0.7",
            path="inswapper_128.onnx",
            format="onnx",
            resolution=128,
            description="InsightFace InSwapper 128×128",
            download_url=(
                "https://github.com/deepinsight/insightface/releases"
                "/download/v0.7/inswapper_128.onnx"
            ),
            min_bytes=50_000_000,
            license="non-commercial (InsightFace)",
        ),
        ModelInfo(
            name="hyperswap",
            version="3.3.0",
            path="hyperswap_1a_256.onnx",
            format="onnx",
            resolution=256,
            description="FaceFusion HyperSwap 1a 256×256",
            download_urls=[
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx",
                "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
            ],
            sha256="c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add",
            min_bytes=50_000_000,
            license="ResearchRAIL",
        ),
        ModelInfo(
            name="gfpgan",
            version="1.4",
            path="gfpgan_1.4.onnx",
            format="onnx",
            resolution=512,
            description="GFPGAN 1.4 face restore (ONNX)",
            download_urls=[
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
            ],
            sha256="accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
            min_bytes=1_000_000,
            license="Apache-2.0 / community ONNX",
        ),
        ModelInfo(
            name="gpen",
            version="bfr_512",
            path="gpen_bfr_512.onnx",
            format="onnx",
            resolution=512,
            description="GPEN-BFR 512 face restore (ONNX)",
            download_urls=[
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.onnx",
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx",
            ],
            sha256="d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54",
            min_bytes=50_000_000,
            license="non-commercial (GPEN / FaceFusion)",
        ),
        ModelInfo(
            name="restoreformer",
            version="plus_plus",
            path="restoreformer_plus_plus.onnx",
            format="onnx",
            resolution=512,
            description="RestoreFormer++ face restore (ONNX)",
            download_urls=[
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/restoreformer_plus_plus.onnx",
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/restoreformer_plus_plus.onnx",
            ],
            sha256="f4db5a89902b6a2d452446f5721245a6f7185f699b6aec7b77285adb4d504337",
            min_bytes=50_000_000,
            license="FaceFusion / RestoreFormer community",
        ),
        ModelInfo(
            name="codeformer",
            version="1.0",
            path="codeformer.onnx",
            format="onnx",
            resolution=512,
            description="CodeFormer face restore (ONNX)",
            download_urls=[
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx",
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
            ],
            sha256="21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f",
            min_bytes=1_000_000,
            license="NTU S-Lab (research)",
        ),
        ModelInfo(
            name="xseg",
            version="1.0",
            path="xseg_1.onnx",
            format="onnx",
            resolution=256,
            description="XSeg face occlusion mask (ONNX)",
            download_urls=[
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx",
                "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx",
            ],
            sha256="c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3",
            min_bytes=1_000_000,
            license="FaceFusion / DeepFaceLab community",
        ),
        # Listed for docs; no auto-download URL (InsightFace handles buffalo_l)
        ModelInfo(
            name="simswap_256",
            version="v1.0",
            path="simswap_256.onnx",
            format="onnx",
            resolution=256,
            description="SimSwap 256×256 (balanced) — provide weights manually",
        ),
        ModelInfo(
            name="simswap_512",
            version="v1.0",
            path="simswap_512.onnx",
            format="onnx",
            resolution=512,
            description="SimSwap 512×512 (best quality) — provide weights manually",
        ),
    ]

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
        self._active_versions: Dict[str, str] = {}

        for m in self._DEFAULT_MODELS:
            full = ModelInfo(
                name=m.name,
                version=m.version,
                path=str(self.models_dir / m.path),
                format=m.format,
                resolution=m.resolution,
                description=m.description,
                sha256=m.sha256,
                download_url=m.download_url,
                download_urls=list(m.download_urls),
                min_bytes=m.min_bytes,
                license=m.license,
            )
            self.registry.register(full)

        self._load_manifest()

    def get_model(
        self, name: str, version: Optional[str] = None
    ) -> Optional[ModelInfo]:
        if version:
            return self.registry.get_version(name, version)
        if name in self._active_versions:
            return self.registry.get_version(name, self._active_versions[name])
        return self.registry.get_latest(name)

    def set_active_version(self, name: str, version: str) -> None:
        info = self.registry.get_version(name, version)
        if info is None:
            raise ValueError(f"Model {name} version {version} not registered.")
        self._active_versions[name] = version
        self._save_manifest()
        logger.info("Pinned %s to version %s", name, version)

    def rollback(self, name: str) -> Optional[ModelInfo]:
        versions = self.registry.list_versions(name)
        if len(versions) < 2:
            logger.warning(
                "Cannot rollback %s — only %d version(s).", name, len(versions)
            )
            return None

        current = self._active_versions.get(name, versions[0])
        try:
            idx = versions.index(current)
        except ValueError:
            idx = 0
        prev = versions[min(idx + 1, len(versions) - 1)]
        self.set_active_version(name, prev)
        return self.registry.get_version(name, prev)

    def ensure_model(
        self,
        name: str,
        version: Optional[str] = None,
        *,
        show_progress: bool = True,
        progress: Optional[ProgressCallback] = None,
    ) -> ModelInfo:
        """Make sure the model is downloaded and verified."""
        info = self.get_model(name, version)
        if info is None:
            raise ValueError(f"Unknown model: {name} (version={version})")

        ensure_downloaded(
            info.path,
            urls=info.mirrors,
            min_bytes=info.min_bytes,
            sha256=info.sha256 or None,
            label=f"{info.name} ({info.version})",
            show_progress=show_progress,
            progress=progress,
        )
        return info

    def ensure_preset(
        self,
        preset: str = "seamless",
        *,
        show_progress: bool = True,
    ) -> List[ModelInfo]:
        """Download every model in a named preset (``core``, ``seamless``, …)."""
        key = preset.lower().strip()
        if key not in MODEL_PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Choose from: {sorted(MODEL_PRESETS)}"
            )
        out: List[ModelInfo] = []
        for name in MODEL_PRESETS[key]:
            out.append(self.ensure_model(name, show_progress=show_progress))
        return out

    def ensure_models(
        self,
        names: Sequence[str],
        *,
        show_progress: bool = True,
    ) -> List[ModelInfo]:
        return [self.ensure_model(n, show_progress=show_progress) for n in names]

    def status(self) -> List[Dict[str, Union[str, bool, int]]]:
        """Return download status rows for all registered models."""
        rows: List[Dict[str, Union[str, bool, int]]] = []
        for name in sorted(self.list_models()):
            info = self.get_model(name)
            if info is None:
                continue
            size = os.path.getsize(info.path) if info.is_downloaded else 0
            rows.append(
                {
                    "name": info.name,
                    "version": info.version,
                    "present": info.is_downloaded,
                    "path": info.path,
                    "bytes": size,
                    "downloadable": bool(info.mirrors),
                    "description": info.description,
                    "license": info.license,
                }
            )
        return rows

    def register_model(self, model: ModelInfo) -> None:
        self.registry.register(model)
        self._save_manifest()

    def list_models(self) -> List[str]:
        return self.registry.list_models()

    def list_versions(self, name: str) -> List[str]:
        return self.registry.list_versions(name)

    def list_presets(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in MODEL_PRESETS.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.models_dir / self.MANIFEST_FILE

    def _save_manifest(self) -> None:
        default_keys = {(d.name, d.version) for d in self._DEFAULT_MODELS}
        data = {
            "active_versions": self._active_versions,
            "user_models": [
                {
                    "name": m.name,
                    "version": m.version,
                    "path": m.path,
                    "format": m.format,
                    "resolution": m.resolution,
                    "description": m.description,
                    "sha256": m.sha256,
                    "download_url": m.download_url,
                    "download_urls": m.download_urls,
                    "min_bytes": m.min_bytes,
                    "license": m.license,
                }
                for name in self.registry.models
                for m in self.registry.models[name]
                if (m.name, m.version) not in default_keys
            ],
        }
        with open(self._manifest_path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._active_versions = data.get("active_versions", {})
            for md in data.get("user_models", []):
                self.registry.register(ModelInfo(**md))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Corrupt manifest ignored: %s", exc)

    # ------------------------------------------------------------------
    # Download / Verify (kept for tests / callers)
    # ------------------------------------------------------------------

    @staticmethod
    def _download(url: str, dest: str) -> None:
        download_with_progress(url, dest, show_progress=False)

    @staticmethod
    def _verify_sha256(path: str, expected: str) -> bool:
        return _verify_sha256(path, expected)
