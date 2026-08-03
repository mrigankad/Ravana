"""
Tests for the model manager module.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from face_swap.core.model_manager import (
    MODEL_PRESETS,
    ModelInfo,
    ModelManager,
    ModelRegistry,
    ensure_downloaded,
)


@pytest.fixture
def tmp_models_dir(tmp_path):
    return str(tmp_path / "models")


@pytest.fixture
def manager(tmp_models_dir):
    return ModelManager(models_dir=tmp_models_dir)


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        info = ModelInfo(
            name="test_model", version="v1.0", path="/tmp/test.onnx", format="onnx"
        )
        reg.register(info)
        assert reg.get_latest("test_model") == info

    def test_multiple_versions(self):
        reg = ModelRegistry()
        v1 = ModelInfo(name="m", version="v1.0", path="/tmp/v1.onnx", format="onnx")
        v2 = ModelInfo(name="m", version="v2.0", path="/tmp/v2.onnx", format="onnx")
        reg.register(v1)
        reg.register(v2)
        assert reg.get_latest("m").version == "v2.0"
        assert len(reg.list_versions("m")) == 2

    def test_no_duplicates(self):
        reg = ModelRegistry()
        info = ModelInfo(name="m", version="v1.0", path="/tmp/m.onnx", format="onnx")
        reg.register(info)
        reg.register(info)
        assert len(reg.list_versions("m")) == 1


class TestModelManager:
    def test_default_models_registered(self, manager):
        models = manager.list_models()
        assert "inswapper" in models
        assert "hyperswap" in models
        assert "gfpgan" in models
        assert "gpen" in models
        assert "xseg" in models
        assert "codeformer" in models

    def test_get_model(self, manager):
        info = manager.get_model("inswapper")
        assert info is not None
        assert info.name == "inswapper"
        assert info.mirrors

    def test_unknown_model(self, manager):
        info = manager.get_model("nonexistent")
        assert info is None

    def test_set_active_version(self, manager):
        manager.set_active_version("inswapper", "v0.7")
        info = manager.get_model("inswapper")
        assert info.version == "v0.7"

    def test_register_custom_model(self, manager, tmp_models_dir):
        custom = ModelInfo(
            name="custom_swap",
            version="v0.1",
            path=str(Path(tmp_models_dir) / "custom.onnx"),
            format="onnx",
            description="Test custom model",
        )
        manager.register_model(custom)
        assert "custom_swap" in manager.list_models()

    def test_manifest_persistence(self, tmp_models_dir):
        mgr1 = ModelManager(models_dir=tmp_models_dir)
        mgr1.set_active_version("inswapper", "v0.7")

        mgr2 = ModelManager(models_dir=tmp_models_dir)
        info = mgr2.get_model("inswapper")
        assert info.version == "v0.7"

    def test_rollback_no_previous(self, manager):
        result = manager.rollback("inswapper")
        assert result is None or result.version == "v0.7"

    def test_status_rows(self, manager):
        rows = manager.status()
        by_name = {r["name"]: r for r in rows}
        assert "hyperswap" in by_name
        assert by_name["hyperswap"]["downloadable"] is True
        assert by_name["simswap_256"]["downloadable"] is False

    def test_presets(self, manager):
        presets = manager.list_presets()
        assert "seamless" in presets
        assert "hyperswap" in presets["seamless"]
        assert presets == MODEL_PRESETS

    def test_ensure_model_skips_when_present(self, manager, tmp_models_dir):
        info = manager.get_model("xseg")
        assert info is not None
        Path(info.path).parent.mkdir(parents=True, exist_ok=True)
        Path(info.path).write_bytes(b"x" * (info.min_bytes + 10))

        with patch(
            "face_swap.core.model_manager._verify_sha256", return_value=True
        ), patch(
            "face_swap.core.model_manager.download_with_progress"
        ) as dl:
            out = manager.ensure_model("xseg", show_progress=False)
            assert out.path == info.path
            dl.assert_not_called()

    def test_ensure_model_downloads(self, manager):
        info = manager.get_model("xseg")
        assert info is not None
        assert not info.is_downloaded

        def fake_dl(url, dest, **kwargs):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"y" * (info.min_bytes + 5))

        with patch(
            "face_swap.core.model_manager.download_with_progress", side_effect=fake_dl
        ):
            out = manager.ensure_model("xseg", show_progress=False)
        assert out.is_downloaded

    def test_ensure_preset(self, manager):
        called = []

        def fake_ensure(name, version=None, **kwargs):
            called.append(name)
            info = manager.get_model(name)
            assert info is not None
            Path(info.path).parent.mkdir(parents=True, exist_ok=True)
            Path(info.path).write_bytes(b"z" * (info.min_bytes + 1))
            return info

        with patch.object(manager, "ensure_model", side_effect=fake_ensure):
            infos = manager.ensure_preset("seamless", show_progress=False)
        assert called == MODEL_PRESETS["seamless"]
        assert [i.name for i in infos] == MODEL_PRESETS["seamless"]


class TestEnsureDownloaded:
    def test_returns_existing(self, tmp_path):
        path = tmp_path / "m.onnx"
        path.write_bytes(b"a" * 2000)
        with patch(
            "face_swap.core.model_manager.download_with_progress"
        ) as dl:
            out = ensure_downloaded(
                str(path), urls=["http://example/m.onnx"], min_bytes=1000, show_progress=False
            )
            assert out == str(path)
            dl.assert_not_called()

    def test_raises_without_urls(self, tmp_path):
        path = tmp_path / "missing.onnx"
        with pytest.raises(FileNotFoundError):
            ensure_downloaded(str(path), urls=[], min_bytes=10, show_progress=False)

    def test_tries_mirrors(self, tmp_path):
        path = tmp_path / "m.onnx"
        calls = []

        def fake_dl(url, dest, **kwargs):
            calls.append(url)
            if "bad" in url:
                raise OSError("fail")
            Path(dest).write_bytes(b"ok" * 1000)

        with patch(
            "face_swap.core.model_manager.download_with_progress", side_effect=fake_dl
        ):
            ensure_downloaded(
                str(path),
                urls=["http://example/bad.onnx", "http://example/good.onnx"],
                min_bytes=100,
                show_progress=False,
            )
        assert len(calls) == 2
        assert path.is_file()
