"""Tests for ONNX Runtime provider selection."""

from face_swap.core.providers import (
    available_providers,
    insightface_ctx_id,
    normalize_device,
    resolve_ort_providers,
)


class TestProviders:
    def test_normalize_aliases(self):
        assert normalize_device("AMD") == "dml"
        assert normalize_device("directml") == "dml"
        assert normalize_device("CUDA") == "cuda"
        assert normalize_device("auto") == "auto"

    def test_auto_returns_valid_providers(self):
        providers = resolve_ort_providers("auto")
        assert "CPUExecutionProvider" in providers
        assert len(providers) >= 1

    def test_cpu_always_available(self):
        assert resolve_ort_providers("cpu") == ["CPUExecutionProvider"]

    def test_dml_falls_back_without_package(self):
        # Without onnxruntime-directml installed, should fall back to CPU
        providers = resolve_ort_providers("dml")
        assert "CPUExecutionProvider" in providers
        avail = available_providers()
        if "DmlExecutionProvider" not in avail:
            assert providers == ["CPUExecutionProvider"]

    def test_ctx_id(self):
        assert insightface_ctx_id("cuda") == 0
        assert insightface_ctx_id("cpu") == -1
        assert insightface_ctx_id("dml") == -1
