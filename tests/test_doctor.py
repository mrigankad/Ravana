from pathlib import Path

from face_swap.core import doctor


def test_duplicate_ort_builds_fail_with_fix():
    vers = {"onnxruntime": "1.20.1", "onnxruntime-directml": "1.24.4"}
    c = doctor.check_ort_packages(vers.get)
    assert c.status == doctor.FAIL
    assert "pip install onnxruntime-directml" in c.fix


def test_single_ort_build_ok():
    assert doctor.check_ort_packages({"onnxruntime-gpu": "1.22"}.get).status == doctor.OK


def test_missing_ort_fails():
    assert doctor.check_ort_packages(lambda d: None).status == doctor.FAIL


def test_cpu_only_providers_warn():
    assert doctor.check_providers(["CPUExecutionProvider"]).status == doctor.WARN
    assert doctor.check_providers(["DmlExecutionProvider", "CPUExecutionProvider"]).status == doctor.OK


def test_nvidia_gpu_without_cuda_provider_warns():
    c = doctor.check_nvidia(["CPUExecutionProvider"], smi=lambda: "RTX 4070, 560.1, 12282 MiB")
    assert c.status == doctor.WARN and "ravana-sdk[gpu]" in c.fix
    assert doctor.check_nvidia(["CPUExecutionProvider"], smi=lambda: None) is None


def test_ffmpeg_and_models(tmp_path: Path):
    assert doctor.check_ffmpeg(lambda n: None).status == doctor.WARN
    assert doctor.check_models(tmp_path).status == doctor.WARN
    (tmp_path / "a.onnx").write_bytes(b"0" * 10)
    assert doctor.check_models(tmp_path).status == doctor.OK


def test_report_rollup_and_format():
    rep = doctor.Report(checks=[doctor.Check("a", doctor.OK, "x"), doctor.Check("b", doctor.WARN, "y", "do z")])
    assert rep.status == doctor.WARN
    text = doctor.format_report(rep)
    assert "fix: do z" in text and rep.to_dict()["status"] == "warn"
