from demos.gui_app.workers import build_config


def test_build_config_defaults():
    cfg = build_config("seamless", "auto", "default", "default", "all")
    assert cfg.quality == "seamless"
    assert cfg.device == "auto"
    assert cfg.enhance_method is None
    assert cfg.pixel_boost is None
    assert cfg.face_select == "all"
    assert cfg.realtime is False


def test_build_config_overrides_and_realtime():
    cfg = build_config("high", "dml", "gpen", "1024", "largest", realtime=True)
    assert cfg.enhance_method == "gpen"
    assert cfg.pixel_boost == 1024
    assert cfg.face_select == "largest"
    assert cfg.realtime is True
