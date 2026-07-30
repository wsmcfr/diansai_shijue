"""现场视觉参数校验、保存和加载测试。"""

import json

import pytest

from maixcam2_app.config import DEFAULT_CONFIG


def test_default_runtime_settings_use_full_frame():
    """没有现场配置时必须使用完整相机画面和默认视觉参数。"""
    from maixcam2_app.settings_store import build_default_runtime_settings

    settings = build_default_runtime_settings(DEFAULT_CONFIG)

    assert settings["roi"] == [0, 0, 640, 480]
    assert settings["fixed_threshold"] is None
    assert settings["min_area_ratio"] == DEFAULT_CONFIG["min_area_ratio"]


def test_runtime_settings_round_trip(tmp_path):
    """合法现场参数必须经过原子JSON保存后无损加载。"""
    from maixcam2_app.settings_store import (
        build_default_runtime_settings,
        load_runtime_settings,
        save_runtime_settings,
    )

    path = tmp_path / "vision_settings.json"
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["roi"] = [160, 30, 300, 420]
    settings["fixed_threshold"] = 134.0

    save_runtime_settings(path, settings, frame_size=(640, 480))
    loaded = load_runtime_settings(path, DEFAULT_CONFIG)

    assert loaded == settings
    assert not (tmp_path / "vision_settings.json.tmp").exists()


def test_missing_runtime_settings_return_independent_defaults(tmp_path):
    """配置文件不存在时每次都应返回独立默认字典，避免跨会话污染。"""
    from maixcam2_app.settings_store import load_runtime_settings

    path = tmp_path / "missing.json"
    first = load_runtime_settings(path, DEFAULT_CONFIG)
    second = load_runtime_settings(path, DEFAULT_CONFIG)
    first["roi"][0] = 99

    assert second["roi"] == [0, 0, 640, 480]


def test_invalid_runtime_settings_raise_without_partial_merge(tmp_path):
    """损坏或越界配置必须整体拒绝，不能把部分字段混入默认参数。"""
    from maixcam2_app.settings_store import load_runtime_settings

    path = tmp_path / "vision_settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "roi": [0, 0, -1, 20],
                "fixed_threshold": None,
                "min_area_ratio": 0.002,
                "open_kernel": 3,
                "close_kernel": 5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ROI"):
        load_runtime_settings(path, DEFAULT_CONFIG)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixed_threshold", 300),
        ("min_area_ratio", 0.0),
        ("open_kernel", 4),
        ("close_kernel", -1),
    ],
)
def test_runtime_settings_reject_invalid_tunable_values(field, value):
    """阈值、面积比例和形态学核超出允许范围时必须拒绝。"""
    from maixcam2_app.settings_store import (
        build_default_runtime_settings,
        validate_runtime_settings,
    )

    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings[field] = value

    with pytest.raises(ValueError):
        validate_runtime_settings(settings, frame_size=(640, 480))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open_kernel", 9),
        ("close_kernel", 11),
    ],
)
def test_runtime_settings_reject_kernels_outside_calibration_options(field, value):
    """持久参数不得接受调参状态机无法枚举的形态学核尺寸。"""
    from maixcam2_app.settings_store import (
        build_default_runtime_settings,
        validate_runtime_settings,
    )

    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings[field] = value

    with pytest.raises(ValueError, match=field):
        validate_runtime_settings(settings, frame_size=(640, 480))


def test_merge_runtime_config_does_not_modify_defaults():
    """合并现场参数必须返回新配置，不能修改全局DEFAULT_CONFIG。"""
    from maixcam2_app.settings_store import (
        build_default_runtime_settings,
        merge_runtime_config,
    )

    original_min_area = DEFAULT_CONFIG["min_area_ratio"]
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["min_area_ratio"] = 0.0005

    merged = merge_runtime_config(DEFAULT_CONFIG, settings)

    assert merged["min_area_ratio"] == 0.0005
    assert DEFAULT_CONFIG["min_area_ratio"] == original_min_area
