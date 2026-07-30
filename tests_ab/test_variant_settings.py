"""验证A/B设置V2的纸张四角、毫米内缩和分组保存行为。"""

import importlib
import json

import pytest


VARIANTS = (
    ("maixcam2_app_A_quad.settings_store", "maixcam2_app_A_quad.config"),
    ("maixcam2_app_B_warp.settings_store", "maixcam2_app_B_warp.config"),
)


def _load_variant_modules(settings_module_name, config_module_name):
    """按参数导入同一变体的设置和配置模块，避免测试误混用稳定版配置。"""
    return (
        importlib.import_module(settings_module_name),
        importlib.import_module(config_module_name),
    )


@pytest.mark.parametrize(("settings_name", "config_name"), VARIANTS)
def test_settings_round_trip_paper_quad_and_inset(
    settings_name,
    config_name,
    tmp_path,
):
    """验证纸张四角和毫米INSET经过JSON保存与重载后保持不变。"""
    module, config_module = _load_variant_modules(settings_name, config_name)
    settings = module.build_default_runtime_settings(config_module.DEFAULT_CONFIG)
    settings["paper_quad"] = [[220, 70], [390, 80], [410, 330], [205, 325]]
    settings["inset_mm"] = 2.0
    settings_path = tmp_path / "settings.json"

    module.save_runtime_settings(settings_path, settings, (640, 480))
    loaded = module.load_runtime_settings(settings_path, config_module.DEFAULT_CONFIG)

    assert loaded["paper_quad"] == settings["paper_quad"]
    assert loaded["inset_mm"] == 2.0
    expected_version = 5 if settings_name.startswith("maixcam2_app_A_quad") else 2
    assert json.loads(settings_path.read_text(encoding="utf-8"))["version"] == expected_version


@pytest.mark.parametrize(("settings_name", "config_name"), VARIANTS)
def test_default_settings_start_without_locked_paper(settings_name, config_name):
    """验证首次启动没有伪造纸张四角，并从0mm内缩开始。"""
    module, config_module = _load_variant_modules(settings_name, config_name)

    settings = module.build_default_runtime_settings(config_module.DEFAULT_CONFIG)

    assert settings["paper_quad"] is None
    assert settings["inset_mm"] == 0.0


@pytest.mark.parametrize(("settings_name", "config_name"), VARIANTS)
def test_settings_reject_non_convex_or_out_of_frame_quad(settings_name, config_name):
    """验证损坏或越界四角不会进入运行时设置。"""
    module, config_module = _load_variant_modules(settings_name, config_name)
    settings = module.build_default_runtime_settings(config_module.DEFAULT_CONFIG)
    settings["paper_quad"] = [[220, 70], [700, 80], [250, 180], [205, 325]]

    with pytest.raises(ValueError, match="paper_quad"):
        module.validate_runtime_settings(settings, (640, 480))


@pytest.mark.parametrize(("settings_name", "config_name"), VARIANTS)
def test_group_merges_only_update_their_owned_fields(settings_name, config_name):
    """验证LOCK ROI和ADV各自只更新所属字段且不修改输入字典。"""
    module, config_module = _load_variant_modules(settings_name, config_name)
    current = module.build_default_runtime_settings(config_module.DEFAULT_CONFIG)
    staged = dict(current)
    staged.update(
        {
            "paper_quad": [[220, 70], [390, 80], [410, 330], [205, 325]],
            "inset_mm": 2.5,
            "fixed_threshold": 125.0,
            "min_area_ratio": 0.01,
            "open_kernel": 5,
            "close_kernel": 7,
        }
    )

    paper_merged = module.merge_paper_settings(current, staged)
    segmentation_merged = module.merge_segmentation_settings(current, staged)

    assert paper_merged["paper_quad"] == staged["paper_quad"]
    assert paper_merged["inset_mm"] == 2.5
    assert paper_merged["fixed_threshold"] == current["fixed_threshold"]
    assert segmentation_merged["paper_quad"] == current["paper_quad"]
    assert segmentation_merged["fixed_threshold"] == 125.0
    assert segmentation_merged["close_kernel"] == 7
    assert current["paper_quad"] is None


def test_a_v5_round_trip_persists_landscape_orientation(tmp_path):
    """A版LOCK ROI所属设置必须把横放方向写入V5并在重启后恢复。"""
    module, config_module = _load_variant_modules(*VARIANTS[0])
    settings = module.build_default_runtime_settings(config_module.DEFAULT_CONFIG)
    settings.update(
        {
            "paper_orientation": "landscape",
            "paper_quad": [[70, 100], [570, 90], [550, 380], [90, 390]],
            "work_x_mm": 33.5,
            "work_y_mm": 0.0,
            "work_width_mm": 230.0,
            "work_height_mm": 210.0,
            "split_y_mm": 105.0,
        }
    )
    settings_path = tmp_path / "settings_v5.json"

    module.save_runtime_settings(settings_path, settings, (640, 480))
    loaded = module.load_runtime_settings(
        settings_path,
        config_module.DEFAULT_CONFIG,
        frame_size=(640, 480),
    )

    assert module.SETTINGS_VERSION == 5
    assert loaded["paper_orientation"] == "landscape"
    assert loaded["work_x_mm"] == pytest.approx(33.5)
    assert loaded["work_height_mm"] == pytest.approx(210.0)
    assert json.loads(settings_path.read_text(encoding="utf-8"))["version"] == 5


def test_a_v4_settings_migrate_to_portrait_orientation(tmp_path):
    """没有方向字段的A版V4文件必须按历史竖放语义迁移，不能根据画面猜测。"""
    module, config_module = _load_variant_modules(*VARIANTS[0])
    payload = module.build_default_runtime_settings(
        config_module.DEFAULT_CONFIG,
        frame_size=(640, 480),
    )
    payload.pop("paper_orientation", None)
    payload["roi"] = [0.0, 0.0, 1.0, 1.0]
    payload["paper_quad"] = None
    settings_path = tmp_path / "settings_v4.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 4,
                "coordinate_space": "normalized",
                **payload,
            }
        ),
        encoding="utf-8",
    )

    loaded = module.load_runtime_settings(
        settings_path,
        config_module.DEFAULT_CONFIG,
        frame_size=(640, 480),
    )

    assert loaded["paper_orientation"] == "portrait"
