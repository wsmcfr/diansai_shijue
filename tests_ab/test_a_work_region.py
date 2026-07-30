"""验证A版设置V5、可调毫米机械区域和分界线调参行为。"""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from maixcam2_app_A_quad import calibration_ui, main, paper_locator, settings_store
from maixcam2_app_A_quad.config import DEFAULT_CONFIG
from maixcam2_app_A_quad.touch_ui import build_calibration_layout
from tests_ab.synthetic_paper import DEFAULT_PAPER_QUAD, make_perspective_scene_with_four_pieces


def _locked_settings():
    """构造带完整A4四角的A版默认设置，供映射和绘制测试复用。"""
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    settings["paper_quad"] = DEFAULT_PAPER_QUAD.astype(float).tolist()
    return settings


def _empty_detection():
    """构造调参绘制所需的最小检测结果，避免测试依赖碎片轮廓细节。"""
    return SimpleNamespace(
        pieces=[],
        mask=np.zeros((120, 160), dtype=np.uint8),
        roi=(0, 0, 160, 120),
        valid_contour_count=0,
        edge_contours=[],
        small_contours=[],
        large_contours=[],
        white_ratio=0.0,
        threshold=128.0,
    )


def test_a_defaults_expose_v5_work_region_and_split():
    """A版首次启动必须使用210×230mm默认机械区域和A4中线。"""
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)

    assert settings_store.SETTINGS_VERSION == 5
    assert settings["paper_orientation"] == "portrait"
    assert settings["work_x_mm"] == 0.0
    assert settings["work_y_mm"] == 33.5
    assert settings["work_width_mm"] == 210.0
    assert settings["work_height_mm"] == 230.0
    assert settings["split_y_mm"] == 148.5


def test_a_v5_settings_round_trip_work_region(tmp_path):
    """方向和五个机械参数保存并重载后必须保持浮点毫米值和版本5。"""
    settings = _locked_settings()
    settings.update(
        {
            "work_x_mm": 5.5,
            "work_y_mm": 40.0,
            "work_width_mm": 198.0,
            "work_height_mm": 220.0,
            "split_y_mm": 151.0,
        }
    )
    settings_path = tmp_path / "vision_settings.json"

    settings_store.save_runtime_settings(settings_path, settings, (640, 480))
    loaded = settings_store.load_runtime_settings(settings_path, DEFAULT_CONFIG)

    assert loaded == settings_store.validate_runtime_settings(settings, (640, 480))
    assert json.loads(settings_path.read_text(encoding="utf-8"))["version"] == 5


def test_a_loads_v2_inset_as_equivalent_v3_region(tmp_path):
    """现有设备V2的INSET必须无损迁移成等价的X/Y/W/H区域。"""
    old_settings = {
        "roi": [0, 0, 640, 480],
        "paper_quad": DEFAULT_PAPER_QUAD.astype(float).tolist(),
        "inset_mm": 2.0,
        "fixed_threshold": None,
        "min_area_ratio": 0.002,
        "open_kernel": 3,
        "close_kernel": 5,
    }
    settings_path = tmp_path / "vision_settings_v2.json"
    settings_path.write_text(
        json.dumps({"version": 2, **old_settings}),
        encoding="utf-8",
    )

    loaded = settings_store.load_runtime_settings(settings_path, DEFAULT_CONFIG)

    assert loaded["work_x_mm"] == 2.0
    assert loaded["work_y_mm"] == 35.5
    assert loaded["work_width_mm"] == 206.0
    assert loaded["work_height_mm"] == 226.0
    assert loaded["split_y_mm"] == 148.5


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"work_x_mm": -0.5}, "work_x_mm"),
        ({"work_width_mm": 210.5}, "work_width_mm"),
        ({"work_height_mm": 230.5}, "work_height_mm"),
        ({"work_x_mm": 20.0, "work_width_mm": 200.0}, "A4"),
        ({"work_y_mm": 100.0, "work_height_mm": 210.0}, "A4"),
        ({"split_y_mm": 20.0}, "split_y_mm"),
        ({"split_y_mm": 280.0}, "split_y_mm"),
    ],
)
def test_a_rejects_work_region_outside_physical_limits(updates, message):
    """机械区域必须位于A4内且不超过电机覆盖高宽，分界线必须在区域中。"""
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    settings.update(updates)

    with pytest.raises(ValueError, match=message):
        settings_store.validate_runtime_settings(settings, (640, 480))


def test_build_work_quad_and_split_use_full_homography():
    """黄色四边形和红线必须由毫米点经完整单应映射得到。"""
    region = (10.0, 40.0, 180.0, 200.0)

    work_quad = paper_locator.build_work_quad(DEFAULT_PAPER_QUAD, region)
    split_line = paper_locator.build_split_segment(
        DEFAULT_PAPER_QUAD,
        region,
        split_y_mm=150.0,
    )

    for image_point, expected_mm in zip(
        work_quad,
        ((10.0, 40.0), (190.0, 40.0), (190.0, 240.0), (10.0, 240.0)),
    ):
        recovered = paper_locator.image_point_to_paper_mm(image_point, DEFAULT_PAPER_QUAD)
        np.testing.assert_allclose(recovered, expected_mm, atol=0.05)
    np.testing.assert_allclose(
        paper_locator.image_point_to_paper_mm(split_line[0], DEFAULT_PAPER_QUAD),
        (10.0, 150.0),
        atol=0.05,
    )
    np.testing.assert_allclose(
        paper_locator.image_point_to_paper_mm(split_line[1], DEFAULT_PAPER_QUAD),
        (190.0, 150.0),
        atol=0.05,
    )


def test_a_runtime_builds_active_quad_from_work_region_not_legacy_inset():
    """A版运行区域必须服从X/Y/W/H，旧INSET不能再次缩小黄色框。"""
    settings = _locked_settings()
    settings["inset_mm"] = 20.0
    settings.update(
        {
            "work_x_mm": 10.0,
            "work_y_mm": 40.0,
            "work_width_mm": 180.0,
            "work_height_mm": 200.0,
        }
    )

    actual = main.build_runtime_active_quad(settings)
    expected = paper_locator.build_work_quad(
        DEFAULT_PAPER_QUAD,
        (10.0, 40.0, 180.0, 200.0),
    )

    np.testing.assert_allclose(actual, expected, atol=0.01)


def test_a_simple_calibration_cycles_and_adjusts_five_work_values():
    """默认调参页中间按钮循环X/Y/W/H/SPLIT，正负键直接调整所选毫米值。"""
    session = calibration_ui.CalibrationSession(
        settings_store.build_default_runtime_settings(DEFAULT_CONFIG),
        (640, 480),
    )

    assert session.bottom_actions() == (
        "auto_roi",
        "work_dec",
        "work_value",
        "work_inc",
        "lock_roi",
    )
    assert session.current_item == "X"
    assert session.adjust_work(1) is False
    assert session.cycle_work_item() == "Y"
    assert session.adjust_work(1) is True
    assert session.settings["work_y_mm"] == 34.0
    assert session.cycle_work_item() == "W"
    assert session.adjust_work(-1) is True
    assert session.settings["work_width_mm"] == 209.5


def test_a_roi_preview_draws_paper_work_region_and_split_colors():
    """ROI页必须同时提供蓝框、黄框和红色分界线的直接视觉反馈。"""
    frame, paper_quad = make_perspective_scene_with_four_pieces()
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    settings["paper_quad"] = paper_quad.astype(float).tolist()
    session = calibration_ui.CalibrationSession(settings, (640, 480))
    detection = _empty_detection()

    output = calibration_ui.draw_calibration_frame(
        frame,
        detection,
        session,
        build_calibration_layout(640, 480),
        calibration_ui.evaluate_calibration(detection),
    )

    assert np.any(np.all(output == np.asarray(calibration_ui.COLOR_PAPER), axis=2))
    assert np.any(np.all(output == np.asarray(calibration_ui.COLOR_ACTIVE), axis=2))
    assert np.any(np.all(output == np.asarray(calibration_ui.COLOR_SPLIT), axis=2))


def test_a_landscape_roi_preview_draws_horizontal_split_at_105_mm():
    """横放调参预览的红线应按210mm纸高画在105mm处，并保持屏幕水平。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    settings.update(
        {
            "paper_orientation": "landscape",
            "paper_quad": [[20, 20], [614, 20], [614, 440], [20, 440]],
            "work_x_mm": 33.5,
            "work_y_mm": 0.0,
            "work_width_mm": 230.0,
            "work_height_mm": 210.0,
            "split_y_mm": 105.0,
        }
    )
    session = calibration_ui.CalibrationSession(settings, (640, 480))
    detection = _empty_detection()

    output = calibration_ui.draw_calibration_frame(
        frame,
        detection,
        session,
        build_calibration_layout(640, 480),
        calibration_ui.evaluate_calibration(detection),
    )

    # 简单矩形纸面中线为y=230；在黄色区中点取样，必须命中红色分界线。
    assert np.array_equal(output[230, 317], np.asarray(calibration_ui.COLOR_SPLIT))


def test_a_landscape_plan_draw_uses_orientation_for_split_line():
    """正常识别叠加即使尚无规划，也必须按横放纸面绘制正确红线。"""
    from maixcam2_app_A_quad.assembly_planner import draw_assembly_plan

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    paper_quad = np.float32([[20, 20], [614, 20], [614, 440], [20, 440]])

    output = draw_assembly_plan(
        frame,
        None,
        paper_quad,
        (33.5, 0.0, 230.0, 210.0),
        105.0,
        paper_orientation="landscape",
    )

    assert np.array_equal(output[230, 317], np.asarray((0, 0, 255)))


def test_a_work_value_action_cycles_selected_parameter():
    """主入口收到中间参数槽时必须循环项目而不是只显示旧INSET。"""
    runtime = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    interface = main.InterfaceState()
    interface.toggle_calibration(runtime, (640, 480))

    unchanged, message = main.handle_calibration_action(
        "control_3",
        interface,
        runtime,
        _empty_detection(),
        "unused.json",
        (640, 480),
        frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    assert unchanged is runtime
    assert message == "WORK Y"
    assert interface.calibration_session.current_item == "Y"
