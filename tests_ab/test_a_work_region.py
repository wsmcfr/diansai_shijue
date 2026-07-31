"""验证A版设置V6、可调毫米机械区域和分界线调参行为。"""

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


def test_a_defaults_expose_v6_work_region_and_split():
    """A版首次启动必须使用完整210×297mm纸面区域和A4中线。"""
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)

    assert settings_store.SETTINGS_VERSION == 6
    assert settings["paper_orientation"] == "portrait"
    assert settings["work_x_mm"] == 0.0
    assert settings["work_y_mm"] == 0.0
    assert settings["work_width_mm"] == 210.0
    assert settings["work_height_mm"] == 297.0
    assert settings["split_y_mm"] == 148.5


def test_a_default_work_region_uses_full_a4_in_both_orientations():
    """横放和竖放默认黄色区域都必须等于完整蓝色A4区域。"""
    assert paper_locator.default_work_region_mm("portrait") == (0.0, 0.0, 210.0, 297.0)
    assert paper_locator.default_work_region_mm("landscape") == (0.0, 0.0, 297.0, 210.0)


def test_a_v6_settings_round_trip_work_region(tmp_path):
    """方向和五个机械参数保存并重载后必须保持浮点毫米值和版本6。"""
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
    assert json.loads(settings_path.read_text(encoding="utf-8"))["version"] == 6


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
        ({"work_height_mm": 297.5}, "work_height_mm"),
        ({"work_x_mm": 20.0, "work_width_mm": 200.0}, "A4"),
        ({"work_y_mm": 100.0, "work_height_mm": 210.0}, "A4"),
        ({"split_y_mm": 0.0}, "split_y_mm"),
        ({"split_y_mm": 297.0}, "split_y_mm"),
    ],
)
def test_a_rejects_work_region_outside_physical_limits(updates, message):
    """黄色区域只受当前A4纸面和内部分界线约束，不再受230mm行程约束。"""
    settings = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    settings.update(updates)

    with pytest.raises(ValueError, match=message):
        settings_store.validate_runtime_settings(settings, (640, 480))


def test_a_accepts_work_region_larger_than_230_mm_when_inside_current_paper():
    """竖纸高度和横纸宽度超过230mm仍应通过，供黄色框扩展到蓝框。"""
    portrait = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    assert settings_store.validate_runtime_settings(portrait, (640, 480))["work_height_mm"] == 297.0

    landscape = dict(portrait)
    landscape.update(
        {
            "paper_orientation": "landscape",
            "work_x_mm": 0.0,
            "work_y_mm": 0.0,
            "work_width_mm": 297.0,
            "work_height_mm": 210.0,
            "split_y_mm": 105.0,
        }
    )
    assert settings_store.validate_runtime_settings(landscape, (640, 480))["work_width_mm"] == 297.0


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


def test_a_simple_calibration_cycles_and_adjusts_six_work_values():
    """默认调参页循环X/Y/W/H/SPLIT/PAPER，并按完整纸面边界限制调整。"""
    session = calibration_ui.CalibrationSession(
        settings_store.build_default_runtime_settings(DEFAULT_CONFIG),
        (640, 480),
    )

    assert session.bottom_actions() == (
        "auto_roi",
        "paper_dec",
        "paper_value",
        "paper_inc",
        "lock_roi",
        "send_a4",
    )
    assert session.current_item == "MODE"
    session.select_view("mask")
    assert session.bottom_actions() == (
        "auto_roi",
        "work_dec",
        "work_value",
        "work_inc",
        "lock_roi",
        "send_a4",
    )
    assert session.current_item == "X"
    assert session.adjust_work(1) is False
    assert session.cycle_work_item() == "Y"
    assert session.adjust_work(1) is False
    assert session.cycle_work_item() == "W"
    assert session.adjust_work(-1) is True
    assert session.settings["work_width_mm"] == 209.5
    assert session.cycle_work_item() == "H"
    assert session.adjust_work(-1) is True
    assert session.settings["work_height_mm"] == 296.5


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
            "work_x_mm": 0.0,
            "work_y_mm": 0.0,
            "work_width_mm": 297.0,
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


def test_a_side_camera_manual_orientation_rotates_paper_axes_and_split_line():
    """侧装相机下手动切到V方向时，纸面坐标轴和红线必须整体旋转90度。

    该四边形模拟A4在侧装相机中呈横向且带透视倾斜的画面。PAPER V表示纸面
    毫米范围仍为210x297，因此297mm的Y轴应沿画面长边，148.5mm红线应连接
    两条长边的中点并在画面中近似竖直，不能继续沿蓝框长边横向绘制。
    """
    side_camera_quad = np.float32(
        ((90.0, 76.0), (220.0, 60.0), (228.0, 151.0), (92.0, 157.0))
    )

    split_line = paper_locator.build_split_segment(
        side_camera_quad,
        (0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
        paper_orientation="portrait",
    )

    delta = split_line[1] - split_line[0]
    assert abs(float(delta[1])) > abs(float(delta[0])) * 4.0
    for image_point, expected_mm in zip(
        split_line,
        ((0.0, 148.5), (210.0, 148.5)),
    ):
        recovered_mm = paper_locator.image_point_to_paper_mm(
            image_point,
            side_camera_quad,
            paper_orientation="portrait",
        )
        np.testing.assert_allclose(recovered_mm, expected_mm, atol=0.05)


def test_a_side_mount_direction_selects_origin_and_lower_region_side():
    """侧装方向枚举必须明确毫米原点，并允许目标下半区位于画面左侧或右侧。"""
    side_camera_quad = np.float32(
        ((90.0, 76.0), (220.0, 60.0), (228.0, 151.0), (92.0, 157.0))
    )

    lower_right_quad = paper_locator.orient_a4_quad_for_coordinates(
        side_camera_quad,
        "portrait",
        camera_mount_direction="side_lower_right",
    )
    lower_left_quad = paper_locator.orient_a4_quad_for_coordinates(
        side_camera_quad,
        "portrait",
        camera_mount_direction="side_lower_left",
    )

    # 目标区在右：机械原点落在画面左下；目标区在左：机械原点落在画面右上。
    np.testing.assert_allclose(lower_right_quad[0], side_camera_quad[3], atol=0.01)
    np.testing.assert_allclose(lower_left_quad[0], side_camera_quad[1], atol=0.01)
    right_target = paper_locator.paper_point_to_image_px(
        (105.0, 260.0),
        side_camera_quad,
        "portrait",
        camera_mount_direction="side_lower_right",
    )
    left_target = paper_locator.paper_point_to_image_px(
        (105.0, 260.0),
        side_camera_quad,
        "portrait",
        camera_mount_direction="side_lower_left",
    )
    assert right_target[0] > float(np.mean(side_camera_quad[:, 0]))
    assert left_target[0] < float(np.mean(side_camera_quad[:, 0]))


def test_a_work_value_action_cycles_selected_parameter():
    """主入口收到中间参数槽时必须循环项目而不是只显示旧INSET。"""
    runtime = settings_store.build_default_runtime_settings(DEFAULT_CONFIG)
    interface = main.InterfaceState()
    interface.toggle_calibration(runtime, (640, 480))
    interface.calibration_session.select_view("mask")

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
