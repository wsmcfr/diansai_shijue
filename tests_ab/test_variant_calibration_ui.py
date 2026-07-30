"""验证A/B共同的五页调参布局、自动ROI和两类独立保存门。"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tests_ab.synthetic_paper import make_perspective_scene_with_four_pieces


VARIANTS = (
    "maixcam2_app_A_quad",
    "maixcam2_app_B_warp",
)


def _variant_modules(package_name):
    """导入同一变体的配置、设置、调参、触摸和入口模块。"""
    return SimpleNamespace(
        config=importlib.import_module(f"{package_name}.config"),
        settings=importlib.import_module(f"{package_name}.settings_store"),
        calibration=importlib.import_module(f"{package_name}.calibration_ui"),
        touch=importlib.import_module(f"{package_name}.touch_ui"),
        main=importlib.import_module(f"{package_name}.main"),
        locator=importlib.import_module(f"{package_name}.paper_locator"),
    )


def _make_detection(piece_count, edge_count=0):
    """构造LOCK ROI与ADV SAVE门槛所需的最小真实字段集合。"""
    pieces = [
        {"complete": True, "vertex_count": 4}
        for _ in range(int(piece_count))
    ]
    return SimpleNamespace(
        pieces=pieces,
        valid_contour_count=int(piece_count),
        edge_contours=[object() for _ in range(int(edge_count))],
        small_contours=[],
        large_contours=[],
        white_ratio=0.10,
        threshold=128.0,
    )


@pytest.mark.parametrize("package_name", VARIANTS)
def test_calibration_layout_has_five_top_tabs_and_five_fixed_controls(package_name):
    """验证640×480布局固定为五个顶部页签和五个底部触摸槽。"""
    modules = _variant_modules(package_name)

    buttons = modules.touch.build_calibration_layout(640, 480)

    assert tuple(buttons) == (
        "roi",
        "mask",
        "result",
        "adv",
        "cal",
        "control_1",
        "control_2",
        "control_3",
        "control_4",
        "control_5",
    )
    assert all(button.width >= 90 for button in buttons.values())


@pytest.mark.parametrize("package_name", VARIANTS)
def test_calibration_session_maps_simple_and_advanced_bottom_actions(package_name):
    """验证默认ROI页操作简单，ADV页才暴露详细分割参数。"""
    modules = _variant_modules(package_name)
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(saved, (640, 480))

    assert session.page_names == ("ROI", "MASK", "RESULT", "ADV")
    if package_name.endswith("A_quad"):
        assert session.bottom_actions() == (
            "auto_roi",
            "work_dec",
            "work_value",
            "work_inc",
            "lock_roi",
        )
    else:
        assert session.bottom_actions() == (
            "auto_roi",
            "inset_dec",
            "inset_value",
            "inset_inc",
            "lock_roi",
        )

    session.select_view("adv")

    assert session.bottom_actions() == (
        "next_param",
        "value_dec",
        "select_value",
        "value_inc",
        "save_segmentation",
    )


@pytest.mark.parametrize("package_name", VARIANTS)
def test_auto_roi_failure_preserves_previous_working_quad(package_name):
    """验证AUTO ROI失败只更新状态文字，不覆盖会话中的旧四角。"""
    modules = _variant_modules(package_name)
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    old_quad = [[220, 70], [390, 80], [410, 330], [205, 325]]
    saved["paper_quad"] = old_quad
    session = modules.calibration.CalibrationSession(saved, (640, 480))

    session.apply_auto_roi(modules.locator.PaperLocation.failed("no_candidate"))

    assert session.settings["paper_quad"] == old_quad
    assert session.status_text == "AUTO ROI FAIL"


@pytest.mark.parametrize("package_name", VARIANTS)
def test_auto_roi_success_updates_work_copy_and_inset_moves_by_half_mm(package_name):
    """验证定位成功只更新会话副本，INSET每次按0.5mm调整。"""
    modules = _variant_modules(package_name)
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(saved, (640, 480))
    paper_quad = np.float32([[220, 70], [390, 80], [410, 330], [205, 325]])
    location = modules.locator.PaperLocation(
        True,
        paper_quad=paper_quad,
        confidence=0.91,
        threshold=112.0,
        reason="ok",
    )

    session.apply_auto_roi(location)
    assert session.status_text.startswith("AUTO ROI OK")
    assert session.adjust_inset(1) is True

    assert session.settings["paper_quad"] == paper_quad.astype(float).tolist()
    assert session.settings["inset_mm"] == 0.5
    assert saved["paper_quad"] is None
    assert session.status_text == "INSET 0.5mm"


@pytest.mark.parametrize("package_name", VARIANTS)
@pytest.mark.parametrize("piece_count", [1, 2, 3, 4])
def test_lock_roi_accepts_one_to_four_complete_pieces(package_name, piece_count):
    """验证黑纸上已有1～4片完整碎片时都允许人工确认锁定。"""
    modules = _variant_modules(package_name)
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    saved["paper_quad"] = [[220, 70], [390, 80], [410, 330], [205, 325]]
    session = modules.calibration.CalibrationSession(saved, (640, 480))

    assert session.can_lock_roi(_make_detection(piece_count)) is True


@pytest.mark.parametrize("package_name", VARIANTS)
def test_lock_roi_rejects_piece_touching_paper_edge(package_name):
    """验证存在EDGE轮廓时不能确认自动四角。"""
    modules = _variant_modules(package_name)
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    saved["paper_quad"] = [[220, 70], [390, 80], [410, 330], [205, 325]]
    session = modules.calibration.CalibrationSession(saved, (640, 480))

    assert session.can_lock_roi(_make_detection(2, edge_count=1)) is False
    assert session.status_text == "ROI NEEDS 1-4 COMPLETE"


@pytest.mark.parametrize("package_name", VARIANTS)
def test_advanced_save_still_requires_good_four_of_four(package_name):
    """验证ADV分割参数保存门与LOCK ROI分离，仍严格要求GOOD 4/4。"""
    modules = _variant_modules(package_name)
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(saved, (640, 480))

    assert session.can_save_segmentation(_make_detection(3)) is False
    assert session.status_text == "NEED GOOD 4/4"
    assert session.can_save_segmentation(_make_detection(4)) is True


@pytest.mark.parametrize("package_name", VARIANTS)
def test_lock_and_advanced_save_persist_only_owned_groups(package_name, tmp_path):
    """验证入口两类保存动作各自持久化所属字段，不覆盖另一组工作值。"""
    modules = _variant_modules(package_name)
    runtime = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    interface = modules.main.InterfaceState()
    interface.toggle_calibration(runtime, (640, 480))
    session = interface.calibration_session
    session.settings["paper_quad"] = [[220, 70], [390, 80], [410, 330], [205, 325]]
    session.settings["inset_mm"] = 2.0
    session.settings["fixed_threshold"] = 123.0
    settings_path = tmp_path / "vision_settings.json"

    runtime, message = modules.main.handle_calibration_action(
        "control_5",
        interface,
        runtime,
        _make_detection(3),
        settings_path,
        (640, 480),
        frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    assert message == "ROI LOCKED"
    assert runtime["paper_quad"] == session.settings["paper_quad"]
    assert runtime["fixed_threshold"] is None

    session.select_view("adv")
    unchanged, message = modules.main.handle_calibration_action(
        "control_5",
        interface,
        runtime,
        _make_detection(3),
        settings_path,
        (640, 480),
        frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert unchanged is runtime
    assert message == "NEED GOOD 4/4"

    updated, message = modules.main.handle_calibration_action(
        "control_5",
        interface,
        runtime,
        _make_detection(4),
        settings_path,
        (640, 480),
        frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert message == "ADV SAVED"
    assert updated["fixed_threshold"] == 123.0
    assert updated["paper_quad"] == runtime["paper_quad"]


@pytest.mark.parametrize("package_name", VARIANTS)
def test_all_calibration_pages_render_fixed_640_by_480_without_mutating_source(package_name):
    """验证ROI/MASK/RESULT/ADV均使用固定画布且不会污染相机或工作原图。"""
    modules = _variant_modules(package_name)
    frame, paper_quad = make_perspective_scene_with_four_pieces()
    settings = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    settings["paper_quad"] = paper_quad.astype(float).tolist()
    session = modules.calibration.CalibrationSession(settings, (640, 480))
    buttons = modules.touch.build_calibration_layout(640, 480)

    if package_name.endswith("A_quad"):
        analysis = modules.main.analyze_quad_frame(frame, session.settings)
        preview_source = frame
    else:
        analysis = modules.main.analyze_warped_frame(
            frame,
            paper_quad,
            runtime_settings=session.settings,
        )
        preview_source = analysis.work_frame
    detection = analysis.detection
    quality = modules.calibration.evaluate_calibration(detection)

    for view in ("roi", "mask", "result", "adv"):
        session.select_view(view)
        source = frame if view == "roi" else preview_source
        original = source.copy()
        output = modules.calibration.draw_calibration_frame(
            source,
            detection,
            session,
            buttons,
            quality,
            session.status_text,
        )
        assert output.shape == (480, 640, 3)
        assert np.array_equal(source, original)
        assert np.count_nonzero(output) > 0


@pytest.mark.parametrize("package_name", VARIANTS)
def test_roi_page_draws_full_paper_and_active_quad_colors(package_name):
    """验证ROI页用青色完整A4和黄色机械有效区提供直观锁定反馈。"""
    modules = _variant_modules(package_name)
    frame, paper_quad = make_perspective_scene_with_four_pieces()
    settings = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    settings["paper_quad"] = paper_quad.astype(float).tolist()
    session = modules.calibration.CalibrationSession(settings, (640, 480))
    detection = _make_detection(4)
    detection.mask = np.zeros((200, 200), dtype=np.uint8)
    detection.roi = (0, 0, 200, 200)

    output = modules.calibration.draw_calibration_frame(
        frame,
        detection,
        session,
        modules.touch.build_calibration_layout(640, 480),
        modules.calibration.evaluate_calibration(detection),
    )

    assert np.any(np.all(output == np.asarray((255, 255, 0), np.uint8), axis=2))
    assert np.any(np.all(output == np.asarray((0, 255, 255), np.uint8), axis=2))


def test_a_roi_page_can_switch_paper_v_h_and_reset_default_work_region():
    """A版ROI参数中的PAPER项应切换V/H，并重置该方向的默认机械区。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(saved, (640, 480))

    # 默认项目为X；循环五次依次越过Y/W/H/SPLIT后应到达PAPER。
    for _ in range(5):
        session.cycle_work_item()

    assert session.current_item == "PAPER"
    assert session.adjust_work(1) is True
    assert session.settings["paper_orientation"] == "landscape"
    assert session.settings["work_x_mm"] == pytest.approx(33.5)
    assert session.settings["work_y_mm"] == pytest.approx(0.0)
    assert session.settings["work_width_mm"] == pytest.approx(230.0)
    assert session.settings["work_height_mm"] == pytest.approx(210.0)
    assert session.settings["split_y_mm"] == pytest.approx(105.0)
    assert session.status_text == "PAPER H"


def test_a_auto_roi_portrait_status_explicitly_reports_v_direction():
    """A版竖纸AUTO成功状态必须显示V，现场可直接确认毫米长宽没有弄反。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(saved, (640, 480))
    location = modules.locator.PaperLocation(
        True,
        paper_quad=np.float32([[180, 40], [460, 60], [440, 440], [160, 420]]),
        paper_orientation=modules.locator.PAPER_ORIENTATION_PORTRAIT,
        confidence=0.88,
        threshold=96.0,
        reason="ok",
    )

    assert session.apply_auto_roi(location) is True
    assert session.status_text == "AUTO ROI OK V 88%"


def test_a_auto_roi_applies_detected_orientation_and_matching_defaults():
    """A版AUTO ROI成功后应同步自动方向，避免横纸仍沿用竖纸毫米坐标。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(saved, (640, 480))
    paper_quad = np.float32([[70, 100], [570, 90], [550, 380], [90, 390]])
    location = modules.locator.PaperLocation(
        True,
        paper_quad=paper_quad,
        paper_orientation=modules.locator.PAPER_ORIENTATION_LANDSCAPE,
        confidence=0.92,
        threshold=110.0,
        reason="ok",
    )

    assert session.apply_auto_roi(location) is True

    assert session.settings["paper_orientation"] == "landscape"
    assert session.settings["work_x_mm"] == pytest.approx(33.5)
    assert session.settings["work_width_mm"] == pytest.approx(230.0)
    assert session.settings["work_height_mm"] == pytest.approx(210.0)
    assert session.settings["split_y_mm"] == pytest.approx(105.0)
    assert session.status_text == "AUTO ROI OK H 92%"
