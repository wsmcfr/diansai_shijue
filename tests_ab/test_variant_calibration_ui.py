"""验证A/B调参布局、自动ROI和两类独立保存门。"""

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


def _make_a_session(frame_size=(640, 480), paper_quad=None):
    """构造A版调参会话，并可注入历史蓝框验证手动模式初始化优先级。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    saved = modules.settings.build_default_runtime_settings(
        modules.config.DEFAULT_CONFIG,
        frame_size=frame_size,
    )
    if paper_quad is not None:
        saved["paper_quad"] = np.asarray(paper_quad, dtype=float).tolist()
    return modules, modules.calibration.CalibrationSession(saved, frame_size)


def _session_work_region(session):
    """返回会话中的黄色毫米工作区，便于直接检查它是否恢复为完整A4。"""
    return (
        session.settings["work_x_mm"],
        session.settings["work_y_mm"],
        session.settings["work_width_mm"],
        session.settings["work_height_mm"],
    )


def _paper_mean_size(paper_quad):
    """计算四边形上下边平均宽和左右边平均高，匹配手动W/H的定义。"""
    quad = np.asarray(paper_quad, dtype=float)
    mean_width = (
        np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
    ) / 2.0
    mean_height = (
        np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
    ) / 2.0
    return mean_width, mean_height


def _paper_edge_directions(paper_quad):
    """返回上下左右四条边的单位方向，用于证明缩放没有抹掉透视斜率。"""
    quad = np.asarray(paper_quad, dtype=float)
    edge_vectors = (
        quad[1] - quad[0],
        quad[2] - quad[3],
        quad[3] - quad[0],
        quad[2] - quad[1],
    )
    return np.asarray(
        [vector / np.linalg.norm(vector) for vector in edge_vectors],
        dtype=float,
    )


def _make_a_manual_session(paper_quad, frame_size=(640, 480)):
    """以指定透视蓝框创建A版MANUAL会话，供几何和边界测试复用。"""
    modules, session = _make_a_session(frame_size=frame_size, paper_quad=paper_quad)
    session.set_roi_mode("MANUAL")
    return modules, session


@pytest.mark.parametrize("package_name", VARIANTS)
def test_calibration_layout_keeps_variant_specific_fixed_controls(package_name):
    """验证A版使用六槽发送A4，B版继续保持原五槽且所有槽互不重叠。"""
    modules = _variant_modules(package_name)

    buttons = modules.touch.build_calibration_layout(640, 480)
    expected_names = (
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
    if package_name.endswith("A_quad"):
        expected_names += ("control_6",)

    assert tuple(buttons) == expected_names
    assert all(button.width >= 90 for button in buttons.values())

    # 分别检查顶部和底部同一行的相邻按钮，防止增加第六槽后触摸区相互覆盖。
    for row_names in (expected_names[:5], expected_names[5:]):
        row = [buttons[name] for name in row_names]
        assert all(
            left.x + left.width < right.x
            for left, right in zip(row, row[1:])
        )


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
            "paper_dec",
            "paper_value",
            "paper_inc",
            "lock_roi",
            "send_a4",
        )
        session.select_view("mask")
        assert session.bottom_actions() == (
            "auto_roi",
            "work_dec",
            "work_value",
            "work_inc",
            "lock_roi",
            "send_a4",
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

    expected_advanced = (
        "next_param",
        "value_dec",
        "select_value",
        "value_inc",
        "save_segmentation",
    )
    if package_name.endswith("A_quad"):
        expected_advanced += ("disabled",)
    assert session.bottom_actions() == expected_advanced


def test_a_roi_session_starts_in_auto_and_cycles_manual_paper_items():
    """A版ROI会话应默认AUTO，并独立循环MODE、蓝框几何和像素步进项目。"""
    _modules, session = _make_a_session()

    assert session.roi_mode == "AUTO"
    assert session.current_roi_item == "MODE"
    assert [session.cycle_roi_item() for _ in range(6)] == [
        "X",
        "Y",
        "W",
        "H",
        "STEP",
        "MODE",
    ]
    assert session.manual_roi_step_px == 5


def test_a_switching_to_manual_builds_centered_a4_when_quad_is_missing():
    """无AUTO和历史蓝框时，MANUAL应生成居中竖版A4框并重置完整黄色区。"""
    _modules, session = _make_a_session(frame_size=(1280, 960))

    assert session.set_roi_mode("MANUAL") is True

    quad = np.asarray(session.settings["paper_quad"], dtype=float)
    mean_width = (
        np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
    ) / 2.0
    mean_height = (
        np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
    ) / 2.0
    assert quad.shape == (4, 2)
    np.testing.assert_allclose(quad.mean(axis=0), (640.0, 480.0), atol=0.01)
    assert mean_width / mean_height == pytest.approx(210.0 / 297.0, rel=1e-4)
    assert _session_work_region(session) == pytest.approx((0.0, 0.0, 210.0, 297.0))
    assert session.settings["split_y_mm"] == pytest.approx(148.5)
    assert session.status_text == "ROI MANUAL"


def test_a_switching_to_manual_preserves_existing_quad_and_resets_full_work_area():
    """已有AUTO或历史蓝框时，MANUAL只把黄色区恢复完整A4，不得重建蓝框。"""
    old_quad = np.float32([[120, 80], [500, 60], [530, 410], [90, 430]])
    _modules, session = _make_a_session(paper_quad=old_quad)
    session.settings.update(
        {
            "work_x_mm": 10.0,
            "work_y_mm": 20.0,
            "work_width_mm": 180.0,
            "work_height_mm": 240.0,
            "split_y_mm": 140.0,
        }
    )

    assert session.set_roi_mode("MANUAL") is True

    np.testing.assert_allclose(session.settings["paper_quad"], old_quad, atol=0.01)
    assert _session_work_region(session) == pytest.approx((0.0, 0.0, 210.0, 297.0))
    assert session.settings["split_y_mm"] == pytest.approx(148.5)


def test_a_switching_back_to_auto_keeps_manual_quad_available_for_retry():
    """MANUAL切回AUTO只改变会话模式，AUTO重试前不得清空当前蓝框。"""
    old_quad = np.float32([[120, 80], [500, 60], [530, 410], [90, 430]])
    _modules, session = _make_a_session(paper_quad=old_quad)
    session.set_roi_mode("MANUAL")

    assert session.set_roi_mode("AUTO") is True

    assert session.roi_mode == "AUTO"
    np.testing.assert_allclose(session.settings["paper_quad"], old_quad, atol=0.01)
    assert session.status_text == "ROI AUTO"


def test_a_auto_roi_success_returns_session_to_auto_mode():
    """MANUAL中重新执行AUTO成功后应切回AUTO，避免减加号继续误改新蓝框。"""
    modules, session = _make_a_session()
    session.set_roi_mode("MANUAL")
    location = modules.locator.PaperLocation(
        True,
        paper_quad=np.float32([[120, 80], [500, 60], [530, 410], [90, 430]]),
        paper_orientation=modules.locator.PAPER_ORIENTATION_PORTRAIT,
        confidence=0.91,
        threshold=120.0,
        reason="ok",
    )

    assert session.apply_auto_roi(location) is True

    assert session.roi_mode == "AUTO"


def test_a_auto_roi_failure_keeps_manual_mode_and_current_quad(monkeypatch, tmp_path):
    """MANUAL中AUTO失败必须保留模式和蓝框，用户可继续按减加号精调。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    runtime = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    interface = modules.main.InterfaceState()
    interface.toggle_calibration(runtime, (640, 480))
    session = interface.calibration_session
    session.set_roi_mode("MANUAL")
    before = session.snapshot()
    monkeypatch.setattr(
        modules.main,
        "locate_black_paper",
        lambda *_args, **_kwargs: modules.locator.PaperLocation.failed("no_candidate"),
    )

    unchanged, message = modules.main.handle_calibration_action(
        "control_1",
        interface,
        runtime,
        _make_detection(1),
        tmp_path / "settings.json",
        (640, 480),
        frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    assert unchanged is runtime
    assert message == "AUTO ROI FAIL"
    assert session.roi_mode == "MANUAL"
    assert session.settings["paper_quad"] == before["paper_quad"]


def test_a_manual_xy_translates_every_corner_by_selected_step():
    """X/Y必须按分析像素刚性平移全部四角，不能改变纸面形状。"""
    paper_quad = np.float32([[120, 80], [500, 60], [530, 410], [90, 430]])
    _modules, session = _make_a_manual_session(paper_quad)
    before = np.asarray(session.settings["paper_quad"], dtype=float)

    assert session.adjust_paper_quad("X", 1) is True
    after_x = np.asarray(session.settings["paper_quad"], dtype=float)
    np.testing.assert_allclose(after_x, before + (5.0, 0.0), atol=1e-6)

    assert session.adjust_paper_quad("Y", -1) is True
    after_y = np.asarray(session.settings["paper_quad"], dtype=float)
    np.testing.assert_allclose(after_y, before + (5.0, -5.0), atol=1e-6)


def test_a_manual_width_preserves_top_and_bottom_perspective_directions():
    """W缩放应保持上下边方向和梯形关系，同时增加平均宽度10px。"""
    paper_quad = np.float32([[120, 80], [500, 60], [530, 410], [90, 430]])
    _modules, session = _make_a_manual_session(paper_quad)
    before = np.asarray(session.settings["paper_quad"], dtype=float)
    before_width, before_height = _paper_mean_size(before)
    before_directions = _paper_edge_directions(before)

    assert session.adjust_paper_quad("W", 1) is True

    after = np.asarray(session.settings["paper_quad"], dtype=float)
    after_width, after_height = _paper_mean_size(after)
    np.testing.assert_allclose(
        _paper_edge_directions(after)[:2],
        before_directions[:2],
        atol=1e-6,
    )
    assert after_width == pytest.approx(before_width + 10.0, abs=0.01)
    # 透视梯形的上下边长度不同，沿两条斜边分别扩宽会产生亚像素级高度耦合；
    # 只要求它小于0.1px，远低于现场最小1px步进，同时严格保留上下边方向。
    assert abs(after_height - before_height) < 0.1


def test_a_manual_height_preserves_left_and_right_perspective_directions():
    """H缩放应保持左右边方向和梯形关系，同时减少平均高度10px。"""
    paper_quad = np.float32([[120, 80], [500, 60], [530, 410], [90, 430]])
    _modules, session = _make_a_manual_session(paper_quad)
    before = np.asarray(session.settings["paper_quad"], dtype=float)
    before_width, before_height = _paper_mean_size(before)
    before_directions = _paper_edge_directions(before)

    assert session.adjust_paper_quad("H", -1) is True

    after = np.asarray(session.settings["paper_quad"], dtype=float)
    after_width, after_height = _paper_mean_size(after)
    np.testing.assert_allclose(
        _paper_edge_directions(after)[2:],
        before_directions[2:],
        atol=1e-6,
    )
    assert after_height == pytest.approx(before_height - 10.0, abs=0.01)
    assert after_width == pytest.approx(before_width, abs=0.01)


def test_a_manual_step_cycles_in_both_directions():
    """STEP应在1/5/10px之间双向循环，便于快速移动后再做1px精调。"""
    _modules, session = _make_a_session()

    assert session.manual_roi_step_px == 5
    assert session.cycle_manual_step(1) == 10
    assert session.cycle_manual_step(1) == 1
    assert session.cycle_manual_step(-1) == 10
    assert session.cycle_manual_step(-1) == 5


def test_a_auto_mode_rejects_manual_geometry_without_changing_quad():
    """AUTO模式下误触X/Y/W/H必须提示切换模式，不能暗中移动蓝框。"""
    paper_quad = np.float32([[120, 80], [500, 60], [530, 410], [90, 430]])
    _modules, session = _make_a_session(paper_quad=paper_quad)
    before = session.snapshot()

    assert session.adjust_paper_quad("X", 1) is False

    assert session.settings["paper_quad"] == before["paper_quad"]
    assert session.status_text == "SWITCH MANUAL"


@pytest.mark.parametrize(
    ("paper_quad", "item", "direction"),
    [
        ([[0, 60], [220, 50], [230, 350], [0, 360]], "X", -1),
        ([[100, 0], [500, 0], [500, 300], [100, 300]], "Y", -1),
        ([[200, 100], [280, 100], [280, 320], [200, 320]], "W", -1),
        ([[200, 100], [420, 100], [420, 180], [200, 180]], "H", -1),
    ],
)
def test_a_manual_adjustment_rejects_out_of_frame_or_too_small_quad(
    paper_quad,
    item,
    direction,
):
    """越界或平均边长将小于80px时必须原子拒绝，不得留下半更新四角。"""
    _modules, session = _make_a_manual_session(paper_quad)
    before = session.snapshot()

    assert session.adjust_paper_quad(item, direction) is False

    assert session.settings["paper_quad"] == before["paper_quad"]
    assert session.status_text == "ROI LIMIT"


def test_a_roi_minus_plus_actions_switch_mode_and_move_selected_geometry(tmp_path):
    """ROI六槽应先用加号切到MANUAL，再通过中间按钮选X并移动蓝框。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    runtime = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    interface = modules.main.InterfaceState()
    interface.toggle_calibration(runtime, (640, 480))
    session = interface.calibration_session
    detection = _make_detection(1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    unchanged, message = modules.main.handle_calibration_action(
        "control_4",
        interface,
        runtime,
        detection,
        tmp_path / "settings.json",
        (640, 480),
        frame_bgr=frame,
    )
    assert unchanged is runtime
    assert message == "ROI MANUAL"
    before = np.asarray(session.settings["paper_quad"], dtype=float)

    unchanged, message = modules.main.handle_calibration_action(
        "control_3",
        interface,
        runtime,
        detection,
        tmp_path / "settings.json",
        (640, 480),
        frame_bgr=frame,
    )
    assert unchanged is runtime
    assert message == "ROI X"

    unchanged, message = modules.main.handle_calibration_action(
        "control_4",
        interface,
        runtime,
        detection,
        tmp_path / "settings.json",
        (640, 480),
        frame_bgr=frame,
    )
    assert unchanged is runtime
    assert message == "MAN X +5px"
    after = np.asarray(session.settings["paper_quad"], dtype=float)
    np.testing.assert_allclose(after, before + (5.0, 0.0), atol=1e-6)


def test_a_lock_roi_persists_current_manual_quad(tmp_path):
    """手动移动后的蓝框只有按LOCK ROI才应合并进运行设置并写入磁盘。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    runtime = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    interface = modules.main.InterfaceState()
    interface.toggle_calibration(runtime, (640, 480))
    session = interface.calibration_session
    session.set_roi_mode("MANUAL")
    assert session.adjust_paper_quad("X", 1) is True
    expected_quad = session.snapshot()["paper_quad"]

    updated, message = modules.main.handle_calibration_action(
        "control_5",
        interface,
        runtime,
        _make_detection(1),
        tmp_path / "settings.json",
        (640, 480),
        frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    assert message == "ROI LOCKED"
    assert updated["paper_quad"] == expected_quad


def test_a_roi_draws_blue_control_label_while_mask_keeps_work_mm_label(monkeypatch):
    """ROI中间槽必须显示蓝框模式，MASK中间槽仍显示黄色毫米工作区。"""
    modules, session = _make_a_session(
        paper_quad=[[120, 80], [500, 60], [530, 410], [90, 430]]
    )
    detection = _make_detection(1)
    detection.mask = np.zeros((480, 640), dtype=np.uint8)
    detection.roi = (0, 0, 640, 480)
    quality = modules.calibration.evaluate_calibration(detection)
    buttons = modules.touch.build_calibration_layout(640, 480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    drawn_labels = []

    def capture_button(_image, _button, label, active=False, enabled=True):
        """记录按钮标签和状态，避免测试依赖OpenCV字体的具体像素。"""
        drawn_labels.append((label, bool(active), bool(enabled)))

    monkeypatch.setattr(modules.calibration, "_draw_button", capture_button)

    modules.calibration.draw_calibration_frame(
        frame,
        detection,
        session,
        buttons,
        quality,
    )
    assert any(label == "MODE AUTO" for label, _active, _enabled in drawn_labels)

    drawn_labels.clear()
    session.select_view("mask")
    modules.calibration.draw_calibration_frame(
        frame,
        detection,
        session,
        buttons,
        quality,
    )
    assert any(label == "X 0.0mm" for label, _active, _enabled in drawn_labels)


def test_a_send_a4_button_is_enabled_only_after_paper_quad_exists(monkeypatch):
    """A版SEND A4仅在蓝框四角存在时启用，避免发送尚未标定的纸面。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    settings = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    session = modules.calibration.CalibrationSession(settings, (640, 480))
    detection = _make_detection(0)
    buttons = modules.touch.build_calibration_layout(640, 480)
    quality = modules.calibration.evaluate_calibration(detection)
    drawn_buttons = []

    def capture_button(_image, _button, label, active=False, enabled=True):
        """记录绘制参数，以验证SEND A4按钮的启用状态而不依赖字体像素。"""
        drawn_buttons.append((label, bool(active), bool(enabled)))

    monkeypatch.setattr(modules.calibration, "_draw_button", capture_button)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    modules.calibration.draw_calibration_frame(
        frame,
        detection,
        session,
        buttons,
        quality,
    )
    assert ("SEND A4", False, False) in drawn_buttons

    drawn_buttons.clear()
    session.settings["paper_quad"] = [
        [100.0, 60.0],
        [540.0, 60.0],
        [540.0, 420.0],
        [100.0, 420.0],
    ]
    modules.calibration.draw_calibration_frame(
        frame,
        detection,
        session,
        buttons,
        quality,
    )
    assert ("SEND A4", True, True) in drawn_buttons


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
    session.select_view("mask")

    # 默认项目为X；循环五次依次越过Y/W/H/SPLIT后应到达PAPER。
    for _ in range(5):
        session.cycle_work_item()

    assert session.current_item == "PAPER"
    assert session.adjust_work(1) is True
    assert session.settings["paper_orientation"] == "landscape"
    assert session.settings["work_x_mm"] == pytest.approx(0.0)
    assert session.settings["work_y_mm"] == pytest.approx(0.0)
    assert session.settings["work_width_mm"] == pytest.approx(297.0)
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
    assert session.settings["work_x_mm"] == pytest.approx(0.0)
    assert session.settings["work_width_mm"] == pytest.approx(297.0)
    assert session.settings["work_height_mm"] == pytest.approx(210.0)
    assert session.settings["split_y_mm"] == pytest.approx(105.0)
    assert session.status_text == "AUTO ROI OK H 92%"


def test_a_auto_roi_resets_same_orientation_region_split_and_inset_to_full_paper():
    """AUTO成功即以本次蓝框为准重置完整纸面，不能保留上次同方向的裁剪值。"""
    modules = _variant_modules("maixcam2_app_A_quad")
    saved = modules.settings.build_default_runtime_settings(modules.config.DEFAULT_CONFIG)
    saved.update(
        {
            "inset_mm": 8.0,
            "work_x_mm": 10.0,
            "work_y_mm": 30.0,
            "work_width_mm": 180.0,
            "work_height_mm": 220.0,
            "split_y_mm": 140.0,
        }
    )
    session = modules.calibration.CalibrationSession(saved, (640, 480))
    location = modules.locator.PaperLocation(
        True,
        paper_quad=np.float32([[180, 40], [460, 60], [440, 440], [160, 420]]),
        paper_orientation=modules.locator.PAPER_ORIENTATION_PORTRAIT,
        confidence=0.90,
        threshold=100.0,
        reason="ok",
    )

    assert session.apply_auto_roi(location) is True
    assert session.settings["inset_mm"] == pytest.approx(0.0)
    assert (
        session.settings["work_x_mm"],
        session.settings["work_y_mm"],
        session.settings["work_width_mm"],
        session.settings["work_height_mm"],
    ) == pytest.approx((0.0, 0.0, 210.0, 297.0))
    assert session.settings["split_y_mm"] == pytest.approx(148.5)
