"""固定ROI调参状态机、质量判断和预览绘制测试。"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests.synthetic_images import make_black_scene


def _make_settings():
    """返回每个测试独立使用的合法现场参数。"""
    return {
        "roi": [20, 20, 500, 400],
        "fixed_threshold": None,
        "min_area_ratio": 0.002,
        "open_kernel": 3,
        "close_kernel": 5,
    }


def _make_result(
    complete_count=4,
    valid_contour_count=4,
    edge_count=0,
    large_count=0,
    small_count=0,
    white_ratio=0.10,
    vertex_count=4,
):
    """构造质量判断所需的最小检测结果替身。"""
    pieces = [
        {"complete": True, "vertex_count": vertex_count}
        for _ in range(complete_count)
    ]
    return SimpleNamespace(
        pieces=pieces,
        valid_contour_count=valid_contour_count,
        edge_contours=[object() for _ in range(edge_count)],
        large_contours=[object() for _ in range(large_count)],
        small_contours=[object() for _ in range(small_count)],
        white_ratio=white_ratio,
    )


def test_calibration_session_edits_copy_not_saved_settings():
    """一次调参会话必须编辑副本，未保存退出不能污染已保存参数。"""
    from maixcam2_app.calibration_ui import CalibrationSession

    saved = _make_settings()
    session = CalibrationSession(saved, frame_size=(640, 480))
    session.select_item("LEFT")
    changed = session.adjust(1)

    assert changed is True
    assert saved["roi"] == [20, 20, 500, 400]
    assert session.settings["roi"] == [25, 20, 495, 400]


def test_roi_edges_respect_frame_and_minimum_size():
    """ROI四条边不能越出画面，也不能把工作区压缩到安全最小尺寸以下。"""
    from maixcam2_app.calibration_ui import CalibrationSession

    settings = _make_settings()
    settings["roi"] = [0, 0, 640, 480]
    session = CalibrationSession(settings, frame_size=(640, 480))

    session.select_item("LEFT")
    assert session.adjust(-1) is False
    session.select_item("TOP")
    assert session.adjust(-1) is False
    session.select_item("RIGHT")
    assert session.adjust(1) is False
    session.select_item("BOTTOM")
    assert session.adjust(1) is False
    assert session.settings["roi"] == [0, 0, 640, 480]


def test_calibration_step_cycles_one_five_ten():
    """步长按钮必须按1、5、10循环并最终回到1。"""
    from maixcam2_app.calibration_ui import CalibrationSession

    session = CalibrationSession(_make_settings(), frame_size=(640, 480))

    assert session.step == 5
    assert session.cycle_step() == 10
    assert session.cycle_step() == 1
    assert session.cycle_step() == 5


def test_threshold_toggles_between_auto_and_fixed_current_value():
    """TH中间按钮必须能锁定当前Otsu值，并再次切回自动模式。"""
    from maixcam2_app.calibration_ui import CalibrationSession

    session = CalibrationSession(_make_settings(), frame_size=(640, 480))
    session.select_item("TH")

    session.toggle_threshold_mode(134.4)
    assert session.settings["fixed_threshold"] == 134.0
    assert session.adjust(1) is True
    assert session.settings["fixed_threshold"] == 139.0
    session.toggle_threshold_mode(120.0)
    assert session.settings["fixed_threshold"] is None
    assert session.adjust(1) is False


def test_min_area_and_kernel_parameters_follow_safe_steps():
    """最小面积按万分比调整，开闭运算核只能取允许的正奇数。"""
    from maixcam2_app.calibration_ui import CalibrationSession

    session = CalibrationSession(_make_settings(), frame_size=(640, 480))

    session.select_item("MIN")
    assert session.adjust(-1) is True
    assert session.settings["min_area_ratio"] == pytest.approx(0.0015)

    session.select_item("OPEN")
    assert session.adjust(1) is True
    assert session.settings["open_kernel"] == 5

    session.select_item("CLOSE")
    assert session.adjust(-1) is True
    assert session.settings["close_kernel"] == 3


@pytest.mark.parametrize(
    ("result", "expected_state"),
    [
        (_make_result(), "GOOD"),
        (_make_result(complete_count=3, valid_contour_count=3), "MISS"),
        (_make_result(valid_contour_count=5), "NOISE"),
        (_make_result(complete_count=3, edge_count=1), "EDGE"),
        (_make_result(complete_count=0, valid_contour_count=0, large_count=1), "BACKGROUND"),
        (_make_result(complete_count=0, valid_contour_count=0, white_ratio=0.8), "BACKGROUND"),
    ],
)
def test_calibration_quality_reports_actionable_state(result, expected_state):
    """质量判断必须按背景、边缘、噪声、漏检和正确的优先级给出状态。"""
    from maixcam2_app.calibration_ui import evaluate_calibration

    quality = evaluate_calibration(result, expected_pieces=4)

    assert quality.state == expected_state
    assert quality.expected_count == 4


def test_calibration_quality_rejects_invalid_vertex_count():
    """四个轮廓存在异常顶点数时不能显示GOOD。"""
    from maixcam2_app.calibration_ui import evaluate_calibration

    quality = evaluate_calibration(_make_result(vertex_count=7), expected_pieces=4)

    assert quality.state == "MISS"


def test_calibration_status_contains_all_decision_metrics():
    """状态栏必须同时显示质量、数量、过滤分类、阈值和白色占比。"""
    from maixcam2_app.calibration_ui import (
        evaluate_calibration,
        format_calibration_status,
    )

    result = _make_result(
        complete_count=3,
        valid_contour_count=3,
        edge_count=1,
        small_count=2,
        white_ratio=0.082,
    )
    result.threshold = 134.0
    quality = evaluate_calibration(result)

    text = format_calibration_status(result, quality)

    assert text == "EDGE 3/4 EDGE=1 SMALL=2 LARGE=0 TH=134 WHITE=8.2%"


def test_draw_calibration_views_return_new_images_without_modifying_source():
    """ROI、MASK和RESULT三个页面都必须绘制到副本而不污染相机原帧。"""
    from maixcam2_app.calibration_ui import (
        CalibrationSession,
        draw_calibration_frame,
        evaluate_calibration,
    )
    from maixcam2_app.puzzle_vision import detect_pieces
    from maixcam2_app.touch_ui import build_calibration_layout

    scene = make_black_scene(
        [
            [(100, 100), (230, 100), (170, 210)],
            [(300, 100), (430, 110), (410, 210), (315, 190)],
        ]
    )
    original = scene.copy()
    session = CalibrationSession(_make_settings(), frame_size=(640, 480))
    result = detect_pieces(scene, tuple(session.settings["roi"]))
    quality = evaluate_calibration(result)
    buttons = build_calibration_layout(640, 480)

    outputs = []
    for view in ("roi", "mask", "result"):
        session.select_view(view)
        outputs.append(
            draw_calibration_frame(scene, result, session, buttons, quality)
        )

    assert np.array_equal(scene, original)
    assert all(output is not scene for output in outputs)
    assert all(output.shape == scene.shape for output in outputs)
    assert all(np.count_nonzero(output) > 0 for output in outputs)
    # MASK页ROI外保持固定深灰色，便于观察亮背景是否已被排除。
    assert tuple(int(value) for value in outputs[1][240, 10]) == (20, 20, 20)


def test_result_preview_uses_distinct_diagnostic_colors():
    """RESULT页必须使用不同颜色显示有效、边缘、过小和过大轮廓。"""
    from maixcam2_app.calibration_ui import (
        CalibrationQuality,
        CalibrationSession,
        COLOR_EDGE,
        COLOR_LARGE,
        COLOR_SMALL,
        COLOR_VALID,
        draw_calibration_frame,
    )
    from maixcam2_app.touch_ui import build_calibration_layout

    valid_contour = np.asarray([[[100, 100]], [[180, 100]], [[140, 170]]], np.int32)
    edge_contour = np.asarray([[[25, 200]], [[100, 200]], [[60, 270]]], np.int32)
    small_contour = np.asarray([[[250, 100]], [[270, 100]], [[260, 120]]], np.int32)
    large_contour = np.asarray([[[320, 100]], [[450, 100]], [[440, 210]], [[330, 210]]], np.int32)
    result = SimpleNamespace(
        pieces=[
            {"contour": valid_contour, "complete": True},
            {"contour": edge_contour, "complete": False},
        ],
        small_contours=[small_contour],
        large_contours=[large_contour],
        edge_contours=[edge_contour],
        valid_contour_count=2,
        white_ratio=0.1,
        threshold=120.0,
        mask=np.zeros((400, 500), dtype=np.uint8),
        roi=(20, 20, 500, 400),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    session = CalibrationSession(_make_settings(), frame_size=(640, 480))
    session.select_view("result")
    output = draw_calibration_frame(
        frame,
        result,
        session,
        build_calibration_layout(640, 480),
        CalibrationQuality("EDGE", 1, 4),
    )

    for color in (COLOR_VALID, COLOR_EDGE, COLOR_SMALL, COLOR_LARGE):
        color_array = np.asarray(color, dtype=np.uint8)
        assert np.any(np.all(output == color_array, axis=2))
