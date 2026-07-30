"""验证A版MEASURE页面使用可量化指标判断像素密度、间隙和毫米稳定性。"""

import copy

from types import SimpleNamespace

import cv2
import numpy as np


def _measurement_result(piece_count=2):
    """构造2px/mm纸面上的检测结果，默认两片之间严格保留4像素黑缝。"""
    mask = np.zeros((594, 420), dtype=np.uint8)
    cv2.rectangle(mask, (40, 100), (180, 400), 255, -1)
    pieces = [
        {
            "complete": True,
            "center_mm": (55.0, 125.0),
            "vertices_mm": [[20, 50], [90, 50], [90, 200], [20, 200]],
        }
    ]
    if piece_count >= 2:
        cv2.rectangle(mask, (185, 100), (325, 400), 255, -1)
        pieces.append(
            {
                "complete": True,
                "center_mm": (127.5, 125.0),
                "vertices_mm": [[92.5, 50], [162.5, 50], [162.5, 200], [92.5, 200]],
            }
        )
    return SimpleNamespace(mask=mask, pieces=pieces)


def test_measurement_reports_two_pixels_per_mm_and_four_pixel_gap():
    """合法420x594像素覆盖与4列黑缝必须分别报告2px/mm和4px。"""
    from maixcam2_app_A_quad.calibration_ui import evaluate_calibration_measurement

    paper_quad = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])
    measurement = evaluate_calibration_measurement(
        _measurement_result(),
        paper_quad,
    )

    assert measurement.scale_px_per_mm == 2.0
    assert measurement.minimum_gap_px == 4.0
    assert measurement.scale_ok is True
    assert measurement.gap_ok is True


def test_scale_is_invariant_to_paper_quad_start_corner():
    """同一A4四角循环换起点后SCALE必须保持2px/mm。"""
    from maixcam2_app_A_quad.calibration_ui import evaluate_calibration_measurement

    paper_quad = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])
    scales = [
        evaluate_calibration_measurement(
            _measurement_result(),
            np.roll(paper_quad, shift, axis=0),
        ).scale_px_per_mm
        for shift in range(4)
    ]

    np.testing.assert_allclose(scales, [2.0, 2.0, 2.0, 2.0], atol=1e-6)


def test_stability_tracker_requires_ten_consistent_frames_and_reports_jitter():
    """固定碎片连续10帧轻微抖动时必须给出10/10和不超过1mm抖动。"""
    from maixcam2_app_A_quad.calibration_ui import CalibrationStabilityTracker

    tracker = CalibrationStabilityTracker(window_size=10)
    for frame_index in range(10):
        offset = 0.3 if frame_index % 2 else -0.3
        tracker.update(
            [
                {"center_mm": (40.0 + offset, 70.0)},
                {"center_mm": (120.0 - offset, 72.0)},
            ]
        )

    assert tracker.stable_frames == 10
    assert tracker.window_size == 10
    assert tracker.jitter_mm <= 0.6 + 1e-6
    assert tracker.is_stable is True


def test_stability_tracker_rejects_more_than_four_noise_pieces_without_permutations():
    """超过题目上限的噪声碎片必须重置稳定门，不能进入阶乘排列。"""
    from maixcam2_app_A_quad.calibration_ui import CalibrationStabilityTracker

    tracker = CalibrationStabilityTracker(window_size=10)
    stable_frames = tracker.update(
        [{"center_mm": (float(index), 40.0)} for index in range(9)]
    )

    assert stable_frames == 0
    assert tracker.stable_frames == 0
    assert tracker.jitter_mm is None


def test_stability_tracker_includes_vertex_motion_in_jitter():
    """中心不动但顶点明显跳动时JITTER必须失败，不能只看质心。"""
    from maixcam2_app_A_quad.calibration_ui import CalibrationStabilityTracker

    tracker = CalibrationStabilityTracker(window_size=10)
    for frame_index in range(10):
        stretch = 0.0 if frame_index % 2 == 0 else 3.0
        tracker.update(
            [
                {
                    "center_mm": (50.0, 70.0),
                    "vertices_mm": [
                        [40.0 - stretch, 60.0],
                        [60.0 + stretch, 60.0],
                        [60.0 + stretch, 80.0],
                        [40.0 - stretch, 80.0],
                    ],
                }
            ]
        )

    assert tracker.stable_frames == 10
    assert tracker.jitter_mm >= 3.0
    assert tracker.is_stable is False


def test_gap_ignores_small_mask_noise_outside_valid_piece_count():
    """GAP只测已识别的最多四片，零散白点不能制造虚假的小间隙。"""
    from maixcam2_app_A_quad.calibration_ui import evaluate_calibration_measurement

    result = _measurement_result()
    for index in range(20):
        result.mask[10 + index * 2, 10 + index * 3] = 255
    paper_quad = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])

    measurement = evaluate_calibration_measurement(result, paper_quad)

    assert measurement.minimum_gap_px == 4.0
    assert measurement.gap_ok is True


def test_gap_uses_valid_piece_contours_instead_of_large_rejected_component():
    """过大亮区不能挤掉有效碎片，GAP必须由result.pieces轮廓重建。"""
    from maixcam2_app_A_quad.calibration_ui import evaluate_calibration_measurement

    result = _measurement_result()
    cv2.rectangle(result.mask, (0, 450), (419, 593), 255, -1)
    result.roi = (0, 0, 420, 594)
    result.pieces[0]["contour"] = np.asarray(
        [[[40, 100]], [[180, 100]], [[180, 400]], [[40, 400]]],
        dtype=np.int32,
    )
    result.pieces[1]["contour"] = np.asarray(
        [[[185, 100]], [[325, 100]], [[325, 400]], [[185, 400]]],
        dtype=np.int32,
    )
    paper_quad = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])

    measurement = evaluate_calibration_measurement(result, paper_quad)

    assert measurement.minimum_gap_px == 4.0
    assert measurement.gap_ok is True


def test_standard_rectangle_measurement_accepts_100_by_60_mm_card():
    """单张100x60mm标准卡必须作为独立尺寸证据通过1.5mm误差门。"""
    from maixcam2_app_A_quad.calibration_ui import evaluate_calibration_measurement

    result = _measurement_result(piece_count=1)
    result.pieces[0]["vertices_mm"] = [[0, 0], [100, 0], [100, 60], [0, 60]]
    result.pieces[0]["center_mm"] = (50.0, 30.0)
    tracker = None
    paper_quad = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])

    measurement = evaluate_calibration_measurement(result, paper_quad, tracker)

    np.testing.assert_allclose(measurement.rectangle_size_mm, (100.0, 60.0), atol=0.01)
    assert measurement.rectangle_ok is True


def test_measurement_does_not_mutate_piece_dictionaries():
    """量化测量只能读取识别数据，不能改写机械规划仍要使用的碎片字典。"""
    from maixcam2_app_A_quad.calibration_ui import evaluate_calibration_measurement

    result = _measurement_result()
    original_pieces = copy.deepcopy(result.pieces)
    paper_quad = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])

    evaluate_calibration_measurement(result, paper_quad)

    assert result.pieces == original_pieces


def test_measure_page_renders_high_resolution_input_to_640_by_480():
    """MEASURE页面必须读取高分辨率检测数据，但最终画布保持640x480。"""
    from maixcam2_app_A_quad.calibration_ui import (
        CalibrationSession,
        draw_calibration_frame,
        evaluate_calibration,
        evaluate_calibration_measurement,
    )
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
    from maixcam2_app_A_quad.settings_store import build_default_runtime_settings
    from maixcam2_app_A_quad.touch_ui import build_calibration_layout

    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    settings = build_default_runtime_settings(DEFAULT_CONFIG, frame_size=(1280, 960))
    settings["paper_quad"] = [[220, 120], [1060, 120], [1060, 900], [220, 900]]
    session = CalibrationSession(settings, (1280, 960))
    session.select_view("measure")
    result = _measurement_result()
    result.roi = (0, 0, 420, 594)
    result.valid_contour_count = 2
    result.edge_contours = []
    result.small_contours = []
    result.large_contours = []
    result.white_ratio = 0.2
    result.threshold = 108.0
    for piece in result.pieces:
        piece["vertex_count"] = 4
    quality = evaluate_calibration(result, expected_pieces=2)
    measurement = evaluate_calibration_measurement(result, settings["paper_quad"])

    output = draw_calibration_frame(
        frame,
        result,
        session,
        build_calibration_layout(640, 480),
        quality,
        measurement=measurement,
        display_size=(640, 480),
    )

    assert output.shape == (480, 640, 3)
    assert np.count_nonzero(output) > 0


def test_result_tab_toggles_between_result_and_measure_pages():
    """A版复用RESULT顶部槽，在轮廓结果页和量化测量页之间往返切换。"""
    from maixcam2_app_A_quad.calibration_ui import CalibrationSession
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
    from maixcam2_app_A_quad.main import InterfaceState, handle_calibration_action
    from maixcam2_app_A_quad.settings_store import build_default_runtime_settings

    runtime_settings = build_default_runtime_settings(
        DEFAULT_CONFIG,
        frame_size=(1280, 960),
    )
    interface = InterfaceState()
    interface.calibration_session = CalibrationSession(
        runtime_settings,
        (1280, 960),
    )

    _unchanged, first_status = handle_calibration_action(
        "result",
        interface,
        runtime_settings,
        _measurement_result(),
        "unused.json",
        (1280, 960),
    )
    assert interface.calibration_session.view == "result"
    assert first_status == "RESULT"

    _unchanged, second_status = handle_calibration_action(
        "result",
        interface,
        runtime_settings,
        _measurement_result(),
        "unused.json",
        (1280, 960),
    )
    assert interface.calibration_session.view == "measure"
    assert second_status == "MEASURE"

    _unchanged, third_status = handle_calibration_action(
        "result",
        interface,
        runtime_settings,
        _measurement_result(),
        "unused.json",
        (1280, 960),
    )
    assert interface.calibration_session.view == "result"
    assert third_status == "RESULT"


def test_runtime_updates_expensive_measurement_only_on_measure_page():
    """ROI/MASK/RESULT/ADV页不得逐帧运行GAP测量，MEASURE页才更新稳定窗口。"""
    from maixcam2_app_A_quad.calibration_ui import CalibrationSession
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
    from maixcam2_app_A_quad.main import evaluate_active_calibration_measurement
    from maixcam2_app_A_quad.settings_store import build_default_runtime_settings

    settings = build_default_runtime_settings(
        DEFAULT_CONFIG,
        frame_size=(1280, 960),
    )
    settings["paper_quad"] = [[0, 0], [419, 0], [419, 593], [0, 593]]
    session = CalibrationSession(settings, (1280, 960))
    result = _measurement_result()

    assert evaluate_active_calibration_measurement(session, result) is None
    assert session.measurement_tracker.stable_frames == 0

    session.select_view("measure")
    measurement = evaluate_active_calibration_measurement(session, result)

    assert measurement is not None
    assert session.measurement_tracker.stable_frames == 1
