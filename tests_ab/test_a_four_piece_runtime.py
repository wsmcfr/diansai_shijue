"""验证FOUR专用轮廓提取、粘连拆分、跨帧稳定锁定和复位。"""

import cv2
import numpy as np
import pytest


PAPER_QUAD = np.float32(
    ((0.0, 0.0), (629.0, 0.0), (629.0, 890.0), (0.0, 890.0))
)


def _draw_piece_mm(frame, x_mm, y_mm, width_mm, height_mm, value=245):
    """在3像素/mm的测试相机帧中绘制一个矩形白片。"""
    x1 = int(round(x_mm * 3.0))
    y1 = int(round(y_mm * 3.0))
    x2 = int(round((x_mm + width_mm) * 3.0))
    y2 = int(round((y_mm + height_mm) * 3.0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (value, value, value), -1)


def _four_piece_frame(offset_mm=0.0):
    """生成四片均位于A4上半区且间距清晰的固定纸面帧。"""
    frame = np.zeros((891, 630, 3), dtype=np.uint8)
    positions = (
        (18.0 + offset_mm, 20.0, 34.0, 28.0),
        (68.0 + offset_mm, 22.0, 32.0, 30.0),
        (116.0 + offset_mm, 18.0, 36.0, 31.0),
        (164.0 + offset_mm, 24.0, 28.0, 26.0),
    )
    for arguments in positions:
        _draw_piece_mm(frame, *arguments)
    return frame


def test_analyze_four_piece_frame_returns_mm_centers_and_original_draw_points():
    """四片检测必须同时提供求解毫米几何和正常页所需的原相机回绘坐标。"""
    from maixcam2_app_A_quad.four_piece_vision import analyze_four_piece_frame

    detection = analyze_four_piece_frame(
        _four_piece_frame(),
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )

    assert detection.valid_contour_count == 4
    assert [piece["id"] for piece in detection.pieces] == ["U1", "U2", "U3", "U4"]
    assert detection.pieces[0]["center_mm"] == pytest.approx((35.0, 34.0), abs=0.8)
    assert len(detection.pieces[0]["vertices_mm"]) >= 3
    assert len(detection.pieces[0]["vertices"]) >= 3
    assert detection.pieces[0]["region"] == "upper"
    assert detection.pieces[0]["complete"] is True


def test_four_piece_detection_splits_only_a_support_bridge_with_four_strict_cores():
    """宽松灰桥造成3个最终连通域时，四个可靠严格核心必须受限拆回四片。"""
    from maixcam2_app_A_quad.four_piece_vision import analyze_four_piece_frame

    frame = _four_piece_frame()
    # 用宽松阈值可见、严格阈值不可见的灰桥连接前两片；真实轮廓仍有两个白核心。
    cv2.rectangle(frame, (156, 102), (204, 108), (115, 115, 115), -1)
    detection = analyze_four_piece_frame(
        frame,
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )

    assert detection.pre_split_count == 3
    assert detection.valid_contour_count == 4
    assert detection.split_applied is True


def test_real_three_piece_scene_is_not_split_to_satisfy_four_mode():
    """只有三个严格核心时必须报告3/4，不能为了满足FOUR模式任意切割轮廓。"""
    from maixcam2_app_A_quad.four_piece_vision import analyze_four_piece_frame

    frame = _four_piece_frame()
    frame[:, 450:] = 0
    detection = analyze_four_piece_frame(
        frame,
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )

    assert detection.valid_contour_count == 3
    assert detection.split_applied is False
    assert detection.reason == "count_3_of_4"


def test_runtime_locks_exact_third_stable_observation_once():
    """第三个稳定四片结果必须被冻结，后续视觉抖动不得覆盖锁定几何。"""
    from maixcam2_app_A_quad.four_piece_vision import FourPieceVisionRuntime

    runtime = FourPieceVisionRuntime(
        stable_frames=3,
        center_tolerance_mm=1.5,
        area_tolerance_ratio=0.08,
    )
    first = runtime.update(
        _four_piece_frame(0.0),
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )
    second = runtime.update(
        _four_piece_frame(0.2),
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )
    locked = runtime.update(
        _four_piece_frame(0.1),
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )
    after_lock = runtime.update(
        _four_piece_frame(8.0),
        PAPER_QUAD,
        "portrait",
        split_y_mm=148.5,
    )

    assert first.locked is False
    assert second.locked is False
    assert locked.locked is True
    assert runtime.snapshot_locked is True
    assert runtime.stable_count == 3
    assert after_lock is locked
    assert after_lock.pieces is locked.pieces


def test_runtime_unstable_frame_restarts_count_and_reset_returns_idle():
    """超过毫米抖动门应重新计数，手动reset必须释放锁定快照和历史参考。"""
    from maixcam2_app_A_quad.four_piece_vision import FourPieceVisionRuntime

    runtime = FourPieceVisionRuntime(stable_frames=3, center_tolerance_mm=1.0)
    runtime.update(_four_piece_frame(0.0), PAPER_QUAD, "portrait", 148.5)
    runtime.update(_four_piece_frame(6.0), PAPER_QUAD, "portrait", 148.5)

    assert runtime.stable_count == 1
    runtime.reset()
    assert runtime.stable_count == 0
    assert runtime.snapshot_locked is False
    assert runtime.locked_pieces == ()
