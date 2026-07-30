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


class _LockedVisionRuntime:
    """为组合运行器测试提供一次锁定快照，并记录是否被错误重复识别。"""

    def __init__(self, pieces):
        """保存固定四片并初始化调用和复位计数。"""
        from types import SimpleNamespace

        self.calls = 0
        self.reset_count = 0
        self.snapshot_locked = True
        self.stable_count = 3
        self.stable_frames = 3
        self.locked_pieces = tuple(pieces)
        self.last_detection = SimpleNamespace(
            pieces=self.locked_pieces,
            locked=True,
            reason="ok",
            valid_contour_count=len(self.locked_pieces),
            split_applied=False,
            pre_split_count=len(self.locked_pieces),
        )

    def update(self, *_args, **_kwargs):
        """返回同一个锁定结果；正常求解期间该方法最多应调用一次。"""
        self.calls += 1
        return self.last_detection

    def reset(self):
        """记录组合运行器是否向视觉层传播手动复位。"""
        self.reset_count += 1


def test_combined_four_runtime_starts_one_solver_and_keeps_locked_snapshot():
    """锁定后只能创建一个专用求解任务，后续帧只推进该任务而不重复视觉。"""
    from maixcam2_app_A_quad.four_piece_solver import FourPieceRuntime
    from tests_ab.test_a_four_piece_solver import _four_grid_pieces

    vision_runtime = _LockedVisionRuntime(_four_grid_pieces())
    runtime = FourPieceRuntime(vision_runtime=vision_runtime)
    common = (
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.float32(((0, 0), (19, 0), (19, 19), (0, 19))),
        "portrait",
        (0.0, 0.0, 210.0, 297.0),
        148.5,
    )

    assert runtime.update(*common) is None
    calls = 1
    while runtime.plan is None and calls < 200:
        runtime.update(*common, time_budget_ms=1000.0, work_unit_limit=64)
        calls += 1

    assert runtime.plan is not None
    assert runtime.plan.success is True
    assert runtime.solve_start_count == 1
    assert vision_runtime.calls == 1
    assert runtime.locked_pieces is vision_runtime.locked_pieces


def test_combined_four_runtime_failure_does_not_restart_detection_or_solver():
    """专用求解失败后必须保留同一失败结果，直到用户手动START触发reset。"""
    from maixcam2_app_A_quad.four_piece_solver import FourPieceRuntime
    from tests_ab.test_a_four_piece_solver import _four_grid_pieces

    vision_runtime = _LockedVisionRuntime(_four_grid_pieces(width=80.0, height=40.0))
    runtime = FourPieceRuntime(vision_runtime=vision_runtime)
    common = (
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.float32(((0, 0), (19, 0), (19, 19), (0, 19))),
        "portrait",
        (0.0, 0.0, 210.0, 297.0),
        148.5,
    )

    runtime.update(*common)
    while runtime.plan is None:
        runtime.update(*common, time_budget_ms=1000.0, work_unit_limit=128)
    failed_plan = runtime.plan
    assert failed_plan.success is False

    assert runtime.update(*common) is failed_plan
    assert vision_runtime.calls == 1
    assert runtime.solve_start_count == 1
    runtime.reset()
    assert runtime.plan is None
    assert vision_runtime.reset_count == 1
