"""验证B版透视展开后的识别、毫米中心和640×480显示画布。"""

import numpy as np
import pytest

from tests_ab.synthetic_paper import make_perspective_scene_with_four_pieces


def test_warp_variant_detects_four_pieces_and_reports_mm_centers():
    """验证透视场景拉正后识别4片并输出机械工作区毫米中心。"""
    from maixcam2_app_B_warp.main import analyze_warped_frame

    frame, paper_quad = make_perspective_scene_with_four_pieces()

    result = analyze_warped_frame(frame, paper_quad, inset_mm=2.0)

    assert result.work_frame.shape[:2] == (460, 420)
    assert result.valid_mask.shape == (460, 420)
    assert len(result.detection.pieces) == 4
    assert all("center_mm" in piece for piece in result.detection.pieces)
    assert all(0.0 < piece["center_mm"][0] < 210.0 for piece in result.detection.pieces)
    assert all(0.0 < piece["center_mm"][1] < 230.0 for piece in result.detection.pieces)


def test_warp_detection_uses_valid_mask_as_area_denominator():
    """验证B版面积和白色占比只统计INSET内部有效像素。"""
    from maixcam2_app_B_warp.main import analyze_warped_frame

    frame, paper_quad = make_perspective_scene_with_four_pieces()
    result = analyze_warped_frame(frame, paper_quad, inset_mm=10.0)

    assert result.detection.active_area == pytest.approx(
        float(np.count_nonzero(result.valid_mask))
    )
    assert result.detection.white_ratio < 0.30


def test_warp_display_canvas_preserves_210_by_230_aspect():
    """验证420×460工作图以约438×480居中显示，不拉伸为640×480。"""
    from maixcam2_app_B_warp.main import build_warp_display_canvas

    work_frame = np.full((460, 420, 3), 180, dtype=np.uint8)

    canvas, content_roi = build_warp_display_canvas(work_frame)

    assert canvas.shape == (480, 640, 3)
    assert content_roi == (101, 0, 438, 480)
    assert np.count_nonzero(canvas[:, :101]) == 0
    assert np.count_nonzero(canvas[:, 539:]) == 0
    assert np.all(canvas[:, 101:539] > 0)


def test_warp_analysis_rejects_missing_paper_quad():
    """验证B版没有锁定完整A4时不会伪造单应性并输出错误坐标。"""
    from maixcam2_app_B_warp.main import analyze_warped_frame

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="paper_quad"):
        analyze_warped_frame(frame, None, inset_mm=0.0)
