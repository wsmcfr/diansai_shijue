"""验证B版完整A4透视展开、工作区裁剪、INSET和毫米坐标。"""

import cv2
import numpy as np
import pytest

from tests_ab.synthetic_paper import DEFAULT_PAPER_QUAD, make_axis_aligned_a4_scene


def test_warp_returns_420_by_460_work_area_at_two_pixels_per_mm():
    """验证完整A4和机械工作区严格使用设计约定的固定像素尺寸。"""
    from maixcam2_app_B_warp.paper_warp import pixels_to_work_mm, warp_to_work_area

    frame, paper_quad = make_axis_aligned_a4_scene()

    result = warp_to_work_area(frame, paper_quad, inset_mm=2.0)

    assert result.full_a4.shape[:2] == (594, 420)
    assert result.work_area.shape[:2] == (460, 420)
    assert result.valid_mask.shape == (460, 420)
    assert pixels_to_work_mm((200.0, 100.0)) == pytest.approx((100.0, 50.0))


def test_warp_straightens_perspective_a4_edges():
    """验证透视四边形展开后四个标准角点保持在420×594画布边缘。"""
    from maixcam2_app_B_warp.paper_warp import build_a4_homography

    matrix = build_a4_homography(DEFAULT_PAPER_QUAD)
    mapped = cv2.perspectiveTransform(DEFAULT_PAPER_QUAD.reshape(1, 4, 2), matrix)[0]
    expected = np.float32([[0, 0], [419, 0], [419, 593], [0, 593]])

    np.testing.assert_allclose(mapped, expected, atol=0.05)


def test_warp_inset_mask_shrinks_all_four_work_edges():
    """验证2mm内缩在固定平面中转换为4像素并清除四边。"""
    from maixcam2_app_B_warp.paper_warp import warp_to_work_area

    frame, paper_quad = make_axis_aligned_a4_scene()
    result = warp_to_work_area(frame, paper_quad, inset_mm=2.0)

    assert np.count_nonzero(result.valid_mask[:4, :]) == 0
    assert np.count_nonzero(result.valid_mask[-4:, :]) == 0
    assert np.count_nonzero(result.valid_mask[:, :4]) == 0
    assert np.count_nonzero(result.valid_mask[:, -4:]) == 0
    assert result.valid_mask[10, 10] == 255


@pytest.mark.parametrize("inset_mm", [-0.5, 20.5])
def test_warp_rejects_out_of_range_inset(inset_mm):
    """验证B版与A版共同遵守0～20mm INSET范围。"""
    from maixcam2_app_B_warp.paper_warp import warp_to_work_area

    frame, paper_quad = make_axis_aligned_a4_scene()

    with pytest.raises(ValueError, match="inset_mm"):
        warp_to_work_area(frame, paper_quad, inset_mm=inset_mm)


def test_warp_rejects_singular_paper_quad():
    """验证重复角点的单应性输入不会产生黑屏结果后继续误识别。"""
    from maixcam2_app_B_warp.paper_warp import build_a4_homography

    invalid_quad = np.float32([[10, 10], [200, 10], [200, 10], [10, 300]])

    with pytest.raises(ValueError, match="四边形"):
        build_a4_homography(invalid_quad)


def test_pixels_to_work_mm_rejects_invalid_point():
    """验证毫米换算拒绝非二维点，防止机械坐标静默错位。"""
    from maixcam2_app_B_warp.paper_warp import pixels_to_work_mm

    with pytest.raises(ValueError, match="point"):
        pixels_to_work_mm((100.0,))
