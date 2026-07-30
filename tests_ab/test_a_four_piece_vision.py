"""验证FOUR专用完整A4透视展开、毫米映射和白片分割。"""

import cv2
import numpy as np
import pytest


PORTRAIT_QUAD = np.float32(
    ((74.0, 28.0), (562.0, 54.0), (588.0, 452.0), (48.0, 426.0))
)
LANDSCAPE_QUAD = np.float32(
    ((42.0, 72.0), (596.0, 42.0), (574.0, 430.0), (62.0, 452.0))
)


@pytest.mark.parametrize(
    ("orientation", "paper_quad", "expected_shape", "sample_mm"),
    (
        ("portrait", PORTRAIT_QUAD, (891, 630), (100.0, 150.0)),
        ("landscape", LANDSCAPE_QUAD, (630, 891), (200.0, 100.0)),
    ),
)
def test_warp_full_paper_uses_orientation_aware_fixed_mm_scale(
    orientation,
    paper_quad,
    expected_shape,
    sample_mm,
):
    """横竖A4都必须按3像素/mm展开，并能无歧义换算纸面毫米坐标。"""
    from maixcam2_app_A_quad.four_piece_vision import warp_full_paper

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = warp_full_paper(
        frame,
        paper_quad,
        orientation,
        pixels_per_mm=3.0,
    )

    assert result.image.shape[:2] == expected_shape
    sample_pixel = tuple(value * 3.0 for value in sample_mm)
    assert result.pixel_to_mm(sample_pixel) == pytest.approx(sample_mm, abs=1e-6)
    assert result.mm_to_image(sample_mm) == pytest.approx(
        _paper_point_to_image(sample_mm, paper_quad, orientation),
        abs=1e-3,
    )


def _paper_point_to_image(point_mm, paper_quad, orientation):
    """用公共纸面映射生成测试期望值，避免依赖FOUR模块内部矩阵。"""
    from maixcam2_app_A_quad.paper_locator import paper_point_to_image_px

    return paper_point_to_image_px(point_mm, paper_quad, orientation)


def test_warp_full_paper_maps_a_known_bright_point_to_expected_mm_pixel():
    """倾斜相机中的已知纸面点必须落在展开图对应的毫米像素附近。"""
    from maixcam2_app_A_quad.four_piece_vision import warp_full_paper

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    source_point = _paper_point_to_image((70.0, 110.0), PORTRAIT_QUAD, "portrait")
    cv2.circle(
        frame,
        tuple(np.rint(source_point).astype(np.int32)),
        6,
        (255, 255, 255),
        -1,
    )

    result = warp_full_paper(
        frame,
        PORTRAIT_QUAD,
        "portrait",
        pixels_per_mm=3.0,
    )

    expected_x = int(round(70.0 * 3.0))
    expected_y = int(round(110.0 * 3.0))
    patch = result.image[
        expected_y - 5 : expected_y + 6,
        expected_x - 5 : expected_x + 6,
    ]
    assert int(np.max(patch)) >= 240


@pytest.mark.parametrize(
    ("paper_quad", "pixels_per_mm"),
    (
        (PORTRAIT_QUAD, 0.0),
        (PORTRAIT_QUAD, -1.0),
        (np.float32(((0, 0), (1, 1), (2, 2), (3, 3))), 3.0),
    ),
)
def test_warp_full_paper_rejects_invalid_scale_or_quad(paper_quad, pixels_per_mm):
    """无效比例或退化蓝框必须明确失败，不能产生可发送的猜测坐标。"""
    from maixcam2_app_A_quad.four_piece_vision import warp_full_paper

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        warp_full_paper(
            frame,
            paper_quad,
            "portrait",
            pixels_per_mm=pixels_per_mm,
        )
