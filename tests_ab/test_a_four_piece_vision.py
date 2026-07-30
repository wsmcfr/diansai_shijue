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


def _four_piece_mask_scene():
    """构造含2mm黑缝、灰色裸露区、内部孔和小亮点的展开纸面测试图。"""
    image = np.zeros((300, 420, 3), dtype=np.uint8)
    rectangles = (
        (30, 30, 130, 120),
        (136, 30, 236, 120),
        (30, 150, 130, 240),
        (136, 150, 236, 240),
    )
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(image, (x1, y1), (x2, y2), (245, 245, 245), -1)

    # 第一片右侧30像素模拟白纸未完全覆盖的灰色金属；它与严格白色核心直接相连。
    cv2.rectangle(image, (100, 30), (130, 120), (115, 115, 115), -1)
    # 第三片内部黑孔应由拓扑填孔恢复，但不能改变对外连通的片间黑缝。
    cv2.circle(image, (80, 195), 7, (0, 0, 0), -1)
    # 远离碎片的单像素亮点属于现场反光噪声，最终掩膜必须删除。
    image[270, 350] = (255, 255, 255)
    return image


def test_dual_mask_recovers_gray_support_without_bridging_two_mm_gap():
    """宽松支撑必须补回同片灰区，同时保持约2mm的黑色片间间隔。"""
    from maixcam2_app_A_quad.four_piece_vision import build_four_piece_masks

    masks = build_four_piece_masks(_four_piece_mask_scene(), pixels_per_mm=3.0)
    component_count, _ = cv2.connectedComponents(masks.final)

    assert component_count - 1 == 4
    assert masks.strict[70, 115] == 0
    assert masks.support[70, 115] == 255
    assert masks.final[70, 115] == 255
    # 两片横向间距为5个有效像素；任何全局膨胀都会使这里变白并导致粘连。
    assert np.all(masks.final[30:121, 131:136] == 0)


def test_dual_mask_fills_internal_hole_and_removes_isolated_bright_noise():
    """单片内部闭合孔应恢复，但没有严格核心支持的小亮点必须被删除。"""
    from maixcam2_app_A_quad.four_piece_vision import build_four_piece_masks

    masks = build_four_piece_masks(_four_piece_mask_scene(), pixels_per_mm=3.0)

    assert masks.final[195, 80] == 255
    assert masks.strict[270, 350] == 255
    assert masks.final[270, 350] == 0


@pytest.mark.parametrize(
    "invalid_image",
    (
        None,
        np.zeros((40, 40), dtype=np.uint8),
        np.zeros((40, 40, 4), dtype=np.uint8),
    ),
)
def test_dual_mask_rejects_non_bgr_input(invalid_image):
    """双掩膜入口只接受三通道BGR图，避免颜色空间静默误用。"""
    from maixcam2_app_A_quad.four_piece_vision import build_four_piece_masks

    with pytest.raises(ValueError):
        build_four_piece_masks(invalid_image, pixels_per_mm=3.0)
