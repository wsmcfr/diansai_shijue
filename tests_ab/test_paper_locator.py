"""验证自动黑纸定位、干扰拒绝和两个变体的一致性。"""

import importlib

import cv2
import numpy as np
import pytest

from tests_ab.synthetic_paper import DEFAULT_PAPER_QUAD, make_paper_scene
from tests_ab.synthetic_paper import make_scene_with_piece_count


LOCATOR_MODULES = (
    "maixcam2_app_A_quad.paper_locator",
    "maixcam2_app_B_warp.paper_locator",
)


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
@pytest.mark.parametrize("piece_count", [1, 2, 3, 4])
def test_locator_finds_a4_with_white_pieces_and_dark_rod(module_name, piece_count):
    """验证纸内有1～4片且纸外有暗杆时仍定位完整A4。"""
    module = importlib.import_module(module_name)
    scene = make_scene_with_piece_count(
        DEFAULT_PAPER_QUAD,
        piece_count,
        add_dark_rod=True,
    )

    result = module.locate_black_paper(scene)

    assert result.success is True
    assert result.confidence >= 0.65
    np.testing.assert_allclose(result.paper_quad, DEFAULT_PAPER_QUAD, atol=12)


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
def test_locator_rejects_scene_without_a4(module_name):
    """验证均匀亮背景不会生成半有效纸张候选。"""
    module = importlib.import_module(module_name)
    result = module.locate_black_paper(np.full((480, 640, 3), 210, dtype=np.uint8))

    assert result.success is False
    assert result.paper_quad is None
    assert result.reason


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
def test_locator_rejects_long_dark_rectangle(module_name):
    """验证长宽比明显错误的暗色矩形不能冒充A4纸。"""
    module = importlib.import_module(module_name)
    long_bar = np.int32([[250, 25], [295, 25], [295, 440], [250, 440]])
    scene = make_paper_scene(long_bar)

    result = module.locate_black_paper(scene)

    assert result.success is False
    assert result.confidence < 0.65


def test_locator_variants_return_equivalent_results():
    """验证A/B共享定位算法对同一帧给出等价角点、阈值和置信度。"""
    scene = make_scene_with_piece_count(DEFAULT_PAPER_QUAD, 4, add_dark_rod=True)
    module_a = importlib.import_module(LOCATOR_MODULES[0])
    module_b = importlib.import_module(LOCATOR_MODULES[1])

    result_a = module_a.locate_black_paper(scene)
    result_b = module_b.locate_black_paper(scene)

    assert result_a.success == result_b.success
    assert result_a.confidence == pytest.approx(result_b.confidence)
    assert result_a.threshold == pytest.approx(result_b.threshold)
    np.testing.assert_allclose(result_a.paper_quad, result_b.paper_quad, atol=0.01)


def test_order_a4_quad_rejects_duplicate_points():
    """验证退化四边形在进入物理映射前被明确拒绝。"""
    module = importlib.import_module(LOCATOR_MODULES[0])
    points = np.float32([[10, 10], [20, 10], [20, 10], [10, 20]])

    with pytest.raises(ValueError, match="四边形"):
        module.order_a4_quad(points)


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
def test_active_quad_crops_long_edges_and_applies_inset_mm(module_name):
    """验证完整A4上下各裁33.5mm后再对四边整体内缩2mm。"""
    module = importlib.import_module(module_name)
    paper_quad = np.float32([[100, 20], [310, 20], [310, 317], [100, 317]])

    active_quad = module.build_active_quad(paper_quad, inset_mm=2.0)

    expected = np.float32([[102, 55.5], [308, 55.5], [308, 281.5], [102, 281.5]])
    np.testing.assert_allclose(active_quad, expected, atol=1.0)


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
def test_image_point_to_paper_mm_reverses_perspective_mapping(module_name):
    """验证透视相机点可以反算回完整A4毫米坐标。"""
    module = importlib.import_module(module_name)
    paper_quad = DEFAULT_PAPER_QUAD.copy()
    source_quad = np.float32([[0, 0], [210, 0], [210, 297], [0, 297]])
    matrix = cv2.getPerspectiveTransform(source_quad, paper_quad)
    physical_point = np.float32([[[87.5, 142.25]]])
    image_point = cv2.perspectiveTransform(physical_point, matrix)[0, 0]

    recovered = module.image_point_to_paper_mm(image_point, paper_quad)

    assert recovered == pytest.approx((87.5, 142.25), abs=0.02)


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
@pytest.mark.parametrize("inset_mm", [-0.5, 20.5])
def test_active_quad_rejects_out_of_range_inset(module_name, inset_mm):
    """验证超出屏幕允许范围的INSET不会静默生成错误机械区域。"""
    module = importlib.import_module(module_name)

    with pytest.raises(ValueError, match="inset_mm"):
        module.build_active_quad(DEFAULT_PAPER_QUAD, inset_mm=inset_mm)


@pytest.mark.parametrize("module_name", LOCATOR_MODULES)
def test_physical_mapping_rejects_non_convex_quad(module_name):
    """验证交叉或非凸纸张四角在计算单应性前被拒绝。"""
    module = importlib.import_module(module_name)
    invalid_quad = np.float32([[100, 20], [310, 20], [150, 120], [100, 317]])

    with pytest.raises(ValueError, match="四边形"):
        module.build_active_quad(invalid_quad, inset_mm=0.0)


def test_a_locator_auto_detects_landscape_from_opposite_edge_averages():
    """A版AUTO ROI应根据两组对边平均长度把横放A4识别为H方向。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    landscape_quad = np.float32(
        [[72, 104], [568, 88], [536, 374], [104, 390]]
    )
    scene = make_paper_scene(landscape_quad)

    result = module.locate_black_paper(scene)

    assert result.success is True
    assert result.paper_orientation == module.PAPER_ORIENTATION_LANDSCAPE


def test_a_locator_detects_strongly_skewed_landscape_paper():
    """明显倾斜的横纸仍应依据两组对边均值识别为H方向。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    landscape_quad = np.float32(
        [[118, 82], [574, 148], [530, 408], [76, 338]]
    )
    scene = make_paper_scene(landscape_quad)

    result = module.locate_black_paper(scene)

    assert result.success is True
    assert result.paper_orientation == module.PAPER_ORIENTATION_LANDSCAPE
    assert module.default_work_region_mm(result.paper_orientation) == pytest.approx(
        (33.5, 0.0, 230.0, 210.0)
    )
    assert module.default_split_y_mm(result.paper_orientation) == pytest.approx(105.0)


def test_a_landscape_homography_round_trip_uses_297_by_210_mm():
    """横放纸面必须按297×210mm反算，不能继续把长宽当成210×297mm。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    paper_quad = np.float32([[70, 90], [570, 70], [530, 390], [110, 410]])
    physical_quad = np.float32([[0, 0], [297, 0], [297, 210], [0, 210]])
    matrix = cv2.getPerspectiveTransform(physical_quad, paper_quad)
    expected_mm = np.float32([[[231.5, 84.25]]])
    image_point = cv2.perspectiveTransform(expected_mm, matrix)[0, 0]

    recovered = module.image_point_to_paper_mm(
        image_point,
        paper_quad,
        paper_orientation=module.PAPER_ORIENTATION_LANDSCAPE,
    )

    assert recovered == pytest.approx((231.5, 84.25), abs=0.02)


def test_a_landscape_default_active_quad_trims_left_and_right():
    """横放默认机械区应左右各裁33.5mm，并完整保留210mm纸面高度。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    paper_quad = np.float32([[0, 0], [594, 0], [594, 420], [0, 420]])

    active_quad = module.build_active_quad(
        paper_quad,
        inset_mm=0.0,
        paper_orientation=module.PAPER_ORIENTATION_LANDSCAPE,
    )

    expected = np.float32([[67, 0], [527, 0], [527, 420], [67, 420]])
    np.testing.assert_allclose(active_quad, expected, atol=0.05)
    assert module.default_work_region_mm(
        module.PAPER_ORIENTATION_LANDSCAPE
    ) == pytest.approx((33.5, 0.0, 230.0, 210.0))
    assert module.default_split_y_mm(
        module.PAPER_ORIENTATION_LANDSCAPE
    ) == pytest.approx(105.0)
