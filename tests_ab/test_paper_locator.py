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


def test_a_locator_reports_oversized_dark_contour_diagnostics():
    """A版AUTO失败时必须指出黑色轮廓因超过画面50%而被拒绝。

    该场景模拟相机过近或黑纸与大面积阴影连接：黑色矩形约占整帧79%，因此不能
    作为A4候选，但诊断数据必须保留面积过大计数和本帧最大轮廓面积，供现场日志
    直接区分“看到了大黑块”和“完全没有暗色轮廓”。
    """
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    oversized_quad = np.int32([[30, 30], [610, 30], [610, 450], [30, 450]])
    scene = make_paper_scene(oversized_quad)

    result = module.locate_black_paper(scene)

    assert result.success is False
    assert result.reason == "no_candidate"
    assert result.diagnostics["contour_count"] >= 1
    assert result.diagnostics["area_large_count"] == 1
    assert result.diagnostics["eligible_count"] == 0
    assert result.diagnostics["largest_area_ratio"] == pytest.approx(
        (580.0 * 420.0) / (640.0 * 480.0),
        abs=0.02,
    )


def test_a_locator_low_confidence_keeps_best_candidate_metrics():
    """A版低置信度结果必须保留最佳四角候选的各评分分量。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    long_bar = np.int32([[250, 25], [295, 25], [295, 440], [250, 440]])

    result = module.locate_black_paper(make_paper_scene(long_bar))

    assert result.success is False
    assert result.reason == "low_confidence"
    assert result.diagnostics["eligible_count"] == 1
    best = result.diagnostics["best_candidate"]
    assert best["area_ratio"] > 0.01
    assert best["observed_aspect"] < 0.20
    assert best["aspect_score"] < 0.10
    assert best["rectangularity"] >= 0.70
    assert best["confidence"] == pytest.approx(result.confidence)


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
    """验证A版从完整A4内缩，B版继续从230mm裁剪区内缩。"""
    module = importlib.import_module(module_name)
    paper_quad = np.float32([[100, 20], [310, 20], [310, 317], [100, 317]])

    if module_name == "maixcam2_app_A_quad.paper_locator":
        active_quad = module.build_active_quad(
            paper_quad,
            inset_mm=2.0,
            camera_mount_direction="top",
        )
    else:
        active_quad = module.build_active_quad(paper_quad, inset_mm=2.0)

    if module_name == "maixcam2_app_A_quad.paper_locator":
        expected = np.float32([[102, 22], [308, 22], [308, 315], [102, 315]])
    else:
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

    if module_name == "maixcam2_app_A_quad.paper_locator":
        recovered = module.image_point_to_paper_mm(
            image_point,
            paper_quad,
            camera_mount_direction="top",
        )
    else:
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

    result = module.locate_black_paper(
        scene,
        {"camera_mount_direction": "top"},
    )

    assert result.success is True
    assert result.paper_orientation == module.PAPER_ORIENTATION_LANDSCAPE


def test_a_locator_detects_strongly_skewed_landscape_paper():
    """明显倾斜的横纸仍应依据两组对边均值识别为H方向。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    landscape_quad = np.float32(
        [[118, 82], [574, 148], [530, 408], [76, 338]]
    )
    scene = make_paper_scene(landscape_quad)

    result = module.locate_black_paper(
        scene,
        {"camera_mount_direction": "top"},
    )

    assert result.success is True
    assert result.paper_orientation == module.PAPER_ORIENTATION_LANDSCAPE
    assert module.default_work_region_mm(result.paper_orientation) == pytest.approx(
        (0.0, 0.0, 297.0, 210.0)
    )
    assert module.default_split_y_mm(result.paper_orientation) == pytest.approx(105.0)


def test_a_side_camera_auto_roi_converts_image_orientation_to_machine_orientation():
    """侧装相机AUTO ROI必须把画面横纸转换为机械纸面的V方向。

    蓝框在相机画面中长边近似水平，但固定侧装配置代表相机相对机械纸面旋转了
    90度；因此AUTO结果应为210x297mm的V方向，后续红线才能自动旋转为竖向。
    """
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    side_camera_quad = np.float32(
        [[90, 76], [220, 60], [228, 151], [92, 157]]
    )
    scene = make_paper_scene(side_camera_quad)

    result = module.locate_black_paper(scene)

    assert result.success is True
    assert result.paper_orientation == module.PAPER_ORIENTATION_PORTRAIT
    split_line = module.build_split_segment(
        result.paper_quad,
        module.default_work_region_mm(result.paper_orientation),
        module.default_split_y_mm(result.paper_orientation),
        result.paper_orientation,
    )
    delta = split_line[1] - split_line[0]
    assert abs(float(delta[1])) > abs(float(delta[0])) * 4.0


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
        camera_mount_direction="top",
    )

    assert recovered == pytest.approx((231.5, 84.25), abs=0.02)


def test_a_landscape_default_active_quad_uses_full_paper():
    """A版横放默认黄色区应等于完整297×210mm蓝框。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    paper_quad = np.float32([[0, 0], [594, 0], [594, 420], [0, 420]])

    active_quad = module.build_active_quad(
        paper_quad,
        inset_mm=0.0,
        paper_orientation=module.PAPER_ORIENTATION_LANDSCAPE,
        camera_mount_direction="top",
    )

    expected = paper_quad
    np.testing.assert_allclose(active_quad, expected, atol=0.05)
    assert module.default_work_region_mm(
        module.PAPER_ORIENTATION_LANDSCAPE
    ) == pytest.approx((0.0, 0.0, 297.0, 210.0))
    assert module.default_split_y_mm(
        module.PAPER_ORIENTATION_LANDSCAPE
    ) == pytest.approx(105.0)
