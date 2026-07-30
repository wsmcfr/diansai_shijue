"""已知碎片模板登记、持久化和匹配测试。"""

import cv2
import numpy as np

from maixcam2_app.puzzle_vision import detect_pieces
from tests.synthetic_images import make_black_scene


def _detect_single_piece(polygon):
    """在统一测试画布中绘制并检测单片多边形，返回其几何字典。"""
    scene = make_black_scene([polygon], size=(400, 400))
    return detect_pieces(scene, roi=(0, 0, 400, 400)).pieces[0]


def _rotate_polygon(polygon, center, angle_deg):
    """绕给定中心旋转测试多边形，并返回四舍五入后的整数坐标。"""
    transform = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    points = np.asarray(polygon, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    rotated = homogeneous @ transform.T
    return np.rint(rotated).astype(np.int32).tolist()


def test_template_json_round_trip(tmp_path):
    """模板保存并重新加载后必须保持版本、编号和数值描述子。"""
    from maixcam2_app.template_store import (
        load_templates,
        register_templates,
        save_templates,
    )

    triangle = _detect_single_piece([(80, 100), (260, 110), (150, 280)])
    quadrilateral = _detect_single_piece(
        [(70, 80), (260, 90), (280, 250), (90, 270)]
    )
    templates = register_templates([triangle, quadrilateral])
    template_path = tmp_path / "known_templates.json"

    save_templates(str(template_path), templates)
    loaded = load_templates(str(template_path))

    assert loaded == templates
    assert [item["id"] for item in loaded] == ["K1", "K2"]


def test_descriptor_is_rotation_invariant():
    """同一多边形旋转后，其形状描述距离必须保持较小。"""
    from maixcam2_app.template_store import (
        build_shape_descriptor,
        descriptor_distance,
    )

    polygon = [(100, 100), (280, 120), (190, 300)]
    rotated_polygon = _rotate_polygon(polygon, center=(190, 190), angle_deg=67)
    original = build_shape_descriptor(_detect_single_piece(polygon))
    rotated = build_shape_descriptor(_detect_single_piece(rotated_polygon))

    assert descriptor_distance(rotated, original) < 0.15


def test_global_match_uses_each_template_once():
    """全局匹配必须保证每个已知模板最多分配给一个观测碎片。"""
    from maixcam2_app.template_store import (
        build_shape_descriptor,
        match_known_pieces,
    )

    triangle_polygon = [(70, 80), (250, 100), (150, 280)]
    quadrilateral_polygon = [(70, 70), (270, 90), (250, 260), (100, 280)]
    triangle = _detect_single_piece(triangle_polygon)
    quadrilateral = _detect_single_piece(quadrilateral_polygon)
    templates = [
        {"id": "K1", **build_shape_descriptor(triangle)},
        {"id": "K2", **build_shape_descriptor(quadrilateral)},
    ]
    observations = [
        _detect_single_piece(
            _rotate_polygon(quadrilateral_polygon, center=(170, 175), angle_deg=35)
        ),
        _detect_single_piece(
            _rotate_polygon(triangle_polygon, center=(160, 170), angle_deg=-42)
        ),
    ]

    matched = match_known_pieces(observations, templates, max_score=1.2)

    assert {piece["id"] for piece in matched} == {"K1", "K2"}
    assert len({piece["id"] for piece in matched}) == len(matched)


def test_known_match_rejects_different_shape_with_strict_threshold():
    """形状距离超过阈值时必须返回 UNKNOWN，避免错误编号。"""
    from maixcam2_app.template_store import (
        build_shape_descriptor,
        match_known_pieces,
    )

    triangle = _detect_single_piece([(80, 90), (270, 110), (160, 290)])
    rectangle = _detect_single_piece([(80, 100), (300, 100), (300, 220), (80, 220)])
    templates = [{"id": "K1", **build_shape_descriptor(triangle)}]

    matched = match_known_pieces([rectangle], templates, max_score=0.01)

    assert matched[0]["id"] == "UNKNOWN"
    assert matched[0]["match_score"] > 0.01
