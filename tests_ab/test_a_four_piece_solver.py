"""验证FOUR专用形状清理、接缝关系、分层搜索和目标位姿。"""

import math

import cv2
import numpy as np
import pytest


def _centroid(vertices):
    """计算测试多边形的面积质心。"""
    contour = np.asarray(vertices, dtype=np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    return np.asarray(
        (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
        dtype=np.float64,
    )


def _transform(vertices, angle_deg, target_center):
    """对目标多边形施加纸面刚体变换，生成随机源位姿。"""
    points = np.asarray(vertices, dtype=np.float64)
    center = _centroid(points)
    angle = math.radians(float(angle_deg))
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    return (points - center) @ rotation.T + np.asarray(target_center, dtype=np.float64)


def _piece(vertices, piece_id="U1", angle_deg=0.0, center=(50.0, 50.0)):
    """构造FOUR求解器需要的最小碎片字典。"""
    transformed = _transform(vertices, angle_deg, center)
    return {
        "id": piece_id,
        "vertices_mm": transformed.astype(float).tolist(),
        "raw_contour_mm": transformed.astype(float).tolist(),
        "center_mm": tuple(float(value) for value in _centroid(transformed)),
        "complete": True,
        "region": "upper",
    }


def test_shape_hypothesis_removes_reflection_short_edge_without_changing_piece():
    """同一直边上的3mm伪角必须清掉，但矩形面积和四条真实边保持不变。"""
    from maixcam2_app_A_quad.four_piece_solver import build_shape_hypotheses

    noisy = _piece(
        ((0, 0), (50, 0), (50, 30), (47, 30), (0, 30)),
        angle_deg=27.0,
        center=(80.0, 70.0),
    )
    hypotheses = build_shape_hypotheses(noisy)

    assert hypotheses
    best = hypotheses[0]
    assert len(best.vertices) == 4
    assert min(edge.length_mm for edge in best.edges) >= 10.0
    assert abs(best.area_mm2 - 1500.0) <= 1.0


def test_pair_relations_are_limited_ranked_rigid_transforms():
    """每个有向片对只保留固定数量最佳关系，且全部禁止镜像。"""
    from maixcam2_app_A_quad.four_piece_solver import (
        FOUR_PAIR_RELATION_LIMIT,
        build_pair_relations,
    )

    fixed = _piece(((0, 0), (40, 0), (40, 60), (0, 60)), "U1", 31.0, (55, 65))
    moving = _piece(((40, 0), (100, 0), (100, 60), (40, 60)), "U2", -48.0, (145, 72))
    relations = build_pair_relations(fixed, moving)

    assert 1 <= len(relations) <= FOUR_PAIR_RELATION_LIMIT
    assert [item.score for item in relations] == sorted(item.score for item in relations)
    assert all(np.linalg.det(item.rotation) > 0.999 for item in relations)
    assert all(np.isfinite(item.translation).all() for item in relations)


def test_build_relations_includes_segmented_long_to_short_alignment():
    """一条80mm长边与40mm短边必须产生分段候选，供后续T形连接组合。"""
    from maixcam2_app_A_quad.four_piece_solver import build_pair_relations

    fixed = _piece(((0, 0), (80, 0), (80, 30), (0, 30)), "U1", 18.0, (70, 60))
    moving = _piece(((0, 0), (40, 0), (40, 25), (0, 25)), "U2", -63.0, (150, 75))
    relations = build_pair_relations(fixed, moving)

    assert any(item.segmented for item in relations)
    assert any(item.overlap_length_mm == pytest.approx(40.0, abs=0.5) for item in relations)
