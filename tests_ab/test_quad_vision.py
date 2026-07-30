"""验证A版四边形掩膜的背景隔离、边界诊断和原图坐标输出。"""

import cv2
import numpy as np
import pytest

from maixcam2_app_A_quad.config import DEFAULT_CONFIG
from maixcam2_app_A_quad.puzzle_vision import detect_pieces
from tests_ab.synthetic_paper import DEFAULT_PAPER_QUAD, make_paper_scene
from tests_ab.synthetic_paper import make_quad_scene_with_four_pieces
from tests_ab.synthetic_paper import _active_quad_for_test, _map_physical_polygon


def _active_bounding_roi(active_quad):
    """把测试有效四边形转换为 OpenCV 外接矩形元组。"""
    return tuple(int(value) for value in cv2.boundingRect(active_quad.astype(np.int32)))


def _noisy_rectangle_contour():
    """构造带一个短凹口的真实矩形轮廓，用于复现“首个五边形不是最佳候选”。

    较小epsilon会保留凹口并得到五边形，稍大epsilon会恢复四边形。该轮廓直接使用
    有序边界点，不依赖随机噪声，确保RED/GREEN结果在不同OpenCV环境中可重复。
    """
    return np.asarray(
        (
            (20, 20),
            (120, 20),
            (120, 80),
            (72, 80),
            (69, 76),
            (66, 80),
            (20, 80),
        ),
        dtype=np.int32,
    ).reshape(-1, 1, 2)


def test_polygon_hypotheses_keep_later_clean_quadrilateral():
    """候选生成不能在首个五边形处停止，必须同时保留后续干净四边形。"""
    from maixcam2_app_A_quad import puzzle_vision

    assert hasattr(puzzle_vision, "approximate_polygon_candidates"), (
        "视觉层尚未提供多多边形候选接口"
    )
    candidates = puzzle_vision.approximate_polygon_candidates(
        _noisy_rectangle_contour(),
        DEFAULT_CONFIG,
    )
    vertex_counts = [len(candidate) for candidate in candidates]

    assert vertex_counts[0] == 5
    assert 4 in vertex_counts
    assert len(vertex_counts) == len(set(tuple(item.reshape(-1)) for item in candidates))


def test_piece_geometry_exposes_independent_pixel_hypotheses():
    """单片结果必须保留独立像素候选，后续排序不能原地修改显示主轮廓。"""
    from maixcam2_app_A_quad.puzzle_vision import compute_piece_geometry

    piece = compute_piece_geometry(
        _noisy_rectangle_contour(),
        roi=(0, 0, 160, 120),
        config=DEFAULT_CONFIG,
    )

    hypotheses = piece["shape_hypotheses_px"]
    assert [len(candidate) for candidate in hypotheses] == [5, 4]
    assert piece["vertices"] == [tuple(point) for point in hypotheses[0]]
    assert hypotheses is not piece["vertices"]


def test_quad_mask_excludes_bright_floor_and_keeps_four_pieces():
    """验证外接矩形角落的亮地面不会成为轮廓或白色占比。"""
    frame, _paper_quad, active_quad = make_quad_scene_with_four_pieces()

    result = detect_pieces(
        frame,
        roi=_active_bounding_roi(active_quad),
        config=DEFAULT_CONFIG,
        active_quad=active_quad,
    )

    assert len(result.pieces) == 4
    assert result.large_contours == []
    assert result.white_ratio < 0.30
    assert result.active_area == pytest.approx(abs(cv2.contourArea(active_quad)), rel=0.01)


def test_quad_mask_clears_pixels_outside_slanted_active_edges():
    """验证返回二值图在四边形外接矩形四角处强制为黑色。"""
    frame, _paper_quad, active_quad = make_quad_scene_with_four_pieces()
    roi = _active_bounding_roi(active_quad)

    result = detect_pieces(frame, roi=roi, config=DEFAULT_CONFIG, active_quad=active_quad)

    assert result.mask[0, 0] == 0
    assert result.mask[0, -1] == 0
    assert result.mask[-1, 0] == 0
    assert result.mask[-1, -1] == 0


def test_contour_touching_slanted_edge_is_incomplete():
    """验证白片接触有效区斜边时使用点到多边形距离判为不完整。"""
    active_quad = _active_quad_for_test(DEFAULT_PAPER_QUAD, inset_mm=0.0)
    touching_piece = _map_physical_polygon(
        [[0, 100], [35, 100], [35, 138], [0, 138]],
        DEFAULT_PAPER_QUAD,
    )
    frame = make_paper_scene(DEFAULT_PAPER_QUAD, white_pieces=[touching_piece])

    result = detect_pieces(
        frame,
        roi=_active_bounding_roi(active_quad),
        config=DEFAULT_CONFIG,
        active_quad=active_quad,
    )

    assert len(result.pieces) == 1
    assert result.pieces[0]["complete"] is False
    assert len(result.edge_contours) == 1


def test_quad_detection_keeps_global_camera_coordinates():
    """验证A版轮廓和中心仍位于原640×480相机坐标系。"""
    frame, _paper_quad, active_quad = make_quad_scene_with_four_pieces()
    roi = _active_bounding_roi(active_quad)

    result = detect_pieces(frame, roi=roi, config=DEFAULT_CONFIG, active_quad=active_quad)

    assert all(roi[0] <= piece["center"][0] <= roi[0] + roi[2] for piece in result.pieces)
    assert all(roi[1] <= piece["center"][1] <= roi[1] + roi[3] for piece in result.pieces)
    assert all(np.min(piece["contour"][:, 0, 0]) >= roi[0] for piece in result.pieces)


def test_quad_detection_rejects_invalid_active_quad():
    """验证非凸有效区不会退回矩形识别造成纸外误检。"""
    frame, _paper_quad, active_quad = make_quad_scene_with_four_pieces()
    invalid_quad = active_quad.copy()
    invalid_quad[2] = invalid_quad[0]

    with pytest.raises(ValueError, match="active_quad"):
        detect_pieces(
            frame,
            roi=_active_bounding_roi(active_quad),
            config=DEFAULT_CONFIG,
            active_quad=invalid_quad,
        )
