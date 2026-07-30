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
