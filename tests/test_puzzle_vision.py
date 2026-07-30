"""拼图视觉核心的单元测试。"""

import cv2
import numpy as np

from tests.synthetic_images import make_black_scene


def test_default_config_limits_piece_count():
    """默认配置必须符合题目最多四片、每片三至五个顶点的约束。"""
    from maixcam2_app.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["max_pieces"] == 4
    assert DEFAULT_CONFIG["min_vertices"] == 3
    assert DEFAULT_CONFIG["max_vertices"] == 5


def test_detects_four_separated_white_pieces():
    """四个互不接触的白色多边形必须被识别为四片。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene(
        [
            [(40, 80), (140, 70), (120, 150)],
            [(190, 60), (290, 80), (270, 160), (180, 140)],
            [(340, 70), (430, 60), (460, 130), (390, 170), (330, 130)],
            [(480, 70), (580, 80), (570, 170), (500, 160)],
        ]
    )

    result = detect_pieces(scene, roi=(0, 0, 640, 240))

    assert len(result.pieces) == 4
    assert result.threshold > 0


def test_ignores_small_white_noise():
    """面积过小的白点不得被当作拼图碎片。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene([[(100, 80), (220, 80), (160, 180)]])
    cv2.circle(scene, (20, 20), 2, (255, 255, 255), -1)

    result = detect_pieces(scene, roi=(0, 0, 640, 240))

    assert len(result.pieces) == 1


def test_extracts_triangle_geometry():
    """三角碎片必须输出三个顶点、有效中心和三条正边长。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene([[(100, 80), (260, 100), (180, 220)]])

    piece = detect_pieces(scene, roi=(0, 0, 640, 300)).pieces[0]

    assert len(piece["vertices"]) == 3
    assert 150 <= piece["center"][0] <= 210
    assert 110 <= piece["center"][1] <= 170
    assert len(piece["edge_lengths"]) == 3
    assert all(length > 0 for length in piece["edge_lengths"])


def test_marks_contour_touching_roi_border_incomplete():
    """接触工作区边界的碎片必须标记为不完整。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene([[(-20, 60), (100, 60), (80, 180), (0, 170)]])

    piece = detect_pieces(scene, roi=(0, 0, 640, 240)).pieces[0]

    assert piece["complete"] is False


def test_assigns_unknown_ids_top_to_bottom_then_left_to_right():
    """未知碎片必须先按行、再按行内横向位置稳定编号。"""
    from maixcam2_app.puzzle_vision import assign_unknown_ids

    pieces = [
        {"center": (300.0, 160.0)},
        {"center": (220.0, 80.0)},
        {"center": (80.0, 80.0)},
    ]

    assign_unknown_ids(pieces, row_tolerance_px=30)

    assert [(piece["id"], piece["center"]) for piece in pieces] == [
        ("U1", (80.0, 80.0)),
        ("U2", (220.0, 80.0)),
        ("U3", (300.0, 160.0)),
    ]


def test_fixed_threshold_override_is_used_and_reported():
    """现场指定固定阈值后，算法必须使用并在结果中返回该阈值。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene([[(100, 80), (240, 80), (170, 200)]])

    result = detect_pieces(
        scene,
        roi=(0, 0, 640, 240),
        config={"fixed_threshold": 180},
    )

    assert result.threshold == 180.0
    assert len(result.pieces) == 1


def test_actionable_pieces_exclude_incomplete_border_contours():
    """编号、模板匹配和机械控制只能使用未接触ROI边界的完整碎片。"""
    from maixcam2_app.puzzle_vision import select_actionable_pieces

    pieces = [
        {"id": "edge", "complete": False},
        {"id": "valid", "complete": True},
    ]

    actionable = select_actionable_pieces(pieces)

    assert actionable == [pieces[1]]


def test_fixed_black_paper_roi_excludes_bright_outer_background():
    """固定ROI必须排除亮地面，使黑纸内四片重新成为最外层轮廓。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    cv2.rectangle(frame, (150, 20), (470, 459), (10, 10, 10), -1)
    polygons = [
        np.array([[190, 90], [300, 125], [195, 150]], dtype=np.int32),
        np.array([[330, 100], [430, 135], [410, 175], [345, 155]], dtype=np.int32),
        np.array([[210, 240], [300, 220], [320, 280], [235, 285]], dtype=np.int32),
        np.array([[350, 260], [410, 245], [430, 300], [370, 315]], dtype=np.int32),
    ]
    for polygon in polygons:
        cv2.fillPoly(frame, [polygon], (245, 245, 245))

    full_result = detect_pieces(frame, (0, 0, 640, 480))
    paper_result = detect_pieces(frame, (160, 30, 300, 420))

    assert len(full_result.pieces) == 0
    assert len(paper_result.pieces) == 4
    assert len(full_result.large_contours) >= 1


def test_detection_reports_small_large_and_edge_contours():
    """调参诊断必须区分过小、过大和接触ROI边界的轮廓。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    small_scene = make_black_scene([[(100, 70), (230, 80), (160, 190)]])
    cv2.circle(small_scene, (25, 25), 2, (255, 255, 255), -1)
    small_result = detect_pieces(small_scene, roi=(0, 0, 640, 240))

    large_scene = make_black_scene(
        [[(20, 20), (620, 20), (620, 220), (20, 220)]]
    )
    large_result = detect_pieces(large_scene, roi=(0, 0, 640, 240))

    edge_scene = make_black_scene([[(-20, 50), (120, 50), (100, 190), (0, 180)]])
    edge_result = detect_pieces(edge_scene, roi=(0, 0, 640, 240))

    assert len(small_result.small_contours) >= 1
    assert len(large_result.large_contours) == 1
    assert len(edge_result.edge_contours) == 1


def test_detection_reports_white_ratio_from_final_mask():
    """白色占比必须与形态学处理后的二值掩膜保持一致。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene([[(100, 80), (240, 80), (170, 200)]])
    result = detect_pieces(scene, roi=(0, 0, 640, 240))
    expected_ratio = float(np.count_nonzero(result.mask)) / float(result.mask.size)

    assert result.white_ratio == expected_ratio
    assert 0.0 < result.white_ratio < 1.0


def test_valid_contour_count_is_measured_before_four_piece_limit():
    """诊断数量必须保留截断前候选数，才能提示画面中存在额外噪声。"""
    from maixcam2_app.puzzle_vision import detect_pieces

    scene = make_black_scene(
        [
            [(30, 60), (100, 60), (65, 125)],
            [(145, 60), (215, 60), (180, 125)],
            [(260, 60), (330, 60), (295, 125)],
            [(375, 60), (445, 60), (410, 125)],
            [(490, 60), (560, 60), (525, 125)],
        ]
    )

    result = detect_pieces(scene, roi=(0, 0, 640, 180))

    assert result.valid_contour_count == 5
    assert len(result.pieces) == 4
