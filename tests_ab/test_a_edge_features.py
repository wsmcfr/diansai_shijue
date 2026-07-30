"""验证扑克牌边缘纹理采样和反向接缝连续性评分。"""

import cv2
import numpy as np

from maixcam2_app_A_quad import main
from maixcam2_app_A_quad.config import DEFAULT_CONFIG
from maixcam2_app_A_quad.settings_store import build_default_runtime_settings
from tests_ab.synthetic_paper import make_quad_scene_with_four_pieces


def _feature(values, energy):
    """构造指定能量的一维灰度边缘特征。"""
    values = np.asarray(values, dtype=np.float64)
    return {
        "colors": np.column_stack((values, values, values)).astype(float).tolist(),
        "gradients": np.gradient(values).astype(float).tolist(),
        "pattern_energy": float(energy),
    }


def test_edge_feature_score_prefers_reversed_color_and_gradient_continuity():
    """两片反向对接时，同一图案序列的分数必须明显低于错误花纹。"""
    from maixcam2_app_A_quad.assembly_planner import edge_feature_match_score

    first = _feature([20, 50, 90, 140, 200], 0.8)
    good = _feature([200, 140, 90, 50, 20], 0.8)
    bad = _feature([220, 210, 205, 200, 195], 0.8)

    assert edge_feature_match_score(first, good) < edge_feature_match_score(first, bad)


def test_edge_feature_score_disables_texture_weight_for_plain_white_edges():
    """纯白边能量不足时接缝纹理项必须为零，未知白片只按几何求解。"""
    from maixcam2_app_A_quad.assembly_planner import edge_feature_match_score

    plain_a = _feature([245, 245, 245, 245, 245], 0.0)
    plain_b = _feature([240, 241, 240, 241, 240], 0.01)

    assert edge_feature_match_score(plain_a, plain_b) == 0.0


def test_sample_piece_edge_features_returns_one_ordered_feature_per_edge():
    """相机帧采样结果必须与多边形边索引一一对应并包含固定数量样本。"""
    from maixcam2_app_A_quad.puzzle_vision import sample_piece_edge_features

    frame = np.zeros((180, 220, 3), dtype=np.uint8)
    polygon = np.int32([[40, 35], [180, 35], [180, 145], [40, 145]])
    cv2.fillConvexPoly(frame, polygon, (245, 245, 245))
    for x_position in range(50, 180, 16):
        cv2.line(frame, (x_position, 38), (x_position, 142), (20, 20, 20), 3)

    features = sample_piece_edge_features(frame, polygon, sample_count=12)

    assert len(features) == 4
    assert all(len(feature["colors"]) == 12 for feature in features)
    assert all(len(feature["gradients"]) == 12 for feature in features)
    assert any(feature["pattern_energy"] > 0.05 for feature in features)


def test_quad_analysis_attaches_edge_features_to_every_piece():
    """锁定纸张后的A版单帧数据必须直接携带供未知求解器使用的边特征。"""
    frame, paper_quad, _ = make_quad_scene_with_four_pieces()
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["paper_quad"] = paper_quad.astype(float).tolist()

    pieces = main.analyze_quad_frame(frame, settings).detection.pieces

    assert len(pieces) == 4
    assert all(len(piece["edge_features"]) == piece["vertex_count"] for piece in pieces)
