"""验证A版未知1～4片的边匹配回溯、矩形校验和搜索上限。"""

import math
import time

import cv2
import numpy as np
import pytest


def _centroid(vertices):
    """计算测试多边形的面积质心。"""
    contour = np.asarray(vertices, dtype=np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _rigid_transform(vertices, angle_deg, target_center):
    """对测试毫米多边形施加任意纸面旋转和平移。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    center = np.asarray(_centroid(vertices), dtype=np.float64)
    angle = math.radians(float(angle_deg))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return (vertices - center) @ rotation.T + np.asarray(target_center)


def _piece(vertices, piece_id, angle_deg=0.0, target_center=(80.0, 80.0)):
    """构造未知求解器只需的毫米碎片字段。"""
    transformed = _rigid_transform(vertices, angle_deg, target_center)
    return {
        "id": str(piece_id),
        "vertices_mm": transformed.astype(float).tolist(),
        "center_mm": _centroid(transformed),
        "region": "upper",
        "complete": True,
    }


def _partition_for_count(piece_count):
    """返回能无缝组成100×60mm矩形的1～4片测试分割。"""
    if piece_count == 1:
        return [((0, 0), (100, 0), (100, 60), (0, 60))]
    if piece_count == 2:
        return [
            ((0, 0), (40, 0), (40, 60), (0, 60)),
            ((40, 0), (100, 0), (100, 60), (40, 60)),
        ]
    if piece_count == 3:
        return [
            ((0, 0), (30, 0), (30, 60), (0, 60)),
            ((30, 0), (60, 0), (60, 60), (30, 60)),
            ((60, 0), (100, 0), (100, 60), (60, 60)),
        ]
    if piece_count == 4:
        return [
            ((0, 0), (50, 0), (50, 30), (0, 30)),
            ((50, 0), (100, 0), (100, 30), (50, 30)),
            ((0, 30), (50, 30), (50, 60), (0, 60)),
            ((50, 30), (100, 30), (100, 60), (50, 60)),
        ]
    raise ValueError("测试分片数量必须位于1到4")


def _patterned_irregular_four_pieces():
    """构造能够组成100×60mm矩形的四片带纹理斜四边形。

    该输入比规则四宫格拥有更多几何候选，并且每条边都超过纹理能量门槛，能够稳定
    复现旧版已经找到合法矩形后仍跑满12000节点的主循环冻结问题。
    返回值：四个已施加随机纸面位姿且带边缘特征的碎片字典。
    """
    target_layout = (
        ((0, 0), (15, 0), (20, 60), (0, 60)),
        ((15, 0), (40, 0), (35, 60), (20, 60)),
        ((40, 0), (70, 0), (75, 60), (35, 60)),
        ((70, 0), (100, 0), (100, 60), (75, 60)),
    )
    angles = (37.0, -61.0, 83.0, -24.0)
    centers = ((35, 75), (85, 78), (135, 72), (180, 80))
    pieces = [
        _piece(vertices, f"U{index + 1}", angles[index], centers[index])
        for index, vertices in enumerate(target_layout)
    ]
    for piece in pieces:
        piece["edge_features"] = [
            _pattern_feature([20, 50, 90, 140, 200, 230])
            for _ in piece["vertices_mm"]
        ]
    return pieces


def _irregular_three_pieces_like_field_mask():
    """构造接近实机MASK的两块四边形加一块三角形。

    三片在目标坐标中无缝覆盖100×60mm矩形，随后分别施加随机平面位姿。该场景
    用于防止测试只覆盖三个规则竖条，而遗漏现场常见的斜接缝和三角形碎片。
    返回值为三个可直接交给UNKNOWN求解器的碎片字典。
    """
    target_layout = (
        ((0, 0), (100, 0), (70, 30), (0, 25)),
        ((0, 25), (70, 30), (100, 60), (0, 60)),
        ((100, 0), (100, 60), (70, 30)),
    )
    angles = (31.0, -47.0, 76.0)
    centers = ((45, 78), (112, 76), (175, 72))
    return [
        _piece(vertices, f"U{index + 1}", angles[index], centers[index])
        for index, vertices in enumerate(target_layout)
    ]


def _noisy_field_three_pieces(noise_mm=1.5, seed=3):
    """给现场近似三片的每个检测顶点加入确定性毫米误差。

    seed=3在1.5mm误差下可由参考GRAPH_AUTO和A版硬验收共同接受，但旧递归搜索会
    稳定返回size_reject，因此适合作为图快路径相对旧实现的回归输入。
    """
    pieces = _irregular_three_pieces_like_field_mask()
    random = np.random.default_rng(int(seed))
    for piece in pieces:
        vertices = np.asarray(piece["vertices_mm"], dtype=np.float64)
        vertices += random.normal(0.0, float(noise_mm), size=vertices.shape)
        piece["vertices_mm"] = vertices.astype(float).tolist()
        piece["center_mm"] = tuple(float(value) for value in np.mean(vertices, axis=0))
    return pieces


def _noisy_corner_three_pieces_from_device_view():
    """构造用户实机截图中的5边形、4边形和3边形近似毫米轮廓。

    这些点来自锁定画面上的红色主角点，并按纸面显示比例换算到合理毫米尺度。
    三片存在白纸覆盖和远距离角点误差：最佳矩形填充约88.8%，但尺寸、重叠及
    每片目标外边均合法，适合验证92%严格门失败后的WHITE容错路径。
    """
    display_vertices = (
        ((96, 84), (132, 66), (148, 72), (137, 115), (102, 105)),
        ((158, 55), (189, 77), (168, 110), (162, 112)),
        ((199, 69), (224, 94), (200, 111)),
    )
    scale_mm_per_display_px = 1.2
    target_centers = ((40.0, 80.0), (100.0, 80.0), (160.0, 80.0))
    pieces = []
    for index, (vertices, target_center) in enumerate(
        zip(display_vertices, target_centers)
    ):
        points = np.asarray(vertices, dtype=np.float64)
        transformed = (
            (points - np.mean(points, axis=0)) * scale_mm_per_display_px
            + np.asarray(target_center, dtype=np.float64)
        )
        pieces.append(
            {
                "id": f"U{index + 1}",
                "vertices_mm": transformed.astype(float).tolist(),
                "center_mm": _centroid(transformed),
                "region": "upper",
                "complete": True,
            }
        )
    return pieces


def _device_short_edge_pieces():
    """返回本次MaixCAM2日志中的U2/U3毫米轮廓，用于复现伪短边。

    U2包含约6.7mm和10.4mm边，U3包含约2.3mm和10.0mm边；题目真实边不短于
    20mm，因此这些短边可作为求解副本清理的确定性回归输入。
    """
    return (
        np.asarray(
            (
                (137.3, 50.5),
                (106.7, 112.6),
                (109.1, 118.8),
                (119.0, 115.7),
                (159.4, 92.4),
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (
                (203.3, 82.0),
                (155.5, 108.6),
                (155.2, 110.9),
                (180.0, 123.4),
                (189.8, 125.4),
            ),
            dtype=np.float64,
        ),
    )


def _field_four_pieces_from_device_log():
    """返回本次四片超时日志中的原始毫米轮廓。

    该快照在旧FALLBACK中能够于第168节点形成填充95.6%、重叠1.5%的严格矩形，
    但MaixCAM2在20秒墙钟内只推进到85节点。测试保留未经人工修正的顶点，确保新的
    四片快路径真正处理现场伪短边和T形分段接缝，而不是只通过规则合成图形。
    """
    raw_pieces = (
        (
            "U1",
            (44.1, 101.2),
            ((17.0, 132.8), (37.0, 126.5), (70.7, 111.7), (74.6, 81.1), (24.4, 74.5)),
        ),
        (
            "U2",
            (114.3, 114.5),
            ((89.5, 94.3), (99.8, 131.9), (113.9, 143.1), (144.6, 100.9), (148.0, 100.3)),
        ),
        (
            "U3",
            (109.5, 62.9),
            ((141.4, 54.5), (118.6, 39.5), (67.6, 59.9), (118.1, 90.2)),
        ),
        (
            "U4",
            (179.1, 80.2),
            ((194.1, 36.6), (150.9, 69.8), (180.3, 115.5), (191.8, 128.0), (193.9, 107.2)),
        ),
    )
    return [
        {
            "id": piece_id,
            "center_mm": center_mm,
            "vertices_mm": [tuple(point) for point in vertices_mm],
            "region": "upper",
            "complete": True,
        }
        for piece_id, center_mm, vertices_mm in raw_pieces
    ]


def _cyclic_edge_lengths(vertices):
    """计算测试多边形首尾闭合后的全部边长。"""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    closed = np.vstack((vertices, vertices[:1]))
    return np.linalg.norm(np.diff(closed, axis=0), axis=1)


def test_solver_piece_uses_outline_and_ranked_shape_hypotheses():
    """UNKNOWN求解片必须分离高保真验收轮廓与三至五边接缝候选。"""
    from maixcam2_app_A_quad import assembly_planner

    outline = np.asarray(
        ((20, 20), (70, 20), (120, 20), (120, 80), (70, 80), (20, 80)),
        dtype=np.float64,
    )
    noisy_five = np.asarray(
        ((20, 20), (120, 20), (120, 80), (69, 77.5), (20, 80)),
        dtype=np.float64,
    )
    clean_four = np.asarray(
        ((20, 20), (120, 20), (120, 80), (20, 80)),
        dtype=np.float64,
    )
    piece = {
        "id": "U1",
        "center_mm": (70.0, 50.0),
        "vertices_mm": noisy_five.astype(float).tolist(),
        "outline_mm": outline.astype(float).tolist(),
        "shape_hypotheses_mm": [
            noisy_five.astype(float).tolist(),
            clean_four.astype(float).tolist(),
        ],
        "shape_edge_features": [
            [None] * len(noisy_five),
            [None] * len(clean_four),
        ],
    }

    solver_piece = assembly_planner._solver_piece(piece, 0)

    assert solver_piece["outline_local"].shape == outline.shape
    assert len(solver_piece["hypotheses"]) == 2
    assert [item["rank"] for item in solver_piece["hypotheses"]] == [0, 1]
    assert len(solver_piece["hypotheses"][0]["local_vertices"]) == 4
    assert all(
        "local_vertices" in item and "edge_lengths" in item
        for item in solver_piece["hypotheses"]
    )
    # 完整轮廓和所有候选必须使用同一个完整轮廓质心作为局部原点。
    np.testing.assert_allclose(
        np.mean(solver_piece["outline_local"], axis=0),
        (0.0, 0.0),
        atol=1e-6,
    )


def test_edge_alignment_pose_is_rigid_and_reverses_shared_edge():
    """候选接缝对齐只能旋转和平移，并使两条共享边反向重合。"""
    from maixcam2_app_A_quad import assembly_planner

    source = np.asarray(((0, 0), (40, 0), (15, 30)), dtype=np.float64)
    target = np.asarray(((80, 20), (80, 60), (45, 45)), dtype=np.float64)

    pose = assembly_planner._edge_alignment_pose(source, 0, target, 0)
    transformed = assembly_planner._transform_polygon_with_pose(source, pose)
    rotation, _translation = pose

    # 源边起点必须落到目标边终点，源边终点必须落到目标边起点。
    np.testing.assert_allclose(transformed[0], target[1], atol=1e-6)
    np.testing.assert_allclose(transformed[1], target[0], atol=1e-6)
    # 正交矩阵且行列式为+1，证明没有缩放或镜像。
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(2), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(
        _cyclic_edge_lengths(transformed),
        _cyclic_edge_lengths(source),
        atol=1e-6,
    )


def test_transform_solver_outline_applies_one_pose_to_seam_and_full_outline():
    """接缝候选与高保真完整轮廓必须共享完全相同的刚体位姿。"""
    from maixcam2_app_A_quad import assembly_planner

    seam = np.asarray(((-20, -15), (20, -15), (20, 15), (-20, 15)), dtype=np.float64)
    outline = np.asarray(
        ((-20, -15), (0, -16), (20, -15), (20, 15), (0, 16), (-20, 15)),
        dtype=np.float64,
    )
    solver_piece = {
        "outline_local": outline,
        "hypotheses": [{"rank": 0, "local_vertices": seam}],
    }
    angle = math.radians(37.0)
    pose = (
        np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
            dtype=np.float64,
        ),
        np.asarray((73.0, 41.0), dtype=np.float64),
    )

    transformed = assembly_planner._transform_solver_outline(solver_piece, 0, pose)

    np.testing.assert_allclose(
        transformed["seam_polygon"],
        assembly_planner._transform_polygon_with_pose(seam, pose),
    )
    np.testing.assert_allclose(
        transformed["outline_polygon"],
        assembly_planner._transform_polygon_with_pose(outline, pose),
    )


def test_hypothesis_edge_graph_records_rank_score_and_consistent_matching_sets():
    """边关系必须携带候选身份，连接集合不得让同一碎片混用候选。"""
    from maixcam2_app_A_quad import assembly_planner

    def solver_piece(index, lengths_by_rank):
        """构造只包含边长和候选分数的最小求解片，隔离测试边图元数据。"""
        hypotheses = []
        for rank, lengths in enumerate(lengths_by_rank):
            vertices = np.asarray(((0, 0), (lengths[0], 0), (0, 30)), dtype=np.float64)
            hypotheses.append(
                {
                    "rank": rank,
                    "local_vertices": vertices,
                    "edge_lengths": tuple(float(value) for value in lengths),
                    "edge_features": [None] * len(lengths),
                    "score": float(rank) + 0.25,
                }
            )
        return {
            "index": index,
            "hypotheses": hypotheses,
            "outline_local": hypotheses[0]["local_vertices"].copy(),
        }

    pieces = [
        solver_piece(0, ((40.0, 50.0, 30.0), (60.0, 50.0, 30.0))),
        solver_piece(1, ((40.0, 45.0, 35.0),)),
        solver_piece(2, ((60.0, 45.0, 35.0),)),
    ]
    candidates = assembly_planner._build_graph_edge_candidates(pieces, candidate_limit=32)

    assert candidates
    assert all(
        {
            "source_hypothesis",
            "target_hypothesis",
            "maximum_rank",
            "hypothesis_score",
        }.issubset(relation)
        for relation in candidates
    )

    # 两条关系分别要求U1使用候选0和候选1；它们不能组成同一个三片布局。
    conflicting = (
        {
            "relative_error": 0.0,
            "first_index": 0,
            "first_edge": 1,
            "second_index": 1,
            "second_edge": 0,
            "source_hypothesis": 0,
            "target_hypothesis": 0,
            "maximum_rank": 0,
            "hypothesis_score": 0.5,
        },
        {
            "relative_error": 0.0,
            "first_index": 0,
            "first_edge": 0,
            "second_index": 2,
            "second_edge": 0,
            "source_hypothesis": 1,
            "target_hypothesis": 0,
            "maximum_rank": 1,
            "hypothesis_score": 1.5,
        },
    )
    matching_sets = assembly_planner._collect_graph_matching_sets(3, conflicting)
    assert matching_sets == tuple()


def test_solver_state_pose_key_includes_hypothesis_identity():
    """相同刚体位姿但候选编号不同的状态不能在硬验收前被去重。"""
    from maixcam2_app_A_quad import assembly_planner

    poses = {
        0: (np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)),
        1: (np.eye(2, dtype=np.float64), np.asarray((40.0, 0.0), dtype=np.float64)),
    }

    first = assembly_planner._solver_state_pose_key(poses, {0: 0, 1: 0})
    second = assembly_planner._solver_state_pose_key(poses, {0: 0, 1: 1})

    assert first != second


def test_graph_propagation_uses_relation_hypothesis_and_transforms_outline():
    """GRAPH传播必须采用关系指定候选，同时生成接缝和完整轮廓两套布局。"""
    from maixcam2_app_A_quad import assembly_planner

    def hypothesis(rank, vertices):
        """由测试顶点构造带完整边长字段的最小形状候选。"""
        points = np.asarray(vertices, dtype=np.float64)
        return {
            "rank": rank,
            "local_vertices": points,
            "edge_lengths": tuple(float(value) for value in _cyclic_edge_lengths(points)),
            "edge_features": [None] * len(points),
            "score": float(rank),
        }

    first_outline = np.asarray(
        ((-30, -20), (0, -21), (30, -20), (30, 20), (0, 21), (-30, 20)),
        dtype=np.float64,
    )
    second_outline = np.asarray(
        ((-20, -20), (0, -21), (20, -20), (20, 20), (0, 21), (-20, 20)),
        dtype=np.float64,
    )
    solver_pieces = [
        {
            "outline_local": first_outline,
            "hypotheses": [
                hypothesis(0, ((-25, -20), (25, -20), (25, 20), (-25, 20))),
                hypothesis(1, ((-30, -20), (30, -20), (30, 20), (-30, 20))),
            ],
        },
        {
            "outline_local": second_outline,
            "hypotheses": [
                hypothesis(0, ((-20, -20), (20, -20), (20, 20), (-20, 20))),
            ],
        },
    ]
    relation = {
        "first_index": 0,
        "first_edge": 1,
        "second_index": 1,
        "second_edge": 3,
        "source_hypothesis": 1,
        "target_hypothesis": 0,
        "relative_error": 0.0,
        "maximum_rank": 1,
        "hypothesis_score": 1.0,
    }

    state, closure_error = assembly_planner._propagate_graph_layout(
        solver_pieces,
        (relation,),
    )

    assert closure_error == pytest.approx(0.0)
    assert state["hypothesis_by_index"] == {0: 1, 1: 0}
    assert set(state["pose_by_index"]) == {0, 1}
    assert len(state["seam_by_index"][0]) == 4
    assert len(state["outline_by_index"][0]) == len(first_outline)
    assert len(state["outline_by_index"][1]) == len(second_outline)
    # 关系指定的两条竖边在目标坐标中必须反向重合。
    first_seam = state["seam_by_index"][0]
    second_seam = state["seam_by_index"][1]
    np.testing.assert_allclose(first_seam[1], second_seam[0], atol=1e-6)
    np.testing.assert_allclose(first_seam[2], second_seam[3], atol=1e-6)


def _ranked_solver_strips(primary_is_correct, piece_count=3):
    """构造三块竖条完整轮廓，并为每片提供正确/错误两个已排序接缝候选。

    正确候选能恢复100x60mm矩形；错误候选是边长互不兼容的等边三角形。参数决定
    正确候选排在0还是1，用于验证分级搜索既能回退，也不会无故进入更高等级。
    返回值为可直接替换`_solver_piece`结果的三个求解片。
    """
    count = int(piece_count)
    if count == 3:
        strip_widths = (30.0, 30.0, 40.0)
    elif count == 4:
        strip_widths = (25.0, 25.0, 25.0, 25.0)
    else:
        raise ValueError("测试竖条只支持三片或四片")
    wrong_sides = (22.0, 48.0, 85.0, 130.0)[:count]
    solver_pieces = []
    left_mm = 0.0
    for index, (width_mm, wrong_side) in enumerate(zip(strip_widths, wrong_sides)):
        outline_source = np.asarray(
            (
                (left_mm, 0.0),
                (left_mm + width_mm * 0.5, -0.4),
                (left_mm + width_mm, 0.0),
                (left_mm + width_mm, 60.0),
                (left_mm + width_mm * 0.5, 60.4),
                (left_mm, 60.0),
            ),
            dtype=np.float64,
        )
        center = np.asarray((left_mm + width_mm * 0.5, 30.0), dtype=np.float64)
        outline_local = outline_source - center
        correct_vertices = np.asarray(
            (
                (-width_mm * 0.5, -30.0),
                (width_mm * 0.5, -30.0),
                (width_mm * 0.5, 30.0),
                (-width_mm * 0.5, 30.0),
            ),
            dtype=np.float64,
        )
        triangle_height = wrong_side * math.sqrt(3.0) * 0.5
        wrong_vertices = np.asarray(
            (
                (-wrong_side * 0.5, triangle_height / 3.0),
                (wrong_side * 0.5, triangle_height / 3.0),
                (0.0, -triangle_height * 2.0 / 3.0),
            ),
            dtype=np.float64,
        )

        def make_hypothesis(rank, vertices, score):
            """为测试候选补齐搜索器依赖的边长、纹理和质量字段。"""
            return {
                "rank": int(rank),
                "local_vertices": vertices.copy(),
                "source_vertices": vertices + center,
                "edge_lengths": tuple(
                    float(value) for value in _cyclic_edge_lengths(vertices)
                ),
                "edge_features": [None] * len(vertices),
                "score": float(score),
            }

        ordered_vertices = (
            (correct_vertices, wrong_vertices)
            if primary_is_correct
            else (wrong_vertices, correct_vertices)
        )
        hypotheses = [
            make_hypothesis(rank, vertices, 0.1 + rank)
            for rank, vertices in enumerate(ordered_vertices)
        ]
        solver_pieces.append(
            {
                "index": index,
                "id": f"U{index + 1}",
                "source_center": tuple(float(value) for value in center),
                "outline_source": outline_source,
                "outline_local": outline_local,
                "hypotheses": hypotheses,
                # 兼容字段故意指向排名0；搜索成功不能依赖这些字段偷换候选。
                "source_vertices": hypotheses[0]["source_vertices"],
                "local_vertices": hypotheses[0]["local_vertices"],
                "edge_lengths": hypotheses[0]["edge_lengths"],
                "edge_features": hypotheses[0]["edge_features"],
            }
        )
        left_mm += width_mm
    return solver_pieces


def test_graph_uses_rank1_when_rank0_edges_cannot_form_layout(monkeypatch):
    """排名0关系无解时，GRAPH必须在同一任务中使用排名1恢复完整轮廓矩形。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_pieces = _ranked_solver_strips(primary_is_correct=False)

    def prepared_solver_piece(_piece, index):
        """返回预制候选，使测试只观察GRAPH的候选分级和完整轮廓验收。"""
        return solver_pieces[index]

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    plan, diagnostics = assembly_planner._solve_unknown_graph_fast_path(
        [{"id": "raw"}] * 3,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan is not None and plan.success is True
    assert diagnostics["maximum_hypothesis_rank"] == 1
    assert diagnostics["selected_hypotheses"] == (1, 1, 1)


def test_graph_rank0_success_does_not_enter_higher_hypothesis(monkeypatch):
    """三片排名0已经通过硬验收时，GRAPH不得继续检查或采用更高候选。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_pieces = _ranked_solver_strips(primary_is_correct=True)

    def prepared_solver_piece(_piece, index):
        """返回首选即正确的预制候选，用于验证早停等级。"""
        return solver_pieces[index]

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    plan, diagnostics = assembly_planner._solve_unknown_graph_fast_path(
        [{"id": "raw"}] * 3,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan is not None and plan.success is True
    assert diagnostics["maximum_hypothesis_rank"] == 0
    assert diagnostics["selected_hypotheses"] == (0, 0, 0)


def test_fallback_hard_acceptance_receives_full_outline(monkeypatch):
    """FALLBACK最终尺寸、填充和重叠验收必须使用完整轮廓。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_piece = _ranked_solver_strips(primary_is_correct=True)[0]
    observed_vertex_counts = []

    def prepared_solver_piece(_piece, _index):
        """返回含4点接缝候选和6点完整轮廓的单片求解结构。"""
        return solver_piece

    def observe_canonicalize(placed_by_index, **_kwargs):
        """记录硬验收收到的顶点数，并以尺寸失败结束该搜索。"""
        observed_vertex_counts.append(len(placed_by_index[0]))
        return None, "size_reject"

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    monkeypatch.setattr(
        assembly_planner,
        "_canonicalize_complete_layout",
        observe_canonicalize,
    )
    plan = assembly_planner._consume_solver_steps(
        assembly_planner._solve_unknown_layout_steps(
            [{"id": "raw"}],
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            stop_at_first_solution=True,
        )
    )

    assert plan.success is False
    assert observed_vertex_counts
    assert all(
        vertex_count == len(solver_piece["outline_local"])
        for vertex_count in observed_vertex_counts
    )


def test_four_fast_hard_acceptance_receives_full_outlines(monkeypatch):
    """FOURFAST中间状态与最终硬验收必须传递四片高保真完整轮廓。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_pieces = _ranked_solver_strips(primary_is_correct=True, piece_count=4)
    observed_vertex_counts = []

    def prepared_solver_piece(_piece, index):
        """返回每片4点接缝候选和6点完整轮廓的预制结构。"""
        return solver_pieces[index]

    def observe_canonicalize(placed_by_index, **_kwargs):
        """记录完整状态顶点数，并用尺寸失败让快路径继续受控结束。"""
        observed_vertex_counts.append(
            tuple(len(placed_by_index[index]) for index in sorted(placed_by_index))
        )
        return None, "size_reject"

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    monkeypatch.setattr(
        assembly_planner,
        "_canonicalize_complete_layout",
        observe_canonicalize,
    )
    _plan, _diagnostics = assembly_planner._solve_unknown_four_fast_path(
        [{"id": "raw"}] * 4,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_work_units=assembly_planner.UNKNOWN_FOUR_FAST_MAX_WORK_UNITS,
    )

    assert observed_vertex_counts
    assert all(vertex_counts == (6, 6, 6, 6) for vertex_counts in observed_vertex_counts)


def test_four_fast_opens_rank1_within_same_work_budget(monkeypatch):
    """四片排名0无解时，FOURFAST必须在同一累计工作预算内采用排名1。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_pieces = _ranked_solver_strips(primary_is_correct=False, piece_count=4)

    def prepared_solver_piece(_piece, index):
        """返回排名1才具有正确接缝的四片预制结构。"""
        return solver_pieces[index]

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    plan, diagnostics = assembly_planner._solve_unknown_four_fast_path(
        [{"id": "raw"}] * 4,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_work_units=assembly_planner.UNKNOWN_FOUR_FAST_MAX_WORK_UNITS,
    )

    assert plan is not None and plan.success is True
    assert (
        diagnostics["four_work_units"]
        <= assembly_planner.UNKNOWN_FOUR_FAST_MAX_WORK_UNITS
    )
    assert diagnostics["maximum_hypothesis_rank"] == 1
    assert diagnostics["selected_hypotheses"] == (1, 1, 1, 1)


def test_fallback_can_select_rank1_for_root_and_remaining_pieces(monkeypatch):
    """FALLBACK不能把第0片锁死为候选0，排名1正确时三片仍须成功。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_pieces = _ranked_solver_strips(primary_is_correct=False)

    def prepared_solver_piece(_piece, index):
        """返回根片和其余碎片均需候选1的三片预制结构。"""
        return solver_pieces[index]

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    plan = assembly_planner._consume_solver_steps(
        assembly_planner._solve_unknown_layout_steps(
            [{"id": "raw"}] * 3,
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            stop_at_first_solution=True,
            max_nodes=12000,
        )
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert plan.diagnostics["maximum_hypothesis_rank"] == 1
    assert plan.diagnostics["selected_hypotheses"] == (1, 1, 1)


def test_fallback_reports_shape_hypothesis_empty(monkeypatch):
    """候选质量门清空任一碎片时必须保留专用失败原因。"""
    from maixcam2_app_A_quad import assembly_planner

    def reject_all_hypotheses(_piece, _index):
        """模拟候选评分器因伪短边或面积保持不足而拒绝全部候选。"""
        raise ValueError("shape_hypothesis_empty")

    monkeypatch.setattr(assembly_planner, "_solver_piece", reject_all_hypotheses)
    plan = assembly_planner._consume_solver_steps(
        assembly_planner._solve_unknown_layout_steps(
            [{"id": "raw"}],
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            stop_at_first_solution=True,
        )
    )

    assert plan.reason == "shape_hypothesis_empty"


def test_fallback_reports_edge_graph_empty(monkeypatch):
    """候选存在但任意片对都没有兼容边时必须返回边图为空。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_pieces = _ranked_solver_strips(primary_is_correct=True)[:2]

    def prepared_solver_piece(_piece, index):
        """返回几何有效求解片，确保失败只来自空边图。"""
        return solver_pieces[index]

    def empty_edge_graph(pieces, base_tolerance_mm):
        """为全部有向片对返回空关系，模拟没有长度兼容接缝。"""
        del base_tolerance_mm
        return {
            (source_index, target_index): tuple()
            for source_index in range(len(pieces))
            for target_index in range(len(pieces))
            if source_index != target_index
        }

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    monkeypatch.setattr(
        assembly_planner,
        "_build_edge_compatibility_graph",
        empty_edge_graph,
    )
    plan = assembly_planner._consume_solver_steps(
        assembly_planner._solve_unknown_layout_steps(
            [{"id": "raw"}] * 2,
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            stop_at_first_solution=True,
        )
    )

    assert plan.reason == "edge_graph_empty"


@pytest.mark.parametrize(
    ("internal_reason", "public_reason"),
    (
        ("size_reject", "layout_size"),
        ("fill_reject", "layout_fill"),
        ("overlap_reject", "layout_overlap"),
    ),
)
def test_fallback_maps_dominant_hard_gate_failure(
    monkeypatch,
    internal_reason,
    public_reason,
):
    """完整候选失败时必须把内部硬门名称映射为稳定现场原因。"""
    from maixcam2_app_A_quad import assembly_planner

    solver_piece = _ranked_solver_strips(primary_is_correct=True)[0]

    def prepared_solver_piece(_piece, _index):
        """返回可直接进入单片完整验收的有效候选结构。"""
        return solver_piece

    def reject_layout(_placed_by_index, **_kwargs):
        """用参数化内部原因拒绝每个完整轮廓候选。"""
        return None, internal_reason

    monkeypatch.setattr(assembly_planner, "_solver_piece", prepared_solver_piece)
    monkeypatch.setattr(
        assembly_planner,
        "_canonicalize_complete_layout",
        reject_layout,
    )
    plan = assembly_planner._consume_solver_steps(
        assembly_planner._solve_unknown_layout_steps(
            [{"id": "raw"}],
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            stop_at_first_solution=False,
        )
    )

    assert plan.reason == public_reason


def test_white_solver_cleanup_macro_and_device_short_edges():
    """默认12mm宏必须清除实机U2/U3伪短边，且不修改输入数组。"""
    from maixcam2_app_A_quad import assembly_planner

    assert assembly_planner.UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM == pytest.approx(12.0)
    for original in _device_short_edge_pieces():
        before = original.copy()

        cleaned, cleanup = assembly_planner._clean_solver_short_edges(
            original,
            min_edge_mm=assembly_planner.UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM,
        )

        assert np.array_equal(original, before)
        assert cleaned is not original
        assert 3 <= len(cleaned) < len(original)
        assert cleanup["removed_count"] == len(original) - len(cleaned)
        assert cleanup["original_min_edge_mm"] < 12.0
        assert np.min(_cyclic_edge_lengths(cleaned)) >= 12.0 - 1e-6


def test_white_solver_cleanup_preserves_real_pentagon_and_zero_disables():
    """真实边均超过20mm的五边形不得变化，门槛0必须完全关闭清理。"""
    from maixcam2_app_A_quad import assembly_planner

    real_pentagon = np.asarray(
        ((0, 0), (35, 0), (55, 25), (35, 55), (0, 40)),
        dtype=np.float64,
    )
    noisy = _device_short_edge_pieces()[0]

    preserved, preserved_cleanup = assembly_planner._clean_solver_short_edges(
        real_pentagon,
        min_edge_mm=12.0,
    )
    disabled, disabled_cleanup = assembly_planner._clean_solver_short_edges(
        noisy,
        min_edge_mm=0.0,
    )

    np.testing.assert_allclose(preserved, real_pentagon)
    assert preserved_cleanup["removed_count"] == 0
    np.testing.assert_allclose(disabled, noisy)
    assert disabled_cleanup["removed_count"] == 0


def test_solver_piece_legacy_input_preserves_features_without_mutation():
    """旧单候选夹具仍可运行，统一构造器不得修改输入顶点或错配边特征。"""
    from maixcam2_app_A_quad import assembly_planner

    vertices = _device_short_edge_pieces()[0]
    piece = {
        "id": "U2",
        "vertices_mm": vertices.astype(float).tolist(),
        "center_mm": _centroid(vertices),
        "edge_features": [
            {"pattern_energy": float(index + 1)} for index in range(len(vertices))
        ],
    }
    before = tuple(tuple(point) for point in piece["vertices_mm"])

    solver_piece = assembly_planner._solver_piece(piece, 0)

    np.testing.assert_allclose(solver_piece["source_vertices"], vertices)
    assert solver_piece["edge_features"] == piece["edge_features"]
    assert len(solver_piece["hypotheses"]) == 1
    assert tuple(tuple(point) for point in piece["vertices_mm"]) == before


def test_graph_and_both_fallback_profiles_share_solver_piece_builder(monkeypatch):
    """GRAPH、WHITE兜底和CARD兜底必须调用同一个无清理模式分支的几何构造器。"""
    from maixcam2_app_A_quad import assembly_planner

    calls = []
    original_solver_piece = assembly_planner._solver_piece

    def recording_solver_piece(piece, index):
        """记录三条求解路径是否都只传公共的碎片与索引参数。"""
        calls.append(int(index))
        return original_solver_piece(piece, index)

    monkeypatch.setattr(assembly_planner, "_solver_piece", recording_solver_piece)
    pieces = [
        _piece(
            ((0, 0), (50, 0), (50, 60), (0, 60)),
            "U1",
            target_center=(55, 80),
        ),
        _piece(
            ((50, 0), (100, 0), (100, 60), (50, 60)),
            "U2",
            target_center=(130, 80),
        ),
    ]

    assembly_planner._solve_unknown_graph_fast_path(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )
    assert calls == [0, 1]

    calls.clear()
    white_steps = assembly_planner._solve_unknown_layout_steps(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
    )
    list(white_steps)
    assert calls == [0, 1]

    calls.clear()
    card_steps = assembly_planner._solve_unknown_layout_steps(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=False,
    )
    list(card_steps)
    assert calls == [0, 1]


@pytest.mark.parametrize("piece_count", [1, 2, 3, 4])
def test_unknown_solver_recovers_rectangle_from_random_piece_poses(piece_count):
    """未知碎片初始位置和角度随机时必须恢复目标范围内矩形并放到下半区。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    angles = (37.0, -61.0, 83.0, -24.0)
    centers = ((35, 75), (85, 78), (135, 72), (180, 80))
    pieces = [
        _piece(vertices, f"U{index + 1}", angles[index], centers[index])
        for index, vertices in enumerate(_partition_for_count(piece_count))
    ]

    plan = solve_unknown_layout(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=12000,
    )

    assert plan.success is True
    assert plan.reason == "ok"
    assert len(plan.placements) == piece_count
    x_mm, y_mm, width_mm, height_mm = plan.target_rect_mm
    assert 88.0 <= width_mm <= 122.0
    assert 48.0 <= height_mm <= 92.0
    assert y_mm >= 148.5
    assert x_mm >= 0.0
    assert plan.search_nodes <= 12000
    if piece_count == 4:
        # 白片一旦得到已通过矩形验收的组合就应快速停止，不能继续跑满节点上限。
        assert plan.search_nodes < 2000


def test_unknown_solver_recovers_irregular_three_piece_field_shape():
    """两块四边形加一块三角形也必须快速恢复矩形，不能只支持规则竖条。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    plan = solve_unknown_layout(
        _irregular_three_pieces_like_field_mask(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=12000,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert len(plan.placements) == 3
    assert plan.search_nodes < 100


def test_white_relaxed_fill_recovers_noisy_corner_device_shape():
    """WHITE严格门无解时应以逐片外边约束接受实机近似三片。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    plan = solve_unknown_layout(
        _noisy_corner_three_pieces_from_device_view(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
        active_timeout_seconds=60.0,
        wall_timeout_seconds=60.0,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert len(plan.placements) == 3
    assert plan.diagnostics["relaxed_accept"] == 1
    assert plan.diagnostics["outer_piece_count"] == 3
    assert plan.diagnostics["fill_permille"] >= 860
    assert plan.search_nodes < 150


def test_card_keeps_strict_fill_for_noisy_corner_device_shape():
    """CARD不得借用WHITE的86%容错门，避免低填充错误拼法绕过花纹比较。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    plan = solve_unknown_layout(
        _noisy_corner_three_pieces_from_device_view(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=False,
        active_timeout_seconds=60.0,
        wall_timeout_seconds=60.0,
    )

    assert plan.success is False
    assert plan.reason == "layout_fill"


def test_relaxed_fill_rejects_piece_without_target_outer_edge():
    """86%填充率候选中只要有一片没有目标外边，WHITE也必须拒绝。

    四片合计覆盖100x60mm目标的约86.7%，其中U4完全位于矩形内部。该构造用于证明
    容错层不是简单降低填充率，而是同时执行题目给出的“每片至少一条外边”硬约束。
    """
    from maixcam2_app_A_quad.assembly_planner import (
        UNKNOWN_RELAXED_MIN_FILL_RATIO,
        _canonicalize_complete_layout,
    )

    placed_by_index = {
        0: np.asarray(((0, 0), (100, 0), (100, 20), (0, 20)), dtype=float),
        1: np.asarray(((0, 40), (100, 40), (100, 60), (0, 60)), dtype=float),
        2: np.asarray(((0, 20), (40, 20), (40, 40), (0, 40)), dtype=float),
        3: np.asarray(((40, 20), (60, 20), (60, 40), (40, 40)), dtype=float),
    }

    relaxed_result, relaxed_reason = _canonicalize_complete_layout(
        placed_by_index,
        min_fill_ratio=UNKNOWN_RELAXED_MIN_FILL_RATIO,
        require_all_outer_edges=True,
    )

    assert relaxed_result is None
    assert relaxed_reason == "outer_edge_reject"


def test_white_graph_fast_path_is_invariant_after_skewed_paper_homography():
    """倾斜相机中的碎片顶点反算到纸面毫米后必须得到同一个GRAPH规划。

    主要流程：把同一组三片物理轮廓分别投影到近正视和明显倾斜的A4像素四边形，
    再通过生产Homography批量反算为毫米输入。两个规划都必须走WHITE图快路径，且
    目标矩形尺寸一致；这能防止求解器错误使用未经校正的相机像素长度。
    """
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout
    from maixcam2_app_A_quad.paper_locator import (
        image_points_to_paper_mm,
        paper_points_to_image_px,
    )

    physical_pieces = _irregular_three_pieces_like_field_mask()
    paper_quads = (
        np.float32(((180, 70), (1020, 75), (1040, 910), (170, 900))),
        np.float32(((300, 80), (1040, 180), (900, 900), (120, 780))),
    )
    plans = []
    for paper_quad in paper_quads:
        recovered_pieces = []
        for piece in physical_pieces:
            image_vertices = paper_points_to_image_px(
                piece["vertices_mm"],
                paper_quad,
            )
            recovered_vertices = image_points_to_paper_mm(
                image_vertices,
                paper_quad,
            )
            recovered_piece = dict(piece)
            recovered_piece["vertices_mm"] = recovered_vertices.astype(float).tolist()
            recovered_piece["center_mm"] = _centroid(recovered_vertices)
            recovered_pieces.append(recovered_piece)

        plans.append(
            solve_unknown_layout(
                recovered_pieces,
                work_region_mm=(0.0, 33.5, 210.0, 230.0),
                split_y_mm=148.5,
                stop_at_first_solution=True,
            )
        )

    assert all(plan.success for plan in plans)
    assert all(plan.diagnostics["graph_fast_path"] == 1 for plan in plans)
    np.testing.assert_allclose(
        plans[0].target_rect_mm[2:],
        plans[1].target_rect_mm[2:],
        atol=0.05,
    )


def test_graph_fast_path_recovers_noisy_three_piece_field_shape():
    """1.5mm顶点误差三片必须由连接图快路径恢复，不能落入旧搜索size_reject。"""
    from maixcam2_app_A_quad.assembly_planner import _solve_unknown_graph_fast_path

    plan, diagnostics = _solve_unknown_graph_fast_path(
        _noisy_field_three_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan is not None
    assert plan.success is True
    assert len(plan.placements) == 3
    assert diagnostics["graph_fast_path"] == 1


def test_graph_fast_path_keeps_candidate_and_matching_sets_bounded():
    """复杂四片的边候选和连通组合必须受32/90硬上限约束。"""
    from maixcam2_app_A_quad.assembly_planner import _solve_unknown_graph_fast_path

    _plan, diagnostics = _solve_unknown_graph_fast_path(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert 0 < diagnostics["graph_edge_candidates"] <= 32
    assert 0 <= diagnostics["graph_matching_sets"] <= 90
    assert diagnostics["graph_layouts_checked"] <= diagnostics["graph_matching_sets"]


def test_four_fast_path_solves_device_log_with_segmented_seam():
    """四片快路径必须在有限工作量内解决本次实机超时快照。

    该输入的正确布局包含短边贴到长边局部区间的分段接缝，因此诊断还必须证明成功
    路径至少使用一条segmented关系，防止实现退化成扩大版整边GRAPH。
    """
    from maixcam2_app_A_quad import assembly_planner

    plan, diagnostics = assembly_planner._solve_unknown_four_fast_path(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan is not None, diagnostics
    assert plan.success is True
    assert len(plan.placements) == 4
    assert diagnostics["four_fast_path"] == 1
    assert diagnostics["four_used_segmented"] >= 1
    assert diagnostics["four_pair_states"] > 0
    assert diagnostics["four_triple_states"] > 0
    assert diagnostics["four_complete_states"] > 0
    assert diagnostics["four_work_units"] <= assembly_planner.UNKNOWN_FOUR_FAST_MAX_WORK_UNITS
    assert plan.diagnostics["fill_permille"] >= 920
    assert plan.diagnostics["overlap_permille"] <= 30


def test_four_fast_expands_every_retained_beam_parent():
    """四片快路径必须展开每层实际保留的全部父状态。

    Beam宽度表示下一层允许继续搜索的状态数量，不能在保存32个状态后只展开前8个；
    诊断分别记录三层父状态数，使现场召回下降时能够区分“没有候选”和“候选未展开”。
    """
    from maixcam2_app_A_quad import assembly_planner

    _plan, diagnostics = assembly_planner._solve_unknown_four_fast_path(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        beam_width=32,
        # 该测试只验证Beam展开语义，因此给足工作量；默认2400上限的中途触顶行为
        # 由后续独立测试验证，不能把两种条件混在同一个断言中。
        max_work_units=10000,
    )

    assert diagnostics["four_pair_parents_expanded"] == 1
    assert diagnostics["four_triple_parents_expanded"] == min(
        diagnostics["four_pair_states"],
        32,
    )
    assert diagnostics["four_complete_parents_expanded"] == min(
        diagnostics["four_triple_states"],
        32,
    )


def test_four_fast_parent_diagnostics_count_started_parents_at_work_limit():
    """工作量中途触顶时，诊断只能统计真正开始展开过的父状态。

    现场调试需要区分“保留了32个三片父状态”和“实际只来得及展开其中一部分”。
    500个工作单元会在三片层展开途中触顶，因此计数必须小于该层保留数。
    """
    from maixcam2_app_A_quad import assembly_planner

    _plan, diagnostics = assembly_planner._solve_unknown_four_fast_path(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        beam_width=32,
        max_work_units=500,
    )

    retained_triple_parents = min(diagnostics["four_pair_states"], 32)
    assert diagnostics["four_work_limit_reached"] == 1
    assert 0 < diagnostics["four_triple_parents_expanded"] < retained_triple_parents


def test_fast_overlap_uses_geometry_for_convex_polygons(monkeypatch):
    """凸多边形必须用几何交集区分分离、共享边和实体重叠。

    把旧栅格函数替换为直接抛错可证明三个凸多边形场景都没有偷偷创建MASK；共享边
    只有零面积接触，应允许作为拼图接缝，实体交叠超过1%时必须拒绝。
    """
    from maixcam2_app_A_quad import assembly_planner

    def forbidden_raster(*args, **kwargs):
        """凸多边形路径若错误回退栅格则立即使测试失败。"""
        del args, kwargs
        raise AssertionError("凸多边形快速重叠判断不应创建栅格MASK")

    monkeypatch.setattr(assembly_planner, "_candidate_overlaps", forbidden_raster)
    placed = np.asarray(((0, 0), (40, 0), (40, 40), (0, 40)), dtype=float)
    separated = np.asarray(((50, 0), (90, 0), (90, 40), (50, 40)), dtype=float)
    touching = np.asarray(((40, 0), (80, 0), (80, 40), (40, 40)), dtype=float)
    overlapping = np.asarray(((20, 0), (60, 0), (60, 40), (20, 40)), dtype=float)

    diagnostics = {}
    assert assembly_planner._candidate_overlaps_fast(
        separated,
        [placed],
        diagnostics=diagnostics,
    ) is False
    assert assembly_planner._candidate_overlaps_fast(
        touching,
        [placed],
        diagnostics=diagnostics,
    ) is False
    assert assembly_planner._candidate_overlaps_fast(
        overlapping,
        [placed],
        diagnostics=diagnostics,
    ) is True
    # 共享边可在AABB阶段直接判定，实体相交至少会执行一次凸多边形求交。
    assert diagnostics["fast_convex_checks"] >= 1
    assert diagnostics.get("fast_raster_fallbacks", 0) == 0


def test_fast_overlap_keeps_shallow_overlap_for_final_three_percent_gate(monkeypatch):
    """中间候选约2%浅交叠必须留给最终3%总重叠硬门判断。

    现场角点误差会让正确接缝产生约1.5%几何交叠；若快速层继续使用旧MASK腐蚀后的
    1%门，精确三角化反而会误删正确路径。明显10%实体重叠仍必须立即拒绝。
    """
    from maixcam2_app_A_quad import assembly_planner

    def forbidden_raster(*args, **kwargs):
        """本测试全为凸矩形，不允许走栅格回退。"""
        del args, kwargs
        raise AssertionError("凸矩形不应创建栅格MASK")

    monkeypatch.setattr(assembly_planner, "_candidate_overlaps", forbidden_raster)
    placed = np.asarray(((0, 0), (100, 0), (100, 40), (0, 40)), dtype=float)
    shallow = np.asarray(((98, 0), (198, 0), (198, 40), (98, 40)), dtype=float)
    excessive = np.asarray(((90, 0), (190, 0), (190, 40), (90, 40)), dtype=float)

    assert assembly_planner._candidate_overlaps_fast(shallow, [placed]) is False
    assert assembly_planner._candidate_overlaps_fast(excessive, [placed]) is True


def test_fast_overlap_preserves_layout_allowed_by_final_total_overlap_gate(monkeypatch):
    """中间门不得按单片分母误删最终总重叠低于3%的合法四片布局。

    最后一条25x60mm碎片向左偏1.01mm时，其自身约4.04%面积发生重叠，但完整四片
    布局按生产栅格计算仅重叠约2.51%。快速层应保留该候选并交给最终硬门决定。
    """
    from maixcam2_app_A_quad import assembly_planner

    def forbidden_raster(*args, **kwargs):
        """规则矩形必须走快速几何路径，避免测试结果依赖中间栅格腐蚀。"""
        del args, kwargs
        raise AssertionError("凸矩形中间预筛不应创建栅格MASK")

    monkeypatch.setattr(assembly_planner, "_candidate_overlaps", forbidden_raster)
    placed_by_index = {
        0: np.asarray(((0, 0), (25, 0), (25, 60), (0, 60)), dtype=float),
        1: np.asarray(((25, 0), (50, 0), (50, 60), (25, 60)), dtype=float),
        2: np.asarray(((50, 0), (75, 0), (75, 60), (50, 60)), dtype=float),
    }
    candidate = np.asarray(
        ((73.99, 0), (98.99, 0), (98.99, 60), (73.99, 60)),
        dtype=float,
    )

    assert assembly_planner._candidate_overlaps_fast(
        candidate,
        list(placed_by_index.values()),
    ) is False
    placed_by_index[3] = candidate
    metrics = {}
    canonical_result, rejection_reason = assembly_planner._canonicalize_complete_layout(
        placed_by_index,
        metrics=metrics,
    )
    assert canonical_result is not None, rejection_reason
    assert metrics["overlap_ratio"] <= assembly_planner.UNKNOWN_MAX_OVERLAP_RATIO


def test_fast_overlap_uses_all_piece_area_during_early_pair_stage(monkeypatch):
    """两片阶段必须用最终全部碎片面积估算重叠率，避免提前误剪合法布局。

    前两片存在1.6mm视觉交叠，按当前两片面积会超过4%；加入另外两片后仍组成
    100x60mm完整矩形，并能通过最终3%栅格硬门。快速预筛只能保留该分支。
    """
    from maixcam2_app_A_quad import assembly_planner

    def forbidden_raster(*args, **kwargs):
        """规则凸矩形必须使用几何交集，不能让测试依赖MASK量化。"""
        del args, kwargs
        raise AssertionError("凸矩形中间预筛不应创建栅格MASK")

    monkeypatch.setattr(assembly_planner, "_candidate_overlaps", forbidden_raster)
    first = np.asarray(((0, 0), (18, 0), (18, 60), (0, 60)), dtype=float)
    second = np.asarray(((16.4, 0), (34.4, 0), (34.4, 60), (16.4, 60)), dtype=float)
    third = np.asarray(((34.4, 0), (64.4, 0), (64.4, 60), (34.4, 60)), dtype=float)
    fourth = np.asarray(((64.4, 0), (100, 0), (100, 60), (64.4, 60)), dtype=float)
    final_total_piece_area = sum(
        abs(float(cv2.contourArea(polygon.astype(np.float32))))
        for polygon in (first, second, third, fourth)
    )

    assert assembly_planner._candidate_overlaps_fast(
        second,
        [first],
        final_total_piece_area=final_total_piece_area,
    ) is False

    complete_layout = {0: first, 1: second, 2: third, 3: fourth}
    metrics = {}
    canonical_result, rejection_reason = assembly_planner._canonicalize_complete_layout(
        complete_layout,
        metrics=metrics,
    )
    assert canonical_result is not None, rejection_reason
    assert metrics["overlap_ratio"] <= assembly_planner.UNKNOWN_MAX_OVERLAP_RATIO


def test_fast_overlap_triangulates_simple_concave_polygon(monkeypatch):
    """简单凹多边形必须三角化后做几何求交，不能为每个候选创建MASK。"""
    from maixcam2_app_A_quad import assembly_planner

    def forbidden_raster(*args, **kwargs):
        """有效简单凹多边形若回退栅格则立即使测试失败。"""
        del args, kwargs
        raise AssertionError("简单凹多边形应使用三角化几何求交")

    monkeypatch.setattr(assembly_planner, "_candidate_overlaps", forbidden_raster)
    placed = np.asarray(((0, 0), (40, 0), (40, 40), (0, 40)), dtype=float)
    overlapping_concave = np.asarray(
        ((10, 10), (50, 10), (30, 20), (50, 40), (10, 40)),
        dtype=float,
    )
    separated_concave = overlapping_concave + np.asarray((60.0, 0.0))
    diagnostics = {}

    assert assembly_planner._candidate_overlaps_fast(
        overlapping_concave,
        [placed],
        pixels_per_mm=1.5,
        diagnostics=diagnostics,
    ) is True
    assert assembly_planner._candidate_overlaps_fast(
        separated_concave,
        [placed],
        pixels_per_mm=1.5,
        diagnostics=diagnostics,
    ) is False
    assert diagnostics["fast_triangle_checks"] > 0
    assert diagnostics.get("fast_raster_fallbacks", 0) == 0


def test_fast_overlap_self_intersection_falls_back_without_raising(monkeypatch):
    """自交或零有向面积轮廓必须安全回退栅格，不能中断整个UNKNOWN任务。"""
    from maixcam2_app_A_quad import assembly_planner

    raster_calls = []

    def recording_raster(candidate, placed, **kwargs):
        """记录异常轮廓是否进入保守栅格路径，并返回无重叠结果。"""
        raster_calls.append((np.asarray(candidate).copy(), len(placed), dict(kwargs)))
        return False

    monkeypatch.setattr(assembly_planner, "_candidate_overlaps", recording_raster)
    bow_tie = np.asarray(((0, 0), (40, 40), (0, 40), (40, 0)), dtype=float)
    separated = np.asarray(((60, 0), (100, 0), (100, 40), (60, 40)), dtype=float)
    diagnostics = {}

    assert assembly_planner._candidate_overlaps_fast(
        bow_tie,
        [separated],
        diagnostics=diagnostics,
    ) is False
    assert len(raster_calls) == 1
    assert diagnostics["fast_raster_fallbacks"] == 1


def test_graph_fast_path_rejects_non_rectangular_triangles():
    """宽松边候选只能生成假设，非矩形输入不得绕过A版毫米硬验收。"""
    from maixcam2_app_A_quad.assembly_planner import _solve_unknown_graph_fast_path

    pieces = [
        _piece(((0, 0), (30, 0), (15, 28)), "U1", 17, (70, 75)),
        _piece(((0, 0), (42, 0), (28, 24)), "U2", -33, (145, 78)),
    ]

    plan, diagnostics = _solve_unknown_graph_fast_path(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan is None
    assert diagnostics["graph_fast_path"] == 0
    assert diagnostics["graph_layouts_checked"] <= 90


def test_white_graph_fallback_preserves_generator_piece_input(monkeypatch):
    """图路径消费输入后必须把同一碎片列表交给旧搜索，不能让生成器变成空集合。"""
    from maixcam2_app_A_quad import assembly_planner

    def force_graph_fallback(*args, **kwargs):
        """强制进入原搜索，以验证同步入口对一次性可迭代对象的所有权。"""
        # 真实图路径会把pieces物化并遍历；替身也必须消费输入才能复现一次性生成器问题。
        list(args[0])
        del args, kwargs
        return None, {"graph_fast_path": 0}

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        force_graph_fallback,
    )
    piece_generator = (
        piece for piece in _irregular_three_pieces_like_field_mask()
    )

    plan = assembly_planner.solve_unknown_layout(
        piece_generator,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
    )

    assert plan.success is True, plan.reason
    assert len(plan.placements) == 3


def test_white_profile_stops_at_first_valid_solution_even_with_reflective_texture():
    """WHITE显式模式必须忽略反光纹理，在首个合法矩形节点立即停止。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    plan = solve_unknown_layout(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        texture_refinement_nodes=400,
        stop_at_first_solution=True,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert plan.diagnostics["first_solution_node"] == plan.search_nodes


def test_unknown_solver_returns_bounded_failure_for_non_rectangular_set():
    """无法组成矩形的碎片必须在节点硬上限内失败，不能阻塞相机主循环。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    pieces = [
        _piece(((0, 0), (30, 0), (15, 28)), "U1", 17, (70, 75)),
        _piece(((0, 0), (42, 0), (28, 24)), "U2", -33, (145, 78)),
    ]

    plan = solve_unknown_layout(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=80,
    )

    assert plan.success is False
    assert plan.reason in {
        "edge_graph_empty",
        "layout_size",
        "layout_fill",
        "layout_overlap",
        "solver_timeout",
    }
    assert plan.search_nodes <= 80
    assert plan.placements == []


def test_unknown_solver_rejects_more_than_four_pieces():
    """未知题最多四片，超出时必须直接拒绝而不是扩大排列搜索。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    pieces = [
        _piece(((0, 0), (20, 0), (20, 20), (0, 20)), f"U{index}")
        for index in range(5)
    ]

    plan = solve_unknown_layout(
        pieces,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "unknown_piece_count"
    assert plan.search_nodes == 0


def test_unknown_solve_job_yields_after_one_work_unit_then_finishes():
    """增量任务每帧只推进一个工作单元时必须立即让出，后续推进仍能得到规划。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    job = UnknownSolveJob(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=12000,
        texture_refinement_nodes=80,
    )

    first_result = job.advance(time_budget_ms=1000.0, work_unit_limit=1)

    assert first_result is None
    assert job.done is False
    assert job.result is None

    result = None
    for _ in range(2000):
        result = job.advance(time_budget_ms=1000.0, work_unit_limit=32)
        if result is not None:
            break

    assert result is not None
    assert result.success is True
    assert job.done is True
    assert job.result is result


def test_patterned_unknown_solver_stops_after_finite_texture_refinement():
    """带纹理首解只允许有限节点择优，不能为全局纹理最优再次跑满12000节点。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    plan = solve_unknown_layout(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=12000,
        texture_refinement_nodes=80,
    )

    assert plan.success is True
    assert plan.search_nodes < 500


def test_unknown_solver_never_calls_expensive_contact_length_scan(monkeypatch):
    """快速搜索不得再对每个候选扫描全部共线边接触长度。

    该扫描只影响旧搜索顺序，不影响最终矩形硬验收，却是实测最大的累计热点。
    用直接抛错的替身可保证生产求解路径已经彻底脱离它，而不是只减少调用次数。
    """
    from maixcam2_app_A_quad import assembly_planner

    def reject_contact_scan(*args, **kwargs):
        """旧热点一旦仍被调用就立即使回归测试失败。"""
        del args, kwargs
        raise AssertionError("UNKNOWN求解仍调用逐候选接触长度扫描")

    monkeypatch.setattr(
        assembly_planner,
        "_candidate_contact_length",
        reject_contact_scan,
    )

    plan = assembly_planner.solve_unknown_layout(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        texture_refinement_nodes=80,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)


def test_unknown_edge_graph_contains_full_and_segmented_compatibility():
    """预计算边图必须保留普通整边和题目允许的T形分段接缝。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests.test_assembly_planner import _make_segmented_seam_pieces

    regular_solver_pieces = [
        assembly_planner._solver_piece(piece, index)
        for index, piece in enumerate(
            [
                _piece(((0, 0), (40, 0), (40, 60), (0, 60)), "U1"),
                _piece(((40, 0), (100, 0), (100, 60), (40, 60)), "U2"),
            ]
        )
    ]
    regular_graph = assembly_planner._build_edge_compatibility_graph(
        regular_solver_pieces,
        base_tolerance_mm=2.5,
    )
    assert any(
        relation["kind"] == "full"
        for relations in regular_graph.values()
        for relation in relations
    )

    segmented_solver_pieces = [
        assembly_planner._solver_piece(piece, index)
        for index, piece in enumerate(_make_segmented_seam_pieces())
    ]
    segmented_graph = assembly_planner._build_edge_compatibility_graph(
        segmented_solver_pieces,
        base_tolerance_mm=2.5,
    )
    # U1的31mm短边可与U3的110mm长边组成分段接缝，剩余37mm和42mm由U2/U4补齐。
    assert any(
        relation["kind"] == "segmented"
        for relation in segmented_graph[(0, 2)]
    )


def test_segmented_edge_graph_does_not_score_unaligned_whole_edge_texture():
    """分段接缝不能把整条长边纹理误当成实际子段纹理参与限宽排序。

    一个短边可以锚定长边两端，实际对应的纹理区间不同；当前边图没有携带子段区间，
    因此最安全的行为是仅用几何排序，不能给两个锚定候选共用一个伪纹理分数。
    """
    from maixcam2_app_A_quad import assembly_planner
    from tests.test_assembly_planner import _make_segmented_seam_pieces

    pieces = _make_segmented_seam_pieces()
    for piece_index, piece in enumerate(pieces):
        values = (
            [10, 30, 70, 120, 180, 240]
            if piece_index == 0
            else [245, 210, 170, 110, 50, 5]
        )
        piece["edge_features"] = [
            _pattern_feature(values) for _ in piece["vertices_mm"]
        ]
    solver_pieces = [
        assembly_planner._solver_piece(piece, index)
        for index, piece in enumerate(pieces)
    ]

    graph = assembly_planner._build_edge_compatibility_graph(
        solver_pieces,
        base_tolerance_mm=2.5,
    )
    segmented_relations = [
        relation
        for relations in graph.values()
        for relation in relations
        if relation["kind"] == "segmented"
    ]

    assert segmented_relations
    assert all(relation["texture_score"] == 0.0 for relation in segmented_relations)


def test_unknown_solver_limits_frontier_and_reports_finite_edge_graph():
    """四片搜索每层最多保留96个状态，并公开有限边候选数供现场诊断。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    plan = solve_unknown_layout(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        texture_refinement_nodes=80,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert plan.diagnostics["max_frontier_width"] <= 96
    # 四个四边形最多只有4*3*4*4=192个有向边对，边图不能在搜索中继续膨胀。
    assert 0 < plan.diagnostics["edge_candidates"] <= 192


def test_complex_unknown_four_piece_pc_solve_is_under_100ms():
    """复杂带纹理四片在PC上必须低于100ms，为MaixCAM2留出足够计算余量。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    elapsed_samples = []
    for _ in range(3):
        started_at = time.perf_counter()
        plan = solve_unknown_layout(
            _patterned_irregular_four_pieces(),
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            texture_refinement_nodes=80,
        )
        elapsed_samples.append(time.perf_counter() - started_at)
        assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)

    # 取三次最小值排除Windows调度偶发抢占；算法稳定慢路径仍会在每次运行中出现。
    assert min(elapsed_samples) < 0.100, elapsed_samples


def test_unknown_solve_job_has_five_second_total_wall_clock_timeout():
    """任务从创建起超过5秒必须结构化超时，不能在主循环中无限跨帧等待。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    current_time = [100.0]

    def fake_clock():
        """返回测试可控的单调时钟秒数。"""
        return current_time[0]

    job = UnknownSolveJob(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        timeout_seconds=5.0,
        clock=fake_clock,
    )
    current_time[0] = 105.01

    result = job.advance(time_budget_ms=12.0, work_unit_limit=32)

    assert result is not None
    assert result.success is False
    assert result.reason == "solver_timeout"
    assert result.placements == []
    assert job.done is True


def test_unknown_job_frame_wait_does_not_consume_active_compute_budget():
    """帧间拍照显示等待不能消耗5秒活动计算预算，只受20秒硬墙钟保护。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    current_time = [0.0]

    def fake_clock():
        """返回由测试控制、工作单元内部保持不变的单调秒数。"""
        return current_time[0]

    def pending_units():
        """持续让出轻量工作单元，模拟尚未完成但本身没有耗时的搜索。"""
        while True:
            yield None

    job = UnknownSolveJob(
        _irregular_three_pieces_like_field_mask(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        active_timeout_seconds=5.0,
        wall_timeout_seconds=20.0,
        clock=fake_clock,
    )
    job._iterator = pending_units()

    assert job.advance(time_budget_ms=12.0, work_unit_limit=1) is None
    current_time[0] = 10.0
    assert job.advance(time_budget_ms=12.0, work_unit_limit=1) is None
    assert job.done is False
    assert job.active_elapsed_ms == 0


def test_unknown_job_stops_when_active_compute_budget_is_exhausted():
    """工作单元真实消耗超过5秒时必须按活动预算停止，不能依赖帧间墙钟。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    current_time = [40.0]

    def fake_clock():
        """返回由慢工作单元推进的单调秒数。"""
        return current_time[0]

    def slow_unit():
        """模拟一个底层OpenCV调用独占5.1秒后才让出。"""
        current_time[0] = 45.1
        yield None

    job = UnknownSolveJob(
        _irregular_three_pieces_like_field_mask(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        active_timeout_seconds=5.0,
        wall_timeout_seconds=20.0,
        clock=fake_clock,
    )
    job._iterator = slow_unit()

    result = job.advance(time_budget_ms=12.0, work_unit_limit=1)

    assert result.success is False
    assert result.reason == "solver_timeout"
    assert result.diagnostics["active_elapsed_ms"] == 5100
    assert result.diagnostics["wall_elapsed_ms"] == 5100


def test_unknown_job_hard_deadline_returns_existing_best_plan():
    """CARD达到硬墙钟时若已有合法矩形，必须返回当前最优解而不是丢弃。"""
    from maixcam2_app_A_quad.assembly_planner import (
        UnknownSolveJob,
        solve_unknown_layout,
    )

    best_plan = solve_unknown_layout(
        _irregular_three_pieces_like_field_mask(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        timeout_seconds=60.0,
    )
    assert best_plan.success is True

    current_time = [100.0]

    def fake_clock():
        """返回测试控制的总墙钟。"""
        return current_time[0]

    job = UnknownSolveJob(
        _irregular_three_pieces_like_field_mask(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        active_timeout_seconds=5.0,
        wall_timeout_seconds=20.0,
        clock=fake_clock,
    )
    job._progress["best_plan"] = best_plan
    current_time[0] = 120.1

    result = job.advance(time_budget_ms=12.0, work_unit_limit=1)

    assert result.success is True
    assert len(result.placements) == 3
    assert result.diagnostics["returned_best_at_timeout"] == 1
    assert result.diagnostics["wall_elapsed_ms"] == 20100


def test_white_fast_plan_finishing_after_deadline_is_preserved(monkeypatch):
    """真实GRAPH核心在慢验收末尾形成合法解时，超时收尾必须返回该解。

    GRAPH生产生成器会先形成规划、再yield工作单元、最后才return。测试让真实硬验收
    跨过活动截止线，证明规划必须在yield前写入共享best_plan，而不能依赖流水线收尾。
    """
    from maixcam2_app_A_quad import assembly_planner

    current_time = [10.0]
    original_canonicalize = assembly_planner._canonicalize_complete_layout

    def fake_clock():
        """返回由慢GRAPH工作单元推进的测试时钟。"""
        return current_time[0]

    def slow_canonicalize(*args, **kwargs):
        """让真实GRAPH的不可抢占栅格验收在返回合法结果时越过截止线。"""
        current_time[0] = 15.1
        return original_canonicalize(*args, **kwargs)

    monkeypatch.setattr(
        assembly_planner,
        "_canonicalize_complete_layout",
        slow_canonicalize,
    )
    job = assembly_planner.UnknownSolveJob(
        [_piece(((0, 0), (100, 0), (100, 60), (0, 60)), "U1")],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
        active_timeout_seconds=5.0,
        wall_timeout_seconds=20.0,
        clock=fake_clock,
    )

    result = job.advance(time_budget_ms=12.0, work_unit_limit=1)

    assert result.success is True
    assert result.diagnostics["graph_fast_path"] == 1
    assert result.diagnostics["returned_best_at_timeout"] == 1


def test_white_stage_event_yields_before_starting_next_fast_stage(monkeypatch):
    """GRAPH事件入队后必须结束当前advance，下一帧才能启动FOUR_FAST。

    这样GRAPH日志的活动耗时不会混入下一阶段，也保证一个24ms时间片不会跨阶段继续
    执行。测试同时覆盖流水线yield边界和UnknownSolveJob的事件感知停止逻辑。
    """
    from maixcam2_app_A_quad import assembly_planner

    four_stage_started = []

    def no_graph_solution(*args, **kwargs):
        """同步返回GRAPH失败，便于精确观察事件边界。"""
        del args, kwargs
        return None, {"graph_fast_path": 0, "graph_layouts_checked": 1}

    def pending_four_stage(*args, **kwargs):
        """记录FOUR_FAST第一次真正恢复，并保持任务未完成。"""
        del args, kwargs

        def stage():
            """每次恢复都让出一个工作单元，直到测试关闭任务。"""
            four_stage_started.append(True)
            while True:
                yield None

        return stage()

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        pending_four_stage,
    )
    job = assembly_planner.UnknownSolveJob(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
    )

    assert job.advance(time_budget_ms=1000.0, work_unit_limit=64) is None
    events = job.consume_stage_events()
    assert [event["source"] for event in events] == ["graph"]
    assert four_stage_started == []
    job.cancel()


def test_four_fast_active_budget_aborts_only_fast_stage_and_continues_fallback(
    monkeypatch,
):
    """四片快路径耗尽独立活动预算后必须进入FALLBACK，不能结束整个任务。

    测试使用真实FOURFAST生成器，只把快速重叠工作单元的假时钟推进0.6秒。两个
    工作单元累计1.2秒后应触发1秒子预算，生成FOURFAST失败事件，并在下一帧启动兜底。
    """
    from maixcam2_app_A_quad import assembly_planner

    current_time = [100.0]
    original_fast_overlap = assembly_planner._candidate_overlaps_fast
    fallback_started = []

    def fake_clock():
        """返回由测试工作单元推进的单调秒数。"""
        return current_time[0]

    def no_graph_solution(*args, **kwargs):
        """同步结束GRAPH，使测试直接观察四片子预算。"""
        del args, kwargs
        return None, {"graph_fast_path": 0}

    def slow_fast_overlap(*args, **kwargs):
        """保持真实几何判断，只让每个候选消耗0.6秒假活动时间。"""
        current_time[0] += 0.6
        return original_fast_overlap(*args, **kwargs)

    def pending_fallback(*args, **kwargs):
        """记录子预算结束后旧FALLBACK确实从下一帧启动。"""
        del args, kwargs
        fallback_started.append(True)
        while True:
            yield None

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_candidate_overlaps_fast",
        slow_fast_overlap,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_layout_steps",
        pending_fallback,
    )
    job = assembly_planner.UnknownSolveJob(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
        active_timeout_seconds=8.0,
        wall_timeout_seconds=20.0,
        four_fast_active_budget_seconds=1.0,
        clock=fake_clock,
    )

    # 第一帧只输出GRAPH失败；第二帧由真实FOURFAST在子预算边界正常结束。
    assert job.advance(time_budget_ms=5000.0, work_unit_limit=64) is None
    assert [event["source"] for event in job.consume_stage_events()] == ["graph"]
    assert job.advance(time_budget_ms=5000.0, work_unit_limit=64) is None
    four_events = job.consume_stage_events()

    assert len(four_events) == 1
    assert four_events[0]["source"] == "four_fast"
    assert four_events[0]["plan"] is None
    assert four_events[0]["diagnostics"]["four_active_limit_reached"] == 1
    assert four_events[0]["diagnostics"]["four_active_elapsed_ms"] >= 1200
    assert job.done is False
    assert job.active_elapsed_ms < 8000
    assert fallback_started == []

    assert job.advance(time_budget_ms=1000.0, work_unit_limit=1) is None
    assert fallback_started == [True]
    assert job.result_source == "fallback"
    job.cancel()


def test_three_piece_path_ignores_four_fast_active_budget(monkeypatch):
    """三片必须完全绕过FOURFAST，极小四片子预算也不能改变原FALLBACK结果。"""
    from maixcam2_app_A_quad import assembly_planner

    current_time = [200.0]
    expected_plan = assembly_planner.AssemblyPlan(
        True,
        placements=[],
        target_rect_mm=(55.0, 176.0, 100.0, 60.0),
        reason="ok",
        diagnostics={"fallback_test": 1},
    )

    def fake_clock():
        """返回三片FALLBACK工作单元使用的假时钟。"""
        return current_time[0]

    def no_graph_solution(*args, **kwargs):
        """让三片进入它原有的FALLBACK路径。"""
        del args, kwargs
        return None, {"graph_fast_path": 0}

    def forbidden_four_fast(*args, **kwargs):
        """三片若错误进入四片快路径则立即失败。"""
        del args, kwargs
        raise AssertionError("三片不得调用FOURFAST")

    def successful_fallback(*args, **kwargs):
        """模拟三片FALLBACK消耗0.2秒后返回合法规划。"""
        del args, kwargs
        current_time[0] += 0.2
        yield None
        return expected_plan

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        forbidden_four_fast,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_layout_steps",
        successful_fallback,
    )
    job = assembly_planner.UnknownSolveJob(
        _irregular_three_pieces_like_field_mask(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
        four_fast_active_budget_seconds=0.01,
        clock=fake_clock,
    )

    assert job.advance(time_budget_ms=1000.0, work_unit_limit=64) is None
    assert [event["source"] for event in job.consume_stage_events()] == ["graph"]
    result = job.advance(time_budget_ms=1000.0, work_unit_limit=64)

    assert result is expected_plan
    assert result.success is True
    assert job.active_elapsed_ms == 200


def test_four_fast_success_before_active_budget_is_preserved(monkeypatch):
    """四片快路径在子预算内完成时必须直接返回，不得误转FALLBACK。"""
    from maixcam2_app_A_quad import assembly_planner

    current_time = [300.0]
    fallback_calls = []
    expected_plan = assembly_planner.AssemblyPlan(
        True,
        placements=[],
        target_rect_mm=(55.0, 176.0, 100.0, 60.0),
        reason="ok",
        diagnostics={"four_fast_path": 1},
    )

    def fake_clock():
        """返回快路径成功工作单元使用的假时钟。"""
        return current_time[0]

    def no_graph_solution(*args, **kwargs):
        """强制进入FOURFAST。"""
        del args, kwargs
        return None, {"graph_fast_path": 0}

    def successful_four_fast(*args, **kwargs):
        """模拟FOURFAST在1秒子预算内用0.8秒完成。"""
        del args, kwargs

        def stage():
            """通过生成器返回值模拟真实增量阶段终态。"""
            current_time[0] += 0.8
            if False:
                yield None
            return expected_plan, {"four_fast_path": 1}

        return stage()

    def forbidden_fallback(*args, **kwargs):
        """快路径成功后不得启动旧FALLBACK。"""
        del args, kwargs
        fallback_calls.append(True)
        yield None

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        successful_four_fast,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_layout_steps",
        forbidden_fallback,
    )
    job = assembly_planner.UnknownSolveJob(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
        four_fast_active_budget_seconds=1.0,
        clock=fake_clock,
    )

    assert job.advance(time_budget_ms=1000.0, work_unit_limit=64) is None
    job.consume_stage_events()
    result = job.advance(time_budget_ms=1000.0, work_unit_limit=64)

    assert result is expected_plan
    assert fallback_calls == []


@pytest.mark.parametrize("invalid_budget", (0.0, -1.0, float("inf"), float("nan")))
def test_four_fast_active_budget_must_be_finite_positive(invalid_budget):
    """四片独立活动预算必须是有限正数，避免任务永久占用或立即中止。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    with pytest.raises(ValueError, match="FOURFAST"):
        UnknownSolveJob(
            _field_four_pieces_from_device_log(),
            work_region_mm=(0.0, 33.5, 210.0, 230.0),
            split_y_mm=148.5,
            stop_at_first_solution=True,
            four_fast_active_budget_seconds=invalid_budget,
        )


def test_white_timeout_reports_the_stage_that_consumed_budget(monkeypatch):
    """GRAPH完成后若FOUR_FAST耗尽预算，结果来源必须标记为FOUR_FAST。"""
    from maixcam2_app_A_quad import assembly_planner

    current_time = [20.0]

    def fake_clock():
        """返回由FOUR_FAST工作单元推进的测试时钟。"""
        return current_time[0]

    def no_graph_solution(*args, **kwargs):
        """同步结束GRAPH并强制进入四片快路径。"""
        del args, kwargs
        return None, {"graph_fast_path": 0}

    def slow_four_stage(*args, **kwargs):
        """模拟FOUR_FAST首个工作单元越过活动预算。"""
        del args, kwargs

        def stage():
            """推进时钟后让出一次，触发统一任务截止检查。"""
            current_time[0] = 25.1
            yield None
            return None, {"four_fast_path": 0}

        return stage()

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        slow_four_stage,
    )
    job = assembly_planner.UnknownSolveJob(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        stop_at_first_solution=True,
        active_timeout_seconds=5.0,
        wall_timeout_seconds=20.0,
        clock=fake_clock,
    )

    # 第一帧只完成GRAPH失败事件；第二帧才允许启动FOUR_FAST并跨过活动截止线。
    assert job.advance(time_budget_ms=12.0, work_unit_limit=1) is None
    graph_events = job.consume_stage_events()
    assert [event["source"] for event in graph_events] == ["graph"]
    result = job.advance(time_budget_ms=12.0, work_unit_limit=1)

    assert result.success is False
    assert result.reason == "solver_timeout"
    assert job.result_source == "four_fast"


def test_unknown_job_exception_after_deadline_is_reported_as_timeout():
    """工作单元在截止线后抛错时必须优先报告超时，而不是笼统solver_error。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    current_time = [20.0]

    def fake_clock():
        """返回测试控制的单调秒数。"""
        return current_time[0]

    def raise_after_deadline():
        """模拟单个工作单元运行到截止线后才抛出底层异常。"""
        current_time[0] = 25.1
        raise RuntimeError("late OpenCV failure")
        yield None

    job = UnknownSolveJob(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        timeout_seconds=5.0,
        clock=fake_clock,
    )
    job._iterator = raise_after_deadline()

    result = job.advance(time_budget_ms=12.0, work_unit_limit=32)

    assert result.success is False
    assert result.reason == "solver_timeout"
    assert result.diagnostics["elapsed_ms"] == 5100


def test_unknown_job_checks_deadline_after_last_work_unit_in_slice():
    """单工作单元时间片结束时也必须立即检查截止线，不能拖到下一帧。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    current_time = [30.0]

    def fake_clock():
        """返回测试控制的单调秒数。"""
        return current_time[0]

    def slow_single_unit():
        """模拟唯一工作单元正常yield，但执行期间已经越过5秒截止线。"""
        current_time[0] = 35.1
        yield None

    job = UnknownSolveJob(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        timeout_seconds=5.0,
        clock=fake_clock,
    )
    job._iterator = slow_single_unit()

    result = job.advance(time_budget_ms=12.0, work_unit_limit=1)

    assert result is not None
    assert result.reason == "solver_timeout"
    assert result.diagnostics["elapsed_ms"] == 5100


def test_unknown_solve_job_cancel_closes_pending_search():
    """模式切换取消增量任务后必须关闭生成器，后续推进只能返回取消结果。"""
    from maixcam2_app_A_quad.assembly_planner import UnknownSolveJob

    job = UnknownSolveJob(
        _patterned_irregular_four_pieces(),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )
    assert job.advance(time_budget_ms=1000.0, work_unit_limit=1) is None

    cancelled = job.cancel()

    assert job.done is True
    assert cancelled.success is False
    assert cancelled.reason == "cancelled"
    assert job.advance() is cancelled


def _pattern_feature(values):
    """把一维灰度序列转换为求解器兼容的高纹理边特征。"""
    values = np.asarray(values, dtype=np.float64)
    colors = np.column_stack((values, values, values))
    gradients = np.gradient(values)
    return {
        "colors": colors.astype(float).tolist(),
        "gradients": gradients.astype(float).tolist(),
        "pattern_energy": 0.8,
    }


def _shared_edge_indices(first_polygon, second_polygon, tolerance=0.6):
    """在两个目标多边形中查找反向重合边并返回各自索引。"""
    first = np.asarray(first_polygon, dtype=np.float64)
    second = np.asarray(second_polygon, dtype=np.float64)
    for first_index in range(len(first)):
        first_start = first[first_index]
        first_end = first[(first_index + 1) % len(first)]
        for second_index in range(len(second)):
            second_start = second[second_index]
            second_end = second[(second_index + 1) % len(second)]
            if (
                np.linalg.norm(first_start - second_end) <= tolerance
                and np.linalg.norm(first_end - second_start) <= tolerance
            ):
                return first_index, second_index
    return None


def test_unknown_solver_uses_pattern_score_to_choose_equivalent_square_seam():
    """几何等价的两方片组合中，求解器必须选择颜色与梯度连续的接缝。"""
    from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout

    square = ((0, 0), (60, 0), (60, 60), (0, 60))
    first = _piece(square, "U1", 31, (60, 75))
    second = _piece(square, "U2", -47, (150, 78))
    ramp = [20, 45, 80, 120, 165, 210]
    mismatch = [220, 210, 200, 190, 180, 170]
    first["edge_features"] = [_pattern_feature(mismatch) for _ in range(4)]
    second["edge_features"] = [_pattern_feature(mismatch) for _ in range(4)]
    first["edge_features"][0] = _pattern_feature(ramp)
    second["edge_features"][2] = _pattern_feature(list(reversed(ramp)))

    plan = solve_unknown_layout(
        [first, second],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=4000,
    )

    assert plan.success is True
    placements = {placement.piece_id: placement for placement in plan.placements}
    assert _shared_edge_indices(
        placements["U1"].target_polygon_mm,
        placements["U2"].target_polygon_mm,
    ) == (0, 2)
