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


def _pieces_from_layout(layout):
    """把目标布局的四片分别施加不同源位姿，形成与模板无关的UNKNOWN输入。"""
    angles = (37.0, -61.0, 83.0, -24.0)
    centers = ((35.0, 72.0), (82.0, 78.0), (132.0, 70.0), (178.0, 80.0))
    return [
        _piece(vertices, f"U{index + 1}", angles[index], centers[index])
        for index, vertices in enumerate(layout)
    ]


def _four_grid_pieces(width=100.0, height=60.0):
    """构造能无缝组成指定矩形的四宫格碎片。"""
    half_width = width * 0.5
    half_height = height * 0.5
    return _pieces_from_layout(
        (
            ((0, 0), (half_width, 0), (half_width, half_height), (0, half_height)),
            ((half_width, 0), (width, 0), (width, half_height), (half_width, half_height)),
            ((0, half_height), (half_width, half_height), (half_width, height), (0, height)),
            ((half_width, half_height), (width, half_height), (width, height), (half_width, height)),
        )
    )


def _t_junction_pieces():
    """构造一条100mm长边同时连接40/30/30mm三条短边的T形布局。"""
    return _pieces_from_layout(
        (
            ((0, 0), (100, 0), (100, 30), (0, 30)),
            ((0, 30), (40, 30), (40, 60), (0, 60)),
            ((40, 30), (70, 30), (70, 60), (40, 60)),
            ((70, 30), (100, 30), (100, 60), (70, 60)),
        )
    )


def _assert_placement_reconstructs_target(piece, placement):
    """按发送的源中心、目标中心和角度重建多边形，验证机械位姿内部一致。"""
    source_vertices = np.asarray(piece["vertices_mm"], dtype=np.float64)
    source_center = np.asarray(placement.source_center_mm, dtype=np.float64)
    target_center = np.asarray(placement.target_center_mm, dtype=np.float64)
    angle = math.radians(float(placement.rotation_delta_deg))
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    reconstructed = (source_vertices - source_center) @ rotation.T + target_center
    assert reconstructed == pytest.approx(
        np.asarray(placement.target_polygon_mm, dtype=np.float64),
        abs=1e-5,
    )


@pytest.mark.parametrize("pieces", (_four_grid_pieces(), _t_junction_pieces()))
def test_four_solver_returns_all_source_and_target_poses(pieces):
    """专用求解器必须对四宫格和T形布局一次返回四个可执行位姿。"""
    from maixcam2_app_A_quad.four_piece_solver import solve_four_piece_layout

    plan = solve_four_piece_layout(
        pieces,
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
    )

    assert plan.success is True, (plan.reason, plan.diagnostics)
    assert len(plan.placements) == 4
    assert {item.piece_id for item in plan.placements} == {"U1", "U2", "U3", "U4"}
    assert plan.target_rect_mm[1] >= 148.5
    long_side = max(plan.target_rect_mm[2:])
    short_side = min(plan.target_rect_mm[2:])
    assert long_side == pytest.approx(100.0, abs=1.5)
    assert short_side == pytest.approx(60.0, abs=1.5)
    pieces_by_id = {piece["id"]: piece for piece in pieces}
    for placement in plan.placements:
        _assert_placement_reconstructs_target(
            pieces_by_id[placement.piece_id],
            placement,
        )


def test_four_solve_job_advances_in_bounded_work_units():
    """单帧只允许一个工作单元时不得同步跑完整搜索，后续帧继续同一任务。"""
    from maixcam2_app_A_quad.four_piece_solver import FourPieceSolveJob

    job = FourPieceSolveJob(
        _four_grid_pieces(),
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
        active_budget_seconds=3.0,
    )

    assert job.advance(time_budget_ms=1000.0, work_unit_limit=1) is None
    calls = 1
    while not job.done and calls < 2000:
        job.advance(time_budget_ms=1000.0, work_unit_limit=1)
        calls += 1

    assert job.done is True
    assert job.result.success is True
    assert calls > 1
    assert calls < 2000


def test_four_solve_job_returns_validated_plan_when_last_unit_crosses_budget():
    """工作单元越过活动截止线时，已验证计划必须优先于solver_timeout返回。"""
    from maixcam2_app_A_quad.four_piece_solver import (
        FourPieceSolveJob,
        solve_four_piece_layout,
    )

    validated = solve_four_piece_layout(
        _four_grid_pieces(),
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
    )
    assert validated.success is True
    clock_values = iter((0.0, 0.0, 3.1))
    job = FourPieceSolveJob(
        _four_grid_pieces(),
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
        active_budget_seconds=3.0,
        clock=lambda: next(clock_values),
    )

    def one_expensive_success_unit():
        """模拟底层在yield前保存计划，但该工作单元执行后已越过截止线。"""
        job._fast_progress["best_plan"] = validated
        yield None
        return validated

    job._generator = one_expensive_success_unit()
    result = job.advance(time_budget_ms=10000.0, work_unit_limit=2)

    # 截止返回前必须克隆并重建位姿，不能直接复用底层快图对象中的旧角度。
    assert result is not validated
    assert result.success is True
    assert [item.piece_id for item in result.placements] == [
        item.piece_id for item in validated.placements
    ]
    np.testing.assert_allclose(
        [item.target_center_mm for item in result.placements],
        [item.target_center_mm for item in validated.placements],
        atol=1e-4,
    )
    assert result.diagnostics["fill_milli"] == validated.diagnostics["fill_milli"]
    assert result.diagnostics["overlap_milli"] == validated.diagnostics["overlap_milli"]
    assert result.diagnostics["returned_best_at_timeout"] == 1
    assert job.done is True


def test_four_solver_rejects_wrong_count_and_out_of_range_rectangle():
    """非四片和80×40mm小矩形都必须失败，且不得携带任何机械目标。"""
    from maixcam2_app_A_quad.four_piece_solver import solve_four_piece_layout

    wrong_count = solve_four_piece_layout(
        _four_grid_pieces()[:3],
        (0.0, 0.0, 210.0, 297.0),
        148.5,
    )
    too_small = solve_four_piece_layout(
        _four_grid_pieces(width=80.0, height=40.0),
        (0.0, 0.0, 210.0, 297.0),
        148.5,
    )

    assert wrong_count.reason == "four_needs_exactly_four"
    assert too_small.reason in ("no_rect", "size_reject")
    assert wrong_count.placements == []
    assert too_small.placements == []


def test_four_solver_applies_dedicated_two_mm_size_tolerance():
    """FOUR应接受题目上限外2mm测量误差，并拒绝超过专用容差的尺寸。"""
    from maixcam2_app_A_quad.four_piece_solver import solve_four_piece_layout

    within_tolerance = solve_four_piece_layout(
        _four_grid_pieces(width=100.0, height=91.0),
        (0.0, 0.0, 210.0, 297.0),
        148.5,
    )
    outside_tolerance = solve_four_piece_layout(
        _four_grid_pieces(width=100.0, height=93.0),
        (0.0, 0.0, 210.0, 297.0),
        148.5,
    )

    assert within_tolerance.success is True
    assert outside_tolerance.success is False
    assert outside_tolerance.reason in ("no_rect", "size_reject")


def test_four_fast_path_isolated_from_unknown_acceptance_globals(monkeypatch):
    """FOUR快图必须显式使用专用验收宏，不能随普通UNKNOWN全局值改变。"""
    from maixcam2_app_A_quad import assembly_planner
    from maixcam2_app_A_quad.four_piece_solver import solve_four_piece_layout
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    # 把普通UNKNOWN验收改成必然拒绝实机矩形；FOUR仍应按自己的题目范围求解成功。
    monkeypatch.setattr(assembly_planner, "UNKNOWN_LONG_SIDE_RANGE_MM", (1.0, 2.0))
    monkeypatch.setattr(assembly_planner, "UNKNOWN_SHORT_SIDE_RANGE_MM", (1.0, 2.0))
    monkeypatch.setattr(assembly_planner, "UNKNOWN_STRICT_MIN_FILL_RATIO", 0.999)
    monkeypatch.setattr(assembly_planner, "UNKNOWN_RELAXED_MIN_FILL_RATIO", 0.999)
    monkeypatch.setattr(assembly_planner, "UNKNOWN_MAX_OVERLAP_RATIO", 0.0)

    plan = solve_four_piece_layout(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
    )

    assert plan.success is True, (plan.reason, plan.diagnostics)


def test_four_solver_handles_device_snapshot_without_generic_fallback():
    """实机噪声轮廓必须走四片快图求解，不能依赖旧通用FALLBACK。"""
    from maixcam2_app_A_quad.four_piece_solver import solve_four_piece_layout
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    plan = solve_four_piece_layout(
        _field_four_pieces_from_device_log(),
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert len(plan.placements) == 4
    assert plan.diagnostics["solver_source_fast"] == 1
    assert plan.diagnostics["native_search_nodes"] == 0
    pieces_by_id = {
        piece["id"]: piece for piece in _field_four_pieces_from_device_log()
    }
    for placement in plan.placements:
        # 实机回退路径同样必须保证UART中心和角度可精确重建目标，不能只看矩形成功。
        _assert_placement_reconstructs_target(
            pieces_by_id[placement.piece_id],
            placement,
        )
