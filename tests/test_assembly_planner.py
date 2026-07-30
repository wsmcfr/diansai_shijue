"""A版拼装规划器的几何求解、KNOWN登记和诊断回归测试。"""

import math
import copy

import cv2
import numpy as np

from maixcam2_app_A_quad.assembly_planner import solve_unknown_layout


WORK_REGION_MM = (0.0, 33.5, 210.0, 230.0)
SPLIT_Y_MM = 148.5


def _rigid_transform(vertices, angle_deg, translation):
    """对目标多边形施加无镜像刚体变换，模拟碎片在上半区随机摆放。

    主要流程：围绕多边形质心旋转指定角度，再整体平移到任意观测位置。
    关键参数：vertices为目标毫米顶点，angle_deg为平面旋转角，translation为位移。
    返回值：保持原顶点顺序的N×2浮点数组。
    """
    polygon = np.asarray(vertices, dtype=np.float64)
    center = np.mean(polygon, axis=0)
    angle = math.radians(float(angle_deg))
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    return (polygon - center) @ rotation.T + center + np.asarray(translation)


def _solver_piece(piece_id, target_vertices, angle_deg, translation):
    """构造未知求解器所需的最小真实碎片字典。

    目标轮廓先独立旋转和平移，确保测试验证的是形状拼装而不是输入当前位置。
    返回值包含编号、毫米顶点和观测质心，不依赖相机或Maix运行时。
    """
    observed = _rigid_transform(target_vertices, angle_deg, translation)
    center = np.mean(observed, axis=0)
    contour = np.rint(observed * 10.0).astype(np.int32).reshape(-1, 1, 2)
    edge_lengths = [
        float(np.linalg.norm(observed[(index + 1) % len(observed)] - observed[index]))
        for index in range(len(observed))
    ]
    return {
        "id": str(piece_id),
        "vertices_mm": observed.astype(float).tolist(),
        "center_mm": tuple(float(value) for value in center),
        # 下列字段来自视觉层，KNOWN模板描述子会使用；UNKNOWN求解只读取上面三项。
        "contour": contour,
        "vertex_count": len(observed),
        "edge_lengths": edge_lengths,
        "interior_angles": [90.0 for _ in observed],
        "area": abs(float(cv2.contourArea(contour))),
        "perimeter": float(cv2.arcLength(contour, True)),
        "complete": True,
        "region": "upper",
    }


def _make_segmented_seam_pieces():
    """生成一条110mm长边由31、37、42mm三段共同对接的四片拼图。

    四片最终组成110×70mm矩形，所有实体边均不短于20mm。输入顺序故意不把
    长条碎片放在第一项，验证求解不能依赖屏幕编号或随机摆放顺序。
    """
    target_polygons = {
        "U1": ((0.0, 23.0), (31.0, 23.0), (31.0, 70.0), (0.0, 70.0)),
        "U2": ((31.0, 23.0), (68.0, 23.0), (68.0, 70.0), (31.0, 70.0)),
        "U3": ((0.0, 0.0), (110.0, 0.0), (110.0, 23.0), (0.0, 23.0)),
        "U4": ((68.0, 23.0), (110.0, 23.0), (110.0, 70.0), (68.0, 70.0)),
    }
    poses = {
        "U1": (37.0, (12.0, 5.0)),
        "U2": (-83.0, (41.0, -9.0)),
        "U3": (128.0, (-18.0, 22.0)),
        "U4": (-24.0, (67.0, 11.0)),
    }
    return [
        _solver_piece(piece_id, target_polygons[piece_id], *poses[piece_id])
        for piece_id in ("U1", "U2", "U3", "U4")
    ]


def _make_known_upper_pieces():
    """生成可组成100×60mm矩形且全部标记为上半区的四片KNOWN输入。"""
    target_polygons = {
        "UNKNOWN": ((0.0, 0.0), (100.0, 0.0), (100.0, 21.0), (0.0, 21.0)),
        "?": ((0.0, 21.0), (29.0, 21.0), (29.0, 60.0), (0.0, 60.0)),
        "": ((29.0, 21.0), (63.0, 21.0), (63.0, 60.0), (29.0, 60.0)),
        "UNKNOWN_DUP": ((63.0, 21.0), (100.0, 21.0), (100.0, 60.0), (63.0, 60.0)),
    }
    poses = ((33.0, (7.0, 5.0)), (-61.0, (45.0, -4.0)), (119.0, (-9.0, 18.0)), (-17.0, (55.0, 9.0)))
    pieces = []
    for (piece_id, polygon), pose in zip(target_polygons.items(), poses):
        piece = _solver_piece(piece_id, polygon, *pose)
        # 模拟KNOWN无旧模板时，视觉匹配会把多片都标为UNKNOWN，自动登记不能依赖旧ID。
        piece["id"] = "UNKNOWN"
        pieces.append(piece)
    return pieces


def _make_known_lower_layout():
    """生成已经在下半区正确拼成100×60mm矩形的四片KNOWN登记输入。"""
    target_polygons = (
        ((0.0, 0.0), (100.0, 0.0), (100.0, 21.0), (0.0, 21.0)),
        ((0.0, 21.0), (29.0, 21.0), (29.0, 60.0), (0.0, 60.0)),
        ((29.0, 21.0), (63.0, 21.0), (63.0, 60.0), (29.0, 60.0)),
        ((63.0, 21.0), (100.0, 21.0), (100.0, 60.0), (63.0, 60.0)),
    )
    pieces = [
        _solver_piece("UNKNOWN", polygon, 0.0, (55.0, 176.0))
        for polygon in target_polygons
    ]
    for piece in pieces:
        piece["complete"] = True
        piece["region"] = "lower"
    return pieces


def _add_far_camera_vertex_noise(pieces):
    """给四片顶点加入确定性的约1mm误差，模拟整张A4视野下的角点抖动。"""
    noisy_pieces = copy.deepcopy(pieces)
    noise_by_piece = (
        ((1.2, -0.8), (-1.0, 0.7), (0.9, 1.1), (-0.7, -1.2)),
        ((-0.8, 1.0), (1.1, -0.9), (-1.2, 0.6), (0.7, -0.8)),
        ((0.8, 1.1), (-1.1, -0.7), (1.0, 0.9), (-0.9, -1.0)),
        ((-1.0, -0.8), (0.9, 1.2), (-0.7, -1.1), (1.1, 0.7)),
    )
    for piece, noise in zip(noisy_pieces, noise_by_piece):
        vertices = np.asarray(piece["vertices_mm"], dtype=np.float64)
        vertices += np.asarray(noise, dtype=np.float64)
        piece["vertices_mm"] = vertices.astype(float).tolist()
        piece["center_mm"] = tuple(float(value) for value in np.mean(vertices, axis=0))
    return noisy_pieces


def test_unknown_solver_supports_one_long_edge_against_multiple_short_edges():
    """UNKNOWN应能求解题目允许的一条长边对应多条共线短边的矩形。"""
    plan = solve_unknown_layout(
        _make_segmented_seam_pieces(),
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=50000,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes)
    assert len(plan.placements) == 4
    assert np.allclose(plan.target_rect_mm[2:], (110.0, 70.0), atol=1.0)


def test_unknown_solver_handles_far_camera_noise_before_2000_node_limit():
    """约1mm顶点误差下应在较小节点预算内找到目标，满足设备实时性要求。"""
    pieces = _add_far_camera_vertex_noise(_make_segmented_seam_pieces())

    plan = solve_unknown_layout(
        pieces,
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=2000,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert len(plan.placements) == 4


def test_unknown_solver_reports_edge_mismatch_when_no_seam_candidate_exists():
    """没有任何可用接缝时应明确报告边不匹配，而不是笼统NO_ASSEMBLY。"""
    pieces = [
        _solver_piece("U1", ((0.0, 0.0), (25.0, 0.0), (0.0, 33.0)), 17.0, (2.0, 3.0)),
        _solver_piece("U2", ((0.0, 0.0), (52.0, 0.0), (0.0, 68.0)), -29.0, (50.0, 4.0)),
    ]

    plan = solve_unknown_layout(pieces, WORK_REGION_MM, SPLIT_Y_MM)

    assert plan.success is False
    assert plan.reason == "edge_mismatch"


def test_unknown_solver_reports_size_reject_for_complete_but_too_small_layout():
    """能沿整边组合但目标尺寸小于题目下限时应明确报告尺寸拒绝。"""
    pieces = [
        _solver_piece(
            "U1",
            ((0.0, 0.0), (25.0, 0.0), (25.0, 30.0), (0.0, 30.0)),
            41.0,
            (5.0, 2.0),
        ),
        _solver_piece(
            "U2",
            ((25.0, 0.0), (50.0, 0.0), (50.0, 30.0), (25.0, 30.0)),
            -67.0,
            (60.0, -7.0),
        ),
    ]

    plan = solve_unknown_layout(pieces, WORK_REGION_MM, SPLIT_Y_MM)

    assert plan.success is False
    assert plan.reason == "size_reject"


def test_unknown_single_piece_reports_geometry_rejection_instead_of_edge_mismatch():
    """单片无需寻找接缝，尺寸不合格时必须报告几何原因而不是边不匹配。"""
    piece = _solver_piece(
        "U1",
        ((0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)),
        27.0,
        (12.0, 8.0),
    )

    plan = solve_unknown_layout([piece], WORK_REGION_MM, SPLIT_Y_MM)

    assert plan.success is False
    assert plan.reason == "size_reject"


def test_known_save_registers_lower_layout_and_builds_exact_target_templates():
    """KNOWN保存应直接登记下半区正确布局，并生成精确100×60mm目标模板。"""
    from maixcam2_app_A_quad.assembly_planner import solve_and_register_known_layout

    plan, templates = solve_and_register_known_layout(
        _make_known_lower_layout(),
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=50000,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes)
    assert [template["id"] for template in templates] == ["K1", "K2", "K3", "K4"]
    assert all(template["layout_size_mm"] == [100.0, 60.0] for template in templates)
    all_target_vertices = np.vstack(
        [np.asarray(template["target_vertices_mm"]) for template in templates]
    )
    assert np.allclose(np.min(all_target_vertices, axis=0), (0.0, 0.0), atol=1e-6)
    assert np.allclose(np.max(all_target_vertices, axis=0), (100.0, 60.0), atol=1e-6)


def test_known_direct_registration_uses_at_most_24_template_assignments():
    """KNOWN直接登记仅允许固定4!模板分配，不能进入未知拼图节点搜索。"""
    from maixcam2_app_A_quad.assembly_planner import solve_and_register_known_layout

    plan, templates = solve_and_register_known_layout(
        _make_known_lower_layout(),
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=100,
    )

    assert plan.success is True, (plan.reason, plan.search_nodes, plan.diagnostics)
    assert plan.search_nodes <= 24
    assert len(templates) == 4


def test_known_direct_registration_ignores_non_actionable_edge_contours():
    """状态栏已是N=4时，额外的不完整EDGE轮廓不得阻止下半区登记。"""
    from maixcam2_app_A_quad.assembly_planner import solve_and_register_known_layout

    pieces = _make_known_lower_layout()
    edge_piece = dict(pieces[0])
    edge_piece["id"] = "EDGE"
    edge_piece["complete"] = False
    edge_piece["region"] = "crossing"

    plan, templates = solve_and_register_known_layout(
        pieces + [edge_piece],
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=100,
    )

    assert plan.success is True, plan.reason
    assert len(templates) == 4


def test_main_known_save_writes_templates_from_lower_layout(tmp_path):
    """主入口SAVE辅助函数应保存正确布局模板，并返回可立即绘制的成功规划。"""
    from maixcam2_app_A_quad.main import register_and_save_known_layout
    from maixcam2_app_A_quad.template_store import load_templates

    template_path = tmp_path / "known_templates.json"
    plan, templates = register_and_save_known_layout(
        _make_known_lower_layout(),
        template_path,
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=50000,
    )

    assert plan.success is True
    assert len(templates) == 4
    assert load_templates(template_path) == templates


def test_runtime_cached_known_plan_survives_following_update():
    """SAVE预装的KNOWN成功规划应在后续帧直接返回，不重新等待三帧。"""
    from maixcam2_app_A_quad.assembly_planner import (
        AssemblyRuntime,
        solve_and_register_known_layout,
    )

    pieces = _make_known_lower_layout()
    plan, templates = solve_and_register_known_layout(
        pieces,
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=50000,
    )
    runtime = AssemblyRuntime(stable_frames=3, position_tolerance_mm=2.0)

    runtime.cache_plan(
        "known",
        plan,
        templates,
        WORK_REGION_MM,
        SPLIT_Y_MM,
    )
    cached = runtime.update(
        "known",
        pieces,
        templates,
        WORK_REGION_MM,
        SPLIT_Y_MM,
    )

    assert cached is plan
    assert runtime.stable_count == 3


def test_known_save_action_caches_plan_and_returns_success_status(tmp_path):
    """完整SAVE动作成功后应更新模板、缓存目标并返回不会误解的成功文字。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from maixcam2_app_A_quad.main import perform_known_save_action

    runtime = AssemblyRuntime(stable_frames=3, position_tolerance_mm=2.0)
    templates, plan, status = perform_known_save_action(
        _make_known_lower_layout(),
        [],
        tmp_path / "known_templates.json",
        WORK_REGION_MM,
        SPLIT_Y_MM,
        runtime,
        max_nodes=50000,
    )

    assert len(templates) == 4
    assert plan.success is True
    assert runtime.plan is plan
    assert status == "KNOWN SAVED PLAN OK"


def test_known_save_action_failure_preserves_previous_templates(tmp_path):
    """KNOWN输入不足四片时应保留内存和磁盘旧模板并显示具体失败原因。"""
    from maixcam2_app_A_quad.assembly_planner import (
        AssemblyRuntime,
        solve_and_register_known_layout,
    )
    from maixcam2_app_A_quad.main import perform_known_save_action
    from maixcam2_app_A_quad.template_store import load_templates, save_templates

    old_plan, old_templates = solve_and_register_known_layout(
        _make_known_lower_layout(),
        WORK_REGION_MM,
        SPLIT_Y_MM,
        max_nodes=50000,
    )
    assert old_plan.success is True
    template_path = tmp_path / "known_templates.json"
    save_templates(template_path, old_templates)
    runtime = AssemblyRuntime(stable_frames=3, position_tolerance_mm=2.0)

    templates, plan, status = perform_known_save_action(
        _make_known_lower_layout()[:3],
        old_templates,
        template_path,
        WORK_REGION_MM,
        SPLIT_Y_MM,
        runtime,
    )

    assert templates is old_templates
    assert plan.success is False
    assert status == "SAVE KNOWN_NEEDS_FOUR"
    assert load_templates(template_path) == old_templates
    assert runtime.plan is None


def test_planning_status_keeps_save_message_for_action_frame():
    """同一帧已经产生SAVE结果时，成功规划不得把它覆盖成普通PLAN OK。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyPlan
    from maixcam2_app_A_quad.main import select_planning_status

    plan = AssemblyPlan(True, placements=[object(), object(), object(), object()])

    status = select_planning_status(
        "KNOWN SAVED PLAN OK",
        plan,
        stable_count=3,
        stable_frames=3,
        preserve_current=True,
    )

    assert status == "KNOWN SAVED PLAN OK"
