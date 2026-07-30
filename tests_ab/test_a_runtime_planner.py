"""验证A版稳定帧求解缓存、主循环接线和下半区目标绘制。"""

from pathlib import Path

import numpy as np

from tests_ab.synthetic_paper import DEFAULT_PAPER_QUAD


def _unknown_rectangle(center=(80.0, 80.0), jitter=(0.0, 0.0)):
    """构造位于上半区的100×60mm单片未知矩形。"""
    center = np.asarray(center, dtype=np.float64) + np.asarray(jitter, dtype=np.float64)
    local = np.asarray(((-50, -30), (50, -30), (50, 30), (-50, 30)), dtype=np.float64)
    vertices = local + center
    return {
        "id": "U1",
        "vertices_mm": vertices.astype(float).tolist(),
        "center_mm": tuple(float(value) for value in center),
        "region": "upper",
        "complete": True,
        "vertex_count": 4,
    }


def _run_runtime_until_plan(runtime, arguments, max_frames=5000):
    """重复推进同一锁定上下文，直到运行器返回终态规划。

    参数arguments直接传给AssemblyRuntime.update；max_frames防止测试在状态机回归时
    无限循环。返回`(plan, frame_count)`，超过上限会给出明确断言。
    """
    for frame_count in range(1, int(max_frames) + 1):
        plan = runtime.update(**arguments)
        if plan is not None:
            return plan, frame_count
    raise AssertionError(f"运行器在{int(max_frames)}帧内没有返回终态")


def test_known_stability_gate_accepts_four_to_five_vertex_edge_jitter():
    """相邻帧同片只多一个2mm毛刺点时稳定计数不得从1/3重新清零。"""
    from maixcam2_app_A_quad.assembly_planner import (
        _observations_are_stable,
        _piece_observation,
    )

    first_piece = {
        "vertices_mm": ((10.0, 10.0), (52.0, 10.0), (48.0, 65.0), (14.0, 58.0)),
        "center_mm": (31.0, 35.75),
    }
    second_piece = {
        "vertices_mm": (
            (10.0, 10.0),
            (31.0, 8.0),
            (52.0, 10.0),
            (48.0, 65.0),
            (14.0, 58.0),
        ),
        "center_mm": (31.0, 35.75),
    }

    stable = _observations_are_stable(
        [_piece_observation(first_piece)],
        [_piece_observation(second_piece)],
        tolerance_mm=3.0,
    )

    assert stable is True


def test_a_known_runtime_defaults_use_relaxed_but_bounded_thresholds():
    """A版默认中心容差应为3mm，KNOWN配置阈值应适度放宽到1.60。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG

    runtime = AssemblyRuntime()

    assert runtime.position_tolerance_mm == 3.0
    assert DEFAULT_CONFIG["known_match_threshold"] == 1.60


def test_unknown_runtime_defaults_give_locked_solver_more_cpu_time():
    """锁定快照后的UNKNOWN兜底应使用24ms/64单元和8s/30s双截止线。"""
    from maixcam2_app_A_quad import assembly_planner

    runtime = assembly_planner.AssemblyRuntime()

    assert runtime.solver_time_budget_ms == 24.0
    assert runtime.solver_work_unit_limit == 64
    assert assembly_planner.UNKNOWN_SOLVER_ACTIVE_TIMEOUT_SECONDS == 8.0
    assert assembly_planner.UNKNOWN_SOLVER_WALL_TIMEOUT_SECONDS == 30.0


def test_runtime_waits_three_stable_frames_then_caches_single_solve():
    """连续三帧稳定前不得搜索，成功后即使碎片开始移动也保持同一规划。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    runtime = AssemblyRuntime(stable_frames=3, position_tolerance_mm=2.0)
    arguments = {
        "mode": "unknown",
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(pieces=[_unknown_rectangle()], **arguments) is None
    assert runtime.update(pieces=[_unknown_rectangle(jitter=(0.4, -0.3))], **arguments) is None
    plan = runtime.update(
        pieces=[_unknown_rectangle(jitter=(-0.2, 0.2))],
        **arguments,
    )

    assert plan.success is True
    assert runtime.stable_count == 3
    assert runtime.solve_count == 1

    cached = runtime.update(pieces=[_unknown_rectangle(center=(150, 90))], **arguments)
    assert cached is plan
    assert runtime.solve_count == 1


def test_runtime_locks_deep_copied_snapshot_after_stability():
    """稳定门满足后必须深复制一次碎片，后续实时坐标变化不能污染求解快照。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    runtime = AssemblyRuntime(stable_frames=1)
    first_frame_piece = _unknown_rectangle(center=(80.0, 80.0))
    original_center = tuple(first_frame_piece["center_mm"])
    arguments = {
        "mode": "unknown",
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    plan = runtime.update(pieces=[first_frame_piece], **arguments)
    first_frame_piece["center_mm"] = (190.0, 140.0)
    cached = runtime.update(
        pieces=[_unknown_rectangle(center=(150.0, 100.0))],
        **arguments,
    )

    assert plan.success is True
    assert cached is plan
    assert runtime.snapshot_locked is True
    assert len(runtime.locked_pieces) == 1
    assert tuple(runtime.locked_pieces[0]["center_mm"]) == original_center


def test_runtime_context_change_clears_stability_and_cached_plan():
    """识别模式或机械设置变化时必须丢弃旧目标，重新累计稳定帧。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    runtime = AssemblyRuntime(stable_frames=2)
    common = {
        "pieces": [_unknown_rectangle()],
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }
    assert runtime.update(mode="unknown", **common) is None
    assert runtime.update(mode="unknown", **common).success is True

    changed = runtime.update(mode="known", **common)

    assert changed is None
    assert runtime.plan is None
    assert runtime.stable_count == 1
    assert runtime.solve_count == 1


def test_runtime_unknown_third_stable_frame_starts_incremental_job():
    """UNKNOWN第3稳定帧只能启动短任务，后续多帧推进完成后才缓存最终规划。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from tests_ab.test_a_unknown_planner import _patterned_irregular_four_pieces

    runtime = AssemblyRuntime(
        stable_frames=3,
        position_tolerance_mm=2.0,
        solver_time_budget_ms=1000.0,
        solver_work_unit_limit=1,
        texture_refinement_nodes=80,
    )
    arguments = {
        "mode": "unknown",
        # CARD保留既有纹理增量搜索；WHITE将在后续版本优先走同步有界图快路径。
        "unknown_profile": "card",
        "pieces": _patterned_irregular_four_pieces(),
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(**arguments) is None
    assert runtime.update(**arguments) is None
    third_frame = runtime.update(**arguments)

    assert third_frame is None
    assert runtime.is_solving is True
    assert runtime.solve_count == 1

    plan = None
    for _ in range(5000):
        plan = runtime.update(**arguments)
        if plan is not None:
            break

    assert plan is not None
    assert plan.success is True
    assert runtime.is_solving is False
    assert runtime.plan is plan


def test_runtime_white_returns_graph_plan_through_unified_incremental_job():
    """WHITE统一任务必须优先返回GRAPH规划，成功后不得进入后续阶段。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _noisy_field_three_pieces

    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)
    plan, _frame_count = _run_runtime_until_plan(
        runtime,
        {
            "mode": "unknown",
            "unknown_profile": "white",
            "pieces": _noisy_field_three_pieces(),
            "templates": [],
            "work_region_mm": (0.0, 33.5, 210.0, 230.0),
            "split_y_mm": 148.5,
        },
    )

    assert plan.success is True
    assert plan.diagnostics["graph_fast_path"] == 1
    assert runtime.plan is plan
    assert runtime.is_solving is False


def test_runtime_debug_log_reports_snapshot_graph_and_result(capsys):
    """调试开启时必须一次性输出锁定几何、GRAPH统计和最终结果。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    runtime = AssemblyRuntime(stable_frames=1, debug_enabled=True)
    plan = runtime.update(
        mode="unknown",
        unknown_profile="white",
        pieces=[_unknown_rectangle()],
        templates=[],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    output = capsys.readouterr().out
    assert plan.success is True
    assert "[SOLVER] SNAPSHOT" in output
    assert "[SOLVER] PIECE" in output
    assert "vertices_mm=" in output
    assert "edges_mm=" in output
    assert "[SOLVER] GRAPH" in output
    assert "layouts=" in output
    assert "[SOLVER] RESULT" in output
    assert "reason=ok" in output


def test_runtime_debug_log_reports_four_fast_geometry_counts(capsys):
    """实机四片成功日志必须显示分层状态、分段接缝和几何求交路径。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    runtime = AssemblyRuntime(stable_frames=1, debug_enabled=True)
    plan, frame_count = _run_runtime_until_plan(
        runtime,
        {
            "mode": "unknown",
            "unknown_profile": "white",
            "pieces": _field_four_pieces_from_device_log(),
            "templates": [],
            "work_region_mm": (0.0, 33.5, 210.0, 230.0),
            "split_y_mm": 148.5,
        },
    )

    output = capsys.readouterr().out
    assert plan.success is True
    assert frame_count > 1
    assert "[SOLVER] FOUR_FAST result=OK" in output
    assert "pairs=" in output
    assert "triples=" in output
    # 默认2400工作量在完整层第24个父状态后触顶；日志必须显示真实展开数，不能
    # 再把“保留32个状态”误报成“展开32个状态”。
    assert "parents=1/32/24" in output
    assert "limit=1" in output
    assert "time_limit=0" in output
    assert "active_ms=" in output
    assert "segmented=" in output
    assert "triangles=" in output
    assert "raster=" in output
    assert "[SOLVER] RESULT source=FOUR_FAST success=1" in output


def test_runtime_debug_switch_disables_all_solver_output(capsys):
    """调试关闭时求解器不得打印任何日志，现场正式运行不承担字符串构造开销。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    runtime = AssemblyRuntime(stable_frames=1, debug_enabled=False)
    plan = runtime.update(
        mode="unknown",
        unknown_profile="white",
        pieces=[_unknown_rectangle()],
        templates=[],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is True
    assert "[SOLVER]" not in capsys.readouterr().out


def test_runtime_debug_snapshot_reports_white_cleanup(capsys):
    """WHITE锁定日志必须显示求解副本清理前后顶点数和原始最短边。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from tests_ab.test_a_unknown_planner import _device_short_edge_pieces

    vertices = _device_short_edge_pieces()[0]
    piece = {
        "id": "U2",
        "vertices_mm": vertices.astype(float).tolist(),
        "center_mm": tuple(float(value) for value in np.mean(vertices, axis=0)),
    }
    runtime = AssemblyRuntime(stable_frames=1, debug_enabled=True)

    runtime._debug_snapshot("unknown", "white", [piece])

    output = capsys.readouterr().out
    assert "[SOLVER] CLEAN id=U2" in output
    assert "vertices=5->" in output
    assert "removed=" in output
    assert "min_edge=6.6" in output or "min_edge=6.7" in output


def test_runtime_disabled_debug_does_not_build_cleanup_preview(monkeypatch, capsys):
    """调试关闭时快照日志不得为了CLEAN文字额外运行轮廓清理。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _device_short_edge_pieces

    def forbidden_cleanup(*args, **kwargs):
        """若关闭日志后仍构造清理预览，则立即使测试失败。"""
        del args, kwargs
        raise AssertionError("关闭调试后不应构造CLEAN日志预览")

    monkeypatch.setattr(
        assembly_planner,
        "_clean_solver_short_edges",
        forbidden_cleanup,
    )
    vertices = _device_short_edge_pieces()[0]
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1, debug_enabled=False)

    runtime._debug_snapshot(
        "unknown",
        "white",
        [
            {
                "id": "U2",
                "vertices_mm": vertices.astype(float).tolist(),
                "center_mm": tuple(float(value) for value in np.mean(vertices, axis=0)),
            }
        ],
    )

    assert "[SOLVER]" not in capsys.readouterr().out


def test_unknown_runtime_never_calls_known_layout_even_when_templates_exist(monkeypatch):
    """UNKNOWN必须完全忽略KNOWN模板，任何模板匹配或固定布局调用都视为架构错误。"""
    from maixcam2_app_A_quad import assembly_planner

    def forbidden_known_layout(*args, **kwargs):
        """若UNKNOWN分支意外进入KNOWN规划，立即给出明确失败。"""
        del args, kwargs
        raise AssertionError("UNKNOWN不允许调用solve_known_layout")

    monkeypatch.setattr(assembly_planner, "solve_known_layout", forbidden_known_layout)
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1, debug_enabled=False)

    plan = runtime.update(
        mode="unknown",
        unknown_profile="white",
        pieces=[_unknown_rectangle()],
        templates=[{"id": "K1", "forbidden": True}],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is True


def test_runtime_debug_log_reports_fallback_rejection_counts(monkeypatch, capsys):
    """GRAPH无解转入兜底后，终态日志必须带节点、前沿和具体拒绝次数。"""
    from maixcam2_app_A_quad import assembly_planner

    def no_graph_solution(*args, **kwargs):
        """模拟GRAPH完成有界枚举但没有通过验收的候选。"""
        del args, kwargs
        return None, {
            "graph_edge_candidates": 7,
            "graph_matching_sets": 5,
            "graph_layouts_checked": 4,
            "graph_fast_path": 0,
            "graph_relaxed_fill_reject": 2,
            "graph_outer_edge_reject": 1,
        }

    class RejectedSolveJob:
        """模拟兜底搜索结束并暴露现场需要的拒绝诊断。"""

        done = False
        search_nodes = 23
        edge_candidates = 34
        max_frontier_width = 13
        first_solution_node = None
        active_elapsed_ms = 17
        result_source = "fallback"

        def __init__(self, *args, **kwargs):
            """接受生产构造参数，并准备一次GRAPH失败阶段事件。"""
            del args, kwargs
            _graph_plan, graph_diagnostics = no_graph_solution()
            self._events = [
                {
                    "source": "graph",
                    "plan": None,
                    "diagnostics": graph_diagnostics,
                }
            ]

        def consume_stage_events(self):
            """只返回一次GRAPH失败事件，模拟统一任务的日志队列。"""
            events = tuple(self._events)
            self._events.clear()
            return events

        def advance(self, **kwargs):
            """返回包含拒绝统计的确定失败结果。"""
            del kwargs
            self.done = True
            return assembly_planner.AssemblyPlan.failed(
                "fill_reject",
                search_nodes=self.search_nodes,
                diagnostics={
                    "fill_reject": 6,
                    "overlap_reject": 3,
                    "outer_edge_reject": 2,
                },
            )

        def cancel(self):
            """提供运行器异常清理所需接口。"""
            self.done = True

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(assembly_planner, "UnknownSolveJob", RejectedSolveJob)
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1, debug_enabled=True)

    plan = runtime.update(
        mode="unknown",
        unknown_profile="white",
        pieces=[_unknown_rectangle()],
        templates=[],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    output = capsys.readouterr().out
    assert plan.reason == "fill_reject"
    assert "[SOLVER] GRAPH result=FAIL" in output
    assert "fill_reject=2" in output
    assert "outer_reject=1" in output
    assert "[SOLVER] FALLBACK" in output
    assert "nodes=23" in output
    assert "fill_reject=6" in output
    assert "overlap_reject=3" in output
    assert "outer_reject=2" in output


def test_runtime_card_skips_white_graph_fast_path(monkeypatch):
    """CARD需要比较花纹，不能被WHITE的GRAPH或FOUR_FAST提前结束。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _patterned_irregular_four_pieces

    def forbidden_graph_path(*args, **kwargs):
        """CARD若错误调用WHITE图快路径则立即失败。"""
        del args, kwargs
        raise AssertionError("CARD不应调用WHITE图快路径")

    class PendingSolveJob:
        """模拟CARD已经进入既有跨帧纹理搜索。"""

        done = False
        search_nodes = 0

        def __init__(self, *args, **kwargs):
            """接受真实任务参数并保持未完成。"""
            del args, kwargs

        def advance(self, **kwargs):
            """当前时间片保持未完成。"""
            del kwargs
            return None

        def cancel(self):
            """提供运行器重置所需的取消接口。"""
            self.done = True

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        forbidden_graph_path,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        forbidden_graph_path,
    )
    monkeypatch.setattr(assembly_planner, "UnknownSolveJob", PendingSolveJob)
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)

    assert runtime.update(
        mode="unknown",
        unknown_profile="card",
        pieces=_patterned_irregular_four_pieces(),
        templates=[],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    ) is None
    assert runtime.is_solving is True


def test_runtime_white_graph_failure_falls_back_to_incremental_job(monkeypatch):
    """WHITE图候选全部未通过硬验收时必须进入同一任务内的增量兜底。"""
    from maixcam2_app_A_quad import assembly_planner

    graph_calls = []
    fallback_calls = []

    def no_graph_solution(*args, **kwargs):
        """模拟有界图快路径没有合法矩形。"""
        del args, kwargs
        graph_calls.append(True)
        return None, {"graph_fast_path": 0}

    def pending_fallback(*args, **kwargs):
        """登记进入旧搜索核心，并持续让出以保持任务跨帧运行。"""
        del args, kwargs
        fallback_calls.append(True)
        while True:
            yield None

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_layout_steps",
        pending_fallback,
    )
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)
    arguments = {
        "mode": "unknown",
        "unknown_profile": "white",
        "pieces": [_unknown_rectangle()],
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    # 第一帧结束GRAPH事件，第二帧才进入旧FALLBACK。
    assert runtime.update(**arguments) is None
    assert graph_calls == [True]
    assert fallback_calls == []
    assert runtime.update(**arguments) is None
    assert fallback_calls == [True]
    assert runtime.is_solving is True


def test_runtime_white_four_pieces_use_four_fast_before_fallback(monkeypatch):
    """WHITE四片在GRAPH失败后必须先运行FOUR_FAST，成功时不得创建旧任务。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    four_calls = []
    expected_plan = assembly_planner.AssemblyPlan(
        True,
        placements=[],
        target_rect_mm=(55.0, 176.0, 100.0, 60.0),
        reason="ok",
        diagnostics={"four_fast_path": 1},
    )

    def no_graph_solution(*args, **kwargs):
        """强制跳过整边GRAPH，以验证下一层路由。"""
        del args, kwargs
        return None, {"graph_fast_path": 0}

    def successful_four_fast(pieces, *args, **kwargs):
        """记录四片输入并返回确定成功规划。"""
        del args, kwargs
        four_calls.append(tuple(piece["id"] for piece in pieces))
        return expected_plan, {
            "four_fast_path": 1,
            "four_pair_states": 8,
            "four_triple_states": 4,
            "four_complete_states": 1,
            "four_work_units": 20,
        }

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
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)
    arguments = {
        "mode": "unknown",
        "unknown_profile": "white",
        "pieces": _field_four_pieces_from_device_log(),
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(**arguments) is None
    assert four_calls == []
    plan = runtime.update(**arguments)

    assert plan is expected_plan
    assert four_calls == [("U1", "U2", "U3", "U4")]
    assert runtime.is_solving is False


def test_runtime_white_four_fast_obeys_cross_frame_work_unit_budget():
    """WHITE四片快路径必须受运行器单帧工作单元上限控制。

    现场快照需要数百个候选；把单帧上限设为1后，第一次update只能锁定并推进一个
    工作单元，不能在同一帧同步返回结果。后续帧继续使用同一快照直至得到规划。
    """
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    pieces = _field_four_pieces_from_device_log()
    runtime = assembly_planner.AssemblyRuntime(
        stable_frames=1,
        solver_time_budget_ms=1000.0,
        solver_work_unit_limit=1,
        debug_enabled=False,
    )
    common = {
        "mode": "unknown",
        "unknown_profile": "white",
        "pieces": pieces,
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(**common) is None
    assert runtime.is_solving is True

    plan = None
    frame_count = 1
    while plan is None and frame_count < 3000:
        plan = runtime.update(**common)
        frame_count += 1

    assert plan is not None
    assert plan.success is True
    assert plan.diagnostics["four_fast_path"] == 1
    assert frame_count > 1
    assert runtime.is_solving is False


def test_runtime_four_fast_failure_preserves_incremental_fallback(monkeypatch):
    """FOUR_FAST达到上限或无解时必须继续同一任务内的增量FALLBACK。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    four_calls = []
    fallback_calls = []

    def no_graph_solution(*args, **kwargs):
        """强制进入四片快路径。"""
        del args, kwargs
        return None, {"graph_fast_path": 0}

    def no_four_solution(*args, **kwargs):
        """模拟FOUR_FAST检查完成但没有合法矩形。"""
        del args, kwargs
        four_calls.append(True)
        return None, {"four_fast_path": 0, "four_work_units": 12}

    def pending_fallback(*args, **kwargs):
        """登记FOUR_FAST失败后的兜底，并持续让出保持未完成。"""
        del args, kwargs
        fallback_calls.append(True)
        while True:
            yield None

    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        no_graph_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        no_four_solution,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_layout_steps",
        pending_fallback,
    )
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)
    arguments = {
        "mode": "unknown",
        "unknown_profile": "white",
        "pieces": _field_four_pieces_from_device_log(),
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    # 三帧分别结束GRAPH、结束FOUR_FAST、启动FALLBACK。
    assert runtime.update(**arguments) is None
    assert four_calls == []
    assert fallback_calls == []
    assert runtime.update(**arguments) is None
    assert four_calls == [True]
    assert fallback_calls == []
    assert runtime.update(**arguments) is None
    assert fallback_calls == [True]
    assert runtime.is_solving is True


def test_runtime_four_fast_macro_can_restore_old_fallback(monkeypatch):
    """关闭现场宏后WHITE四片必须完全跳过FOUR_FAST并进入旧FALLBACK。"""
    from maixcam2_app_A_quad import assembly_planner
    from tests_ab.test_a_unknown_planner import _field_four_pieces_from_device_log

    fallback_calls = []

    def forbidden_four_fast(*args, **kwargs):
        """关闭宏后若仍调用四片快路径则立即失败。"""
        del args, kwargs
        raise AssertionError("UNKNOWN_FOUR_FAST_ENABLED=False时不应调用快路径")

    def pending_fallback(*args, **kwargs):
        """记录关闭FOUR_FAST后统一任务是否直接进入旧搜索核心。"""
        del args, kwargs
        fallback_calls.append(True)
        while True:
            yield None

    monkeypatch.setattr(assembly_planner, "UNKNOWN_FOUR_FAST_ENABLED", False)
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        lambda *args, **kwargs: (None, {"graph_fast_path": 0}),
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_four_fast_path",
        forbidden_four_fast,
    )
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_layout_steps",
        pending_fallback,
    )
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)
    arguments = {
        "mode": "unknown",
        "unknown_profile": "white",
        "pieces": _field_four_pieces_from_device_log(),
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(**arguments) is None
    assert fallback_calls == []
    assert runtime.update(**arguments) is None
    assert fallback_calls == [True]


def test_runtime_context_change_cancels_incremental_unknown_job():
    """模式或机械上下文变化时必须取消未完成任务，不能让旧结果污染新模式。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from tests_ab.test_a_unknown_planner import _patterned_irregular_four_pieces

    runtime = AssemblyRuntime(
        stable_frames=1,
        solver_time_budget_ms=1000.0,
        solver_work_unit_limit=1,
    )
    common = {
        # 本用例专门验证未完成增量任务的取消，因此显式使用保留旧任务路径的CARD。
        "unknown_profile": "card",
        "pieces": _patterned_irregular_four_pieces(),
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }
    assert runtime.update(mode="unknown", **common) is None
    assert runtime.is_solving is True

    runtime.reset()

    assert runtime.is_solving is False
    assert runtime.plan is None
    assert runtime.stable_count == 0


def test_runtime_unknown_profile_changes_context_and_white_stops_at_first_solution(
    monkeypatch,
):
    """WHITE必须启用首解停止；切到CARD必须取消旧任务并创建继续择优的新任务。"""
    from maixcam2_app_A_quad import assembly_planner

    created_jobs = []

    class PendingSolveJob:
        """记录求解构造参数并持续挂起，便于观察子模式切换是否取消旧上下文。"""

        done = False
        search_nodes = 0
        edge_candidates = 0
        max_frontier_width = 0
        first_solution_node = None

        def __init__(self, *args, **kwargs):
            """保存stop_at_first_solution参数，并登记当前任务实例。"""
            del args
            self.stop_at_first_solution = bool(kwargs["stop_at_first_solution"])
            self.cancelled = False
            created_jobs.append(self)

        def advance(self, **kwargs):
            """模拟跨帧任务尚未完成。"""
            del kwargs
            return None

        def cancel(self):
            """记录上下文切换对旧任务的取消动作。"""
            self.cancelled = True
            self.done = True

    monkeypatch.setattr(assembly_planner, "UnknownSolveJob", PendingSolveJob)
    monkeypatch.setattr(
        assembly_planner,
        "_solve_unknown_graph_fast_path",
        lambda *args, **kwargs: (None, {"graph_fast_path": 0}),
    )
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)
    arguments = {
        "mode": "unknown",
        "pieces": [_unknown_rectangle()],
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(unknown_profile="white", **arguments) is None
    assert created_jobs[0].stop_at_first_solution is True

    assert runtime.update(unknown_profile="card", **arguments) is None
    assert created_jobs[0].cancelled is True
    assert created_jobs[1].stop_at_first_solution is False


def test_runtime_returns_solver_error_when_incremental_job_raises(monkeypatch):
    """UNKNOWN工作单元异常必须转换为失败规划，不能穿透相机主循环。"""
    from maixcam2_app_A_quad import assembly_planner

    class RaisingSolveJob:
        """模拟底层OpenCV工作单元抛出运行异常的增量任务。"""

        done = False
        search_nodes = 0

        def __init__(self, *args, **kwargs):
            """接受真实任务构造参数，但不执行几何初始化。"""
            del args, kwargs

        def advance(self, **kwargs):
            """模拟某个不可恢复的底层求解异常。"""
            del kwargs
            raise RuntimeError("solver failed")

        def cancel(self):
            """提供运行器清理任务所需的取消接口。"""
            self.done = True

    monkeypatch.setattr(assembly_planner, "UnknownSolveJob", RaisingSolveJob)
    runtime = assembly_planner.AssemblyRuntime(stable_frames=1)

    plan = runtime.update(
        mode="unknown",
        # 本用例验证增量任务异常边界，使用CARD确保不会被WHITE图快路径提前完成。
        unknown_profile="card",
        pieces=[_unknown_rectangle()],
        templates=[],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "solver_error"
    assert runtime.is_solving is False


def test_runtime_timeout_without_solution_keeps_locked_snapshot_until_manual_reset(
    monkeypatch,
):
    """无首解超时必须锁定失败和原快照，不能受后续抖动影响而自动重新识别。"""
    from maixcam2_app_A_quad import assembly_planner

    class TimeoutSolveJob:
        """模拟一启动就达到截止线、且进度中没有任何合法规划的UNKNOWN任务。"""

        done = False
        search_nodes = 23
        edge_candidates = 34
        max_frontier_width = 13
        first_solution_node = None

        def __init__(self, *args, **kwargs):
            """接受真实构造参数，保持测试只替换求解结果而不改变运行器入口。"""
            del args, kwargs

        def advance(self, **kwargs):
            """返回不含机械目标的结构化超时，并把任务标记为结束。"""
            del kwargs
            self.done = True
            return assembly_planner.AssemblyPlan.failed(
                "solver_timeout",
                search_nodes=self.search_nodes,
            )

        def cancel(self):
            """提供运行器异常或上下文重置时需要的安全取消接口。"""
            self.done = True

    monkeypatch.setattr(assembly_planner, "UnknownSolveJob", TimeoutSolveJob)
    runtime = assembly_planner.AssemblyRuntime(stable_frames=3)
    arguments = {
        "mode": "unknown",
        # 本用例模拟UnknownSolveJob超时，必须显式进入CARD增量路径。
        "unknown_profile": "card",
        "pieces": [_unknown_rectangle()],
        "templates": [],
        "work_region_mm": (0.0, 33.5, 210.0, 230.0),
        "split_y_mm": 148.5,
    }

    assert runtime.update(**arguments) is None
    assert runtime.update(**arguments) is None
    timeout_result = runtime.update(**arguments)

    assert timeout_result.reason == "solver_timeout"
    assert runtime.plan is timeout_result
    assert runtime.is_solving is False
    assert runtime.stable_count == 3
    assert runtime.snapshot_locked is True
    assert len(runtime.locked_pieces) == 1

    # 后续实时轮廓即使明显移动也只能返回同一个失败，用户显式reset后才允许重新采集。
    moved_arguments = dict(arguments)
    moved_arguments["pieces"] = [_unknown_rectangle(center=(150.0, 100.0))]
    assert runtime.update(**moved_arguments) is timeout_result
    assert runtime.stable_count == 3
    assert runtime.solve_count == 1

    runtime.reset()

    assert runtime.plan is None
    assert runtime.snapshot_locked is False
    assert runtime.locked_pieces == ()


def test_runtime_damaged_template_fingerprint_returns_layout_failure():
    """损坏目标顶点不得在上下文指纹阶段抛异常，应返回模板布局失败。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime
    from tests_ab.test_a_known_planner import _registered_layout

    templates, pieces = _registered_layout()
    for piece in pieces:
        piece["region"] = "upper"
    templates[0]["target_vertices_mm"] = "damaged"
    runtime = AssemblyRuntime(stable_frames=1)

    plan = runtime.update(
        mode="known",
        pieces=pieces,
        templates=templates,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "template_layout_invalid"


def test_runtime_ignores_lower_or_crossing_pieces_before_solve():
    """只有完整上半区碎片可以触发自动拼装规划。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    runtime = AssemblyRuntime(stable_frames=1)
    piece = _unknown_rectangle()
    piece["region"] = "lower"

    plan = runtime.update(
        mode="unknown",
        pieces=[piece],
        templates=[],
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan is None
    assert runtime.solve_count == 0


def test_draw_assembly_plan_renders_split_target_contours_and_arrow():
    """正常界面必须用红线和下半区目标位姿直观展示规划结果。"""
    from maixcam2_app_A_quad.assembly_planner import (
        AssemblyPlacement,
        AssemblyPlan,
        COLOR_PLAN_ARROW,
        COLOR_PLAN_SPLIT,
        COLOR_PLAN_TARGET,
        draw_assembly_plan,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    placement = AssemblyPlacement(
        "U1",
        source_center_mm=(80.0, 80.0),
        target_center_mm=(105.0, 206.0),
        target_polygon_mm=((55, 176), (155, 176), (155, 236), (55, 236)),
        rotation_delta_deg=37.0,
    )
    plan = AssemblyPlan(
        True,
        placements=[placement],
        target_rect_mm=(55.0, 176.0, 100.0, 60.0),
        score=0.0,
        reason="ok",
    )

    output = draw_assembly_plan(
        frame,
        plan,
        DEFAULT_PAPER_QUAD,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert output.shape == frame.shape
    assert np.array_equal(frame, np.zeros_like(frame))
    assert np.any(np.all(output == np.asarray(COLOR_PLAN_SPLIT), axis=2))
    assert np.any(np.all(output == np.asarray(COLOR_PLAN_TARGET), axis=2))
    assert np.any(np.all(output == np.asarray(COLOR_PLAN_ARROW), axis=2))


def test_main_source_connects_runtime_update_and_plan_overlay():
    """实际部署入口必须调用稳定运行器并把规划对象传入正常叠加层。"""
    from maixcam2_app_A_quad import main

    source = Path(main.__file__).read_text(encoding="utf-8")

    assert "planner_runtime = AssemblyRuntime(" in source
    assert "planner_runtime.update(" in source
    assert "assembly_plan=assembly_plan" in source
    assert "planner_runtime.reset(" in source


def test_planning_status_displays_incremental_solver_progress():
    """增量任务状态必须同时显示节点、边图、前沿和是否已有首解。"""
    from maixcam2_app_A_quad.main import select_planning_status

    status = select_planning_status(
        "UNKNOWN MODE",
        assembly_plan=None,
        stable_count=3,
        stable_frames=3,
        solving=True,
        search_nodes=17,
        edge_candidates=34,
        max_frontier_width=13,
        first_solution_node=9,
    )

    assert status == "SOLVING N=17 E=34 F=13 S=1"


def test_planning_status_marks_solver_and_result_as_locked_snapshot():
    """锁定后状态必须明确显示LOCKED，避免用户把实时红点误认为求解输入。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyPlan
    from maixcam2_app_A_quad.main import select_planning_status

    solving_status = select_planning_status(
        "UNKNOWN MODE",
        assembly_plan=None,
        stable_count=3,
        stable_frames=3,
        solving=True,
        search_nodes=17,
        edge_candidates=34,
        max_frontier_width=13,
        first_solution_node=None,
        snapshot_locked=True,
    )
    timeout_status = select_planning_status(
        solving_status,
        assembly_plan=AssemblyPlan.failed("solver_timeout"),
        stable_count=3,
        stable_frames=3,
        snapshot_locked=True,
    )

    assert solving_status == "LOCKED SOLVING N=17 E=34 F=13 S=0"
    assert timeout_status == "LOCKED PLAN SOLVER_TIMEOUT"


def test_planning_status_keeps_save_failure_until_next_user_action():
    """SAVE失败不能在下一帧被STABLE覆盖，否则现场来不及读取具体原因。"""
    from maixcam2_app_A_quad.main import select_planning_status

    status = select_planning_status(
        "SAVE KNOWN_NEEDS_FOUR",
        assembly_plan=None,
        stable_count=1,
        stable_frames=3,
    )

    assert status == "SAVE KNOWN_NEEDS_FOUR"


def test_a_known_layout_uses_persistent_path_outside_maixvision_tmp():
    """赛前保存一次的已知布局必须位于/root持久目录，不能依赖/tmp运行目录。"""
    from maixcam2_app_A_quad import config, main

    source = Path(main.__file__).read_text(encoding="utf-8")

    assert config.PERSISTENT_TEMPLATE_PATH == (
        "/root/maixcam2_puzzle_A/known_templates.json"
    )
    assert "template_path = PERSISTENT_TEMPLATE_PATH" in source
