"""验证A版毫米多边形、上下区分类和已知四片快速规划。"""

import math

import cv2
import numpy as np
import pytest

from maixcam2_app_A_quad import main, paper_locator, template_store
from maixcam2_app_A_quad.config import DEFAULT_CONFIG
from maixcam2_app_A_quad.puzzle_vision import compute_piece_geometry
from maixcam2_app_A_quad.settings_store import build_default_runtime_settings
from tests_ab.synthetic_paper import DEFAULT_PAPER_QUAD, make_quad_scene_with_four_pieces


TARGET_LAYOUT = (
    ((0.0, 0.0), (15.0, 0.0), (20.0, 60.0), (0.0, 60.0)),
    ((15.0, 0.0), (40.0, 0.0), (35.0, 60.0), (20.0, 60.0)),
    ((40.0, 0.0), (70.0, 0.0), (75.0, 60.0), (35.0, 60.0)),
    ((70.0, 0.0), (100.0, 0.0), (100.0, 60.0), (75.0, 60.0)),
)


def _polygon_centroid(vertices):
    """使用OpenCV图像矩计算测试毫米多边形质心。"""
    contour = np.asarray(vertices, dtype=np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _piece_from_mm(vertices_mm):
    """从毫米顶点构造带真实轮廓特征的碎片字典，供模板描述子测试使用。"""
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    contour = np.rint((vertices_mm + np.asarray((140.0, 140.0))) * 3.0).astype(
        np.int32
    ).reshape(-1, 1, 2)
    piece = compute_piece_geometry(
        contour,
        roi=(0, 0, 1000, 1000),
        config=DEFAULT_CONFIG,
    )
    piece["vertices_mm"] = vertices_mm.astype(float).tolist()
    piece["center_mm"] = _polygon_centroid(vertices_mm)
    piece["region"] = "lower"
    return piece


def _transform_piece(piece, angle_deg, target_center):
    """对模板毫米多边形施加无缩放刚体变换，模拟上半区随机位置与角度。"""
    vertices = np.asarray(piece["vertices_mm"], dtype=np.float64)
    center = np.asarray(piece["center_mm"], dtype=np.float64)
    angle_rad = math.radians(float(angle_deg))
    rotation = np.asarray(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ]
    )
    transformed = (vertices - center) @ rotation.T + np.asarray(target_center)
    return _piece_from_mm(transformed)


def _registered_layout():
    """登记精确100×60mm四片目标布局并返回模板和原始目标碎片。"""
    from maixcam2_app_A_quad.assembly_planner import register_known_layout

    pieces = [_piece_from_mm(vertices) for vertices in TARGET_LAYOUT]
    return register_known_layout(pieces), pieces


def _random_upper_known_pieces(with_pattern=False):
    """构造四片随机位姿的KNOWN登记输入，可选附加高能量牌面边特征。

    主要流程：从固定100×60mm目标布局生成真实形状描述子，分别施加纸面内旋转和平移，
    最后标记为上半区。返回值为四个可直接送入KNOWN登记任务的碎片字典。
    """
    target_pieces = [_piece_from_mm(vertices) for vertices in TARGET_LAYOUT]
    angles = (31.0, -47.0, 82.0, -19.0)
    centers = ((35.0, 72.0), (82.0, 78.0), (133.0, 70.0), (160.0, 80.0))
    pieces = [
        _transform_piece(piece, angle, center)
        for piece, angle, center in zip(target_pieces, angles, centers)
    ]
    for piece in pieces:
        piece["region"] = "upper"
        if with_pattern:
            # 所有边跨过0.05纹理门槛，稳定复现旧SAVE同步跑满搜索的问题。
            piece["edge_features"] = [
                {
                    "colors": [[20, 20, 20], [80, 80, 80], [160, 160, 160]],
                    "gradients": [60.0, 70.0, 80.0],
                    "pattern_energy": 0.8,
                }
                for _ in piece["vertices_mm"]
            ]
    return pieces


def _correct_lower_known_pieces():
    """把固定100×60mm四片正确布局平移到红线下方，作为赛前SAVE输入。"""
    offset = np.asarray((55.0, 176.0), dtype=np.float64)
    pieces = [
        _piece_from_mm(np.asarray(vertices, dtype=np.float64) + offset)
        for vertices in TARGET_LAYOUT
    ]
    for piece in pieces:
        # 模拟视觉层已经确认完整的SAVE快照；假像素轮廓不参与下半区边界判断。
        piece["complete"] = True
        piece["region"] = "lower"
    return pieces


def _rotate_vertices(vertices, angle_deg, target_center):
    """将原始毫米顶点绕自身均值旋转并平移，保留测试指定的4/5顶点数量。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    angle_rad = math.radians(float(angle_deg))
    rotation = np.asarray(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=np.float64,
    )
    return (
        (vertices - np.mean(vertices, axis=0)) @ rotation.T
        + np.asarray(target_center, dtype=np.float64)
    )


def test_batch_image_points_to_paper_mm_reverses_perspective():
    """毫米顶点批量反算必须逐点恢复透视前的完整A4坐标。"""
    expected_mm = np.float32([[12, 44], [98, 60], [180, 210], [33, 250]])
    image_points = paper_locator.paper_points_to_image_px(
        expected_mm,
        DEFAULT_PAPER_QUAD,
    )

    recovered = paper_locator.image_points_to_paper_mm(
        image_points,
        DEFAULT_PAPER_QUAD,
    )

    np.testing.assert_allclose(recovered, expected_mm, atol=0.05)


def test_quad_analysis_adds_mm_vertices_and_classifies_both_regions():
    """A版单帧必须按全部毫米顶点把碎片分成上半区、下半区或跨线。"""
    frame, paper_quad, _ = make_quad_scene_with_four_pieces()
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["paper_quad"] = paper_quad.astype(float).tolist()

    pieces = main.analyze_quad_frame(frame, settings).detection.pieces

    assert len(pieces) == 4
    assert all(len(piece["vertices_mm"]) == piece["vertex_count"] for piece in pieces)
    assert [piece["region"] for piece in pieces].count("upper") == 2
    assert [piece["region"] for piece in pieces].count("lower") == 2


@pytest.mark.parametrize(
    ("vertices", "expected"),
    [
        (((10, 100), (30, 100), (30, 120)), "upper"),
        (((10, 170), (30, 170), (30, 190)), "lower"),
        (((10, 140), (30, 140), (30, 160)), "crossing"),
    ],
)
def test_region_classification_uses_all_vertices(vertices, expected):
    """只要多边形跨越分界线就不能仅凭质心误判为可操作上片或下片。"""
    from maixcam2_app_A_quad.assembly_planner import classify_piece_region

    assert classify_piece_region(vertices, split_y_mm=148.5) == expected


def test_register_known_layout_persists_shape_and_target_polygon(tmp_path):
    """赛前登记必须把每片描述子和100×60目标局部多边形一起持久化。"""
    templates, _ = _registered_layout()
    template_path = tmp_path / "known_templates.json"

    template_store.save_templates(template_path, templates)
    loaded = template_store.load_templates(template_path)

    assert loaded == templates
    assert len(loaded) == 4
    assert {item["id"] for item in loaded} == {"K1", "K2", "K3", "K4"}
    assert all("target_vertices_mm" in item for item in loaded)
    assert all(item["layout_size_mm"] == [100.0, 60.0] for item in loaded)


def test_known_solver_matches_random_poses_and_places_fixed_rect_below_split():
    """四片初始位姿随机时，最多24种全局分配后必须直接输出固定下半区目标。"""
    from maixcam2_app_A_quad.assembly_planner import solve_known_layout

    templates, target_pieces = _registered_layout()
    observations = [
        _transform_piece(target_pieces[2], 37.0, (45.0, 75.0)),
        _transform_piece(target_pieces[0], -51.0, (95.0, 80.0)),
        _transform_piece(target_pieces[3], 73.0, (150.0, 72.0)),
        _transform_piece(target_pieces[1], -18.0, (185.0, 82.0)),
    ]

    plan = solve_known_layout(
        observations,
        templates,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_match_score=1.2,
    )

    assert plan.success is True
    assert plan.reason == "ok"
    assert plan.target_rect_mm == pytest.approx((55.0, 176.0, 100.0, 60.0))
    assert len(plan.placements) == 4
    assert {placement.piece_id for placement in plan.placements} == {
        "K1",
        "K2",
        "K3",
        "K4",
    }
    assert all(placement.target_center_mm[1] > 148.5 for placement in plan.placements)
    assert all(-180.0 <= placement.rotation_delta_deg < 180.0 for placement in plan.placements)


def test_known_contour_match_accepts_two_mm_bump_with_extra_vertex():
    """同一片边缘多出2mm反光毛刺时应继续匹配模板，不能因4变5顶点判UNKNOWN。"""
    base_vertices = np.asarray(
        ((0.0, 0.0), (42.0, 0.0), (36.0, 58.0), (5.0, 46.0)),
        dtype=np.float64,
    )
    bumped_vertices = np.asarray(
        ((0.0, 0.0), (21.0, -2.0), (42.0, 0.0), (36.0, 58.0), (5.0, 46.0)),
        dtype=np.float64,
    )
    template_piece = _piece_from_mm(base_vertices)
    observation = _piece_from_mm(
        _rotate_vertices(bumped_vertices, 37.0, (92.0, 74.0))
    )
    template = {
        "id": "K1",
        **template_store.build_shape_descriptor(template_piece),
        "target_vertices_mm": base_vertices.astype(float).tolist(),
        "layout_size_mm": [100.0, 60.0],
    }

    template_store.match_known_pieces(
        [observation],
        [template],
        max_score=1.60,
    )

    assert observation["id"] == "K1"
    assert observation["match_score"] <= 1.60


def test_known_contour_distance_rejects_mirrored_asymmetric_piece():
    """无镜像轮廓距离必须拒绝翻面形状，适度放宽阈值不能牺牲K编号安全。"""
    original = np.asarray(
        ((0.0, 0.0), (52.0, 4.0), (43.0, 23.0), (55.0, 61.0), (7.0, 48.0)),
        dtype=np.float64,
    )
    mirrored = original.copy()
    mirrored[:, 0] = np.max(original[:, 0]) - original[:, 0]

    score = template_store.contour_shape_distance(original, mirrored)

    assert score > 1.60


def test_known_pose_alignment_accepts_four_to_five_vertex_jitter():
    """姿态配准应先重采样再求旋转，使带毛刺的5点观测仍能对准4点目标。"""
    from maixcam2_app_A_quad.assembly_planner import _best_rotation_delta_deg

    target = np.asarray(
        ((0.0, 0.0), (42.0, 0.0), (36.0, 58.0), (5.0, 46.0)),
        dtype=np.float64,
    )
    bumped = np.asarray(
        ((0.0, 0.0), (21.0, -2.0), (42.0, 0.0), (36.0, 58.0), (5.0, 46.0)),
        dtype=np.float64,
    )
    source = _rotate_vertices(bumped, 37.0, (120.0, 80.0))

    rotation_delta, alignment_error = _best_rotation_delta_deg(source, target)

    assert rotation_delta == pytest.approx(-37.0, abs=2.0)
    assert alignment_error < 2.0


def test_known_solver_rejects_legacy_templates_without_layout():
    """只有形状描述子的旧模板不能伪造目标位置，必须显示缺少布局。"""
    from maixcam2_app_A_quad.assembly_planner import solve_known_layout

    pieces = [_piece_from_mm(vertices) for vertices in TARGET_LAYOUT]
    legacy_templates = template_store.register_templates(pieces)
    for template in legacy_templates:
        template.pop("target_vertices_mm", None)
        template.pop("layout_size_mm", None)

    plan = solve_known_layout(
        pieces,
        legacy_templates,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "template_no_layout"
    assert plan.placements == []


def test_known_solver_contains_corrupt_layout_fields_as_failed_plan():
    """持久模板的新布局字段损坏时必须返回失败规划，不能让设备主循环抛异常。"""
    from maixcam2_app_A_quad.assembly_planner import solve_known_layout

    templates, pieces = _registered_layout()
    templates[0]["layout_size_mm"] = "damaged"

    plan = solve_known_layout(
        pieces,
        templates,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "template_layout_invalid"
    assert plan.placements == []


def test_known_solver_rejects_lower_region_too_small_for_target():
    """下半区不足100×60mm时不得输出越界机械目标。"""
    from maixcam2_app_A_quad.assembly_planner import solve_known_layout

    templates, pieces = _registered_layout()

    plan = solve_known_layout(
        pieces,
        templates,
        work_region_mm=(0.0, 33.5, 90.0, 100.0),
        split_y_mm=80.0,
    )

    assert plan.success is False
    assert plan.reason == "target_out_of_work_region"


def test_main_directly_saves_correct_lower_known_layout(tmp_path):
    """下半区正确100×60mm布局必须同步登记，不能再运行未知拼图搜索。"""
    pieces = _correct_lower_known_pieces()
    template_path = tmp_path / "known_templates.json"

    plan, templates = main.register_and_save_known_layout(
        pieces,
        template_path,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is True
    assert 1 <= plan.search_nodes <= 24
    assert template_store.load_templates(template_path) == templates
    assert all("target_vertices_mm" in template for template in templates)


def test_main_rejects_upper_known_save_with_specific_layout_message(tmp_path):
    """上半区随机四片不是登记姿态，SAVE必须提示先放到下半区正确拼好。"""
    template_path = tmp_path / "known_templates.json"

    plan, templates = main.register_and_save_known_layout(
        _random_upper_known_pieces(),
        template_path,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "known_layout_must_be_lower"
    assert templates == []
    assert not template_path.exists()


def test_direct_known_save_never_constructs_incremental_registration_job(
    tmp_path,
    monkeypatch,
):
    """设备SAVE入口必须直接登记下半区布局，禁止回到UnknownSolveJob慢路径。"""
    from maixcam2_app_A_quad import assembly_planner
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    class ForbiddenRegistrationJob:
        """只要旧增量登记任务被构造就立即让测试失败。"""

        def __init__(self, *args, **kwargs):
            """拒绝构造旧任务；参数仅用于兼容原签名。"""
            del args, kwargs
            raise AssertionError("KNOWN SAVE不得构造KnownRegistrationJob")

    monkeypatch.setattr(
        assembly_planner,
        "KnownRegistrationJob",
        ForbiddenRegistrationJob,
    )
    template_path = tmp_path / "known_templates.json"
    runtime = AssemblyRuntime(stable_frames=3)

    templates, plan, status = main.perform_known_save_action(
        _correct_lower_known_pieces(),
        current_templates=[],
        template_path=template_path,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        planner_runtime=runtime,
    )

    assert status == "KNOWN SAVED PLAN OK"
    assert plan.success is True
    assert 1 <= plan.search_nodes <= 24
    assert runtime.plan is plan
    assert runtime.snapshot_locked is True
    assert len(runtime.locked_pieces) == 4
    assert all(piece["region"] == "lower" for piece in runtime.locked_pieces)
    assert template_store.load_templates(template_path) == templates


def test_direct_known_save_rejects_overlapping_incomplete_rectangle(tmp_path):
    """外框仍接近100×60但碎片明显重叠并留下大洞时不得保存伪布局。"""
    pieces = _correct_lower_known_pieces()
    pieces[1] = _piece_from_mm(pieces[0]["vertices_mm"])
    pieces[1]["complete"] = True
    pieces[1]["region"] = "lower"
    template_path = tmp_path / "known_templates.json"

    plan, templates = main.register_and_save_known_layout(
        pieces,
        template_path,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "known_layout_invalid"
    assert templates == []
    assert not template_path.exists()


def test_main_known_save_rejects_fewer_than_four_upper_pieces_without_writing(tmp_path):
    """上半区可操作碎片不足四片时必须返回具体失败且不得创建模板文件。"""
    pieces = [_piece_from_mm(vertices) for vertices in TARGET_LAYOUT]
    for piece in pieces:
        piece["region"] = "upper"
    template_path = tmp_path / "known_templates.json"

    plan, templates = main.register_and_save_known_layout(
        pieces[:3],
        template_path,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert plan.success is False
    assert plan.reason == "known_needs_four"
    assert templates == []
    assert not template_path.exists()


def test_known_registration_job_yields_then_builds_templates_incrementally():
    """KNOWN登记必须能跨多次短推进恢复，并在完成后同时给出规划和四个模板。"""
    from maixcam2_app_A_quad.assembly_planner import KnownRegistrationJob

    job = KnownRegistrationJob(
        _random_upper_known_pieces(with_pattern=True),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        max_nodes=12000,
        texture_refinement_nodes=80,
    )

    first_result = job.advance(time_budget_ms=1000.0, work_unit_limit=1)

    assert first_result is None
    assert job.done is False
    assert job.templates == []

    result = None
    for _ in range(3000):
        result = job.advance(time_budget_ms=1000.0, work_unit_limit=32)
        if result is not None:
            break

    assert result is not None
    assert result.success is True
    assert job.done is True
    assert len(job.templates) == 4


def test_known_save_controller_writes_only_after_incremental_job_finishes(tmp_path):
    """SAVE启动和未完成时间片不得写文件，成功完成后才保存模板并缓存规划。"""
    from maixcam2_app_A_quad.assembly_planner import AssemblyRuntime

    template_path = tmp_path / "known_templates.json"
    planner_runtime = AssemblyRuntime(stable_frames=3)
    controller = main.KnownSaveController(
        time_budget_ms=1000.0,
        work_unit_limit=1,
        texture_refinement_nodes=80,
    )
    status = controller.start(
        _random_upper_known_pieces(with_pattern=True),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    assert status == "SAVE SOLVING N=0"
    assert controller.active is True
    assert not template_path.exists()

    templates, plan, status = controller.advance(
        current_templates=[],
        template_path=template_path,
        planner_runtime=planner_runtime,
    )

    assert templates == []
    assert plan is None
    assert status.startswith("SAVE SOLVING N=")
    assert not template_path.exists()

    for _ in range(10000):
        templates, plan, status = controller.advance(
            current_templates=templates,
            template_path=template_path,
            planner_runtime=planner_runtime,
        )
        if not controller.active:
            break

    assert controller.active is False
    assert plan is not None
    assert plan.success is True
    assert status == "KNOWN SAVED PLAN OK"
    assert template_store.load_templates(template_path) == templates
    assert planner_runtime.plan is plan


def test_no_template_hint_does_not_overwrite_known_save_status():
    """没有旧模板时，SAVE失败或求解进度必须优先显示，不能被NO TEMPLATE覆盖。"""
    assert main.select_known_template_status(
        templates=[],
        current_status="SAVE KNOWN_NEEDS_FOUR",
        save_active=False,
    ) == "SAVE KNOWN_NEEDS_FOUR"
    assert main.select_known_template_status(
        templates=[],
        current_status="SAVE SOLVING N=17",
        save_active=True,
    ) == "SAVE SOLVING N=17"
    assert main.select_known_template_status(
        templates=[],
        current_status="KNOWN MODE",
        save_active=False,
    ) == "NO TEMPLATE"


def test_safe_known_matching_converts_bad_descriptor_to_template_error():
    """损坏模板描述子导致匹配异常时必须清空内存模板并返回可见错误状态。"""
    templates, pieces = _registered_layout()
    templates[0]["edge_ratios"] = "damaged"

    kept_templates, status = main.match_known_pieces_safely(
        pieces,
        templates,
        max_score=1.2,
    )

    assert kept_templates == []
    assert status == "TEMPLATE ERROR ValueError"
    assert all(piece["id"] == "UNKNOWN" for piece in pieces)


def test_known_save_cache_failure_keeps_old_template_file(tmp_path):
    """规划缓存失败必须发生在写盘前，确保旧模板文件和旧内存模板保持一致。"""
    old_templates, _ = _registered_layout()
    template_path = tmp_path / "known_templates.json"
    template_store.save_templates(template_path, old_templates)

    class FailingPlannerRuntime:
        """模拟模板可生成但运行规划缓存失败的极端异常。"""

        def cache_plan(self, *args, **kwargs):
            """在任何文件写入之前抛出缓存异常。"""
            del args, kwargs
            raise RuntimeError("cache failed")

    controller = main.KnownSaveController(
        time_budget_ms=1000.0,
        work_unit_limit=32,
        texture_refinement_nodes=80,
    )
    controller.start(
        _random_upper_known_pieces(with_pattern=True),
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
    )

    status = ""
    for _ in range(3000):
        _, _, status = controller.advance(
            current_templates=old_templates,
            template_path=template_path,
            planner_runtime=FailingPlannerRuntime(),
        )
        if not controller.active:
            break

    assert status == "SAVE ERROR RuntimeError"
    assert template_store.load_templates(template_path) == old_templates


def test_known_save_controller_converts_job_exception_to_status(tmp_path):
    """KNOWN增量工作单元异常必须取消任务并显示SAVE ERROR，不能退出主循环。"""
    class RaisingRegistrationJob:
        """模拟执行期间抛出异常的KNOWN登记任务。"""

        done = False
        search_nodes = 7
        templates = []

        def advance(self, **kwargs):
            """模拟OpenCV或NumPy内部异常。"""
            del kwargs
            raise RuntimeError("registration failed")

        def cancel(self):
            """记录任务已被控制器终止。"""
            self.done = True

    controller = main.KnownSaveController()
    controller._job = RaisingRegistrationJob()
    controller._work_region_mm = (0.0, 33.5, 210.0, 230.0)
    controller._split_y_mm = 148.5

    templates, plan, status = controller.advance(
        current_templates=[],
        template_path=tmp_path / "known_templates.json",
        planner_runtime=object(),
    )

    assert templates == []
    assert plan is None
    assert status == "SAVE ERROR RuntimeError"
    assert controller.active is False
