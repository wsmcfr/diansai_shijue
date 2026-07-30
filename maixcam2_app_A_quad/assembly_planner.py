"""A版拼图的毫米区域判断、布局求解、稳定缓存和目标位姿绘制。"""

import copy
import itertools
import math
import time

import cv2
import numpy as np


# ======================== 现场调试常量（用户可直接修改） ========================
# 第一轮严格矩形验收的最低填充率。提高会减少误接受，降低会容忍更多角点缺口。
UNKNOWN_STRICT_MIN_FILL_RATIO = 0.92
# 仅WHITE严格轮失败后使用的最低填充率；尺寸、重叠和逐片外边约束不会随之放宽。
UNKNOWN_RELAXED_MIN_FILL_RATIO = 0.86
# 旧CLEAN诊断使用的短边门槛；生产求解已改为下面的多候选评分，不再破坏性删点。
UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM = 12.0
# 每片最多保留的接缝候选数；增大会提高异常轮廓召回，也会扩大后续边图。
UNKNOWN_SHAPE_MAX_HYPOTHESES = 3
# 题目规定真实边不短于20mm，该值用于候选短边惩罚和现场诊断。
UNKNOWN_REAL_EDGE_MIN_MM = 20.0
# 考虑Homography和远距离角点误差后的候选硬下限；低于该值默认视为伪边。
UNKNOWN_EDGE_HARD_FLOOR_MM = 14.0
# 简化候选相对高保真完整轮廓必须保留的最小面积比例。
UNKNOWN_SHAPE_AREA_RETENTION_MIN = 0.96
# 简化候选边界相对完整轮廓允许的最大双向偏差，单位mm。
UNKNOWN_SHAPE_MAX_DEVIATION_MM = 3.0
# True时在WHITE四片的整边GRAPH失败后启用分层Beam快路径；False可恢复旧FALLBACK。
UNKNOWN_FOUR_FAST_ENABLED = True
# FOUR_FAST每层全局保留的状态数；增大可提高复杂轮廓召回，但会增加候选检查量。
UNKNOWN_FOUR_FAST_BEAM_WIDTH = 32
# 每个有向片对只展开排序最前的关系；现场正确T形接缝排名为2和5，默认6完整覆盖。
UNKNOWN_FOUR_FAST_RELATION_LIMIT = 6
# FOUR_FAST单次最多检查的对齐候选数；中间层触顶会转FALLBACK，完整层触顶仍会
# 验收已经生成的完整候选，候选无解时再转FALLBACK。
UNKNOWN_FOUR_FAST_MAX_WORK_UNITS = 2400
# FOURFAST独立CPU活动预算，单位秒；到期只结束四片快路径并转FALLBACK，不结束任务。
UNKNOWN_FOUR_FAST_ACTIVE_BUDGET_SECONDS = 1.5
# True时向电脑串口/控制台输出一次性求解诊断；调试结束后改为False即可关闭。
UNKNOWN_SOLVER_DEBUG = True
# ============================================================================

try:
    from maixcam2_app_A_quad.paper_locator import (
        PAPER_ORIENTATION_PORTRAIT,
        build_split_segment,
        paper_points_to_image_px,
        validate_split_y_mm,
        validate_work_region_mm,
    )
    from maixcam2_app_A_quad.template_store import (
        CONTOUR_SAMPLE_COUNT,
        align_closed_contours,
        build_shape_descriptor,
        match_known_pieces,
        resample_closed_contour,
    )
except ModuleNotFoundError as error:
    # MaixVision平铺运行时顶层包不存在，改用main.py同级模块。
    if error.name != "maixcam2_app_A_quad":
        raise
    from paper_locator import (
        PAPER_ORIENTATION_PORTRAIT,
        build_split_segment,
        paper_points_to_image_px,
        validate_split_y_mm,
        validate_work_region_mm,
    )
    from template_store import (
        CONTOUR_SAMPLE_COUNT,
        align_closed_contours,
        build_shape_descriptor,
        match_known_pieces,
        resample_closed_contour,
    )


KNOWN_TARGET_SIZE_MM = (100.0, 60.0)
KNOWN_LAYOUT_SIZE_TOLERANCE_MM = 15.0


class AssemblyPlacement:
    """保存单片从当前位姿移动到目标位姿所需的屏幕与机械数据。"""

    def __init__(
        self,
        piece_id,
        source_center_mm,
        target_center_mm,
        target_polygon_mm,
        rotation_delta_deg,
    ):
        """规范化编号、中心、目标轮廓和纸面内旋转增量。

        关键参数均使用完整A4左上角为原点的毫米坐标；旋转角规范到[-180,180)。
        返回值：构造函数无返回值，数据保存为公开只读约定属性。
        """
        self.piece_id = str(piece_id)
        self.source_center_mm = tuple(float(value) for value in source_center_mm)
        self.target_center_mm = tuple(float(value) for value in target_center_mm)
        self.target_polygon_mm = [
            (float(point[0]), float(point[1])) for point in target_polygon_mm
        ]
        self.rotation_delta_deg = _normalize_angle_deg(rotation_delta_deg)


class AssemblyPlan:
    """保存一次拼装求解的成功状态、目标矩形、单片位姿和诊断分数。"""

    def __init__(
        self,
        success,
        placements=None,
        target_rect_mm=None,
        score=float("inf"),
        reason="",
        search_nodes=0,
        diagnostics=None,
    ):
        """初始化结构化规划结果，失败结果默认不携带任何机械目标。

        diagnostics保存搜索阶段的拒绝次数，只用于屏幕和测试诊断，不参与电机控制。
        """
        self.success = bool(success)
        self.placements = list(placements or [])
        self.target_rect_mm = (
            None
            if target_rect_mm is None
            else tuple(float(value) for value in target_rect_mm)
        )
        self.score = float(score)
        self.reason = str(reason)
        self.search_nodes = int(search_nodes)
        self.diagnostics = {
            str(name): int(count) for name, count in (diagnostics or {}).items()
        }

    @classmethod
    def failed(cls, reason, search_nodes=0, diagnostics=None):
        """构造不含目标位置的失败结果，避免调用方误驱动电机。"""
        return cls(
            False,
            reason=reason,
            search_nodes=search_nodes,
            diagnostics=diagnostics,
        )


def _normalize_angle_deg(angle_deg):
    """把任意角度规范到[-180,180)，便于屏幕显示和电机选择短路径。"""
    angle_deg = float(angle_deg)
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def _normalize_vertices(vertices_mm, minimum_count=3):
    """校验毫米多边形并返回N×2 float64数组。

    关键参数：minimum_count默认3；所有坐标必须有限且多边形面积非零。
    返回值：独立数组；字段缺失、退化或非二维输入抛出ValueError。
    """
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < minimum_count:
        raise ValueError("毫米多边形必须包含至少三个二维顶点")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("毫米多边形必须包含有限数字")
    if abs(float(cv2.contourArea(vertices.astype(np.float32)))) <= 1e-6:
        raise ValueError("毫米多边形面积过小或已经退化")
    return vertices.copy()


def _point_to_segment_distance(point, segment_start, segment_end):
    """计算二维点到有限线段的欧氏距离。

    主要流程：把点投影到线段方向并将比例限制在0～1；线段退化时直接返回点到端点
    距离。关键参数均为二维有限坐标。返回值为非负毫米距离。
    """
    point = np.asarray(point, dtype=np.float64).reshape(2)
    start = np.asarray(segment_start, dtype=np.float64).reshape(2)
    end = np.asarray(segment_end, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(np.concatenate((point, start, end)))):
        raise ValueError("点和线段端点必须是有限二维坐标")
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, direction) / length_squared)
    ratio = max(0.0, min(1.0, ratio))
    projection = start + ratio * direction
    return float(np.linalg.norm(point - projection))


def _clean_solver_short_edges(
    vertices_mm,
    min_edge_mm=UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM,
):
    """合并WHITE求解副本中的毫米伪短边并返回清理诊断。

    主要流程：先复制并校验有序多边形，再反复寻找最短且低于门槛的边；分别尝试
    删除短边起点或终点，使用被删点到新相邻弦线的距离衡量形变，优先采用距离较小且
    面积仍有效的候选。至少保留三个顶点，门槛为0时直接返回副本。

    关键参数：vertices_mm为N×2毫米坐标，min_edge_mm为非负有限值。
    返回值：``(独立顶点数组, 诊断字典)``；不会修改视觉层传入数组或碎片字典。
    """
    vertices = _normalize_vertices(vertices_mm)
    try:
        minimum_edge = float(min_edge_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("求解短边门槛必须是非负有限毫米值") from error
    if minimum_edge < 0.0 or not math.isfinite(minimum_edge):
        raise ValueError("求解短边门槛必须是非负有限毫米值")

    original_vertex_count = int(len(vertices))
    original_lengths = np.linalg.norm(
        np.roll(vertices, -1, axis=0) - vertices,
        axis=1,
    )
    original_min_edge = float(np.min(original_lengths))
    cleaned = vertices.copy()
    removed_count = 0

    while minimum_edge > 0.0 and len(cleaned) > 3:
        edge_lengths = np.linalg.norm(
            np.roll(cleaned, -1, axis=0) - cleaned,
            axis=1,
        )
        shortest_edge_index = int(np.argmin(edge_lengths))
        if float(edge_lengths[shortest_edge_index]) >= minimum_edge:
            break

        start_index = shortest_edge_index
        end_index = (shortest_edge_index + 1) % len(cleaned)
        previous_index = (start_index - 1) % len(cleaned)
        following_index = (end_index + 1) % len(cleaned)
        delete_candidates = (
            (
                _point_to_segment_distance(
                    cleaned[start_index],
                    cleaned[previous_index],
                    cleaned[end_index],
                ),
                start_index,
            ),
            (
                _point_to_segment_distance(
                    cleaned[end_index],
                    cleaned[start_index],
                    cleaned[following_index],
                ),
                end_index,
            ),
        )

        accepted = None
        for _distortion_mm, delete_index in sorted(delete_candidates):
            candidate = np.delete(cleaned, delete_index, axis=0)
            try:
                accepted = _normalize_vertices(candidate)
            except ValueError:
                continue
            break
        if accepted is None:
            # 两种局部合并都会使轮廓退化时停止，保留最后一个有效多边形。
            break
        cleaned = accepted
        removed_count += 1

    cleaned_lengths = np.linalg.norm(
        np.roll(cleaned, -1, axis=0) - cleaned,
        axis=1,
    )
    diagnostics = {
        "original_vertex_count": original_vertex_count,
        "cleaned_vertex_count": int(len(cleaned)),
        "removed_count": int(removed_count),
        "original_min_edge_mm": original_min_edge,
        "cleaned_min_edge_mm": float(np.min(cleaned_lengths)),
    }
    return cleaned.copy(), diagnostics


def _polygon_centroid(vertices_mm):
    """计算毫米多边形面积质心，退化时回退顶点均值。"""
    vertices = _normalize_vertices(vertices_mm)
    moments = cv2.moments(vertices.astype(np.float32).reshape(-1, 1, 2))
    if abs(float(moments["m00"])) <= 1e-9:
        center = np.mean(vertices, axis=0)
    else:
        center = np.asarray(
            (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ),
            dtype=np.float64,
        )
    return float(center[0]), float(center[1])


def classify_piece_region(vertices_mm, split_y_mm, tolerance_mm=0.5):
    """按全部毫米顶点判断碎片位于分界线上方、下方或跨线。

    顶点最高Y不超过分界线减容差时为upper，最低Y不小于分界线加容差时为lower；
    其余情况统一为crossing，防止只看质心导致机械抓取跨线碎片。
    """
    vertices = _normalize_vertices(vertices_mm)
    split_y_mm = float(split_y_mm)
    tolerance_mm = max(0.0, float(tolerance_mm))
    maximum_y = float(np.max(vertices[:, 1]))
    minimum_y = float(np.min(vertices[:, 1]))
    if maximum_y <= split_y_mm - tolerance_mm:
        return "upper"
    if minimum_y >= split_y_mm + tolerance_mm:
        return "lower"
    return "crossing"


def _descriptor_sort_key(descriptor):
    """复用模板登记的稳定几何排序规则，为布局和K编号建立一一对应。"""
    return (
        int(descriptor["vertex_count"]),
        round(float(descriptor["compactness"]), 9),
        tuple(round(float(value), 9) for value in sorted(descriptor["edge_ratios"])),
    )


def register_known_layout(
    pieces,
    expected_size_mm=KNOWN_TARGET_SIZE_MM,
    size_tolerance_mm=KNOWN_LAYOUT_SIZE_TOLERANCE_MM,
):
    """把赛前正确四片布局登记为形状模板和目标局部毫米轮廓。

    主要流程：检查四片完整毫米多边形，求整体外接轴对齐矩形，确认接近100×60mm，
    再把检测坐标按两轴归一到精确目标尺寸。描述子与目标轮廓使用同一稳定排序编号。
    返回值：四个可直接交给template_store保存的模板字典。
    """
    pieces = list(pieces)
    if len(pieces) != 4:
        raise ValueError("已知布局必须恰好包含四片")
    expected_width, expected_height = (float(value) for value in expected_size_mm)
    if expected_width <= 0.0 or expected_height <= 0.0:
        raise ValueError("已知目标宽高必须大于零")

    vertices_by_piece = [_normalize_vertices(piece["vertices_mm"]) for piece in pieces]
    all_vertices = np.vstack(vertices_by_piece)
    minimum = np.min(all_vertices, axis=0)
    maximum = np.max(all_vertices, axis=0)
    measured_width, measured_height = maximum - minimum
    tolerance = max(0.0, float(size_tolerance_mm))
    if (
        abs(float(measured_width) - expected_width) > tolerance
        or abs(float(measured_height) - expected_height) > tolerance
    ):
        raise ValueError("已知布局外框必须接近100×60mm")

    scale = np.asarray(
        (expected_width / measured_width, expected_height / measured_height),
        dtype=np.float64,
    )
    candidates = []
    for piece, vertices in zip(pieces, vertices_by_piece):
        descriptor = build_shape_descriptor(piece)
        target_vertices = (vertices - minimum) * scale
        candidates.append((descriptor, target_vertices))
    candidates.sort(key=lambda item: _descriptor_sort_key(item[0]))

    templates = []
    for index, (descriptor, target_vertices) in enumerate(candidates, start=1):
        templates.append(
            {
                "id": f"K{index}",
                **descriptor,
                "target_vertices_mm": target_vertices.astype(float).tolist(),
                "layout_size_mm": [expected_width, expected_height],
            }
        )
    return templates


def _target_rect_in_lower_region(work_region_mm, split_y_mm, target_size_mm):
    """把目标矩形居中放入黄色区域下半区，空间不足时返回None。"""
    work_x, work_y, work_width, work_height = validate_work_region_mm(work_region_mm)
    split_y = validate_split_y_mm(work_region_mm, split_y_mm)
    target_width, target_height = (float(value) for value in target_size_mm)
    lower_height = work_y + work_height - split_y
    if target_width > work_width + 1e-6 or target_height > lower_height + 1e-6:
        return None
    target_x = work_x + (work_width - target_width) / 2.0
    target_y = split_y + (lower_height - target_height) / 2.0
    return target_x, target_y, target_width, target_height


def _best_rotation_delta_deg(source_vertices_mm, target_vertices_mm):
    """用不含镜像的二维刚体配准求源多边形到目标多边形的旋转增量。

    主要流程：枚举目标顶点循环起点和正反遍历，仅接受行列式为正的旋转矩阵；
    以中心化均方根误差选最佳对应。返回值：``(角度, 误差)``。
    """
    source = _normalize_vertices(source_vertices_mm)
    target = _normalize_vertices(target_vertices_mm)
    angle_deg, alignment_error = align_closed_contours(
        source,
        target,
        sample_count=CONTOUR_SAMPLE_COUNT,
        normalize_scale=False,
    )
    return _normalize_angle_deg(angle_deg), float(alignment_error)


def solve_known_layout(
    pieces,
    templates,
    work_region_mm,
    split_y_mm,
    max_match_score=1.2,
):
    """匹配已知四片并直接生成固定100×60mm下半区目标位姿。

    主要流程：检查模板布局字段，计算下半区目标框，复用最多24种全局模板分配，
    再逐片执行无镜像刚体配准。返回AssemblyPlan；任何不安全条件返回结构化失败。
    """
    pieces = list(pieces)
    templates = list(templates)
    if len(pieces) != 4 or len(templates) != 4:
        return AssemblyPlan.failed("known_needs_four")
    if any(
        "target_vertices_mm" not in template or "layout_size_mm" not in template
        for template in templates
    ):
        return AssemblyPlan.failed("template_no_layout")

    try:
        target_sizes = {
            tuple(float(value) for value in item["layout_size_mm"])
            for item in templates
        }
        if any(
            len(size) != 2
            or not np.all(np.isfinite(np.asarray(size)))
            or min(size) <= 0.0
            for size in target_sizes
        ):
            raise ValueError("目标布局尺寸无效")
        # 在模板匹配前先校验所有目标多边形，损坏持久文件只能产生失败规划，
        # 不能把ValueError泄漏到MaixCAM2主循环。
        for template in templates:
            _normalize_vertices(template["target_vertices_mm"])
    except (KeyError, TypeError, ValueError):
        return AssemblyPlan.failed("template_layout_invalid")
    if len(target_sizes) != 1:
        return AssemblyPlan.failed("template_layout_invalid")
    target_size = next(iter(target_sizes))
    try:
        target_rect = _target_rect_in_lower_region(
            work_region_mm,
            split_y_mm,
            target_size,
        )
    except ValueError:
        return AssemblyPlan.failed("target_out_of_work_region")
    if target_rect is None:
        return AssemblyPlan.failed("target_out_of_work_region")

    match_known_pieces(pieces, templates, float(max_match_score))
    if any(piece.get("id") == "UNKNOWN" for piece in pieces):
        return AssemblyPlan.failed("known_match_failed")

    template_by_id = {template["id"]: template for template in templates}
    target_origin = np.asarray(target_rect[:2], dtype=np.float64)
    placements = []
    total_score = 0.0
    for piece in pieces:
        template = template_by_id[piece["id"]]
        local_target = _normalize_vertices(template["target_vertices_mm"])
        target_polygon = local_target + target_origin
        try:
            rotation_delta, alignment_error = _best_rotation_delta_deg(
                piece["vertices_mm"],
                target_polygon,
            )
        except ValueError:
            return AssemblyPlan.failed("known_pose_failed")
        target_center = _polygon_centroid(target_polygon)
        placements.append(
            AssemblyPlacement(
                piece["id"],
                piece["center_mm"],
                target_center,
                target_polygon,
                rotation_delta,
            )
        )
        total_score += float(piece.get("match_score", 0.0)) + alignment_error * 0.01

    placements.sort(key=lambda placement: placement.piece_id)
    return AssemblyPlan(
        True,
        placements=placements,
        target_rect_mm=target_rect,
        score=total_score,
        reason="ok",
        search_nodes=24,
    )


PATTERN_ENERGY_THRESHOLD = 0.05
UNKNOWN_LONG_SIDE_RANGE_MM = (88.0, 122.0)
UNKNOWN_SHORT_SIDE_RANGE_MM = (48.0, 92.0)
# 兼容旧测试和离线工具导入；生产严格门统一读取文件顶部的现场常量。
UNKNOWN_MIN_FILL_RATIO = UNKNOWN_STRICT_MIN_FILL_RATIO
UNKNOWN_MAX_OVERLAP_RATIO = 0.03
# 题目保证现场碎片的实体边不短于20mm。识别顶点存在毫米级误差，因此搜索时
# 接受18mm以上的公共接缝，既覆盖题目下限，也避免把拟合产生的短毛刺当成接缝。
UNKNOWN_MIN_SEAM_LENGTH_MM = 18.0
# 白片没有花纹对应关系，任意通过尺寸、填充和重叠硬验收的矩形都是可执行答案。
# 理论最高合格几何分为(1-0.92)*50+0.03*50=5.5，因此首个合格解即可停止。
UNKNOWN_WHITE_EARLY_STOP_SCORE = 5.5
# 每层只保留最有希望的固定数量状态，避免错误边组合让搜索宽度失控。
UNKNOWN_SEARCH_BEAM_WIDTH = 96
# UNKNOWN只累计生成器工作单元真正占用CPU的时间，拍照、显示和帧间等待不消耗该预算。
UNKNOWN_SOLVER_ACTIVE_TIMEOUT_SECONDS = 8.0
# 硬墙钟用于兜住相机异常缓慢、长期没有可运行时间片等现场极端情况。
UNKNOWN_SOLVER_WALL_TIMEOUT_SECONDS = 30.0
# 保留旧常量供既有调用兼容；它表达旧接口timeout_seconds的默认活动预算长度。
UNKNOWN_SOLVER_TIMEOUT_SECONDS = UNKNOWN_SOLVER_ACTIVE_TIMEOUT_SECONDS
# UNKNOWN子模式由用户按现场材料显式选择，避免白片反光被误判成需要纹理择优的牌面。
UNKNOWN_PROFILE_WHITE = "white"
UNKNOWN_PROFILE_CARD = "card"
# 参考K230 GRAPH_AUTO的相对边长门槛只用于提出连接假设；最终仍由毫米矩形硬验收。
UNKNOWN_GRAPH_MATCH_RATIO = 0.16
UNKNOWN_GRAPH_MAX_EDGE_CANDIDATES = 32
UNKNOWN_GRAPH_MAX_MATCHING_SETS = 90
# WHITE首次得到容错解后只再比较固定数量节点，避免为了择优重新跑满12000节点。
UNKNOWN_RELAXED_REFINEMENT_NODES = 64
# 中间放片按“累计碎片面积/近似联合面积”估算最终重叠率；额外1%只吸收浮点、
# 三角化和最终栅格离散误差。最终机械结果仍严格执行UNKNOWN_MAX_OVERLAP_RATIO。
UNKNOWN_FAST_INTERMEDIATE_OVERLAP_RATIO = UNKNOWN_MAX_OVERLAP_RATIO + 0.01
# 题目保证每片至少一条边属于目标矩形外框；该比例用于吸收毫米角点和栅格误差。
UNKNOWN_OUTER_EDGE_MARGIN_RATIO = 0.045
UNKNOWN_OUTER_EDGE_MIN_MARGIN_MM = 1.5


def _resample_feature(values, target_count):
    """把一维或多通道边缘样本线性重采样到统一长度。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2) or len(array) < 2:
        raise ValueError("边缘特征至少需要两个样本")
    source_positions = np.linspace(0.0, 1.0, len(array))
    target_positions = np.linspace(0.0, 1.0, int(target_count))
    if array.ndim == 1:
        return np.interp(target_positions, source_positions, array)
    channels = [
        np.interp(target_positions, source_positions, array[:, channel])
        for channel in range(array.shape[1])
    ]
    return np.column_stack(channels)


def edge_feature_match_score(first_feature, second_feature):
    """计算两条反向对接边的牌面颜色与梯度不连续分数。

    只有任一边的pattern_energy达到门槛时才启用纹理项；纯白边直接返回0。第二条边
    按拼装方向反转，切向梯度同时取反。返回值越小表示花纹越连续。
    """
    if not first_feature or not second_feature:
        return 0.0
    first_energy = float(first_feature.get("pattern_energy", 0.0))
    second_energy = float(second_feature.get("pattern_energy", 0.0))
    if max(first_energy, second_energy) < PATTERN_ENERGY_THRESHOLD:
        return 0.0
    try:
        sample_count = max(
            4,
            min(
                len(first_feature["colors"]),
                len(second_feature["colors"]),
            ),
        )
        first_colors = _resample_feature(first_feature["colors"], sample_count)
        second_colors = _resample_feature(second_feature["colors"], sample_count)[::-1]
        first_gradients = _resample_feature(first_feature["gradients"], sample_count)
        second_gradients = -_resample_feature(
            second_feature["gradients"], sample_count
        )[::-1]
    except (KeyError, TypeError, ValueError):
        return 0.0
    color_error = float(np.mean(np.abs(first_colors - second_colors))) / 255.0
    gradient_error = float(np.mean(np.abs(first_gradients - second_gradients))) / 255.0
    return color_error + 0.5 * gradient_error


def _validate_shape_hypothesis_constants():
    """校验UNKNOWN多候选现场宏并返回规范化数值。

    主要流程：检查候选上限、真实边基准、硬下限、面积保持率和边界偏差；硬下限不得
    高于题目真实边基准。返回值按构造器需要的顺序排列。配置错误抛出ValueError，
    由外层求解入口转换成结构化几何失败，不能让异常退出相机主循环。
    """
    maximum_hypotheses = int(UNKNOWN_SHAPE_MAX_HYPOTHESES)
    real_edge_minimum = float(UNKNOWN_REAL_EDGE_MIN_MM)
    hard_edge_floor = float(UNKNOWN_EDGE_HARD_FLOOR_MM)
    area_retention_minimum = float(UNKNOWN_SHAPE_AREA_RETENTION_MIN)
    maximum_deviation = float(UNKNOWN_SHAPE_MAX_DEVIATION_MM)
    numeric_values = (
        real_edge_minimum,
        hard_edge_floor,
        area_retention_minimum,
        maximum_deviation,
    )
    if (
        maximum_hypotheses <= 0
        or not all(math.isfinite(value) for value in numeric_values)
        or real_edge_minimum <= 0.0
        or hard_edge_floor <= 0.0
        or hard_edge_floor > real_edge_minimum
        or not 0.0 < area_retention_minimum <= 1.0
        or maximum_deviation <= 0.0
    ):
        raise ValueError("UNKNOWN轮廓候选宏无效")
    return (
        maximum_hypotheses,
        real_edge_minimum,
        hard_edge_floor,
        area_retention_minimum,
        maximum_deviation,
    )


def _polygon_boundary_max_distance(source_vertices, target_vertices):
    """计算一组源点到目标闭合多边形边界的最大最短距离。

    每个源点分别计算到目标全部有限线段的最短距离，再取这些最短距离中的最大值。
    该方向量用于发现候选删角后遗漏的完整轮廓边界；调用方会交换参数再算一次，形成
    对称偏差。输入必须已经通过`_normalize_vertices()`校验。
    """
    source = np.asarray(source_vertices, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_vertices, dtype=np.float64).reshape(-1, 2)
    maximum = 0.0
    for point in source:
        minimum = min(
            _point_to_segment_distance(
                point,
                target[edge_index],
                target[(edge_index + 1) % len(target)],
            )
            for edge_index in range(len(target))
        )
        maximum = max(maximum, float(minimum))
    return maximum


def _polygon_hypothesis_key(vertices):
    """生成与循环起点和顺逆方向无关的毫米候选去重键。

    坐标按0.01mm量化，足以合并相邻epsilon产生的数值重复，同时不会把现场可分辨的
    不同角点假设合并。返回值只用于单片候选去重，不参与机械坐标计算。
    """
    points = np.rint(np.asarray(vertices, dtype=np.float64) * 100.0).astype(np.int64)
    point_tuples = tuple((int(point[0]), int(point[1])) for point in points)
    variants = []
    for ordered in (point_tuples, tuple(reversed(point_tuples))):
        for offset in range(len(ordered)):
            variants.append(ordered[offset:] + ordered[:offset])
    return min(variants)


def _polygon_turn_penalty(vertices):
    """计算尖刺和近共线重复角点的软惩罚。

    真实拼图角点可以是锐角，因此这里只惩罚小于8度的极端尖刺和大于155度的近共线
    点；结果按顶点累计。该项只参与候选排序，不代替面积、偏差和短边硬门。
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    penalty = 0.0
    for index, current in enumerate(points):
        previous_vector = points[(index - 1) % len(points)] - current
        next_vector = points[(index + 1) % len(points)] - current
        denominator = float(np.linalg.norm(previous_vector) * np.linalg.norm(next_vector))
        if denominator <= 1e-9:
            return float("inf")
        cosine = float(np.dot(previous_vector, next_vector) / denominator)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle < 8.0:
            penalty += (8.0 - angle) / 8.0
        elif angle > 155.0:
            penalty += (angle - 155.0) / 25.0
    return float(penalty)


def _hypothesis_feature_group(piece, candidate_index, vertex_count, candidate_count):
    """为指定候选复制与边数一致的CARD纹理特征。

    新数据优先读取`shape_edge_features`对应组；旧单候选夹具回退`edge_features`。
    缺失纹理时返回等长None列表，禁止把另一个候选的边特征按错误索引复用。
    """
    feature_groups = piece.get("shape_edge_features") or ()
    if candidate_index < len(feature_groups):
        features = list(feature_groups[candidate_index] or ())
        if len(features) != vertex_count:
            raise ValueError("shape_edge_features必须与对应候选边数一致")
        return features
    legacy_features = list(piece.get("edge_features") or ())
    if candidate_count == 1 and legacy_features:
        if len(legacy_features) != vertex_count:
            raise ValueError("edge_features必须与多边形边数一致")
        return legacy_features
    return [None for _ in range(vertex_count)]


def _build_solver_shape_hypotheses(piece, outline_vertices, outline_center):
    """评分、去重并构造单片最多三个UNKNOWN接缝候选。

    主要流程：读取视觉层毫米候选；对新格式执行14mm硬下限、96%面积保持和3mm双向
    偏差门，再按面积误差、偏差、20mm短边惩罚和近共线惩罚排序。旧测试或旧调用若
    只有`vertices_mm`，将其作为受信任单候选兼容输入，不在缺少完整视觉候选时误删。
    所有局部顶点都减去同一个完整轮廓质心。返回值为按score排序并重新编号的字典列表。
    """
    (
        maximum_hypotheses,
        real_edge_minimum,
        hard_edge_floor,
        area_retention_minimum,
        maximum_deviation,
    ) = _validate_shape_hypothesis_constants()
    outline = _normalize_vertices(outline_vertices)
    outline_area = abs(float(cv2.contourArea(outline.astype(np.float32))))
    if outline_area <= 1e-6:
        raise ValueError("完整轮廓面积无效")

    raw_candidates = piece.get("shape_hypotheses_mm")
    legacy_input = not raw_candidates
    if legacy_input:
        raw_candidates = [piece["vertices_mm"]]
    raw_candidates = list(raw_candidates)
    deduplicated = {}
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        candidate = _normalize_vertices(raw_candidate)
        if not 3 <= len(candidate) <= 5:
            continue
        edge_lengths_array = np.linalg.norm(
            np.roll(candidate, -1, axis=0) - candidate,
            axis=1,
        )
        minimum_edge = float(np.min(edge_lengths_array))
        if not legacy_input and minimum_edge < hard_edge_floor:
            continue

        candidate_area = abs(float(cv2.contourArea(candidate.astype(np.float32))))
        area_retention = min(outline_area, candidate_area) / max(outline_area, candidate_area)
        forward_deviation = _polygon_boundary_max_distance(outline, candidate)
        reverse_deviation = _polygon_boundary_max_distance(candidate, outline)
        boundary_deviation = max(forward_deviation, reverse_deviation)
        if not legacy_input and (
            area_retention < area_retention_minimum
            or boundary_deviation > maximum_deviation
        ):
            continue

        short_edge_penalty = float(
            np.sum(
                np.maximum(0.0, real_edge_minimum - edge_lengths_array)
                / real_edge_minimum
            )
        )
        turn_penalty = _polygon_turn_penalty(candidate)
        score = (
            (1.0 - area_retention) * 4.0
            + (boundary_deviation / maximum_deviation) * 0.5
            + short_edge_penalty * 0.5
            + turn_penalty * 0.75
        )
        features = _hypothesis_feature_group(
            piece,
            candidate_index,
            len(candidate),
            len(raw_candidates),
        )
        hypothesis = {
            "input_index": int(candidate_index),
            "source_vertices": candidate.copy(),
            "local_vertices": candidate - outline_center,
            "edge_lengths": tuple(float(value) for value in edge_lengths_array),
            "edge_features": features,
            "score": float(score),
            "area_retention": float(area_retention),
            "max_deviation_mm": float(boundary_deviation),
            "minimum_edge_mm": float(minimum_edge),
            "short_edge_count": int(np.count_nonzero(edge_lengths_array < real_edge_minimum)),
        }
        key = _polygon_hypothesis_key(candidate)
        previous = deduplicated.get(key)
        if previous is None or hypothesis["score"] < previous["score"]:
            deduplicated[key] = hypothesis

    hypotheses = sorted(
        deduplicated.values(),
        key=lambda item: (
            item["score"],
            item["max_deviation_mm"],
            -item["area_retention"],
            item["input_index"],
        ),
    )[:maximum_hypotheses]
    if not hypotheses:
        raise ValueError("shape_hypothesis_empty")
    for rank, hypothesis in enumerate(hypotheses):
        hypothesis["rank"] = int(rank)
    return hypotheses


def _solver_piece(piece, index):
    """把识别碎片规范为统一的UNKNOWN完整轮廓和候选结构。

    主要流程：高保真轮廓和全部候选使用同一个完整轮廓面积质心作为局部原点；候选经
    公共评分器排序。返回结构中的`outline_local`与`hypotheses`是新搜索器数据源；
    `source_vertices/local_vertices/edge_lengths/edge_features`暂时映射首选候选，供后续
    Task 4/5逐步迁移旧GRAPH、FOURFAST和FALLBACK，不修改视觉层传入字典。
    """
    fallback_vertices = _normalize_vertices(piece["vertices_mm"])
    outline = _normalize_vertices(piece.get("outline_mm", fallback_vertices))
    outline_center = np.asarray(_polygon_centroid(outline), dtype=np.float64)
    hypotheses = _build_solver_shape_hypotheses(piece, outline, outline_center)
    primary = hypotheses[0]
    primary_lengths = tuple(float(value) for value in primary["edge_lengths"])
    cleanup = {
        "original_vertex_count": int(len(fallback_vertices)),
        "cleaned_vertex_count": int(len(primary["source_vertices"])),
        "removed_count": 0,
        "original_min_edge_mm": float(
            np.min(
                np.linalg.norm(
                    np.roll(fallback_vertices, -1, axis=0) - fallback_vertices,
                    axis=1,
                )
            )
        ),
        "cleaned_min_edge_mm": float(min(primary_lengths)),
    }
    return {
        "index": int(index),
        "id": str(piece.get("id", f"U{index + 1}")),
        "outline_source": outline.copy(),
        "outline_local": outline - outline_center,
        "source_center": tuple(float(value) for value in outline_center),
        "hypotheses": hypotheses,
        # 以下首选候选别名将在Task 4/5完成后仅保留给兼容辅助函数。
        "source_vertices": primary["source_vertices"].copy(),
        "local_vertices": primary["local_vertices"].copy(),
        "cleanup": cleanup,
        "edge_features": list(primary["edge_features"]),
        "edge_lengths": primary_lengths,
    }


def _edge_length(vertices, edge_index):
    """返回循环多边形指定边的毫米长度。"""
    start = vertices[edge_index]
    end = vertices[(edge_index + 1) % len(vertices)]
    return float(np.linalg.norm(end - start))


def _edge_alignment_pose(
    source_polygon,
    source_edge,
    target_polygon,
    target_edge,
    anchor="midpoint",
):
    """计算源边反向贴合目标边的二维刚体位姿。

    主要流程：根据两条有向边的夹角生成行列式为正一的旋转矩阵，再按`anchor`
    选择中点、源起点或源终点作为平移锚点。整个过程不引入缩放和镜像；等长边在
    三种锚点下得到相同结果，长短边分段接缝则可用两端锚点生成左右两个候选。

    关键参数：`source_edge`和`target_edge`是闭合多边形边号；`anchor`可为
    `midpoint`、`source_start`或`source_end`。返回值为`(2x2旋转矩阵, 2维平移)`。
    """
    source = np.asarray(source_polygon, dtype=np.float64)
    target = np.asarray(target_polygon, dtype=np.float64)
    source_edge = int(source_edge)
    target_edge = int(target_edge)
    source_start = source[source_edge]
    source_end = source[(source_edge + 1) % len(source)]
    target_start = target[target_edge]
    target_end = target[(target_edge + 1) % len(target)]
    source_vector = source_end - source_start
    target_reverse_vector = target_start - target_end
    if (
        float(np.linalg.norm(source_vector)) <= 1e-9
        or float(np.linalg.norm(target_reverse_vector)) <= 1e-9
    ):
        raise ValueError("接缝边长度必须大于零")

    source_angle = math.atan2(source_vector[1], source_vector[0])
    target_angle = math.atan2(target_reverse_vector[1], target_reverse_vector[0])
    angle = target_angle - source_angle
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    rotated_start = source_start @ rotation.T
    rotated_end = source_end @ rotation.T
    if anchor == "midpoint":
        source_anchor = (rotated_start + rotated_end) * 0.5
        target_anchor = (target_start + target_end) * 0.5
    elif anchor == "source_start":
        source_anchor = rotated_start
        target_anchor = target_end
    elif anchor == "source_end":
        source_anchor = rotated_end
        target_anchor = target_start
    else:
        raise ValueError("未知接缝锚点模式")
    translation = target_anchor - source_anchor
    return rotation, translation


def _transform_solver_outline(solver_piece, hypothesis_index, pose):
    """用同一刚体位姿转换一片碎片的接缝候选和完整轮廓。

    主要流程：按候选编号读取`local_vertices`，并与`outline_local`分别通过同一个
    旋转矩阵和平移向量。返回的两套多边形共享坐标系：接缝多边形只用于提出后续
    连接，完整轮廓专门用于重叠和最终矩形硬验收。

    关键参数：`solver_piece`来自`_solver_piece`；`hypothesis_index`是候选列表下标；
    `pose`为`(旋转矩阵, 平移向量)`。返回含候选编号、位姿和两套多边形的字典。
    """
    hypotheses = solver_piece["hypotheses"]
    selected_index = int(hypothesis_index)
    if not 0 <= selected_index < len(hypotheses):
        raise ValueError("候选编号超出范围")
    hypothesis = hypotheses[selected_index]
    return {
        "hypothesis": selected_index,
        "pose": pose,
        "seam_polygon": _transform_polygon_with_pose(
            hypothesis["local_vertices"],
            pose,
        ),
        "outline_polygon": _transform_polygon_with_pose(
            solver_piece["outline_local"],
            pose,
        ),
    }


def _solver_state_pose_key(pose_by_index, hypothesis_by_index, precision=100.0):
    """生成包含候选身份和量化刚体位姿的稳定搜索状态键。

    每片记录候选编号、旋转矩阵和平移向量，避免外框相近的不同接缝候选在完整轮廓
    硬验收前被错误合并。`precision`是每个浮点单位的量化倍数，默认保留0.01精度；
    返回按碎片索引排序的不可变元组，可直接用于集合和字典去重。
    """
    scale = float(precision)
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("位姿量化精度必须是有限正数")
    result = []
    for piece_index in sorted(pose_by_index):
        if piece_index not in hypothesis_by_index:
            raise ValueError("位姿状态缺少候选编号")
        rotation, translation = pose_by_index[piece_index]
        pose_values = np.concatenate(
            (
                np.asarray(rotation, dtype=np.float64).reshape(-1),
                np.asarray(translation, dtype=np.float64).reshape(-1),
            )
        )
        result.append(
            (
                int(piece_index),
                int(hypothesis_by_index[piece_index]),
                tuple(int(value) for value in np.rint(pose_values * scale)),
            )
        )
    return tuple(result)


def _align_source_edge_to_target(source_vertices, source_edge, target_vertices, target_edge):
    """生成使源边与目标边反向重合的无缩放、无镜像刚体变换多边形。"""
    source_start = source_vertices[source_edge]
    source_end = source_vertices[(source_edge + 1) % len(source_vertices)]
    target_start = target_vertices[target_edge]
    target_end = target_vertices[(target_edge + 1) % len(target_vertices)]
    source_vector = source_end - source_start
    target_vector = target_start - target_end
    source_angle = math.atan2(source_vector[1], source_vector[0])
    target_angle = math.atan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    translation = target_end - source_start @ rotation.T
    return source_vertices @ rotation.T + translation


def _align_source_edge_candidates(
    source_vertices,
    source_edge,
    target_vertices,
    target_edge,
):
    """生成源边与目标边反向共线时从两个端点开始对接的刚体候选。

    主要流程：先计算不缩放、不镜像的旋转，再分别让“源起点对目标终点”和
    “源终点对目标起点”。等长边的两个结果相同，会自动去重；边长不等时两个
    结果分别覆盖长边左端和右端，允许一条长边由多条短边共同组成接缝。
    返回值：一至两个保持源多边形形状不变的N×2候选数组。
    """
    source_start = source_vertices[source_edge]
    source_end = source_vertices[(source_edge + 1) % len(source_vertices)]
    target_start = target_vertices[target_edge]
    target_end = target_vertices[(target_edge + 1) % len(target_vertices)]
    source_vector = source_end - source_start
    target_reverse_vector = target_start - target_end
    source_angle = math.atan2(source_vector[1], source_vector[0])
    target_angle = math.atan2(target_reverse_vector[1], target_reverse_vector[0])
    angle = target_angle - source_angle
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    rotated_polygon = source_vertices @ rotation.T
    rotated_start = rotated_polygon[source_edge]
    rotated_end = rotated_polygon[(source_edge + 1) % len(source_vertices)]
    candidates = [rotated_polygon + (target_end - rotated_start)]
    anchored_at_start = rotated_polygon + (target_start - rotated_end)
    if not np.allclose(candidates[0], anchored_at_start, atol=1e-6):
        candidates.append(anchored_at_start)
    return candidates


def _segmented_seam_is_possible(
    solver_pieces,
    source_index,
    source_length,
    target_index,
    target_length,
    base_tolerance_mm,
):
    """判断一条长边能否由当前短边和其他碎片的边共同完整覆盖。

    主要流程：把较短边作为已占用接缝，再从其余每片最多选择一条边累加；只有
    某个组合与长边长度在毫米误差内一致时才允许分段候选进入回溯。该边长守恒
    剪枝可保留T形接缝，同时阻止任意长短边组合造成搜索节点爆炸。
    返回值：存在可完成的分段组合时为True，否则为False。
    """
    long_length = max(float(source_length), float(target_length))
    short_length = min(float(source_length), float(target_length))
    if short_length < UNKNOWN_MIN_SEAM_LENGTH_MM:
        return False

    # 每个列表项保存“当前累计长度、已经使用的接缝段数”。初始短边算一段。
    partial_sums = [(short_length, 1)]
    for piece_index, solver_piece in enumerate(solver_pieces):
        if piece_index in (source_index, target_index):
            continue
        # _solver_piece已经缓存边长；兼容旧测试构造的内部字典时才现场补算。
        # 其余碎片可以尚未确定形状候选，因此这里汇总全部候选的边长，仅回答
        # “是否存在补齐长边的可能”。真正搜索状态会固定候选编号，最终不会混用。
        hypotheses = solver_piece.get("hypotheses")
        if hypotheses:
            edge_lengths = tuple(
                float(edge_length)
                for hypothesis in hypotheses
                for edge_length in hypothesis["edge_lengths"]
            )
        else:
            edge_lengths = solver_piece.get("edge_lengths")
            if edge_lengths is None:
                edge_lengths = tuple(
                    _edge_length(solver_piece["local_vertices"], edge_index)
                    for edge_index in range(len(solver_piece["local_vertices"]))
                )
        next_sums = list(partial_sums)
        for current_sum, segment_count in partial_sums:
            for edge_length in edge_lengths:
                candidate_sum = current_sum + edge_length
                candidate_count = segment_count + 1
                tolerance = max(
                    float(base_tolerance_mm) * candidate_count,
                    0.06 * long_length,
                )
                if abs(candidate_sum - long_length) <= tolerance:
                    return True
                if candidate_sum < long_length + tolerance:
                    next_sums.append((candidate_sum, candidate_count))
        partial_sums = next_sums
    return False


def _build_edge_compatibility_graph(solver_pieces, base_tolerance_mm):
    """一次构建全部形状候选的有向整边与T形分段边兼容关系。

    主要流程：枚举不同碎片的候选对和有向边对；等长边直接登记为``full``，长度
    不同的边只有通过长度守恒检查才登记为``segmented``。每条关系记录源/目标候选
    编号、最高等级、候选总分、纹理分数和边长误差，后续搜索可按等级逐步开放。

    关键参数：solver_pieces来自_solver_piece；base_tolerance_mm是远距离毫米边长容差。
    返回值：``{(源片索引, 目标片索引): (关系字典, ...)}``，无兼容边的片对也保留
    空元组，便于搜索使用常数时间查询。
    """
    tolerance_mm = float(base_tolerance_mm)
    if tolerance_mm < 0.0 or not math.isfinite(tolerance_mm):
        raise ValueError("边长容差必须是有限非负数")

    graph = {}
    for source_index, source_piece in enumerate(solver_pieces):
        for target_index, target_piece in enumerate(solver_pieces):
            if source_index == target_index:
                continue
            relations = []
            for source_hypothesis_index, source_hypothesis in enumerate(
                source_piece["hypotheses"]
            ):
                source_lengths = tuple(
                    float(value) for value in source_hypothesis["edge_lengths"]
                )
                source_rank = int(
                    source_hypothesis.get("rank", source_hypothesis_index)
                )
                for target_hypothesis_index, target_hypothesis in enumerate(
                    target_piece["hypotheses"]
                ):
                    target_lengths = tuple(
                        float(value) for value in target_hypothesis["edge_lengths"]
                    )
                    target_rank = int(
                        target_hypothesis.get("rank", target_hypothesis_index)
                    )
                    maximum_rank = max(source_rank, target_rank)
                    hypothesis_score = float(
                        source_hypothesis.get("score", 0.0)
                    ) + float(target_hypothesis.get("score", 0.0))
                    for source_edge, source_length in enumerate(source_lengths):
                        for target_edge, target_length in enumerate(target_lengths):
                            pair_tolerance = max(
                                tolerance_mm,
                                0.04 * max(source_length, target_length),
                            )
                            length_error = abs(source_length - target_length)
                            lengths_are_equal = length_error <= pair_tolerance
                            if not lengths_are_equal and not _segmented_seam_is_possible(
                                solver_pieces,
                                source_index,
                                source_length,
                                target_index,
                                target_length,
                                tolerance_mm,
                            ):
                                continue

                            relation_kind = "full" if lengths_are_equal else "segmented"
                            # 只有整边反向重合时特征区间一一对应；分段边没有实际子段
                            # 区间，继续只使用几何关系，防止虚假的整边纹理分数。
                            texture_score = 0.0
                            if lengths_are_equal:
                                texture_score = edge_feature_match_score(
                                    source_hypothesis["edge_features"][source_edge],
                                    target_hypothesis["edge_features"][target_edge],
                                )
                            relations.append(
                                {
                                    "source_hypothesis": int(source_hypothesis_index),
                                    "target_hypothesis": int(target_hypothesis_index),
                                    "source_edge": int(source_edge),
                                    "target_edge": int(target_edge),
                                    "maximum_rank": int(maximum_rank),
                                    "hypothesis_score": float(hypothesis_score),
                                    "kind": relation_kind,
                                    "texture_score": float(texture_score),
                                    "matched_length_mm": float(
                                        min(source_length, target_length)
                                    ),
                                    "length_error_mm": float(length_error),
                                }
                            )

            # 候选等级和质量优先，随后沿用整边、纹理、长接缝和误差排序。
            relations.sort(
                key=lambda relation: (
                    relation["maximum_rank"],
                    relation["hypothesis_score"],
                    0 if relation["kind"] == "full" else 1,
                    relation["texture_score"],
                    -relation["matched_length_mm"],
                    relation["length_error_mm"],
                    relation["source_hypothesis"],
                    relation["target_hypothesis"],
                    relation["source_edge"],
                    relation["target_edge"],
                )
            )
            graph[(source_index, target_index)] = tuple(relations)
    return graph


def _rasterize_polygons(polygons, pixels_per_mm=2.0, erode=False):
    """把毫米多边形放入共同局部画布并返回逐片二值掩膜。

    该函数支持凹多边形，专门用于小规模重叠与填充校验。erode=True时收缩一个像素，
    消除共享边因离散填充产生的单像素假重叠。
    """
    polygons = [np.asarray(polygon, dtype=np.float64) for polygon in polygons]
    all_points = np.vstack(polygons)
    minimum = np.floor(np.min(all_points, axis=0) * pixels_per_mm) - 3.0
    maximum = np.ceil(np.max(all_points, axis=0) * pixels_per_mm) + 3.0
    canvas_width = max(8, int(maximum[0] - minimum[0] + 1))
    canvas_height = max(8, int(maximum[1] - minimum[1] + 1))
    masks = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for polygon in polygons:
        pixel_polygon = np.rint(polygon * pixels_per_mm - minimum).astype(np.int32)
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_polygon], 255)
        if erode:
            mask = cv2.erode(mask, kernel, iterations=1)
        masks.append(mask)
    return masks


def _candidate_overlaps(
    candidate_polygon,
    placed_polygons,
    pixels_per_mm=2.0,
    max_total_overlap_ratio=None,
    final_total_piece_area=None,
):
    """用共同栅格判断新候选是否与已放置碎片发生不可接受的内部重叠。

    默认保持CARD和旧搜索的“重叠像素/候选单片像素超过1%”语义；当
    max_total_overlap_ratio给出时，改用所有碎片累计像素相对联合像素的比例，供WHITE
    快速几何异常回退使用。final_total_piece_area可提供最终全部碎片毫米面积，使两片、
    三片阶段与最终四片硬门使用同一分母口径。返回True表示应剪除当前候选。
    """
    if not placed_polygons:
        return False
    masks = _rasterize_polygons(
        list(placed_polygons) + [candidate_polygon],
        pixels_per_mm=pixels_per_mm,
        erode=True,
    )
    existing_union = np.zeros_like(masks[0])
    for mask in masks[:-1]:
        existing_union = cv2.bitwise_or(existing_union, mask)
    if max_total_overlap_ratio is None:
        overlap_pixels = int(np.count_nonzero(cv2.bitwise_and(existing_union, masks[-1])))
        candidate_pixels = max(1, int(np.count_nonzero(masks[-1])))
        return overlap_pixels / candidate_pixels > 0.01

    ratio_limit = float(max_total_overlap_ratio)
    if ratio_limit < 0.0 or not math.isfinite(ratio_limit):
        raise ValueError("累计重叠率上限必须是有限非负数")
    total_pixels = sum(int(np.count_nonzero(mask)) for mask in masks)
    union = existing_union.copy()
    union = cv2.bitwise_or(union, masks[-1])
    union_pixels = max(1, int(np.count_nonzero(union)))
    duplicated_pixels = max(0, int(total_pixels - union_pixels))
    if final_total_piece_area is None:
        total_overlap_ratio = float(duplicated_pixels) / union_pixels
    else:
        # MASK回退也必须使用全部碎片面积。把当前重复像素换回毫米面积后，从最终
        # 碎片总面积中扣除，得到与几何快路径相同的近似联合面积分母。
        selected_final_area = float(final_total_piece_area)
        if selected_final_area <= 0.0 or not math.isfinite(selected_final_area):
            raise ValueError("最终碎片总面积必须是有限正数")
        overlap_area_mm2 = float(duplicated_pixels) / (float(pixels_per_mm) ** 2)
        estimated_final_union_area = max(
            1e-9,
            selected_final_area - overlap_area_mm2,
        )
        total_overlap_ratio = overlap_area_mm2 / estimated_final_union_area
    return total_overlap_ratio > ratio_limit


def _point_in_ccw_triangle(point, first, second, third, tolerance=1e-9):
    """判断二维点是否位于逆时针三角形内部或边界上。

    该函数只供耳切三角化使用；通过三条有向边叉积判断，容差吸收毫米浮点旋转误差。
    关键参数均为二维坐标，返回布尔值。
    """
    point = np.asarray(point, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    third = np.asarray(third, dtype=np.float64)

    def cross(start, end, tested):
        """返回有向边start->end到tested的二维叉积。"""
        edge = end - start
        offset = tested - start
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    minimum = -abs(float(tolerance))
    return bool(
        cross(first, second, point) >= minimum
        and cross(second, third, point) >= minimum
        and cross(third, first, point) >= minimum
    )


def _triangulate_simple_polygon(vertices_mm):
    """用耳切法把3～少量顶点的简单多边形分解为互不重叠三角形。

    主要流程：按有向面积统一为逆时针索引，循环寻找凸顶点；候选耳三角形内部没有
    其他顶点时输出并删除该顶点，最后保留一个三角形。现场轮廓通常只有3～5点，
    O(N^3)上界远小于反复分配整幅毫米栅格的成本。

    返回值：成功时为float32三角形轮廓元组；自交、严重共线或数值异常时返回None，
    调用方必须回退原MASK判断，不能把三角化失败当成无重叠。
    """
    try:
        vertices = _normalize_vertices(vertices_mm)
    except ValueError:
        return None
    if len(vertices) == 3:
        return (vertices.astype(np.float32).reshape(-1, 1, 2),)

    contour = vertices.astype(np.float32).reshape(-1, 1, 2)
    oriented_area = float(cv2.contourArea(contour, oriented=True))
    if not math.isfinite(oriented_area) or abs(oriented_area) <= 1e-9:
        return None
    indices = list(range(len(vertices)))
    if oriented_area < 0.0:
        indices.reverse()
    triangles = []
    maximum_iterations = len(vertices) * len(vertices)
    iterations = 0
    while len(indices) > 3 and iterations < maximum_iterations:
        iterations += 1
        ear_found = False
        for position, current_index in enumerate(indices):
            previous_index = indices[(position - 1) % len(indices)]
            next_index = indices[(position + 1) % len(indices)]
            previous = vertices[previous_index]
            current = vertices[current_index]
            following = vertices[next_index]
            first_edge = current - previous
            second_edge = following - current
            cross_value = float(
                first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
            )
            if cross_value <= 1e-9:
                continue
            other_inside = any(
                _point_in_ccw_triangle(
                    vertices[other_index],
                    previous,
                    current,
                    following,
                )
                for other_index in indices
                if other_index not in (previous_index, current_index, next_index)
            )
            if other_inside:
                continue
            triangle = np.asarray(
                (previous, current, following),
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            if abs(float(cv2.contourArea(triangle))) <= 1e-9:
                continue
            triangles.append(triangle)
            del indices[position]
            ear_found = True
            break
        if not ear_found:
            return None

    if len(indices) != 3:
        return None
    final_triangle = vertices[indices].astype(np.float32).reshape(-1, 1, 2)
    if abs(float(cv2.contourArea(final_triangle))) <= 1e-9:
        return None
    triangles.append(final_triangle)

    # 三角形面积和必须与原多边形一致；不一致通常表示输入自交或耳切数值失真。
    triangle_area = sum(abs(float(cv2.contourArea(triangle))) for triangle in triangles)
    polygon_area = abs(oriented_area)
    if abs(triangle_area - polygon_area) > max(0.05, polygon_area * 1e-4):
        return None
    return tuple(triangles)


def _polygon_is_simple(vertices_mm, tolerance=1e-9):
    """判断闭合多边形的非相邻边是否相交。

    主要流程：枚举少量轮廓边，跳过共享端点的相邻边，通过方向叉积和投影范围检测
    真正交叉或异常重叠。现场轮廓通常只有3～5点，O(N^2)检查远小于一次MASK分配。
    参数tolerance吸收旋转后的浮点误差；返回False时调用方必须走保守栅格路径。
    """
    vertices = np.asarray(vertices_mm, dtype=np.float64).reshape(-1, 2)
    if len(vertices) < 3 or not np.all(np.isfinite(vertices)):
        return False
    epsilon = abs(float(tolerance))

    def orientation(first, second, third):
        """返回三点有向面积的二维叉积。"""
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        third = np.asarray(third, dtype=np.float64)
        edge = second - first
        offset = third - first
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    def point_on_segment(point, start, end):
        """判断共线点是否位于闭线段投影范围内。"""
        return bool(
            np.all(point >= np.minimum(start, end) - epsilon)
            and np.all(point <= np.maximum(start, end) + epsilon)
        )

    def segments_intersect(first_start, first_end, second_start, second_end):
        """判断两条闭线段是否交叉或共线重叠。"""
        first_cross_start = orientation(first_start, first_end, second_start)
        first_cross_end = orientation(first_start, first_end, second_end)
        second_cross_start = orientation(second_start, second_end, first_start)
        second_cross_end = orientation(second_start, second_end, first_end)
        if (
            first_cross_start * first_cross_end < -(epsilon * epsilon)
            and second_cross_start * second_cross_end < -(epsilon * epsilon)
        ):
            return True
        checks = (
            (first_cross_start, second_start, first_start, first_end),
            (first_cross_end, second_end, first_start, first_end),
            (second_cross_start, first_start, second_start, second_end),
            (second_cross_end, first_end, second_start, second_end),
        )
        return any(
            abs(cross_value) <= epsilon and point_on_segment(point, start, end)
            for cross_value, point, start, end in checks
        )

    edge_count = len(vertices)
    for first_index in range(edge_count):
        first_end_index = (first_index + 1) % edge_count
        for second_index in range(first_index + 1, edge_count):
            second_end_index = (second_index + 1) % edge_count
            # 相邻边按定义共享一个合法顶点；首边和末边也属于相邻边。
            if (
                first_index == second_index
                or first_end_index == second_index
                or second_end_index == first_index
            ):
                continue
            if segments_intersect(
                vertices[first_index],
                vertices[first_end_index],
                vertices[second_index],
                vertices[second_end_index],
            ):
                return False
    return True


def _candidate_overlaps_fast(
    candidate_polygon,
    placed_polygons,
    pixels_per_mm=2.0,
    diagnostics=None,
    final_total_piece_area=None,
):
    """优先用凸多边形几何交集判断候选重叠，凹多边形回退原栅格。

    主要流程：先用轴对齐外框排除完全分离或只在边界接触的片对；全部相关片对均为
    凸多边形时，累计`cv2.intersectConvexConvex`返回的真实交叠面积，并沿用旧搜索
    最终3%重叠门。简单凹多边形先耳切为三角形精确累计；三角化失败、面积无效或
    OpenCV求交异常时，对整个已放置集合调用原`_candidate_overlaps`保守兜底。

    关键参数：diagnostics可选字典用于累计凸检查和栅格回退次数；final_total_piece_area
    为最终全部碎片毫米面积，供早期两片/三片阶段使用最终布局口径估算重叠率。
    返回值：估算累计重叠率超过中间门时为True，分离或共享边接触为False。
    """
    diagnostic_values = diagnostics if isinstance(diagnostics, dict) else None

    def raster_fallback():
        """记录一次异常几何回退，并按累计重叠口径执行保守MASK判断。"""
        if diagnostic_values is not None:
            diagnostic_values["fast_raster_fallbacks"] = (
                int(diagnostic_values.get("fast_raster_fallbacks", 0)) + 1
            )
        return _candidate_overlaps(
            candidate_polygon,
            placed_polygons,
            pixels_per_mm=pixels_per_mm,
            max_total_overlap_ratio=UNKNOWN_FAST_INTERMEDIATE_OVERLAP_RATIO,
            final_total_piece_area=final_total_piece_area,
        )

    try:
        candidate = _normalize_vertices(candidate_polygon)
        placed = [_normalize_vertices(polygon) for polygon in placed_polygons]
    except (TypeError, ValueError):
        return raster_fallback()
    if not placed:
        return False

    if not _polygon_is_simple(candidate) or any(
        not _polygon_is_simple(polygon) for polygon in placed
    ):
        return raster_fallback()

    candidate_minimum = np.min(candidate, axis=0)
    candidate_maximum = np.max(candidate, axis=0)
    candidate_contour = candidate.astype(np.float32).reshape(-1, 1, 2)
    candidate_is_convex = bool(cv2.isContourConvex(candidate_contour))
    candidate_area = abs(float(cv2.contourArea(candidate_contour)))
    if candidate_area <= 1e-9 or not math.isfinite(candidate_area):
        return raster_fallback()

    overlapping_aabbs = []
    for polygon in placed:
        polygon_minimum = np.min(polygon, axis=0)
        polygon_maximum = np.max(polygon, axis=0)
        intersection_span = (
            np.minimum(candidate_maximum, polygon_maximum)
            - np.maximum(candidate_minimum, polygon_minimum)
        )
        # 任一轴没有正长度时，两片最多共享一个端点或一条边，不存在内部面积重叠。
        if float(intersection_span[0]) <= 1e-9 or float(intersection_span[1]) <= 1e-9:
            continue
        overlapping_aabbs.append(polygon)

    if not overlapping_aabbs:
        return False
    intersection_area = 0.0
    try:
        candidate_parts = (
            (candidate_contour,)
            if candidate_is_convex
            else _triangulate_simple_polygon(candidate)
        )
        if not candidate_parts:
            raise ValueError("候选多边形无法安全三角化")
        for polygon in overlapping_aabbs:
            polygon_contour = polygon.astype(np.float32).reshape(-1, 1, 2)
            polygon_is_convex = bool(cv2.isContourConvex(polygon_contour))
            polygon_parts = (
                (polygon_contour,)
                if polygon_is_convex
                else _triangulate_simple_polygon(polygon)
            )
            if not polygon_parts:
                raise ValueError("已放置多边形无法安全三角化")
            uses_triangles = len(candidate_parts) > 1 or len(polygon_parts) > 1
            for candidate_part in candidate_parts:
                candidate_part_points = candidate_part.reshape(-1, 2)
                candidate_part_minimum = np.min(candidate_part_points, axis=0)
                candidate_part_maximum = np.max(candidate_part_points, axis=0)
                for polygon_part in polygon_parts:
                    polygon_part_points = polygon_part.reshape(-1, 2)
                    part_span = (
                        np.minimum(
                            candidate_part_maximum,
                            np.max(polygon_part_points, axis=0),
                        )
                        - np.maximum(
                            candidate_part_minimum,
                            np.min(polygon_part_points, axis=0),
                        )
                    )
                    if float(part_span[0]) <= 1e-9 or float(part_span[1]) <= 1e-9:
                        continue
                    if diagnostic_values is not None:
                        diagnostic_key = (
                            "fast_triangle_checks" if uses_triangles else "fast_convex_checks"
                        )
                        diagnostic_values[diagnostic_key] = (
                            int(diagnostic_values.get(diagnostic_key, 0)) + 1
                        )
                    area, _intersection = cv2.intersectConvexConvex(
                        candidate_part,
                        polygon_part,
                        handleNested=True,
                    )
                    area = float(area)
                    if not math.isfinite(area) or area < 0.0:
                        raise ValueError("多边形交集面积无效")
                    intersection_area += area
            # 用所有碎片面积减去本次交叠估算加入候选后的联合面积。已放片之间若存在
            # 少量历史重叠会让该估算更宽松，但完整状态仍执行精确3%栅格硬门；快速层
            # 的职责是避免误删合法路径，而不是代替最终机械验收。
            current_piece_area = candidate_area + sum(
                abs(
                    float(
                        cv2.contourArea(
                            polygon.astype(np.float32).reshape(-1, 1, 2)
                        )
                    )
                )
                for polygon in placed
            )
            if final_total_piece_area is None:
                total_piece_area = current_piece_area
            else:
                selected_final_area = float(final_total_piece_area)
                if selected_final_area <= 0.0 or not math.isfinite(selected_final_area):
                    raise ValueError("最终碎片总面积必须是有限正数")
                # 调用方传入值理论上不会小于当前面积；max用于吸收轮廓浮点量化误差，
                # 防止异常偏小参数反而让快速预筛比原逻辑更严格。
                total_piece_area = max(current_piece_area, selected_final_area)
            estimated_union_area = max(1e-9, total_piece_area - intersection_area)
            if (
                intersection_area / estimated_union_area
                > UNKNOWN_FAST_INTERMEDIATE_OVERLAP_RATIO
            ):
                return True
    except (cv2.error, TypeError, ValueError):
        return raster_fallback()
    return False


def _partial_layout_priority(
    placed_polygons,
    tolerance_mm=3.0,
    target_size_hint_mm=None,
):
    """计算部分布局的紧凑搜索优先级，并拒绝不可能装入最大目标框的状态。

    主要流程：先用点集直径和凸包面积做方向无关的安全上界剪枝；再计算最小外接
    矩形面积相对碎片总面积的空隙率。KNOWN可额外提供100×60mm尺寸提示，优先
    搜索外框更接近已知目标的状态；提示只改变顺序，不参与合法性判定。
    返回None表示不可能装入122×92mm目标，否则返回可排序浮点元组。
    """
    polygons = [np.asarray(polygon, dtype=np.float64) for polygon in placed_polygons]
    all_points = np.vstack(polygons)
    maximum_diagonal = math.hypot(
        UNKNOWN_LONG_SIDE_RANGE_MM[1],
        UNKNOWN_SHORT_SIDE_RANGE_MM[1],
    ) + max(0.0, float(tolerance_mm))
    point_differences = all_points[:, None, :] - all_points[None, :, :]
    maximum_distance = float(
        np.sqrt(np.max(np.sum(point_differences * point_differences, axis=2)))
    )
    if maximum_distance > maximum_diagonal:
        return None

    hull = cv2.convexHull(all_points.astype(np.float32).reshape(-1, 1, 2))
    maximum_area = (
        UNKNOWN_LONG_SIDE_RANGE_MM[1] * UNKNOWN_SHORT_SIDE_RANGE_MM[1]
    )
    if abs(float(cv2.contourArea(hull))) > maximum_area * 1.05:
        return None

    rectangle = cv2.minAreaRect(all_points.astype(np.float32).reshape(-1, 1, 2))
    rectangle_sides = sorted(
        (float(rectangle[1][0]), float(rectangle[1][1])),
        reverse=True,
    )
    rectangle_area = max(1.0, rectangle_sides[0] * rectangle_sides[1])
    piece_area = sum(
        abs(float(cv2.contourArea(polygon.astype(np.float32))))
        for polygon in polygons
    )
    gap_ratio = max(0.0, rectangle_area - piece_area) / max(1.0, piece_area)
    hint_error = 0.0
    if target_size_hint_mm is not None:
        hinted_sides = sorted(
            (float(target_size_hint_mm[0]), float(target_size_hint_mm[1])),
            reverse=True,
        )
        if min(hinted_sides) <= 0.0 or not np.all(np.isfinite(hinted_sides)):
            raise ValueError("目标尺寸提示必须包含两个有限正数")
        hint_error = (
            abs(rectangle_sides[0] - hinted_sides[0]) / hinted_sides[0]
            + abs(rectangle_sides[1] - hinted_sides[1]) / hinted_sides[1]
        )
    return float(hint_error), float(gap_ratio), float(rectangle_area)


def _collinear_edge_overlap_length(first_start, first_end, second_start, second_end, tolerance_mm):
    """返回两条近似共线线段的投影重合长度，不共线时返回0。

    该兼容诊断函数不再参与生产搜索；保留它是为了旧回归工具仍可独立测量接触长度。
    角度采用约5度上限，法向距离采用调用方给出的现场毫米容差。
    """
    first_start = np.asarray(first_start, dtype=np.float64)
    first_end = np.asarray(first_end, dtype=np.float64)
    second_start = np.asarray(second_start, dtype=np.float64)
    second_end = np.asarray(second_end, dtype=np.float64)
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    first_length = float(np.linalg.norm(first_vector))
    second_length = float(np.linalg.norm(second_vector))
    if first_length <= 1e-9 or second_length <= 1e-9:
        return 0.0
    cross_value = first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
    cross_ratio = abs(float(cross_value)) / (first_length * second_length)
    if cross_ratio > math.sin(math.radians(5.0)):
        return 0.0

    unit = first_vector / first_length
    normal = np.asarray((-unit[1], unit[0]), dtype=np.float64)
    first_distance = abs(float((second_start - first_start) @ normal))
    second_distance = abs(float((second_end - first_start) @ normal))
    if max(first_distance, second_distance) > max(0.0, float(tolerance_mm)):
        return 0.0
    projections = (
        float((second_start - first_start) @ unit),
        float((second_end - first_start) @ unit),
    )
    overlap_start = max(0.0, min(projections))
    overlap_end = min(first_length, max(projections))
    return max(0.0, overlap_end - overlap_start)


def _candidate_contact_length(candidate_polygon, placed_polygons, tolerance_mm=3.0):
    """为兼容诊断累计新候选与全部已放置碎片的近似共线接触长度。

    返回值只供旧测试和性能对照使用。v1.4.0生产搜索改用预计算有限边图，不能在
    每个候选上调用本函数，否则会重新引入随已放置边数增长的重复扫描热点。
    """
    candidate = np.asarray(candidate_polygon, dtype=np.float64)
    total_length = 0.0
    for placed_polygon in placed_polygons:
        placed = np.asarray(placed_polygon, dtype=np.float64)
        for candidate_edge in range(len(candidate)):
            candidate_start = candidate[candidate_edge]
            candidate_end = candidate[(candidate_edge + 1) % len(candidate)]
            for placed_edge in range(len(placed)):
                total_length += _collinear_edge_overlap_length(
                    candidate_start,
                    candidate_end,
                    placed[placed_edge],
                    placed[(placed_edge + 1) % len(placed)],
                    tolerance_mm,
                )
    return float(total_length)


def _count_outer_edge_pieces(canonical_polygons, width_mm, height_mm):
    """统计至少有一条边落在目标矩形外框附近的碎片数量。

    主要流程：按目标短边计算尺度相关余量，并检查每条边的两个端点是否同时靠近
    左、右、上、下同一条外框边。题目保证每片至少一条目标外边，因此WHITE容错层
    可用该条件抵消降低填充率带来的错误紧凑组合风险。

    关键参数：canonical_polygons已经旋正并平移到第一象限；width_mm和height_mm是
    目标外接矩形尺寸。返回满足条件的碎片数量。
    """
    short_side_mm = min(float(width_mm), float(height_mm))
    margin_mm = max(
        UNKNOWN_OUTER_EDGE_MIN_MARGIN_MM,
        short_side_mm * UNKNOWN_OUTER_EDGE_MARGIN_RATIO,
    )
    outer_piece_count = 0
    for polygon in canonical_polygons:
        polygon = np.asarray(polygon, dtype=np.float64)
        has_outer_edge = False
        for edge_index in range(len(polygon)):
            start = polygon[edge_index]
            end = polygon[(edge_index + 1) % len(polygon)]
            has_outer_edge = bool(
                (abs(float(start[0])) <= margin_mm and abs(float(end[0])) <= margin_mm)
                or (
                    abs(float(start[0]) - width_mm) <= margin_mm
                    and abs(float(end[0]) - width_mm) <= margin_mm
                )
                or (abs(float(start[1])) <= margin_mm and abs(float(end[1])) <= margin_mm)
                or (
                    abs(float(start[1]) - height_mm) <= margin_mm
                    and abs(float(end[1]) - height_mm) <= margin_mm
                )
            )
            if has_outer_edge:
                break
        outer_piece_count += int(has_outer_edge)
    return int(outer_piece_count)


def _canonicalize_complete_layout(
    placed_by_index,
    pixels_per_mm=2.0,
    min_fill_ratio=UNKNOWN_STRICT_MIN_FILL_RATIO,
    require_all_outer_edges=False,
    metrics=None,
):
    """把完整组合旋正到长边X轴，并用栅格验证目标矩形尺寸、重叠和缝隙。

    主要流程：统一旋正全部碎片，计算目标尺寸、栅格填充率、重叠率和逐片外边数量，
    再按调用方指定的填充门与外边策略验收。严格层使用92%且不额外要求外边；WHITE
    容错层使用文件顶部可调门槛，并强制每片至少拥有一条目标外边。

    关键参数：min_fill_ratio必须位于0～1；require_all_outer_edges只允许容错层启用；
    metrics若为字典会写入本次候选的浮点诊断，不参与机械结果。
    返回值：``(结果, 原因)``；成功时结果为按索引多边形、宽、高和几何分数，
    原因为None；失败原因区分尺寸、填充、重叠或逐片外边拒绝。
    """
    try:
        minimum_fill = float(min_fill_ratio)
    except (TypeError, ValueError) as error:
        raise ValueError("最低填充率必须是0到1之间的有限数字") from error
    if not 0.0 <= minimum_fill <= 1.0 or not math.isfinite(minimum_fill):
        raise ValueError("最低填充率必须是0到1之间的有限数字")
    metric_values = metrics if isinstance(metrics, dict) else None

    indices = sorted(placed_by_index)
    polygons = [np.asarray(placed_by_index[index], dtype=np.float64) for index in indices]
    all_points = np.vstack(polygons).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points.reshape(-1, 1, 2))
    center = np.asarray(rectangle[0], dtype=np.float64)
    angle = math.radians(float(rectangle[2]))
    width_axis = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    height_axis = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    canonical = []
    for polygon in polygons:
        relative = polygon - center
        canonical.append(
            np.column_stack((relative @ width_axis, relative @ height_axis))
        )
    combined = np.vstack(canonical)
    span = np.max(combined, axis=0) - np.min(combined, axis=0)
    if span[0] < span[1]:
        # 交换轴并翻转一轴，仍保持行列式为正，避免把实体镜像。
        canonical = [np.column_stack((polygon[:, 1], -polygon[:, 0])) for polygon in canonical]
        combined = np.vstack(canonical)
    minimum = np.min(combined, axis=0)
    canonical = [polygon - minimum for polygon in canonical]
    combined = np.vstack(canonical)
    width_mm, height_mm = np.max(combined, axis=0)
    outer_piece_count = _count_outer_edge_pieces(canonical, width_mm, height_mm)
    if metric_values is not None:
        metric_values.update(
            {
                "width_mm": float(width_mm),
                "height_mm": float(height_mm),
                "outer_piece_count": int(outer_piece_count),
                "piece_count": int(len(canonical)),
            }
        )
    if not (
        UNKNOWN_LONG_SIDE_RANGE_MM[0] <= width_mm <= UNKNOWN_LONG_SIDE_RANGE_MM[1]
        and UNKNOWN_SHORT_SIDE_RANGE_MM[0] <= height_mm <= UNKNOWN_SHORT_SIDE_RANGE_MM[1]
    ):
        return None, "size_reject"

    masks = _rasterize_polygons(canonical, pixels_per_mm=pixels_per_mm, erode=False)
    union = np.zeros_like(masks[0])
    total_pixels = 0
    for mask in masks:
        total_pixels += int(np.count_nonzero(mask))
        union = cv2.bitwise_or(union, mask)
    union_pixels = max(1, int(np.count_nonzero(union)))
    overlap_ratio = max(0.0, float(total_pixels - union_pixels) / float(union_pixels))

    # 在旋正后的目标外框内统计填充率，黑色缝隙和缺片都会直接降低该值。
    rectangle_pixels = max(
        1.0,
        (width_mm * pixels_per_mm + 1.0) * (height_mm * pixels_per_mm + 1.0),
    )
    fill_ratio = min(1.0, float(union_pixels) / rectangle_pixels)
    if metric_values is not None:
        metric_values.update(
            {
                "fill_ratio": float(fill_ratio),
                "overlap_ratio": float(overlap_ratio),
            }
        )
    if overlap_ratio > UNKNOWN_MAX_OVERLAP_RATIO:
        return None, "overlap_reject"
    if fill_ratio < minimum_fill:
        return None, "fill_reject"
    if require_all_outer_edges and outer_piece_count != len(canonical):
        return None, "outer_edge_reject"
    geometry_score = (1.0 - fill_ratio) * 50.0 + overlap_ratio * 50.0
    return (
        (
            {index: polygon for index, polygon in zip(indices, canonical)},
            float(width_mm),
            float(height_mm),
            float(geometry_score),
        ),
        None,
    )


def _build_unknown_success_plan(
    solver_pieces,
    canonical_result,
    work_region_mm,
    split_y_mm,
    score,
    search_nodes,
    diagnostics,
):
    """把已通过硬验收的规范布局转换成下半区机械规划。

    主要流程：在红线下方计算居中的目标矩形，再逐片换算目标轮廓、中心和最短旋转角。
    canonical_result来自`_canonicalize_complete_layout()`，因此本函数不再重复尺寸、填充
    和重叠判断。返回独立AssemblyPlan；下半工作区放不下时返回None。
    """
    canonical_by_index, target_width, target_height, _geometry_score = canonical_result
    target_rect = _target_rect_in_lower_region(
        work_region_mm,
        split_y_mm,
        (target_width, target_height),
    )
    if target_rect is None:
        return None
    target_origin = np.asarray(target_rect[:2], dtype=np.float64)
    placements = []
    for index, solver_piece in enumerate(solver_pieces):
        target_polygon = canonical_by_index[index] + target_origin
        rotation_delta, _ = _best_rotation_delta_deg(
            solver_piece["source_vertices"],
            target_polygon,
        )
        placements.append(
            AssemblyPlacement(
                solver_piece["id"],
                solver_piece["source_center"],
                _polygon_centroid(target_polygon),
                target_polygon,
                rotation_delta,
            )
        )
    placements.sort(key=lambda placement: placement.piece_id)
    return AssemblyPlan(
        True,
        placements=placements,
        target_rect_mm=target_rect,
        score=score,
        reason="ok",
        search_nodes=search_nodes,
        diagnostics=diagnostics,
    )


def _build_graph_edge_candidates(
    solver_pieces,
    match_ratio=UNKNOWN_GRAPH_MATCH_RATIO,
    candidate_limit=UNKNOWN_GRAPH_MAX_EDGE_CANDIDATES,
):
    """构造不同碎片、不同形状候选之间最多32条无向整边连接假设。

    每条关系同时保存两片的候选编号、边号、最高候选等级和候选质量总分。16%门槛
    用于容忍远距离角点误差，但短于题目最短接缝的边不会进入图。候选先按等级和
    候选质量排序，再比较边长误差和接缝长度，保证首选形状不被次级候选挤出有限图。
    返回元组；后续连接集合必须固定同一碎片的候选编号，不能跨候选借边。
    """
    ratio_limit = float(match_ratio)
    maximum_candidates = int(candidate_limit)
    if not 0.0 <= ratio_limit < 1.0 or maximum_candidates <= 0:
        raise ValueError("图边误差和候选上限参数无效")

    candidates = []
    for first_index, first_piece in enumerate(solver_pieces):
        for second_index in range(first_index + 1, len(solver_pieces)):
            second_piece = solver_pieces[second_index]
            for first_hypothesis_index, first_hypothesis in enumerate(
                first_piece["hypotheses"]
            ):
                first_rank = int(first_hypothesis.get("rank", first_hypothesis_index))
                for second_hypothesis_index, second_hypothesis in enumerate(
                    second_piece["hypotheses"]
                ):
                    second_rank = int(
                        second_hypothesis.get("rank", second_hypothesis_index)
                    )
                    maximum_rank = max(first_rank, second_rank)
                    hypothesis_score = float(first_hypothesis.get("score", 0.0)) + float(
                        second_hypothesis.get("score", 0.0)
                    )
                    for first_edge, first_length in enumerate(
                        first_hypothesis["edge_lengths"]
                    ):
                        first_length = float(first_length)
                        if first_length < UNKNOWN_MIN_SEAM_LENGTH_MM:
                            continue
                        for second_edge, second_length in enumerate(
                            second_hypothesis["edge_lengths"]
                        ):
                            second_length = float(second_length)
                            if second_length < UNKNOWN_MIN_SEAM_LENGTH_MM:
                                continue
                            long_length = max(first_length, second_length)
                            if long_length <= 1e-9:
                                continue
                            relative_error = abs(first_length - second_length) / long_length
                            if relative_error > ratio_limit:
                                continue
                            candidates.append(
                                {
                                    "relative_error": float(relative_error),
                                    "first_index": int(first_index),
                                    "first_edge": int(first_edge),
                                    "second_index": int(second_index),
                                    "second_edge": int(second_edge),
                                    # 无向GRAPH沿用first/second索引，但候选字段统一采用
                                    # source/target命名，分别与first/second一一对应。
                                    "source_hypothesis": int(first_hypothesis_index),
                                    "target_hypothesis": int(second_hypothesis_index),
                                    "maximum_rank": int(maximum_rank),
                                    "hypothesis_score": float(hypothesis_score),
                                    "matched_length_mm": float(
                                        min(first_length, second_length)
                                    ),
                                }
                            )
    candidates.sort(
        key=lambda candidate: (
            candidate["maximum_rank"],
            candidate["hypothesis_score"],
            candidate["relative_error"],
            -candidate["matched_length_mm"],
            candidate["first_index"],
            candidate["second_index"],
            candidate["source_hypothesis"],
            candidate["target_hypothesis"],
            candidate["first_edge"],
            candidate["second_edge"],
        )
    )
    return tuple(candidates[:maximum_candidates])


def _graph_indices_are_connected(piece_count, relations):
    """判断连接关系是否覆盖全部碎片并形成单个连通分量。"""
    if piece_count <= 0:
        return False
    adjacency = [[] for _ in range(piece_count)]
    for relation in relations:
        first_index = relation["first_index"]
        second_index = relation["second_index"]
        adjacency[first_index].append(second_index)
        adjacency[second_index].append(first_index)
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == piece_count


def _collect_graph_matching_sets(
    piece_count,
    candidates,
    matching_set_limit=UNKNOWN_GRAPH_MAX_MATCHING_SETS,
):
    """枚举最多90组边唯一、度数受限且覆盖全部碎片的连接关系。

    两片只需一条连接边；三至四片先找N-1条生成树，只有找不到时才尝试N条闭环。
    同一碎片物理边在一组中只能使用一次，每片连接度不超过3；同一碎片一旦由某条
    关系选择了形状候选，后续关系必须使用同一候选。返回值先按最高候选等级和候选
    分数排序，再比较累计边长误差，使WHITE优先检查排名0的可信组合。
    """
    count = int(piece_count)
    limit = int(matching_set_limit)
    if not 1 <= count <= 4 or limit <= 0:
        raise ValueError("图碎片数量和组合上限参数无效")
    if count == 1:
        return (tuple(),)
    candidates = tuple(candidates)
    if not candidates:
        return tuple()

    matching_sets = []

    def search(
        start_index,
        target_count,
        selected,
        used_edges,
        degrees,
        selected_hypotheses,
    ):
        """深度优先选择固定条数关系，并固定每片已经选定的候选编号。"""
        if len(matching_sets) >= limit:
            return
        if len(selected) == target_count:
            if _graph_indices_are_connected(count, selected):
                matching_sets.append(tuple(selected))
            return
        remaining_needed = target_count - len(selected)
        if len(candidates) - start_index < remaining_needed:
            return

        for candidate_index in range(start_index, len(candidates)):
            relation = candidates[candidate_index]
            first_index = relation["first_index"]
            second_index = relation["second_index"]
            first_hypothesis = int(relation.get("source_hypothesis", 0))
            second_hypothesis = int(relation.get("target_hypothesis", 0))
            if (
                first_index in selected_hypotheses
                and selected_hypotheses[first_index] != first_hypothesis
            ) or (
                second_index in selected_hypotheses
                and selected_hypotheses[second_index] != second_hypothesis
            ):
                continue
            first_edge = (first_index, first_hypothesis, relation["first_edge"])
            second_edge = (second_index, second_hypothesis, relation["second_edge"])
            if first_edge in used_edges or second_edge in used_edges:
                continue
            if degrees[first_index] >= 3 or degrees[second_index] >= 3:
                continue

            used_edges.add(first_edge)
            used_edges.add(second_edge)
            degrees[first_index] += 1
            degrees[second_index] += 1
            selected.append(relation)
            previous_first = selected_hypotheses.get(first_index)
            previous_second = selected_hypotheses.get(second_index)
            selected_hypotheses[first_index] = first_hypothesis
            selected_hypotheses[second_index] = second_hypothesis
            search(
                candidate_index + 1,
                target_count,
                selected,
                used_edges,
                degrees,
                selected_hypotheses,
            )
            selected.pop()
            if previous_first is None:
                selected_hypotheses.pop(first_index, None)
            else:
                selected_hypotheses[first_index] = previous_first
            if previous_second is None:
                selected_hypotheses.pop(second_index, None)
            else:
                selected_hypotheses[second_index] = previous_second
            degrees[first_index] -= 1
            degrees[second_index] -= 1
            used_edges.remove(first_edge)
            used_edges.remove(second_edge)
            if len(matching_sets) >= limit:
                return

    target_counts = (1,) if count == 2 else (count - 1, count)
    for target_count in target_counts:
        search(0, target_count, [], set(), [0] * count, {})
        if matching_sets:
            break
    matching_sets.sort(
        key=lambda relations: (
            max((int(relation.get("maximum_rank", 0)) for relation in relations), default=0),
            sum(float(relation.get("hypothesis_score", 0.0)) for relation in relations),
            sum(relation["relative_error"] for relation in relations),
        )
    )
    return tuple(matching_sets[:limit])


def _transform_polygon_with_pose(polygon, pose):
    """用`(旋转矩阵, 平移向量)`刚体位姿转换毫米多边形。"""
    rotation, translation = pose
    return np.asarray(polygon, dtype=np.float64) @ rotation.T + translation


def _align_edge_midpoint_pose(source_polygon, source_edge, target_start, target_end):
    """计算源边反向对齐目标边中点的刚体位姿。

    检测边长不完全相等时不能缩放真实碎片；中点对齐会把剩余长度误差均分到两个
    端点，比单端点锚定更不容易在候选矩形一侧产生集中缺口或重叠。
    """
    source = np.asarray(source_polygon, dtype=np.float64)
    source_start = source[source_edge]
    source_end = source[(source_edge + 1) % len(source)]
    source_vector = source_end - source_start
    target_vector = np.asarray(target_start, dtype=np.float64) - np.asarray(
        target_end,
        dtype=np.float64,
    )
    source_angle = math.atan2(source_vector[1], source_vector[0])
    target_angle = math.atan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    source_midpoint = (source_start + source_end) * 0.5
    rotated_midpoint = source_midpoint @ rotation.T
    target_midpoint = (
        np.asarray(target_start, dtype=np.float64)
        + np.asarray(target_end, dtype=np.float64)
    ) * 0.5
    translation = target_midpoint - rotated_midpoint
    return rotation, translation


def _propagate_graph_layout(solver_pieces, relations):
    """按关系指定的形状候选传播全部碎片刚体位姿。

    主要流程：先从关系中建立每片唯一的候选编号；若同片出现候选冲突立即拒绝。
    随后固定第0片位姿，通过反向接缝逐片传播旋转和平移。每片最终同时生成接缝
    多边形和完整轮廓多边形；闭环再次到达已放置片时，只比较同一候选的接缝位置。

    返回值：成功时为`(状态字典, 闭合误差)`，状态含`hypothesis_by_index`、
    `pose_by_index`、`seam_by_index`和`outline_by_index`；关系冲突、图断开或数值异常
    时返回`(None, inf)`。
    """
    piece_count = len(solver_pieces)
    adjacency = [[] for _ in range(piece_count)]
    hypothesis_by_index = {}
    for relation in relations:
        first_index = int(relation["first_index"])
        second_index = int(relation["second_index"])
        first_hypothesis = int(relation.get("source_hypothesis", 0))
        second_hypothesis = int(relation.get("target_hypothesis", 0))
        if (
            first_index in hypothesis_by_index
            and hypothesis_by_index[first_index] != first_hypothesis
        ) or (
            second_index in hypothesis_by_index
            and hypothesis_by_index[second_index] != second_hypothesis
        ):
            return None, float("inf")
        hypothesis_by_index[first_index] = first_hypothesis
        hypothesis_by_index[second_index] = second_hypothesis
        adjacency[first_index].append(relation)
        adjacency[second_index].append(relation)

    # 单片布局没有连接关系；多片连通关系则应覆盖全部碎片。未覆盖项在后续断开检查
    # 中拒绝，这里只为合法单片和第0片提供排名0默认值。
    if piece_count == 1:
        hypothesis_by_index[0] = 0
    elif 0 not in hypothesis_by_index:
        return None, float("inf")

    poses = [None] * piece_count
    poses[0] = (np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64))
    pending = [0]
    closure_error = 0.0
    while pending:
        current_index = pending.pop()
        current_hypothesis_index = hypothesis_by_index[current_index]
        current_hypothesis = solver_pieces[current_index]["hypotheses"][
            current_hypothesis_index
        ]
        current_polygon = _transform_polygon_with_pose(
            current_hypothesis["local_vertices"],
            poses[current_index],
        )
        for relation in adjacency[current_index]:
            if relation["first_index"] == current_index:
                current_edge = relation["first_edge"]
                neighbor_index = relation["second_index"]
                neighbor_edge = relation["second_edge"]
                neighbor_hypothesis_index = int(
                    relation.get("target_hypothesis", 0)
                )
            else:
                current_edge = relation["second_edge"]
                neighbor_index = relation["first_index"]
                neighbor_edge = relation["first_edge"]
                neighbor_hypothesis_index = int(
                    relation.get("source_hypothesis", 0)
                )
            if hypothesis_by_index.get(neighbor_index) != neighbor_hypothesis_index:
                return None, float("inf")
            neighbor_hypothesis = solver_pieces[neighbor_index]["hypotheses"][
                neighbor_hypothesis_index
            ]
            proposed_pose = _edge_alignment_pose(
                neighbor_hypothesis["local_vertices"],
                neighbor_edge,
                current_polygon,
                current_edge,
                anchor="midpoint",
            )
            if poses[neighbor_index] is None:
                poses[neighbor_index] = proposed_pose
                pending.append(neighbor_index)
                continue
            existing_polygon = _transform_polygon_with_pose(
                neighbor_hypothesis["local_vertices"],
                poses[neighbor_index],
            )
            proposed_polygon = _transform_polygon_with_pose(
                neighbor_hypothesis["local_vertices"],
                proposed_pose,
            )
            closure_error += float(
                np.mean(np.linalg.norm(existing_polygon - proposed_polygon, axis=1))
            )

    if any(pose is None for pose in poses):
        return None, float("inf")
    transformed_by_index = {
        index: _transform_solver_outline(
            solver_pieces[index],
            hypothesis_by_index[index],
            pose,
        )
        for index, pose in enumerate(poses)
    }
    if not all(
        np.all(np.isfinite(transformed["seam_polygon"]))
        and np.all(np.isfinite(transformed["outline_polygon"]))
        for transformed in transformed_by_index.values()
    ):
        return None, float("inf")
    return (
        {
            "hypothesis_by_index": dict(hypothesis_by_index),
            "pose_by_index": {
                index: poses[index]
                for index in range(piece_count)
            },
            "seam_by_index": {
                index: transformed_by_index[index]["seam_polygon"]
                for index in range(piece_count)
            },
            "outline_by_index": {
                index: transformed_by_index[index]["outline_polygon"]
                for index in range(piece_count)
            },
        },
        float(closure_error),
    )


def _solve_unknown_graph_fast_path_steps(
    pieces,
    work_region_mm,
    split_y_mm,
    pixels_per_mm=2.0,
    candidate_limit=UNKNOWN_GRAPH_MAX_EDGE_CANDIDATES,
    matching_set_limit=UNKNOWN_GRAPH_MAX_MATCHING_SETS,
    progress=None,
):
    """逐候选执行有界连接图WHITE快解，并在每组完整布局后让出一次。

    主要流程：规范化1～4片输入，构造最多32条候选边和90组连通关系，逐组传播刚体
    位姿；每组先用92%严格门验收，严格解立即返回。只有所有严格候选都失败时，才从
    WHITE的86%容错候选中返回评分最低且每片都有目标外边的一组。所有假设均失败时
    返回None，由调用方进入下一阶段。每检查一组连通关系就yield一次，结束时通过
    StopIteration.value返回`(plan, diagnostics)`。progress可选共享字典用于在yield前保存
    已验证规划，避免单个硬验收越过截止线时由外层超时收尾误丢结果。
    """
    diagnostics = {
        "graph_edge_candidates": 0,
        "graph_matching_sets": 0,
        "graph_layouts_checked": 0,
        "graph_fast_path": 0,
        "graph_strict_accept": 0,
        "graph_relaxed_candidates": 0,
        "graph_relaxed_fill_reject": 0,
        "graph_outer_edge_reject": 0,
    }
    progress = progress if isinstance(progress, dict) else {}
    pieces = list(pieces)
    if not 1 <= len(pieces) <= 4:
        return None, diagnostics
    try:
        pixels_per_mm = float(pixels_per_mm)
        if pixels_per_mm <= 0.0 or not math.isfinite(pixels_per_mm):
            raise ValueError
        solver_pieces = [
            _solver_piece(piece, index)
            for index, piece in enumerate(pieces)
        ]
        candidates = _build_graph_edge_candidates(
            solver_pieces,
            candidate_limit=candidate_limit,
        )
        matching_sets = _collect_graph_matching_sets(
            len(solver_pieces),
            candidates,
            matching_set_limit=matching_set_limit,
        )
    except (KeyError, TypeError, ValueError):
        return None, diagnostics

    diagnostics["graph_edge_candidates"] = int(len(candidates))
    diagnostics["graph_matching_sets"] = int(len(matching_sets))
    best_relaxed = None
    best_relaxed_score = float("inf")
    best_relaxed_metrics = {}

    def evaluate_relations(relations):
        """验收一组连接关系，并在得到严格解时返回规划。

        该辅助函数把一次最昂贵的传播、栅格验收和计划构造收拢成一个工作单元；
        容错候选只更新外层最优值，不会因枚举顺序提前结束。
        """
        nonlocal best_relaxed, best_relaxed_score, best_relaxed_metrics
        layout_state, closure_error = _propagate_graph_layout(
            solver_pieces,
            relations,
        )
        if layout_state is None:
            return None
        # 接缝候选只负责传播位姿；尺寸、重叠和填充率必须读取同一位姿下的完整轮廓。
        placed_by_index = layout_state["outline_by_index"]
        diagnostics["graph_layouts_checked"] += 1
        strict_metrics = {}
        canonical_result, rejection_reason = _canonicalize_complete_layout(
            placed_by_index,
            pixels_per_mm=pixels_per_mm,
            min_fill_ratio=UNKNOWN_STRICT_MIN_FILL_RATIO,
            metrics=strict_metrics,
        )
        accepted_relaxed = False
        candidate_metrics = strict_metrics
        if canonical_result is None and rejection_reason == "fill_reject":
            # 只有填充率不足可以进入WHITE容错层；尺寸和重叠属于机械安全硬门，
            # 在这一层绝不放宽。逐片外边约束用于排除偶然紧凑但不是矩形的组合。
            relaxed_metrics = {}
            canonical_result, relaxed_reason = _canonicalize_complete_layout(
                placed_by_index,
                pixels_per_mm=pixels_per_mm,
                min_fill_ratio=UNKNOWN_RELAXED_MIN_FILL_RATIO,
                require_all_outer_edges=True,
                metrics=relaxed_metrics,
            )
            if canonical_result is None:
                if relaxed_reason == "fill_reject":
                    diagnostics["graph_relaxed_fill_reject"] += 1
                elif relaxed_reason == "outer_edge_reject":
                    diagnostics["graph_outer_edge_reject"] += 1
                return None
            accepted_relaxed = True
            candidate_metrics = relaxed_metrics
            diagnostics["graph_relaxed_candidates"] += 1
        elif canonical_result is None:
            return None
        relation_error = sum(
            relation["relative_error"] for relation in relations
        )
        geometry_score = float(canonical_result[3])
        score = geometry_score + relation_error * 10.0 + closure_error * 0.1
        if accepted_relaxed:
            # 不能在首个86%候选处结束，否则候选枚举顺序会影响目标。记录最多90组中
            # 的最低分容错候选，循环结束且始终没有严格解时才允许返回它。
            if score < best_relaxed_score:
                best_relaxed_score = float(score)
                best_relaxed = canonical_result
                best_relaxed_metrics = dict(candidate_metrics)
            return None

        success_diagnostics = dict(diagnostics)
        success_diagnostics["graph_fast_path"] = 1
        success_diagnostics["graph_strict_accept"] = 1
        success_diagnostics["relaxed_accept"] = 0
        success_diagnostics["fill_permille"] = int(
            round(float(candidate_metrics.get("fill_ratio", 0.0)) * 1000.0)
        )
        success_diagnostics["overlap_permille"] = int(
            round(float(candidate_metrics.get("overlap_ratio", 0.0)) * 1000.0)
        )
        success_diagnostics["outer_piece_count"] = int(
            candidate_metrics.get("outer_piece_count", 0)
        )
        # 图快路径在第一个硬验收合格布局处返回；以0节点记录首解，兼容WHITE既有诊断。
        success_diagnostics["first_solution_node"] = 0
        plan = _build_unknown_success_plan(
            solver_pieces,
            canonical_result,
            work_region_mm,
            split_y_mm,
            score,
            search_nodes=0,
            diagnostics=success_diagnostics,
        )
        if plan is not None:
            diagnostics.update(success_diagnostics)
            return plan
        return None

    for relations in matching_sets:
        plan = evaluate_relations(relations)
        if plan is not None:
            # 严格规划已经通过全部机械硬门。必须先保存再yield；否则外层在这个慢工作
            # 单元后发现预算耗尽并关闭生成器时，流水线永远收不到下一次恢复后的return。
            progress["current_stage"] = "graph"
            progress["result_source"] = "graph"
            progress["best_plan"] = plan
        # 一个工作单元只包含一组传播与最多两轮栅格验收；先把控制权交还相机主循环，
        # 下一次恢复时再返回已经形成的严格解，确保单帧工作单元上限真实生效。
        yield None
        if plan is not None:
            return plan, diagnostics

    if best_relaxed is not None:
        # 走到这里证明90组内没有92%严格解，才允许采用86%容错结果。
        success_diagnostics = dict(diagnostics)
        success_diagnostics["graph_fast_path"] = 1
        success_diagnostics["relaxed_accept"] = 1
        success_diagnostics["fill_permille"] = int(
            round(float(best_relaxed_metrics.get("fill_ratio", 0.0)) * 1000.0)
        )
        success_diagnostics["overlap_permille"] = int(
            round(float(best_relaxed_metrics.get("overlap_ratio", 0.0)) * 1000.0)
        )
        success_diagnostics["outer_piece_count"] = int(
            best_relaxed_metrics.get("outer_piece_count", 0)
        )
        success_diagnostics["first_solution_node"] = 0
        plan = _build_unknown_success_plan(
            solver_pieces,
            best_relaxed,
            work_region_mm,
            split_y_mm,
            best_relaxed_score,
            search_nodes=0,
            diagnostics=success_diagnostics,
        )
        if plan is not None:
            diagnostics.update(success_diagnostics)
            progress["current_stage"] = "graph"
            progress["result_source"] = "graph"
            progress["best_plan"] = plan
            return plan, diagnostics
    return None, diagnostics


def _consume_solver_steps(iterator):
    """同步消费一个增量求解生成器，并返回其StopIteration.value。

    该兼容辅助函数只供PC测试和旧同步API使用；相机运行器必须直接保存生成器并按帧
    调用next，不能使用本函数一次性跑完。参数必须是Python迭代器，返回值由生成器决定。
    """
    while True:
        try:
            next(iterator)
        except StopIteration as completed:
            return completed.value


def _solve_unknown_graph_fast_path(
    pieces,
    work_region_mm,
    split_y_mm,
    pixels_per_mm=2.0,
    candidate_limit=UNKNOWN_GRAPH_MAX_EDGE_CANDIDATES,
    matching_set_limit=UNKNOWN_GRAPH_MAX_MATCHING_SETS,
    incremental=False,
    progress=None,
):
    """创建GRAPH增量核心；默认同步消费以保持既有调用兼容。

    incremental=True时返回尚未执行的生成器，供AssemblyRuntime纳入统一帧预算；默认
    False时返回原有`(plan, diagnostics)`。输入碎片会在生成器首次推进时独立物化。
    """
    steps = _solve_unknown_graph_fast_path_steps(
        pieces,
        work_region_mm,
        split_y_mm,
        pixels_per_mm=pixels_per_mm,
        candidate_limit=candidate_limit,
        matching_set_limit=matching_set_limit,
        progress=progress,
    )
    if bool(incremental):
        return steps
    return _consume_solver_steps(steps)


def _solve_unknown_four_fast_path_steps(
    pieces,
    work_region_mm,
    split_y_mm,
    edge_length_tolerance_mm=2.5,
    pixels_per_mm=2.0,
    beam_width=UNKNOWN_FOUR_FAST_BEAM_WIDTH,
    max_work_units=UNKNOWN_FOUR_FAST_MAX_WORK_UNITS,
    progress=None,
):
    """逐候选执行UNKNOWN WHITE四片分层Beam快解。

    主要流程：固定第0片消除整体自由度，复用现有整边与`segmented`兼容图；依次生成
    两片、三片和四片状态，每层按量化位姿全局去重并按外框紧凑度限宽。中间候选用
    `_candidate_overlaps_fast`剪枝，完整状态仍调用现有92%严格与86%容错硬验收。

    关键参数：beam_width限制每个未完成层保留状态数，max_work_units限制对齐候选检查
    总数；中间层达到上限时直接返回无解，完整层达到上限时仍验收已经生成的候选，
    只有这些候选无解才返回None并由调用方继续FALLBACK。每个候选或完整验收后yield；
    StopIteration.value返回`(plan, diagnostics)`。progress可选共享字典用于在yield前保存
    已通过硬门的规划，机械目标仍由公共构造函数生成。
    """
    diagnostics = {
        "four_fast_path": 0,
        "four_pair_states": 0,
        "four_triple_states": 0,
        "four_complete_states": 0,
        "four_pair_parents_expanded": 0,
        "four_triple_parents_expanded": 0,
        "four_complete_parents_expanded": 0,
        "four_complete_checked": 0,
        "four_deduplicated": 0,
        "four_overlap_reject": 0,
        "four_partial_reject": 0,
        "four_size_reject": 0,
        "four_fill_reject": 0,
        "four_outer_edge_reject": 0,
        "four_work_units": 0,
        "four_work_limit_reached": 0,
        "four_active_limit_reached": 0,
        "four_active_elapsed_ms": 0,
        "four_used_segmented": 0,
        "fast_convex_checks": 0,
        "fast_raster_fallbacks": 0,
    }
    progress = progress if isinstance(progress, dict) else {}
    pieces = list(pieces)
    if len(pieces) != 4:
        return None, diagnostics
    try:
        selected_beam_width = int(beam_width)
        selected_work_limit = int(max_work_units)
        selected_pixels_per_mm = float(pixels_per_mm)
        length_tolerance = float(edge_length_tolerance_mm)
        if (
            selected_beam_width <= 0
            or selected_work_limit <= 0
            or selected_pixels_per_mm <= 0.0
            or not math.isfinite(selected_pixels_per_mm)
            or length_tolerance < 0.0
            or not math.isfinite(length_tolerance)
        ):
            raise ValueError
        # GRAPH、FOURFAST和FALLBACK必须共享同一候选构造器；当前阶段旧搜索字段统一
        # 指向评分最高候选，Task 4/5会继续扩展候选等级而不改变完整轮廓。
        solver_pieces = [
            _solver_piece(piece, index)
            for index, piece in enumerate(pieces)
        ]
        # 中间两片/三片阶段必须预先知道最终四片总面积，否则同一接缝误差会因当前
        # 分母过小被放大，提前剪掉最终能够通过3%硬门的合法布局。
        final_total_piece_area = sum(
            abs(
                float(
                    cv2.contourArea(
                        solver_piece["local_vertices"].astype(np.float32)
                    )
                )
            )
            for solver_piece in solver_pieces
        )
        edge_graph = _build_edge_compatibility_graph(
            solver_pieces,
            base_tolerance_mm=length_tolerance,
        )
    except (KeyError, TypeError, ValueError):
        return None, diagnostics

    def pose_key(placed_by_index):
        """把完整状态量化为稳定键，合并由不同接缝关系得到的同一刚体布局。"""
        return tuple(
            (
                int(index),
                tuple(
                    int(value)
                    for value in np.rint(
                        np.asarray(placed_by_index[index], dtype=np.float64).flatten()
                        * 100.0
                    )
                ),
            )
            for index in sorted(placed_by_index)
        )

    def build_priority(partial_priority, segmented_count, full_error, matched_length):
        """构造分层全局排序键，优先紧凑布局并轻度偏向可靠整边和长接缝。"""
        _hint_error, gap_ratio, rectangle_area = partial_priority
        return (
            float(gap_ratio) + float(segmented_count) * 0.015,
            float(full_error),
            -float(matched_length),
            float(rectangle_area),
        )

    def active_budget_reached():
        """读取外层任务的FOURFAST中止请求，并把阶段活动耗时写入诊断。

        外层`UnknownSolveJob.advance()`只能在生成器yield后的工作单元边界统计真实CPU
        时间，因此本函数不自行读时钟。返回True表示四片快路径应正常结束并交给FALLBACK；
        这不是整个UNKNOWN任务超时，不能生成`solver_timeout`。
        """
        active_seconds = max(
            0.0,
            float(progress.get("four_fast_active_elapsed_seconds", 0.0)),
        )
        diagnostics["four_active_elapsed_ms"] = int(
            round(active_seconds * 1000.0)
        )
        if not bool(progress.get("four_fast_abort_requested", False)):
            return False
        diagnostics["four_active_limit_reached"] = 1
        return True

    if active_budget_reached():
        return None, diagnostics

    initial_polygon = solver_pieces[0]["local_vertices"].copy()
    states = [
        {
            "placed": {0: initial_polygon},
            "remaining": (1, 2, 3),
            "segmented_count": 0,
            "full_error": 0.0,
            "matched_length": 0.0,
            "priority": (0.0, 0.0, 0.0, 0.0),
        }
    ]

    for placed_count in (2, 3, 4):
        if active_budget_reached():
            return None, diagnostics
        # `states`在上一层已经按beam_width截断，因此这里必须展开全部保留状态。
        # 若再次只取四分之一，排名9～32的状态虽然被记录为“保留”，实际永远无法进入
        # 下一层，会无故降低快路径召回并把可解输入推回慢FALLBACK。
        next_by_pose = {}
        work_limit_reached = False
        parent_states = states[:selected_beam_width]
        parent_diagnostic_key = {
            2: "four_pair_parents_expanded",
            3: "four_triple_parents_expanded",
            4: "four_complete_parents_expanded",
        }[placed_count]
        for state in parent_states:
            if active_budget_reached():
                return None, diagnostics
            if work_limit_reached:
                break
            # 只有真正进入展开循环的父状态才计数。工作量在层中途触顶时，这个值会
            # 小于保留状态数，现场即可判断召回受限于候选不足还是预算不足。
            diagnostics[parent_diagnostic_key] += 1
            placed_by_index = state["placed"]
            placed_polygons = list(placed_by_index.values())
            for source_index in state["remaining"]:
                if active_budget_reached():
                    return None, diagnostics
                if work_limit_reached:
                    break
                source_polygon = solver_pieces[source_index]["local_vertices"]
                for target_index, target_polygon in placed_by_index.items():
                    if active_budget_reached():
                        return None, diagnostics
                    if work_limit_reached:
                        break
                    relations = edge_graph.get((source_index, target_index), ())
                    for relation in relations[:UNKNOWN_FOUR_FAST_RELATION_LIMIT]:
                        if active_budget_reached():
                            return None, diagnostics
                        if work_limit_reached:
                            break
                        candidates = _align_source_edge_candidates(
                            source_polygon,
                            relation["source_edge"],
                            target_polygon,
                            relation["target_edge"],
                        )
                        for candidate_polygon in candidates:
                            if active_budget_reached():
                                return None, diagnostics
                            if diagnostics["four_work_units"] >= selected_work_limit:
                                diagnostics["four_work_limit_reached"] = 1
                                work_limit_reached = True
                                break
                            diagnostics["four_work_units"] += 1
                            if _candidate_overlaps_fast(
                                candidate_polygon,
                                placed_polygons,
                                pixels_per_mm=selected_pixels_per_mm,
                                diagnostics=diagnostics,
                                final_total_piece_area=final_total_piece_area,
                            ):
                                diagnostics["four_overlap_reject"] += 1
                                yield None
                                continue
                            updated = dict(placed_by_index)
                            updated[source_index] = candidate_polygon
                            partial_priority = _partial_layout_priority(
                                list(updated.values()),
                                tolerance_mm=length_tolerance,
                            )
                            if partial_priority is None:
                                diagnostics["four_partial_reject"] += 1
                                yield None
                                continue
                            relation_is_segmented = relation["kind"] == "segmented"
                            segmented_count = (
                                int(state["segmented_count"])
                                + int(relation_is_segmented)
                            )
                            full_error = float(state["full_error"])
                            if not relation_is_segmented:
                                matched_length = max(
                                    1e-6,
                                    float(relation["matched_length_mm"]),
                                )
                                full_error += float(relation["length_error_mm"]) / matched_length
                            total_matched_length = (
                                float(state["matched_length"])
                                + float(relation["matched_length_mm"])
                            )
                            remaining = tuple(
                                index
                                for index in state["remaining"]
                                if index != source_index
                            )
                            candidate_state = {
                                "placed": updated,
                                "remaining": remaining,
                                "segmented_count": segmented_count,
                                "full_error": full_error,
                                "matched_length": total_matched_length,
                                "priority": build_priority(
                                    partial_priority,
                                    segmented_count,
                                    full_error,
                                    total_matched_length,
                                ),
                            }
                            key = pose_key(updated)
                            previous = next_by_pose.get(key)
                            if previous is not None:
                                diagnostics["four_deduplicated"] += 1
                                if previous["priority"] <= candidate_state["priority"]:
                                    yield None
                                    continue
                            next_by_pose[key] = candidate_state
                            # 对齐、重叠、外框和去重组成一个高成本工作单元。保存候选后
                            # 立即让出，使24ms/64上限能够在任意Beam层之间恢复。
                            yield None

        if active_budget_reached():
            return None, diagnostics
        next_states = sorted(
            next_by_pose.values(),
            key=lambda state: state["priority"],
        )
        if placed_count == 2:
            diagnostics["four_pair_states"] = int(len(next_states))
        elif placed_count == 3:
            diagnostics["four_triple_states"] = int(len(next_states))
        else:
            diagnostics["four_complete_states"] = int(len(next_states))

        if not next_states:
            return None, diagnostics
        if placed_count < 4:
            states = next_states[:selected_beam_width]
            if work_limit_reached:
                # 未完整生成中间层会使全局排序带有父状态顺序偏差；交给旧FALLBACK比
                # 在不完整Beam上继续给出机械结果更稳妥。
                return None, diagnostics
            continue

        # 完整层最多验收四倍Beam；尺寸不合格在栅格化前即可快速拒绝。达到工作上限时
        # 已生成的完整状态仍然是合法候选，可安全完成硬验收。
        complete_states = next_states[: selected_beam_width * 4]
        best_relaxed = None
        best_relaxed_score = float("inf")
        for state in complete_states:
            if active_budget_reached():
                return None, diagnostics
            diagnostics["four_complete_checked"] += 1
            strict_metrics = {}
            canonical_result, rejection_reason = _canonicalize_complete_layout(
                state["placed"],
                pixels_per_mm=selected_pixels_per_mm,
                min_fill_ratio=UNKNOWN_STRICT_MIN_FILL_RATIO,
                metrics=strict_metrics,
            )
            accepted_relaxed = False
            candidate_metrics = strict_metrics
            if canonical_result is None:
                if rejection_reason == "size_reject":
                    diagnostics["four_size_reject"] += 1
                elif rejection_reason == "fill_reject":
                    diagnostics["four_fill_reject"] += 1
                if rejection_reason != "fill_reject":
                    yield None
                    continue
                relaxed_metrics = {}
                canonical_result, relaxed_reason = _canonicalize_complete_layout(
                    state["placed"],
                    pixels_per_mm=selected_pixels_per_mm,
                    min_fill_ratio=UNKNOWN_RELAXED_MIN_FILL_RATIO,
                    require_all_outer_edges=True,
                    metrics=relaxed_metrics,
                )
                if canonical_result is None:
                    if relaxed_reason == "outer_edge_reject":
                        diagnostics["four_outer_edge_reject"] += 1
                    yield None
                    continue
                accepted_relaxed = True
                candidate_metrics = relaxed_metrics

            geometry_score = float(canonical_result[3])
            success_diagnostics = dict(diagnostics)
            success_diagnostics["four_fast_path"] = 1
            success_diagnostics["four_used_segmented"] = int(
                state["segmented_count"]
            )
            success_diagnostics["relaxed_accept"] = int(accepted_relaxed)
            success_diagnostics["fill_permille"] = int(
                round(float(candidate_metrics.get("fill_ratio", 0.0)) * 1000.0)
            )
            success_diagnostics["overlap_permille"] = int(
                round(float(candidate_metrics.get("overlap_ratio", 0.0)) * 1000.0)
            )
            success_diagnostics["outer_piece_count"] = int(
                candidate_metrics.get("outer_piece_count", 0)
            )
            if accepted_relaxed:
                if geometry_score < best_relaxed_score:
                    best_relaxed_score = geometry_score
                    best_relaxed = (
                        state,
                        canonical_result,
                        success_diagnostics,
                    )
                yield None
                continue

            plan = _build_unknown_success_plan(
                solver_pieces,
                canonical_result,
                work_region_mm,
                split_y_mm,
                geometry_score,
                search_nodes=diagnostics["four_work_units"],
                diagnostics=success_diagnostics,
            )
            if plan is not None:
                diagnostics.update(success_diagnostics)
                # 与GRAPH相同，规划必须在工作单元yield前进入共享进度，保证超时收尾
                # 能返回它，而不是把刚找到的四片解误报为solver_timeout。
                progress["current_stage"] = "four_fast"
                progress["result_source"] = "four_fast"
                progress["best_plan"] = plan
                # 最终栅格硬验收也作为一个工作单元让出；下一帧恢复后再返回规划，
                # 避免“找到答案”的那一帧额外跨过UI刷新预算。
                yield None
                return plan, diagnostics
            yield None

        if best_relaxed is not None:
            _state, canonical_result, success_diagnostics = best_relaxed
            plan = _build_unknown_success_plan(
                solver_pieces,
                canonical_result,
                work_region_mm,
                split_y_mm,
                best_relaxed_score,
                search_nodes=diagnostics["four_work_units"],
                diagnostics=success_diagnostics,
            )
            if plan is not None:
                diagnostics.update(success_diagnostics)
                progress["current_stage"] = "four_fast"
                progress["result_source"] = "four_fast"
                progress["best_plan"] = plan
                return plan, diagnostics
        return None, diagnostics

    return None, diagnostics


def _solve_unknown_four_fast_path(
    pieces,
    work_region_mm,
    split_y_mm,
    edge_length_tolerance_mm=2.5,
    pixels_per_mm=2.0,
    beam_width=UNKNOWN_FOUR_FAST_BEAM_WIDTH,
    max_work_units=UNKNOWN_FOUR_FAST_MAX_WORK_UNITS,
    incremental=False,
    progress=None,
):
    """创建FOUR_FAST增量核心；默认同步消费以保持既有调用兼容。

    incremental=True时返回可跨帧恢复的生成器；默认False时返回原有
    `(plan, diagnostics)`。beam_width和max_work_units仍分别约束召回宽度与总候选数。
    """
    steps = _solve_unknown_four_fast_path_steps(
        pieces,
        work_region_mm,
        split_y_mm,
        edge_length_tolerance_mm=edge_length_tolerance_mm,
        pixels_per_mm=pixels_per_mm,
        beam_width=beam_width,
        max_work_units=max_work_units,
        progress=progress,
    )
    if bool(incremental):
        return steps
    return _consume_solver_steps(steps)


def _coincident_edge(first_polygon, first_edge, second_polygon, second_edge, tolerance_mm=1.0):
    """判断两个目标边是否以反向端点在毫米容差内重合。"""
    first_start = first_polygon[first_edge]
    first_end = first_polygon[(first_edge + 1) % len(first_polygon)]
    second_start = second_polygon[second_edge]
    second_end = second_polygon[(second_edge + 1) % len(second_polygon)]
    return (
        np.linalg.norm(first_start - second_end) <= tolerance_mm
        and np.linalg.norm(first_end - second_start) <= tolerance_mm
    )


def _complete_layout_texture_score(canonical_by_index, solver_pieces):
    """汇总完整布局中所有反向重合内部边的牌面接缝分数。"""
    total_score = 0.0
    seam_count = 0
    indices = sorted(canonical_by_index)
    for first_position, first_index in enumerate(indices):
        first_polygon = canonical_by_index[first_index]
        for second_index in indices[first_position + 1 :]:
            second_polygon = canonical_by_index[second_index]
            for first_edge in range(len(first_polygon)):
                for second_edge in range(len(second_polygon)):
                    if not _coincident_edge(
                        first_polygon,
                        first_edge,
                        second_polygon,
                        second_edge,
                    ):
                        continue
                    total_score += edge_feature_match_score(
                        solver_pieces[first_index]["edge_features"][first_edge],
                        solver_pieces[second_index]["edge_features"][second_edge],
                    )
                    seam_count += 1
    return 0.0 if seam_count == 0 else total_score / seam_count


def _solve_unknown_layout_steps(
    pieces,
    work_region_mm,
    split_y_mm,
    edge_length_tolerance_mm=2.5,
    max_nodes=12000,
    pixels_per_mm=2.0,
    target_size_hint_mm=None,
    texture_refinement_nodes=400,
    search_width=UNKNOWN_SEARCH_BEAM_WIDTH,
    stop_at_first_solution=False,
    progress=None,
):
    """逐工作单元求解1～4片未知多边形，并通过生成器让出主循环。

    主要流程：固定第一片消除整体自由度，枚举未放置边与已放置边的刚体对齐，
    用毫米栅格剪除重叠；完整组合通过尺寸、填充率和重叠率验收，并以牌面接缝
    连续性打破几何同分。每完成一个较重候选检查就yield一次，使调用方可以跨帧
    恢复；生成器结束时通过StopIteration.value返回AssemblyPlan。

    关键参数：texture_refinement_nodes限制找到首个带纹理解后继续比较的节点数；
    search_width限制每层排序后保留的状态数；stop_at_first_solution用于WHITE显式模式，
    使反光纹理不再触发额外择优；progress用于公开节点、边图和前沿诊断。
    """
    pieces = list(pieces)
    if not 1 <= len(pieces) <= 4:
        return AssemblyPlan.failed("unknown_piece_count")
    try:
        max_nodes = int(max_nodes)
        pixels_per_mm = float(pixels_per_mm)
        length_tolerance = float(edge_length_tolerance_mm)
        texture_refinement_nodes = int(texture_refinement_nodes)
        search_width = int(search_width)
        if (
            max_nodes <= 0
            or pixels_per_mm <= 0.0
            or length_tolerance < 0.0
            or texture_refinement_nodes < 0
            or search_width <= 0
        ):
            raise ValueError
        # 三条搜索路径使用同一公共构造器，禁止在FALLBACK重新引入另一套原始伪短边。
        solver_pieces = [
            _solver_piece(piece, index)
            for index, piece in enumerate(pieces)
        ]
        # WHITE旧FALLBACK也会经历两片/三片中间状态，必须与FOUR_FAST共享最终总面积
        # 口径；CARD仍走原候选单片MASK门，不受该值影响。
        final_total_piece_area = sum(
            abs(
                float(
                    cv2.contourArea(
                        solver_piece["local_vertices"].astype(np.float32)
                    )
                )
            )
            for solver_piece in solver_pieces
        )
        edge_graph = _build_edge_compatibility_graph(
            solver_pieces,
            base_tolerance_mm=length_tolerance,
        )
    except (KeyError, TypeError, ValueError):
        return AssemblyPlan.failed("unknown_geometry_invalid")

    # progress由增量任务持有。同步入口不需要进度时仍使用局部字典，保持同一核心。
    progress = progress if isinstance(progress, dict) else {}
    progress["search_nodes"] = 0
    progress["first_solution_node"] = None
    progress["edge_candidates"] = int(
        sum(len(relations) for relations in edge_graph.values())
    )
    progress["max_frontier_width"] = 0
    progress["first_solution_node"] = None

    has_pattern = any(
        feature
        and float(feature.get("pattern_energy", 0.0)) >= PATTERN_ENERGY_THRESHOLD
        for solver_piece in solver_pieces
        for feature in solver_piece["edge_features"]
    )
    first_polygon = solver_pieces[0]["local_vertices"].copy()
    best_candidate = None
    best_score = float("inf")
    best_is_relaxed = False
    best_metrics = {}
    search_nodes = 0
    first_solution_node = None
    reached_limit = False
    stop_search = False
    rejection_counts = {
        "edge_mismatch": 0,
        "size_reject": 0,
        "fill_reject": 0,
        "overlap_reject": 0,
        "outer_edge_reject": 0,
        "relaxed_fill_reject": 0,
        "relaxed_candidates": 0,
        "relaxed_accept": 0,
        "complete_candidates": 0,
        "edge_candidates": int(progress["edge_candidates"]),
        "max_frontier_width": 0,
        "first_solution_node": 0,
    }
    # progress由UnknownSolveJob跨帧持有。保存同一个字典引用后，超时分支无需等待
    # 生成器正常return，也能读取搜索当时累计到的尺寸、填充、重叠和外边拒绝次数。
    progress["rejection_counts"] = rejection_counts

    def build_success_plan(candidate, score):
        """把规范化候选转换为下半区可执行规划，目标区放不下时返回None。

        主要流程：先在下半工作区计算目标矩形，再逐片换算目标轮廓、中心和最短旋转角。
        candidate包含按碎片索引保存的规范化轮廓及矩形宽高；score为当前综合评分。
        返回值为独立AssemblyPlan快照，使任务达到截止线时仍能安全返回已经找到的答案。
        """
        canonical_by_index, target_width, target_height = candidate
        # AssemblyPlan诊断只保存整数；比例统一乘1000，便于控制台和测试无浮点漂移地读取。
        rejection_counts["relaxed_accept"] = int(best_is_relaxed)
        rejection_counts["fill_permille"] = int(
            round(float(best_metrics.get("fill_ratio", 0.0)) * 1000.0)
        )
        rejection_counts["overlap_permille"] = int(
            round(float(best_metrics.get("overlap_ratio", 0.0)) * 1000.0)
        )
        rejection_counts["outer_piece_count"] = int(
            best_metrics.get("outer_piece_count", 0)
        )
        return _build_unknown_success_plan(
            solver_pieces,
            (canonical_by_index, target_width, target_height, score),
            work_region_mm,
            split_y_mm,
            score,
            search_nodes,
            rejection_counts,
        )

    def evaluate_complete(placed_by_index):
        """先严格、后WHITE容错验收完整组合，并保存同层内分数最低的候选。"""
        nonlocal best_candidate, best_score, best_is_relaxed, best_metrics
        nonlocal first_solution_node, stop_search
        rejection_counts["complete_candidates"] += 1
        strict_metrics = {}
        canonical_result, rejection_reason = _canonicalize_complete_layout(
            placed_by_index,
            pixels_per_mm=pixels_per_mm,
            min_fill_ratio=UNKNOWN_STRICT_MIN_FILL_RATIO,
            metrics=strict_metrics,
        )
        accepted_relaxed = False
        if canonical_result is None:
            rejection_counts[rejection_reason] += 1
            # 尺寸和重叠硬门在两层完全相同，只有严格填充不足才值得进入WHITE容错层。
            if not (stop_at_first_solution and rejection_reason == "fill_reject"):
                return
            relaxed_metrics = {}
            canonical_result, relaxed_reason = _canonicalize_complete_layout(
                placed_by_index,
                pixels_per_mm=pixels_per_mm,
                min_fill_ratio=UNKNOWN_RELAXED_MIN_FILL_RATIO,
                require_all_outer_edges=True,
                metrics=relaxed_metrics,
            )
            if canonical_result is None:
                if relaxed_reason == "fill_reject":
                    rejection_counts["relaxed_fill_reject"] += 1
                elif relaxed_reason == "outer_edge_reject":
                    rejection_counts["outer_edge_reject"] += 1
                return
            accepted_relaxed = True
            rejection_counts["relaxed_candidates"] += 1
            candidate_metrics = relaxed_metrics
        else:
            candidate_metrics = strict_metrics
        canonical_by_index, width_mm, height_mm, geometry_score = canonical_result
        texture_score = _complete_layout_texture_score(
            canonical_by_index,
            solver_pieces,
        )
        score = geometry_score + texture_score * 20.0
        candidate_is_better = bool(
            best_candidate is None
            or (best_is_relaxed and not accepted_relaxed)
            or (best_is_relaxed == accepted_relaxed and score < best_score)
        )
        if candidate_is_better:
            best_score = score
            best_is_relaxed = bool(accepted_relaxed)
            best_metrics = dict(candidate_metrics)
            best_candidate = (canonical_by_index, width_mm, height_mm)
            # 每个更优候选都立即生成独立规划快照。CARD后续可以继续比较花纹，但即使
            # 活动预算或硬墙钟随后到期，也不会丢掉已经验证通过的矩形布局。
            best_plan = build_success_plan(best_candidate, best_score)
            if best_plan is not None:
                progress["best_plan"] = best_plan
            if first_solution_node is None:
                # 首解节点是有限纹理优化窗口的起点，避免已有答案后仍跑满总上限。
                first_solution_node = search_nodes
                progress["first_solution_node"] = int(search_nodes)
                rejection_counts["first_solution_node"] = int(search_nodes)
            # WHITE由用户明确选择，只关心几何合法性；即使裸露铁片或反光让边缘
            # 纹理能量偏高，严格解可以立即结束；容错解继续固定64节点比较同层候选，
            # 避免降低填充门后返回遇到的第一个次优组合。
            if stop_at_first_solution and not accepted_relaxed:
                stop_search = True
                return
            # 白片没有纹理同分问题，首个近乎无缝矩形即可结束以保障实时性。
            if not has_pattern and geometry_score <= UNKNOWN_WHITE_EARLY_STOP_SCORE:
                stop_search = True
            elif has_pattern and texture_refinement_nodes == 0:
                stop_search = True

    def search(placed_by_index, remaining_indices):
        """按紧凑度递归枚举，并在每个高成本候选后yield一个工作单元。"""
        nonlocal search_nodes, reached_limit, stop_search
        if stop_search or reached_limit:
            return
        if not remaining_indices:
            evaluate_complete(placed_by_index)
            # 完整布局的栅格验收和纹理评分也计为一个可让出的高成本工作单元。
            yield None
            return

        expansions = []
        seen_poses = set()
        for source_index in tuple(remaining_indices):
            source_polygon = solver_pieces[source_index]["local_vertices"]
            for target_index, target_polygon in tuple(placed_by_index.items()):
                for relation in edge_graph.get((source_index, target_index), ()):
                    source_edge = relation["source_edge"]
                    target_edge = relation["target_edge"]
                    lengths_are_equal = relation["kind"] == "full"
                    candidates = _align_source_edge_candidates(
                        source_polygon,
                        source_edge,
                        target_polygon,
                        target_edge,
                    )
                    for candidate_polygon in candidates:
                        # WHITE已有最终3%硬验收，可使用三角化快速剪枝；CARD继续保留
                        # 原MASK语义，避免牌面纹理择优路径在本次优化中发生额外变化。
                        if stop_at_first_solution:
                            overlaps = _candidate_overlaps_fast(
                                candidate_polygon,
                                list(placed_by_index.values()),
                                pixels_per_mm=pixels_per_mm,
                                diagnostics=rejection_counts,
                                final_total_piece_area=final_total_piece_area,
                            )
                        else:
                            overlaps = _candidate_overlaps(
                                candidate_polygon,
                                list(placed_by_index.values()),
                                pixels_per_mm=pixels_per_mm,
                            )
                        if overlaps:
                            rejection_counts["overlap_reject"] += 1
                            yield None
                            continue
                        # 同一碎片可能通过多组共线边得到完全相同位姿，只保留一次。
                        pose_key = (
                            source_index,
                            tuple(np.rint(candidate_polygon.flatten() * 100.0).astype(int)),
                        )
                        if pose_key in seen_poses:
                            yield None
                            continue
                        seen_poses.add(pose_key)
                        updated = dict(placed_by_index)
                        updated[source_index] = candidate_polygon
                        partial_priority = _partial_layout_priority(
                            list(updated.values()),
                            tolerance_mm=length_tolerance,
                            target_size_hint_mm=target_size_hint_mm,
                        )
                        if partial_priority is None:
                            yield None
                            continue
                        # 排序只使用预计算边属性和O(顶点数)外框紧凑度，不再扫描所有接触边。
                        relation_priority = (
                            0 if lengths_are_equal else 1,
                            relation["texture_score"],
                            -relation["matched_length_mm"],
                            relation["length_error_mm"],
                        )
                        if target_size_hint_mm is None:
                            priority = (
                                partial_priority[1],
                                *relation_priority,
                                partial_priority[2],
                            )
                        else:
                            priority = (
                                partial_priority[0],
                                partial_priority[1],
                                *relation_priority,
                                partial_priority[2],
                            )
                        next_remaining = tuple(
                            index for index in remaining_indices if index != source_index
                        )
                        expansions.append((priority, updated, next_remaining))
                        # 重叠和外框计算完成后立即让出，继续保持相机主循环可响应。
                        yield None

        expansions.sort(key=lambda item: item[0])
        frontier_width = min(len(expansions), search_width)
        progress["max_frontier_width"] = max(
            int(progress["max_frontier_width"]),
            int(frontier_width),
        )
        rejection_counts["max_frontier_width"] = int(progress["max_frontier_width"])
        for _, updated, next_remaining in expansions[:search_width]:
            search_nodes += 1
            progress["search_nodes"] = int(search_nodes)
            if search_nodes > max_nodes:
                search_nodes = max_nodes
                progress["search_nodes"] = int(search_nodes)
                reached_limit = True
                return
            refinement_nodes = (
                UNKNOWN_RELAXED_REFINEMENT_NODES
                if stop_at_first_solution and best_is_relaxed
                else texture_refinement_nodes
            )
            if (
                first_solution_node is not None
                and search_nodes - first_solution_node >= refinement_nodes
            ):
                # WHITE容错首解后只再比较固定64节点；CARD仍使用原纹理优化窗口。
                # 这条停止条件既减少候选顺序影响，又防止已有答案后跑满总上限。
                stop_search = True
                return
            yield from search(updated, next_remaining)
            if stop_search or reached_limit:
                return

    yield from search({0: first_polygon}, tuple(range(1, len(solver_pieces))))
    if best_candidate is None:
        if reached_limit:
            reason = "search_limit"
        elif rejection_counts["complete_candidates"] > 0:
            # 单片无需接缝也会直接进入完整验收；只要存在完整候选，就应报告真实的
            # 尺寸、填充或重叠原因，不能因为search_nodes为0误报EDGE_MISMATCH。
            reason_priority = ("size_reject", "fill_reject", "overlap_reject")
            reason = max(
                reason_priority,
                key=lambda name: (rejection_counts[name], -reason_priority.index(name)),
            )
        elif search_nodes == 0:
            reason = "edge_mismatch"
            rejection_counts["edge_mismatch"] += 1
        else:
            # 有边候选却始终无法放完全部碎片，通常是接缝组合不闭合；若所有扩展
            # 都被实体重叠剪掉，则优先提示重叠，便于现场判断顶点误差。
            reason = (
                "overlap_reject"
                if rejection_counts["overlap_reject"] > 0
                else "edge_mismatch"
            )
            rejection_counts[reason] += 1
        return AssemblyPlan.failed(
            reason,
            search_nodes=search_nodes,
            diagnostics=rejection_counts,
        )

    best_plan = build_success_plan(best_candidate, best_score)
    if best_plan is None:
        return AssemblyPlan.failed("target_out_of_work_region", search_nodes=search_nodes)
    progress["best_plan"] = best_plan
    return best_plan


def _resume_solver_stage(stage):
    """兼容增量生成器与测试替身，并返回阶段终值。

    生产GRAPH/FOUR_FAST在incremental=True时返回生成器，本函数用yield from逐单元转发；
    单元测试可能把阶段替换成同步`(plan, diagnostics)`元组，此时直接返回该值。
    """
    if hasattr(stage, "__next__"):
        return (yield from stage)
    return stage


def _solve_unknown_white_pipeline_steps(
    pieces,
    work_region_mm,
    split_y_mm,
    edge_length_tolerance_mm=2.5,
    max_nodes=12000,
    pixels_per_mm=2.0,
    target_size_hint_mm=None,
    texture_refinement_nodes=400,
    search_width=UNKNOWN_SEARCH_BEAM_WIDTH,
    progress=None,
    stage_events=None,
):
    """按GRAPH、FOUR_FAST、FALLBACK顺序增量求解WHITE布局。

    三个阶段共享同一个外层UnknownSolveJob，因此从锁定快照开始就统一受单帧预算、
    8秒活动计算和30秒墙钟约束。GRAPH/FOUR_FAST完成后写入stage_events供运行器输出
    一次性日志；快路径无解才yield from旧FALLBACK。返回最终AssemblyPlan。
    """
    pieces = list(pieces)
    progress = progress if isinstance(progress, dict) else {}
    stage_events = stage_events if isinstance(stage_events, list) else []

    progress["current_stage"] = "graph"
    graph_stage = _solve_unknown_graph_fast_path(
        pieces,
        work_region_mm,
        split_y_mm,
        pixels_per_mm=pixels_per_mm,
        incremental=True,
        progress=progress,
    )
    graph_plan, graph_diagnostics = yield from _resume_solver_stage(graph_stage)
    stage_events.append(
        {
            "source": "graph",
            "diagnostics": dict(graph_diagnostics),
            "plan": graph_plan,
        }
    )
    if graph_plan is not None:
        progress["result_source"] = "graph"
        # 单个OpenCV硬验收无法在执行中抢占；若它返回合法解时刚好越过总预算，
        # UnknownSolveJob的统一超时收尾必须能够返回这份已验证规划，而不是丢成超时。
        progress["best_plan"] = graph_plan
        return graph_plan
    # 阶段事件必须形成独立的可让出边界。运行器可先输出纯GRAPH累计耗时，下一帧
    # 再启动FOUR_FAST，避免一条GRAPH日志混入后续阶段的计算时间。
    yield None

    if UNKNOWN_FOUR_FAST_ENABLED and len(pieces) == 4:
        progress["current_stage"] = "four_fast"
        four_stage = _solve_unknown_four_fast_path(
            pieces,
            work_region_mm,
            split_y_mm,
            edge_length_tolerance_mm=edge_length_tolerance_mm,
            pixels_per_mm=pixels_per_mm,
            incremental=True,
            progress=progress,
        )
        four_plan, four_diagnostics = yield from _resume_solver_stage(four_stage)
        # 子生成器返回时，外层advance已经累计了此前所有FOURFAST工作单元；在复制
        # 阶段事件前补齐该值，使成功、工作量失败和活动时间失败使用同一日志字段。
        four_diagnostics["four_active_elapsed_ms"] = int(
            round(
                max(
                    0.0,
                    float(progress.get("four_fast_active_elapsed_seconds", 0.0)),
                )
                * 1000.0
            )
        )
        stage_events.append(
            {
                "source": "four_fast",
                "diagnostics": dict(four_diagnostics),
                "plan": four_plan,
            }
        )
        if four_plan is not None:
            progress["result_source"] = "four_fast"
            progress["best_plan"] = four_plan
            return four_plan
        # FOUR_FAST失败统计同样先交给运行器；旧FALLBACK从下一帧开始使用剩余预算。
        yield None

    progress["current_stage"] = "fallback"
    progress["result_source"] = "fallback"
    return (
        yield from _solve_unknown_layout_steps(
            pieces,
            work_region_mm,
            split_y_mm,
            edge_length_tolerance_mm=edge_length_tolerance_mm,
            max_nodes=max_nodes,
            pixels_per_mm=pixels_per_mm,
            target_size_hint_mm=target_size_hint_mm,
            texture_refinement_nodes=texture_refinement_nodes,
            search_width=search_width,
            stop_at_first_solution=True,
            progress=progress,
        )
    )


class UnknownSolveJob:
    """把未知拼装生成器封装为可在相机多帧之间恢复的增量任务。"""

    def __init__(
        self,
        pieces,
        work_region_mm,
        split_y_mm,
        edge_length_tolerance_mm=2.5,
        max_nodes=12000,
        pixels_per_mm=2.0,
        target_size_hint_mm=None,
        texture_refinement_nodes=400,
        search_width=UNKNOWN_SEARCH_BEAM_WIDTH,
        stop_at_first_solution=False,
        active_timeout_seconds=UNKNOWN_SOLVER_ACTIVE_TIMEOUT_SECONDS,
        wall_timeout_seconds=UNKNOWN_SOLVER_WALL_TIMEOUT_SECONDS,
        timeout_seconds=None,
        clock=None,
        four_fast_active_budget_seconds=UNKNOWN_FOUR_FAST_ACTIVE_BUDGET_SECONDS,
    ):
        """创建尚未执行的求解任务并复制调用参数。

        主要流程：记录任务创建时刻并构造共享增量生成器；clock默认使用单调时钟，
        测试可注入替代时钟。active_timeout_seconds只累计生成器工作单元耗时，
        wall_timeout_seconds限制任务总存活时间；four_fast_active_budget_seconds只限制
        四片WHITE快路径的CPU活动时间，耗尽后继续FALLBACK；兼容参数timeout_seconds
        非None时仅覆盖硬墙钟。stop_at_first_solution由显式WHITE模式启用；输入碎片由
        求解核心转成独立列表，主循环后续帧不会改变本任务的集合引用。
        返回值：构造函数无返回值；结果通过advance()、done和result读取。
        """
        self._progress = {
            "search_nodes": 0,
            "first_solution_node": None,
            "best_plan": None,
            "current_stage": "fallback",
            "result_source": "fallback",
            # 这两个字段由外层任务计时、FOURFAST生成器读取。三片和CARD不会进入
            # four_fast阶段，因此不会累计或触发该中止请求。
            "four_fast_active_elapsed_seconds": 0.0,
            "four_fast_abort_requested": False,
        }
        self._stage_events = []
        self._clock = clock if clock is not None else time.monotonic
        try:
            self._active_timeout_seconds = float(active_timeout_seconds)
            selected_wall_timeout = (
                wall_timeout_seconds if timeout_seconds is None else timeout_seconds
            )
            self._wall_timeout_seconds = float(selected_wall_timeout)
            self._created_at = float(self._clock())
            if (
                self._active_timeout_seconds <= 0.0
                or not math.isfinite(self._active_timeout_seconds)
                or self._wall_timeout_seconds <= 0.0
                or not math.isfinite(self._wall_timeout_seconds)
                or not math.isfinite(self._created_at)
            ):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("求解活动预算和硬墙钟必须是有限正数")
        try:
            self._four_fast_active_budget_seconds = float(
                four_fast_active_budget_seconds
            )
            if (
                self._four_fast_active_budget_seconds <= 0.0
                or not math.isfinite(self._four_fast_active_budget_seconds)
            ):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("FOURFAST活动预算必须是有限正数")
        self._active_elapsed_seconds = 0.0
        # WHITE把两个快路径与旧FALLBACK放入同一生成器，三阶段从任务创建起共享预算；
        # CARD仍直接使用原增量核心，保持纹理择优和MASK重叠语义不变。
        pieces = list(pieces)
        if bool(stop_at_first_solution):
            self._progress["current_stage"] = "graph"
            self._progress["result_source"] = "graph"
            self._iterator = _solve_unknown_white_pipeline_steps(
                pieces,
                work_region_mm,
                split_y_mm,
                edge_length_tolerance_mm=edge_length_tolerance_mm,
                max_nodes=max_nodes,
                pixels_per_mm=pixels_per_mm,
                target_size_hint_mm=target_size_hint_mm,
                texture_refinement_nodes=texture_refinement_nodes,
                search_width=search_width,
                progress=self._progress,
                stage_events=self._stage_events,
            )
        else:
            self._iterator = _solve_unknown_layout_steps(
                pieces,
                work_region_mm,
                split_y_mm,
                edge_length_tolerance_mm=edge_length_tolerance_mm,
                max_nodes=max_nodes,
                pixels_per_mm=pixels_per_mm,
                target_size_hint_mm=target_size_hint_mm,
                texture_refinement_nodes=texture_refinement_nodes,
                search_width=search_width,
                stop_at_first_solution=False,
                progress=self._progress,
            )
        self.done = False
        self.result = None

    @property
    def search_nodes(self):
        """返回任务目前已经深入的搜索节点数，供状态栏显示进度。"""
        return int(self._progress.get("search_nodes", 0))

    @property
    def first_solution_node(self):
        """返回首个合法矩形出现的节点；尚无合法解时返回None。"""
        value = self._progress.get("first_solution_node")
        return None if value is None else int(value)

    @property
    def edge_candidates(self):
        """返回预计算边图中的有限有向候选数；生成器尚未首次推进时返回0。"""
        return int(self._progress.get("edge_candidates", 0))

    @property
    def max_frontier_width(self):
        """返回搜索至今实际保留过的最大单层状态数，供性能诊断。"""
        return int(self._progress.get("max_frontier_width", 0))

    @property
    def active_elapsed_ms(self):
        """返回生成器工作单元累计占用的整数毫秒数，不包含相机帧间等待。"""
        return max(0, int(round(self._active_elapsed_seconds * 1000.0)))

    @property
    def result_source(self):
        """返回当前/最终求解阶段名，供运行器区分GRAPH、FOUR_FAST和FALLBACK日志。"""
        return str(
            self._progress.get(
                "current_stage",
                self._progress.get("result_source", "fallback"),
            )
        )

    def consume_stage_events(self):
        """取出尚未输出的快路径完成事件，并清空内部队列。

        每个事件只返回一次，包含source、diagnostics和可选plan；运行器在每次advance后
        调用，避免跨帧重复打印GRAPH/FOUR_FAST统计。返回不可变元组。
        """
        events = tuple(self._stage_events)
        self._stage_events.clear()
        return events

    def _deadline_reached(self, current_time):
        """判断活动计算预算或总墙钟是否到期。

        current_time必须来自任务注入的同一单调时钟。返回True表示任务必须在当前工作
        单元边界结束；活动预算只由advance包围next()的计时更新，不受帧间等待影响。
        """
        wall_elapsed = float(current_time) - self._created_at
        return (
            self._active_elapsed_seconds >= self._active_timeout_seconds
            or wall_elapsed >= self._wall_timeout_seconds
        )

    def _record_work_unit_elapsed(self, unit_started_at, unit_finished_at):
        """累计一个生成器工作单元的CPU活动时间，并更新四片快路径中止信号。

        关键参数必须来自任务注入的同一单调时钟；返回本工作单元非负耗时秒数。
        `current_stage`在`next()`返回后读取，因此从FOURFAST切入FALLBACK的首个工作单元
        会正确计入FALLBACK，不会反向消耗四片子预算。达到子预算只设置共享标志，
        FOURFAST在下一个安全yield边界正常返回，整个UNKNOWN任务仍保留剩余统一预算。
        """
        elapsed_seconds = max(
            0.0,
            float(unit_finished_at) - float(unit_started_at),
        )
        self._active_elapsed_seconds += elapsed_seconds
        if str(self._progress.get("current_stage", "fallback")) == "four_fast":
            four_elapsed = (
                float(self._progress.get("four_fast_active_elapsed_seconds", 0.0))
                + elapsed_seconds
            )
            self._progress["four_fast_active_elapsed_seconds"] = four_elapsed
            if four_elapsed >= self._four_fast_active_budget_seconds:
                self._progress["four_fast_abort_requested"] = True
        return elapsed_seconds

    def _finish_timeout(self, current_time):
        """关闭生成器，并优先返回搜索期间已经验证通过的最优规划。

        关键参数current_time必须来自同一单调时钟；诊断保留活动耗时、总墙钟、边候选、
        最大前沿和首解节点。若progress中已有best_plan，则克隆为成功结果并标记截止返回；
        否则返回不含机械目标的solver_timeout。重复advance会直接返回同一结果。
        """
        if self.done:
            return self.result
        self._iterator.close()
        wall_elapsed_ms = max(
            0,
            int(round((float(current_time) - self._created_at) * 1000.0)),
        )
        timeout_diagnostics = dict(self._progress.get("rejection_counts", {}))
        timeout_diagnostics.update({
            "edge_candidates": self.edge_candidates,
            "max_frontier_width": self.max_frontier_width,
            "first_solution_node": self.first_solution_node or 0,
            "active_elapsed_ms": self.active_elapsed_ms,
            "wall_elapsed_ms": wall_elapsed_ms,
            # elapsed_ms是v1.4.0诊断字段，继续映射到总墙钟以兼容旧测试和现场日志。
            "elapsed_ms": wall_elapsed_ms,
        })
        best_plan = self._progress.get("best_plan")
        if isinstance(best_plan, AssemblyPlan) and best_plan.success:
            diagnostics = dict(best_plan.diagnostics)
            diagnostics.update(timeout_diagnostics)
            diagnostics["returned_best_at_timeout"] = 1
            self.result = AssemblyPlan(
                True,
                placements=best_plan.placements,
                target_rect_mm=best_plan.target_rect_mm,
                score=best_plan.score,
                reason=best_plan.reason,
                search_nodes=self.search_nodes,
                diagnostics=diagnostics,
            )
        else:
            self.result = AssemblyPlan.failed(
                "solver_timeout",
                search_nodes=self.search_nodes,
                diagnostics=timeout_diagnostics,
            )
        self.done = True
        return self.result

    def advance(self, time_budget_ms=12.0, work_unit_limit=32):
        """在当前帧预算内推进任务，完成时返回规划，否则返回None。

        关键参数：time_budget_ms限制单帧墙钟，work_unit_limit限制单帧让出点数量；
        任务同时受累计活动计算预算和总墙钟硬截止线约束。每次next()前后读取同一时钟，
        仅把该工作单元耗时加入活动预算；单个慢OpenCV调用返回或抛错后也会立即截止。
        重复调用返回同一结果。
        """
        if self.done:
            return self.result
        try:
            budget_seconds = float(time_budget_ms) / 1000.0
            unit_limit = int(work_unit_limit)
            if budget_seconds <= 0.0 or unit_limit <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("求解时间预算和工作单元上限必须大于零")

        started_at = float(self._clock())
        if self._deadline_reached(started_at):
            return self._finish_timeout(started_at)
        processed_units = 0
        while processed_units < unit_limit:
            if processed_units > 0:
                current_time = float(self._clock())
                if self._deadline_reached(current_time):
                    return self._finish_timeout(current_time)
                if current_time - started_at >= budget_seconds:
                    break
            unit_started_at = float(self._clock())
            # 记录本工作单元前的事件数；若生成器在该单元完成一个快路径阶段，当前
            # advance必须立即结束，不能在同一帧继续消耗下一阶段的预算。
            stage_event_count = len(self._stage_events)
            try:
                next(self._iterator)
            except StopIteration as completed:
                completed_at = float(self._clock())
                self._record_work_unit_elapsed(unit_started_at, completed_at)
                if self._deadline_reached(completed_at):
                    return self._finish_timeout(completed_at)
                self.result = completed.value
                self.done = True
                return self.result
            except Exception:
                # 底层工作单元若在截止线之后才抛错，现场真正发生的是超时；截止线前
                # 的异常继续上抛，由AssemblyRuntime转换为solver_error并释放任务。
                failed_at = float(self._clock())
                self._record_work_unit_elapsed(unit_started_at, failed_at)
                if self._deadline_reached(failed_at):
                    return self._finish_timeout(failed_at)
                raise
            unit_finished_at = float(self._clock())
            self._record_work_unit_elapsed(unit_started_at, unit_finished_at)
            processed_units += 1
            if self._deadline_reached(unit_finished_at):
                return self._finish_timeout(unit_finished_at)
            if len(self._stage_events) > stage_event_count:
                break

        # 循环可能因为达到work_unit_limit而结束，此时最后一个工作单元已经执行过，
        # 却不会再进入下一轮的“单元前检查”。必须在返回控制权给主循环之前再读取一次
        # 同一单调时钟，避免唯一/最后一个慢工作单元越过总截止线后仍延迟到下一帧报错。
        slice_finished_at = float(self._clock())
        if self._deadline_reached(slice_finished_at):
            return self._finish_timeout(slice_finished_at)
        return None

    def cancel(self):
        """关闭尚未完成的生成器并返回不含机械目标的取消结果。

        模式切换、进入CAL或运行器重置时调用。已经完成的任务保持原结果；未完成任务
        立即关闭递归生成器，释放其候选列表和栅格引用，后续advance返回同一取消结果。
        """
        if self.done:
            return self.result
        self._iterator.close()
        self.result = AssemblyPlan.failed("cancelled", search_nodes=self.search_nodes)
        self.done = True
        return self.result

    def run_to_completion(self):
        """连续消费增量核心直到结束，供兼容的同步API和PC测试复用。"""
        while not self.done:
            self.advance(time_budget_ms=60000.0, work_unit_limit=1000000)
        return self.result


def solve_unknown_layout(
    pieces,
    work_region_mm,
    split_y_mm,
    edge_length_tolerance_mm=2.5,
    max_nodes=12000,
    pixels_per_mm=2.0,
    target_size_hint_mm=None,
    texture_refinement_nodes=400,
    search_width=UNKNOWN_SEARCH_BEAM_WIDTH,
    stop_at_first_solution=False,
    active_timeout_seconds=UNKNOWN_SOLVER_ACTIVE_TIMEOUT_SECONDS,
    wall_timeout_seconds=UNKNOWN_SOLVER_WALL_TIMEOUT_SECONDS,
    timeout_seconds=None,
    four_fast_active_budget_seconds=UNKNOWN_FOUR_FAST_ACTIVE_BUDGET_SECONDS,
):
    """同步求解未知布局，内部完整消费与主循环相同的增量搜索核心。

    该接口用于既有测试和非实时调用；stop_at_first_solution=True代表WHITE时，任务
    内部依次执行GRAPH、四片FOUR_FAST和原FALLBACK。同步入口完整消费统一生成器，
    相机运行器则按帧推进同一个任务，两个入口不会再出现不同的快路径调度语义。
    """
    job = UnknownSolveJob(
        pieces,
        work_region_mm,
        split_y_mm,
        edge_length_tolerance_mm=edge_length_tolerance_mm,
        max_nodes=max_nodes,
        pixels_per_mm=pixels_per_mm,
        target_size_hint_mm=target_size_hint_mm,
        texture_refinement_nodes=texture_refinement_nodes,
        search_width=search_width,
        stop_at_first_solution=stop_at_first_solution,
        active_timeout_seconds=active_timeout_seconds,
        wall_timeout_seconds=wall_timeout_seconds,
        timeout_seconds=timeout_seconds,
        four_fast_active_budget_seconds=four_fast_active_budget_seconds,
    )
    return job.run_to_completion()


def _prepare_known_registration(pieces, expected_size_mm):
    """筛选四个上半区碎片并生成不会污染视觉结果的临时求解输入。

    主要流程：复用运行器的可操作碎片定义，校验目标尺寸，为四片浅复制字典分配
    R1～R4临时编号。返回值为``(solver_inputs, expected_size, failure_plan)``；成功时
    failure_plan为None，失败时前两项为None并给出具体KNOWN错误。
    """
    source_pieces = [
        piece
        for piece in pieces
        if piece.get("complete") is True and piece.get("region") == "upper"
    ]
    if len(source_pieces) != 4:
        return None, None, AssemblyPlan.failed("known_needs_four")

    try:
        expected_width, expected_height = (float(value) for value in expected_size_mm)
        if (
            expected_width <= 0.0
            or expected_height <= 0.0
            or not np.all(np.isfinite((expected_width, expected_height)))
        ):
            raise ValueError
        solver_inputs = []
        for index, piece in enumerate(source_pieces, start=1):
            # 当前帧可能没有UNKNOWN编号；浅复制后改临时ID可保留轮廓和边缘特征。
            solver_piece = dict(piece)
            solver_piece["id"] = f"R{index}"
            solver_inputs.append(solver_piece)
    except (TypeError, ValueError):
        return None, None, AssemblyPlan.failed("known_geometry_invalid")
    return solver_inputs, (expected_width, expected_height), None


def _finish_known_registration(
    solver_inputs,
    unknown_plan,
    expected_size_mm,
    work_region_mm,
    split_y_mm,
    max_match_score,
):
    """把未知增量求解结果转换成精确KNOWN模板和可立即绘制的规划。

    关键参数：unknown_plan必须成功并携带目标框；expected_size_mm通常为100×60mm。
    主要流程：按测量目标归一缩放、稳定排序生成K1～K4，再用模板快速规划验证。
    返回值为``(known_plan, templates)``；任何几何或匹配异常返回空模板失败规划。
    """
    if not unknown_plan.success or unknown_plan.target_rect_mm is None:
        return unknown_plan, []

    expected_width, expected_height = expected_size_mm
    try:
        target_x, target_y, measured_width, measured_height = unknown_plan.target_rect_mm
        if measured_width <= 0.0 or measured_height <= 0.0:
            raise ValueError("求解目标尺寸无效")
        placement_by_id = {
            placement.piece_id: placement for placement in unknown_plan.placements
        }
        target_origin = np.asarray((target_x, target_y), dtype=np.float64)
        target_scale = np.asarray(
            (expected_width / measured_width, expected_height / measured_height),
            dtype=np.float64,
        )
        candidates = []
        for solver_piece in solver_inputs:
            placement = placement_by_id[solver_piece["id"]]
            descriptor = build_shape_descriptor(solver_piece)
            normalized_target = (
                np.asarray(placement.target_polygon_mm, dtype=np.float64) - target_origin
            ) * target_scale
            _normalize_vertices(normalized_target)
            candidates.append((descriptor, normalized_target))
        candidates.sort(key=lambda item: _descriptor_sort_key(item[0]))
    except (KeyError, TypeError, ValueError):
        return AssemblyPlan.failed("known_registration_failed"), []

    templates = []
    for index, (descriptor, target_vertices) in enumerate(candidates, start=1):
        templates.append(
            {
                "id": f"K{index}",
                **descriptor,
                "target_vertices_mm": target_vertices.astype(float).tolist(),
                "layout_size_mm": [expected_width, expected_height],
            }
        )

    # 重新规划确保返回目标已经是精确100×60mm且编号为K1～K4，可直接写入运行缓存。
    known_plan = solve_known_layout(
        solver_inputs,
        templates,
        work_region_mm,
        split_y_mm,
        max_match_score=max_match_score,
    )
    if not known_plan.success:
        return AssemblyPlan.failed("known_registration_failed"), []
    return known_plan, templates


class KnownRegistrationJob:
    """跨帧执行KNOWN布局登记，并在完成后提供规划与持久化模板。"""

    def __init__(
        self,
        pieces,
        work_region_mm,
        split_y_mm,
        expected_size_mm=KNOWN_TARGET_SIZE_MM,
        edge_length_tolerance_mm=2.5,
        max_nodes=12000,
        pixels_per_mm=2.0,
        max_match_score=1.2,
        texture_refinement_nodes=400,
        clock=None,
    ):
        """准备KNOWN登记输入并创建内部未知增量任务。

        四片数量或目标尺寸不合法时会立即形成失败结果；合法输入直到advance()才执行
        搜索。模板仅在未知计划和KNOWN快速复核都成功后写入templates属性。
        """
        solver_inputs, expected_size, failure_plan = _prepare_known_registration(
            pieces,
            expected_size_mm,
        )
        self._solver_inputs = solver_inputs
        self._expected_size = expected_size
        self._work_region_mm = work_region_mm
        self._split_y_mm = split_y_mm
        self._max_match_score = float(max_match_score)
        self._unknown_job = None
        self.templates = []
        self.result = failure_plan
        self.done = failure_plan is not None
        if not self.done:
            self._unknown_job = UnknownSolveJob(
                solver_inputs,
                work_region_mm,
                split_y_mm,
                edge_length_tolerance_mm=edge_length_tolerance_mm,
                max_nodes=max_nodes,
                pixels_per_mm=pixels_per_mm,
                target_size_hint_mm=expected_size,
                texture_refinement_nodes=texture_refinement_nodes,
                clock=clock,
            )

    @property
    def search_nodes(self):
        """返回内部未知任务已经深入的节点数，供SAVE状态栏显示。"""
        if self._unknown_job is None:
            return 0 if self.result is None else int(self.result.search_nodes)
        return self._unknown_job.search_nodes

    def advance(self, time_budget_ms=12.0, work_unit_limit=32):
        """推进一个KNOWN SAVE时间片；未完成返回None，完成返回最终KNOWN规划。"""
        if self.done:
            return self.result
        unknown_plan = self._unknown_job.advance(
            time_budget_ms=time_budget_ms,
            work_unit_limit=work_unit_limit,
        )
        if unknown_plan is None:
            return None
        self.result, self.templates = _finish_known_registration(
            self._solver_inputs,
            unknown_plan,
            self._expected_size,
            self._work_region_mm,
            self._split_y_mm,
            self._max_match_score,
        )
        self.done = True
        return self.result

    def cancel(self):
        """取消内部未知搜索并清除尚未持久化的KNOWN模板。"""
        if self.done:
            return self.result
        if self._unknown_job is not None:
            self._unknown_job.cancel()
        self.templates = []
        self.result = AssemblyPlan.failed("cancelled", search_nodes=self.search_nodes)
        self.done = True
        return self.result

    def run_to_completion(self):
        """连续推进直到登记完成，供既有同步SAVE接口保持兼容。"""
        while not self.done:
            self.advance(time_budget_ms=60000.0, work_unit_limit=1000000)
        return self.result, self.templates


def solve_and_register_known_layout(
    pieces,
    work_region_mm,
    split_y_mm,
    expected_size_mm=KNOWN_TARGET_SIZE_MM,
    edge_length_tolerance_mm=2.5,
    max_nodes=12000,
    pixels_per_mm=2.0,
    max_match_score=1.2,
    texture_refinement_nodes=400,
):
    """从下半区已经拼好的四片布局同步生成模板和即时规划。

    主要流程：只接受四片完整lower碎片；复用未知求解器的旋正、填充率和重叠率
    验收函数验证联合矩形；将验收后的轮廓缩放到精确100×60mm，按稳定描述子编号
    K1～K4；最后运行固定规模KNOWN匹配，直接生成下半区目标。该路径不创建
    KnownRegistrationJob或UnknownSolveJob。返回``(AssemblyPlan, templates)``。

    edge_length_tolerance_mm、max_nodes和texture_refinement_nodes仅为旧调用签名兼容，
    直接登记不使用搜索预算；pixels_per_mm仍用于小画布几何验收。
    """
    del edge_length_tolerance_mm, max_nodes, texture_refinement_nodes
    source_pieces = [
        piece for piece in pieces if piece.get("complete") is True
    ]
    if len(source_pieces) != 4:
        return AssemblyPlan.failed("known_needs_four"), []
    if any(piece.get("region") != "lower" for piece in source_pieces):
        return AssemblyPlan.failed("known_layout_must_be_lower"), []

    try:
        expected_width, expected_height = (
            float(value) for value in expected_size_mm
        )
        if (
            expected_width <= 0.0
            or expected_height <= 0.0
            or not np.all(np.isfinite((expected_width, expected_height)))
        ):
            raise ValueError("KNOWN目标尺寸无效")
        placed_by_index = {
            index: _normalize_vertices(piece["vertices_mm"])
            for index, piece in enumerate(source_pieces)
        }
        canonical_result, rejection_reason = _canonicalize_complete_layout(
            placed_by_index,
            pixels_per_mm=pixels_per_mm,
        )
    except (KeyError, TypeError, ValueError):
        return AssemblyPlan.failed("known_layout_invalid"), []

    if canonical_result is None:
        return AssemblyPlan.failed(
            "known_layout_invalid",
            diagnostics={str(rejection_reason): 1},
        ), []
    canonical_by_index, measured_width, measured_height, geometry_score = (
        canonical_result
    )
    tolerance = max(0.0, float(KNOWN_LAYOUT_SIZE_TOLERANCE_MM))
    if (
        abs(measured_width - expected_width) > tolerance
        or abs(measured_height - expected_height) > tolerance
    ):
        return AssemblyPlan.failed(
            "known_layout_invalid",
            diagnostics={"size_reject": 1},
        ), []

    try:
        target_scale = np.asarray(
            (
                expected_width / measured_width,
                expected_height / measured_height,
            ),
            dtype=np.float64,
        )
        candidates = []
        for index, piece in enumerate(source_pieces):
            descriptor = build_shape_descriptor(piece)
            target_vertices = canonical_by_index[index] * target_scale
            _normalize_vertices(target_vertices)
            candidates.append((descriptor, target_vertices))
        candidates.sort(key=lambda item: _descriptor_sort_key(item[0]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return AssemblyPlan.failed("known_layout_invalid"), []

    templates = []
    for index, (descriptor, target_vertices) in enumerate(candidates, start=1):
        templates.append(
            {
                "id": f"K{index}",
                **descriptor,
                "target_vertices_mm": target_vertices.astype(float).tolist(),
                "layout_size_mm": [expected_width, expected_height],
            }
        )

    known_plan = solve_known_layout(
        source_pieces,
        templates,
        work_region_mm,
        split_y_mm,
        max_match_score=max_match_score,
    )
    if not known_plan.success:
        if known_plan.reason == "target_out_of_work_region":
            return known_plan, []
        return AssemblyPlan.failed(
            "known_registration_failed",
            diagnostics={"geometry_score_milli": int(round(geometry_score * 1000.0))},
        ), []
    return known_plan, templates


def _template_context_fingerprint(templates):
    """生成不抛异常的模板指纹，损坏布局使用稳定哨兵交给规划阶段拒绝。

    指纹只负责判断上下文是否变化，不能在相机主循环前置阶段验证持久模板。无效字段
    会编码为invalid类型键；随后solve_known_layout返回template_layout_invalid。
    """
    fingerprint = []
    for template in templates:
        if not isinstance(template, dict):
            fingerprint.append(("<invalid>", ("invalid_template", type(template).__name__)))
            continue
        target_vertices = template.get("target_vertices_mm")
        if target_vertices is None:
            layout_key = ("none",)
        else:
            try:
                normalized_vertices = tuple(
                    tuple(round(float(value), 3) for value in point)
                    for point in target_vertices
                )
                layout_key = ("valid", normalized_vertices)
            except (TypeError, ValueError):
                layout_key = ("invalid_layout", type(target_vertices).__name__)
        fingerprint.append((str(template.get("id", "")), layout_key))
    return tuple(sorted(fingerprint))


def _piece_observation(piece):
    """提取稳定帧比较所需的顶点数、毫米中心和有序毫米顶点。"""
    vertices = _normalize_vertices(piece["vertices_mm"])
    center = tuple(float(value) for value in piece.get("center_mm", _polygon_centroid(vertices)))
    return {
        "vertex_count": len(vertices),
        "center": np.asarray(center, dtype=np.float64),
        "vertices": vertices,
    }


def _polygon_frame_distance(first_vertices, second_vertices):
    """把两帧轮廓重采样后枚举循环起点，计算最小RMS毫米位移。

    稳定门保留绝对纸面位置和角度，不执行旋转或尺度配准；它只消除3～5顶点近似
    数量变化。轮廓遍历方向不反转，避免翻面形状被当成同一稳定观测。
    """
    first_resampled = resample_closed_contour(
        first_vertices,
        sample_count=CONTOUR_SAMPLE_COUNT,
    )
    second_resampled = resample_closed_contour(
        second_vertices,
        sample_count=CONTOUR_SAMPLE_COUNT,
    )
    best_distance = float("inf")
    for shift in range(CONTOUR_SAMPLE_COUNT):
        aligned = np.roll(second_resampled, shift, axis=0)
        distance = float(
            np.sqrt(np.mean(np.sum((first_resampled - aligned) ** 2, axis=1)))
        )
        best_distance = min(best_distance, distance)
    return best_distance


def _observations_are_stable(reference, current, tolerance_mm):
    """对不超过四片的两帧观测穷举一对一对应，判断全部顶点位移是否在容差内。"""
    if len(reference) != len(current):
        return False
    for assignment in itertools.permutations(range(len(current))):
        assignment_valid = True
        for reference_index, current_index in enumerate(assignment):
            first = reference[reference_index]
            second = current[current_index]
            if np.linalg.norm(first["center"] - second["center"]) > tolerance_mm:
                assignment_valid = False
                break
            if _polygon_frame_distance(first["vertices"], second["vertices"]) > tolerance_mm:
                assignment_valid = False
                break
        if assignment_valid:
            return True
    return False


class AssemblyRuntime:
    """维护连续稳定帧门、一次求解和跨帧缓存。

    规划缓存用于机械开始移动后的画面：碎片位置变化不会触发重算；识别模式、模板、
    黄色机械区域或分界线变化会自动建立新上下文并清除旧目标。
    """

    def __init__(
        self,
        stable_frames=3,
        position_tolerance_mm=3.0,
        solver_time_budget_ms=24.0,
        solver_work_unit_limit=64,
        texture_refinement_nodes=400,
        debug_enabled=None,
    ):
        """初始化稳定门与UNKNOWN单帧求解预算。

        关键参数：solver_time_budget_ms和solver_work_unit_limit共同限制每帧计算量；
        texture_refinement_nodes限制带纹理首解后的附加择优节点；debug_enabled为None时
        读取文件顶部UNKNOWN_SOLVER_DEBUG，测试也可显式传True/False覆盖。所有参数必须有效。
        返回值：构造函数无返回值，运行状态通过update和公开属性读取。
        """
        self.stable_frames = int(stable_frames)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.solver_time_budget_ms = float(solver_time_budget_ms)
        self.solver_work_unit_limit = int(solver_work_unit_limit)
        self.texture_refinement_nodes = int(texture_refinement_nodes)
        self.debug_enabled = bool(
            UNKNOWN_SOLVER_DEBUG if debug_enabled is None else debug_enabled
        )
        if (
            self.stable_frames <= 0
            or self.position_tolerance_mm <= 0.0
            or self.solver_time_budget_ms <= 0.0
            or self.solver_work_unit_limit <= 0
            or self.texture_refinement_nodes < 0
        ):
            raise ValueError("稳定门、单帧预算和纹理优化节点参数无效")
        self.solve_count = 0
        self._context_key = None
        self._reference_observation = None
        # 达到稳定门时只深复制一次可操作碎片。后续相机帧可以继续刷新界面和触摸，
        # 但求解、成功结果和失败诊断都必须绑定这份快照，避免轮廓抖动改变机械目标。
        self._locked_pieces = None
        self._solve_job = None
        self._solve_started_at = None
        self.stable_count = 0
        self.plan = None

    @property
    def snapshot_locked(self):
        """返回当前上下文是否已经锁定一次稳定碎片快照。

        该状态从稳定计数达到门槛开始保持为True，求解成功、确定失败或超时均不自动
        清除；只有reset、模式/参数上下文变化才恢复为False。
        """
        return self._locked_pieces is not None

    @property
    def locked_pieces(self):
        """返回供正常界面只读绘制的锁定碎片元组。

        元组中的字典和NumPy轮廓已经在锁定时深复制，不会引用相机当前帧。调用方只能
        读取，不应原地修改；尚未达到稳定门时返回空元组，便于界面回退到实时轮廓。
        """
        return () if self._locked_pieces is None else self._locked_pieces

    @property
    def is_solving(self):
        """返回当前是否存在尚未完成的UNKNOWN增量求解任务。"""
        return self._solve_job is not None and not self._solve_job.done

    @property
    def search_nodes(self):
        """返回当前增量任务或已缓存规划的节点数，供状态栏显示。"""
        if self._solve_job is not None:
            return self._solve_job.search_nodes
        if self.plan is not None:
            return int(self.plan.search_nodes)
        return 0

    @property
    def edge_candidates(self):
        """返回当前UNKNOWN任务预计算的有向边候选数；没有活动任务时返回0。"""
        if self._solve_job is None:
            return 0
        return int(getattr(self._solve_job, "edge_candidates", 0))

    @property
    def max_frontier_width(self):
        """返回当前UNKNOWN任务搜索至今的最大前沿宽度；没有活动任务时返回0。"""
        if self._solve_job is None:
            return 0
        return int(getattr(self._solve_job, "max_frontier_width", 0))

    @property
    def first_solution_node(self):
        """返回当前UNKNOWN任务发现首解时的节点；尚无首解或无活动任务时返回None。"""
        if self._solve_job is None:
            return None
        value = getattr(self._solve_job, "first_solution_node", None)
        return None if value is None else int(value)

    def _debug_log(self, event, detail_factory):
        """按开关惰性输出一条求解日志，关闭时不构造任何明细字符串。

        关键参数：event为固定事件名；detail_factory必须是无参数可调用对象，仅在调试
        开启时才执行并返回明细文本。返回值始终为None，日志只写标准输出供电脑查看。
        """
        if not self.debug_enabled:
            return
        details = str(detail_factory()).strip()
        suffix = f" {details}" if details else ""
        print(f"[SOLVER] {str(event).upper()}{suffix}")

    def _debug_snapshot(self, mode, unknown_profile, pieces):
        """输出锁定快照摘要以及每片的毫米顶点、中心和循环边长。

        该函数只在稳定门刚满足时调用一次。全部NumPy换算位于_debug_log回调中，因此
        文件顶部开关为False时不会给MaixCAM2增加轮廓格式化和边长计算开销。
        """
        self._debug_log(
            "SNAPSHOT",
            lambda: (
                f"mode={str(mode).upper()} profile={str(unknown_profile).upper()} "
                f"count={len(pieces)} strict_fill={UNKNOWN_STRICT_MIN_FILL_RATIO:.2f} "
                f"relaxed_fill={UNKNOWN_RELAXED_MIN_FILL_RATIO:.2f}"
            ),
        )
        for fallback_index, piece in enumerate(pieces, start=1):
            def build_piece_details(piece=piece, fallback_index=fallback_index):
                """惰性计算当前单片的现场诊断文本。"""
                vertices = np.asarray(piece["vertices_mm"], dtype=np.float64).reshape(-1, 2)
                closed = np.vstack((vertices, vertices[:1]))
                edge_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
                center = np.asarray(piece["center_mm"], dtype=np.float64).reshape(2)
                vertex_text = ",".join(
                    f"({point[0]:.1f},{point[1]:.1f})" for point in vertices
                )
                edge_text = ",".join(f"{length:.1f}" for length in edge_lengths)
                return (
                    f"id={piece.get('id', f'U{fallback_index}')} "
                    f"center_mm=({center[0]:.1f},{center[1]:.1f}) "
                    f"vertices_mm=[{vertex_text}] edges_mm=[{edge_text}]"
                )

            self._debug_log("PIECE", build_piece_details)
            if (
                str(mode).strip().lower() == "unknown"
                and str(unknown_profile).strip().lower() == UNKNOWN_PROFILE_WHITE
            ):
                def build_cleanup_details(piece=piece, fallback_index=fallback_index):
                    """仅在调试开启时预览WHITE求解副本的短边清理结果。"""
                    _cleaned, cleanup = _clean_solver_short_edges(
                        piece["vertices_mm"],
                        min_edge_mm=UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM,
                    )
                    return (
                        f"id={piece.get('id', f'U{fallback_index}')} "
                        f"vertices={cleanup['original_vertex_count']}->"
                        f"{cleanup['cleaned_vertex_count']} "
                        f"removed={cleanup['removed_count']} "
                        f"min_edge={cleanup['original_min_edge_mm']:.1f}mm "
                        f"cleaned_min={cleanup['cleaned_min_edge_mm']:.1f}mm"
                    )

                # _debug_log会在开关关闭时直接返回，因此不会执行上面的清理预览。
                self._debug_log("CLEAN", build_cleanup_details)

    def _debug_graph(self, diagnostics, plan, elapsed_ms):
        """输出一次GRAPH有界枚举统计和容错拒绝原因。"""
        self._debug_log(
            "GRAPH",
            lambda: (
                f"result={'OK' if plan is not None and plan.success else 'FAIL'} "
                f"edges={int(diagnostics.get('graph_edge_candidates', 0))} "
                f"sets={int(diagnostics.get('graph_matching_sets', 0))} "
                f"layouts={int(diagnostics.get('graph_layouts_checked', 0))} "
                f"strict={int(diagnostics.get('graph_strict_accept', 0))} "
                f"relaxed={int(diagnostics.get('graph_relaxed_candidates', 0))} "
                f"fill_reject={int(diagnostics.get('graph_relaxed_fill_reject', 0))} "
                f"outer_reject={int(diagnostics.get('graph_outer_edge_reject', 0))} "
                f"elapsed_ms={int(elapsed_ms)}"
            ),
        )

    def _debug_four_fast(self, diagnostics, plan, elapsed_ms):
        """输出四片分层Beam的状态规模、剪枝数量和终态。

        关键参数：diagnostics来自`_solve_unknown_four_fast_path`，plan可为None；日志只在
        总调试开关开启时由`_debug_log`惰性格式化，不参与后续FALLBACK决策。
        """
        self._debug_log(
            "FOUR_FAST",
            lambda: (
                f"result={'OK' if plan is not None and plan.success else 'FAIL'} "
                f"pairs={int(diagnostics.get('four_pair_states', 0))} "
                f"triples={int(diagnostics.get('four_triple_states', 0))} "
                f"complete={int(diagnostics.get('four_complete_states', 0))} "
                f"parents={int(diagnostics.get('four_pair_parents_expanded', 0))}/"
                f"{int(diagnostics.get('four_triple_parents_expanded', 0))}/"
                f"{int(diagnostics.get('four_complete_parents_expanded', 0))} "
                f"checked={int(diagnostics.get('four_complete_checked', 0))} "
                f"dedupe={int(diagnostics.get('four_deduplicated', 0))} "
                f"overlap={int(diagnostics.get('four_overlap_reject', 0))} "
                f"units={int(diagnostics.get('four_work_units', 0))} "
                f"limit={int(diagnostics.get('four_work_limit_reached', 0))} "
                f"time_limit={int(diagnostics.get('four_active_limit_reached', 0))} "
                f"active_ms={int(diagnostics.get('four_active_elapsed_ms', 0))} "
                f"segmented={int(diagnostics.get('four_used_segmented', 0))} "
                f"triangles={int(diagnostics.get('fast_triangle_checks', 0))} "
                f"raster={int(diagnostics.get('fast_raster_fallbacks', 0))} "
                f"elapsed_ms={int(elapsed_ms)}"
            ),
        )

    def _debug_fast_stage_events(self, job):
        """输出当前时间片刚结束的GRAPH/FOUR_FAST阶段事件。

        job.consume_stage_events保证每个阶段只输出一次；elapsed_ms使用任务累计活动计算
        时间，不包含拍照、显示和帧间等待。FOUR_FAST值包含此前GRAPH耗时，便于直接
        对照从锁定到该阶段结束的CPU预算消耗。
        """
        consume_events = getattr(job, "consume_stage_events", None)
        if not callable(consume_events):
            return
        elapsed_ms = int(getattr(job, "active_elapsed_ms", 0))
        for event in consume_events():
            source = str(event.get("source", "")).lower()
            diagnostics = event.get("diagnostics", {})
            plan = event.get("plan")
            if source == "graph":
                self._debug_graph(diagnostics, plan, elapsed_ms)
            elif source == "four_fast":
                self._debug_four_fast(diagnostics, plan, elapsed_ms)

    def _debug_fallback(self, job, result):
        """输出递归兜底终态、搜索规模和验收拒绝统计。"""
        diagnostics = result.diagnostics if isinstance(result, AssemblyPlan) else {}
        self._debug_log(
            "FALLBACK",
            lambda: (
                f"reason={getattr(result, 'reason', 'unknown')} "
                f"nodes={int(getattr(job, 'search_nodes', 0))} "
                f"edges={int(getattr(job, 'edge_candidates', 0))} "
                f"frontier={int(getattr(job, 'max_frontier_width', 0))} "
                f"first={int(getattr(job, 'first_solution_node', 0) or 0)} "
                f"active_ms={int(getattr(job, 'active_elapsed_ms', 0))} "
                f"size_reject={int(diagnostics.get('size_reject', 0))} "
                f"fill_reject={int(diagnostics.get('fill_reject', 0))} "
                f"overlap_reject={int(diagnostics.get('overlap_reject', 0))} "
                f"outer_reject={int(diagnostics.get('outer_edge_reject', 0))}"
            ),
        )

    def _debug_result(self, plan, source):
        """输出一次最终成功/失败、采用路径、填充/重叠和总耗时。"""
        elapsed_ms = (
            0
            if self._solve_started_at is None
            else max(0, int(round((time.monotonic() - self._solve_started_at) * 1000.0)))
        )
        diagnostics = plan.diagnostics if isinstance(plan, AssemblyPlan) else {}
        self._debug_log(
            "RESULT",
            lambda: (
                f"source={str(source).upper()} success={int(bool(plan.success))} "
                f"reason={plan.reason} nodes={int(plan.search_nodes)} "
                f"fill={int(diagnostics.get('fill_permille', 0)) / 10.0:.1f}% "
                f"overlap={int(diagnostics.get('overlap_permille', 0)) / 10.0:.1f}% "
                f"outer={int(diagnostics.get('outer_piece_count', 0))} "
                f"elapsed_ms={elapsed_ms}"
            ),
        )

    def reset(self):
        """清除上下文、锁定快照、未完成任务和规划缓存，但保留求解计数。"""
        if self._solve_job is not None:
            self._solve_job.cancel()
        self._context_key = None
        self._reference_observation = None
        self._locked_pieces = None
        self._solve_job = None
        self._solve_started_at = None
        self.stable_count = 0
        self.plan = None

    def _reset_tracking(self):
        """上下文变化时清除帧跟踪、锁定快照、任务和目标，再从当前帧累计。"""
        if self._solve_job is not None:
            self._solve_job.cancel()
        self._reference_observation = None
        self._locked_pieces = None
        self._solve_job = None
        self._solve_started_at = None
        self.stable_count = 0
        self.plan = None

    def _build_context_key(
        self,
        mode,
        templates,
        work_region_mm,
        split_y_mm,
        unknown_profile=UNKNOWN_PROFILE_WHITE,
    ):
        """校验规划上下文并生成决定缓存是否仍可复用的稳定键。

        关键参数包括模式、UNKNOWN子模式、模板目标轮廓、黄色机械区域和红色分界线；
        任一项变化都必须使旧机械目标失效。返回值同时包含规范后的模式、子模式、
        区域、分界线和键。
        """
        normalized_mode = str(mode).lower()
        if normalized_mode not in ("known", "unknown"):
            raise ValueError("规划模式必须是known或unknown")
        normalized_unknown_profile = str(unknown_profile).lower()
        if normalized_unknown_profile not in (
            UNKNOWN_PROFILE_WHITE,
            UNKNOWN_PROFILE_CARD,
        ):
            raise ValueError("UNKNOWN子模式必须是white或card")
        work_region = validate_work_region_mm(work_region_mm)
        split_y = validate_split_y_mm(work_region, split_y_mm)
        template_key = (
            _template_context_fingerprint(templates)
            if normalized_mode == "known"
            else ()
        )
        context_key = (
            normalized_mode,
            (
                normalized_unknown_profile
                if normalized_mode == "unknown"
                else ""
            ),
            tuple(round(float(value), 3) for value in work_region),
            round(split_y, 3),
            template_key,
        )
        return (
            normalized_mode,
            normalized_unknown_profile,
            work_region,
            split_y,
            context_key,
        )

    def cache_plan(
        self,
        mode,
        plan,
        templates,
        work_region_mm,
        split_y_mm,
        pieces=None,
    ):
        """预装一次已经验收成功的规划，供SAVE后立即显示并跨帧复用。

        主要流程：使用与update完全相同的上下文键绑定规划，把稳定计数置为已满足；
        只接受成功AssemblyPlan。可选pieces用于设备直接SAVE路径，把本次识别的四片一并
        深复制锁定，使下一帧跳过重复视觉分割；旧离线调用不提供时保持兼容空快照。
        返回值：输入plan本身，便于调用方直接用于当前帧绘制。
        """
        if not isinstance(plan, AssemblyPlan) or not plan.success:
            raise ValueError("只能缓存成功的AssemblyPlan")
        _, _, _, _, context_key = self._build_context_key(
            mode,
            templates,
            work_region_mm,
            split_y_mm,
        )
        if self._solve_job is not None:
            self._solve_job.cancel()
        self._context_key = context_key
        self._reference_observation = None
        if pieces is None:
            self._locked_pieces = None
        else:
            pieces = list(pieces)
            if not pieces:
                raise ValueError("预装规划的锁定碎片不能为空")
            self._locked_pieces = tuple(copy.deepcopy(pieces))
        self._solve_job = None
        self.stable_count = self.stable_frames
        self.plan = plan
        return plan

    def _advance_unknown_job(self):
        """推进当前UNKNOWN任务，并把不可恢复异常转换成安全失败规划。

        主要流程：按构造时预算调用advance；异常时读取已有节点数形成solver_error，
        显式取消任务释放生成器。成功、确定性几何失败和无首解超时都写入plan，并继续
        保留任务启动时的锁定快照；程序不会自动换一组抖动轮廓重新搜索，用户必须通过
        模式按钮或CAL显式reset后才能重新采集。返回值：未完成为None，结束帧为plan。
        """
        job = self._solve_job
        try:
            result = job.advance(
                time_budget_ms=self.solver_time_budget_ms,
                work_unit_limit=self.solver_work_unit_limit,
            )
        except Exception:
            search_nodes = int(getattr(job, "search_nodes", 0))
            try:
                job.cancel()
            except Exception:
                # 异常清理不能覆盖原始求解失败；丢弃引用后由Python回收对象。
                pass
            result = AssemblyPlan.failed("solver_error", search_nodes=search_nodes)
        self._debug_fast_stage_events(job)
        if result is None:
            return None
        self._solve_job = None
        # 包括solver_timeout在内的终态都绑定同一份快照。自动丢弃超时会造成“重新识别
        # 三帧—再次搜索”的无限循环，也会让每轮顶点略有变化而难以复现实机问题。
        self.plan = result
        result_source = str(getattr(job, "result_source", "fallback")).lower()
        if result_source == "fallback":
            self._debug_fallback(job, result)
        self._debug_result(result, result_source)
        return self.plan

    def update(
        self,
        mode,
        pieces,
        templates,
        work_region_mm,
        split_y_mm,
        known_match_threshold=1.6,
        unknown_max_nodes=12000,
        unknown_profile=UNKNOWN_PROFILE_WHITE,
    ):
        """接收一帧识别数据，在稳定门满足后求解一次并返回缓存。

        关键参数：mode为known或unknown；unknown_profile为white或card；只使用complete
        且region=upper的1～4片。UNKNOWN达到稳定门后只创建并推进一个短时间片；WHITE
        首解即停，CARD继续有限纹理择优。任务执行期间返回None，完成后返回成功或失败
        AssemblyPlan。KNOWN模板匹配规模固定且很小，继续同步完成。
        """
        mode, unknown_profile, work_region, split_y, context_key = self._build_context_key(
            mode,
            templates,
            work_region_mm,
            split_y_mm,
            unknown_profile=unknown_profile,
        )
        if context_key != self._context_key:
            self._context_key = context_key
            self._reset_tracking()
        if self.plan is not None:
            return self.plan
        if self._solve_job is not None:
            # 已启动任务使用第3稳定帧的几何快照；后续帧只推进任务，不因识别抖动重启。
            return self._advance_unknown_job()

        actionable = [
            piece
            for piece in pieces
            if piece.get("complete") is True and piece.get("region") == "upper"
        ]
        if not 1 <= len(actionable) <= 4:
            self._reference_observation = None
            self.stable_count = 0
            return None
        try:
            current_observation = [_piece_observation(piece) for piece in actionable]
        except (KeyError, TypeError, ValueError):
            self._reference_observation = None
            self.stable_count = 0
            return None

        if self._reference_observation is None:
            self.stable_count = 1
        elif _observations_are_stable(
            self._reference_observation,
            current_observation,
            self.position_tolerance_mm,
        ):
            self.stable_count += 1
        else:
            self.stable_count = 1
        self._reference_observation = current_observation
        if self.stable_count < self.stable_frames:
            return None

        # 第3稳定帧是本次识别唯一的几何输入。必须在调用同步GRAPH或创建增量生成器前
        # 深复制，因为视觉层下一帧会创建或修改新的ID、轮廓和边缘特征字典。
        self._locked_pieces = tuple(copy.deepcopy(actionable))
        solve_pieces = self._locked_pieces
        self._solve_started_at = time.monotonic()
        self._debug_snapshot(mode, unknown_profile, solve_pieces)
        self.solve_count += 1
        if mode == "known":
            self.plan = solve_known_layout(
                solve_pieces,
                templates,
                work_region,
                split_y,
                max_match_score=known_match_threshold,
            )
            self._debug_result(self.plan, "known")
        else:
            # WHITE和CARD都只创建一个UnknownSolveJob。WHITE任务内部按
            # GRAPH→FOUR_FAST→FALLBACK推进，三个阶段从锁定帧起共享单帧与总超时预算；
            # CARD任务直接进入保留纹理语义的FALLBACK。
            self._solve_job = UnknownSolveJob(
                solve_pieces,
                work_region,
                split_y,
                max_nodes=unknown_max_nodes,
                texture_refinement_nodes=self.texture_refinement_nodes,
                stop_at_first_solution=(unknown_profile == UNKNOWN_PROFILE_WHITE),
            )
            # UNKNOWN结束结果是否进入长期缓存由_advance_unknown_job统一决定；这里若把
            # 它的单帧超时返回值再次赋给self.plan，会破坏“无首解自动重试”语义。
            return self._advance_unknown_job()
        return self.plan


COLOR_PLAN_SPLIT = (0, 0, 255)
COLOR_PLAN_TARGET = (255, 0, 255)
COLOR_PLAN_ARROW = (0, 165, 255)
COLOR_PLAN_PIECES = (
    (0, 255, 0),
    (255, 180, 0),
    (0, 220, 220),
    (220, 100, 255),
)


def _rect_corners(rect_mm):
    """把X/Y/W/H目标矩形转换为顺时针四个毫米角点。"""
    rect_x, rect_y, rect_width, rect_height = (float(value) for value in rect_mm)
    return np.asarray(
        (
            (rect_x, rect_y),
            (rect_x + rect_width, rect_y),
            (rect_x + rect_width, rect_y + rect_height),
            (rect_x, rect_y + rect_height),
        ),
        dtype=np.float32,
    )


def draw_assembly_plan(
    frame_bgr,
    plan,
    paper_quad,
    work_region_mm,
    split_y_mm,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """在相机画面副本上绘制毫米分界线、目标矩形、单片轮廓、箭头和位姿文字。

    失败或尚未求解时仍绘制红色分界线；只有成功plan才输出机械目标。所有毫米点
    通过完整A4单应性映射，输入frame_bgr保持不变。返回值：同尺寸BGR新图像。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效numpy图像")
    output = frame_bgr.copy()
    split_segment = build_split_segment(
        paper_quad,
        work_region_mm,
        split_y_mm,
        paper_orientation=paper_orientation,
    )
    cv2.line(
        output,
        tuple(np.rint(split_segment[0]).astype(np.int32)),
        tuple(np.rint(split_segment[1]).astype(np.int32)),
        COLOR_PLAN_SPLIT,
        2,
    )
    if plan is None or not plan.success or plan.target_rect_mm is None:
        return output

    target_rect_px = paper_points_to_image_px(
        _rect_corners(plan.target_rect_mm),
        paper_quad,
        paper_orientation,
    )
    for index, placement in enumerate(plan.placements):
        color = COLOR_PLAN_PIECES[index % len(COLOR_PLAN_PIECES)]
        target_polygon_px = paper_points_to_image_px(
            placement.target_polygon_mm,
            paper_quad,
            paper_orientation,
        )
        source_center_px = paper_points_to_image_px(
            [placement.source_center_mm],
            paper_quad,
            paper_orientation,
        )[0]
        target_center_px = paper_points_to_image_px(
            [placement.target_center_mm],
            paper_quad,
            paper_orientation,
        )[0]
        target_polygon_int = np.rint(target_polygon_px).astype(np.int32)
        source_center_int = tuple(np.rint(source_center_px).astype(np.int32))
        target_center_int = tuple(np.rint(target_center_px).astype(np.int32))
        cv2.polylines(output, [target_polygon_int], True, color, 2)
        cv2.arrowedLine(
            output,
            source_center_int,
            target_center_int,
            COLOR_PLAN_ARROW,
            2,
            tipLength=0.12,
        )
        label = (
            f"{placement.piece_id} "
            f"({placement.target_center_mm[0]:.1f},{placement.target_center_mm[1]:.1f}) "
            f"R{placement.rotation_delta_deg:+.1f}"
        )
        cv2.putText(
            output,
            label,
            (max(0, target_center_int[0] + 4), max(14, target_center_int[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    # 总目标外框最后绘制，避免单片轮廓与外框重合时把矩形边界完全覆盖。
    cv2.polylines(
        output,
        [np.rint(target_rect_px).astype(np.int32)],
        True,
        COLOR_PLAN_TARGET,
        2,
    )
    return output
