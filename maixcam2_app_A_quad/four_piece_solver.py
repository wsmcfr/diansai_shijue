"""UNKNOWN FOUR模式的轮廓假设、接缝关系和四片专用图搜索。"""

import math
import time

import cv2
import numpy as np

try:
    from maixcam2_app_A_quad.assembly_planner import (
        AssemblyPlacement,
        AssemblyPlan,
        _solve_unknown_four_fast_path,
    )
    from maixcam2_app_A_quad.four_piece_vision import FourPieceVisionRuntime
except ModuleNotFoundError as error:
    # MaixVision平铺部署时顶层包不存在，四片求解器仍复用同级兼容结果结构。
    if error.name != "maixcam2_app_A_quad":
        raise
    from assembly_planner import (
        AssemblyPlacement,
        AssemblyPlan,
        _solve_unknown_four_fast_path,
    )
    from four_piece_vision import FourPieceVisionRuntime


# ======================== FOUR求解调试常量（不影响1～3片） ========================
# 小于该值的边按远距离反光伪角处理；题目真实结构边应不短于约20mm。
FOUR_MIN_EDGE_MM = 10.0
# 两条边长度差不超过该值时按完整接缝评分，超过后才考虑长短分段关系。
FOUR_EDGE_LENGTH_TOLERANCE_MM = 3.0
# 每个有向片对进入图搜索的最佳关系上限，控制四片组合分支数量。
FOUR_PAIR_RELATION_LIMIT = 6
# 长短边比例低于该值时视为分段接缝，高于该值的轻微差异按轮廓误差处理。
FOUR_SEGMENTED_LENGTH_RATIO = 0.75
# 过短的分段覆盖不具备可靠机械接缝意义，直接拒绝。
FOUR_MIN_PARTIAL_RATIO = 0.35
# 四片分层搜索每层最多保留的全局状态数；只影响FOUR，不改变旧UNKNOWN beam。
FOUR_BEAM_WIDTH = 32
# 完整矩形长边和短边的题目范围，单位毫米。
FOUR_RECT_LONG_RANGE_MM = (90.0, 120.0)
FOUR_RECT_SHORT_RANGE_MM = (50.0, 90.0)
# 视觉轮廓存在毫米级角点误差；该容差只放宽测量验收，不缩放最终机械目标。
FOUR_RECT_SIZE_TOLERANCE_MM = 2.0
# 严格轮和宽松轮最低填充率；宽松轮仍保持相同尺寸和重叠硬门。
FOUR_STRICT_MIN_FILL_RATIO = 0.92
FOUR_RELAXED_MIN_FILL_RATIO = 0.85
# 最终总重叠硬门和中间单片加入时的稍宽松剪枝门。
FOUR_MAX_OVERLAP_RATIO = 0.03
FOUR_INTERMEDIATE_MAX_OVERLAP_RATIO = 0.06
# 专用任务累计CPU活动预算；拍照、显示和帧间等待不计入。
FOUR_ACTIVE_BUDGET_SECONDS = 3.0
# 中间几何重叠使用较低像素密度，仅消除共享边栅格误差，不参与最终填充计算。
FOUR_OVERLAP_PIXELS_PER_MM = 2.0
# True时向电脑控制台输出一次锁定快照和最终结果；关闭后不构造明细字符串。
FOUR_SOLVER_DEBUG = True
# ===============================================================================


class EdgeSegment:
    """保存一个形状假设中的有向直边及其毫米长度。"""

    def __init__(self, index, start, end):
        """初始化边序号和二维端点；退化边立即抛ValueError。"""
        self.index = int(index)
        self.start = _normalize_point(start, "边起点")
        self.end = _normalize_point(end, "边终点")
        self.vector = self.end - self.start
        self.length_mm = float(np.linalg.norm(self.vector))
        if self.length_mm <= 1e-6:
            raise ValueError("形状假设不能包含零长度边")
        self.unit = self.vector / self.length_mm


class ShapeHypothesis:
    """保存一片碎片的清理后多边形、边线、面积和轮廓拟合分数。"""

    def __init__(self, piece_id, vertices, score, source_index=0):
        """规范顶点为逆时针并建立循环边列表，构造函数无返回值。"""
        normalized = _normalize_polygon(vertices)
        self.piece_id = str(piece_id)
        self.vertices = normalized
        self.score = float(score)
        self.source_index = int(source_index)
        self.area_mm2 = abs(float(cv2.contourArea(normalized.astype(np.float32))))
        self.edges = tuple(
            EdgeSegment(
                index,
                normalized[index],
                normalized[(index + 1) % len(normalized)],
            )
            for index in range(len(normalized))
        )


class PairRelation:
    """保存把moving碎片放到fixed碎片坐标系中的一个刚体接缝候选。"""

    def __init__(
        self,
        fixed_piece_id,
        moving_piece_id,
        fixed_hypothesis,
        moving_hypothesis,
        fixed_edge_index,
        moving_edge_index,
        rotation,
        translation,
        score,
        overlap_length_mm,
        segmented,
    ):
        """初始化接缝身份、2×2旋转、二维平移和排序指标。

        rotation行列式必须为正且接近1，确保关系只含旋转和平移、不允许镜像。
        """
        normalized_rotation = np.asarray(rotation, dtype=np.float64)
        normalized_translation = _normalize_point(translation, "接缝平移")
        if normalized_rotation.shape != (2, 2) or not np.all(
            np.isfinite(normalized_rotation)
        ):
            raise ValueError("接缝旋转必须是有限2×2矩阵")
        determinant = float(np.linalg.det(normalized_rotation))
        if determinant <= 0.0 or abs(determinant - 1.0) > 1e-5:
            raise ValueError("接缝关系只能包含不带镜像的平面旋转")
        self.fixed_piece_id = str(fixed_piece_id)
        self.moving_piece_id = str(moving_piece_id)
        self.fixed_hypothesis = fixed_hypothesis
        self.moving_hypothesis = moving_hypothesis
        self.fixed_edge_index = int(fixed_edge_index)
        self.moving_edge_index = int(moving_edge_index)
        self.rotation = normalized_rotation.copy()
        self.translation = normalized_translation
        self.score = float(score)
        self.overlap_length_mm = float(overlap_length_mm)
        self.segmented = bool(segmented)

    def transform_points(self, points):
        """把N×2 moving局部点应用本关系后返回fixed坐标系点。"""
        normalized = _normalize_points(points, "待变换点")
        return normalized @ self.rotation.T + self.translation


def _normalize_point(point, field_name):
    """校验一个有限二维点并返回float64副本。"""
    normalized = np.asarray(point, dtype=np.float64)
    if normalized.shape != (2,) or not np.all(np.isfinite(normalized)):
        raise ValueError(f"{field_name}必须包含两个有限数字")
    return normalized.copy()


def _normalize_points(points, field_name):
    """校验非空N×2有限点集并返回float64副本。"""
    normalized = np.asarray(points, dtype=np.float64)
    if (
        normalized.ndim != 2
        or normalized.shape[1] != 2
        or len(normalized) < 1
        or not np.all(np.isfinite(normalized))
    ):
        raise ValueError(f"{field_name}必须是非空N×2有限数组")
    return normalized.copy()


def _normalize_polygon(vertices):
    """校验至少三点的非退化多边形，并统一为逆时针顶点顺序。"""
    normalized = _normalize_points(vertices, "多边形顶点")
    if len(normalized) < 3:
        raise ValueError("多边形至少需要三个顶点")
    signed_area = float(cv2.contourArea(normalized.astype(np.float32), oriented=True))
    if abs(signed_area) <= 1e-6:
        raise ValueError("多边形面积过小或已经退化")
    if signed_area < 0.0:
        normalized = normalized[::-1].copy()
    return normalized


def _remove_one_short_edge(vertices, reference_area):
    """删除一条最短伪边的最佳端点，并尽量保持原多边形面积。

    返回``(新顶点, 是否删除)``。对最短边的两个端点分别试删，选择面积变化较小且
    仍为有效多边形的方案；这样能保留真实转角并去掉同一直线上的反光伪点。
    """
    if len(vertices) <= 3:
        return vertices, False
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    edge_index = int(np.argmin(lengths))
    if float(lengths[edge_index]) >= FOUR_MIN_EDGE_MM:
        return vertices, False
    candidate_removals = (edge_index, (edge_index + 1) % len(vertices))
    candidates = []
    for remove_index in candidate_removals:
        candidate = np.delete(vertices, remove_index, axis=0)
        try:
            candidate = _normalize_polygon(candidate)
        except ValueError:
            continue
        area = abs(float(cv2.contourArea(candidate.astype(np.float32))))
        area_error = abs(area - reference_area) / max(1.0, reference_area)
        candidates.append((area_error, remove_index, candidate))
    if not candidates:
        return vertices, False
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2], True


def _clean_short_edges(vertices, reference_area):
    """反复清理小于FOUR_MIN_EDGE_MM的伪边，最低保留三角形。"""
    cleaned = _normalize_polygon(vertices)
    while len(cleaned) > 3:
        cleaned, removed = _remove_one_short_edge(cleaned, reference_area)
        if not removed:
            break
    return cleaned


def _hypothesis_key(vertices):
    """生成与循环起点无关的毫米顶点键，用于删除多尺度重复假设。"""
    rounded = [tuple(np.round(point, 2)) for point in vertices]
    rotations = [tuple(rounded[index:] + rounded[:index]) for index in range(len(rounded))]
    return min(rotations)


def build_shape_hypotheses(piece, max_hypotheses=3):
    """从单片稠密轮廓生成少量清理后多边形假设并按拟合误差排序。

    主要流程：优先读取raw_contour_mm，不存在时使用vertices_mm；按多个绝对毫米误差
    执行approxPolyDP；清理伪短边；按面积差和matchShapes评分并去重。关键参数piece
    必须提供id及至少一种毫米轮廓；返回ShapeHypothesis列表。
    """
    if not isinstance(piece, dict):
        raise ValueError("piece必须是碎片字典")
    piece_id = str(piece.get("id", ""))
    if not piece_id:
        raise ValueError("piece必须提供非空id")
    source = piece.get("raw_contour_mm", piece.get("vertices_mm"))
    raw_vertices = _normalize_polygon(source)
    raw_contour = raw_vertices.astype(np.float32).reshape(-1, 1, 2)
    reference_area = abs(float(cv2.contourArea(raw_contour)))
    perimeter = float(cv2.arcLength(raw_contour, True))
    if perimeter <= 1e-6:
        raise ValueError("碎片轮廓周长无效")
    max_hypotheses = int(max_hypotheses)
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses必须至少为1")

    candidates = []
    seen = set()
    for source_index, epsilon_mm in enumerate((0.35, 0.60, 0.90, 1.30, 1.90, 2.60)):
        approximated = cv2.approxPolyDP(raw_contour, float(epsilon_mm), True)
        if len(approximated) < 3:
            continue
        try:
            cleaned = _clean_short_edges(
                approximated.reshape(-1, 2).astype(np.float64),
                reference_area,
            )
        except ValueError:
            continue
        key = _hypothesis_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        candidate_contour = cleaned.astype(np.float32).reshape(-1, 1, 2)
        candidate_area = abs(float(cv2.contourArea(candidate_contour)))
        area_error = abs(candidate_area - reference_area) / max(1.0, reference_area)
        shape_error = float(
            cv2.matchShapes(raw_contour, candidate_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        )
        score = area_error * 2.0 + shape_error + len(cleaned) * 1e-4
        candidates.append(
            ShapeHypothesis(
                piece_id,
                cleaned,
                score,
                source_index=source_index,
            )
        )
    candidates.sort(key=lambda item: (item.score, len(item.vertices), item.source_index))
    return candidates[:max_hypotheses]


def _rotation_between(source_vector, target_vector):
    """返回把source单位方向旋转到target单位方向的正交2×2矩阵。"""
    source_angle = math.atan2(float(source_vector[1]), float(source_vector[0]))
    target_angle = math.atan2(float(target_vector[1]), float(target_vector[0]))
    angle = target_angle - source_angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _relation_translations(fixed_edge, moving_edge, rotation):
    """为完整或分段接缝生成首端、末端和居中三种平移锚点。"""
    rotated_start = moving_edge.start @ rotation.T
    rotated_end = moving_edge.end @ rotation.T
    fixed_midpoint = (fixed_edge.start + fixed_edge.end) * 0.5
    moving_midpoint = (rotated_start + rotated_end) * 0.5
    return (
        fixed_edge.start - rotated_end,
        fixed_edge.end - rotated_start,
        fixed_midpoint - moving_midpoint,
    )


def _relation_key(rotation, translation):
    """把刚体变换量化成稳定键，删除等长边产生的重复锚点。"""
    angle = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    return (
        round(angle, 3),
        round(float(translation[0]), 3),
        round(float(translation[1]), 3),
    )


def _build_hypothesis_pair_relations(fixed_hypothesis, moving_hypothesis):
    """枚举两个形状假设的反向边对齐关系，不在此阶段判断完整矩形。"""
    candidates = []
    seen = set()
    for fixed_edge in fixed_hypothesis.edges:
        for moving_edge in moving_hypothesis.edges:
            shorter = min(fixed_edge.length_mm, moving_edge.length_mm)
            longer = max(fixed_edge.length_mm, moving_edge.length_mm)
            length_ratio = shorter / longer
            if length_ratio < FOUR_MIN_PARTIAL_RATIO:
                continue
            segmented = length_ratio <= FOUR_SEGMENTED_LENGTH_RATIO
            rotation = _rotation_between(moving_edge.unit, -fixed_edge.unit)
            for anchor_index, translation in enumerate(
                _relation_translations(fixed_edge, moving_edge, rotation)
            ):
                key = _relation_key(rotation, translation)
                if key in seen:
                    continue
                seen.add(key)
                length_error = abs(fixed_edge.length_mm - moving_edge.length_mm) / longer
                score = (
                    fixed_hypothesis.score
                    + moving_hypothesis.score
                    + length_error
                    + (0.08 if segmented else 0.0)
                    + (0.01 if anchor_index == 2 and segmented else 0.0)
                )
                candidates.append(
                    PairRelation(
                        fixed_hypothesis.piece_id,
                        moving_hypothesis.piece_id,
                        fixed_hypothesis,
                        moving_hypothesis,
                        fixed_edge.index,
                        moving_edge.index,
                        rotation,
                        translation,
                        score,
                        shorter,
                        segmented,
                    )
                )
    return candidates


def build_pair_relations(fixed_piece, moving_piece):
    """构建一个有向片对的最佳刚体接缝关系，数量不超过固定现场上限。

    主要流程：各取最多三个形状假设，枚举反向边对齐，按拟合和长度误差排序去重；
    如果存在长短分段关系但总榜前列全是完整边，则保留最佳分段关系作为T形兜底。
    """
    fixed_hypotheses = build_shape_hypotheses(fixed_piece)
    moving_hypotheses = build_shape_hypotheses(moving_piece)
    candidates = []
    for fixed_hypothesis in fixed_hypotheses:
        for moving_hypothesis in moving_hypotheses:
            candidates.extend(
                _build_hypothesis_pair_relations(fixed_hypothesis, moving_hypothesis)
            )
    candidates.sort(
        key=lambda item: (
            item.score,
            item.fixed_edge_index,
            item.moving_edge_index,
            _relation_key(item.rotation, item.translation),
        )
    )
    selected = list(candidates[:FOUR_PAIR_RELATION_LIMIT])
    segmented_candidates = [item for item in candidates if item.segmented]
    if segmented_candidates:
        # 分段榜不仅保留最低误差，还保留覆盖长度最大的关系。否则30/40等偶然边对会
        # 占据唯一分段名额，真正80/40的T形长边关系会在总榜截断时丢失。
        required_segmented = (
            segmented_candidates[0],
            max(
                segmented_candidates,
                key=lambda item: (item.overlap_length_mm, -item.score),
            ),
        )
        for required in required_segmented:
            if any(
                _relation_key(item.rotation, item.translation)
                == _relation_key(required.rotation, required.translation)
                for item in selected
            ):
                continue
            if len(selected) >= FOUR_PAIR_RELATION_LIMIT:
                # 优先替换最差的非分段关系；如果已全是分段，则替换当前总榜末项。
                replacement_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if not selected[index].segmented
                    ),
                    len(selected) - 1,
                )
                selected[replacement_index] = required
            else:
                selected.append(required)
        selected.sort(key=lambda item: item.score)
    return tuple(selected)


class _LayoutState:
    """保存分层搜索中已放碎片的刚体变换、形状假设和累计评分。"""

    def __init__(self, transforms, hypotheses, score=0.0, overlap_area_mm2=0.0):
        """复制状态字典，避免不同beam分支共享可变矩阵。"""
        self.transforms = {
            int(index): (
                np.asarray(rotation, dtype=np.float64).copy(),
                np.asarray(translation, dtype=np.float64).copy(),
            )
            for index, (rotation, translation) in transforms.items()
        }
        self.hypotheses = dict(hypotheses)
        self.score = float(score)
        self.overlap_area_mm2 = float(overlap_area_mm2)


def _transform_points(points, rotation, translation):
    """使用行向量约定对N×2点执行二维刚体旋转和平移。"""
    normalized = _normalize_points(points, "刚体变换点")
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = _normalize_point(translation, "刚体平移")
    if rotation.shape != (2, 2) or not np.all(np.isfinite(rotation)):
        raise ValueError("刚体旋转必须是有限2×2矩阵")
    return normalized @ rotation.T + translation


def _state_polygons(state):
    """按状态中的形状假设和变换返回已放碎片毫米多边形字典。"""
    polygons = {}
    for index, (rotation, translation) in state.transforms.items():
        hypothesis = state.hypotheses.get(index)
        if hypothesis is None:
            continue
        polygons[index] = _transform_points(
            hypothesis.vertices,
            rotation,
            translation,
        )
    return polygons


def _pair_overlap_area_mm2(first_polygon, second_polygon, pixels_per_mm=None):
    """栅格估算两个任意简单多边形的内部重叠面积，并排除共享边像素。

    主要流程：建立只覆盖两片局部边界的小画布，分别填充并轻蚀一像素后求交。轻蚀
    只用于重叠判断，不改变目标轮廓或填充率，可避免正确共享接缝被算成一像素重叠。
    """
    scale = (
        FOUR_OVERLAP_PIXELS_PER_MM
        if pixels_per_mm is None
        else float(pixels_per_mm)
    )
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("重叠栅格比例必须是正有限数")
    first = _normalize_polygon(first_polygon)
    second = _normalize_polygon(second_polygon)
    all_points = np.vstack((first, second))
    minimum = np.floor(np.min(all_points, axis=0) * scale) - 3.0
    maximum = np.ceil(np.max(all_points, axis=0) * scale) + 3.0
    width, height = np.maximum(1, (maximum - minimum + 1.0).astype(np.int32))
    if width * height > 4_000_000:
        raise ValueError("重叠检查画布异常过大")
    masks = []
    erosion_kernel = np.ones((3, 3), dtype=np.uint8)
    for polygon in (first, second):
        mask = np.zeros((int(height), int(width)), dtype=np.uint8)
        pixels = np.rint(polygon * scale - minimum).astype(np.int32)
        cv2.fillPoly(mask, [pixels.reshape(-1, 1, 2)], 255)
        masks.append(cv2.erode(mask, erosion_kernel, iterations=1))
    overlap_pixels = cv2.countNonZero(cv2.bitwise_and(masks[0], masks[1]))
    return float(overlap_pixels) / (scale * scale)


def _compose_moving_transform(fixed_transform, relation):
    """把moving→fixed接缝变换组合到fixed→assembly状态变换之后。"""
    fixed_rotation, fixed_translation = fixed_transform
    moving_rotation = fixed_rotation @ relation.rotation
    moving_translation = relation.translation @ fixed_rotation.T + fixed_translation
    return moving_rotation, moving_translation


def _state_key(state):
    """量化已放碎片位姿和形状键，删除不同扩展顺序产生的等价状态。"""
    entries = []
    for index in sorted(state.transforms):
        rotation, translation = state.transforms[index]
        angle = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
        hypothesis = state.hypotheses.get(index)
        entries.append(
            (
                index,
                round(angle, 1),
                round(float(translation[0]), 1),
                round(float(translation[1]), 1),
                None if hypothesis is None else _hypothesis_key(hypothesis.vertices),
            )
        )
    return tuple(entries)


def _partial_bounds_are_possible(polygons):
    """用题目最大尺寸对中间组合做保守剪枝，明显过大的状态立即拒绝。"""
    if not polygons:
        return False
    points = np.vstack(tuple(polygons.values())).astype(np.float32)
    rectangle = cv2.minAreaRect(points.reshape(-1, 1, 2))
    sides = sorted((float(rectangle[1][0]), float(rectangle[1][1])))
    if sides[0] <= 1e-6 or sides[1] <= 1e-6:
        return False
    return (
        sides[1] <= FOUR_RECT_LONG_RANGE_MM[1] + 5.0
        and sides[0] <= FOUR_RECT_SHORT_RANGE_MM[1] + 5.0
    )


def _expand_state(state, moving_index, fixed_index, relation):
    """尝试按一条接缝把未放碎片加入状态；重叠或尺寸越界返回None。"""
    fixed_hypothesis = state.hypotheses.get(fixed_index)
    relation_fixed_key = _hypothesis_key(relation.fixed_hypothesis.vertices)
    if (
        fixed_hypothesis is not None
        and _hypothesis_key(fixed_hypothesis.vertices) != relation_fixed_key
    ):
        return None
    transforms = dict(state.transforms)
    hypotheses = dict(state.hypotheses)
    hypotheses[fixed_index] = relation.fixed_hypothesis
    hypotheses[moving_index] = relation.moving_hypothesis
    transforms[moving_index] = _compose_moving_transform(
        transforms[fixed_index],
        relation,
    )
    candidate = _LayoutState(
        transforms,
        hypotheses,
        score=state.score + relation.score,
        overlap_area_mm2=state.overlap_area_mm2,
    )
    polygons = _state_polygons(candidate)
    moving_polygon = polygons[moving_index]
    added_overlap = 0.0
    for other_index, other_polygon in polygons.items():
        if other_index == moving_index:
            continue
        added_overlap += _pair_overlap_area_mm2(moving_polygon, other_polygon)
    moving_area = abs(float(cv2.contourArea(moving_polygon.astype(np.float32))))
    if added_overlap / max(1.0, moving_area) > FOUR_INTERMEDIATE_MAX_OVERLAP_RATIO:
        return None
    candidate.overlap_area_mm2 += added_overlap
    candidate.score += added_overlap / max(1.0, moving_area)
    if not _partial_bounds_are_possible(polygons):
        return None
    return candidate


def _rotation_matrix(angle_rad):
    """按弧度构造行列式为1的二维旋转矩阵。"""
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _canonicalize_state(state):
    """把完整布局旋正到左上原点，返回多边形、全局变换和矩形宽高。

    旋正方向取最小外接矩形最长边，允许整体相差180度；这种差异不会改变机械可执行
    性。返回``(多边形字典, 旋转, 平移, 宽, 高)``，退化时返回None。
    """
    polygons = _state_polygons(state)
    if len(polygons) != 4:
        return None
    all_points = np.vstack(tuple(polygons.values())).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points.reshape(-1, 1, 2))
    box = cv2.boxPoints(rectangle).astype(np.float64)
    edge_vectors = np.roll(box, -1, axis=0) - box
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    longest_index = int(np.argmax(edge_lengths))
    longest_vector = edge_vectors[longest_index]
    if float(np.linalg.norm(longest_vector)) <= 1e-6:
        return None
    angle = math.atan2(float(longest_vector[1]), float(longest_vector[0]))
    canonical_rotation = _rotation_matrix(-angle)
    rotated = {
        index: polygon @ canonical_rotation.T
        for index, polygon in polygons.items()
    }
    minimum = np.min(np.vstack(tuple(rotated.values())), axis=0)
    canonical_translation = -minimum
    canonical = {
        index: polygon + canonical_translation
        for index, polygon in rotated.items()
    }
    maximum = np.max(np.vstack(tuple(canonical.values())), axis=0)
    width_mm, height_mm = float(maximum[0]), float(maximum[1])
    if width_mm < height_mm:
        # 数值或OpenCV边序导致长边落在Y轴时再旋转90度，统一长边沿目标X方向。
        quarter_turn = _rotation_matrix(-math.pi * 0.5)
        rerotated = {
            index: polygon @ quarter_turn.T
            for index, polygon in canonical.items()
        }
        second_minimum = np.min(np.vstack(tuple(rerotated.values())), axis=0)
        canonical = {
            index: polygon - second_minimum
            for index, polygon in rerotated.items()
        }
        canonical_rotation = quarter_turn @ canonical_rotation
        canonical_translation = (
            canonical_translation @ quarter_turn.T - second_minimum
        )
        maximum = np.max(np.vstack(tuple(canonical.values())), axis=0)
        width_mm, height_mm = float(maximum[0]), float(maximum[1])
    return (
        canonical,
        canonical_rotation,
        canonical_translation,
        width_mm,
        height_mm,
    )


def _evaluate_complete_state(state):
    """执行四片矩形尺寸、填充率和重叠率硬验收并返回结构化指标。"""
    canonicalized = _canonicalize_state(state)
    if canonicalized is None:
        return None, "geometry_reject"
    canonical, rotation, translation, width_mm, height_mm = canonicalized
    long_side = max(width_mm, height_mm)
    short_side = min(width_mm, height_mm)
    size_tolerance = float(FOUR_RECT_SIZE_TOLERANCE_MM)
    if not (
        FOUR_RECT_LONG_RANGE_MM[0] - size_tolerance
        <= long_side
        <= FOUR_RECT_LONG_RANGE_MM[1] + size_tolerance
        and FOUR_RECT_SHORT_RANGE_MM[0] - size_tolerance
        <= short_side
        <= FOUR_RECT_SHORT_RANGE_MM[1] + size_tolerance
    ):
        return None, "size_reject"

    polygon_areas = [
        abs(float(cv2.contourArea(polygon.astype(np.float32))))
        for polygon in canonical.values()
    ]
    total_area = sum(polygon_areas)
    overlap_area = 0.0
    polygon_items = list(canonical.items())
    for first_position, (_first_index, first_polygon) in enumerate(polygon_items):
        for _second_index, second_polygon in polygon_items[first_position + 1 :]:
            overlap_area += _pair_overlap_area_mm2(first_polygon, second_polygon)
    overlap_ratio = overlap_area / max(1.0, total_area)
    if overlap_ratio > FOUR_MAX_OVERLAP_RATIO:
        return None, "overlap_reject"
    rectangle_area = max(1.0, width_mm * height_mm)
    fill_ratio = max(0.0, min(1.0, (total_area - overlap_area) / rectangle_area))
    if fill_ratio < FOUR_RELAXED_MIN_FILL_RATIO:
        return None, "fill_reject"
    tier = "strict" if fill_ratio >= FOUR_STRICT_MIN_FILL_RATIO else "relaxed"
    return {
        "canonical": canonical,
        "rotation": rotation,
        "translation": translation,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "fill_ratio": fill_ratio,
        "overlap_ratio": overlap_ratio,
        "tier": tier,
    }, "ok"


def _normalize_work_target(work_region_mm, split_y_mm):
    """校验工作区域与下半区并返回可放置目标矩形的毫米边界。"""
    try:
        work_values = tuple(float(value) for value in work_region_mm)
        split_y = float(split_y_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("工作区域和split_y_mm必须包含有限数字") from error
    if len(work_values) != 4 or not np.all(np.isfinite(work_values)):
        raise ValueError("work_region_mm必须包含X/Y/W/H四个有限数字")
    if not math.isfinite(split_y):
        raise ValueError("split_y_mm必须是有限数字")
    work_x, work_y, work_width, work_height = work_values
    if work_width <= 0.0 or work_height <= 0.0:
        raise ValueError("工作区域宽高必须大于零")
    lower_top = max(work_y, split_y)
    lower_bottom = work_y + work_height
    if lower_top >= lower_bottom:
        raise ValueError("分界线下方没有可用目标区域")
    return work_x, lower_top, work_width, lower_bottom - lower_top


def _place_canonical_in_lower_region(canonical_result, work_region_mm, split_y_mm):
    """把旋正矩形居中放入下半区，必要时尝试整体旋转90度。

    返回目标多边形、assembly到目标的全局旋转/平移以及目标矩形；两种方向都放不下
    时返回None，调用方不得生成机械placements。
    """
    work_x, lower_top, available_width, available_height = _normalize_work_target(
        work_region_mm,
        split_y_mm,
    )
    canonical = canonical_result["canonical"]
    base_rotation = canonical_result["rotation"]
    base_translation = canonical_result["translation"]
    width_mm = canonical_result["width_mm"]
    height_mm = canonical_result["height_mm"]
    orientation_candidates = [
        (
            canonical,
            base_rotation,
            base_translation,
            width_mm,
            height_mm,
        )
    ]
    quarter_turn = _rotation_matrix(math.pi * 0.5)
    rotated = {
        index: polygon @ quarter_turn.T
        for index, polygon in canonical.items()
    }
    rotated_minimum = np.min(np.vstack(tuple(rotated.values())), axis=0)
    rotated = {
        index: polygon - rotated_minimum
        for index, polygon in rotated.items()
    }
    orientation_candidates.append(
        (
            rotated,
            quarter_turn @ base_rotation,
            base_translation @ quarter_turn.T - rotated_minimum,
            height_mm,
            width_mm,
        )
    )

    for polygons, rotation, translation, target_width, target_height in orientation_candidates:
        if target_width > available_width + 1e-6 or target_height > available_height + 1e-6:
            continue
        offset = np.asarray(
            (
                work_x + (available_width - target_width) * 0.5,
                lower_top + (available_height - target_height) * 0.5,
            ),
            dtype=np.float64,
        )
        target_polygons = {
            index: polygon + offset
            for index, polygon in polygons.items()
        }
        return {
            "polygons": target_polygons,
            "rotation": rotation,
            "translation": translation + offset,
            "target_rect_mm": (
                float(offset[0]),
                float(offset[1]),
                float(target_width),
                float(target_height),
            ),
        }
    return None


def _build_success_plan(pieces, state, evaluated, work_region_mm, split_y_mm, search_nodes):
    """把通过硬验收的布局转换成兼容UART和绘制层的AssemblyPlan。"""
    target = _place_canonical_in_lower_region(
        evaluated,
        work_region_mm,
        split_y_mm,
    )
    if target is None:
        return AssemblyPlan.failed("target_range", search_nodes=search_nodes)
    placements = []
    global_rotation = target["rotation"]
    global_translation = target["translation"]
    for index, piece in enumerate(pieces):
        piece_rotation, piece_translation = state.transforms[index]
        total_rotation = global_rotation @ piece_rotation
        total_translation = piece_translation @ global_rotation.T + global_translation
        source_center = _normalize_point(piece["center_mm"], "碎片源中心")
        target_center = source_center @ total_rotation.T + total_translation
        angle_deg = math.degrees(
            math.atan2(float(total_rotation[1, 0]), float(total_rotation[0, 0]))
        )
        placements.append(
            AssemblyPlacement(
                piece["id"],
                source_center,
                target_center,
                target["polygons"][index],
                angle_deg,
            )
        )
    diagnostics = {
        "fill_milli": int(round(evaluated["fill_ratio"] * 1000.0)),
        "overlap_milli": int(round(evaluated["overlap_ratio"] * 1000.0)),
        "relaxed": 1 if evaluated["tier"] == "relaxed" else 0,
    }
    score = (
        state.score
        + (1.0 - evaluated["fill_ratio"]) * 10.0
        + evaluated["overlap_ratio"] * 10.0
    )
    return AssemblyPlan(
        True,
        placements=placements,
        target_rect_mm=target["target_rect_mm"],
        score=score,
        reason="ok",
        search_nodes=search_nodes,
        diagnostics=diagnostics,
    )


def _normalize_fast_diagnostics(diagnostics):
    """把FOUR_FAST诊断转换为FOUR运行器使用的统一整数字段。

    参数diagnostics来自四片快速图核心；返回新的字典，不修改底层对象。原始字段全部
    保留，同时增加界面需要的填充率、重叠率和四类拒绝计数别名。
    """
    normalized = {
        str(name): int(value)
        for name, value in (diagnostics or {}).items()
    }
    normalized["fill_milli"] = int(normalized.get("fill_permille", 0))
    normalized["overlap_milli"] = int(normalized.get("overlap_permille", 0))
    normalized["overlap_reject"] = int(normalized.get("four_overlap_reject", 0))
    normalized["size_reject"] = int(normalized.get("four_size_reject", 0))
    normalized["fill_reject"] = int(normalized.get("four_fill_reject", 0))
    normalized["geometry_reject"] = int(normalized.get("four_partial_reject", 0))
    return normalized


def _fast_failure_reason(diagnostics):
    """根据四片快速图计数返回稳定的FOUR失败原因字符串。

    尺寸不合格优先返回size_reject；已形成完整候选但没有通过填充或外边验收时返回
    no_rect；连两片候选都没有时返回no_edge。返回值只用于状态显示，不携带机械目标。
    """
    if int(diagnostics.get("size_reject", 0)) > 0:
        return "size_reject"
    if (
        int(diagnostics.get("fill_reject", 0)) > 0
        or int(diagnostics.get("four_outer_edge_reject", 0)) > 0
        or int(diagnostics.get("four_complete_states", 0)) > 0
    ):
        return "no_rect"
    if int(diagnostics.get("four_pair_states", 0)) <= 0:
        return "no_edge"
    return "no_rect"


def _plan_size_is_within_four_tolerance(plan):
    """复核成功计划的实测矩形尺寸是否位于题目范围加测量容差内。

    plan.target_rect_mm中的宽高仍保持实际毫米值，不会被缩放到题目边界；这里只允许
    视觉轮廓在长短边上下限各偏差FOUR_RECT_SIZE_TOLERANCE_MM。无目标矩形返回False。
    """
    target_rect = getattr(plan, "target_rect_mm", None)
    if target_rect is None or len(target_rect) != 4:
        return False
    measured_sides = sorted((float(target_rect[2]), float(target_rect[3])))
    if not np.all(np.isfinite(measured_sides)):
        return False
    short_side, long_side = measured_sides
    tolerance = float(FOUR_RECT_SIZE_TOLERANCE_MM)
    return bool(
        FOUR_RECT_LONG_RANGE_MM[0] - tolerance
        <= long_side
        <= FOUR_RECT_LONG_RANGE_MM[1] + tolerance
        and FOUR_RECT_SHORT_RANGE_MM[0] - tolerance
        <= short_side
        <= FOUR_RECT_SHORT_RANGE_MM[1] + tolerance
    )


def _rebuild_fast_plan_placements(pieces, plan):
    """由源顶点和快图目标顶点重新计算可直接发送的中心与旋转角。

    旧FOUR_FAST使用最小外接矩形方向计算角度，规则矩形可能出现180度二义性；此外
    视觉提供的center_mm可能与轮廓面积质心有小偏差。这里对同序顶点执行无缩放、
    无镜像Kabsch拟合，并用同一个刚体变换计算target_center_mm。返回值仍是传入的
    独立AssemblyPlan，但placements会替换为几何自洽的新对象。
    """
    pieces_by_id = {str(piece.get("id", "")): piece for piece in pieces}
    rebuilt = []
    for placement in plan.placements:
        piece = pieces_by_id.get(str(placement.piece_id))
        if piece is None:
            raise ValueError("FOUR_FAST结果包含未知碎片编号")
        source_vertices = _normalize_points(piece.get("vertices_mm"), "碎片源顶点")
        target_vertices = _normalize_points(
            placement.target_polygon_mm,
            "碎片目标顶点",
        )
        if len(source_vertices) != len(target_vertices):
            raise ValueError("FOUR_FAST源目标顶点数量不一致")

        source_mean = np.mean(source_vertices, axis=0)
        target_mean = np.mean(target_vertices, axis=0)
        source_centered = source_vertices - source_mean
        target_centered = target_vertices - target_mean
        # 二维Kabsch可以用点积/叉积闭式求解，无需MaixPy设备端额外依赖SVD。
        dot_sum = float(np.sum(source_centered * target_centered))
        cross_sum = float(np.sum(
            source_centered[:, 0] * target_centered[:, 1]
            - source_centered[:, 1] * target_centered[:, 0]
        ))
        rotation_scale = math.hypot(dot_sum, cross_sum)
        if rotation_scale <= 1e-12:
            raise ValueError("FOUR_FAST源目标轮廓无法确定旋转角")
        cosine = dot_sum / rotation_scale
        sine = cross_sum / rotation_scale
        # 行向量右乘矩阵等价于列向量的标准二维旋转；行列式恒为1，不允许镜像。
        row_rotation = np.asarray(
            ((cosine, sine), (-sine, cosine)),
            dtype=np.float64,
        )
        translation = target_mean - source_mean @ row_rotation
        rebuilt_target = source_vertices @ row_rotation + translation
        maximum_error = float(np.max(np.linalg.norm(
            rebuilt_target - target_vertices,
            axis=1,
        )))
        if maximum_error > 1e-4:
            raise ValueError("FOUR_FAST目标不是源轮廓的刚体变换")

        source_center = _normalize_point(piece.get("center_mm"), "碎片源中心")
        target_center = source_center @ row_rotation + translation
        rotation_delta_deg = math.degrees(
            math.atan2(float(row_rotation[0, 1]), float(row_rotation[0, 0]))
        )
        rebuilt.append(
            AssemblyPlacement(
                placement.piece_id,
                source_center,
                target_center,
                rebuilt_target,
                rotation_delta_deg,
            )
        )
    rebuilt.sort(key=lambda item: item.piece_id)
    plan.placements = rebuilt
    return plan


def _prefer_fast_graph_first(pieces):
    """判断四片快照是否应优先使用抗噪声快图。

    干净三角形/四边形由原生FOUR图直接处理；任一简化轮廓超过四个顶点，通常代表
    反光、裸露铁片或远距离角点形成了额外折点，此时实机验证快图的边长守恒与紧凑
    度排序更可靠。只读取vertices_mm，不使用稠密raw_contour_mm，避免把正常轮廓点
    采样数量误当成噪声。
    """
    for piece in pieces:
        try:
            if len(piece.get("vertices_mm", ())) > 4:
                return True
        except (AttributeError, TypeError):
            return False
    return False


class FourPieceSolveJob:
    """按有限工作单元跨帧推进的FOUR专用接缝图搜索任务。"""

    def __init__(
        self,
        pieces,
        work_region_mm,
        split_y_mm,
        beam_width=FOUR_BEAM_WIDTH,
        active_budget_seconds=FOUR_ACTIVE_BUDGET_SECONDS,
        clock=None,
    ):
        """准备四片快照、候选关系图和增量生成器。

        关键参数pieces必须恰有四个完整上半区碎片；active_budget_seconds只累计advance
        内实际CPU时间。无效数量立即形成失败结果；合法输入直到advance才展开组合。
        """
        self.pieces = tuple(pieces)
        self.work_region_mm = tuple(work_region_mm)
        self.split_y_mm = float(split_y_mm)
        self.beam_width = int(beam_width)
        self.active_budget_seconds = float(active_budget_seconds)
        self._clock = time.monotonic if clock is None else clock
        if self.beam_width < 1:
            raise ValueError("beam_width必须至少为1")
        if self.active_budget_seconds <= 0.0 or not math.isfinite(
            self.active_budget_seconds
        ):
            raise ValueError("active_budget_seconds必须是正有限数")
        self.search_nodes = 0
        self.active_seconds = 0.0
        self.result = None
        self.done = False
        self._stage = "validate"
        self._fast_progress = {}
        self._relation_graph = {}
        if len(self.pieces) != 4:
            self.result = AssemblyPlan.failed("four_needs_exactly_four")
            self.done = True
            self._generator = None
            return
        try:
            for piece in self.pieces:
                if (
                    not isinstance(piece, dict)
                    or not piece.get("id")
                    or piece.get("complete") is not True
                    or piece.get("region") != "upper"
                ):
                    raise ValueError("FOUR只接受四个完整上半区碎片")
                _normalize_polygon(piece.get("raw_contour_mm", piece.get("vertices_mm")))
                _normalize_point(piece["center_mm"], "碎片源中心")
            _normalize_work_target(self.work_region_mm, self.split_y_mm)
            # 原生FOUR图仍作为低成本第一阶段；只有它无解时才启动实机验证快图。
            for fixed_index, fixed_piece in enumerate(self.pieces):
                for moving_index, moving_piece in enumerate(self.pieces):
                    if fixed_index == moving_index:
                        continue
                    self._relation_graph[(fixed_index, moving_index)] = (
                        build_pair_relations(fixed_piece, moving_piece)
                    )
        except (KeyError, TypeError, ValueError):
            self.result = AssemblyPlan.failed("four_input_invalid")
            self.done = True
            self._generator = None
            return
        self._stage = "search"
        self._generator = self._search_generator()

    @property
    def stage(self):
        """返回当前关系准备、搜索或完成阶段，供设备状态栏显示。"""
        return self._stage

    def _search_generator(self):
        """按轮廓复杂度选择首个四片图核心，无解时再运行另一核心。"""
        if _prefer_fast_graph_first(self.pieces):
            fast_result = yield from self._fast_search_generator()
            if fast_result.success:
                fast_result.diagnostics["solver_source_fast"] = 1
                fast_result.diagnostics["native_search_nodes"] = 0
                return fast_result

            native_result = yield from self._native_search_generator()
            native_result.diagnostics["solver_source_native"] = 1
            native_result.diagnostics["fast_search_nodes"] = int(
                fast_result.search_nodes
            )
            for name, value in fast_result.diagnostics.items():
                native_result.diagnostics[f"fast_{name}"] = int(value)
            return native_result

        native_result = yield from self._native_search_generator()
        if native_result.success:
            native_result.diagnostics["solver_source_native"] = 1
            return native_result

        fast_result = yield from self._fast_search_generator()
        fast_result.diagnostics["solver_source_fast"] = 1
        # 保存第一阶段失败概况，现场日志可以确认是否进入了实机快图兜底。
        fast_result.diagnostics["native_search_nodes"] = int(
            native_result.search_nodes
        )
        for name, value in native_result.diagnostics.items():
            fast_result.diagnostics[f"native_{name}"] = int(value)
        return fast_result

    def _native_search_generator(self):
        """运行原生FOUR关系图搜索；逐候选yield，失败时不启动通用FALLBACK。"""
        states = [
            _LayoutState(
                {0: (np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64))},
                {},
            )
        ]
        rejection_counts = {
            "overlap_reject": 0,
            "size_reject": 0,
            "fill_reject": 0,
            "geometry_reject": 0,
        }
        for _target_count in (2, 3, 4):
            next_states = []
            seen = set()
            for state in states:
                placed = tuple(sorted(state.transforms))
                unplaced = tuple(
                    index for index in range(4) if index not in state.transforms
                )
                for fixed_index in placed:
                    for moving_index in unplaced:
                        relations = self._relation_graph.get(
                            (fixed_index, moving_index),
                            (),
                        )
                        for relation in relations:
                            self.search_nodes += 1
                            candidate = _expand_state(
                                state,
                                moving_index,
                                fixed_index,
                                relation,
                            )
                            yield None
                            if candidate is None:
                                rejection_counts["overlap_reject"] += 1
                                continue
                            key = _state_key(candidate)
                            if key in seen:
                                continue
                            seen.add(key)
                            next_states.append(candidate)
            if not next_states:
                return AssemblyPlan.failed(
                    "no_rect",
                    search_nodes=self.search_nodes,
                    diagnostics=rejection_counts,
                )
            next_states.sort(key=lambda item: item.score)
            states = next_states[: self.beam_width]

        strict_candidates = []
        relaxed_candidates = []
        for state in states:
            self.search_nodes += 1
            evaluated, reason = _evaluate_complete_state(state)
            yield None
            if evaluated is None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            entry = (state.score, state, evaluated)
            if evaluated["tier"] == "strict":
                strict_candidates.append(entry)
            else:
                relaxed_candidates.append(entry)
        candidates = strict_candidates if strict_candidates else relaxed_candidates
        if not candidates:
            primary_reason = (
                "size_reject" if rejection_counts["size_reject"] else "no_rect"
            )
            return AssemblyPlan.failed(
                primary_reason,
                search_nodes=self.search_nodes,
                diagnostics=rejection_counts,
            )
        candidates.sort(
            key=lambda item: (
                item[0] + (1.0 - item[2]["fill_ratio"]) * 10.0,
                item[2]["overlap_ratio"],
            )
        )
        _score, best_state, best_evaluated = candidates[0]
        return _build_success_plan(
            self.pieces,
            best_state,
            best_evaluated,
            self.work_region_mm,
            self.split_y_mm,
            self.search_nodes,
        )

    def _fast_search_generator(self):
        """逐工作单元驱动经过实机验证的四片快速图核心。

        旧通用UNKNOWN链路包含GRAPH、FOUR_FAST和FALLBACK三个阶段；这里仅复用其中
        与四片白色几何完全对应的FOUR_FAST生成器，不会进入GRAPH、通用FALLBACK或
        KNOWN模板。生成器仍逐候选yield，因此MaixCAM2主循环可以持续刷新触摸和心跳。
        返回值统一转换为FOUR自己的失败原因和诊断字段。
        """
        fast_generator = _solve_unknown_four_fast_path(
            self.pieces,
            self.work_region_mm,
            self.split_y_mm,
            beam_width=self.beam_width,
            incremental=True,
            progress=self._fast_progress,
            strict_min_fill_ratio=FOUR_STRICT_MIN_FILL_RATIO,
            relaxed_min_fill_ratio=FOUR_RELAXED_MIN_FILL_RATIO,
            long_side_range_mm=(
                FOUR_RECT_LONG_RANGE_MM[0] - FOUR_RECT_SIZE_TOLERANCE_MM,
                FOUR_RECT_LONG_RANGE_MM[1] + FOUR_RECT_SIZE_TOLERANCE_MM,
            ),
            short_side_range_mm=(
                FOUR_RECT_SHORT_RANGE_MM[0] - FOUR_RECT_SIZE_TOLERANCE_MM,
                FOUR_RECT_SHORT_RANGE_MM[1] + FOUR_RECT_SIZE_TOLERANCE_MM,
            ),
            max_overlap_ratio=FOUR_MAX_OVERLAP_RATIO,
        )
        while True:
            try:
                next(fast_generator)
                # 运行中没有暴露内部局部诊断字典，因此用实际yield次数显示搜索进度；
                # 完成时会用FOUR_FAST的精确工作单元数覆盖该值。
                self.search_nodes += 1
                yield None
            except StopIteration as completed:
                plan, diagnostics = completed.value
                normalized_diagnostics = _normalize_fast_diagnostics(diagnostics)
                self.search_nodes = max(
                    self.search_nodes,
                    int(normalized_diagnostics.get("four_work_units", 0)),
                )
                if plan is None:
                    return AssemblyPlan.failed(
                        _fast_failure_reason(normalized_diagnostics),
                        search_nodes=self.search_nodes,
                        diagnostics=normalized_diagnostics,
                    )
                if not _plan_size_is_within_four_tolerance(plan):
                    normalized_diagnostics["size_reject"] = (
                        int(normalized_diagnostics.get("size_reject", 0)) + 1
                    )
                    return AssemblyPlan.failed(
                        "size_reject",
                        search_nodes=self.search_nodes,
                        diagnostics=normalized_diagnostics,
                    )
                try:
                    plan = _rebuild_fast_plan_placements(self.pieces, plan)
                except (KeyError, TypeError, ValueError):
                    # 源目标若不能构成同一个刚体变换，绝不能向F4发送不自洽的位姿。
                    normalized_diagnostics["geometry_reject"] = (
                        int(normalized_diagnostics.get("geometry_reject", 0)) + 1
                    )
                    return AssemblyPlan.failed(
                        "geometry_reject",
                        search_nodes=self.search_nodes,
                        diagnostics=normalized_diagnostics,
                    )
                # AssemblyPlan是本轮刚创建的独立快照，可以安全补充FOUR界面所需别名。
                plan.search_nodes = self.search_nodes
                plan.diagnostics.update(normalized_diagnostics)
                return plan

    def _finish_timeout(self):
        """在FOUR活动预算到期时关闭任务并优先返回已验证快图计划。

        主要流程：先关闭仍停在yield边界的生成器，读取FOUR_FAST写入共享进度的
        best_plan；缓存存在时克隆计划、复核题目尺寸并重新计算每片刚体位姿，全部
        通过后标记returned_best_at_timeout并返回。缓存缺失或复核失败时才返回
        solver_timeout，绝不能把未经复核的目标坐标发送给F4。

        返回值：成功时为独立的AssemblyPlan快照，失败时为不含placements的超时结果。
        本函数同时设置done和stage，后续advance会稳定返回同一个终态对象。
        """
        if self.done:
            return self.result
        if self._generator is not None:
            self._generator.close()

        progress_diagnostics = dict(
            self._fast_progress.get("rejection_counts", {})
        )
        for name, value in self._fast_progress.items():
            if name in (
                "best_plan",
                "rejection_counts",
                "current_stage",
                "result_source",
            ):
                continue
            if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
                progress_diagnostics[str(name)] = int(value)
        normalized_diagnostics = _normalize_fast_diagnostics(progress_diagnostics)
        normalized_diagnostics["active_elapsed_ms"] = int(
            round(max(0.0, self.active_seconds) * 1000.0)
        )
        self.search_nodes = max(
            int(self.search_nodes),
            int(self._fast_progress.get("search_nodes", 0)),
            int(normalized_diagnostics.get("four_work_units", 0)),
        )

        best_plan = self._fast_progress.get("best_plan")
        if isinstance(best_plan, AssemblyPlan) and best_plan.success:
            # 必须先构造独立对象再重建placements，不能修改底层共享进度保存的原计划。
            candidate = AssemblyPlan(
                True,
                placements=list(best_plan.placements),
                target_rect_mm=best_plan.target_rect_mm,
                score=best_plan.score,
                reason=best_plan.reason,
                search_nodes=self.search_nodes,
                diagnostics=dict(best_plan.diagnostics),
            )
            if _plan_size_is_within_four_tolerance(candidate):
                try:
                    candidate = _rebuild_fast_plan_placements(
                        self.pieces,
                        candidate,
                    )
                except (KeyError, TypeError, ValueError):
                    normalized_diagnostics["geometry_reject"] = (
                        int(normalized_diagnostics.get("geometry_reject", 0)) + 1
                    )
                else:
                    candidate.search_nodes = self.search_nodes
                    # 共享进度补充活动耗时和拒绝计数，但缓存计划中的最终填充率、
                    # 重叠率和验收层级更权威，不能被空进度归一化产生的0覆盖。
                    merged_diagnostics = dict(normalized_diagnostics)
                    merged_diagnostics.update(candidate.diagnostics)
                    merged_diagnostics["active_elapsed_ms"] = int(
                        normalized_diagnostics["active_elapsed_ms"]
                    )
                    merged_diagnostics["returned_best_at_timeout"] = 1
                    candidate.diagnostics = merged_diagnostics
                    self.result = candidate
            else:
                normalized_diagnostics["size_reject"] = (
                    int(normalized_diagnostics.get("size_reject", 0)) + 1
                )

        if self.result is None:
            self.result = AssemblyPlan.failed(
                "solver_timeout",
                search_nodes=self.search_nodes,
                diagnostics=normalized_diagnostics,
            )
        self.done = True
        self._stage = "done"
        return self.result

    def advance(self, time_budget_ms=24.0, work_unit_limit=64):
        """推进有限CPU时间和候选数量；未完成返回None，完成返回缓存计划。

        单帧预算与累计活动预算双重生效。每次生成、检查或拒绝一个组合关系计一个工作
        单元；达到任一当前帧门立即让出，保证触摸、显示和UART心跳继续刷新。
        """
        if self.done:
            return self.result
        try:
            time_budget_ms = float(time_budget_ms)
            work_unit_limit = int(work_unit_limit)
        except (TypeError, ValueError) as error:
            raise ValueError("求解时间片参数无效") from error
        if time_budget_ms <= 0.0 or work_unit_limit < 1:
            raise ValueError("求解时间片必须为正且至少包含一个工作单元")
        started_at = self._clock()
        units = 0
        while units < work_unit_limit:
            elapsed_this_call = self._clock() - started_at
            if elapsed_this_call * 1000.0 >= time_budget_ms:
                break
            if self.active_seconds + elapsed_this_call >= self.active_budget_seconds:
                self.active_seconds += elapsed_this_call
                return self._finish_timeout()
            try:
                next(self._generator)
                units += 1
            except StopIteration as completed:
                self.active_seconds += self._clock() - started_at
                self.result = completed.value
                self.done = True
                self._stage = "done"
                return self.result
        self.active_seconds += self._clock() - started_at
        return None

    def run_to_completion(self):
        """连续消费增量任务直到结束，供离线测试和兼容同步入口使用。"""
        while not self.done:
            self.advance(time_budget_ms=1000.0, work_unit_limit=100000)
        return self.result


def solve_four_piece_layout(
    pieces,
    work_region_mm,
    split_y_mm,
    beam_width=FOUR_BEAM_WIDTH,
    active_budget_seconds=FOUR_ACTIVE_BUDGET_SECONDS,
):
    """同步运行独立FOUR求解器并返回兼容AssemblyPlan。

    设备主循环应直接使用FourPieceSolveJob.advance保持非阻塞；本函数用于单元测试、
    PC回放和少量同步工具。它不调用KNOWN模板或旧UNKNOWN GRAPH/FALLBACK。
    """
    job = FourPieceSolveJob(
        pieces,
        work_region_mm,
        split_y_mm,
        beam_width=beam_width,
        active_budget_seconds=active_budget_seconds,
    )
    return job.run_to_completion()


class FourPieceRuntime:
    """组合FOUR视觉稳定门和增量求解任务，供设备主循环逐帧调用。"""

    def __init__(
        self,
        vision_runtime=None,
        beam_width=FOUR_BEAM_WIDTH,
        active_budget_seconds=FOUR_ACTIVE_BUDGET_SECONDS,
        solver_time_budget_ms=24.0,
        solver_work_unit_limit=64,
        debug_enabled=FOUR_SOLVER_DEBUG,
    ):
        """初始化独立视觉运行器和求解时间片参数并进入完全待机。

        vision_runtime允许测试或回放注入兼容对象；设备默认创建FourPieceVisionRuntime。
        关键参数beam和活动预算只影响FOUR求解器。构造函数不启动视觉或搜索。
        """
        self.vision_runtime = (
            FourPieceVisionRuntime() if vision_runtime is None else vision_runtime
        )
        if not callable(getattr(self.vision_runtime, "update", None)) or not callable(
            getattr(self.vision_runtime, "reset", None)
        ):
            raise ValueError("vision_runtime必须提供update和reset方法")
        self.beam_width = int(beam_width)
        self.active_budget_seconds = float(active_budget_seconds)
        self.solver_time_budget_ms = float(solver_time_budget_ms)
        self.solver_work_unit_limit = int(solver_work_unit_limit)
        self.debug_enabled = bool(debug_enabled)
        if self.beam_width < 1:
            raise ValueError("beam_width必须至少为1")
        if self.active_budget_seconds <= 0.0:
            raise ValueError("active_budget_seconds必须大于零")
        if self.solver_time_budget_ms <= 0.0 or self.solver_work_unit_limit < 1:
            raise ValueError("FOUR单帧求解预算无效")
        self.reset(propagate=False)

    def reset(self, propagate=True):
        """取消本轮求解、清空计划，并按需把复位传播到视觉稳定门。

        设备模式/CAL/START调用默认传播；构造函数使用propagate=False避免把外部注入
        运行器的初始化状态误记为一次用户复位。
        """
        if propagate:
            self.vision_runtime.reset()
        self.solve_job = None
        self.plan = None
        self.solve_start_count = 0
        self._result_logged = False

    def _debug_snapshot(self):
        """调试开启时输出一次冻结四片的编号、重心和形状顶点数。"""
        if not self.debug_enabled:
            return
        print(f"[FOUR] SNAPSHOT count={len(self.locked_pieces)}")
        for piece in self.locked_pieces:
            center = np.asarray(piece.get("center_mm", (0.0, 0.0)), dtype=np.float64)
            vertices = piece.get("vertices_mm", ())
            print(
                "[FOUR] PIECE "
                f"id={piece.get('id', '?')} "
                f"center_mm=({center[0]:.1f},{center[1]:.1f}) "
                f"vertices={len(vertices)}"
            )

    def _cache_result(self, result):
        """缓存成功或失败计划，并在调试开启时只输出一次最终诊断。"""
        self.plan = result
        if self._result_logged or not self.debug_enabled or result is None:
            return
        diagnostics = getattr(result, "diagnostics", {})
        print(
            "[FOUR] RESULT "
            f"success={1 if bool(getattr(result, 'success', False)) else 0} "
            f"reason={getattr(result, 'reason', 'unknown')} "
            f"nodes={int(getattr(result, 'search_nodes', 0))} "
            f"fill={int(diagnostics.get('fill_milli', 0)) / 10.0:.1f}% "
            f"overlap={int(diagnostics.get('overlap_milli', 0)) / 10.0:.1f}%"
        )
        self._result_logged = True

    @property
    def stable_count(self):
        """返回视觉运行器当前连续稳定帧数。"""
        return int(getattr(self.vision_runtime, "stable_count", 0))

    @property
    def stable_frames(self):
        """返回视觉运行器要求的稳定帧总数。"""
        return int(getattr(self.vision_runtime, "stable_frames", 3))

    @property
    def snapshot_locked(self):
        """返回本轮是否已经冻结四片视觉快照。"""
        return bool(getattr(self.vision_runtime, "snapshot_locked", False))

    @property
    def locked_pieces(self):
        """返回冻结四片元组；未锁定时为空元组。"""
        return getattr(self.vision_runtime, "locked_pieces", ())

    @property
    def last_detection(self):
        """返回最近实时或锁定的FOUR检测结果。"""
        return getattr(self.vision_runtime, "last_detection", None)

    @property
    def is_solving(self):
        """返回是否存在尚未完成的四片专用求解任务。"""
        return self.solve_job is not None and not self.solve_job.done and self.plan is None

    @property
    def search_nodes(self):
        """返回专用求解器已经检查的候选工作单元数。"""
        return 0 if self.solve_job is None else int(self.solve_job.search_nodes)

    @property
    def stage(self):
        """返回DETECTING、SOLVING、DONE或IDLE，供正常页状态显示。"""
        if self.plan is not None:
            return "done"
        if self.is_solving:
            return str(self.solve_job.stage)
        if self.snapshot_locked:
            return "locked"
        if self.last_detection is not None:
            return "detecting"
        return "idle"

    def update(
        self,
        frame_bgr,
        paper_quad,
        paper_orientation,
        work_region_mm,
        split_y_mm,
        time_budget_ms=None,
        work_unit_limit=None,
    ):
        """推进一次FOUR视觉或求解时间片并返回当前完成计划。

        主要流程：已有成功或失败计划时直接返回；已有求解任务时只advance；否则调用
        视觉运行器，锁定后恰好创建一次FourPieceSolveJob且本帧不继续重算视觉。
        返回None表示仍在检测/求解，返回AssemblyPlan表示本轮已永久结束。
        """
        if self.plan is not None:
            return self.plan
        if self.solve_job is not None:
            result = self.solve_job.advance(
                time_budget_ms=(
                    self.solver_time_budget_ms
                    if time_budget_ms is None
                    else float(time_budget_ms)
                ),
                work_unit_limit=(
                    self.solver_work_unit_limit
                    if work_unit_limit is None
                    else int(work_unit_limit)
                ),
            )
            if result is not None:
                self._cache_result(result)
            return self.plan

        detection = self.vision_runtime.update(
            frame_bgr,
            paper_quad,
            paper_orientation,
            split_y_mm=split_y_mm,
        )
        if not bool(getattr(detection, "locked", False)):
            return None
        self.solve_job = FourPieceSolveJob(
            self.locked_pieces,
            work_region_mm,
            split_y_mm,
            beam_width=self.beam_width,
            active_budget_seconds=self.active_budget_seconds,
        )
        self.solve_start_count += 1
        self._debug_snapshot()
        if self.solve_job.done:
            self._cache_result(self.solve_job.result)
        return self.plan
