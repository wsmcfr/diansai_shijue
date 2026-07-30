"""UNKNOWN FOUR模式的轮廓假设、接缝关系和四片专用图搜索。"""

import math

import cv2
import numpy as np


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
