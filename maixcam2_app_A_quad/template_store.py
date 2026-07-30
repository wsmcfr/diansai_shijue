"""已知拼图碎片的形状模板登记、持久化和一对一匹配。"""

import itertools
import json
import math
import os

import cv2
import numpy as np


# 模板文件版本用于阻止未来字段变化后静默读取不兼容数据。
TEMPLATE_VERSION = 1
# 32点在MaixCAM2上只产生固定规模的小矩阵，同时足以消除3～5顶点近似抖动。
CONTOUR_SAMPLE_COUNT = 32
# 归一化轮廓RMS误差乘以固定系数后与既有1.x匹配阈值处于相近量级。
CONTOUR_SCORE_SCALE = 8.0


def resample_closed_contour(vertices, sample_count=CONTOUR_SAMPLE_COUNT):
    """沿闭合多边形周长等距离重采样为固定数量二维点。

    主要流程：校验至少三个有限顶点，补上首尾闭合边，按累计弧长定位每个等距采样点
    并在线段内插值。关键参数sample_count必须不少于3。返回N×2 float64数组；退化
    轮廓抛出ValueError。原始顶点不会修改。
    """
    points = np.asarray(vertices, dtype=np.float64)
    try:
        sample_count = int(sample_count)
    except (TypeError, ValueError) as error:
        raise ValueError("轮廓采样点数必须是整数") from error
    if (
        points.ndim != 2
        or points.shape[0] < 3
        or points.shape[1] != 2
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("闭合轮廓必须包含至少三个有限二维点")
    if sample_count < 3:
        raise ValueError("轮廓采样点数不能少于3")

    next_points = np.roll(points, -1, axis=0)
    segment_vectors = next_points - points
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    perimeter = float(np.sum(segment_lengths))
    if perimeter <= 1e-9 or np.count_nonzero(segment_lengths > 1e-9) < 3:
        raise ValueError("闭合轮廓周长过小或已经退化")

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_distances = np.arange(sample_count, dtype=np.float64) * (
        perimeter / float(sample_count)
    )
    segment_indices = np.searchsorted(
        cumulative[1:],
        sample_distances,
        side="right",
    )
    segment_indices = np.minimum(segment_indices, len(points) - 1)
    local_distances = sample_distances - cumulative[segment_indices]
    denominators = segment_lengths[segment_indices]
    fractions = np.divide(
        local_distances,
        denominators,
        out=np.zeros_like(local_distances),
        where=denominators > 1e-12,
    )
    return points[segment_indices] + segment_vectors[segment_indices] * fractions[:, None]


def align_closed_contours(
    source_vertices,
    target_vertices,
    sample_count=CONTOUR_SAMPLE_COUNT,
    normalize_scale=False,
):
    """用循环起点和无镜像Kabsch配准两个闭合轮廓。

    主要流程：两侧先等弧长采样并中心化，可选按各自RMS半径归一尺度；随后枚举目标
    采样起点，只接受行列式为正的二维旋转。这里故意不反转遍历方向，因为反转会让
    翻面轮廓错误地成为合法匹配。返回``(旋转角度, RMS误差)``。
    """
    source = resample_closed_contour(source_vertices, sample_count)
    target = resample_closed_contour(target_vertices, sample_count)
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)

    if normalize_scale:
        source_scale = float(np.sqrt(np.mean(np.sum(source_centered**2, axis=1))))
        target_scale = float(np.sqrt(np.mean(np.sum(target_centered**2, axis=1))))
        if source_scale <= 1e-9 or target_scale <= 1e-9:
            raise ValueError("轮廓尺度过小，无法归一化")
        source_centered = source_centered / source_scale
        target_centered = target_centered / target_scale

    # 二维正旋转可用复数单位相位闭式求解，等价于2×2 Kabsch，但避免每个循环
    # 起点都执行一次SVD。KNOWN四片最多16组形状比较，固定点积能显著降低设备耗时。
    source_complex = source_centered[:, 0] + 1j * source_centered[:, 1]
    target_complex = target_centered[:, 0] + 1j * target_centered[:, 1]
    best_angle = 0.0
    best_error = float("inf")
    for shift in range(sample_count):
        aligned_target = np.roll(target_complex, shift)
        correlation = np.vdot(source_complex, aligned_target)
        correlation_magnitude = float(abs(correlation))
        if correlation_magnitude <= 1e-12:
            continue
        rotation_complex = correlation / correlation_magnitude
        transformed = source_complex * rotation_complex
        error = float(np.sqrt(np.mean(np.abs(transformed - aligned_target) ** 2)))
        if error < best_error:
            best_error = error
            best_angle = math.degrees(math.atan2(rotation_complex.imag, rotation_complex.real))
    if not np.isfinite(best_error):
        raise ValueError("轮廓无法建立有效正旋转配准")
    normalized_angle = (best_angle + 180.0) % 360.0 - 180.0
    return float(normalized_angle), best_error


def contour_shape_distance(source_vertices, target_vertices):
    """返回平移、旋转和统一尺度不敏感，但不允许镜像的轮廓匹配分数。"""
    _angle, normalized_error = align_closed_contours(
        source_vertices,
        target_vertices,
        sample_count=CONTOUR_SAMPLE_COUNT,
        normalize_scale=True,
    )
    return float(normalized_error * CONTOUR_SCORE_SCALE)


def _normalize_sequence(values):
    """将非负数序列按总和归一化，并转换为可写入 JSON 的 Python 浮点数。"""
    values = [float(value) for value in values]
    total = sum(values)
    if total <= 1e-12:
        return [0.0 for _ in values]
    return [value / total for value in values]


def _hu_log_values(contour):
    """计算对尺度和旋转不敏感的 Hu 矩对数值，压缩跨数量级差异。"""
    hu_values = cv2.HuMoments(cv2.moments(contour)).flatten()
    transformed = []
    for value in hu_values:
        value = float(value)
        if abs(value) <= 1e-30:
            transformed.append(0.0)
        else:
            transformed.append(-math.copysign(math.log10(abs(value)), value))
    return transformed


def build_shape_descriptor(piece):
    """构造对平移、缩放和旋转不敏感的碎片形状描述子。

    主要流程：归一化相邻边长与内角，计算 Hu 矩，并记录顶点数和无量纲紧致度。
    关键参数：piece 必须含 contour、edge_lengths、interior_angles、area 和 perimeter。
    返回值：只含 JSON 原生类型的字典，可直接保存或用于匹配。
    """
    perimeter = float(piece["perimeter"])
    compactness = 0.0
    if perimeter > 1e-12:
        compactness = float(piece["area"]) / (perimeter * perimeter)

    return {
        "vertex_count": int(piece["vertex_count"]),
        "edge_ratios": _normalize_sequence(piece["edge_lengths"]),
        "angle_ratios": [
            float(angle) / 180.0 for angle in piece["interior_angles"]
        ],
        "hu": _hu_log_values(piece["contour"]),
        "compactness": float(compactness),
    }


def register_templates(pieces):
    """按稳定几何顺序将一至四片已知碎片登记为 K1 至 K4。

    主要流程：先计算描述子，再按顶点数、紧致度和排序边长组成的键排序并依次编号。
    关键参数：pieces 为已经完成几何提取的碎片列表。
    返回值：包含 id 和形状描述子的模板列表，不保留不可序列化的原始轮廓。
    """
    if not 1 <= len(pieces) <= 4:
        raise ValueError("已知模板数量必须在 1 到 4 之间")

    descriptors = [build_shape_descriptor(piece) for piece in pieces]
    descriptors.sort(
        key=lambda item: (
            item["vertex_count"],
            round(item["compactness"], 9),
            tuple(round(value, 9) for value in sorted(item["edge_ratios"])),
        )
    )

    templates = []
    for index, descriptor in enumerate(descriptors, start=1):
        templates.append({"id": f"K{index}", **descriptor})
    return templates


def save_templates(path, templates):
    """使用临时文件和原子替换保存模板，避免断电产生半个 JSON。

    主要流程：写入同目录 `.tmp` 文件、刷新并同步磁盘，然后使用 os.replace 原子替换目标。
    关键参数：path 为目标 JSON 路径，templates 为 register_templates 的返回值。
    返回值：无；写入或替换失败时清理临时文件并重新抛出异常。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    payload = {"version": TEMPLATE_VERSION, "templates": templates}

    try:
        with open(temporary_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        # 原子替换前失败时删除临时文件，避免下次启动误认为它是有效模板。
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _validate_loaded_templates(payload):
    """校验模板文件顶层结构和必要字段，拒绝损坏或不兼容的数据。"""
    if not isinstance(payload, dict) or payload.get("version") != TEMPLATE_VERSION:
        raise ValueError("模板文件版本无效")
    templates = payload.get("templates")
    if not isinstance(templates, list) or len(templates) > 4:
        raise ValueError("模板列表无效")

    required_fields = {
        "id",
        "vertex_count",
        "edge_ratios",
        "angle_ratios",
        "hu",
        "compactness",
    }
    for template in templates:
        if not isinstance(template, dict) or not required_fields.issubset(template):
            raise ValueError("模板字段不完整")
    return templates


def load_templates(path):
    """读取并校验已知碎片模板，文件不存在时返回空列表。

    主要流程：判断文件是否存在、解析 JSON、检查版本和每个模板的必要字段。
    关键参数：path 为模板 JSON 路径。
    返回值：模板列表；不存在返回空列表，损坏文件抛出 ValueError。
    """
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return _validate_loaded_templates(payload)


def _cyclic_sequence_distance(first, second):
    """计算两个等长循环序列在正向或反向最佳对齐后的均方根误差。"""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if len(first_array) != len(second_array) or len(first_array) == 0:
        return 1.0

    best_distance = float("inf")
    for candidate in (second_array, second_array[::-1]):
        for shift in range(len(candidate)):
            aligned = np.roll(candidate, shift)
            distance = float(np.sqrt(np.mean((first_array - aligned) ** 2)))
            best_distance = min(best_distance, distance)
    return best_distance


def descriptor_distance(observation, template):
    """综合顶点数、循环边长、循环内角、紧致度和 Hu 矩计算形状距离。

    主要流程：顶点数不一致时先加入显著惩罚；其余连续特征使用归一化误差加权。
    关键参数：observation 和 template 都是 build_shape_descriptor 兼容字典。
    返回值：非负浮点距离，越小表示形状越接近。
    """
    vertex_penalty = abs(
        int(observation["vertex_count"]) - int(template["vertex_count"])
    )
    edge_distance = _cyclic_sequence_distance(
        observation["edge_ratios"], template["edge_ratios"]
    )
    angle_distance = _cyclic_sequence_distance(
        observation["angle_ratios"], template["angle_ratios"]
    )
    compactness_distance = abs(
        float(observation["compactness"]) - float(template["compactness"])
    )

    observation_hu = np.asarray(observation["hu"], dtype=np.float64)
    template_hu = np.asarray(template["hu"], dtype=np.float64)
    hu_denominator = 1.0 + np.abs(template_hu)
    hu_distance = float(np.mean(np.abs(observation_hu - template_hu) / hu_denominator))

    return float(
        vertex_penalty
        + 2.0 * edge_distance
        + 0.5 * angle_distance
        + compactness_distance
        + 0.02 * hu_distance
    )


def _candidate_assignments(piece_count, template_count):
    """生成不重复使用模板的全部小规模分配方案，缺少模板时以 -1 表示未知。"""
    if piece_count <= template_count:
        return itertools.permutations(range(template_count), piece_count)

    padded = list(range(template_count)) + [-1] * (piece_count - template_count)
    # 最多四片，使用集合去除重复 -1 造成的相同排列仍然足够轻量。
    return iter(set(itertools.permutations(padded, piece_count)))


def match_known_pieces(pieces, templates, max_score):
    """对不超过四片观测执行总代价最低的一对一已知模板匹配。

    主要流程：构造描述子和代价矩阵，穷举模板排列，选取总代价最低方案并应用拒识阈值。
    关键参数：pieces 为观测碎片，templates 为已登记模板，max_score 为单片最大接受距离。
    返回值：输入碎片列表本身；每项会更新 id 和 match_score，顺序保持不变。
    """
    if max_score < 0:
        raise ValueError("匹配阈值不能为负数")
    if len(pieces) > 4 or len(templates) > 4:
        raise ValueError("模板匹配最多支持四片")
    if not pieces:
        return pieces

    observations = [build_shape_descriptor(piece) for piece in pieces]
    if not templates:
        for piece in pieces:
            piece["id"] = "UNKNOWN"
            piece["match_score"] = float("inf")
        return pieces

    # 即使新版模板将使用target_vertices_mm轮廓快路，也必须先读取一次全部V1描述子。
    # 这样损坏JSON仍沿用既有异常边界，由主循环转换成可见TEMPLATE ERROR，而不会
    # 因快路绕过损坏字段后继续输出看似合法的K编号。
    validation_observation = observations[0]
    for template in templates:
        descriptor_distance(validation_observation, template)

    cost_matrix = []
    for piece, observation in zip(pieces, observations):
        piece_costs = []
        for template in templates:
            try:
                if "vertices_mm" in piece and "target_vertices_mm" in template:
                    # 新版KNOWN模板有目标毫米轮廓时优先使用固定点轮廓距离，避免
                    # approxPolyDP在4/5顶点间跳变触发整项顶点惩罚。
                    score = contour_shape_distance(
                        piece["vertices_mm"],
                        template["target_vertices_mm"],
                    )
                else:
                    # 旧模板或稳定版风格调用没有毫米目标，继续沿用V1描述子兼容路径。
                    score = descriptor_distance(observation, template)
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                score = float("inf")
            piece_costs.append(float(score))
        cost_matrix.append(piece_costs)

    best_assignment = None
    best_total = float("inf")
    for assignment in _candidate_assignments(len(pieces), len(templates)):
        total = 0.0
        for piece_index, template_index in enumerate(assignment):
            if template_index < 0:
                total += max_score + 1.0
            else:
                total += cost_matrix[piece_index][template_index]
        if total < best_total:
            best_assignment = assignment
            best_total = total

    for piece_index, piece in enumerate(pieces):
        template_index = best_assignment[piece_index]
        if template_index < 0:
            score = float("inf")
            piece_id = "UNKNOWN"
        else:
            score = float(cost_matrix[piece_index][template_index])
            piece_id = templates[template_index]["id"] if score <= max_score else "UNKNOWN"
        piece["id"] = piece_id
        piece["match_score"] = score

    return pieces
