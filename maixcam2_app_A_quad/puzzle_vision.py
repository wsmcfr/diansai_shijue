"""黑底白色拼图碎片的 OpenCV 识别核心。"""

import cv2
import numpy as np

try:
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
except ModuleNotFoundError as error:
    # MaixVision运行工程时会把文件平铺到临时目录，此时改为导入同级config.py。
    if error.name != "maixcam2_app_A_quad":
        raise
    from config import DEFAULT_CONFIG


class DetectionResult:
    """保存单帧检测结果。

    主要内容：有效碎片列表、工作区二值掩膜、自动阈值和本次使用的 ROI。
    该结构只承载数据，不依赖 MaixPy，便于 PC 测试与实拍图回放。
    """

    def __init__(
        self,
        pieces,
        mask,
        threshold,
        roi,
        small_contours=None,
        large_contours=None,
        edge_contours=None,
        valid_contour_count=0,
        white_ratio=0.0,
        active_area=None,
    ):
        """初始化单帧结果和调参诊断数据。

        主要流程：保存业务使用的最多四片结果，同时保留面积过小、面积过大和接触边界
        的轮廓集合，供调参界面解释漏检原因。
        关键参数：轮廓列表均使用原始相机坐标，valid_contour_count 为截断前有效数量。
        返回值：构造函数无返回值；所有输入会保存为同名公开属性。
        """
        self.pieces = pieces
        self.mask = mask
        self.threshold = threshold
        self.roi = roi
        self.small_contours = list(small_contours or [])
        self.large_contours = list(large_contours or [])
        self.edge_contours = list(edge_contours or [])
        self.valid_contour_count = int(valid_contour_count)
        self.white_ratio = float(white_ratio)
        self.active_area = (
            float(roi[2] * roi[3]) if active_area is None else float(active_area)
        )


def _normalize_active_quad(active_quad, roi, frame_size):
    """校验可选的全局有效四边形并返回4×2 float32数组。

    主要流程：允许 None 保留旧矩形行为；否则检查形状、有限值、凸性、面积以及是否
    完整落在相机画面和传入ROI中。返回值：None或独立数组；非法输入抛出 ValueError。
    """
    if active_quad is None:
        return None
    quad = np.asarray(active_quad, dtype=np.float32)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        raise ValueError("active_quad 必须包含四个有限二维角点")
    contour = quad.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour.astype(np.int32)):
        raise ValueError("active_quad 必须是凸四边形")
    area = abs(float(cv2.contourArea(contour)))
    if area <= 1.0:
        raise ValueError("active_quad 面积过小或已经退化")

    frame_width, frame_height = frame_size
    if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= frame_width):
        raise ValueError("active_quad 必须完整位于相机画面内部")
    if np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= frame_height):
        raise ValueError("active_quad 必须完整位于相机画面内部")

    roi_x, roi_y, roi_width, roi_height = roi
    if np.any(quad[:, 0] < roi_x) or np.any(quad[:, 0] > roi_x + roi_width - 1):
        raise ValueError("active_quad 必须完整位于ROI内部")
    if np.any(quad[:, 1] < roi_y) or np.any(quad[:, 1] > roi_y + roi_height - 1):
        raise ValueError("active_quad 必须完整位于ROI内部")
    return quad.copy()


def _build_active_mask(roi, active_quad):
    """在ROI局部坐标中生成矩形全白或四边形有效像素掩膜。"""
    roi_x, roi_y, roi_width, roi_height = roi
    if active_quad is None:
        return np.full((roi_height, roi_width), 255, dtype=np.uint8)
    local_quad = np.rint(
        active_quad - np.float32([roi_x, roi_y])
    ).astype(np.int32)
    active_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    cv2.fillConvexPoly(active_mask, local_quad, 255)
    return active_mask


def _fill_internal_mask_holes(mask):
    """填充白色碎片内部的黑色孔洞，同时保持碎片外边界和相邻黑缝不变。

    主要流程：先给掩膜增加一圈确定的黑色边界，再从边界背景执行泛洪；泛洪无法
    到达的黑色区域才是真正封闭的牌面纹理孔洞。最终只把这些孔洞置白，不使用膨胀
    或闭运算，因此不会跨越两片之间的黑色间隙，也不会填平与背景相通的凹多边形。
    关键参数：mask必须是二维uint8二值图。返回值为独立的同尺寸uint8掩膜。
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("mask必须是二维uint8掩膜")

    # 外加一圈背景保证(0,0)始终是可泛洪的黑色像素，即使碎片接触原ROI边缘。
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded_background = padded.copy()
    flood_mask = np.zeros(
        (padded.shape[0] + 2, padded.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flooded_background, flood_mask, (0, 0), 255)

    # 泛洪结果取反后只剩封闭孔洞；与原前景合并不会改变任何已有白色外边界。
    enclosed_holes = cv2.bitwise_not(flooded_background)
    filled = cv2.bitwise_or(padded, enclosed_holes)
    return filled[1:-1, 1:-1].copy()


def build_foreground_mask(frame_bgr, roi, config=None, active_quad=None):
    """在指定工作区内分割黑色背景上的亮色碎片。

    主要流程：校验输入和 ROI，执行灰度化、高斯滤波、Otsu 阈值以及开闭运算。
    关键参数：frame_bgr 为 BGR 图像，roi 为 ``(x, y, 宽, 高)``，config 可覆盖默认参数。
    active_quad 为可选全局四角；提供时其外部像素在形态学处理后强制清零。
    返回值：ROI 内的二值掩膜和 Otsu 自动计算的灰度阈值。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr 必须是有效的 numpy 图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr 必须是三通道 BGR 图像")

    x, y, width, height = roi
    frame_height, frame_width = frame_bgr.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("ROI 宽高必须大于零")
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("ROI 必须完整位于输入图像内部")
    normalized_active_quad = _normalize_active_quad(
        active_quad,
        roi,
        (frame_width, frame_height),
    )

    # 复制配置后再覆盖，避免现场传入局部参数时意外修改全局默认值。
    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    for key in ("gaussian_kernel", "open_kernel", "close_kernel"):
        kernel_size = int(merged_config[key])
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"{key} 必须是正奇数")

    roi_frame = frame_bgr[y : y + height, x : x + width]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (merged_config["gaussian_kernel"], merged_config["gaussian_kernel"]),
        0,
    )
    fixed_threshold = merged_config.get("fixed_threshold")
    if fixed_threshold is None:
        threshold, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        fixed_threshold = float(fixed_threshold)
        if not 0.0 <= fixed_threshold <= 255.0:
            raise ValueError("fixed_threshold 必须位于 0 到 255 之间")
        # 固定阈值便于相机曝光和照明稳定后锁定现场参数，避免 Otsu 随画面占比波动。
        threshold, mask = cv2.threshold(
            gray,
            fixed_threshold,
            255,
            cv2.THRESH_BINARY,
        )

    # 小核开运算去除白色亮点；闭运算默认使用1x1核，防止跨过2mm物理黑缝。
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (merged_config["open_kernel"], merged_config["open_kernel"]),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (merged_config["close_kernel"], merged_config["close_kernel"]),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    # 先清零四边形外部，确保亮地面不会把整块黑纸包围成一个“内部孔洞”。
    active_mask = _build_active_mask(roi, normalized_active_quad)
    mask = cv2.bitwise_and(mask, active_mask)
    if bool(merged_config.get("fill_internal_holes", True)):
        # 纹理孔洞必须通过拓扑填孔处理，不能依赖会扩大碎片外轮廓的闭运算。
        mask = _fill_internal_mask_holes(mask)
    # 填孔后再次应用有效区，作为纸外像素绝不进入轮廓的最终防线。
    mask = cv2.bitwise_and(mask, active_mask)

    return mask, float(threshold)


def _polygon_approximation_parameters(config):
    """读取并校验多边形拟合参数。

    主要流程：把配置中的epsilon范围、步长和顶点范围转换为明确数值，并验证步长、
    上下界和顶点数关系。关键参数config与检测主配置共用。返回值为按调用顺序排列的
    五元组；非法设置抛出ValueError，避免视觉循环进入无法终止的epsilon遍历。
    """
    epsilon = float(config["approx_epsilon_min"])
    epsilon_max = float(config["approx_epsilon_max"])
    epsilon_step = float(config["approx_epsilon_step"])
    min_vertices = int(config["min_vertices"])
    max_vertices = int(config["max_vertices"])
    if (
        epsilon < 0.0
        or epsilon_step <= 0.0
        or epsilon_max < epsilon
        or min_vertices < 3
        or max_vertices < min_vertices
    ):
        raise ValueError("多边形拟合误差范围或步长无效")
    return epsilon, epsilon_max, epsilon_step, min_vertices, max_vertices


def _polygon_candidate_key(candidate):
    """生成与循环起点和顺逆方向无关的整数多边形去重键。

    OpenCV在相邻epsilon上经常返回同一组顶点。这里枚举正向、反向的所有循环移位并
    选择字典序最小者，确保同一候选只保留第一次出现的低epsilon版本。
    """
    points = np.asarray(candidate, dtype=np.int64).reshape(-1, 2)
    point_tuples = tuple((int(point[0]), int(point[1])) for point in points)
    variants = []
    for ordered in (point_tuples, tuple(reversed(point_tuples))):
        for offset in range(len(ordered)):
            variants.append(ordered[offset:] + ordered[:offset])
    return min(variants)


def _polygon_candidate_boundary_distance(first_candidate, second_candidate):
    """计算两个像素候选闭合边界之间的对称最大距离。

    主要流程：把两组候选转换为OpenCV浮点轮廓，逐顶点计算到另一条闭合边界的
    有符号距离并取绝对值，最后取双向最大值。参数必须是至少三点的像素多边形；
    返回值单位为像素，用于合并相邻epsilon产生的轻微坐标抖动候选。
    """
    first = np.asarray(first_candidate, dtype=np.float32).reshape(-1, 1, 2)
    second = np.asarray(second_candidate, dtype=np.float32).reshape(-1, 1, 2)

    def directed_distance(source, target):
        """返回source全部顶点到target闭合边界的最大绝对距离。"""
        return max(
            abs(
                float(
                    cv2.pointPolygonTest(
                        target,
                        (float(point[0]), float(point[1])),
                        True,
                    )
                )
            )
            for point in source.reshape(-1, 2)
        )

    return max(
        directed_distance(first, second),
        directed_distance(second, first),
    )


def approximate_polygon_candidates(contour, config):
    """收集像素轮廓在全部epsilon下得到的三至五边候选。

    主要流程：完整遍历配置的拟合误差范围，保留顶点数符合题目约束的简单候选，并按
    循环起点无关的键去重。关键参数contour为OpenCV闭合轮廓，config提供epsilon与
    顶点范围。返回值为独立OpenCV轮廓列表，按epsilon从小到大排列；没有候选时为空。
    """
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return []

    epsilon, epsilon_max, epsilon_step, min_vertices, max_vertices = (
        _polygon_approximation_parameters(config)
    )
    candidates = []
    seen = set()
    # 一像素是远景轮廓常见的量化抖动；较大碎片按周长给少量比例余量。只比较相同
    # 顶点数，避免把“带伪角的五边形”和应保留的四边形错误聚为一类。
    duplicate_distance_px = max(1.0, perimeter * 0.0025)
    while epsilon <= epsilon_max + 1e-12:
        candidate = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if min_vertices <= len(candidate) <= max_vertices:
            key = _polygon_candidate_key(candidate)
            is_near_duplicate = any(
                len(existing) == len(candidate)
                and _polygon_candidate_boundary_distance(candidate, existing)
                <= duplicate_distance_px
                for existing in candidates
            )
            if key not in seen and not is_near_duplicate:
                seen.add(key)
                candidates.append(candidate.copy())
        epsilon += epsilon_step
    return candidates


def approximate_polygon(contour, config):
    """返回兼容显示和KNOWN描述子的首个三至五边像素多边形。

    UNKNOWN会另外读取`approximate_polygon_candidates()`的全部候选；本函数继续返回
    最小epsilon下的首个合法候选，避免改变既有显示、编号和KNOWN模板行为。没有合法
    候选时，返回顶点数最接近配置范围的异常显示轮廓。
    """
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return contour.copy()

    candidates = approximate_polygon_candidates(contour, config)
    if candidates:
        return candidates[0].copy()

    epsilon, epsilon_max, epsilon_step, min_vertices, max_vertices = (
        _polygon_approximation_parameters(config)
    )

    best_candidate = contour.copy()
    best_distance = float("inf")
    while epsilon <= epsilon_max + 1e-12:
        candidate = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        vertex_count = len(candidate)

        # 没有候选落入目标范围时，保留顶点数最接近三至五的结果用于异常显示。
        if vertex_count < min_vertices:
            distance = min_vertices - vertex_count
        else:
            distance = vertex_count - max_vertices
        if distance < best_distance:
            best_candidate = candidate
            best_distance = distance

        epsilon += epsilon_step

    return best_candidate


def compute_piece_geometry(contour, roi, config, active_quad=None):
    """计算单片碎片的顶点、中心、角度、边长、内角和完整性。

    主要流程：拟合多边形，用图像矩计算中心，规范化最小外接矩形角度，再计算相邻顶点特征。
    关键参数：contour 使用原图坐标；active_quad 提供时按斜边真实距离判断完整性。
    返回值：可供未知编号、模板匹配和显示层直接使用的碎片字典。
    """
    polygon_candidates = approximate_polygon_candidates(contour, config)
    polygon = (
        polygon_candidates[0].copy()
        if polygon_candidates
        else approximate_polygon(contour, config)
    )
    vertices_array = polygon.reshape(-1, 2).astype(np.float64)
    vertices = [(int(point[0]), int(point[1])) for point in vertices_array]
    # 候选使用独立Python列表保存，后续毫米评分和排序不能原地改变显示主轮廓。
    shape_hypotheses_px = [
        [
            (int(point[0]), int(point[1]))
            for point in candidate.reshape(-1, 2)
        ]
        for candidate in polygon_candidates
    ]

    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        center = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    else:
        # 极小或退化轮廓没有有效面积，此时用外接矩形中心保证显示层仍有可用坐标。
        center = tuple(float(value) for value in cv2.minAreaRect(contour)[0])

    edge_lengths = []
    interior_angles = []
    vertex_count = len(vertices_array)
    for index in range(vertex_count):
        current = vertices_array[index]
        next_point = vertices_array[(index + 1) % vertex_count]
        previous_point = vertices_array[(index - 1) % vertex_count]

        edge_lengths.append(float(np.linalg.norm(next_point - current)))

        previous_vector = previous_point - current
        next_vector = next_point - current
        denominator = float(np.linalg.norm(previous_vector) * np.linalg.norm(next_vector))
        if denominator <= 1e-9:
            interior_angles.append(0.0)
        else:
            # 点积可能因浮点误差略超出 [-1, 1]，夹紧后再计算反余弦可避免 NaN。
            cosine = float(np.dot(previous_vector, next_vector) / denominator)
            cosine = max(-1.0, min(1.0, cosine))
            interior_angles.append(float(np.degrees(np.arccos(cosine))))

    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rect[1]
    angle_deg = float(rect[2])
    if rect_width < rect_height:
        angle_deg += 90.0
    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0

    margin = int(config["border_margin_px"])
    contour_points = contour.reshape(-1, 2)
    if active_quad is None:
        roi_x, roi_y, roi_width, roi_height = roi
        min_x = int(np.min(contour_points[:, 0]))
        max_x = int(np.max(contour_points[:, 0]))
        min_y = int(np.min(contour_points[:, 1]))
        max_y = int(np.max(contour_points[:, 1]))
        complete = not (
            min_x <= roi_x + margin
            or min_y <= roi_y + margin
            or max_x >= roi_x + roi_width - 1 - margin
            or max_y >= roi_y + roi_height - 1 - margin
        )
    else:
        polygon = np.asarray(active_quad, dtype=np.float32).reshape(-1, 1, 2)
        # pointPolygonTest的正距离表示在内部；距离不大于边缘余量即视为接触斜边。
        distances = [
            cv2.pointPolygonTest(
                polygon,
                (float(point[0]), float(point[1])),
                True,
            )
            for point in contour_points
        ]
        complete = bool(distances) and min(distances) > margin

    return {
        "contour": contour,
        "vertices": vertices,
        "shape_hypotheses_px": shape_hypotheses_px,
        "center": center,
        "angle_deg": angle_deg,
        "area": float(cv2.contourArea(contour)),
        "perimeter": float(cv2.arcLength(contour, True)),
        "edge_lengths": edge_lengths,
        "interior_angles": interior_angles,
        "vertex_count": vertex_count,
        "complete": bool(complete),
    }


def sample_piece_edge_features(
    frame_bgr,
    vertices,
    sample_count=16,
    inward_offsets_px=(2.0, 4.0),
):
    """沿每条多边形边的内侧采样BGR颜色、切向梯度和纹理能量。

    主要流程：根据顶点均值选择指向碎片内部的单位法线，在每条边10%～90%位置
    进行多层内缩采样并平均，随后计算灰度切向梯度。关键参数vertices使用相机像素
    坐标，sample_count至少4。返回值：与边索引一一对应的JSON兼容特征字典列表。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效numpy图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr必须是三通道BGR图像")
    vertices_array = np.asarray(vertices, dtype=np.float64)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 2 or len(vertices_array) < 3:
        raise ValueError("vertices必须包含至少三个二维顶点")
    if not np.all(np.isfinite(vertices_array)):
        raise ValueError("vertices必须包含有限坐标")
    sample_count = int(sample_count)
    if sample_count < 4:
        raise ValueError("sample_count必须至少为4")
    offsets = tuple(float(value) for value in inward_offsets_px)
    if not offsets or any(value <= 0.0 for value in offsets):
        raise ValueError("inward_offsets_px必须包含正数")

    frame_height, frame_width = frame_bgr.shape[:2]
    polygon_center = np.mean(vertices_array, axis=0)
    interpolation = np.linspace(0.10, 0.90, sample_count, dtype=np.float64)
    features = []
    for edge_index in range(len(vertices_array)):
        start = vertices_array[edge_index]
        end = vertices_array[(edge_index + 1) % len(vertices_array)]
        edge_vector = end - start
        edge_length = float(np.linalg.norm(edge_vector))
        if edge_length <= 1e-6:
            raise ValueError("多边形不能包含零长度边")
        normal = np.asarray((-edge_vector[1], edge_vector[0]), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        midpoint = (start + end) * 0.5
        if float(np.dot(polygon_center - midpoint, normal)) < 0.0:
            normal = -normal

        base_points = start + interpolation[:, None] * edge_vector
        sampled_layers = []
        for offset in offsets:
            sample_points = base_points + normal * offset
            sample_x = np.clip(np.rint(sample_points[:, 0]).astype(np.int32), 0, frame_width - 1)
            sample_y = np.clip(np.rint(sample_points[:, 1]).astype(np.int32), 0, frame_height - 1)
            sampled_layers.append(frame_bgr[sample_y, sample_x].astype(np.float64))
        colors = np.mean(np.stack(sampled_layers, axis=0), axis=0)
        gray = (
            0.114 * colors[:, 0]
            + 0.587 * colors[:, 1]
            + 0.299 * colors[:, 2]
        )
        gradients = np.gradient(gray)
        # 颜色标准差与切向梯度共同表示牌面纹理；纯白片两项都接近零。
        pattern_energy = min(
            1.0,
            float(np.mean(np.std(colors, axis=0)) / 64.0)
            + float(np.mean(np.abs(gradients)) / 64.0),
        )
        features.append(
            {
                "colors": colors.astype(float).tolist(),
                "gradients": gradients.astype(float).tolist(),
                "pattern_energy": float(pattern_energy),
            }
        )
    return features


def select_actionable_pieces(pieces):
    """筛选可用于编号、模板匹配和后续机械控制的完整碎片。

    主要流程：保留 complete 明确为 True 的碎片，维持原始相对顺序且不复制碎片字典。
    关键参数：pieces 为 detect_pieces 返回的碎片列表。
    返回值：新的列表；接触 ROI 边界的不完整轮廓不会进入可操作结果。
    """
    return [piece for piece in pieces if piece.get("complete") is True]


def assign_unknown_ids(pieces, row_tolerance_px):
    """按从上到下、同一行从左到右的顺序编号未知碎片。

    主要流程：先按中心 Y 排序并依据行容差动态分行，再按每行中心 X 排序并原地重排列表。
    关键参数：pieces 必须包含 center，row_tolerance_px 决定两个中心是否属于同一行。
    返回值：原始列表本身；列表顺序和每项的 id 字段都会被更新为 U1 至 U4。
    """
    if row_tolerance_px < 0:
        raise ValueError("行容差不能为负数")

    rows = []
    for piece in sorted(pieces, key=lambda item: (item["center"][1], item["center"][0])):
        center_y = float(piece["center"][1])
        if not rows or abs(center_y - rows[-1]["mean_y"]) > row_tolerance_px:
            # 新行保存当前均值，后续加入同一行时持续更新，避免首个点偏置分组结果。
            rows.append({"mean_y": center_y, "pieces": [piece]})
            continue

        current_row = rows[-1]
        current_row["pieces"].append(piece)
        current_row["mean_y"] = sum(
            float(item["center"][1]) for item in current_row["pieces"]
        ) / len(current_row["pieces"])

    ordered_pieces = []
    for row in rows:
        ordered_pieces.extend(
            sorted(row["pieces"], key=lambda item: float(item["center"][0]))
        )

    pieces[:] = ordered_pieces
    for index, piece in enumerate(pieces, start=1):
        piece["id"] = f"U{index}"

    return pieces


def detect_pieces(frame_bgr, roi, config=None, active_quad=None):
    """提取工作区内的有效碎片外轮廓。

    主要流程：生成前景掩膜、提取最外层轮廓、按面积比例过滤并最多保留四片。
    关键参数：frame_bgr 和 roi 与 ``build_foreground_mask`` 一致，config 可覆盖默认阈值。
    active_quad 提供时只统计四边形内部；返回轮廓仍转换回原图坐标系。
    """
    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    frame_height, frame_width = frame_bgr.shape[:2]
    normalized_active_quad = _normalize_active_quad(
        active_quad,
        roi,
        (frame_width, frame_height),
    )
    mask, threshold = build_foreground_mask(
        frame_bgr,
        roi,
        merged_config,
        active_quad=normalized_active_quad,
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    x, y, width, height = roi
    active_area = (
        float(width * height)
        if normalized_active_quad is None
        else abs(float(cv2.contourArea(normalized_active_quad)))
    )
    min_area = active_area * float(merged_config["min_area_ratio"])
    max_area = active_area * float(merged_config["max_area_ratio"])

    # 先保留完整分类结果，调参界面才能区分“阈值没分出来”和“面积过滤掉了”。
    small_contours = []
    valid_contours = []
    large_contours = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            small_contours.append(contour)
        elif contour_area > max_area:
            large_contours.append(contour)
        else:
            valid_contours.append(contour)

    # 最多四片的限制只作用于业务结果，截断前数量保留为噪声诊断依据。
    valid_contour_count = len(valid_contours)
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    valid_contours = valid_contours[: int(merged_config["max_pieces"])]

    pieces = []
    offset = np.array([[[x, y]]], dtype=np.int32)
    for contour in valid_contours:
        # findContours 返回 ROI 内局部坐标，统一平移到原始相机图像坐标。
        global_contour = contour.astype(np.int32) + offset
        pieces.append(
            compute_piece_geometry(
                global_contour,
                roi,
                merged_config,
                active_quad=normalized_active_quad,
            )
        )

    # 被面积过滤的轮廓也转换回相机坐标，保证RESULT页面可以直接叠加绘制。
    global_small_contours = [
        contour.astype(np.int32) + offset for contour in small_contours
    ]
    global_large_contours = [
        contour.astype(np.int32) + offset for contour in large_contours
    ]
    edge_contours = [
        piece["contour"] for piece in pieces if piece.get("complete") is False
    ]
    active_mask = _build_active_mask(roi, normalized_active_quad)
    valid_pixel_count = int(np.count_nonzero(active_mask))
    white_ratio = (
        0.0
        if valid_pixel_count <= 0
        else float(np.count_nonzero(mask)) / float(valid_pixel_count)
    )

    return DetectionResult(
        pieces,
        mask,
        threshold,
        roi,
        small_contours=global_small_contours,
        large_contours=global_large_contours,
        edge_contours=edge_contours,
        valid_contour_count=valid_contour_count,
        white_ratio=white_ratio,
        active_area=active_area,
    )
