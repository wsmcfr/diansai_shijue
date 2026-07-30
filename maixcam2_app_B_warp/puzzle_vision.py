"""黑底白色拼图碎片的 OpenCV 识别核心。"""

import cv2
import numpy as np

try:
    from maixcam2_app_B_warp.config import DEFAULT_CONFIG
except ModuleNotFoundError as error:
    # MaixVision运行工程时会把文件平铺到临时目录，此时改为导入同级config.py。
    if error.name != "maixcam2_app_B_warp":
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


def _normalize_valid_mask(valid_mask, roi):
    """校验B版可选有效掩膜并规范为0/255二值图。

    valid_mask 必须与ROI宽高一致；None返回全白矩形以保持旧行为。返回值为独立
    uint8数组，空掩膜和错误尺寸抛出 ValueError。
    """
    _roi_x, _roi_y, roi_width, roi_height = roi
    if valid_mask is None:
        return np.full((roi_height, roi_width), 255, dtype=np.uint8)
    mask_array = np.asarray(valid_mask)
    if mask_array.ndim != 2 or mask_array.shape != (roi_height, roi_width):
        raise ValueError("valid_mask 尺寸必须与ROI一致")
    normalized = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(normalized) <= 0:
        raise ValueError("valid_mask 不能是空掩膜")
    return normalized


def build_foreground_mask(frame_bgr, roi, config=None, valid_mask=None):
    """在指定工作区内分割黑色背景上的亮色碎片。

    主要流程：校验输入和 ROI，执行灰度化、高斯滤波、Otsu 阈值以及开闭运算。
    关键参数：frame_bgr 为 BGR 图像，roi 为 ``(x, y, 宽, 高)``，config 可覆盖默认参数。
    valid_mask 可进一步清除INSET边缘；返回值为二值掩膜和实际灰度阈值。
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
    normalized_valid_mask = _normalize_valid_mask(valid_mask, roi)

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

    # 小核开运算去除白色亮点；随后闭运算填补纸面纹理造成的细小黑孔。
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
    # 在形态学操作后应用INSET，避免闭运算把边缘外白色重新扩入机械有效区。
    mask = cv2.bitwise_and(mask, normalized_valid_mask)

    return mask, float(threshold)


def approximate_polygon(contour, config):
    """将像素轮廓拟合为题目允许的三至五边主要多边形。

    主要流程：从较小到较大的周长比例逐步增加拟合误差，优先返回首个三至五顶点结果。
    关键参数：contour 为 OpenCV 轮廓，config 提供误差范围、步长和顶点限制。
    返回值：形状为 ``(顶点数, 1, 2)`` 的 OpenCV 多边形轮廓。
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return contour.copy()

    epsilon = float(config["approx_epsilon_min"])
    epsilon_max = float(config["approx_epsilon_max"])
    epsilon_step = float(config["approx_epsilon_step"])
    min_vertices = int(config["min_vertices"])
    max_vertices = int(config["max_vertices"])
    if epsilon_step <= 0 or epsilon_max < epsilon:
        raise ValueError("多边形拟合误差范围或步长无效")

    best_candidate = contour.copy()
    best_distance = float("inf")
    while epsilon <= epsilon_max + 1e-12:
        candidate = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        vertex_count = len(candidate)

        if min_vertices <= vertex_count <= max_vertices:
            return candidate

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


def compute_piece_geometry(contour, roi, config, valid_mask=None):
    """计算单片碎片的顶点、中心、角度、边长、内角和完整性。

    主要流程：拟合多边形，用图像矩计算中心，规范化最小外接矩形角度，再计算相邻顶点特征。
    valid_mask 提供时按INSET安全内区判断完整性，否则保持旧矩形边界判断。
    返回值：可供未知编号、模板匹配和显示层直接使用的碎片字典。
    """
    polygon = approximate_polygon(contour, config)
    vertices_array = polygon.reshape(-1, 2).astype(np.float64)
    vertices = [(int(point[0]), int(point[1])) for point in vertices_array]

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

    roi_x, roi_y, roi_width, roi_height = roi
    margin = int(config["border_margin_px"])
    contour_points = contour.reshape(-1, 2)
    if valid_mask is None:
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
        kernel_size = max(1, margin * 2 + 1)
        safe_mask = cv2.erode(
            valid_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
        local_points = contour_points - np.int32([roi_x, roi_y])
        complete = all(
            0 <= int(point[0]) < roi_width
            and 0 <= int(point[1]) < roi_height
            and safe_mask[int(point[1]), int(point[0])] != 0
            for point in local_points
        )

    return {
        "contour": contour,
        "vertices": vertices,
        "center": center,
        "angle_deg": angle_deg,
        "area": float(cv2.contourArea(contour)),
        "perimeter": float(cv2.arcLength(contour, True)),
        "edge_lengths": edge_lengths,
        "interior_angles": interior_angles,
        "vertex_count": vertex_count,
        "complete": bool(complete),
    }


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


def detect_pieces(frame_bgr, roi, config=None, valid_mask=None):
    """提取工作区内的有效碎片外轮廓。

    主要流程：生成前景掩膜、提取最外层轮廓、按面积比例过滤并最多保留四片。
    关键参数：frame_bgr 和 roi 与 ``build_foreground_mask`` 一致，config 可覆盖默认阈值。
    valid_mask 用于B版INSET；返回轮廓坐标位于工作图坐标系。
    """
    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    normalized_valid_mask = _normalize_valid_mask(valid_mask, roi)
    mask, threshold = build_foreground_mask(
        frame_bgr,
        roi,
        merged_config,
        valid_mask=normalized_valid_mask,
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    x, y, width, height = roi
    active_area = float(np.count_nonzero(normalized_valid_mask))
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
                valid_mask=(normalized_valid_mask if valid_mask is not None else None),
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
    white_ratio = float(np.count_nonzero(mask)) / active_area

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
