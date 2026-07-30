"""从固定相机画面中单次定位完整黑色A4纸。"""

import cv2
import numpy as np

try:
    from maixcam2_app_B_warp.config import DEFAULT_CONFIG
except ModuleNotFoundError as error:
    # MaixVision会把发布包平铺到临时目录，只有包本身不存在时才回退同级导入。
    if error.name != "maixcam2_app_B_warp":
        raise
    from config import DEFAULT_CONFIG


# 完整A4和龙门架机械覆盖范围都使用毫米，两个变体共享同一物理定义。
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
WORK_HEIGHT_MM = 230.0
WORK_TRIM_MM = (A4_HEIGHT_MM - WORK_HEIGHT_MM) / 2.0
MAX_INSET_MM = 20.0


class PaperLocation:
    """保存一次黑纸定位的完整结果和失败原因。

    成功结果包含按左上、右上、右下、左下排序的完整A4四角；失败结果强制把
    ``paper_quad`` 和 ``active_quad`` 设为 None，避免调用方误锁定半有效候选。
    """

    def __init__(
        self,
        success,
        paper_quad=None,
        active_quad=None,
        confidence=0.0,
        threshold=0.0,
        reason="",
    ):
        """初始化定位结果。

        关键参数：confidence 位于0～1；threshold 为本次Otsu阈值；reason 用于屏幕状态。
        返回值：构造函数无返回值，输入会规范化后保存为公开属性。
        """
        self.success = bool(success)
        self.paper_quad = (
            None if paper_quad is None else np.asarray(paper_quad, dtype=np.float32)
        )
        self.active_quad = (
            None if active_quad is None else np.asarray(active_quad, dtype=np.float32)
        )
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.threshold = float(threshold)
        self.reason = str(reason)

    @classmethod
    def failed(cls, reason, threshold=0.0, confidence=0.0):
        """构造不携带任何四角的失败结果，供检测和UI回退分支统一使用。"""
        return cls(
            False,
            paper_quad=None,
            active_quad=None,
            confidence=confidence,
            threshold=threshold,
            reason=reason,
        )


def _validate_frame(frame_bgr):
    """校验定位输入必须是非空三通道BGR图像，非法输入直接抛出 ValueError。"""
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr 必须是有效的 numpy 图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr 必须是三通道 BGR 图像")
    if frame_bgr.shape[0] < 8 or frame_bgr.shape[1] < 8:
        raise ValueError("frame_bgr 尺寸过小，无法定位A4纸")


def order_a4_quad(points):
    """把四个无序角点规范为左上、右上、右下、左下。

    主要流程：检查有限值和重复点，按点相对中心的极角形成连续凸轮廓，再把左上角
    旋转到首位。关键参数 points 必须可转换为4×2浮点数组。
    返回值：4×2 float32 数组；退化、重复或非凸输入抛出 ValueError。
    """
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        raise ValueError("A4四边形必须包含四个有限二维角点")
    if len(np.unique(quad, axis=0)) != 4:
        raise ValueError("A4四边形不能包含重复角点")

    center = np.mean(quad, axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles)]
    start_index = int(np.argmin(np.sum(ordered, axis=1)))
    ordered = np.roll(ordered, -start_index, axis=0).astype(np.float32)

    contour = ordered.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour.astype(np.int32)):
        raise ValueError("A4四边形必须是凸四边形")
    if abs(float(cv2.contourArea(contour))) <= 1.0:
        raise ValueError("A4四边形面积过小或已经退化")
    return ordered


def _physical_to_image_homography(paper_quad):
    """构造完整A4毫米平面到相机四边形的单应性矩阵。

    主要流程：规范四角顺序，建立210×297毫米标准平面并求透视矩阵，再检查矩阵
    是否有限和可逆。返回值：``(矩阵, 有序四角)``；退化输入抛出 ValueError。
    """
    ordered_quad = order_a4_quad(paper_quad)
    physical_quad = np.float32(
        [[0, 0], [A4_WIDTH_MM, 0], [A4_WIDTH_MM, A4_HEIGHT_MM], [0, A4_HEIGHT_MM]]
    )
    matrix = cv2.getPerspectiveTransform(physical_quad, ordered_quad)
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-9:
        raise ValueError("A4四边形无法建立有效单应性矩阵")
    return matrix, ordered_quad


def build_active_quad(paper_quad, inset_mm=0.0):
    """由完整A4四角生成210×230mm机械有效四边形。

    主要流程：长边上下各裁33.5mm，再把左、右、上、下四边整体内缩 inset_mm，
    最后通过完整A4单应性映射回相机坐标。inset_mm 允许0～20mm。
    返回值：按左上、右上、右下、左下排序的4×2 float32相机坐标。
    """
    try:
        inset_mm = float(inset_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("inset_mm 必须是0到20之间的数字") from error
    if not 0.0 <= inset_mm <= MAX_INSET_MM:
        raise ValueError("inset_mm 必须位于0到20之间")

    matrix, _ = _physical_to_image_homography(paper_quad)
    active_physical = np.float32(
        [
            [inset_mm, WORK_TRIM_MM + inset_mm],
            [A4_WIDTH_MM - inset_mm, WORK_TRIM_MM + inset_mm],
            [A4_WIDTH_MM - inset_mm, A4_HEIGHT_MM - WORK_TRIM_MM - inset_mm],
            [inset_mm, A4_HEIGHT_MM - WORK_TRIM_MM - inset_mm],
        ]
    ).reshape(1, 4, 2)
    active_quad = cv2.perspectiveTransform(active_physical, matrix)[0]
    if not np.all(np.isfinite(active_quad)):
        raise ValueError("A4四边形映射出的机械有效区无效")
    return active_quad.astype(np.float32)


def image_point_to_paper_mm(point, paper_quad):
    """把原相机像素点反算为完整A4的毫米坐标。

    关键参数：point 为 ``(x, y)`` 相机坐标；paper_quad 为已锁定完整A4四角。
    返回值：``(x_mm, y_mm)`` 浮点元组，原点位于完整A4左上角。
    """
    _, ordered_quad = _physical_to_image_homography(paper_quad)
    physical_quad = np.float32(
        [[0, 0], [A4_WIDTH_MM, 0], [A4_WIDTH_MM, A4_HEIGHT_MM], [0, A4_HEIGHT_MM]]
    )
    inverse_matrix = cv2.getPerspectiveTransform(ordered_quad, physical_quad)
    point_array = np.asarray(point, dtype=np.float32)
    if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
        raise ValueError("point 必须包含两个有限坐标")
    mapped = cv2.perspectiveTransform(point_array.reshape(1, 1, 2), inverse_matrix)[0, 0]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("相机点无法映射到A4毫米坐标")
    return float(mapped[0]), float(mapped[1])


def _candidate_quad(hull):
    """从单个暗色凸包拟合四边形，无法形成稳定四角时返回 None。"""
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1.0:
        return None
    approximation = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(approximation) != 4:
        return None
    try:
        return order_a4_quad(approximation.reshape(4, 2))
    except ValueError:
        return None


def _score_candidate(contour, hull, quad, gray, area_ratio, config):
    """按A4比例、矩形度、凸性、面积与内部暗度计算0～1候选分数。

    权重设计：A4长宽比最能排除龙门架细杆，权重0.42；矩形度0.18；凸性0.10；
    可见面积0.10；纸内暗度0.20。内部白色碎片只降低暗度项，不改变外轮廓。
    返回值：``(总分, 矩形度)``，供调用方同时执行硬门槛和候选排序。
    """
    edge_lengths = [
        float(np.linalg.norm(quad[(index + 1) % 4] - quad[index]))
        for index in range(4)
    ]
    opposite_pair_a = (edge_lengths[0] + edge_lengths[2]) * 0.5
    opposite_pair_b = (edge_lengths[1] + edge_lengths[3]) * 0.5
    long_side = max(opposite_pair_a, opposite_pair_b)
    short_side = min(opposite_pair_a, opposite_pair_b)
    if short_side <= 1.0 or long_side <= 1.0:
        return 0.0, 0.0

    observed_aspect = short_side / long_side
    expected_aspect = float(config["paper_expected_aspect"])
    aspect_score = max(0.0, 1.0 - abs(observed_aspect - expected_aspect) / 0.25)

    min_rect = cv2.minAreaRect(hull)
    rect_area = float(min_rect[1][0] * min_rect[1][1])
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull))
    rectangularity = 0.0 if rect_area <= 1.0 else min(1.0, contour_area / rect_area)
    convexity = 0.0 if hull_area <= 1.0 else min(1.0, contour_area / hull_area)

    min_area_ratio = float(config["paper_min_area_ratio"])
    # 面积达到画面8%后给满分，小尺寸纸仍可凭其它几项通过，不强制依赖固定拍摄距离。
    area_score = float(
        np.clip((area_ratio - min_area_ratio) / max(0.08 - min_area_ratio, 1e-6), 0, 1)
    )

    interior_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(interior_mask, quad.round().astype(np.int32), 255)
    mean_gray = float(cv2.mean(gray, mask=interior_mask)[0])
    darkness_score = float(np.clip(1.0 - mean_gray / 255.0, 0.0, 1.0))

    confidence = (
        0.42 * aspect_score
        + 0.18 * rectangularity
        + 0.10 * convexity
        + 0.10 * area_score
        + 0.20 * darkness_score
    )
    return float(np.clip(confidence, 0.0, 1.0)), rectangularity


def locate_black_paper(frame_bgr, config=None):
    """在整帧中定位最符合黑色A4纸的四边形候选。

    主要流程：灰度与反向Otsu分割、闭运算、外轮廓提取、凸包四角拟合和加权评分。
    关键参数：config 可覆盖 DEFAULT_CONFIG 中 ``paper_*`` 参数。
    返回值：成功时返回 PaperLocation 四角与置信度；失败时返回不含四角的失败对象。
    """
    _validate_frame(frame_bgr)
    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    close_kernel_size = int(merged_config["paper_close_kernel"])
    if close_kernel_size <= 0 or close_kernel_size % 2 == 0:
        raise ValueError("paper_close_kernel 必须是正奇数")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, dark_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (close_kernel_size, close_kernel_size),
    )
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
    min_area_ratio = float(merged_config["paper_min_area_ratio"])
    max_area_ratio = float(merged_config["paper_max_area_ratio"])
    best_quad = None
    best_confidence = 0.0

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        area_ratio = contour_area / frame_area
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            continue

        hull = cv2.convexHull(contour)
        quad = _candidate_quad(hull)
        if quad is None:
            continue
        confidence, rectangularity = _score_candidate(
            contour,
            hull,
            quad,
            gray,
            area_ratio,
            merged_config,
        )
        if rectangularity < float(merged_config["paper_min_rectangularity"]):
            continue
        if confidence > best_confidence:
            best_quad = quad
            best_confidence = confidence

    min_confidence = float(merged_config["paper_min_confidence"])
    if best_quad is None:
        return PaperLocation.failed("no_candidate", threshold=threshold)
    if best_confidence < min_confidence:
        return PaperLocation.failed(
            "low_confidence",
            threshold=threshold,
            confidence=best_confidence,
        )
    return PaperLocation(
        True,
        paper_quad=best_quad,
        active_quad=None,
        confidence=best_confidence,
        threshold=threshold,
        reason="ok",
    )
