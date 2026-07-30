"""UNKNOWN FOUR模式的完整A4透视展开与白色碎片视觉前端。"""

import math

import cv2
import numpy as np

try:
    from maixcam2_app_A_quad.paper_locator import (
        order_a4_quad,
        paper_size_mm,
        validate_paper_orientation,
    )
except ModuleNotFoundError as error:
    # MaixVision会把发布包平铺到临时目录；只有顶层包不存在时才改用同级导入。
    if error.name != "maixcam2_app_A_quad":
        raise
    from paper_locator import (
        order_a4_quad,
        paper_size_mm,
        validate_paper_orientation,
    )


# ======================== FOUR现场调试常量（仅影响四片模式） ========================
# 完整A4展开图的默认像素密度。3px/mm让2mm黑缝在处理图上约有6像素宽。
FOUR_WARP_PIXELS_PER_MM = 3.0
# 严格白色核心需要低饱和、高HSV亮度和高LAB亮度同时成立，优先保护片间黑缝。
FOUR_HSV_S_MAX = 80
FOUR_HSV_V_MIN = 150
FOUR_LAB_L_MIN = 150
# 宽松支撑只允许被可靠核心选中的连通域进入结果，用于补回阴影和裸露金属。
FOUR_SUPPORT_S_MAX = 160
FOUR_SUPPORT_V_MIN = 90
FOUR_SUPPORT_L_MIN = 95
# 没有达到该严格核心面积的亮点不能独立成为碎片，单位平方毫米。
FOUR_MIN_CORE_AREA_MM2 = 4.0
# 单片候选的最小面积，排除大于亮点但明显不可能参与题目矩形的小区域。
FOUR_MIN_PIECE_AREA_MM2 = 80.0
# 单片候选最大面积不应超过题目最大120×90mm矩形，超出通常是纸外或大面积反光。
FOUR_MAX_PIECE_AREA_MM2 = 10800.0
# 轮廓距离A4边缘或上下分界线小于该值时视为不完整，避免发送被裁断的几何。
FOUR_BORDER_MARGIN_MM = 1.0
# 稳定锁定默认要求连续三帧；重心和面积门均只用于FOUR视觉运行器。
FOUR_STABLE_FRAMES = 3
FOUR_CENTER_JITTER_MM = 2.0
FOUR_AREA_JITTER_RATIO = 0.10
# True时允许3个支撑连通域在确有4个可靠严格核心时执行一次受限拆分。
FOUR_SPLIT_CONNECTED_ENABLED = True
# True时允许构造并输出四片视觉与求解诊断；关闭后应跳过高成本调试字符串。
FOUR_SOLVER_DEBUG = True
# ===============================================================================


class PaperWarpResult:
    """保存完整A4透视图、坐标矩阵和毫米换算参数。"""

    def __init__(
        self,
        image,
        image_to_warp_matrix,
        warp_to_image_matrix,
        paper_size,
        pixels_per_mm,
        paper_orientation,
    ):
        """初始化一次不可混用坐标系的纸面展开结果。

        关键参数：image为透视展开后的BGR图；两个矩阵分别完成相机像素到展开像素和
        反向映射；paper_size单位毫米；pixels_per_mm为固定展开比例。构造函数不返回值。
        """
        self.image = image
        self.image_to_warp_matrix = np.asarray(
            image_to_warp_matrix,
            dtype=np.float64,
        ).copy()
        self.warp_to_image_matrix = np.asarray(
            warp_to_image_matrix,
            dtype=np.float64,
        ).copy()
        self.paper_size_mm = tuple(float(value) for value in paper_size)
        self.pixels_per_mm = float(pixels_per_mm)
        self.paper_orientation = str(paper_orientation)

    def pixel_to_mm(self, point_px):
        """把一个展开图像素点转换为完整A4左上角原点的毫米坐标。

        关键参数point_px必须包含两个有限数；返回``(x_mm, y_mm)``。该转换只除以
        固定展开比例，不重复求Homography，因此可用于每片全部轮廓点。
        """
        point = _normalize_point(point_px, "展开像素点")
        return (
            float(point[0] / self.pixels_per_mm),
            float(point[1] / self.pixels_per_mm),
        )

    def pixels_to_mm(self, points_px):
        """批量把N×2展开图像素转换为N×2纸面毫米数组。"""
        points = _normalize_points(points_px, "展开像素点")
        return (points / self.pixels_per_mm).astype(np.float32)

    def mm_to_image(self, point_mm):
        """把一个纸面毫米点经本次逆矩阵映射回原相机像素浮点元组。"""
        point = _normalize_point(point_mm, "纸面毫米点")
        warp_point = (point * self.pixels_per_mm).astype(np.float32)
        mapped = cv2.perspectiveTransform(
            warp_point.reshape(1, 1, 2),
            self.warp_to_image_matrix,
        )[0, 0]
        if not np.all(np.isfinite(mapped)):
            raise ValueError("纸面毫米点无法映射回相机图像")
        return float(mapped[0]), float(mapped[1])

    def mm_points_to_image(self, points_mm):
        """批量把N×2纸面毫米点映射回原相机像素数组，供正常页回绘轮廓。"""
        points = _normalize_points(points_mm, "纸面毫米点")
        warp_points = (points * self.pixels_per_mm).astype(np.float32)
        mapped = cv2.perspectiveTransform(
            warp_points.reshape(1, -1, 2),
            self.warp_to_image_matrix,
        )[0]
        if not np.all(np.isfinite(mapped)):
            raise ValueError("纸面毫米点无法批量映射回相机图像")
        return mapped.astype(np.float32)


class FourPieceMasks:
    """保存FOUR分割的严格核心、宽松支撑和最终恢复掩膜。"""

    def __init__(self, strict, support, final):
        """复制并保存三张同尺寸uint8二值掩膜，构造函数无返回值。"""
        masks = []
        for name, mask in (("strict", strict), ("support", support), ("final", final)):
            if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
                raise ValueError(f"{name}必须是二维uint8掩膜")
            masks.append(mask.copy())
        if len({mask.shape for mask in masks}) != 1:
            raise ValueError("strict、support和final掩膜尺寸必须一致")
        self.strict, self.support, self.final = masks


class FourPieceDetection:
    """保存一帧FOUR检测、三张掩膜、轮廓数量和稳定锁定状态。"""

    def __init__(
        self,
        pieces,
        warp,
        masks,
        pre_split_count,
        split_applied=False,
        reason="",
        locked=False,
    ):
        """初始化一帧检测结果，所有碎片以只读约定元组保存。

        关键参数pre_split_count表示受限拆分前的支撑连通域数量；split_applied用于现场
        判断是否靠拆分得到四片；reason提供可直接显示的失败原因。构造函数无返回值。
        """
        self.pieces = tuple(pieces)
        self.warp = warp
        self.masks = masks
        self.pre_split_count = int(pre_split_count)
        self.valid_contour_count = len(self.pieces)
        self.split_applied = bool(split_applied)
        self.reason = str(reason)
        self.locked = bool(locked)


def _normalize_point(point, field_name):
    """校验单个二维有限点并返回独立float64数组。"""
    normalized = np.asarray(point, dtype=np.float64)
    if normalized.shape != (2,) or not np.all(np.isfinite(normalized)):
        raise ValueError(f"{field_name}必须包含两个有限数字")
    return normalized.copy()


def _normalize_points(points, field_name):
    """校验N×2有限点集并返回独立float64数组。"""
    normalized = np.asarray(points, dtype=np.float64)
    if (
        normalized.ndim != 2
        or normalized.shape[1] != 2
        or len(normalized) < 1
        or not np.all(np.isfinite(normalized))
    ):
        raise ValueError(f"{field_name}必须是非空N×2有限数组")
    return normalized.copy()


def _validate_frame(frame_bgr):
    """校验相机输入必须是非空三通道BGR图像。"""
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效numpy图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr必须是三通道BGR图像")
    if min(frame_bgr.shape[:2]) < 8:
        raise ValueError("frame_bgr尺寸过小，无法展开A4纸面")


def _validate_pixels_per_mm(pixels_per_mm):
    """校验展开比例并返回正的有限浮点数。"""
    try:
        normalized = float(pixels_per_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("pixels_per_mm必须是正有限数") from error
    if normalized <= 0.0 or not math.isfinite(normalized):
        raise ValueError("pixels_per_mm必须是正有限数")
    return normalized


def _fill_internal_mask_holes(mask):
    """只填补与图像外部背景不连通的黑孔，不扩大任何白色外边界。

    主要流程：外加一圈确定黑边，从(0,0)泛洪所有外部背景；取反后仅剩封闭黑孔，
    再与原掩膜合并。对外开放的凹口和片间黑缝都与背景连通，因此不会被误填。
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("mask必须是二维uint8掩膜")
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    flood_mask = np.zeros(
        (padded.shape[0] + 2, padded.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    enclosed_holes = cv2.bitwise_not(flooded)
    return cv2.bitwise_or(padded, enclosed_holes)[1:-1, 1:-1].copy()


def _select_supported_components(strict_mask, support_mask, min_core_pixels):
    """保留拥有足够严格白色核心的宽松连通域，并删除孤立反光噪声。

    关键参数两张掩膜必须同尺寸；min_core_pixels是每个支撑域内所需的最小严格核心
    像素数。返回独立uint8掩膜。该过程不做膨胀，支撑域之间的黑缝不会被改变。
    """
    if strict_mask.shape != support_mask.shape:
        raise ValueError("严格核心和宽松支撑掩膜尺寸必须一致")
    component_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        support_mask,
        connectivity=8,
    )
    selected = np.zeros_like(support_mask)
    for label in range(1, int(component_count)):
        component = labels == label
        core_pixels = int(np.count_nonzero(strict_mask[component]))
        if core_pixels < min_core_pixels:
            continue
        selected[component] = 255
    return selected


def build_four_piece_masks(
    warped_bgr,
    pixels_per_mm=FOUR_WARP_PIXELS_PER_MM,
):
    """在完整A4展开图中构造不跨越黑缝的HSV/LAB双阈值白片掩膜。

    主要流程：严格核心同时满足HSV低饱和高亮和LAB高亮；宽松支撑降低亮度并允许
    更高饱和度，用于包含灰色裸露金属。随后只保留内部拥有足够严格核心的支撑连通
    域，删除孤立反光，并用拓扑泛洪填补单片内部孔洞。函数不使用膨胀或闭运算。
    关键参数warped_bgr必须是透视展开后的三通道BGR图；返回FourPieceMasks。
    """
    _validate_frame(warped_bgr)
    scale = _validate_pixels_per_mm(pixels_per_mm)
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]
    hsv_value = hsv[:, :, 2]
    lab_lightness = lab[:, :, 0]

    strict_condition = (
        (saturation <= int(FOUR_HSV_S_MAX))
        & (hsv_value >= int(FOUR_HSV_V_MIN))
        & (lab_lightness >= int(FOUR_LAB_L_MIN))
    )
    support_condition = (
        (saturation <= int(FOUR_SUPPORT_S_MAX))
        & (hsv_value >= int(FOUR_SUPPORT_V_MIN))
        & (lab_lightness >= int(FOUR_SUPPORT_L_MIN))
    )
    strict_mask = np.where(strict_condition, 255, 0).astype(np.uint8)
    # 严格核心必须始终属于支撑域，防止现场阈值调整后出现核心被支撑排除的矛盾状态。
    support_mask = np.where(support_condition | strict_condition, 255, 0).astype(np.uint8)
    min_core_pixels = max(
        1,
        int(round(float(FOUR_MIN_CORE_AREA_MM2) * scale * scale)),
    )
    final_mask = _select_supported_components(
        strict_mask,
        support_mask,
        min_core_pixels,
    )
    final_mask = _fill_internal_mask_holes(final_mask)
    return FourPieceMasks(strict_mask, support_mask, final_mask)


def _mask_source_region(masks, split_y_mm, pixels_per_mm):
    """把三张掩膜限制到A4上半源区域，防止下半目标区白物体进入检测。

    split_y_mm为None时返回原对象；否则复制掩膜并把分界线及其下方清零。返回新的
    FourPieceMasks，原调试掩膜不会被调用方意外修改。
    """
    if split_y_mm is None:
        return masks
    try:
        split_y = float(split_y_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("split_y_mm必须是正有限数") from error
    if split_y <= 0.0 or not math.isfinite(split_y):
        raise ValueError("split_y_mm必须是正有限数")
    split_row = int(round(split_y * float(pixels_per_mm)))
    if not 1 <= split_row <= masks.final.shape[0]:
        raise ValueError("split_y_mm必须位于展开A4高度内部")
    clipped = []
    for source in (masks.strict, masks.support, masks.final):
        target = source.copy()
        target[split_row:, :] = 0
        clipped.append(target)
    return FourPieceMasks(*clipped)


def _significant_component_masks(mask, pixels_per_mm, min_area_mm2):
    """按真实平方毫米面积提取二值掩膜中的有效8连通域。"""
    scale = _validate_pixels_per_mm(pixels_per_mm)
    min_pixels = max(1, int(round(float(min_area_mm2) * scale * scale)))
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    components = []
    for label in range(1, int(component_count)):
        area_pixels = int(stats[label, cv2.CC_STAT_AREA])
        if area_pixels < min_pixels:
            continue
        component = np.zeros_like(mask)
        component[labels == label] = 255
        components.append(component)
    return components


def _split_support_bridge(final_components, strict_mask, pixels_per_mm):
    """在3个支撑域确实包含4个严格核心时，把唯一双核心父域按最近核心拆开。

    该函数不会按数量任意切割：严格核心必须恰好四个；两个核心必须落在同一个最终
    父域；其余父域各含一个核心；拆出的每个子域都必须达到单片面积下限。返回
    ``(组件列表, 是否拆分)``。
    """
    if not FOUR_SPLIT_CONNECTED_ENABLED or len(final_components) != 3:
        return final_components, False
    strict_components = _significant_component_masks(
        strict_mask,
        pixels_per_mm,
        FOUR_MIN_CORE_AREA_MM2,
    )
    if len(strict_components) != 4:
        return final_components, False

    core_groups = []
    for parent in final_components:
        overlapping = [
            core
            for core in strict_components
            if cv2.countNonZero(cv2.bitwise_and(parent, core)) > 0
        ]
        core_groups.append(overlapping)
    group_sizes = sorted(len(group) for group in core_groups)
    if group_sizes != [1, 1, 2]:
        return final_components, False

    result = []
    min_piece_pixels = int(
        round(float(FOUR_MIN_PIECE_AREA_MM2) * float(pixels_per_mm) ** 2)
    )
    for parent, cores in zip(final_components, core_groups):
        if len(cores) == 1:
            result.append(parent)
            continue

        # distanceTransform计算每个父域像素到两个核心的最近距离；argmin形成只在父域
        # 内生效的Voronoi分界，不会改变任何其他碎片或纸面背景。
        distances = []
        for core in cores:
            distance_input = np.where(core > 0, 0, 255).astype(np.uint8)
            distances.append(cv2.distanceTransform(distance_input, cv2.DIST_L2, 3))
        nearest = np.argmin(np.stack(distances, axis=0), axis=0)
        children = []
        for index in range(len(cores)):
            child = np.zeros_like(parent)
            child[(parent > 0) & (nearest == index)] = 255
            if cv2.countNonZero(child) < min_piece_pixels:
                return final_components, False
            children.append(child)
        result.extend(children)
    return result, len(result) == 4


def _component_contour(component_mask):
    """返回单连通域的最大外轮廓；没有有效轮廓时返回None。"""
    contours, _hierarchy = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _approximate_piece_vertices(contour):
    """从稠密轮廓生成保留主要直边的初始多边形，详细多假设留给求解器。"""
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        raise ValueError("碎片轮廓周长无效")
    for epsilon_ratio in (0.006, 0.009, 0.012, 0.016, 0.022):
        candidate = cv2.approxPolyDP(contour, perimeter * epsilon_ratio, True)
        if 3 <= len(candidate) <= 10:
            return candidate.reshape(-1, 2).astype(np.float32)
    raise ValueError("碎片轮廓无法形成有效多边形")


def _build_piece_from_component(component, warp, split_y_mm):
    """把单个展开掩膜连通域转换为兼容显示和求解的碎片字典。

    返回字段同时包含展开稠密轮廓、纸面毫米顶点与重心、原相机像素顶点与重心。
    如果面积、几何或边界完整性无效则返回None。
    """
    contour = _component_contour(component)
    if contour is None:
        return None
    area_pixels = abs(float(cv2.contourArea(contour)))
    area_mm2 = area_pixels / (warp.pixels_per_mm ** 2)
    if not FOUR_MIN_PIECE_AREA_MM2 <= area_mm2 <= FOUR_MAX_PIECE_AREA_MM2:
        return None
    moments = cv2.moments(contour)
    if abs(float(moments["m00"])) <= 1e-9:
        return None
    center_warp = np.asarray(
        (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ),
        dtype=np.float64,
    )
    try:
        vertices_warp = _approximate_piece_vertices(contour)
    except ValueError:
        return None
    center_mm = np.asarray(warp.pixel_to_mm(center_warp), dtype=np.float64)
    vertices_mm = warp.pixels_to_mm(vertices_warp)
    raw_contour_mm = warp.pixels_to_mm(contour.reshape(-1, 2))
    image_vertices = warp.mm_points_to_image(vertices_mm)
    image_center = warp.mm_to_image(center_mm)

    paper_width_mm, paper_height_mm = warp.paper_size_mm
    minimum = np.min(raw_contour_mm, axis=0)
    maximum = np.max(raw_contour_mm, axis=0)
    complete = (
        minimum[0] > FOUR_BORDER_MARGIN_MM
        and minimum[1] > FOUR_BORDER_MARGIN_MM
        and maximum[0] < paper_width_mm - FOUR_BORDER_MARGIN_MM
        and maximum[1] < paper_height_mm - FOUR_BORDER_MARGIN_MM
    )
    if split_y_mm is not None:
        complete = complete and maximum[1] < float(split_y_mm) - FOUR_BORDER_MARGIN_MM
    region = "upper" if split_y_mm is None or center_mm[1] < float(split_y_mm) else "lower"
    return {
        "id": "",
        "contour": np.rint(warp.mm_points_to_image(raw_contour_mm)).astype(np.int32).reshape(-1, 1, 2),
        "warp_contour": contour.copy(),
        "raw_contour_mm": raw_contour_mm.astype(float).tolist(),
        "vertices": [tuple(np.rint(point).astype(np.int32)) for point in image_vertices],
        "vertices_mm": vertices_mm.astype(float).tolist(),
        "center": tuple(float(value) for value in image_center),
        "center_mm": tuple(float(value) for value in center_mm),
        "area": area_pixels,
        "area_mm2": float(area_mm2),
        "region": region,
        "complete": bool(complete),
    }


def analyze_four_piece_frame(
    frame_bgr,
    paper_quad,
    paper_orientation,
    split_y_mm=None,
    pixels_per_mm=FOUR_WARP_PIXELS_PER_MM,
):
    """执行一帧FOUR专用透视、双阈值、受限拆分和毫米几何提取。

    返回FourPieceDetection。该函数只做单帧视觉，不累计稳定状态、不启动求解；只有
    FourPieceVisionRuntime负责跨帧锁定。四片按纸面X中心从左到右编号U1～U4。
    """
    warp = warp_full_paper(
        frame_bgr,
        paper_quad,
        paper_orientation,
        pixels_per_mm=pixels_per_mm,
    )
    masks = build_four_piece_masks(warp.image, pixels_per_mm=warp.pixels_per_mm)
    masks = _mask_source_region(masks, split_y_mm, warp.pixels_per_mm)
    final_components = _significant_component_masks(
        masks.final,
        warp.pixels_per_mm,
        FOUR_MIN_PIECE_AREA_MM2,
    )
    pre_split_count = len(final_components)
    final_components, split_applied = _split_support_bridge(
        final_components,
        masks.strict,
        warp.pixels_per_mm,
    )
    pieces = []
    for component in final_components:
        piece = _build_piece_from_component(component, warp, split_y_mm)
        if piece is not None:
            pieces.append(piece)
    pieces.sort(key=lambda item: (float(item["center_mm"][0]), float(item["center_mm"][1])))
    for index, piece in enumerate(pieces, start=1):
        piece["id"] = f"U{index}"
    reason = "ok" if len(pieces) == 4 else f"count_{len(pieces)}_of_4"
    return FourPieceDetection(
        pieces,
        warp,
        masks,
        pre_split_count,
        split_applied=split_applied,
        reason=reason,
    )


def _detections_are_stable(reference, current, center_tolerance_mm, area_tolerance_ratio):
    """比较两帧按X排序的四片重心和面积是否同时位于稳定门内。"""
    if len(reference.pieces) != 4 or len(current.pieces) != 4:
        return False
    for previous_piece, current_piece in zip(reference.pieces, current.pieces):
        previous_center = np.asarray(previous_piece["center_mm"], dtype=np.float64)
        current_center = np.asarray(current_piece["center_mm"], dtype=np.float64)
        if float(np.linalg.norm(current_center - previous_center)) > center_tolerance_mm:
            return False
        previous_area = max(1e-6, float(previous_piece["area_mm2"]))
        current_area = float(current_piece["area_mm2"])
        if abs(current_area - previous_area) / previous_area > area_tolerance_ratio:
            return False
    return True


class FourPieceVisionRuntime:
    """累计FOUR单帧检测并在第三个稳定结果后永久冻结本轮几何。"""

    def __init__(
        self,
        stable_frames=FOUR_STABLE_FRAMES,
        center_tolerance_mm=FOUR_CENTER_JITTER_MM,
        area_tolerance_ratio=FOUR_AREA_JITTER_RATIO,
        pixels_per_mm=FOUR_WARP_PIXELS_PER_MM,
    ):
        """初始化稳定门参数并进入未锁定待机状态。

        关键参数分别控制所需连续帧、重心最大毫米位移、面积最大相对变化和展开比例。
        非法参数立即抛ValueError，避免设备运行后静默无法锁定。
        """
        self.stable_frames = int(stable_frames)
        self.center_tolerance_mm = float(center_tolerance_mm)
        self.area_tolerance_ratio = float(area_tolerance_ratio)
        self.pixels_per_mm = _validate_pixels_per_mm(pixels_per_mm)
        if self.stable_frames < 1:
            raise ValueError("stable_frames必须至少为1")
        if self.center_tolerance_mm < 0.0 or not math.isfinite(self.center_tolerance_mm):
            raise ValueError("center_tolerance_mm必须是非负有限数")
        if not 0.0 <= self.area_tolerance_ratio < 1.0:
            raise ValueError("area_tolerance_ratio必须位于0到1之间")
        self.reset()

    def reset(self):
        """释放参考帧和锁定快照，使下一次START从完全待机重新采集。"""
        self.stable_count = 0
        self.snapshot_locked = False
        self.locked_pieces = ()
        self.last_detection = None
        self._reference_detection = None
        self._locked_detection = None
        self._context_key = None

    def update(self, frame_bgr, paper_quad, paper_orientation, split_y_mm=None):
        """分析一帧并推进稳定门；锁定后始终返回同一个冻结检测对象。

        主要流程：标定上下文变化时清空旧参考；单帧必须恰有四片且全部完整才参与
        稳定计数；达到stable_frames后把当前结果标为locked并停止后续分割。
        返回FourPieceDetection，便于界面显示实时失败原因或冻结轮廓。
        """
        if self.snapshot_locked:
            return self._locked_detection
        normalized_quad = order_a4_quad(paper_quad)
        orientation = validate_paper_orientation(paper_orientation)
        context_key = (
            orientation,
            None if split_y_mm is None else round(float(split_y_mm), 4),
            tuple(np.round(normalized_quad.reshape(-1), 3)),
        )
        if self._context_key is not None and context_key != self._context_key:
            self.stable_count = 0
            self._reference_detection = None
        self._context_key = context_key
        detection = analyze_four_piece_frame(
            frame_bgr,
            normalized_quad,
            orientation,
            split_y_mm=split_y_mm,
            pixels_per_mm=self.pixels_per_mm,
        )
        self.last_detection = detection
        actionable = (
            detection.valid_contour_count == 4
            and all(piece.get("complete") is True for piece in detection.pieces)
        )
        if not actionable:
            self.stable_count = 0
            self._reference_detection = None
            return detection

        if self._reference_detection is None:
            self.stable_count = 1
        elif _detections_are_stable(
            self._reference_detection,
            detection,
            self.center_tolerance_mm,
            self.area_tolerance_ratio,
        ):
            self.stable_count += 1
        else:
            self.stable_count = 1
        self._reference_detection = detection
        if self.stable_count < self.stable_frames:
            return detection

        detection.locked = True
        self.snapshot_locked = True
        self.locked_pieces = detection.pieces
        self._locked_detection = detection
        return detection


def build_paper_warp(
    paper_quad,
    paper_orientation,
    pixels_per_mm=FOUR_WARP_PIXELS_PER_MM,
):
    """构造相机像素与完整A4展开像素之间的一对单应矩阵。

    主要流程：规范蓝框四角和横竖方向，按毫米尺寸乘固定像素密度建立目标矩形，
    再分别计算正向和反向Homography。关键参数paper_quad为相机像素四角；返回
    ``(正向矩阵, 反向矩阵, 输出宽, 输出高, 纸面尺寸, 规范方向, 比例)``。
    """
    scale = _validate_pixels_per_mm(pixels_per_mm)
    orientation = validate_paper_orientation(paper_orientation)
    ordered_quad = order_a4_quad(paper_quad)
    paper_width_mm, paper_height_mm = paper_size_mm(orientation)
    output_width = int(round(paper_width_mm * scale))
    output_height = int(round(paper_height_mm * scale))
    if output_width < 2 or output_height < 2:
        raise ValueError("A4展开图尺寸过小")

    # 目标矩形采用半开像素范围：毫米点乘比例后可直接换算，纸张最右/下边界位于
    # 输出画布边界线上，不会给内部重心引入(width-1)/width比例误差。
    destination_quad = np.float32(
        (
            (0.0, 0.0),
            (float(output_width), 0.0),
            (float(output_width), float(output_height)),
            (0.0, float(output_height)),
        )
    )
    image_to_warp = cv2.getPerspectiveTransform(ordered_quad, destination_quad)
    warp_to_image = cv2.getPerspectiveTransform(destination_quad, ordered_quad)
    if (
        not np.all(np.isfinite(image_to_warp))
        or not np.all(np.isfinite(warp_to_image))
        or abs(float(np.linalg.det(image_to_warp))) <= 1e-12
        or abs(float(np.linalg.det(warp_to_image))) <= 1e-12
    ):
        raise ValueError("A4蓝框无法建立有效透视矩阵")
    return (
        image_to_warp,
        warp_to_image,
        output_width,
        output_height,
        (paper_width_mm, paper_height_mm),
        orientation,
        scale,
    )


def warp_full_paper(
    frame_bgr,
    paper_quad,
    paper_orientation,
    pixels_per_mm=FOUR_WARP_PIXELS_PER_MM,
):
    """把固定相机中的完整A4蓝框展开为比例固定的俯视BGR图。

    主要流程：校验相机帧，调用build_paper_warp建立矩阵，再用线性插值展开颜色图。
    线性插值只作用于BGR图；后续二值分割禁止使用会跨越2mm黑缝的大形态学核。
    返回PaperWarpResult，包含展开图、正逆矩阵、毫米尺寸和坐标换算方法。
    """
    _validate_frame(frame_bgr)
    (
        image_to_warp,
        warp_to_image,
        output_width,
        output_height,
        paper_size,
        orientation,
        scale,
    ) = build_paper_warp(
        paper_quad,
        paper_orientation,
        pixels_per_mm=pixels_per_mm,
    )
    warped = cv2.warpPerspective(
        frame_bgr,
        image_to_warp,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return PaperWarpResult(
        warped,
        image_to_warp,
        warp_to_image,
        paper_size,
        scale,
        orientation,
    )
