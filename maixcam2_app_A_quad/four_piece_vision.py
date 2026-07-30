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
