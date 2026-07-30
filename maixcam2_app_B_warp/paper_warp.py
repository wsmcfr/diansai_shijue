"""把锁定的完整黑色A4纸透视展开为固定机械工作平面。"""

import cv2
import numpy as np

try:
    from maixcam2_app_B_warp.paper_locator import order_a4_quad
except ModuleNotFoundError as error:
    # MaixVision平铺部署时顶层包不存在，改用与main.py同级的定位模块。
    if error.name != "maixcam2_app_B_warp":
        raise
    from paper_locator import order_a4_quad


# B版固定使用2像素/mm，使工作图尺寸与机械毫米坐标形成稳定一一比例。
PIXELS_PER_MM = 2.0
A4_SIZE_PX = (420, 594)
WORK_SIZE_PX = (420, 460)
WORK_TOP_PX = 67
WORK_BOTTOM_PX = 527
MAX_INSET_MM = 20.0


class WarpResult:
    """保存完整A4展开图、机械工作图、有效掩膜和单应性矩阵。"""

    def __init__(self, full_a4, work_area, valid_mask, homography):
        """初始化展开结果；所有图像尺寸由模块常量固定，不在构造函数中再次缩放。"""
        self.full_a4 = full_a4
        self.work_area = work_area
        self.valid_mask = valid_mask
        self.homography = homography


def build_a4_homography(paper_quad):
    """构造原相机完整A4四角到420×594标准画布的单应性矩阵。

    主要流程：调用共用角点排序与凸性校验，再把四角映射到目标像素边缘0～419、
    0～593。返回值：3×3 float64矩阵；退化或不可逆输入抛出 ValueError。
    """
    ordered_quad = order_a4_quad(paper_quad)
    target_quad = np.float32(
        [[0, 0], [A4_SIZE_PX[0] - 1, 0], [A4_SIZE_PX[0] - 1, A4_SIZE_PX[1] - 1], [0, A4_SIZE_PX[1] - 1]]
    )
    homography = cv2.getPerspectiveTransform(ordered_quad, target_quad)
    if not np.all(np.isfinite(homography)) or abs(float(np.linalg.det(homography))) <= 1e-9:
        raise ValueError("A4四边形无法建立有效透视展开矩阵")
    return homography


def _build_work_valid_mask(inset_mm):
    """在420×460固定工作平面中生成四边整体内缩后的二值有效掩膜。"""
    try:
        inset_mm = float(inset_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("inset_mm 必须是0到20之间的数字") from error
    if not np.isfinite(inset_mm) or not 0.0 <= inset_mm <= MAX_INSET_MM:
        raise ValueError("inset_mm 必须位于0到20之间")

    inset_px = int(round(inset_mm * PIXELS_PER_MM))
    work_width, work_height = WORK_SIZE_PX
    valid_mask = np.zeros((work_height, work_width), dtype=np.uint8)
    cv2.rectangle(
        valid_mask,
        (inset_px, inset_px),
        (work_width - 1 - inset_px, work_height - 1 - inset_px),
        255,
        -1,
    )
    return valid_mask


def warp_to_work_area(frame_bgr, paper_quad, inset_mm=0.0):
    """把相机帧展开为完整A4，并裁取中间210×230mm机械工作区。

    主要流程：校验三通道输入、计算单应性、展开420×594、裁取纵向67:527，
    再按毫米INSET生成420×460有效掩膜。
    返回值：``WarpResult``；工作图保持原亮度，调用识别时再应用 valid_mask。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr 必须是有效的 numpy 图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr 必须是三通道 BGR 图像")

    valid_mask = _build_work_valid_mask(inset_mm)
    homography = build_a4_homography(paper_quad)
    full_a4 = cv2.warpPerspective(
        frame_bgr,
        homography,
        A4_SIZE_PX,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    work_area = full_a4[WORK_TOP_PX:WORK_BOTTOM_PX, :].copy()
    if work_area.shape[:2] != (WORK_SIZE_PX[1], WORK_SIZE_PX[0]):
        raise ValueError("透视展开后的机械工作区尺寸无效")
    return WarpResult(full_a4, work_area, valid_mask, homography)


def pixels_to_work_mm(point):
    """把420×460工作图像素点换算为机械区域左上原点的毫米坐标。

    关键参数：point 必须包含有限的 ``(x, y)`` 像素坐标。
    返回值：``(x_mm, y_mm)``，每个分量除以固定的2像素/mm比例。
    """
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
        raise ValueError("point 必须包含两个有限坐标")
    return (
        float(point_array[0] / PIXELS_PER_MM),
        float(point_array[1] / PIXELS_PER_MM),
    )
