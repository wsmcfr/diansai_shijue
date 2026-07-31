"""从固定相机画面中单次定位完整黑色A4纸。"""

import cv2
import numpy as np

try:
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
except ModuleNotFoundError as error:
    # MaixVision会把发布包平铺到临时目录，只有包本身不存在时才回退同级导入。
    if error.name != "maixcam2_app_A_quad":
        raise
    from config import DEFAULT_CONFIG


# 完整A4和龙门架机械覆盖范围都使用毫米。旧常量继续表示竖放纸面，避免已有
# 离线工具和测试导入时失效；横放纸面的宽高由paper_size_mm统一返回。
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
WORK_HEIGHT_MM = 230.0
WORK_TRIM_MM = (A4_HEIGHT_MM - WORK_HEIGHT_MM) / 2.0
MAX_INSET_MM = 20.0
PAPER_ORIENTATION_PORTRAIT = "portrait"
PAPER_ORIENTATION_LANDSCAPE = "landscape"
PAPER_ORIENTATIONS = (
    PAPER_ORIENTATION_PORTRAIT,
    PAPER_ORIENTATION_LANDSCAPE,
)
CAMERA_MOUNT_TOP = "top"
CAMERA_MOUNT_SIDE_LOWER_RIGHT = "side_lower_right"
CAMERA_MOUNT_SIDE_LOWER_LEFT = "side_lower_left"
CAMERA_MOUNT_UPSIDE_DOWN = "upside_down"
CAMERA_MOUNT_DIRECTIONS = (
    CAMERA_MOUNT_TOP,
    CAMERA_MOUNT_SIDE_LOWER_RIGHT,
    CAMERA_MOUNT_SIDE_LOWER_LEFT,
    CAMERA_MOUNT_UPSIDE_DOWN,
)
CAMERA_MOUNT_QUAD_SHIFTS = {
    CAMERA_MOUNT_TOP: 0,
    CAMERA_MOUNT_SIDE_LOWER_RIGHT: 1,
    CAMERA_MOUNT_SIDE_LOWER_LEFT: -1,
    CAMERA_MOUNT_UPSIDE_DOWN: 2,
}


def validate_paper_orientation(paper_orientation):
    """校验并返回规范化纸张方向字符串。

    关键参数：paper_orientation只接受``portrait``或``landscape``。
    返回值：小写方向字符串；非法值抛出ValueError，防止错误毫米坐标静默传播。
    """
    normalized = str(paper_orientation).strip().lower()
    if normalized not in PAPER_ORIENTATIONS:
        raise ValueError("paper_orientation必须是portrait或landscape")
    return normalized


def validate_camera_mount_direction(camera_mount_direction):
    """校验并返回固定相机安装方向。

    可选值：top为正常顶置；side_lower_right/side_lower_left分别表示侧装后目标下半区
    出现在原始CAL画面右侧/左侧；upside_down表示顶置画面旋转180度。非法值抛出
    ValueError，防止机械毫米原点和UART坐标静默反向。
    """
    normalized = str(camera_mount_direction).strip().lower()
    if normalized not in CAMERA_MOUNT_DIRECTIONS:
        raise ValueError(
            "camera_mount_direction必须是top、side_lower_right、"
            "side_lower_left或upside_down"
        )
    return normalized


def paper_size_mm(paper_orientation=PAPER_ORIENTATION_PORTRAIT):
    """按纸张方向返回完整A4纸面的``(宽, 高)``毫米尺寸。"""
    orientation = validate_paper_orientation(paper_orientation)
    if orientation == PAPER_ORIENTATION_LANDSCAPE:
        return A4_HEIGHT_MM, A4_WIDTH_MM
    return A4_WIDTH_MM, A4_HEIGHT_MM


def default_work_region_mm(paper_orientation=PAPER_ORIENTATION_PORTRAIT):
    """返回当前方向下完整A4纸面的默认黄色区域。

    该区域用于视觉检测和目标显示，不再用230mm电机行程裁剪；机械端只需按协议
    接收能够到达的毫米目标。返回值为``(x, y, width, height)``毫米元组。
    """
    paper_width_mm, paper_height_mm = paper_size_mm(paper_orientation)
    return (0.0, 0.0, paper_width_mm, paper_height_mm)


def default_split_y_mm(paper_orientation=PAPER_ORIENTATION_PORTRAIT):
    """返回当前方向的水平纸面中线Y坐标，作为默认上下区分界线。"""
    _, paper_height_mm = paper_size_mm(paper_orientation)
    return paper_height_mm / 2.0


def validate_work_region_mm(
    work_region_mm,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """校验A4纸内的机械工作矩形并返回规范化毫米元组。

    主要流程：读取X/Y/W/H四个有限数字，按纸张方向取得完整纸面宽高，并确保
    区域完整位于纸面内。视觉区域不再受230mm电机行程限制。关键参数可为四元素序列。
    返回值：``(x_mm, y_mm, width_mm, height_mm)``；非法输入抛出ValueError。
    """
    try:
        values = tuple(float(value) for value in work_region_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("机械区域必须包含X/Y/W/H四个数字") from error
    if len(values) != 4 or not np.all(np.isfinite(np.asarray(values))):
        raise ValueError("机械区域必须包含X/Y/W/H四个有限数字")

    paper_width_mm, paper_height_mm = paper_size_mm(paper_orientation)
    work_x_mm, work_y_mm, work_width_mm, work_height_mm = values
    if work_x_mm < 0.0:
        raise ValueError("work_x_mm不能为负数")
    if work_y_mm < 0.0:
        raise ValueError("work_y_mm不能为负数")
    if not 0.0 < work_width_mm <= paper_width_mm:
        raise ValueError(f"work_width_mm必须位于0到{paper_width_mm:g}之间")
    if not 0.0 < work_height_mm <= paper_height_mm:
        raise ValueError(f"work_height_mm必须位于0到{paper_height_mm:g}之间")
    if work_x_mm + work_width_mm > paper_width_mm + 1e-6:
        raise ValueError("机械区域X方向必须完整位于A4纸内")
    if work_y_mm + work_height_mm > paper_height_mm + 1e-6:
        raise ValueError("机械区域Y方向必须完整位于A4纸内")
    return values


def validate_split_y_mm(
    work_region_mm,
    split_y_mm,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """校验毫米分界线必须严格位于机械区域的上、下边之间。"""
    work_x_mm, work_y_mm, work_width_mm, work_height_mm = validate_work_region_mm(
        work_region_mm,
        paper_orientation,
    )
    del work_x_mm, work_width_mm
    try:
        split_y_mm = float(split_y_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("split_y_mm必须是有限数字") from error
    if not np.isfinite(split_y_mm):
        raise ValueError("split_y_mm必须是有限数字")
    if not work_y_mm < split_y_mm < work_y_mm + work_height_mm:
        raise ValueError("split_y_mm必须位于机械区域内部")
    return split_y_mm


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
        paper_orientation=PAPER_ORIENTATION_PORTRAIT,
        confidence=0.0,
        threshold=0.0,
        reason="",
        diagnostics=None,
    ):
        """初始化定位结果。

        关键参数：confidence 位于0～1；threshold 为本次Otsu阈值；reason 用于屏幕状态；
        diagnostics保存本次各候选门的统计，只供电脑端调试，不参与定位结果判断。
        返回值：构造函数无返回值，输入会规范化后保存为公开属性。
        """
        self.success = bool(success)
        self.paper_quad = (
            None if paper_quad is None else np.asarray(paper_quad, dtype=np.float32)
        )
        self.active_quad = (
            None if active_quad is None else np.asarray(active_quad, dtype=np.float32)
        )
        self.paper_orientation = validate_paper_orientation(paper_orientation)
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.threshold = float(threshold)
        self.reason = str(reason)
        # 复制顶层字典，避免调用方后续替换统计字段时改变已经返回的定位结果。
        self.diagnostics = dict(diagnostics or {})

    @classmethod
    def failed(cls, reason, threshold=0.0, confidence=0.0, diagnostics=None):
        """构造不携带四角的失败结果，并保留可选的候选门诊断统计。"""
        return cls(
            False,
            paper_quad=None,
            active_quad=None,
            confidence=confidence,
            threshold=threshold,
            reason=reason,
            diagnostics=diagnostics,
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


def infer_paper_orientation(paper_quad):
    """根据有序A4两组对边平均像素长度判断横放或竖放。

    主要流程：先统一四角顺序，再分别平均上/下边和左/右边长度；水平组更长返回
    landscape，否则返回portrait。使用对边均值可减小透视倾斜时远端单边缩短的影响。
    """
    ordered_quad = order_a4_quad(paper_quad)
    edge_lengths = [
        float(np.linalg.norm(ordered_quad[(index + 1) % 4] - ordered_quad[index]))
        for index in range(4)
    ]
    horizontal_average = (edge_lengths[0] + edge_lengths[2]) * 0.5
    vertical_average = (edge_lengths[1] + edge_lengths[3]) * 0.5
    if horizontal_average > vertical_average:
        return PAPER_ORIENTATION_LANDSCAPE
    return PAPER_ORIENTATION_PORTRAIT


def _machine_orientation_from_camera(observed_orientation, camera_mount_direction):
    """把画面横竖方向转换为固定机械纸面方向。

    关键参数：observed_orientation为蓝框在相机画面中的H/V；安装方向由枚举明确
    90度正负和180度。侧装的奇数四分之一圈使H与V互换，顶置或倒置保持不变。
    返回值：供工作区域、Homography和UART共同使用的逻辑PAPER方向。
    """
    orientation = validate_paper_orientation(observed_orientation)
    direction = validate_camera_mount_direction(camera_mount_direction)
    quarter_turns = abs(int(CAMERA_MOUNT_QUAD_SHIFTS[direction]))
    if quarter_turns % 2 == 0:
        return orientation
    if orientation == PAPER_ORIENTATION_LANDSCAPE:
        return PAPER_ORIENTATION_PORTRAIT
    return PAPER_ORIENTATION_LANDSCAPE


def orient_a4_quad_for_coordinates(
    paper_quad,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """按纸面逻辑方向返回毫米坐标系使用的四角顺序。

    主要流程：先按画面左上、右上、右下、左下规范蓝框，再比较蓝框的画面方向与
    固定安装方向先确定机械原点和Y轴正方向，再比较自动机械方向与用户保存的PAPER
    方向。手动方向不一致时额外旋转一位作为兜底。这样两个侧装方向不会共用一个
    模糊的90度布尔值，Homography也不会只交换210和297两个数值。

    关键参数：paper_quad为相机像素四角；paper_orientation为用户确认的逻辑方向；
    camera_mount_direction为空时读取config.py固定值，测试或离线工具可显式覆盖。
    返回值：与物理(0,0)、(W,0)、(W,H)、(0,H)一一对应的4x2 float32四角。
    循环移位始终保持四边形绕序，因此只旋转坐标系，不会镜像轮廓或反转角度手性。
    """
    ordered_quad = order_a4_quad(paper_quad)
    requested_orientation = validate_paper_orientation(paper_orientation)
    observed_orientation = infer_paper_orientation(ordered_quad)
    direction = validate_camera_mount_direction(
        DEFAULT_CONFIG.get("camera_mount_direction", CAMERA_MOUNT_TOP)
        if camera_mount_direction is None
        else camera_mount_direction
    )
    machine_orientation = _machine_orientation_from_camera(
        observed_orientation,
        direction,
    )
    corner_shift = int(CAMERA_MOUNT_QUAD_SHIFTS[direction])
    if requested_orientation != machine_orientation:
        # PAPER手动兜底选择另一组物理轴；安装方向仍决定90度旋转的正负。
        corner_shift += 1 if corner_shift >= 0 else -1
    return np.roll(ordered_quad, corner_shift, axis=0).astype(np.float32)


def _physical_to_image_homography(
    paper_quad,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """构造完整A4毫米平面到相机四边形的单应性矩阵。

    主要流程：规范四角顺序，建立210×297毫米标准平面并求透视矩阵，再检查矩阵
    是否有限和可逆。返回值：``(矩阵, 有序四角)``；退化输入抛出 ValueError。
    """
    orientation = validate_paper_orientation(paper_orientation)
    ordered_quad = orient_a4_quad_for_coordinates(
        paper_quad,
        orientation,
        camera_mount_direction=camera_mount_direction,
    )
    paper_width_mm, paper_height_mm = paper_size_mm(orientation)
    physical_quad = np.float32(
        [
            [0, 0],
            [paper_width_mm, 0],
            [paper_width_mm, paper_height_mm],
            [0, paper_height_mm],
        ]
    )
    matrix = cv2.getPerspectiveTransform(physical_quad, ordered_quad)
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-9:
        raise ValueError("A4四边形无法建立有效单应性矩阵")
    return matrix, ordered_quad


def build_active_quad(
    paper_quad,
    inset_mm=0.0,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """由完整A4四角生成当前默认工作区的有效四边形。

    主要流程：读取当前纸张方向的默认毫米工作区，再把四边整体内缩inset_mm，最后
    通过完整A4单应性映射回相机坐标。A版默认工作区等于完整A4，inset_mm允许0～20mm。
    camera_mount_direction为空时使用设备固定配置；离线顶置图可显式传入top。
    返回值：按左上、右上、右下、左下排序的4×2 float32相机坐标。
    """
    try:
        inset_mm = float(inset_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("inset_mm 必须是0到20之间的数字") from error
    if not 0.0 <= inset_mm <= MAX_INSET_MM:
        raise ValueError("inset_mm 必须位于0到20之间")

    orientation = validate_paper_orientation(paper_orientation)
    matrix, _ = _physical_to_image_homography(
        paper_quad,
        orientation,
        camera_mount_direction=camera_mount_direction,
    )
    work_x_mm, work_y_mm, work_width_mm, work_height_mm = default_work_region_mm(
        orientation
    )
    active_physical = np.float32(
        [
            [work_x_mm + inset_mm, work_y_mm + inset_mm],
            [work_x_mm + work_width_mm - inset_mm, work_y_mm + inset_mm],
            [
                work_x_mm + work_width_mm - inset_mm,
                work_y_mm + work_height_mm - inset_mm,
            ],
            [work_x_mm + inset_mm, work_y_mm + work_height_mm - inset_mm],
        ]
    ).reshape(1, 4, 2)
    active_quad = cv2.perspectiveTransform(active_physical, matrix)[0]
    if not np.all(np.isfinite(active_quad)):
        raise ValueError("A4四边形映射出的机械有效区无效")
    return active_quad.astype(np.float32)


def build_work_quad(
    paper_quad,
    work_region_mm,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """把X/Y/W/H毫米机械区域通过完整A4单应性映射成相机四边形。

    关键参数：paper_quad为蓝色完整A4四角，work_region_mm为机械区域毫米矩形。
    返回值：左上、右上、右下、左下顺序的4×2 float32相机坐标。
    """
    work_x_mm, work_y_mm, work_width_mm, work_height_mm = validate_work_region_mm(
        work_region_mm,
        paper_orientation,
    )
    matrix, _ = _physical_to_image_homography(paper_quad, paper_orientation)
    physical_quad = np.float32(
        [
            [work_x_mm, work_y_mm],
            [work_x_mm + work_width_mm, work_y_mm],
            [work_x_mm + work_width_mm, work_y_mm + work_height_mm],
            [work_x_mm, work_y_mm + work_height_mm],
        ]
    ).reshape(1, 4, 2)
    work_quad = cv2.perspectiveTransform(physical_quad, matrix)[0]
    if not np.all(np.isfinite(work_quad)):
        raise ValueError("机械区域映射出的相机四边形无效")
    return work_quad.astype(np.float32)


def paper_points_to_image_px(
    points_mm,
    paper_quad,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """把一个或多个完整A4毫米点批量映射为相机像素坐标。

    关键参数：points_mm必须能转换为N×2数组；paper_quad为完整A4蓝框。
    返回值：N×2 float32数组，便于绘制分界线、目标轮廓和机械箭头。
    """
    points_array = np.asarray(points_mm, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points_mm必须是N×2毫米坐标")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("points_mm必须包含有限数字")
    matrix, _ = _physical_to_image_homography(
        paper_quad,
        paper_orientation,
        camera_mount_direction=camera_mount_direction,
    )
    mapped = cv2.perspectiveTransform(points_array.reshape(1, -1, 2), matrix)[0]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("毫米点无法映射到相机坐标")
    return mapped.astype(np.float32)


def paper_point_to_image_px(
    point_mm,
    paper_quad,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """把单个完整A4毫米点映射为相机像素浮点元组。"""
    mapped = paper_points_to_image_px(
        [point_mm],
        paper_quad,
        paper_orientation,
        camera_mount_direction=camera_mount_direction,
    )[0]
    return float(mapped[0]), float(mapped[1])


def build_split_segment(
    paper_quad,
    work_region_mm,
    split_y_mm,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """把机械区域内的水平毫米分界线映射为两个相机端点。"""
    work_x_mm, _, work_width_mm, _ = validate_work_region_mm(
        work_region_mm,
        paper_orientation,
    )
    split_y_mm = validate_split_y_mm(
        work_region_mm,
        split_y_mm,
        paper_orientation,
    )
    return paper_points_to_image_px(
        [
            (work_x_mm, split_y_mm),
            (work_x_mm + work_width_mm, split_y_mm),
        ],
        paper_quad,
        paper_orientation,
    )


def image_point_to_paper_mm(
    point,
    paper_quad,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """把原相机像素点反算为完整A4的毫米坐标。

    关键参数：point为``(x, y)``相机坐标；paper_quad为已锁定完整A4四角；
    camera_mount_direction为空时使用固定设备配置，测试可显式覆盖。
    返回值：``(x_mm, y_mm)``浮点元组，原点位于安装方向定义的A4逻辑起点。
    """
    _, ordered_quad = _physical_to_image_homography(
        paper_quad,
        paper_orientation,
        camera_mount_direction=camera_mount_direction,
    )
    paper_width_mm, paper_height_mm = paper_size_mm(paper_orientation)
    physical_quad = np.float32(
        [
            [0, 0],
            [paper_width_mm, 0],
            [paper_width_mm, paper_height_mm],
            [0, paper_height_mm],
        ]
    )
    inverse_matrix = cv2.getPerspectiveTransform(ordered_quad, physical_quad)
    point_array = np.asarray(point, dtype=np.float32)
    if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
        raise ValueError("point 必须包含两个有限坐标")
    mapped = cv2.perspectiveTransform(point_array.reshape(1, 1, 2), inverse_matrix)[0, 0]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("相机点无法映射到A4毫米坐标")
    return float(mapped[0]), float(mapped[1])


def image_points_to_paper_mm(
    points,
    paper_quad,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
    camera_mount_direction=None,
):
    """把一个或多个相机像素点批量反算为完整A4毫米坐标。

    主要流程：用完整A4四角构造一次逆单应矩阵并批量透视变换，避免逐顶点重复求矩阵。
    关键参数：points必须为N×2有限坐标；camera_mount_direction决定纸面逻辑原点和
    两轴方向，为空时使用固定设备配置。返回值：N×2 float32毫米数组。
    """
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points必须是N×2相机坐标")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("points必须包含有限坐标")
    _, ordered_quad = _physical_to_image_homography(
        paper_quad,
        paper_orientation,
        camera_mount_direction=camera_mount_direction,
    )
    paper_width_mm, paper_height_mm = paper_size_mm(paper_orientation)
    physical_quad = np.float32(
        [
            [0, 0],
            [paper_width_mm, 0],
            [paper_width_mm, paper_height_mm],
            [0, paper_height_mm],
        ]
    )
    inverse_matrix = cv2.getPerspectiveTransform(ordered_quad, physical_quad)
    mapped = cv2.perspectiveTransform(points_array.reshape(1, -1, 2), inverse_matrix)[0]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("相机点无法批量映射到A4毫米坐标")
    return mapped.astype(np.float32)


def _candidate_quad_with_vertex_count(hull):
    """从暗色凸包拟合四边形，并返回近似顶点数供失败诊断。

    主要流程：按凸包周长2%执行多边形近似，只有恰好四个顶点且四边形有效时返回
    排序后的角点。返回值为``(quad, vertex_count)``；失败时quad为None，顶点数仍用于
    判断暗色轮廓是与龙门架粘连，还是尚未形成稳定纸张四角。
    """
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1.0:
        return None, 0
    approximation = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    vertex_count = int(len(approximation))
    if vertex_count != 4:
        return None, vertex_count
    try:
        return order_a4_quad(approximation.reshape(4, 2)), vertex_count
    except ValueError:
        return None, vertex_count


def _candidate_quad(hull):
    """兼容旧调用：只返回暗色凸包拟合出的四边形或None。"""
    quad, _vertex_count = _candidate_quad_with_vertex_count(hull)
    return quad


def _score_candidate(contour, hull, quad, gray, area_ratio, config):
    """按A4比例、矩形度、凸性、面积与内部暗度计算0～1候选分数。

    权重设计：A4长宽比最能排除龙门架细杆，权重0.42；矩形度0.18；凸性0.10；
    可见面积0.10；纸内暗度0.20。内部白色碎片只降低暗度项，不改变外轮廓。
    返回值：``(总分, 矩形度, 指标字典)``。指标字典只用于AUTO单次日志，不改变
    原有加权公式、硬门槛或候选排序。
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
    confidence = float(np.clip(confidence, 0.0, 1.0))
    metrics = {
        "area_ratio": float(area_ratio),
        "observed_aspect": float(observed_aspect),
        "aspect_score": float(aspect_score),
        "rectangularity": float(rectangularity),
        "convexity": float(convexity),
        "darkness_score": float(darkness_score),
        "confidence": confidence,
    }
    return confidence, rectangularity, metrics


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

    # 诊断统计覆盖每一道硬门，但只在单次AUTO结束后由main.py打印一行，不在轮廓循环
    # 中直接输出，避免现场控制台被逐候选日志淹没。
    diagnostics = {
        "contour_count": int(len(contours)),
        "area_small_count": 0,
        "area_large_count": 0,
        "not_quad_count": 0,
        "rectangularity_reject_count": 0,
        "eligible_count": 0,
        "largest_area_ratio": 0.0,
        "approx_vertex_counts": {},
        "best_candidate": None,
    }
    diagnostic_best_confidence = -1.0

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        area_ratio = contour_area / frame_area
        diagnostics["largest_area_ratio"] = max(
            float(diagnostics["largest_area_ratio"]),
            float(area_ratio),
        )
        if area_ratio < min_area_ratio:
            diagnostics["area_small_count"] += 1
            continue
        if area_ratio > max_area_ratio:
            diagnostics["area_large_count"] += 1
            continue

        hull = cv2.convexHull(contour)
        quad, vertex_count = _candidate_quad_with_vertex_count(hull)
        if quad is None:
            diagnostics["not_quad_count"] += 1
            vertex_counts = diagnostics["approx_vertex_counts"]
            vertex_counts[vertex_count] = int(vertex_counts.get(vertex_count, 0)) + 1
            continue
        confidence, rectangularity, metrics = _score_candidate(
            contour,
            hull,
            quad,
            gray,
            area_ratio,
            merged_config,
        )
        # 即使候选随后因矩形度被拒绝，也保留分数最高的一组指标，便于现场区分
        # “已经得到四角但矩形度不足”和“从未得到四角”。
        if confidence > diagnostic_best_confidence:
            diagnostics["best_candidate"] = dict(metrics)
            diagnostic_best_confidence = float(confidence)
        if rectangularity < float(merged_config["paper_min_rectangularity"]):
            diagnostics["rectangularity_reject_count"] += 1
            continue
        diagnostics["eligible_count"] += 1
        if confidence > best_confidence:
            best_quad = quad
            best_confidence = confidence

    min_confidence = float(merged_config["paper_min_confidence"])
    if best_quad is None:
        return PaperLocation.failed(
            "no_candidate",
            threshold=threshold,
            diagnostics=diagnostics,
        )
    if best_confidence < min_confidence:
        return PaperLocation.failed(
            "low_confidence",
            threshold=threshold,
            confidence=best_confidence,
            diagnostics=diagnostics,
        )
    observed_orientation = infer_paper_orientation(best_quad)
    paper_orientation = _machine_orientation_from_camera(
        observed_orientation,
        merged_config.get("camera_mount_direction", CAMERA_MOUNT_TOP),
    )
    return PaperLocation(
        True,
        paper_quad=best_quad,
        active_quad=None,
        paper_orientation=paper_orientation,
        confidence=best_confidence,
        threshold=threshold,
        reason="ok",
        diagnostics=diagnostics,
    )
