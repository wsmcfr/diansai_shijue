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


def _validate_paper_quad_epsilon_ratios(raw_ratios):
    """校验并返回AUTO ROI由严格到宽松的四角拟合比例。

    关键参数raw_ratios必须是非空可迭代对象；每项转换为浮点数后必须有限、位于
    ``(0, 0.10]``且严格递增。返回值为浮点元组。这样现场修改宏时不会因重复、倒序
    或过大的epsilon静默把任意暗色物体强行简化成四边形。
    """
    try:
        ratios = tuple(float(value) for value in raw_ratios)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "paper_quad_epsilon_ratios必须是有限正数序列"
        ) from error
    if not ratios:
        raise ValueError("paper_quad_epsilon_ratios不能为空")
    previous = 0.0
    for ratio in ratios:
        if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 0.10:
            raise ValueError(
                "paper_quad_epsilon_ratios每项必须位于0到0.10之间"
            )
        if ratio <= previous:
            raise ValueError("paper_quad_epsilon_ratios必须严格递增")
        previous = ratio
    return ratios


def _validate_threshold_offsets(raw_offsets):
    """校验并规范化AUTO ROI多阈值偏移序列。

    关键参数raw_offsets必须是非空、非字符串的可迭代对象；每项必须是有限整数且位于
    ``[-127, 127]``。返回值会按用户顺序去重，并在缺少0时把0插入首位，确保严格
    Otsu阈值也能进入后续宽松验收。非法配置明确指出配置键名，避免现场静默失效。
    """
    if isinstance(raw_offsets, (str, bytes)):
        raise ValueError("paper_auto_threshold_offsets必须是整数序列")
    try:
        raw_values = tuple(raw_offsets)
    except TypeError as error:
        raise ValueError("paper_auto_threshold_offsets必须是整数序列") from error
    if not raw_values:
        raise ValueError("paper_auto_threshold_offsets不能为空")

    offsets = []
    for raw_value in raw_values:
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "paper_auto_threshold_offsets必须只包含有限整数"
            ) from error
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("paper_auto_threshold_offsets必须只包含有限整数")
        offset = int(numeric_value)
        if not -127 <= offset <= 127:
            raise ValueError("paper_auto_threshold_offsets每项必须位于-127到127")
        if offset not in offsets:
            offsets.append(offset)
    if 0 not in offsets:
        offsets.insert(0, 0)
    return tuple(offsets)


def _validate_ratio_config(config, key, allow_zero=True, upper=1.0):
    """校验配置中的有限比例并返回浮点值。

    关键参数key用于读取配置并构造可定位的错误信息；allow_zero决定下界是否允许0；
    upper为闭区间上界。返回值是规范浮点数，非法值抛出ValueError。
    """
    try:
        value = float(config[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{key}必须是有限比例") from error
    lower_valid = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not lower_valid or value > float(upper):
        lower_text = "0到" if allow_zero else "大于0且不超过"
        raise ValueError(f"{key}必须位于{lower_text}{float(upper):g}")
    return value


def _validate_auto_roi_recovery_config(config):
    """一次性校验多阈值、宽松安全门和旧ROI兜底配置。

    主要流程：规范阈值偏移，逐项校验0～1比例，再校验单边数量和每掩膜候选上限。
    返回规范化字段字典供定位主流程复用；在处理图像前失败，避免非法宏运行到一半。
    """
    validated = {
        "threshold_offsets": _validate_threshold_offsets(
            config["paper_auto_threshold_offsets"]
        ),
        "relaxed_min_area_ratio": _validate_ratio_config(
            config,
            "paper_auto_relaxed_min_area_ratio",
            allow_zero=False,
        ),
        "min_area_to_largest": _validate_ratio_config(
            config,
            "paper_auto_min_area_to_largest",
            allow_zero=False,
        ),
        "relaxed_min_rectangularity": _validate_ratio_config(
            config,
            "paper_auto_relaxed_min_rectangularity",
        ),
        "relaxed_min_aspect_score": _validate_ratio_config(
            config,
            "paper_auto_relaxed_min_aspect_score",
        ),
        "relaxed_min_darkness": _validate_ratio_config(
            config,
            "paper_auto_relaxed_min_darkness",
        ),
        "min_edge_support": _validate_ratio_config(
            config,
            "paper_auto_min_edge_support",
        ),
        "prior_min_iou": _validate_ratio_config(
            config,
            "paper_auto_prior_min_iou",
        ),
        "prior_max_shift_ratio": _validate_ratio_config(
            config,
            "paper_auto_prior_max_shift_ratio",
            allow_zero=False,
            upper=0.25,
        ),
    }

    integer_specs = (
        ("paper_auto_min_supported_sides", 1, 4, "min_supported_sides"),
        ("paper_auto_max_contours_per_mask", 1, 64, "max_contours_per_mask"),
    )
    for key, minimum, maximum, output_key in integer_specs:
        try:
            raw_value = float(config[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{key}必须是整数") from error
        if not np.isfinite(raw_value) or not raw_value.is_integer():
            raise ValueError(f"{key}必须是整数")
        value = int(raw_value)
        if not minimum <= value <= maximum:
            raise ValueError(f"{key}必须位于{minimum}到{maximum}")
        validated[output_key] = value
    return validated


def _candidate_quad_with_vertex_count(hull, epsilon_ratios):
    """按严格到宽松的多个epsilon拟合四边形，并返回本次诊断信息。

    主要流程：第一个比例是严格路径；只有它不是有效四角时才继续后续比例。遇到第一
    个有效四角立即返回，避免干净纸张无条件使用宽松近似。返回值为
    ``(quad, strict_vertex_count, selected_epsilon)``；完全失败时quad和epsilon为None，
    strict_vertex_count仍用于判断原始轮廓是5角、6角还是其它形状。
    """
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1.0:
        return None, 0, None
    strict_vertex_count = 0
    for index, epsilon_ratio in enumerate(epsilon_ratios):
        approximation = cv2.approxPolyDP(
            hull,
            float(epsilon_ratio) * perimeter,
            True,
        )
        vertex_count = int(len(approximation))
        if index == 0:
            strict_vertex_count = vertex_count
        if vertex_count != 4:
            continue
        try:
            quad = order_a4_quad(approximation.reshape(4, 2))
        except ValueError:
            continue
        return quad, strict_vertex_count, float(epsilon_ratio)
    return None, strict_vertex_count, None


def _candidate_quad(hull):
    """兼容旧调用：只返回暗色凸包拟合出的四边形或None。"""
    epsilon_ratios = _validate_paper_quad_epsilon_ratios(
        DEFAULT_CONFIG["paper_quad_epsilon_ratios"]
    )
    quad, _vertex_count, _epsilon_ratio = _candidate_quad_with_vertex_count(
        hull,
        epsilon_ratios,
    )
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


def _build_dark_mask(blurred_gray, threshold, close_kernel_size):
    """按显式阈值生成暗色掩膜并执行一次闭运算。

    关键参数blurred_gray必须是已模糊的二维灰度图；threshold会限制到1～254，避免
    极端偏移把整帧变成同一颜色；close_kernel_size必须是正奇数。返回值为uint8二值
    掩膜，黑纸和其它暗色结构为255，供严格与宽松候选路径共用。
    """
    clipped_threshold = float(np.clip(float(threshold), 1.0, 254.0))
    _, dark_mask = cv2.threshold(
        blurred_gray,
        clipped_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(close_kernel_size), int(close_kernel_size)),
    )
    return cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)


def _find_mask_contours(dark_mask):
    """提取暗色外轮廓，并返回与轮廓顺序一致的面积列表。

    使用RETR_EXTERNAL保持旧AUTO行为：白色碎片只形成黑纸内部孔洞，不应被当成纸张
    候选。返回值为``(轮廓列表, 面积列表)``，调用方据此计算本掩膜最大暗区。
    """
    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    areas = [float(cv2.contourArea(contour)) for contour in contours]
    return contours, areas


def _new_paper_diagnostics(contour_count, source="STRICT"):
    """建立字段稳定的AUTO ROI候选诊断字典。

    关键参数contour_count是当前严格掩膜的外轮廓数；source标记候选路径。返回字典
    继续兼容旧日志字段，并预留后续宽松路径拒绝计数，避免失败分支缺键。
    """
    return {
        "source": str(source),
        "contour_count": int(contour_count),
        "area_small_count": 0,
        "area_large_count": 0,
        "not_quad_count": 0,
        "rectangularity_reject_count": 0,
        "eligible_count": 0,
        "largest_area_ratio": 0.0,
        "approx_vertex_counts": {},
        "best_candidate": None,
        "relaxed_area_reject_count": 0,
        "relaxed_rectangularity_reject_count": 0,
        "relaxed_aspect_reject_count": 0,
        "relaxed_darkness_reject_count": 0,
        "relaxed_edge_reject_count": 0,
        "prior_mismatch_reject_count": 0,
    }


def _search_strict_paper_candidate(
    gray,
    dark_mask,
    threshold,
    config,
    epsilon_ratios,
):
    """使用旧硬门搜索当前掩膜中的严格A4候选。

    主要流程完整保留旧面积、凸包四角、矩形度、加权置信度和诊断排序；只把原先
    位于``locate_black_paper``中的循环提取为可测试单元。返回
    ``(最佳四角或None, 最佳置信度, 诊断字典)``，低置信度候选仍返回四角供调用方
    区分``low_confidence``与``no_candidate``。
    """
    contours, contour_areas = _find_mask_contours(dark_mask)
    diagnostics = _new_paper_diagnostics(len(contours), source="STRICT")
    frame_area = float(gray.shape[0] * gray.shape[1])
    min_area_ratio = float(config["paper_min_area_ratio"])
    max_area_ratio = float(config["paper_max_area_ratio"])
    best_quad = None
    best_confidence = 0.0
    diagnostic_best_confidence = -1.0

    for contour, contour_area in zip(contours, contour_areas):
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
        quad, vertex_count, selected_epsilon = _candidate_quad_with_vertex_count(
            hull,
            epsilon_ratios,
        )
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
            config,
        )
        metrics["strict_vertex_count"] = int(vertex_count)
        metrics["quad_epsilon_ratio"] = float(selected_epsilon)
        metrics["used_threshold"] = float(threshold)
        if confidence > diagnostic_best_confidence:
            diagnostics["best_candidate"] = dict(metrics)
            diagnostic_best_confidence = float(confidence)
        if rectangularity < float(config["paper_min_rectangularity"]):
            diagnostics["rectangularity_reject_count"] += 1
            continue
        diagnostics["eligible_count"] += 1
        if confidence > best_confidence:
            best_quad = quad
            best_confidence = confidence

    largest_area_ratio = float(diagnostics["largest_area_ratio"])
    best_metrics = diagnostics.get("best_candidate")
    if isinstance(best_metrics, dict):
        area_to_largest = (
            0.0
            if largest_area_ratio <= 1e-9
            else float(best_metrics["area_ratio"]) / largest_area_ratio
        )
        best_metrics["area_to_largest"] = float(
            np.clip(area_to_largest, 0.0, 1.0)
        )

    # 即使四角来自严格2%，当候选远小于同帧主暗区时也不能立即锁定。这个门只排除
    # 明显的相对小框，不提高旧绝对面积下限，因此远距离但画面中没有更大暗区的A4
    # 仍保持旧行为。
    if best_quad is not None and isinstance(best_metrics, dict):
        if best_metrics["area_to_largest"] < float(
            config["paper_auto_min_area_to_largest"]
        ):
            diagnostics["relaxed_area_reject_count"] += 1
            best_quad = None
            best_confidence = 0.0

    return best_quad, float(best_confidence), diagnostics


def _build_supported_edge_map(gray):
    """生成带少量位置容差的灰度边缘图。

    主要流程：按全帧中位灰度生成自适应Canny阈值，再把边缘膨胀为约画面短边1%的
    容差带。返回uint8二值图，供候选四边验收和旧ROI局部修正共用，避免重复计算。
    """
    gray_array = np.asarray(gray, dtype=np.uint8)
    median_gray = float(np.median(gray_array))
    canny_lower = int(np.clip(0.50 * median_gray, 10, 220))
    canny_upper = int(np.clip(1.25 * median_gray, canny_lower + 1, 250))
    edge_map = cv2.Canny(gray_array, canny_lower, canny_upper)
    tolerance_px = max(2, int(round(min(gray_array.shape[:2]) * 0.01)))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance_px + 1, 2 * tolerance_px + 1),
    )
    return cv2.dilate(edge_map, dilation_kernel)


def _measure_line_support(supported_edge_map, start, end):
    """返回一条像素线在膨胀边缘图中的命中比例。

    关键参数start/end是二维像素坐标；退化线返回0。该函数只分配单通道线掩膜，
    不重复运行Canny，便于旧ROI在多个法向偏移上快速比较。
    """
    start_point = np.rint(start).astype(np.int32)
    end_point = np.rint(end).astype(np.int32)
    line_mask = np.zeros(supported_edge_map.shape, dtype=np.uint8)
    cv2.line(
        line_mask,
        tuple(int(value) for value in start_point),
        tuple(int(value) for value in end_point),
        255,
        1,
        cv2.LINE_8,
    )
    line_pixels = int(cv2.countNonZero(line_mask))
    if line_pixels <= 1:
        return 0.0
    supported_pixels = int(
        cv2.countNonZero(cv2.bitwise_and(line_mask, supported_edge_map))
    )
    return float(np.clip(supported_pixels / float(line_pixels), 0.0, 1.0))


def _measure_quad_edge_support(gray, quad, supported_edge_map=None):
    """测量候选四边附近真实灰度边缘的支持比例。

    可选supported_edge_map允许多候选共用一次Canny结果；未传时由gray现场生成。
    返回值为``(四边平均值, 四项单边值)``；只验证已有候选四边，不做全帧直线搜索，
    减少龙门架对AUTO的干扰。
    """
    ordered_quad = order_a4_quad(quad)
    if supported_edge_map is None:
        supported_edge_map = _build_supported_edge_map(gray)

    side_supports = []
    for index in range(4):
        side_supports.append(
            _measure_line_support(
                supported_edge_map,
                ordered_quad[index],
                ordered_quad[(index + 1) % 4],
            )
        )
    return float(np.mean(side_supports)), tuple(side_supports)


def _quad_fill_ratio(contour_area, quad):
    """返回暗色轮廓面积占候选四边形面积的比例，退化四边形返回0。"""
    quad_area = abs(float(cv2.contourArea(np.asarray(quad, dtype=np.float32))))
    if quad_area <= 1.0:
        return 0.0
    return float(np.clip(float(contour_area) / quad_area, 0.0, 1.0))


def _search_relaxed_paper_candidate(
    gray,
    blurred,
    otsu_threshold,
    close_kernel_size,
    config,
    recovery_config,
    epsilon_ratios,
    diagnostics,
):
    """扫描有限阈值并使用联合安全门恢复主A4轮廓。

    主要流程：为每个去重后的阈值生成暗色掩膜，只处理面积最大的有限轮廓；候选
    必须依次通过绝对面积、相对最大轮廓面积、宽松矩形度、A4比例、内部暗度和四边
    边缘支持。返回``(四角或None, 置信度, 实际阈值)``，并原位补充诊断字典。
    """
    frame_area = float(gray.shape[0] * gray.shape[1])
    max_area_ratio = float(config["paper_max_area_ratio"])
    min_confidence = float(config["paper_min_confidence"])
    best_quad = None
    best_confidence = 0.0
    best_threshold = float(otsu_threshold)
    diagnostic_best_confidence = float(
        (diagnostics.get("best_candidate") or {}).get("confidence", -1.0)
    )
    visited_thresholds = set()
    # 边缘图只与原始灰度帧有关，与扫描阈值无关；单次计算后供全部候选复用，避免
    # 最坏20个候选重复运行Canny拖慢MaixCAM2触摸响应。
    supported_edge_map = _build_supported_edge_map(gray)

    for offset in recovery_config["threshold_offsets"]:
        threshold = float(np.clip(float(otsu_threshold) + int(offset), 1.0, 254.0))
        threshold_key = int(round(threshold))
        if threshold_key in visited_thresholds:
            continue
        visited_thresholds.add(threshold_key)
        dark_mask = _build_dark_mask(blurred, threshold, close_kernel_size)
        contours, contour_areas = _find_mask_contours(dark_mask)
        if not contour_areas:
            continue
        largest_contour_area = max(contour_areas)
        diagnostics["largest_area_ratio"] = max(
            float(diagnostics["largest_area_ratio"]),
            largest_contour_area / frame_area,
        )
        ordered_candidates = sorted(
            zip(contours, contour_areas),
            key=lambda item: item[1],
            reverse=True,
        )[: recovery_config["max_contours_per_mask"]]

        for contour, contour_area in ordered_candidates:
            area_ratio = float(contour_area / frame_area)
            area_to_largest = float(
                contour_area / max(largest_contour_area, 1.0)
            )
            if (
                area_ratio < recovery_config["relaxed_min_area_ratio"]
                or area_ratio > max_area_ratio
                or area_to_largest < recovery_config["min_area_to_largest"]
            ):
                diagnostics["relaxed_area_reject_count"] += 1
                continue

            hull = cv2.convexHull(contour)
            quad, vertex_count, selected_epsilon = _candidate_quad_with_vertex_count(
                hull,
                epsilon_ratios,
            )
            if quad is None:
                continue
            _old_confidence, rectangularity, metrics = _score_candidate(
                contour,
                hull,
                quad,
                gray,
                area_ratio,
                config,
            )
            edge_support, side_supports = _measure_quad_edge_support(
                gray,
                quad,
                supported_edge_map=supported_edge_map,
            )
            supported_side_count = sum(
                support >= recovery_config["min_edge_support"]
                for support in side_supports
            )
            recovery_confidence = float(
                np.clip(
                    0.30 * metrics["aspect_score"]
                    + 0.30 * edge_support
                    + 0.15 * area_to_largest
                    + 0.15 * metrics["darkness_score"]
                    + 0.10 * rectangularity,
                    0.0,
                    1.0,
                )
            )
            metrics.update(
                {
                    "confidence": recovery_confidence,
                    "strict_vertex_count": int(vertex_count),
                    "quad_epsilon_ratio": float(selected_epsilon),
                    "used_threshold": float(threshold),
                    "area_to_largest": float(area_to_largest),
                    "quad_fill": _quad_fill_ratio(contour_area, quad),
                    "edge_support": float(edge_support),
                    "supported_side_count": int(supported_side_count),
                }
            )
            if recovery_confidence > diagnostic_best_confidence:
                diagnostics["best_candidate"] = dict(metrics)
                diagnostic_best_confidence = recovery_confidence

            if rectangularity < recovery_config["relaxed_min_rectangularity"]:
                diagnostics["relaxed_rectangularity_reject_count"] += 1
                continue
            if metrics["aspect_score"] < recovery_config["relaxed_min_aspect_score"]:
                diagnostics["relaxed_aspect_reject_count"] += 1
                continue
            if metrics["darkness_score"] < recovery_config["relaxed_min_darkness"]:
                diagnostics["relaxed_darkness_reject_count"] += 1
                continue
            if (
                edge_support < recovery_config["min_edge_support"]
                or supported_side_count < recovery_config["min_supported_sides"]
            ):
                diagnostics["relaxed_edge_reject_count"] += 1
                continue
            if recovery_confidence < min_confidence:
                continue
            diagnostics["eligible_count"] += 1
            if recovery_confidence > best_confidence:
                best_quad = quad
                best_confidence = recovery_confidence
                best_threshold = threshold

    if best_quad is not None:
        diagnostics["source"] = "TH_SCAN"
    return best_quad, float(best_confidence), float(best_threshold)


def _convex_quad_iou(left_quad, right_quad):
    """计算两个有效凸四边形的交并比，退化或无交集时返回0。"""
    try:
        left = order_a4_quad(left_quad).astype(np.float32)
        right = order_a4_quad(right_quad).astype(np.float32)
    except (TypeError, ValueError):
        return 0.0
    left_area = abs(float(cv2.contourArea(left)))
    right_area = abs(float(cv2.contourArea(right)))
    if left_area <= 1.0 or right_area <= 1.0:
        return 0.0
    try:
        intersection_area, _ = cv2.intersectConvexConvex(left, right)
    except cv2.error:
        return 0.0
    union_area = left_area + right_area - float(intersection_area)
    if union_area <= 1.0:
        return 0.0
    return float(np.clip(float(intersection_area) / union_area, 0.0, 1.0))


def _intersect_infinite_lines(first_start, first_end, second_start, second_end):
    """求两条无限直线交点，近平行或坐标非法时返回None。"""
    point = np.asarray(first_start, dtype=np.float64)
    direction = np.asarray(first_end, dtype=np.float64) - point
    other_point = np.asarray(second_start, dtype=np.float64)
    other_direction = np.asarray(second_end, dtype=np.float64) - other_point
    # NumPy 2已弃用二维向量np.cross；显式计算标量叉积也能避免MaixPy版本差异。
    cross_value = float(
        direction[0] * other_direction[1]
        - direction[1] * other_direction[0]
    )
    if abs(cross_value) <= 1e-6:
        return None
    relative = other_point - point
    relative_cross = float(
        relative[0] * other_direction[1]
        - relative[1] * other_direction[0]
    )
    factor = relative_cross / cross_value
    intersection = point + factor * direction
    if not np.all(np.isfinite(intersection)):
        return None
    return intersection.astype(np.float32)


def _refine_prior_quad(gray, prior_quad, max_shift_ratio):
    """在旧ROI四边法向的小范围内寻找当前帧最强边缘并重建四角。

    主要流程：每条旧边保持方向不变，只沿法向扫描有限整数偏移；同分时优先最小
    位移，避免被附近碎片边缘拉走。随后由相邻修正线求交得到四角。返回有效凸四边形
    或None；该函数只提出候选，最终仍由IoU、比例、暗度和边缘门验收。
    """
    try:
        ordered_prior = order_a4_quad(prior_quad)
    except (TypeError, ValueError):
        return None
    supported_edge_map = _build_supported_edge_map(gray)
    max_shift_px = max(
        2,
        int(round(min(gray.shape[:2]) * float(max_shift_ratio))),
    )
    refined_lines = []
    for index in range(4):
        start = ordered_prior[index].astype(np.float64)
        end = ordered_prior[(index + 1) % 4].astype(np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1.0:
            return None
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64) / length
        best_line = None
        best_key = (-1.0, float("-inf"))
        for offset in range(-max_shift_px, max_shift_px + 1):
            shift = normal * float(offset)
            shifted_start = start + shift
            shifted_end = end + shift
            support = _measure_line_support(
                supported_edge_map,
                shifted_start,
                shifted_end,
            )
            candidate_key = (support, -abs(offset))
            if candidate_key > best_key:
                best_key = candidate_key
                best_line = (shifted_start, shifted_end)
        if best_line is None:
            return None
        refined_lines.append(best_line)

    intersections = []
    adjacent_line_pairs = ((3, 0), (0, 1), (1, 2), (2, 3))
    for previous_index, next_index in adjacent_line_pairs:
        previous_line = refined_lines[previous_index]
        next_line = refined_lines[next_index]
        intersection = _intersect_infinite_lines(
            previous_line[0],
            previous_line[1],
            next_line[0],
            next_line[1],
        )
        if intersection is None:
            return None
        intersections.append(intersection)
    try:
        return order_a4_quad(np.asarray(intersections, dtype=np.float32))
    except ValueError:
        return None


def _search_prior_edge_candidate(
    gray,
    prior_quad,
    threshold,
    config,
    recovery_config,
    diagnostics,
):
    """使用当前图像证据验收旧ROI附近的局部边缘修正候选。

    返回``(四角或None, 置信度)``。旧ROI只限定搜索范围并参与IoU评分；没有足够
    当前边缘、A4比例或暗度时增加拒绝计数并返回None，不能直接复用历史蓝框。
    """
    refined_quad = _refine_prior_quad(
        gray,
        prior_quad,
        recovery_config["prior_max_shift_ratio"],
    )
    if refined_quad is None:
        diagnostics["prior_mismatch_reject_count"] += 1
        return None, 0.0

    contour = np.rint(refined_quad).astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    area_ratio = abs(float(cv2.contourArea(contour))) / float(gray.size)
    _old_confidence, rectangularity, metrics = _score_candidate(
        contour,
        hull,
        refined_quad,
        gray,
        area_ratio,
        config,
    )
    edge_support, side_supports = _measure_quad_edge_support(gray, refined_quad)
    supported_side_count = sum(
        support >= recovery_config["min_edge_support"]
        for support in side_supports
    )
    prior_iou = _convex_quad_iou(prior_quad, refined_quad)
    confidence = float(
        np.clip(
            0.30 * metrics["aspect_score"]
            + 0.30 * edge_support
            + 0.20 * metrics["darkness_score"]
            + 0.20 * prior_iou,
            0.0,
            1.0,
        )
    )
    metrics.update(
        {
            "confidence": confidence,
            "strict_vertex_count": 4,
            "quad_epsilon_ratio": 0.0,
            "used_threshold": float(threshold),
            "area_to_largest": 1.0,
            "quad_fill": 1.0,
            "edge_support": float(edge_support),
            "supported_side_count": int(supported_side_count),
            "prior_iou": float(prior_iou),
        }
    )
    diagnostics["best_candidate"] = dict(metrics)

    prior_valid = prior_iou >= recovery_config["prior_min_iou"]
    aspect_valid = (
        metrics["aspect_score"] >= recovery_config["relaxed_min_aspect_score"]
    )
    darkness_valid = (
        metrics["darkness_score"] >= recovery_config["relaxed_min_darkness"]
    )
    edge_valid = (
        edge_support >= recovery_config["min_edge_support"]
        and supported_side_count >= recovery_config["min_supported_sides"]
    )
    confidence_valid = confidence >= float(config["paper_min_confidence"])
    if not edge_valid:
        diagnostics["relaxed_edge_reject_count"] += 1
    if not aspect_valid:
        diagnostics["relaxed_aspect_reject_count"] += 1
    if not darkness_valid:
        diagnostics["relaxed_darkness_reject_count"] += 1
    if not (prior_valid and aspect_valid and darkness_valid and edge_valid and confidence_valid):
        diagnostics["prior_mismatch_reject_count"] += 1
        return None, 0.0

    diagnostics["source"] = "PRIOR_EDGE"
    diagnostics["eligible_count"] += 1
    return refined_quad, confidence


def locate_black_paper(frame_bgr, config=None, prior_quad=None):
    """在整帧中定位最符合黑色A4纸的四边形候选。

    主要流程：校验配置，使用反向Otsu取得基准阈值，再调用严格候选路径执行旧面积、
    四角、矩形度和置信度验收。后续多阈值恢复会在严格失败后接入，不改变严格成功
    行为。关键参数config可覆盖DEFAULT_CONFIG中的``paper_*``参数；prior_quad为相机
    固定场景下上次已保存的完整A4蓝框，只在严格和多阈值都失败后用于局部边缘修正。
    返回值：成功时返回 PaperLocation 四角与置信度；失败时返回不含四角的失败对象。
    """
    _validate_frame(frame_bgr)
    merged_config = dict(DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    close_kernel_size = int(merged_config["paper_close_kernel"])
    if close_kernel_size <= 0 or close_kernel_size % 2 == 0:
        raise ValueError("paper_close_kernel 必须是正奇数")
    epsilon_ratios = _validate_paper_quad_epsilon_ratios(
        merged_config["paper_quad_epsilon_ratios"]
    )
    # 即使本阶段尚未使用宽松结果，也必须在处理图像前验证全部现场宏；这样下一阶段
    # 接入多阈值时不会改变非法配置的失败时机。
    recovery_config = _validate_auto_roi_recovery_config(merged_config)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    dark_mask = _build_dark_mask(
        blurred,
        threshold,
        close_kernel_size,
    )
    strict_quad, strict_confidence, diagnostics = _search_strict_paper_candidate(
        gray,
        dark_mask,
        threshold,
        merged_config,
        epsilon_ratios,
    )

    min_confidence = float(merged_config["paper_min_confidence"])
    if strict_quad is not None and strict_confidence >= min_confidence:
        best_quad = strict_quad
        best_confidence = strict_confidence
        result_threshold = float(threshold)
    else:
        best_quad, best_confidence, result_threshold = (
            _search_relaxed_paper_candidate(
                gray,
                blurred,
                threshold,
                close_kernel_size,
                merged_config,
                recovery_config,
                epsilon_ratios,
                diagnostics,
            )
        )

    if best_quad is None and prior_quad is not None:
        best_quad, best_confidence = _search_prior_edge_candidate(
            gray,
            prior_quad,
            threshold,
            merged_config,
            recovery_config,
            diagnostics,
        )
        if best_quad is not None:
            result_threshold = float(threshold)

    if best_quad is None and strict_quad is None:
        return PaperLocation.failed(
            "no_candidate",
            threshold=threshold,
            diagnostics=diagnostics,
        )
    if best_quad is None:
        return PaperLocation.failed(
            "low_confidence",
            threshold=threshold,
            confidence=strict_confidence,
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
        threshold=result_threshold,
        reason="ok",
        diagnostics=diagnostics,
    )
