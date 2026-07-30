"""现场视觉参数的校验、持久化和运行配置合并。"""

import json
import math
import os

try:
    from maixcam2_app_A_quad.paper_locator import (
        PAPER_ORIENTATION_PORTRAIT,
        default_split_y_mm,
        default_work_region_mm,
        validate_paper_orientation,
        validate_split_y_mm,
        validate_work_region_mm,
    )
except ModuleNotFoundError as error:
    # MaixVision平铺运行时顶层包不存在，仍从main.py同级模块加载物理约束。
    if error.name != "maixcam2_app_A_quad":
        raise
    from paper_locator import (
        PAPER_ORIENTATION_PORTRAIT,
        default_split_y_mm,
        default_work_region_mm,
        validate_paper_orientation,
        validate_split_y_mm,
        validate_work_region_mm,
    )


# V4开始把ROI和A4四角保存为0～1归一化值；V5新增纸张横竖方向字段；V6明确
# 固定相机的四种安装方向。当前设备使用的V5蓝框没有侧装正负方向语义，升级时
# 不能直接沿用到V6；V2～V4继续执行各自已有的历史兼容规则。
# V2/V3设备文件固定来自旧版640x480相机，加载到高分辨率时必须显式等比迁移。
SETTINGS_VERSION = 6
LEGACY_SETTINGS_VERSION = 2
PIXEL_SETTINGS_VERSION = 3
NORMALIZED_SETTINGS_VERSION = 4
ORIENTATION_SETTINGS_VERSION = 5
LEGACY_PIXEL_FRAME_SIZE = (640, 480)

# 只有这些字段允许由触摸调参界面修改并写入持久文件。
RUNTIME_SETTING_KEYS = (
    "roi",
    "paper_quad",
    "paper_orientation",
    "inset_mm",
    "work_x_mm",
    "work_y_mm",
    "work_width_mm",
    "work_height_mm",
    "split_y_mm",
    "fixed_threshold",
    "min_area_ratio",
    "open_kernel",
    "close_kernel",
)

# 形态学核尺寸既是持久参数约束，也是屏幕调参状态机的可选项。
# 两处共同引用这里的元组，避免配置文件能加载但调参界面无法定位当前值。
OPEN_KERNEL_VALUES = (1, 3, 5, 7)
CLOSE_KERNEL_VALUES = (1, 3, 5, 7, 9)
KERNEL_VALUES_BY_KEY = {
    "open_kernel": OPEN_KERNEL_VALUES,
    "close_kernel": CLOSE_KERNEL_VALUES,
}

# 纸张锁定和高级分割拥有不同保存按钮，字段分组是防止两类保存互相覆盖的边界。
PAPER_SETTING_KEYS = (
    "paper_quad",
    "paper_orientation",
    "inset_mm",
    "work_x_mm",
    "work_y_mm",
    "work_width_mm",
    "work_height_mm",
    "split_y_mm",
)
SEGMENTATION_SETTING_KEYS = (
    "fixed_threshold",
    "min_area_ratio",
    "open_kernel",
    "close_kernel",
)


def _resolve_frame_size(config, frame_size=None):
    """返回设置层使用的实际采集尺寸。

    关键参数：frame_size由相机初始化成功后传入；None用于兼容旧调用，读取既有
    camera_width/camera_height。返回值为正整数``(宽, 高)``，非法尺寸抛出ValueError。
    """
    if frame_size is None:
        frame_size = (config["camera_width"], config["camera_height"])
    try:
        frame_width, frame_height = (int(value) for value in frame_size)
    except (TypeError, ValueError) as error:
        raise ValueError("相机画面宽高必须是整数") from error
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("相机画面宽高必须大于零")
    return frame_width, frame_height


def build_default_runtime_settings(config, frame_size=None):
    """从默认视觉配置构造一份独立的现场参数。

    主要流程：读取相机宽高作为整帧ROI，再复制阈值、面积和形态学参数。
    关键参数：config 必须包含 DEFAULT_CONFIG 中对应的相机与视觉字段。
    返回值：只包含可调字段的新字典，调用方修改它不会影响全局默认配置。
    """
    frame_width, frame_height = _resolve_frame_size(config, frame_size)
    default_region = default_work_region_mm(PAPER_ORIENTATION_PORTRAIT)
    return {
        "roi": [
            0,
            0,
            frame_width,
            frame_height,
        ],
        "paper_quad": None,
        "paper_orientation": PAPER_ORIENTATION_PORTRAIT,
        "inset_mm": 0.0,
        "work_x_mm": float(default_region[0]),
        "work_y_mm": float(default_region[1]),
        "work_width_mm": float(default_region[2]),
        "work_height_mm": float(default_region[3]),
        "split_y_mm": float(default_split_y_mm(PAPER_ORIENTATION_PORTRAIT)),
        "fixed_threshold": config.get("fixed_threshold"),
        "min_area_ratio": float(config["min_area_ratio"]),
        "open_kernel": int(config["open_kernel"]),
        "close_kernel": int(config["close_kernel"]),
    }


def _normalize_paper_quad(paper_quad, frame_size):
    """校验并规范化可选的完整A4四角。

    主要流程：允许未锁定时为 None；否则检查4×2数字、有限值、画面边界、连续凸性
    和非零面积。关键参数 frame_size 为 ``(宽, 高)``。
    返回值：None 或由浮点坐标组成的新列表；任何损坏输入抛出 ValueError。
    """
    if paper_quad is None:
        return None
    if not isinstance(paper_quad, (list, tuple)) or len(paper_quad) != 4:
        raise ValueError("paper_quad必须包含四个二维角点")

    frame_width, frame_height = frame_size
    normalized = []
    for point in paper_quad:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("paper_quad每个角点必须包含x和y")
        try:
            point_x, point_y = (float(value) for value in point)
        except (TypeError, ValueError) as error:
            raise ValueError("paper_quad角点必须是数字") from error
        if not math.isfinite(point_x) or not math.isfinite(point_y):
            raise ValueError("paper_quad角点必须是有限数字")
        if not (0.0 <= point_x < frame_width and 0.0 <= point_y < frame_height):
            raise ValueError("paper_quad必须完整位于相机画面内部")
        normalized.append([point_x, point_y])

    cross_products = []
    for index in range(4):
        first = normalized[index]
        second = normalized[(index + 1) % 4]
        third = normalized[(index + 2) % 4]
        vector_a = (second[0] - first[0], second[1] - first[1])
        vector_b = (third[0] - second[0], third[1] - second[1])
        cross_products.append(vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0])

    # 有序凸四边形的四个转向叉积必须同号且不能接近零，交叉或凹陷输入在此拒绝。
    if any(abs(value) <= 1e-6 for value in cross_products) or not (
        all(value > 0 for value in cross_products)
        or all(value < 0 for value in cross_products)
    ):
        raise ValueError("paper_quad必须是按边连续排列的凸四边形")
    return normalized


def validate_runtime_settings(settings, frame_size):
    """校验并规范化现场参数。

    主要流程：检查字段完整性、ROI边界、阈值范围、面积比例和奇数核尺寸。
    关键参数：settings 为待校验字典，frame_size 为 ``(宽, 高)``。
    返回值：规范化后的新字典；任何字段非法时抛出 ValueError，绝不部分接受。
    """
    if not isinstance(settings, dict):
        raise ValueError("现场参数必须是字典")
    missing_keys = [key for key in RUNTIME_SETTING_KEYS if key not in settings]
    if missing_keys:
        raise ValueError(f"现场参数字段不完整: {','.join(missing_keys)}")

    frame_width, frame_height = (int(value) for value in frame_size)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("相机画面宽高必须大于零")

    roi = settings["roi"]
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("ROI必须是[x, y, width, height]")
    try:
        roi_x, roi_y, roi_width, roi_height = (int(value) for value in roi)
    except (TypeError, ValueError) as error:
        raise ValueError("ROI必须包含四个整数") from error
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI宽高必须大于零")
    if (
        roi_x < 0
        or roi_y < 0
        or roi_x + roi_width > frame_width
        or roi_y + roi_height > frame_height
    ):
        raise ValueError("ROI必须完整位于相机画面内部")

    paper_quad = _normalize_paper_quad(
        settings["paper_quad"],
        (frame_width, frame_height),
    )
    paper_orientation = validate_paper_orientation(settings["paper_orientation"])
    try:
        inset_mm = float(settings["inset_mm"])
    except (TypeError, ValueError) as error:
        raise ValueError("inset_mm必须是0到20之间的数字") from error
    if not math.isfinite(inset_mm) or not 0.0 <= inset_mm <= 20.0:
        raise ValueError("inset_mm必须位于0到20之间")

    work_region = validate_work_region_mm(
        (
            settings["work_x_mm"],
            settings["work_y_mm"],
            settings["work_width_mm"],
            settings["work_height_mm"],
        ),
        paper_orientation,
    )
    split_y_mm = validate_split_y_mm(
        work_region,
        settings["split_y_mm"],
        paper_orientation,
    )

    fixed_threshold = settings["fixed_threshold"]
    if fixed_threshold is not None:
        try:
            fixed_threshold = float(fixed_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("固定阈值必须是0到255之间的数字") from error
        if not 0.0 <= fixed_threshold <= 255.0:
            raise ValueError("固定阈值必须位于0到255之间")

    try:
        min_area_ratio = float(settings["min_area_ratio"])
    except (TypeError, ValueError) as error:
        raise ValueError("最小面积比例必须是数字") from error
    if not 0.0 < min_area_ratio < 1.0:
        raise ValueError("最小面积比例必须位于0到1之间")

    kernels = {}
    for key, allowed_values in KERNEL_VALUES_BY_KEY.items():
        try:
            kernel_size = int(settings[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key}必须是允许的正奇数") from error
        if kernel_size not in allowed_values:
            allowed_text = "/".join(str(value) for value in allowed_values)
            raise ValueError(f"{key}必须是{allowed_text}之一")
        kernels[key] = kernel_size

    return {
        "roi": [roi_x, roi_y, roi_width, roi_height],
        "paper_quad": paper_quad,
        "paper_orientation": paper_orientation,
        "inset_mm": inset_mm,
        "work_x_mm": work_region[0],
        "work_y_mm": work_region[1],
        "work_width_mm": work_region[2],
        "work_height_mm": work_region[3],
        "split_y_mm": split_y_mm,
        "fixed_threshold": fixed_threshold,
        "min_area_ratio": min_area_ratio,
        "open_kernel": kernels["open_kernel"],
        "close_kernel": kernels["close_kernel"],
    }


def _merge_owned_settings(current_settings, staged_settings, owned_keys):
    """复制当前设置并只合并指定字段，避免两个保存动作污染对方参数。

    关键参数：current_settings 为已生效设置，staged_settings 为调参工作副本，owned_keys
    指定本次保存的所有权边界。返回值：新的字典，不修改任何输入对象。
    """
    if not isinstance(current_settings, dict) or not isinstance(staged_settings, dict):
        raise ValueError("合并设置必须使用字典")
    merged = dict(current_settings)
    for key in owned_keys:
        if key not in staged_settings:
            raise ValueError(f"待合并设置缺少字段: {key}")
        value = staged_settings[key]
        # 纸张四角是嵌套列表，显式复制可防止后续拖动工作副本时影响已生效设置。
        if key == "paper_quad" and value is not None:
            value = [list(point) for point in value]
        merged[key] = value
    return merged


def merge_paper_settings(current_settings, staged_settings):
    """只合并 ``paper_quad/inset_mm``，对应屏幕上的 LOCK ROI 动作。"""
    return _merge_owned_settings(current_settings, staged_settings, PAPER_SETTING_KEYS)


def merge_segmentation_settings(current_settings, staged_settings):
    """只合并阈值、面积和形态学核，对应 ADV 页的 SAVE 动作。"""
    return _merge_owned_settings(
        current_settings,
        staged_settings,
        SEGMENTATION_SETTING_KEYS,
    )


def _scale_pixel_coordinate_fields(settings, source_frame_size, target_frame_size):
    """把运行设置中的ROI和A4四角从一个像素尺寸等比换算到另一个尺寸。

    主要流程：ROI使用四个独立比例后四舍五入为整数；A4四角保留浮点精度；毫米
    机械区域和分割参数原样复制。返回值为新字典，不修改输入设置。
    """
    source_width, source_height = _resolve_frame_size({}, source_frame_size)
    target_width, target_height = _resolve_frame_size({}, target_frame_size)
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    scaled = dict(settings)
    roi_x, roi_y, roi_width, roi_height = settings["roi"]
    scaled["roi"] = [
        int(round(float(roi_x) * scale_x)),
        int(round(float(roi_y) * scale_y)),
        int(round(float(roi_width) * scale_x)),
        int(round(float(roi_height) * scale_y)),
    ]
    paper_quad = settings.get("paper_quad")
    if paper_quad is not None:
        scaled["paper_quad"] = [
            [float(point[0]) * scale_x, float(point[1]) * scale_y]
            for point in paper_quad
        ]
    return scaled


def _coordinates_to_normalized(settings, frame_size):
    """把合法运行时像素坐标转换成V4磁盘使用的0～1坐标。"""
    frame_width, frame_height = _resolve_frame_size({}, frame_size)
    normalized = dict(settings)
    roi_x, roi_y, roi_width, roi_height = settings["roi"]
    normalized["roi"] = [
        float(roi_x) / frame_width,
        float(roi_y) / frame_height,
        float(roi_width) / frame_width,
        float(roi_height) / frame_height,
    ]
    paper_quad = settings.get("paper_quad")
    if paper_quad is not None:
        normalized["paper_quad"] = [
            [float(point[0]) / frame_width, float(point[1]) / frame_height]
            for point in paper_quad
        ]
    return normalized


def _normalized_to_pixel_coordinates(settings, frame_size):
    """把V4磁盘归一化坐标恢复成当前实际采集尺寸的运行时像素坐标。"""
    frame_width, frame_height = _resolve_frame_size({}, frame_size)
    restored = dict(settings)
    try:
        roi_x, roi_y, roi_width, roi_height = (float(value) for value in settings["roi"])
    except (TypeError, ValueError) as error:
        raise ValueError("V4 ROI归一化坐标无效") from error
    if not all(0.0 <= value <= 1.0 for value in (roi_x, roi_y, roi_width, roi_height)):
        raise ValueError("V4 ROI归一化坐标必须位于0到1")
    restored["roi"] = [
        int(round(roi_x * frame_width)),
        int(round(roi_y * frame_height)),
        int(round(roi_width * frame_width)),
        int(round(roi_height * frame_height)),
    ]
    paper_quad = settings.get("paper_quad")
    if paper_quad is not None:
        restored_quad = []
        for point in paper_quad:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("V4 paper_quad归一化角点无效")
            point_x, point_y = (float(value) for value in point)
            if not 0.0 <= point_x <= 1.0 or not 0.0 <= point_y <= 1.0:
                raise ValueError("V4 paper_quad归一化坐标必须位于0到1")
            restored_quad.append([point_x * frame_width, point_y * frame_height])
        restored["paper_quad"] = restored_quad
    return restored


def load_runtime_settings(path, config, frame_size=None):
    """从JSON加载现场参数，文件不存在时返回默认参数。

    主要流程：构造默认参数、读取JSON、校验版本，再整体校验所有可调字段。
    关键参数：path 为设置文件路径，config 提供默认值和相机尺寸。
    返回值：独立的合法现场参数字典；损坏或不兼容文件抛出 ValueError。
    """
    actual_frame_size = _resolve_frame_size(config, frame_size)
    default_settings = build_default_runtime_settings(config, actual_frame_size)
    path = os.fspath(path)
    if not os.path.exists(path):
        return default_settings

    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict) or payload.get("version") not in (
        LEGACY_SETTINGS_VERSION,
        PIXEL_SETTINGS_VERSION,
        NORMALIZED_SETTINGS_VERSION,
        ORIENTATION_SETTINGS_VERSION,
        SETTINGS_VERSION,
    ):
        raise ValueError("现场参数文件版本无效")

    version = int(payload["version"])
    if version == LEGACY_SETTINGS_VERSION:
        # V2只支持四边整体INSET。迁移时把它展开成等价X/Y/W/H，确保设备升级后
        # 黄色区域位置和大小不发生跳变；分界线仍使用A4物理中线。
        try:
            legacy_inset = float(payload.get("inset_mm", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError("V2 inset_mm无法迁移") from error
        migrated = dict(default_settings)
        for key in (
            "roi",
            "paper_quad",
            "inset_mm",
            "fixed_threshold",
            "min_area_ratio",
            "open_kernel",
            "close_kernel",
        ):
            migrated[key] = payload.get(key)
        # V2发布时黄色区域固定为竖纸中间210×230mm。新版本默认已扩展为完整
        # A4，因此迁移不能再调用当前默认函数，否则旧设备升级后区域会静默跳变。
        legacy_portrait_region = (0.0, 33.5, 210.0, 230.0)
        migrated.update(
            {
                "work_x_mm": legacy_portrait_region[0] + legacy_inset,
                "work_y_mm": legacy_portrait_region[1] + legacy_inset,
                "work_width_mm": legacy_portrait_region[2] - 2.0 * legacy_inset,
                "work_height_mm": legacy_portrait_region[3] - 2.0 * legacy_inset,
                "split_y_mm": default_split_y_mm(PAPER_ORIENTATION_PORTRAIT),
            }
        )
        # 旧大核会在远距离把2mm黑缝连接；迁移时强制切换到不扩张的1x1闭运算。
        migrated["close_kernel"] = 1
        migrated = validate_runtime_settings(migrated, LEGACY_PIXEL_FRAME_SIZE)
        migrated = _scale_pixel_coordinate_fields(
            migrated,
            LEGACY_PIXEL_FRAME_SIZE,
            actual_frame_size,
        )
        return validate_runtime_settings(migrated, actual_frame_size)

    loaded_settings = {key: payload.get(key) for key in RUNTIME_SETTING_KEYS}
    if version == PIXEL_SETTINGS_VERSION:
        # V3没有记录采集尺寸，A版历史发布物固定为640x480，因此来源尺寸是确定的。
        loaded_settings["paper_orientation"] = PAPER_ORIENTATION_PORTRAIT
        loaded_settings["close_kernel"] = 1
        loaded_settings = validate_runtime_settings(
            loaded_settings,
            LEGACY_PIXEL_FRAME_SIZE,
        )
        loaded_settings = _scale_pixel_coordinate_fields(
            loaded_settings,
            LEGACY_PIXEL_FRAME_SIZE,
            actual_frame_size,
        )
    else:
        if payload.get("coordinate_space") != "normalized":
            raise ValueError("V4/V5/V6现场参数缺少normalized坐标声明")
        # V4还没有方向字段，严格沿用历史竖放语义；V5/V6使用文件中的显式方向。
        if version == NORMALIZED_SETTINGS_VERSION:
            loaded_settings["paper_orientation"] = PAPER_ORIENTATION_PORTRAIT
        if version == ORIENTATION_SETTINGS_VERSION:
            # V5虽然已经保存H/V，但当时毫米原点固定跟随画面左上角，没有记录相机
            # 是向左还是向右侧装。继续使用旧蓝框会让红线、目标位置及UART毫米坐标
            # 一起反向。ROI和分割阈值仍与坐标原点无关，因此保留；纸面相关字段则
            # 全部恢复为V6默认值，迫使现场重新执行AUTO ROI和LOCK ROI。
            loaded_settings["paper_quad"] = None
        loaded_settings = _normalized_to_pixel_coordinates(
            loaded_settings,
            actual_frame_size,
        )
        if version == ORIENTATION_SETTINGS_VERSION:
            for key in PAPER_SETTING_KEYS:
                loaded_settings[key] = default_settings[key]
    return validate_runtime_settings(loaded_settings, actual_frame_size)


def save_runtime_settings(path, settings, frame_size):
    """使用临时文件和原子替换保存合法现场参数。

    主要流程：先整体校验参数，再写入同目录临时文件、刷新磁盘并原子替换目标。
    关键参数：path 为持久文件路径，settings 为运行参数，frame_size 为相机宽高。
    返回值：规范化后的已保存参数；失败时清理临时文件并重新抛出异常。
    """
    normalized = validate_runtime_settings(settings, frame_size)
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    disk_settings = _coordinates_to_normalized(normalized, frame_size)
    payload = {
        "version": SETTINGS_VERSION,
        "coordinate_space": "normalized",
        **disk_settings,
    }

    try:
        with open(temporary_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        # 原子替换前失败时清理临时文件，防止下次启动误读取残留内容。
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise

    return normalized


def merge_runtime_config(default_config, settings):
    """把现场参数合并到默认视觉配置并返回新字典。

    主要流程：复制默认配置，再覆盖阈值、最小面积和形态学核；ROI由调用方单独传递。
    关键参数：default_config 为完整算法配置，settings 为已经校验的现场参数。
    返回值：可直接传给 detect_pieces 的新配置，不修改两个输入对象。
    """
    merged = dict(default_config)
    for key in ("fixed_threshold", "min_area_ratio", "open_kernel", "close_kernel"):
        merged[key] = settings[key]
    return merged
