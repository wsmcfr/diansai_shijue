"""现场视觉参数的校验、持久化和运行配置合并。"""

import json
import math
import os


# 文件版本用于阻止未来字段变化后静默加载不兼容的现场参数。
SETTINGS_VERSION = 2

# 只有这些字段允许由触摸调参界面修改并写入持久文件。
RUNTIME_SETTING_KEYS = (
    "roi",
    "paper_quad",
    "inset_mm",
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
PAPER_SETTING_KEYS = ("paper_quad", "inset_mm")
SEGMENTATION_SETTING_KEYS = (
    "fixed_threshold",
    "min_area_ratio",
    "open_kernel",
    "close_kernel",
)


def build_default_runtime_settings(config):
    """从默认视觉配置构造一份独立的现场参数。

    主要流程：读取相机宽高作为整帧ROI，再复制阈值、面积和形态学参数。
    关键参数：config 必须包含 DEFAULT_CONFIG 中对应的相机与视觉字段。
    返回值：只包含可调字段的新字典，调用方修改它不会影响全局默认配置。
    """
    return {
        "roi": [
            0,
            0,
            int(config["camera_width"]),
            int(config["camera_height"]),
        ],
        "paper_quad": None,
        "inset_mm": 0.0,
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
    try:
        inset_mm = float(settings["inset_mm"])
    except (TypeError, ValueError) as error:
        raise ValueError("inset_mm必须是0到20之间的数字") from error
    if not math.isfinite(inset_mm) or not 0.0 <= inset_mm <= 20.0:
        raise ValueError("inset_mm必须位于0到20之间")

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
        "inset_mm": inset_mm,
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


def load_runtime_settings(path, config):
    """从JSON加载现场参数，文件不存在时返回默认参数。

    主要流程：构造默认参数、读取JSON、校验版本，再整体校验所有可调字段。
    关键参数：path 为设置文件路径，config 提供默认值和相机尺寸。
    返回值：独立的合法现场参数字典；损坏或不兼容文件抛出 ValueError。
    """
    default_settings = build_default_runtime_settings(config)
    path = os.fspath(path)
    if not os.path.exists(path):
        return default_settings

    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict) or payload.get("version") != SETTINGS_VERSION:
        raise ValueError("现场参数文件版本无效")

    loaded_settings = {key: payload.get(key) for key in RUNTIME_SETTING_KEYS}
    frame_size = (int(config["camera_width"]), int(config["camera_height"]))
    return validate_runtime_settings(loaded_settings, frame_size)


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
    payload = {"version": SETTINGS_VERSION, **normalized}

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
