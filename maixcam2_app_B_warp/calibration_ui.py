"""固定ROI调参会话、质量判断和屏幕预览绘制。"""

import cv2
import numpy as np

try:
    from maixcam2_app_B_warp.paper_locator import build_active_quad
    from maixcam2_app_B_warp.settings_store import (
        CLOSE_KERNEL_VALUES,
        OPEN_KERNEL_VALUES,
        validate_runtime_settings,
    )
except ModuleNotFoundError as error:
    # MaixVision平铺工程文件后顶层包不存在，此时使用同级模块导入。
    if error.name != "maixcam2_app_B_warp":
        raise
    from paper_locator import build_active_quad
    from settings_store import (
        CLOSE_KERNEL_VALUES,
        OPEN_KERNEL_VALUES,
        validate_runtime_settings,
    )


# 四个预览页面名称同时作为界面动作和状态值；CAL由主循环负责退出。
VIEW_ROI = "roi"
VIEW_MASK = "mask"
VIEW_RESULT = "result"
VIEW_ADV = "adv"
CALIBRATION_VIEWS = (VIEW_ROI, VIEW_MASK, VIEW_RESULT, VIEW_ADV)

# ROI与分割参数分别循环，保证小屏一次只显示和调整一个项目。
ROI_ITEMS = ("LEFT", "RIGHT", "TOP", "BOTTOM")
SEGMENT_ITEMS = ("TH", "MIN", "OPEN", "CLOSE")
CALIBRATION_STEPS = (1, 5, 10)
MIN_ROI_SIZE_PX = 40
BACKGROUND_WHITE_RATIO = 0.65
INSET_STEP_MM = 0.5

# RESULT页面使用稳定BGR颜色区分轮廓去留原因。
COLOR_VALID = (0, 210, 0)
COLOR_EDGE = (0, 140, 255)
COLOR_SMALL = (0, 0, 255)
COLOR_LARGE = (200, 0, 200)
COLOR_ROI = (255, 200, 0)
COLOR_PAPER = (255, 255, 0)
COLOR_ACTIVE = (0, 255, 255)
COLOR_SELECTED = COLOR_ACTIVE
COLOR_MASK_OUTSIDE = (20, 20, 20)


class CalibrationQuality:
    """保存一次调参质量判断以及界面需要的关键计数。"""

    def __init__(self, state, complete_count, expected_count):
        """初始化质量状态、完整轮廓数和校准期望数量。"""
        self.state = str(state)
        self.complete_count = int(complete_count)
        self.expected_count = int(expected_count)


class CalibrationSession:
    """维护一次未保存的现场调参会话。

    主要职责：保存工作参数副本、当前预览页面、当前参数项和调节步长。
    会话不会修改外部已保存字典，只有 snapshot 返回值才能交给持久化层保存。
    """

    def __init__(self, saved_settings, frame_size):
        """校验并复制已保存参数，初始化为简化ROI页面。

        关键参数：saved_settings 为当前生效参数，frame_size 为相机 ``(宽, 高)``。
        返回值：构造函数无返回值；非法初始参数直接抛出 ValueError。
        """
        self.frame_size = tuple(int(value) for value in frame_size)
        self.settings = validate_runtime_settings(saved_settings, self.frame_size)
        self.view = VIEW_ROI
        self._roi_item_index = 0
        self._segment_item_index = 0
        self._step_index = CALIBRATION_STEPS.index(5)
        self.status_text = "ROI NOT SET" if self.settings["paper_quad"] is None else "ROI READY"
        self.paper_confidence = None

    @property
    def page_names(self):
        """返回屏幕显示使用的四个预览页名称，CAL退出按钮不属于预览页。"""
        return tuple(view.upper() for view in CALIBRATION_VIEWS)

    @property
    def step(self):
        """返回当前像素、灰度或面积比例调节使用的整数步长。"""
        return CALIBRATION_STEPS[self._step_index]

    @property
    def current_item(self):
        """ADV页返回当前分割参数，其余简化页统一返回INSET。"""
        if self.view == VIEW_ADV:
            return SEGMENT_ITEMS[self._segment_item_index]
        return "INSET"

    def select_view(self, view):
        """切换ROI、MASK、RESULT或ADV预览页面并返回新页面名称。"""
        view = str(view).lower()
        if view not in CALIBRATION_VIEWS:
            raise ValueError("未知调参页面")
        self.view = view
        return self.view

    def select_item(self, item):
        """选择ADV分割参数；旧ROI边名称仅作为兼容输入返回ROI页。"""
        item = str(item).upper()
        if item in ROI_ITEMS:
            self._roi_item_index = ROI_ITEMS.index(item)
            self.view = VIEW_ROI
            return item
        if item in SEGMENT_ITEMS:
            self._segment_item_index = SEGMENT_ITEMS.index(item)
            self.view = VIEW_ADV
            return item
        raise ValueError("未知调参项目")

    def cycle_item(self):
        """循环选择下一项高级分割参数并返回名称。"""
        self._segment_item_index = (
            self._segment_item_index + 1
        ) % len(SEGMENT_ITEMS)
        return self.current_item

    def bottom_actions(self):
        """按当前页面返回五个固定触摸槽对应的逻辑动作。

        ROI/MASK/RESULT保持简化纸张操作；只有ADV显示详细分割参数，避免默认界面
        同时堆叠过多现场参数。
        """
        if self.view == VIEW_ADV:
            return (
                "next_param",
                "value_dec",
                "select_value",
                "value_inc",
                "save_segmentation",
            )
        return (
            "auto_roi",
            "inset_dec",
            "inset_value",
            "inset_inc",
            "lock_roi",
        )

    def resolve_control_action(self, control_name):
        """把control_1～control_5转换为当前页面逻辑动作。"""
        if not str(control_name).startswith("control_"):
            return str(control_name)
        try:
            index = int(str(control_name).split("_", 1)[1]) - 1
        except (TypeError, ValueError) as error:
            raise ValueError("未知底部控制槽") from error
        actions = self.bottom_actions()
        if not 0 <= index < len(actions):
            raise ValueError("未知底部控制槽")
        return actions[index]

    def apply_auto_roi(self, location):
        """把一次自动定位结果应用到会话副本，失败时完整保留旧四角。

        成功只更新内存工作副本和置信度，必须再按LOCK ROI才会由入口持久化。
        返回值：成功返回True，失败返回False。
        """
        if not getattr(location, "success", False) or location.paper_quad is None:
            self.status_text = "AUTO ROI FAIL"
            return False
        candidate = location.paper_quad.astype(float).tolist()
        updated = dict(self.settings)
        updated["paper_quad"] = candidate
        # 复用完整设置校验，确保候选四角在保存前已经满足画面边界与凸性约束。
        self.settings = validate_runtime_settings(updated, self.frame_size)
        self.paper_confidence = float(location.confidence)
        self.status_text = f"AUTO ROI OK {self.paper_confidence * 100.0:.0f}%"
        return True

    def adjust_inset(self, direction):
        """按0.5mm步进整体增减四边INSET，达到0或20mm边界时返回False。"""
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("调节方向必须是-1或1")
        current = float(self.settings["inset_mm"])
        candidate = round(max(0.0, min(20.0, current + direction * INSET_STEP_MM)), 1)
        if candidate == current:
            self.status_text = "INSET LIMIT"
            return False
        self.settings["inset_mm"] = candidate
        self.status_text = f"INSET {candidate:.1f}mm"
        return True

    def can_lock_roi(self, result):
        """判断当前纸张四角能否在1～4片完整且无触边轮廓时锁定。"""
        complete_count = sum(
            1 for piece in result.pieces if piece.get("complete") is True
        )
        allowed = (
            self.settings.get("paper_quad") is not None
            and 1 <= complete_count <= 4
            and 1 <= int(result.valid_contour_count) <= 4
            and not result.edge_contours
            and not result.large_contours
        )
        self.status_text = "ROI READY" if allowed else "ROI NEEDS 1-4 COMPLETE"
        return bool(allowed)

    def can_save_segmentation(self, result):
        """判断高级分割参数是否达到GOOD 4/4保存门槛。"""
        allowed = evaluate_calibration(result).state == "GOOD"
        self.status_text = "GOOD 4/4" if allowed else "NEED GOOD 4/4"
        return bool(allowed)

    def cycle_step(self):
        """按1、5、10的顺序循环调节步长并返回新值。"""
        self._step_index = (self._step_index + 1) % len(CALIBRATION_STEPS)
        return self.step

    def toggle_threshold_mode(self, current_otsu_threshold):
        """在Otsu自动阈值和当前固定阈值之间切换。

        自动模式切到固定模式时锁定本帧Otsu值；固定模式再次切换则恢复None。
        返回值：切换后的 fixed_threshold，自动模式返回 None。
        """
        if self.settings["fixed_threshold"] is None:
            threshold = max(0.0, min(255.0, float(current_otsu_threshold)))
            self.settings["fixed_threshold"] = float(round(threshold))
        else:
            self.settings["fixed_threshold"] = None
        return self.settings["fixed_threshold"]

    def adjust(self, direction):
        """按当前步长增减选中参数，非法调整不修改参数并返回False。

        关键参数：direction 只允许 -1 或 1；ROI使用像素，TH使用灰度，MIN每单位
        对应万分之一面积比例，OPEN/CLOSE在允许的奇数序列中移动。
        返回值：参数实际改变返回 True，到达边界或AUTO阈值不可调时返回 False。
        """
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("调节方向必须是-1或1")

        if self.view != VIEW_ADV:
            return self.adjust_inset(direction)
        item = self.current_item
        if item == "TH":
            return self._adjust_threshold(direction)
        if item == "MIN":
            return self._adjust_min_area(direction)
        return self._adjust_kernel(item, direction)

    def _adjust_roi(self, item, direction):
        """移动选中的ROI边，同时保持画面边界和最小宽高约束。"""
        frame_width, frame_height = self.frame_size
        x, y, width, height = self.settings["roi"]
        delta = direction * self.step
        candidate = [x, y, width, height]

        if item == "LEFT":
            candidate[0] += delta
            candidate[2] -= delta
        elif item == "RIGHT":
            candidate[2] += delta
        elif item == "TOP":
            candidate[1] += delta
            candidate[3] -= delta
        else:
            candidate[3] += delta

        candidate_x, candidate_y, candidate_width, candidate_height = candidate
        valid = (
            candidate_x >= 0
            and candidate_y >= 0
            and candidate_width >= MIN_ROI_SIZE_PX
            and candidate_height >= MIN_ROI_SIZE_PX
            and candidate_x + candidate_width <= frame_width
            and candidate_y + candidate_height <= frame_height
        )
        if not valid:
            return False
        self.settings["roi"] = candidate
        return True

    def _adjust_threshold(self, direction):
        """调整固定阈值；AUTO模式下保持不变并返回False。"""
        threshold = self.settings["fixed_threshold"]
        if threshold is None:
            return False
        candidate = max(0.0, min(255.0, float(threshold) + direction * 5.0))
        if candidate == threshold:
            return False
        self.settings["fixed_threshold"] = candidate
        return True

    def _adjust_min_area(self, direction):
        """按0.05%固定步长调整最小面积比例并限制到合法范围。"""
        current = float(self.settings["min_area_ratio"])
        candidate = current + direction * 0.0005
        candidate = round(max(0.0001, min(0.25, candidate)), 7)
        if candidate == current:
            return False
        self.settings["min_area_ratio"] = candidate
        return True

    def _adjust_kernel(self, item, direction):
        """在预定义正奇数序列中调整开运算或闭运算核。"""
        key = "open_kernel" if item == "OPEN" else "close_kernel"
        values = OPEN_KERNEL_VALUES if item == "OPEN" else CLOSE_KERNEL_VALUES
        current = int(self.settings[key])
        current_index = values.index(current)
        candidate_index = max(0, min(len(values) - 1, current_index + direction))
        if candidate_index == current_index:
            return False
        self.settings[key] = values[candidate_index]
        return True

    def snapshot(self):
        """返回经过完整校验的工作参数副本，供保存成功后替换运行参数。"""
        return validate_runtime_settings(self.settings, self.frame_size)


def evaluate_calibration(result, expected_pieces=4):
    """根据检测诊断给出现场参数质量状态。

    判定顺序：过大亮背景或异常白色占比优先，其次是边缘轮廓、额外有效噪声、
    漏检或异常顶点，只有恰好四片完整且顶点有效时才返回 GOOD。
    关键参数：result 为 DetectionResult 兼容对象，expected_pieces 默认四片。
    返回值：CalibrationQuality，供状态栏和保存门共同使用。
    """
    expected_pieces = int(expected_pieces)
    if expected_pieces <= 0:
        raise ValueError("期望碎片数量必须大于零")

    pieces = list(result.pieces)
    complete_pieces = [piece for piece in pieces if piece.get("complete") is True]
    complete_count = len(complete_pieces)
    valid_count = int(result.valid_contour_count)
    vertex_counts_valid = all(
        3 <= int(piece.get("vertex_count", 0)) <= 5 for piece in complete_pieces
    )

    if result.large_contours or float(result.white_ratio) >= BACKGROUND_WHITE_RATIO:
        state = "BACKGROUND"
    elif result.edge_contours:
        state = "EDGE"
    elif valid_count > expected_pieces:
        state = "NOISE"
    elif (
        valid_count != expected_pieces
        or complete_count != expected_pieces
        or not vertex_counts_valid
    ):
        state = "MISS"
    else:
        state = "GOOD"

    return CalibrationQuality(state, complete_count, expected_pieces)


def format_calibration_status(result, quality):
    """生成固定字段顺序的ASCII调参诊断状态文本。

    主要流程：汇总质量状态、完整数量、三类拒绝轮廓、当前阈值和白色占比。
    关键参数：result 为 DetectionResult 兼容对象，quality 为 CalibrationQuality。
    返回值：适合OpenCV默认字体绘制的单行字符串。
    """
    return (
        f"{quality.state} {quality.complete_count}/{quality.expected_count} "
        f"EDGE={len(result.edge_contours)} "
        f"SMALL={len(result.small_contours)} "
        f"LARGE={len(result.large_contours)} "
        f"TH={float(result.threshold):.0f} "
        f"WHITE={float(result.white_ratio) * 100.0:.1f}%"
    )


def _put_fitted_text(image, text, rect, color=(255, 255, 255), max_scale=0.52):
    """把单行文字缩放到指定矩形内并居中绘制。

    关键参数：rect 为含x、y、width、height属性的按钮矩形或兼容对象。
    返回值：实际使用的字体缩放值，供测试或后续布局诊断使用。
    """
    text = str(text)
    available_width = max(1, int(rect.width) - 8)
    available_height = max(1, int(rect.height) - 8)
    base_size, base_line = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        max_scale,
        1,
    )
    width_scale = available_width / max(1, base_size[0])
    height_scale = available_height / max(1, base_size[1] + base_line)
    scale = max(0.28, min(max_scale, max_scale * width_scale, max_scale * height_scale))
    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1,
    )
    text_x = int(round(rect.x + (rect.width - text_size[0]) / 2.0))
    text_y = int(round(rect.y + (rect.height + text_size[1] - baseline) / 2.0))
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )
    return scale


def _draw_button(image, button, label, active=False, enabled=True):
    """绘制一个固定矩形按钮，并根据活动与禁用状态选择颜色。"""
    x1 = int(round(button.x))
    y1 = int(round(button.y))
    x2 = int(round(button.x + button.width))
    y2 = int(round(button.y + button.height))
    if active:
        fill_color = (32, 150, 48)
    elif enabled:
        fill_color = (55, 55, 55)
    else:
        fill_color = (24, 24, 24)
    cv2.rectangle(image, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (220, 220, 220), 1)
    text_color = (255, 255, 255) if enabled else (100, 100, 100)
    _put_fitted_text(image, label, button, color=text_color)


def _draw_roi_preview(frame_bgr, session):
    """在固定640×480画布上绘制完整A4和机械有效四边形。

    完整纸张使用青色，33.5mm裁剪和INSET后的有效区使用黄色；没有锁定四角时
    保留兼容矩形ROI，确保首次标定仍有明确参照。
    """
    target_width, target_height = session.frame_size
    output = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    scale_x = target_width / frame_bgr.shape[1]
    scale_y = target_height / frame_bgr.shape[0]
    paper_quad = session.settings.get("paper_quad")
    if paper_quad is None:
        x, y, width, height = session.settings["roi"]
        cv2.rectangle(
            output,
            (int(round(x * scale_x)), int(round(y * scale_y))),
            (int(round((x + width - 1) * scale_x)), int(round((y + height - 1) * scale_y))),
            COLOR_ROI,
            2,
        )
        return output

    paper_array = np.asarray(paper_quad, dtype=np.float32)
    active_array = build_active_quad(paper_array, session.settings["inset_mm"])
    display_scale = np.float32([scale_x, scale_y])
    cv2.polylines(
        output,
        [np.rint(paper_array * display_scale).astype(np.int32)],
        True,
        COLOR_PAPER,
        2,
    )
    cv2.polylines(
        output,
        [np.rint(active_array * display_scale).astype(np.int32)],
        True,
        COLOR_ACTIVE,
        2,
    )
    return output


def _place_preview_content(content_bgr, frame_size):
    """把任意比例的预览内容等比放入顶部页签和底部状态栏之间。"""
    target_width, target_height = frame_size
    preview_top = 64
    preview_bottom = target_height - 128
    preview_height = max(1, preview_bottom - preview_top)
    scale = min(target_width / content_bgr.shape[1], preview_height / content_bgr.shape[0])
    display_width = max(1, int(round(content_bgr.shape[1] * scale)))
    display_height = max(1, int(round(content_bgr.shape[0] * scale)))
    resized = cv2.resize(
        content_bgr,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.full((target_height, target_width, 3), COLOR_MASK_OUTSIDE, dtype=np.uint8)
    offset_x = (target_width - display_width) // 2
    offset_y = preview_top + (preview_height - display_height) // 2
    output[offset_y : offset_y + display_height, offset_x : offset_x + display_width] = resized
    return output


def _draw_mask_preview(frame_bgr, result, session):
    """把有效区二值掩膜等比显示在固定预览区，外部保持深灰。"""
    if result.mask is None or result.mask.ndim != 2:
        raise ValueError("二值掩膜必须是二维图像")
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    return _place_preview_content(mask_bgr, session.frame_size)


def _draw_result_preview(frame_bgr, result, session):
    """在检测ROI局部图上按有效、边缘、过小和过大分类绘制轮廓。"""
    x, y, width, height = (int(value) for value in result.roi)
    frame_height, frame_width = frame_bgr.shape[:2]
    if x >= 0 and y >= 0 and x + width <= frame_width and y + height <= frame_height:
        content = frame_bgr[y : y + height, x : x + width].copy()
    else:
        # B版或诊断回退的坐标空间与传入帧不一致时，使用缩放副本而不是越界切片。
        content = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    offset = np.asarray([[[x, y]]], dtype=np.int32)
    for contour in result.small_contours:
        cv2.drawContours(content, [contour.astype(np.int32) - offset], -1, COLOR_SMALL, 2)
    for contour in result.large_contours:
        cv2.drawContours(content, [contour.astype(np.int32) - offset], -1, COLOR_LARGE, 2)
    for piece in result.pieces:
        contour = piece.get("contour")
        if contour is None:
            continue
        color = COLOR_VALID if piece.get("complete") is True else COLOR_EDGE
        cv2.drawContours(content, [contour.astype(np.int32) - offset], -1, color, 2)
    return _place_preview_content(content, session.frame_size)


def _current_value_label(session, result):
    """根据当前参数项生成底部中间按钮的简短数值标签。"""
    item = session.current_item
    if item == "INSET":
        return f"{float(session.settings['inset_mm']):.1f}mm"
    if item in ROI_ITEMS:
        x, y, width, height = session.settings["roi"]
        values = {
            "LEFT": x,
            "RIGHT": x + width,
            "TOP": y,
            "BOTTOM": y + height,
        }
        return str(values[item])
    if item == "TH":
        fixed_threshold = session.settings["fixed_threshold"]
        if fixed_threshold is None:
            return f"AUTO:{float(result.threshold):.0f}"
        return f"FIX:{float(fixed_threshold):.0f}"
    if item == "MIN":
        return f"{float(session.settings['min_area_ratio']) * 100.0:.2f}%"
    key = "open_kernel" if item == "OPEN" else "close_kernel"
    return str(int(session.settings[key]))


def draw_calibration_frame(
    frame_bgr,
    result,
    session,
    buttons,
    quality,
    status_message="",
):
    """绘制固定640×480的ROI、MASK、RESULT或ADV调参界面。

    主要流程：按当前页面生成预览副本，覆盖顶部页签、诊断状态栏和底部参数按钮。
    关键参数：result 为当前工作参数产生的检测结果，buttons 为固定触摸布局，
    quality 控制状态颜色和保存按钮启用状态，status_message 可追加保存或错误提示。
    返回值：与输入同尺寸的BGR新图像；输入相机帧保持不变。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr必须是三通道BGR图像")

    if session.view == VIEW_ROI:
        output = _draw_roi_preview(frame_bgr, session)
    elif session.view == VIEW_MASK:
        output = _draw_mask_preview(frame_bgr, result, session)
    else:
        output = _draw_result_preview(frame_bgr, result, session)
        if session.view == VIEW_ADV:
            cv2.putText(
                output,
                f"PARAM {session.current_item} {_current_value_label(session, result)}",
                (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_ACTIVE,
                1,
                cv2.LINE_AA,
            )

    top_labels = {
        "roi": "ROI",
        "mask": "MASK",
        "result": "RESULT",
        "adv": "ADV",
        "cal": "CAL",
    }
    for name in ("roi", "mask", "result", "adv", "cal"):
        _draw_button(
            output,
            buttons[name],
            top_labels[name],
            active=(name == session.view or name == "cal"),
        )

    control_names = tuple(f"control_{index}" for index in range(1, 6))
    control_y = int(round(min(buttons[name].y for name in control_names)))
    status_height = 52
    status_y = max(0, control_y - status_height - 4)
    status_colors = {
        "GOOD": (18, 92, 28),
        "MISS": (30, 30, 130),
        "NOISE": (20, 95, 145),
        "EDGE": (20, 95, 145),
        "BACKGROUND": (70, 20, 100),
    }
    cv2.rectangle(
        output,
        (0, status_y),
        (output.shape[1], status_y + status_height),
        status_colors.get(quality.state, (0, 0, 0)),
        -1,
    )
    confidence_text = (
        "" if session.paper_confidence is None else f" CONF {session.paper_confidence * 100.0:.0f}%"
    )
    first_line = (status_message or session.status_text) + confidence_text
    second_line = (
        f"N {quality.complete_count}/{quality.expected_count} | "
        f"EDGE {len(result.edge_contours)} | SMALL {len(result.small_contours)} | "
        f"LARGE {len(result.large_contours)}"
    )

    # 两行状态分别自适应，避免长诊断文字覆盖顶部预览或底部按钮。
    first_rect = type(
        "StatusRect",
        (),
        {
            "x": 4.0,
            "y": float(status_y),
            "width": float(output.shape[1] - 8),
            "height": float(status_height / 2),
        },
    )()
    second_rect = type(
        "StatusRect",
        (),
        {
            "x": 4.0,
            "y": float(status_y + status_height / 2),
            "width": float(output.shape[1] - 8),
            "height": float(status_height / 2),
        },
    )()
    _put_fitted_text(output, first_line, first_rect, max_scale=0.48)
    _put_fitted_text(output, second_line, second_rect, max_scale=0.44)

    if session.view == VIEW_ADV:
        bottom_labels = (
            f"PARAM:{session.current_item}",
            "-",
            _current_value_label(session, result),
            "+",
            "SAVE",
        )
    else:
        bottom_labels = (
            "AUTO ROI",
            "-",
            f"INSET {session.settings['inset_mm']:.1f}",
            "+",
            "LOCK ROI",
        )
    lock_enabled = session.settings.get("paper_quad") is not None
    for index, (name, label) in enumerate(zip(control_names, bottom_labels), start=1):
        if session.view == VIEW_ADV:
            enabled = index != 5 or quality.state == "GOOD"
        else:
            enabled = index != 5 or lock_enabled
        _draw_button(
            output,
            buttons[name],
            label,
            active=(index == 5 and enabled),
            enabled=enabled,
        )

    return output
