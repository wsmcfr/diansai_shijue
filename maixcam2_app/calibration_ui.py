"""固定ROI调参会话、质量判断和屏幕预览绘制。"""

import cv2
import numpy as np

try:
    from maixcam2_app.settings_store import (
        CLOSE_KERNEL_VALUES,
        OPEN_KERNEL_VALUES,
        validate_runtime_settings,
    )
except ModuleNotFoundError as error:
    # MaixVision平铺工程文件后顶层包不存在，此时使用同级模块导入。
    if error.name != "maixcam2_app":
        raise
    from settings_store import (
        CLOSE_KERNEL_VALUES,
        OPEN_KERNEL_VALUES,
        validate_runtime_settings,
    )


# 三个页面名称同时作为界面动作和状态值，避免主循环使用额外映射。
VIEW_ROI = "roi"
VIEW_MASK = "mask"
VIEW_RESULT = "result"
CALIBRATION_VIEWS = (VIEW_ROI, VIEW_MASK, VIEW_RESULT)

# ROI与分割参数分别循环，保证小屏一次只显示和调整一个项目。
ROI_ITEMS = ("LEFT", "RIGHT", "TOP", "BOTTOM")
SEGMENT_ITEMS = ("TH", "MIN", "OPEN", "CLOSE")
CALIBRATION_STEPS = (1, 5, 10)
MIN_ROI_SIZE_PX = 40
BACKGROUND_WHITE_RATIO = 0.65

# RESULT页面使用稳定BGR颜色区分轮廓去留原因。
COLOR_VALID = (0, 210, 0)
COLOR_EDGE = (0, 140, 255)
COLOR_SMALL = (0, 0, 255)
COLOR_LARGE = (200, 0, 200)
COLOR_ROI = (255, 200, 0)
COLOR_SELECTED = (0, 255, 255)
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
        """校验并复制已保存参数，初始化为ROI页面和5像素步长。

        关键参数：saved_settings 为当前生效参数，frame_size 为相机 ``(宽, 高)``。
        返回值：构造函数无返回值；非法初始参数直接抛出 ValueError。
        """
        self.frame_size = tuple(int(value) for value in frame_size)
        self.settings = validate_runtime_settings(saved_settings, self.frame_size)
        self.view = VIEW_ROI
        self._roi_item_index = 0
        self._segment_item_index = 0
        self._step_index = CALIBRATION_STEPS.index(5)

    @property
    def step(self):
        """返回当前像素、灰度或面积比例调节使用的整数步长。"""
        return CALIBRATION_STEPS[self._step_index]

    @property
    def current_item(self):
        """根据当前页面返回正在调整的ROI边或分割参数名称。"""
        if self.view == VIEW_ROI:
            return ROI_ITEMS[self._roi_item_index]
        return SEGMENT_ITEMS[self._segment_item_index]

    def select_view(self, view):
        """切换ROI、MASK或RESULT预览页面并返回新页面名称。"""
        view = str(view).lower()
        if view not in CALIBRATION_VIEWS:
            raise ValueError("未知调参页面")
        self.view = view
        return self.view

    def select_item(self, item):
        """选择指定ROI边或分割参数，主要供触摸动作和自动测试调用。"""
        item = str(item).upper()
        if item in ROI_ITEMS:
            self._roi_item_index = ROI_ITEMS.index(item)
            self.view = VIEW_ROI
            return item
        if item in SEGMENT_ITEMS:
            self._segment_item_index = SEGMENT_ITEMS.index(item)
            # MASK和RESULT共用分割参数；直接选择参数时进入可观察二值结果的MASK页。
            self.view = VIEW_MASK
            return item
        raise ValueError("未知调参项目")

    def cycle_item(self):
        """在当前页面支持的参数列表中循环选择下一项。"""
        if self.view == VIEW_ROI:
            self._roi_item_index = (self._roi_item_index + 1) % len(ROI_ITEMS)
        else:
            self._segment_item_index = (
                self._segment_item_index + 1
            ) % len(SEGMENT_ITEMS)
        return self.current_item

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

        item = self.current_item
        if item in ROI_ITEMS:
            return self._adjust_roi(item, direction)
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
        candidate = max(0.0, min(255.0, float(threshold) + direction * self.step))
        if candidate == threshold:
            return False
        self.settings["fixed_threshold"] = candidate
        return True

    def _adjust_min_area(self, direction):
        """按万分之一乘以步长调整最小面积比例并限制到合法范围。"""
        current = float(self.settings["min_area_ratio"])
        candidate = current + direction * self.step * 0.0001
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
    """暗化ROI外部并用黄色高亮当前选中的ROI边。"""
    output = np.multiply(frame_bgr, 0.25).astype(np.uint8)
    x, y, width, height = session.settings["roi"]
    output[y : y + height, x : x + width] = frame_bgr[
        y : y + height,
        x : x + width,
    ]
    right = x + width - 1
    bottom = y + height - 1
    cv2.rectangle(output, (x, y), (right, bottom), COLOR_ROI, 1)

    selected_lines = {
        "LEFT": ((x, y), (x, bottom)),
        "RIGHT": ((right, y), (right, bottom)),
        "TOP": ((x, y), (right, y)),
        "BOTTOM": ((x, bottom), (right, bottom)),
    }
    start, end = selected_lines[session.current_item]
    cv2.line(output, start, end, COLOR_SELECTED, 3)
    return output


def _draw_mask_preview(frame_bgr, result, session):
    """在深灰全帧中仅显示ROI对应的二值掩膜。"""
    output = np.full_like(frame_bgr, COLOR_MASK_OUTSIDE, dtype=np.uint8)
    x, y, width, height = session.settings["roi"]
    if result.mask.shape[:2] != (height, width):
        raise ValueError("二值掩膜尺寸必须与当前ROI一致")
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    output[y : y + height, x : x + width] = mask_bgr
    cv2.rectangle(
        output,
        (x, y),
        (x + width - 1, y + height - 1),
        COLOR_ROI,
        1,
    )
    return output


def _draw_result_preview(frame_bgr, result, session):
    """在相机帧副本上按有效、边缘、过小和过大分类绘制轮廓。"""
    output = frame_bgr.copy()
    x, y, width, height = session.settings["roi"]
    cv2.rectangle(
        output,
        (x, y),
        (x + width - 1, y + height - 1),
        COLOR_ROI,
        1,
    )
    for contour in result.small_contours:
        cv2.drawContours(output, [contour], -1, COLOR_SMALL, 2)
    for contour in result.large_contours:
        cv2.drawContours(output, [contour], -1, COLOR_LARGE, 2)
    for piece in result.pieces:
        color = COLOR_VALID if piece.get("complete") is True else COLOR_EDGE
        cv2.drawContours(output, [piece["contour"]], -1, color, 2)
    return output


def _current_value_label(session, result):
    """根据当前参数项生成底部中间按钮的简短数值标签。"""
    item = session.current_item
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
    """绘制完整的ROI、MASK或RESULT调参界面。

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

    top_labels = {"roi": "ROI", "mask": "MASK", "result": "RESULT", "cal": "CAL"}
    for name in ("roi", "mask", "result", "cal"):
        _draw_button(
            output,
            buttons[name],
            top_labels[name],
            active=(name == session.view or name == "cal"),
        )

    control_y = int(round(min(buttons[name].y for name in (
        "item", "minus", "value", "plus", "step", "save_settings"
    ))))
    status_height = 38
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
    status_text = format_calibration_status(result, quality)
    if status_message:
        status_text = f"{status_text} {status_message}"

    # 状态栏使用兼容矩形对象复用自适应文字绘制，保证最长状态也不覆盖底部按钮。
    status_rect = type(
        "StatusRect",
        (),
        {
            "x": 4.0,
            "y": float(status_y),
            "width": float(output.shape[1] - 8),
            "height": float(status_height),
        },
    )()
    _put_fitted_text(output, status_text, status_rect, max_scale=0.48)

    bottom_labels = {
        "item": f"ITEM:{session.current_item}",
        "minus": "-",
        "value": _current_value_label(session, result),
        "plus": "+",
        "step": f"STEP:{session.step}",
        "save_settings": "SAVE",
    }
    for name in ("item", "minus", "value", "plus", "step", "save_settings"):
        save_enabled = name != "save_settings" or quality.state == "GOOD"
        _draw_button(
            output,
            buttons[name],
            bottom_labels[name],
            active=(name == "save_settings" and save_enabled),
            enabled=save_enabled,
        )

    return output
