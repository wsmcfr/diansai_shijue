"""MaixCAM2 触摸按键的纯逻辑层，不直接依赖 MaixPy。"""


class ButtonRect:
    """保存一个触摸按钮在相机显示图像坐标系中的矩形区域。"""

    def __init__(self, name, x, y, width, height):
        """初始化按钮名称和矩形；宽高必须为正数。"""
        if width <= 0 or height <= 0:
            raise ValueError("按钮宽高必须大于零")
        self.name = str(name)
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    @property
    def center(self):
        """返回按钮中心点，主要用于测试、绘制文字和触摸命中。"""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def contains(self, x, y):
        """判断给定图像坐标是否位于按钮边界内，边界点视为命中。"""
        return (
            self.x <= x <= self.x + self.width
            and self.y <= y <= self.y + self.height
        )


class TouchReleaseTracker:
    """把连续触摸采样转换为每次松开时唯一的一次点击事件。"""

    def __init__(self):
        """初始化为未按下状态，并清空最近一次触摸坐标。"""
        self._pressed_already = False
        self._last_x = 0.0
        self._last_y = 0.0

    def update(self, x, y, pressed):
        """接收一次触摸采样，在按下后的首次松开时返回点击坐标。

        主要流程：按住期间只更新最后坐标；松开沿产生一次点击并复位状态。
        关键参数：x、y 为物理屏幕坐标，pressed 为当前按压状态。
        返回值：无点击时为 None，松开沿为 ``(x, y)``。
        """
        if pressed:
            self._pressed_already = True
            self._last_x = float(x)
            self._last_y = float(y)
            return None

        if not self._pressed_already:
            return None

        self._pressed_already = False
        # 官方触摸示例在松开采样时仍提供有效坐标，优先使用当前值以贴近用户最终落点。
        self._last_x = float(x)
        self._last_y = float(y)
        return (x, y)


def build_button_layout(width, height):
    """根据相机显示图像尺寸创建正常识别界面的五个按钮。

    主要流程：从左排列已知、未知和模板保存按钮，在剩余区域加入START确认按钮，
    并复用右上角CAL切换按钮。若显示宽度不足以保持按钮间隔则明确报错，避免触摸区
    静默重叠。
    关键参数：width、height 为处理后显示图像尺寸。
    返回值：以 ``known``、``unknown``、``save``、``start``、``cal`` 为键的按钮字典。
    """
    if width <= 0 or height <= 0:
        raise ValueError("图像宽高必须大于零")

    margin = max(6, int(round(width * 0.015625)))
    gap = max(6, int(round(width * 0.0125)))
    button_height = max(36, min(52, int(round(height * 0.10))))
    known_width = max(96, int(round(width * 0.18)))
    unknown_width = max(112, int(round(width * 0.21)))
    save_width = max(84, int(round(width * 0.15)))

    known = ButtonRect("known", margin, margin, known_width, button_height)
    unknown = ButtonRect(
        "unknown",
        known.x + known.width + gap,
        margin,
        unknown_width,
        button_height,
    )
    save = ButtonRect(
        "save",
        unknown.x + unknown.width + gap,
        margin,
        save_width,
        button_height,
    )
    cal = build_cal_toggle_button(width, height)
    start_width = max(84, min(100, int(round(width * 0.14))))
    start_x = save.x + save.width + gap
    available_start_width = cal.x - gap - start_x
    if available_start_width < start_width:
        raise ValueError("显示宽度不足，无法放置互不重叠的START与CAL按钮")
    start = ButtonRect("start", start_x, margin, start_width, button_height)
    return {
        "known": known,
        "unknown": unknown,
        "save": save,
        "start": start,
        "cal": cal,
    }


def build_cal_toggle_button(width, height):
    """创建正常与调参界面共用的CAL切换按钮。

    主要流程：使用与顶部模式按钮相同的高度，把固定宽度按钮锚定到右上角。
    关键参数：width、height 为相机显示图像尺寸。
    返回值：名称为 ``cal`` 的 ButtonRect；两个界面调用本函数可保证位置完全一致。
    """
    if width <= 0 or height <= 0:
        raise ValueError("图像宽高必须大于零")
    margin = max(6, int(round(width * 0.015625)))
    button_height = max(36, min(52, int(round(height * 0.10))))
    cal_width = max(84, min(100, int(round(width * 0.14))))
    return ButtonRect(
        "cal",
        width - margin - cal_width,
        margin,
        cal_width,
        button_height,
    )


def build_calibration_layout(width, height):
    """创建五个顶部页签和五个固定底部控制槽。

    顶部固定为ROI/MASK/RESULT/ADV/CAL；底部使用control_1～control_5通用槽，
    由当前页面映射成AUTO ROI/INSET/LOCK或高级分割动作。固定槽避免页面切换时
    触摸区域漂移。返回值：按屏幕从左到右顺序排列的十个 ButtonRect 字典。
    """
    if width <= 0 or height <= 0:
        raise ValueError("图像宽高必须大于零")

    margin = max(6, int(round(width * 0.015625)))
    gap = max(4, int(round(width * 0.008)))
    top_height = max(36, min(52, int(round(height * 0.10))))
    available_width = width - 2 * margin - 4 * gap
    slot_widths = [available_width // 5 for _ in range(5)]
    slot_widths[-1] += available_width - sum(slot_widths)

    top_buttons = {}
    top_x = margin
    for name, slot_width in zip(("roi", "mask", "result", "adv", "cal"), slot_widths):
        top_buttons[name] = ButtonRect(name, top_x, margin, slot_width, top_height)
        top_x += slot_width + gap

    control_height = max(48, min(64, int(round(height * 0.12))))
    control_y = height - margin - control_height
    controls = {}
    control_x = margin
    for index, control_width in enumerate(slot_widths, start=1):
        name = f"control_{index}"
        controls[name] = ButtonRect(
            name,
            control_x,
            control_y,
            control_width,
            control_height,
        )
        control_x += control_width + gap

    return {
        **top_buttons,
        **controls,
    }


def map_display_to_image(point, image_size, display_size):
    """将 FIT_CONTAIN 显示模式下的物理屏幕触点反算为图像坐标。

    主要流程：计算保持宽高比的缩放值和两侧黑边偏移，再判断触点是否位于有效图像内。
    关键参数：point 为屏幕坐标，image_size 和 display_size 均为 ``(宽, 高)``。
    返回值：有效时返回浮点图像坐标，触摸黑边时返回 None。
    """
    image_width, image_height = image_size
    display_width, display_height = display_size
    if min(image_width, image_height, display_width, display_height) <= 0:
        raise ValueError("图像和屏幕宽高必须大于零")

    scale = min(display_width / image_width, display_height / image_height)
    rendered_width = image_width * scale
    rendered_height = image_height * scale
    offset_x = (display_width - rendered_width) / 2.0
    offset_y = (display_height - rendered_height) / 2.0
    display_x, display_y = point

    if not (
        offset_x <= display_x <= offset_x + rendered_width
        and offset_y <= display_y <= offset_y + rendered_height
    ):
        return None

    return (
        (display_x - offset_x) / scale,
        (display_y - offset_y) / scale,
    )


def hit_test(point, buttons):
    """返回包含指定图像触点的按钮名称，没有命中时返回 None。"""
    if point is None:
        return None
    x, y = point
    for name, button in buttons.items():
        if button.contains(x, y):
            return name
    return None
