"""固定ROI调参会话、量化标定、质量判断和屏幕预览绘制。"""

import itertools

import cv2
import numpy as np

try:
    from maixcam2_app_A_quad.paper_locator import (
        PAPER_ORIENTATION_LANDSCAPE,
        PAPER_ORIENTATION_PORTRAIT,
        build_split_segment,
        build_work_quad,
        default_split_y_mm,
        default_work_region_mm,
    )
    from maixcam2_app_A_quad.settings_store import (
        CLOSE_KERNEL_VALUES,
        OPEN_KERNEL_VALUES,
        validate_runtime_settings,
    )
except ModuleNotFoundError as error:
    # MaixVision平铺工程文件后顶层包不存在，此时使用同级模块导入。
    if error.name != "maixcam2_app_A_quad":
        raise
    from paper_locator import (
        PAPER_ORIENTATION_LANDSCAPE,
        PAPER_ORIENTATION_PORTRAIT,
        build_split_segment,
        build_work_quad,
        default_split_y_mm,
        default_work_region_mm,
    )
    from settings_store import (
        CLOSE_KERNEL_VALUES,
        OPEN_KERNEL_VALUES,
        validate_runtime_settings,
    )


# 四个预览页面名称同时作为界面动作和状态值；CAL由主循环负责退出。
VIEW_ROI = "roi"
VIEW_MASK = "mask"
VIEW_RESULT = "result"
VIEW_MEASURE = "measure"
VIEW_ADV = "adv"
# 顶部仍只有四个可见页签；MEASURE复用RESULT页签，通过重复点击往返切换。
CALIBRATION_VIEWS = (VIEW_ROI, VIEW_MASK, VIEW_RESULT, VIEW_ADV)
SELECTABLE_CALIBRATION_VIEWS = CALIBRATION_VIEWS + (VIEW_MEASURE,)

# ROI与分割参数分别循环，保证小屏一次只显示和调整一个项目。
ROI_ITEMS = ("LEFT", "RIGHT", "TOP", "BOTTOM")
SEGMENT_ITEMS = ("TH", "MIN", "OPEN", "CLOSE")
CALIBRATION_STEPS = (1, 5, 10)
MIN_ROI_SIZE_PX = 40
BACKGROUND_WHITE_RATIO = 0.65
INSET_STEP_MM = 0.5
WORK_STEP_MM = 0.5
WORK_ITEMS = ("X", "Y", "W", "H", "SPLIT", "PAPER")
WORK_SETTING_KEYS = {
    "X": "work_x_mm",
    "Y": "work_y_mm",
    "W": "work_width_mm",
    "H": "work_height_mm",
    "SPLIT": "split_y_mm",
}

# RESULT页面使用稳定BGR颜色区分轮廓去留原因。
COLOR_VALID = (0, 210, 0)
COLOR_EDGE = (0, 140, 255)
COLOR_SMALL = (0, 0, 255)
COLOR_LARGE = (200, 0, 200)
COLOR_ROI = (255, 200, 0)
COLOR_PAPER = (255, 255, 0)
COLOR_ACTIVE = (0, 255, 255)
COLOR_SPLIT = (0, 0, 255)
COLOR_SELECTED = COLOR_ACTIVE
COLOR_MASK_OUTSIDE = (20, 20, 20)

# 量化标定门槛来自当前固定机位目标：至少2px/mm，2mm黑缝至少留下4px，
# 连续10帧最大位置跨度不超过1mm，100×60mm标准卡每边误差不超过1.5mm。
MIN_SCALE_PX_PER_MM = 2.0
MIN_GAP_PX = 4.0
MAX_JITTER_MM = 1.0
RECTANGLE_TARGET_MM = (100.0, 60.0)
RECTANGLE_TOLERANCE_MM = 1.5


class CalibrationQuality:
    """保存一次调参质量判断以及界面需要的关键计数。"""

    def __init__(self, state, complete_count, expected_count):
        """初始化质量状态、完整轮廓数和校准期望数量。"""
        self.state = str(state)
        self.complete_count = int(complete_count)
        self.expected_count = int(expected_count)


class CalibrationMeasurement:
    """保存MEASURE页面的一组可量化标定结果。

    各数值在证据不足时使用None，而不是用0伪装成失败测量；对应的 ``*_ok``
    字段始终为布尔值，便于界面直接绘制PASS、FAIL或WAIT。
    """

    def __init__(
        self,
        scale_px_per_mm=None,
        minimum_gap_px=None,
        stable_frames=0,
        stability_window=10,
        jitter_mm=None,
        rectangle_size_mm=None,
    ):
        """初始化测量数值并按统一门槛生成判定字段。

        关键参数：scale和gap为像素尺度，jitter及rectangle为毫米尺度；None表示
        当前画面没有足够证据。返回值：构造函数无返回值，判定结果保存在实例字段中。
        """
        self.scale_px_per_mm = (
            None if scale_px_per_mm is None else float(scale_px_per_mm)
        )
        self.minimum_gap_px = (
            None if minimum_gap_px is None else float(minimum_gap_px)
        )
        self.stable_frames = int(stable_frames)
        self.stability_window = int(stability_window)
        self.jitter_mm = None if jitter_mm is None else float(jitter_mm)
        self.rectangle_size_mm = (
            None
            if rectangle_size_mm is None
            else tuple(float(value) for value in rectangle_size_mm)
        )

        self.scale_ok = (
            self.scale_px_per_mm is not None
            and self.scale_px_per_mm >= MIN_SCALE_PX_PER_MM
        )
        self.gap_ok = (
            self.minimum_gap_px is not None
            and self.minimum_gap_px >= MIN_GAP_PX
        )
        self.jitter_ok = (
            self.jitter_mm is not None and self.jitter_mm <= MAX_JITTER_MM
        )
        self.stability_ok = (
            self.stable_frames >= self.stability_window and self.jitter_ok
        )
        self.rectangle_ok = False
        if self.rectangle_size_mm is not None:
            long_side, short_side = self.rectangle_size_mm
            self.rectangle_ok = (
                abs(long_side - RECTANGLE_TARGET_MM[0]) <= RECTANGLE_TOLERANCE_MM
                and abs(short_side - RECTANGLE_TARGET_MM[1])
                <= RECTANGLE_TOLERANCE_MM
            )


class CalibrationStabilityTracker:
    """在固定机位下跟踪最多四片碎片的跨帧毫米位置稳定性。

    主要流程：第一帧中心作为关联参考；后续帧枚举最多24种排列并选择总位移最小
    的对应关系；数量变化时立即重新计数。这样碎片字典顺序抖动不会被误判为位移。
    """

    def __init__(self, window_size=10):
        """创建固定长度观测窗口。

        关键参数：window_size为要求的连续有效帧数，必须大于1。返回值：构造函数
        无返回值；窗口数据通过update、reset及只读属性访问。
        """
        self.window_size = int(window_size)
        if self.window_size <= 1:
            raise ValueError("稳定窗口必须大于1帧")
        self._frames = []

    @property
    def stable_frames(self):
        """返回当前连续且碎片数量一致的有效观测帧数。"""
        return len(self._frames)

    @property
    def jitter_mm(self):
        """返回窗口内中心或顶点集合的最大跨帧Hausdorff位移。

        只看质心会漏掉轮廓尺寸、旋转和顶点抖动；这里对每两帧的同一碎片同时比较
        中心距离与顶点集合双向最近距离，取整个窗口最大值作为保守JITTER。
        """
        if not self._frames:
            return None
        if len(self._frames) == 1:
            return 0.0

        maximum_jitter = 0.0
        for first_frame, second_frame in itertools.combinations(self._frames, 2):
            for first_piece, second_piece in zip(first_frame, second_frame):
                center_distance = float(
                    np.linalg.norm(first_piece["center"] - second_piece["center"])
                )
                first_vertices = first_piece["vertices"]
                second_vertices = second_piece["vertices"]
                pairwise = np.linalg.norm(
                    first_vertices[:, None, :] - second_vertices[None, :, :],
                    axis=2,
                )
                hausdorff_distance = max(
                    float(np.max(np.min(pairwise, axis=1))),
                    float(np.max(np.min(pairwise, axis=0))),
                )
                maximum_jitter = max(
                    maximum_jitter,
                    center_distance,
                    hausdorff_distance,
                )
        return float(maximum_jitter)

    @property
    def is_stable(self):
        """返回窗口是否已满且最大毫米抖动不超过门槛。"""
        jitter_mm = self.jitter_mm
        return bool(
            self.stable_frames >= self.window_size
            and jitter_mm is not None
            and jitter_mm <= MAX_JITTER_MM
        )

    def reset(self):
        """清空连续帧窗口；参数、ROI或碎片数量改变时调用。"""
        self._frames = []

    def update(self, pieces):
        """加入一帧完整碎片的毫米中心并返回当前连续帧数。

        关键参数：pieces为检测字典序列，仅接收含合法center_mm的完整碎片。
        没有有效中心时清空窗口；数量改变时从当前帧重新开始；正常情况下按第一帧
        最小总位移关联并只保留最近window_size帧。
        """
        observations = []
        for piece in pieces:
            if piece.get("complete", True) is not True:
                continue
            center = piece.get("center_mm")
            if center is None or len(center) != 2:
                continue
            center_array = np.asarray(center, dtype=np.float64)
            if not np.all(np.isfinite(center_array)):
                continue
            vertices = np.asarray(
                piece.get("vertices_mm", [center_array]),
                dtype=np.float64,
            )
            if (
                vertices.ndim != 2
                or vertices.shape[0] < 1
                or vertices.shape[1] != 2
                or not np.all(np.isfinite(vertices))
            ):
                continue
            observations.append(
                {
                    "center": center_array,
                    "vertices": vertices,
                }
            )

        # 题目最多四片；超过上限说明分割存在噪声，必须立即WAIT，不能枚举n!关联。
        if not 1 <= len(observations) <= 4:
            self.reset()
            return self.stable_frames

        if self._frames and len(observations) != len(self._frames[0]):
            self.reset()

        if self._frames:
            reference = self._frames[0]
            best_order = None
            best_score = float("inf")
            for order in itertools.permutations(range(len(observations))):
                ordered = [observations[index] for index in order]
                score = sum(
                    float(np.linalg.norm(item["center"] - reference_item["center"]))
                    for item, reference_item in zip(ordered, reference)
                )
                if score < best_score:
                    best_score = score
                    best_order = order
            current = [observations[index] for index in best_order]
        else:
            # 第一帧按坐标排序，令后续关联参考与检测轮廓返回顺序无关。
            current = sorted(
                observations,
                key=lambda item: (float(item["center"][0]), float(item["center"][1])),
            )

        self._frames.append(current)
        if len(self._frames) > self.window_size:
            self._frames.pop(0)
        return self.stable_frames


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
        self._work_item_index = 0
        self._segment_item_index = 0
        self._step_index = CALIBRATION_STEPS.index(5)
        self.status_text = "ROI NOT SET" if self.settings["paper_quad"] is None else "ROI READY"
        self.paper_confidence = None
        # 跟踪器属于未保存会话；退出CAL后自动丢弃，避免跨标定批次沿用旧稳定帧。
        self.measurement_tracker = CalibrationStabilityTracker(window_size=10)

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
        """ADV页返回当前分割参数，其余简化页返回当前机械毫米参数。"""
        if self.view == VIEW_ADV:
            return SEGMENT_ITEMS[self._segment_item_index]
        return WORK_ITEMS[self._work_item_index]

    def select_view(self, view):
        """切换可见页或隐藏MEASURE页并返回新页面名称。"""
        view = str(view).lower()
        if view not in SELECTABLE_CALIBRATION_VIEWS:
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
            "work_dec",
            "work_value",
            "work_inc",
            "lock_roi",
        )

    def cycle_work_item(self):
        """按X、Y、W、H、SPLIT、PAPER顺序循环机械参数并返回新名称。"""
        self._work_item_index = (self._work_item_index + 1) % len(WORK_ITEMS)
        return self.current_item

    def _switch_paper_orientation(self):
        """切换V/H方向并重置对应默认机械区和水平分界线。

        方向改变后旧X/Y/W/H可能超出新纸面边界，因此必须作为一组替换后再整体校验。
        返回值：规范化后的新设置字典；本方法只修改会话，不写入磁盘。
        """
        current_orientation = self.settings["paper_orientation"]
        if current_orientation == PAPER_ORIENTATION_PORTRAIT:
            target_orientation = PAPER_ORIENTATION_LANDSCAPE
        else:
            target_orientation = PAPER_ORIENTATION_PORTRAIT
        work_region = default_work_region_mm(target_orientation)
        updated = dict(self.settings)
        updated.update(
            {
                "paper_orientation": target_orientation,
                "work_x_mm": work_region[0],
                "work_y_mm": work_region[1],
                "work_width_mm": work_region[2],
                "work_height_mm": work_region[3],
                "split_y_mm": default_split_y_mm(target_orientation),
            }
        )
        return validate_runtime_settings(updated, self.frame_size)

    def adjust_work(self, direction):
        """以0.5mm步进调整当前机械参数，并复用完整设置校验保护物理边界。

        关键参数：direction只允许-1或1。返回值：参数改变返回True；到达A4、
        230mm行程或分界线边界时保持原值、显示WORK LIMIT并返回False。
        """
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("调节方向必须是-1或1")
        if self.current_item == "PAPER":
            # PAPER是二值选项，左右两个按钮都执行V/H切换，避免在小屏上引入第三种状态。
            self.settings = self._switch_paper_orientation()
            self.measurement_tracker.reset()
            orientation_label = (
                "H"
                if self.settings["paper_orientation"] == PAPER_ORIENTATION_LANDSCAPE
                else "V"
            )
            self.status_text = f"PAPER {orientation_label}"
            return True
        key = WORK_SETTING_KEYS[self.current_item]
        updated = dict(self.settings)
        updated[key] = round(float(updated[key]) + direction * WORK_STEP_MM, 1)
        try:
            normalized = validate_runtime_settings(updated, self.frame_size)
        except ValueError:
            self.status_text = "WORK LIMIT"
            return False
        self.settings = normalized
        self.measurement_tracker.reset()
        self.status_text = f"WORK {self.current_item} {self.settings[key]:.1f}mm"
        return True

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
        detected_orientation = getattr(
            location,
            "paper_orientation",
            self.settings["paper_orientation"],
        )
        if detected_orientation != self.settings["paper_orientation"]:
            # 自动判向改变时同步重置毫米区域；否则保留用户此前对同方向做的手动微调。
            work_region = default_work_region_mm(detected_orientation)
            updated.update(
                {
                    "paper_orientation": detected_orientation,
                    "work_x_mm": work_region[0],
                    "work_y_mm": work_region[1],
                    "work_width_mm": work_region[2],
                    "work_height_mm": work_region[3],
                    "split_y_mm": default_split_y_mm(detected_orientation),
                }
            )
        # 复用完整设置校验，确保候选四角在保存前已经满足画面边界与凸性约束。
        self.settings = validate_runtime_settings(updated, self.frame_size)
        self.measurement_tracker.reset()
        self.paper_confidence = float(location.confidence)
        # AUTO成功后把方向直接显示在屏幕状态栏：H表示297mm长边水平，V表示长边
        # 竖直。现场无需再从X/Y裁剪值反推方向，可立即发现横竖判断是否符合实物。
        orientation_label = (
            "H"
            if self.settings["paper_orientation"] == PAPER_ORIENTATION_LANDSCAPE
            else "V"
        )
        self.status_text = (
            f"AUTO ROI OK {orientation_label} "
            f"{self.paper_confidence * 100.0:.0f}%"
        )
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
        self.measurement_tracker.reset()
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
        self.measurement_tracker.reset()
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
            return self.adjust_work(direction)
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
        self.measurement_tracker.reset()
        return True

    def _adjust_min_area(self, direction):
        """按0.05%固定步长调整最小面积比例并限制到合法范围。"""
        current = float(self.settings["min_area_ratio"])
        candidate = current + direction * 0.0005
        candidate = round(max(0.0001, min(0.25, candidate)), 7)
        if candidate == current:
            return False
        self.settings["min_area_ratio"] = candidate
        self.measurement_tracker.reset()
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
        self.measurement_tracker.reset()
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


def _measure_paper_scale(paper_quad):
    """由完整A4四边形的四条边计算保守像素密度。

    主要流程：像素角点是覆盖范围内的像素中心，因此边长加1得到覆盖像素数；再比较
    两种210/297mm交替方向，选择四边尺度离散度更小的一组并取最小值。这样四角
    循环换起点不会交换A4长短边。无合法四角时返回None。
    """
    if paper_quad is None:
        return None
    quad = np.asarray(paper_quad, dtype=np.float32)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return None
    edge_coverages = [
        float(np.linalg.norm(quad[(index + 1) % 4] - quad[index])) + 1.0
        for index in range(4)
    ]
    if min(edge_coverages) <= 1.0:
        return None
    assignments = (
        (210.0, 297.0, 210.0, 297.0),
        (297.0, 210.0, 297.0, 210.0),
    )
    candidates = []
    for physical_lengths in assignments:
        scales = np.asarray(
            [
                coverage / physical_length
                for coverage, physical_length in zip(
                    edge_coverages,
                    physical_lengths,
                )
            ],
            dtype=np.float64,
        )
        relative_spread = float(np.std(scales) / max(np.mean(scales), 1e-9))
        candidates.append((relative_spread, scales))
    _spread, selected_scales = min(candidates, key=lambda item: item[0])
    return float(np.min(selected_scales))


def _measure_minimum_component_gap(mask, max_components=4):
    """测量二值掩膜中任意两块白色连通域之间的最窄黑缝像素数。

    先按面积保留与有效碎片数量一致且最多四个的最大连通域，排除小白点噪声；再对
    每个保留域执行一次距离变换。两个边界像素中心相距d时，中间黑缝为d-1。
    少于两个保留连通域时返回None，整图距离变换次数永远不超过4次。
    """
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    component_count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 2:
        return None

    component_limit = max(0, min(4, int(max_components)))
    if component_limit < 2:
        return None
    selected_components = sorted(
        range(1, component_count),
        key=lambda index: int(statistics[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:component_limit]
    if len(selected_components) < 2:
        return None

    minimum_distance = float("inf")
    for selected_index, component_index in enumerate(selected_components):
        source = np.where(labels == component_index, 0, 255).astype(np.uint8)
        distances = cv2.distanceTransform(
            source,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        for other_index in selected_components[selected_index + 1 :]:
            other_pixels = labels == other_index
            if not np.any(other_pixels):
                continue
            minimum_distance = min(
                minimum_distance,
                float(np.min(distances[other_pixels])),
            )
    if not np.isfinite(minimum_distance):
        return None
    return float(max(0.0, minimum_distance - 1.0))


def _build_valid_piece_gap_mask(result, pieces):
    """优先用已通过视觉过滤的完整碎片轮廓重建GAP专用掩膜。

    DetectionResult.mask仍含过大、过小和其他阈值前景，不能直接代表有效碎片。若
    所有完整碎片都有真实轮廓，本函数把全帧轮廓减去ROI偏移后填入局部掩膜；测试
    替身或旧调用缺少轮廓时返回原mask，由面积上限回退逻辑处理。输入字典不修改。
    """
    source_mask = getattr(result, "mask", None)
    if source_mask is None or not isinstance(source_mask, np.ndarray):
        return source_mask
    complete_pieces = [
        piece for piece in pieces if piece.get("complete", True) is True
    ][:4]
    if len(complete_pieces) < 2 or any(
        piece.get("contour") is None for piece in complete_pieces
    ):
        return source_mask

    roi = getattr(result, "roi", (0, 0, source_mask.shape[1], source_mask.shape[0]))
    try:
        roi_x, roi_y = int(roi[0]), int(roi[1])
        gap_mask = np.zeros(source_mask.shape[:2], dtype=np.uint8)
        offset = np.asarray([[[roi_x, roi_y]]], dtype=np.int32)
        for piece in complete_pieces:
            contour = np.asarray(piece["contour"], dtype=np.int32)
            if contour.ndim != 3 or contour.shape[0] < 3 or contour.shape[-1] != 2:
                return source_mask
            local_contour = contour - offset
            cv2.fillPoly(gap_mask, [local_contour], 255)
    except (IndexError, TypeError, ValueError):
        return source_mask
    return gap_mask


def _measure_standard_rectangle(pieces):
    """在仅有一片完整标准卡时测量其最小外接矩形毫米长短边。

    关键参数：pieces为检测字典序列，vertices_mm必须已经由A4单应反算。
    返回值：按长边、短边排列的二元组；数量不是一片或顶点无效时返回None。
    """
    complete_pieces = [
        piece for piece in pieces if piece.get("complete", True) is True
    ]
    if len(complete_pieces) != 1:
        return None
    vertices = np.asarray(
        complete_pieces[0].get("vertices_mm", []),
        dtype=np.float32,
    )
    if (
        vertices.ndim != 2
        or vertices.shape[0] < 3
        or vertices.shape[1] != 2
        or not np.all(np.isfinite(vertices))
    ):
        return None
    _center, size, _angle = cv2.minAreaRect(vertices)
    long_side, short_side = sorted(
        (float(size[0]), float(size[1])),
        reverse=True,
    )
    if short_side <= 0.0:
        return None
    return long_side, short_side


def evaluate_calibration_measurement(result, paper_quad, tracker=None):
    """计算MEASURE页面的SCALE、GAP、JITTER、稳定帧和RECT证据。

    主要流程：A4四边给出最差像素密度，二值掩膜给出最窄真实黑缝，毫米中心进入
    可选跨帧跟踪器，单张完整卡给出100×60mm尺寸误差。关键参数tracker为None时
    只计算单帧指标；传入跟踪器时本函数会加入当前帧。返回CalibrationMeasurement。
    """
    pieces = list(getattr(result, "pieces", []))
    scale_px_per_mm = _measure_paper_scale(paper_quad)
    complete_piece_count = min(
        4,
        sum(1 for piece in pieces if piece.get("complete", True) is True),
    )
    gap_mask = _build_valid_piece_gap_mask(result, pieces)
    minimum_gap_px = _measure_minimum_component_gap(
        gap_mask,
        max_components=complete_piece_count,
    )
    rectangle_size_mm = _measure_standard_rectangle(pieces)

    stable_frames = 0
    stability_window = 10
    jitter_mm = None
    if tracker is not None:
        tracker.update(pieces)
        stable_frames = tracker.stable_frames
        stability_window = tracker.window_size
        jitter_mm = tracker.jitter_mm

    return CalibrationMeasurement(
        scale_px_per_mm=scale_px_per_mm,
        minimum_gap_px=minimum_gap_px,
        stable_frames=stable_frames,
        stability_window=stability_window,
        jitter_mm=jitter_mm,
        rectangle_size_mm=rectangle_size_mm,
    )


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


def _draw_roi_preview(frame_bgr, session, display_size=None):
    """在固定640×480画布上绘制完整A4和机械有效四边形。

    完整纸张使用青色，33.5mm裁剪和INSET后的有效区使用黄色；没有锁定四角时
    保留兼容矩形ROI，确保首次标定仍有明确参照。display_size为空时沿用会话尺寸。
    """
    target_width, target_height = (
        session.frame_size if display_size is None else display_size
    )
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
    work_region = (
        session.settings["work_x_mm"],
        session.settings["work_y_mm"],
        session.settings["work_width_mm"],
        session.settings["work_height_mm"],
    )
    paper_orientation = session.settings["paper_orientation"]
    active_array = build_work_quad(
        paper_array,
        work_region,
        paper_orientation=paper_orientation,
    )
    split_segment = build_split_segment(
        paper_array,
        work_region,
        session.settings["split_y_mm"],
        paper_orientation=paper_orientation,
    )
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
    cv2.line(
        output,
        tuple(np.rint(split_segment[0] * display_scale).astype(np.int32)),
        tuple(np.rint(split_segment[1] * display_scale).astype(np.int32)),
        COLOR_SPLIT,
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


def _draw_mask_preview(frame_bgr, result, session, display_size=None):
    """把有效区二值掩膜等比显示到目标画布，外部保持深灰。"""
    if result.mask is None or result.mask.ndim != 2:
        raise ValueError("二值掩膜必须是二维图像")
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    target_size = session.frame_size if display_size is None else display_size
    return _place_preview_content(mask_bgr, target_size)


def _draw_result_preview(frame_bgr, result, session, display_size=None):
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
    target_size = session.frame_size if display_size is None else display_size
    return _place_preview_content(content, target_size)


def _measurement_state_text(available, passed):
    """把测量证据状态转换成屏幕使用的WAIT、PASS或FAIL短文本。"""
    if not available:
        return "WAIT"
    return "PASS" if passed else "FAIL"


def _draw_measurement_preview(frame_bgr, result, session, measurement, display_size):
    """在RESULT轮廓预览上覆盖五行量化标定证据。

    主要流程：先绘制实际轮廓，再在预览区左侧覆盖半高深色信息面板；每行同时显示
    数值与PASS/FAIL/WAIT，现场人员无需凭肉眼猜阈值是否合适。返回目标尺寸画布。
    """
    output = _draw_result_preview(
        frame_bgr,
        result,
        session,
        display_size=display_size,
    )
    if measurement is None:
        measurement = CalibrationMeasurement()

    scale_text = (
        "SCALE -- WAIT"
        if measurement.scale_px_per_mm is None
        else (
            f"SCALE {measurement.scale_px_per_mm:.2f}px/mm "
            f"{_measurement_state_text(True, measurement.scale_ok)}"
        )
    )
    gap_text = (
        "GAP -- WAIT"
        if measurement.minimum_gap_px is None
        else (
            f"GAP {measurement.minimum_gap_px:.1f}px "
            f"{_measurement_state_text(True, measurement.gap_ok)}"
        )
    )
    stable_available = measurement.stable_frames >= measurement.stability_window
    stable_text = (
        f"STABLE {measurement.stable_frames}/{measurement.stability_window} "
        f"{_measurement_state_text(stable_available, measurement.stability_ok)}"
    )
    jitter_text = (
        "JITTER -- WAIT"
        if measurement.jitter_mm is None
        else (
            f"JITTER {measurement.jitter_mm:.2f}mm "
            f"{_measurement_state_text(stable_available, measurement.jitter_ok)}"
        )
    )
    rectangle_text = (
        "RECT -- WAIT"
        if measurement.rectangle_size_mm is None
        else (
            f"RECT {measurement.rectangle_size_mm[0]:.1f}x"
            f"{measurement.rectangle_size_mm[1]:.1f}mm "
            f"{_measurement_state_text(True, measurement.rectangle_ok)}"
        )
    )
    lines = (scale_text, gap_text, stable_text, jitter_text, rectangle_text)

    panel_x1 = 8
    panel_y1 = 72
    panel_x2 = min(output.shape[1] - 8, 350)
    panel_y2 = min(output.shape[0] - 136, panel_y1 + 174)
    cv2.rectangle(output, (panel_x1, panel_y1), (panel_x2, panel_y2), (8, 8, 8), -1)
    cv2.rectangle(output, (panel_x1, panel_y1), (panel_x2, panel_y2), COLOR_ACTIVE, 1)
    for index, line in enumerate(lines):
        passed = line.endswith("PASS")
        waiting = line.endswith("WAIT")
        color = (180, 180, 180) if waiting else ((0, 230, 0) if passed else (0, 0, 255))
        cv2.putText(
            output,
            line,
            (panel_x1 + 8, panel_y1 + 27 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _current_value_label(session, result):
    """根据当前参数项生成底部中间按钮的简短数值标签。"""
    item = session.current_item
    if item == "PAPER":
        return (
            "H"
            if session.settings["paper_orientation"] == PAPER_ORIENTATION_LANDSCAPE
            else "V"
        )
    if item in WORK_ITEMS:
        return f"{float(session.settings[WORK_SETTING_KEYS[item]]):.1f}mm"
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
    measurement=None,
    display_size=None,
):
    """绘制固定显示尺寸的ROI、MASK、RESULT、MEASURE或ADV调参界面。

    主要流程：按当前页面生成预览副本，覆盖顶部页签、诊断状态栏和底部参数按钮。
    关键参数：result 为当前工作参数产生的检测结果，buttons 为固定触摸布局，
    quality 控制状态颜色；measurement提供量化证据；display_size为空时沿用会话尺寸。
    返回值：目标显示尺寸的BGR新图像；高分辨率输入相机帧保持不变。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr必须是三通道BGR图像")

    target_size = session.frame_size if display_size is None else tuple(
        int(value) for value in display_size
    )
    if min(target_size) <= 0:
        raise ValueError("显示宽高必须大于零")

    if session.view == VIEW_ROI:
        output = _draw_roi_preview(frame_bgr, session, target_size)
    elif session.view == VIEW_MASK:
        output = _draw_mask_preview(frame_bgr, result, session, target_size)
    elif session.view == VIEW_MEASURE:
        output = _draw_measurement_preview(
            frame_bgr,
            result,
            session,
            measurement,
            target_size,
        )
    else:
        output = _draw_result_preview(frame_bgr, result, session, target_size)
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
        "result": "MEASURE" if session.view == VIEW_MEASURE else "RESULT",
        "adv": "ADV",
        "cal": "CAL",
    }
    for name in ("roi", "mask", "result", "adv", "cal"):
        _draw_button(
            output,
            buttons[name],
            top_labels[name],
            active=(
                name == session.view
                or (name == VIEW_RESULT and session.view == VIEW_MEASURE)
                or name == "cal"
            ),
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
            f"{session.current_item} {_current_value_label(session, result)}",
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
