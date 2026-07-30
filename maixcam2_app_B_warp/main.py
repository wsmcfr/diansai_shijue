"""MaixCAM2 拼图碎片识别应用入口。"""

import os
import sys

import cv2
import numpy as np

# MaixVision直接运行本文件时补入项目父目录；作为maixcam2_app包导入时不改变搜索路径。
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from maixcam2_app_B_warp.calibration_ui import (
        CalibrationSession,
        draw_calibration_frame,
        evaluate_calibration,
    )
    from maixcam2_app_B_warp.config import DEFAULT_CONFIG, PERSISTENT_SETTINGS_PATH
    from maixcam2_app_B_warp.paper_locator import locate_black_paper
    from maixcam2_app_B_warp.paper_warp import (
        WORK_SIZE_PX,
        pixels_to_work_mm,
        warp_to_work_area,
    )
    from maixcam2_app_B_warp.puzzle_vision import (
        DetectionResult,
        assign_unknown_ids,
        detect_pieces,
    )
    from maixcam2_app_B_warp.settings_store import (
        build_default_runtime_settings,
        load_runtime_settings,
        merge_paper_settings,
        merge_runtime_config,
        merge_segmentation_settings,
        save_runtime_settings,
    )
    from maixcam2_app_B_warp.template_store import (
        load_templates,
        match_known_pieces,
        register_templates,
        save_templates,
    )
    from maixcam2_app_B_warp.touch_ui import (
        TouchReleaseTracker,
        build_button_layout,
        build_calibration_layout,
        hit_test,
        map_display_to_image,
    )
except ModuleNotFoundError as error:
    # MaixVision会把工程文件平铺到/tmp/maixpy_run，此时顶层包不存在，
    # 需要从main.py同级位置加载模块；其他模块内部缺失仍应原样抛出，避免掩盖真实依赖错误。
    if error.name != "maixcam2_app_B_warp":
        raise
    from calibration_ui import (
        CalibrationSession,
        draw_calibration_frame,
        evaluate_calibration,
    )
    from config import DEFAULT_CONFIG, PERSISTENT_SETTINGS_PATH
    from paper_locator import locate_black_paper
    from paper_warp import WORK_SIZE_PX, pixels_to_work_mm, warp_to_work_area
    from puzzle_vision import DetectionResult, assign_unknown_ids, detect_pieces
    from settings_store import (
        build_default_runtime_settings,
        load_runtime_settings,
        merge_paper_settings,
        merge_runtime_config,
        merge_segmentation_settings,
        save_runtime_settings,
    )
    from template_store import (
        load_templates,
        match_known_pieces,
        register_templates,
        save_templates,
    )
    from touch_ui import (
        TouchReleaseTracker,
        build_button_layout,
        build_calibration_layout,
        hit_test,
        map_display_to_image,
    )


# 模式使用稳定字符串，既用于逻辑判断，也用于屏幕状态栏显示。
MODE_KNOWN = "known"
MODE_UNKNOWN = "unknown"


class WarpFrameAnalysis:
    """保存B版展开工作图、有效掩膜和碎片检测结果。"""

    def __init__(self, work_frame, valid_mask, detection, full_a4, homography):
        """初始化单帧展开结果，字段保持公开以供设备显示和PC对比工具复用。"""
        self.work_frame = work_frame
        self.valid_mask = valid_mask
        self.detection = detection
        self.full_a4 = full_a4
        self.homography = homography


def analyze_warped_frame(
    frame_bgr,
    paper_quad,
    inset_mm=0.0,
    runtime_settings=None,
    config=None,
):
    """执行B版完整A4展开、机械区裁剪和碎片识别。

    主要流程：使用锁定四角生成420×460工作图与INSET掩膜，在固定全工作区ROI内
    识别碎片，再按2像素/mm补充机械区毫米中心。
    返回值：``WarpFrameAnalysis``；paper_quad 缺失或单应性无效时抛出 ValueError。
    """
    if paper_quad is None:
        raise ValueError("paper_quad 未锁定，不能执行透视展开")
    warp_result = warp_to_work_area(frame_bgr, paper_quad, inset_mm)
    if runtime_settings is None:
        runtime_settings = build_default_runtime_settings(DEFAULT_CONFIG)
    base_config = DEFAULT_CONFIG if config is None else config
    detection_config = merge_runtime_config(base_config, runtime_settings)
    work_width, work_height = WORK_SIZE_PX
    detection = detect_pieces(
        warp_result.work_area,
        (0, 0, work_width, work_height),
        detection_config,
        valid_mask=warp_result.valid_mask,
    )
    for piece in detection.pieces:
        piece["center_mm"] = pixels_to_work_mm(piece["center"])
    return WarpFrameAnalysis(
        warp_result.work_area,
        warp_result.valid_mask,
        detection,
        warp_result.full_a4,
        warp_result.homography,
    )


def build_warp_display_canvas(work_frame, canvas_size=(640, 480)):
    """按210:230比例把工作图居中放入固定显示画布。

    主要流程：以画布高度为限制等比缩放，左右填充黑色，不把420×460拉伸为640×480。
    返回值：``(画布, 内容ROI)``，内容ROI用于后续绘制边框或映射触摸坐标。
    """
    if work_frame is None or not isinstance(work_frame, np.ndarray):
        raise ValueError("work_frame 必须是有效的 numpy 图像")
    if work_frame.ndim != 3 or work_frame.shape[2] != 3:
        raise ValueError("work_frame 必须是三通道 BGR 图像")
    canvas_width, canvas_height = (int(value) for value in canvas_size)
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas_size 宽高必须大于零")

    source_height, source_width = work_frame.shape[:2]
    scale = min(canvas_width / source_width, canvas_height / source_height)
    display_width = max(1, int(round(source_width * scale)))
    display_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        work_frame,
        (display_width, display_height),
        interpolation=cv2.INTER_LINEAR,
    )
    offset_x = (canvas_width - display_width) // 2
    offset_y = (canvas_height - display_height) // 2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + display_height, offset_x : offset_x + display_width] = resized
    return canvas, (offset_x, offset_y, display_width, display_height)


class InterfaceState:
    """维护正常/调参界面切换和当前未保存调参会话。

    该结构故意不保存KNOWN/UNKNOWN识别模式，因此同一个CAL按钮往返切换时不会
    意外改变碎片识别模式。
    """

    def __init__(self):
        """初始化为正常识别界面，当前没有未保存调参会话。"""
        self.calibration_session = None

    @property
    def is_calibrating(self):
        """返回当前是否处于调参界面。"""
        return self.calibration_session is not None

    def toggle_calibration(self, saved_settings, frame_size):
        """使用同一个CAL动作进入或退出调参界面。

        主要流程：正常界面时复制已保存参数并创建会话；调参界面时直接丢弃会话，
        因而未保存修改不会污染运行参数。
        关键参数：saved_settings 为当前生效参数，frame_size 为相机 ``(宽, 高)``。
        返回值：切换后处于调参界面返回 True，返回正常界面时返回 False。
        """
        if self.calibration_session is None:
            self.calibration_session = CalibrationSession(saved_settings, frame_size)
            return True
        self.calibration_session = None
        return False


def handle_calibration_action(
    action,
    interface_state,
    runtime_settings,
    detection,
    settings_path,
    frame_size,
    frame_bgr=None,
):
    """处理五页调参动作并执行两组互不覆盖的持久化。

    顶部切换预览页；底部固定槽由会话映射为纸张或高级分割动作。AUTO ROI只读取
    当前原相机帧一次，LOCK和ADV SAVE分别合并所属字段。
    返回值为 ``(运行参数, 状态文字)``。
    """
    if not interface_state.is_calibrating:
        raise ValueError("只有调参界面可以处理调参动作")

    session = interface_state.calibration_session
    action = str(action)
    if action in ("roi", "mask", "result", "adv"):
        session.select_view(action)
        return runtime_settings, action.upper()
    logical_action = session.resolve_control_action(action)
    if logical_action == "auto_roi":
        if frame_bgr is None:
            raise ValueError("AUTO ROI需要当前相机帧")
        location = locate_black_paper(frame_bgr)
        session.apply_auto_roi(location)
        return runtime_settings, session.status_text
    if logical_action == "inset_dec":
        session.adjust_inset(-1)
        return runtime_settings, session.status_text
    if logical_action == "inset_inc":
        session.adjust_inset(1)
        return runtime_settings, session.status_text
    if logical_action == "inset_value":
        return runtime_settings, f"INSET {session.settings['inset_mm']:.1f}mm"
    if logical_action == "lock_roi":
        if not session.can_lock_roi(detection):
            return runtime_settings, session.status_text
        paper_settings = merge_paper_settings(runtime_settings, session.snapshot())
        saved_settings = save_runtime_settings(
            settings_path,
            paper_settings,
            frame_size,
        )
        session.status_text = "ROI LOCKED"
        return saved_settings, session.status_text
    if logical_action == "next_param":
        return runtime_settings, f"PARAM {session.cycle_item()}"
    if logical_action == "value_dec":
        changed = session.adjust(-1)
        return runtime_settings, "ADJUSTED" if changed else "LIMIT"
    if logical_action == "value_inc":
        changed = session.adjust(1)
        return runtime_settings, "ADJUSTED" if changed else "LIMIT"
    if logical_action == "select_value":
        if session.current_item != "TH":
            return runtime_settings, f"PARAM {session.current_item}"
        session.toggle_threshold_mode(detection.threshold)
        return runtime_settings, "TH MODE"
    if logical_action == "save_segmentation":
        if not session.can_save_segmentation(detection):
            return runtime_settings, session.status_text
        segmentation_settings = merge_segmentation_settings(
            runtime_settings,
            session.snapshot(),
        )
        saved_settings = save_runtime_settings(
            settings_path,
            segmentation_settings,
            frame_size,
        )
        session.status_text = "ADV SAVED"
        return saved_settings, session.status_text
    raise ValueError(f"未知调参动作: {action}")


def format_status_text(mode, pieces, threshold, status_message):
    """生成区分有效碎片与边界轮廓的ASCII状态栏文本。

    主要流程：统计 complete 为真的可操作碎片和其余 EDGE 轮廓，再拼接模式、阈值和状态。
    关键参数：pieces 可包含完整及不完整轮廓，threshold 为当前分割阈值。
    返回值：适合 OpenCV 默认字体绘制的单行字符串。
    """
    actionable_count = sum(1 for piece in pieces if piece.get("complete") is True)
    edge_count = len(pieces) - actionable_count
    return (
        f"{mode.upper()} N={actionable_count} EDGE={edge_count} "
        f"TH={threshold:.0f} {status_message}"
    ).strip()


def draw_overlay(
    frame_bgr,
    pieces,
    roi,
    buttons,
    mode,
    threshold,
    status_message,
):
    """在相机帧副本上绘制工作区、碎片几何信息、模式按钮和状态栏。

    主要流程：复制输入帧，画 ROI 和每片轮廓/顶点/中心，再绘制顶部模式按钮与底部状态。
    关键参数：pieces 为视觉核心输出，buttons 使用图像坐标，mode 为 known 或 unknown。
    返回值：带调试叠加的 BGR 新图像；原始 frame_bgr 保持不变。
    """
    output = frame_bgr.copy()
    frame_height, frame_width = output.shape[:2]
    scale = max(0.45, min(0.75, frame_width / 900.0))
    text_thickness = 1 if frame_width < 960 else 2

    roi_x, roi_y, roi_width, roi_height = roi
    cv2.rectangle(
        output,
        (roi_x, roi_y),
        (roi_x + roi_width - 1, roi_y + roi_height - 1),
        (255, 200, 0),
        1,
    )

    for piece in pieces:
        # 完整轮廓使用绿色，不完整轮廓使用橙色，防止边界截断结果被误用于机械控制。
        contour_color = (0, 210, 0) if piece.get("complete", False) else (0, 140, 255)
        cv2.drawContours(output, [piece["contour"]], -1, contour_color, 2)

        for vertex in piece.get("vertices", []):
            cv2.circle(output, tuple(int(value) for value in vertex), 4, (0, 0, 255), -1)

        center_x = int(round(piece["center"][0]))
        center_y = int(round(piece["center"][1]))
        cv2.drawMarker(
            output,
            (center_x, center_y),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            14,
            2,
        )
        label = f'{piece.get("id", "?")} V{len(piece.get("vertices", []))} {piece.get("angle_deg", 0.0):.1f}deg'
        cv2.putText(
            output,
            label,
            (max(0, center_x + 6), max(16, center_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    for name, button in buttons.items():
        x1 = int(round(button.x))
        y1 = int(round(button.y))
        x2 = int(round(button.x + button.width))
        y2 = int(round(button.y + button.height))
        active = name == mode
        enabled = name != "save" or mode == MODE_KNOWN
        if active:
            fill_color = (32, 150, 48)
        elif enabled:
            fill_color = (60, 60, 60)
        else:
            fill_color = (25, 25, 25)
        cv2.rectangle(output, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(output, (x1, y1), (x2, y2), (220, 220, 220), 1)
        cv2.putText(
            output,
            name.upper(),
            (x1 + 7, y1 + int(button.height * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255) if enabled else (100, 100, 100),
            text_thickness,
            cv2.LINE_AA,
        )

    status_text = format_status_text(mode, pieces, threshold, status_message)
    status_height = max(24, int(round(frame_height * 0.065)))
    cv2.rectangle(
        output,
        (0, frame_height - status_height),
        (frame_width, frame_height),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        status_text[:96],
        (8, frame_height - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return output


def run_app():
    """初始化 MaixCAM2 并运行识别与调参双界面主循环。

    主要流程：加载持久现场参数、读取固定相机画面、按RUN或CAL选择参数，处理同一个
    CAL按钮的往返切换，再显示正常识别叠加或ROI/MASK/RESULT调参画面。
    关键参数：无；默认算法来自 config.DEFAULT_CONFIG，现场参数来自持久JSON。
    返回值：正常退出应用时返回 None。
    """
    from maix import app, camera, display, image, time, touchscreen

    camera_width = int(DEFAULT_CONFIG["camera_width"])
    camera_height = int(DEFAULT_CONFIG["camera_height"])
    frame_size = (camera_width, camera_height)
    template_path = os.path.join(os.path.dirname(__file__), "known_templates.json")

    disp = display.Display()
    cam = camera.Camera(camera_width, camera_height, image.Format.FMT_BGR888)
    cam.skip_frames(30)
    touch = touchscreen.TouchScreen()
    touch_tracker = TouchReleaseTracker()
    run_buttons = build_button_layout(camera_width, camera_height)
    calibration_buttons = build_calibration_layout(camera_width, camera_height)
    interface_state = InterfaceState()

    mode = MODE_UNKNOWN
    status_message = "READY"
    try:
        runtime_settings = load_runtime_settings(
            PERSISTENT_SETTINGS_PATH,
            DEFAULT_CONFIG,
        )
    except Exception as error:
        # 损坏设置不能阻止相机启动，回退默认整帧并把错误类型显示给现场人员。
        runtime_settings = build_default_runtime_settings(DEFAULT_CONFIG)
        status_message = f"SETTINGS ERROR {type(error).__name__}"

    try:
        templates = load_templates(template_path)
    except Exception as error:
        # 模板损坏不能阻止未知模式工作，错误保留在状态栏供现场判断。
        templates = []
        status_message = f"TEMPLATE ERROR {type(error).__name__}"

    while not app.need_exit():
        camera_image = cam.read()
        frame_bgr = image.image2cv(camera_image, ensure_bgr=False, copy=False)

        # CAL使用未保存工作副本，RUN只使用已经保存并生效的参数。
        if interface_state.is_calibrating:
            active_settings = interface_state.calibration_session.settings
        else:
            active_settings = runtime_settings
        roi = tuple(int(value) for value in active_settings["roi"])
        detection_config = merge_runtime_config(DEFAULT_CONFIG, active_settings)
        paper_quad = active_settings.get("paper_quad")
        warped_analysis = None

        try:
            if paper_quad is None:
                # 首次标定前显示原相机画面并沿用兼容矩形ROI，用户仍可进入CAL设置四角。
                detection = detect_pieces(frame_bgr, roi, detection_config)
                if status_message in ("READY", "RUN MODE"):
                    status_message = "ROI NOT SET"
            else:
                warped_analysis = analyze_warped_frame(
                    frame_bgr,
                    paper_quad,
                    inset_mm=active_settings.get("inset_mm", 0.0),
                    runtime_settings=active_settings,
                )
                detection = warped_analysis.detection
                roi = detection.roi
        except Exception as error:
            # 单应性错误使用固定工作区空结果；矩形回退错误仍使用当前相机ROI尺寸。
            if paper_quad is None:
                roi_x, roi_y, roi_width, roi_height = roi
                empty_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
                error_prefix = "VISION ERROR"
            else:
                roi_width, roi_height = WORK_SIZE_PX
                roi = (0, 0, roi_width, roi_height)
                empty_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
                error_prefix = "WARP ERROR"
            detection = DetectionResult(
                [],
                empty_mask,
                0.0,
                roi,
                valid_contour_count=0,
                white_ratio=0.0,
            )
            status_message = f"{error_prefix} {type(error).__name__}"

        pieces = detection.pieces
        threshold = detection.threshold
        active_buttons = (
            calibration_buttons if interface_state.is_calibrating else run_buttons
        )

        touch_x, touch_y, pressed = touch.read()
        clicked_display = touch_tracker.update(touch_x, touch_y, pressed)
        if clicked_display is not None:
            clicked_image = map_display_to_image(
                clicked_display,
                (camera_width, camera_height),
                (disp.width(), disp.height()),
            )
            action = hit_test(clicked_image, active_buttons)
            if action == "cal":
                entered_calibration = interface_state.toggle_calibration(
                    runtime_settings,
                    frame_size,
                )
                status_message = "CAL MODE" if entered_calibration else "RUN MODE"
            elif interface_state.is_calibrating and action is not None:
                try:
                    runtime_settings, status_message = handle_calibration_action(
                        action,
                        interface_state,
                        runtime_settings,
                        detection,
                        PERSISTENT_SETTINGS_PATH,
                        frame_size,
                        frame_bgr=frame_bgr,
                    )
                except Exception as error:
                    # 调参动作失败保持旧运行参数和当前会话，便于用户修正后重试。
                    status_message = f"CAL ERROR {type(error).__name__}"
            elif action == MODE_KNOWN:
                mode = MODE_KNOWN
                status_message = "KNOWN MODE"
            elif action == MODE_UNKNOWN:
                mode = MODE_UNKNOWN
                status_message = "UNKNOWN MODE"
            elif action == "save" and mode == MODE_KNOWN:
                if len(pieces) == 4 and all(piece["complete"] for piece in pieces):
                    try:
                        templates = register_templates(pieces)
                        save_templates(template_path, templates)
                        status_message = "4 TEMPLATES SAVED"
                    except Exception as error:
                        status_message = f"SAVE ERROR {type(error).__name__}"
                else:
                    status_message = "SAVE NEEDS 4 COMPLETE"

        if interface_state.is_calibrating:
            quality = evaluate_calibration(detection)
            calibration_source = frame_bgr
            if (
                interface_state.calibration_session.view != "roi"
                and warped_analysis is not None
            ):
                # B版非ROI页面使用展开工作图，保证MASK和轮廓与检测坐标完全一致。
                calibration_source = warped_analysis.work_frame
            display_frame = draw_calibration_frame(
                calibration_source,
                detection,
                interface_state.calibration_session,
                calibration_buttons,
                quality,
                status_message,
            )
        else:
            if mode == MODE_KNOWN:
                match_known_pieces(
                    pieces,
                    templates,
                    float(DEFAULT_CONFIG["known_match_threshold"]),
                )
                if not templates and not status_message.startswith("TEMPLATE ERROR"):
                    status_message = "NO TEMPLATE"
            else:
                assign_unknown_ids(
                    pieces,
                    row_tolerance_px=max(
                        20,
                        (WORK_SIZE_PX[1] if warped_analysis is not None else camera_height)
                        * 0.08,
                    ),
                )

            if warped_analysis is None:
                # 尚未锁定或本帧展开失败时显示原相机画面，明确保留进入CAL的路径。
                display_roi = tuple(int(value) for value in runtime_settings["roi"])
                display_frame = draw_overlay(
                    frame_bgr,
                    pieces,
                    display_roi,
                    run_buttons,
                    mode,
                    threshold,
                    status_message,
                )
            else:
                # 先在420×460原比例工作图绘制几何结果，再居中到640×480显示画布。
                work_overlay = draw_overlay(
                    warped_analysis.work_frame,
                    pieces,
                    (0, 0, WORK_SIZE_PX[0], WORK_SIZE_PX[1]),
                    {},
                    mode,
                    threshold,
                    status_message,
                )
                display_frame, content_roi = build_warp_display_canvas(work_overlay)
                # 第二层只绘制640×480坐标下的运行按钮和状态，触摸命中区域不会被压缩。
                display_frame = draw_overlay(
                    display_frame,
                    [],
                    content_roi,
                    run_buttons,
                    mode,
                    threshold,
                    status_message,
                )
        display_image = image.cv2image(display_frame, bgr=True, copy=False)
        disp.show(display_image, fit=image.Fit.FIT_CONTAIN)
        time.sleep_ms(1)


if __name__ == "__main__":
    run_app()
