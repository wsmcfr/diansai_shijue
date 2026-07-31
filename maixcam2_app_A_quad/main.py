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
    from maixcam2_app_A_quad.assembly_planner import (
        AssemblyRuntime,
        KnownRegistrationJob,
        UNKNOWN_PROFILE_CARD,
        UNKNOWN_PROFILE_WHITE,
        UNKNOWN_SOLVER_DEBUG,
        classify_piece_region,
        draw_assembly_plan,
        solve_and_register_known_layout,
    )
    from maixcam2_app_A_quad.calibration_ui import (
        CalibrationSession,
        VIEW_MEASURE,
        VIEW_RESULT,
        draw_calibration_frame,
        evaluate_calibration,
        evaluate_calibration_measurement,
    )
    from maixcam2_app_A_quad.config import (
        DEFAULT_CONFIG,
        PERSISTENT_SETTINGS_PATH,
        PERSISTENT_TEMPLATE_PATH,
    )
    from maixcam2_app_A_quad.four_piece_solver import FourPieceRuntime
    from maixcam2_app_A_quad.paper_locator import (
        PAPER_ORIENTATION_PORTRAIT,
        build_work_quad,
        image_point_to_paper_mm,
        image_points_to_paper_mm,
        infer_paper_orientation,
        locate_black_paper,
        orient_a4_quad_for_coordinates,
        order_a4_quad,
        paper_size_mm,
        validate_paper_orientation,
    )
    from maixcam2_app_A_quad.puzzle_vision import (
        DetectionResult,
        assign_unknown_ids,
        detect_pieces,
        sample_piece_edge_features,
    )
    from maixcam2_app_A_quad.settings_store import (
        build_default_runtime_settings,
        load_runtime_settings,
        merge_paper_settings,
        merge_runtime_config,
        merge_segmentation_settings,
        save_runtime_settings,
    )
    from maixcam2_app_A_quad.serial_protocol import (
        VisionSerialRuntime,
        create_maix_uart4,
    )
    from maixcam2_app_A_quad.template_store import (
        load_templates,
        match_known_pieces,
        save_templates,
    )
    from maixcam2_app_A_quad.touch_ui import (
        TouchReleaseTracker,
        build_button_layout,
        build_calibration_layout,
        hit_test,
        map_display_to_image,
    )
except ModuleNotFoundError as error:
    # MaixVision会把工程文件平铺到/tmp/maixpy_run，此时顶层包不存在，
    # 需要从main.py同级位置加载模块；其他模块内部缺失仍应原样抛出，避免掩盖真实依赖错误。
    if error.name != "maixcam2_app_A_quad":
        raise
    from assembly_planner import (
        AssemblyRuntime,
        KnownRegistrationJob,
        UNKNOWN_PROFILE_CARD,
        UNKNOWN_PROFILE_WHITE,
        UNKNOWN_SOLVER_DEBUG,
        classify_piece_region,
        draw_assembly_plan,
        solve_and_register_known_layout,
    )
    from calibration_ui import (
        CalibrationSession,
        VIEW_MEASURE,
        VIEW_RESULT,
        draw_calibration_frame,
        evaluate_calibration,
        evaluate_calibration_measurement,
    )
    from config import DEFAULT_CONFIG, PERSISTENT_SETTINGS_PATH, PERSISTENT_TEMPLATE_PATH
    from four_piece_solver import FourPieceRuntime
    from paper_locator import (
        PAPER_ORIENTATION_PORTRAIT,
        build_work_quad,
        image_point_to_paper_mm,
        image_points_to_paper_mm,
        infer_paper_orientation,
        locate_black_paper,
        orient_a4_quad_for_coordinates,
        order_a4_quad,
        paper_size_mm,
        validate_paper_orientation,
    )
    from puzzle_vision import (
        DetectionResult,
        assign_unknown_ids,
        detect_pieces,
        sample_piece_edge_features,
    )
    from settings_store import (
        build_default_runtime_settings,
        load_runtime_settings,
        merge_paper_settings,
        merge_runtime_config,
        merge_segmentation_settings,
        save_runtime_settings,
    )
    from serial_protocol import VisionSerialRuntime, create_maix_uart4
    from template_store import (
        load_templates,
        match_known_pieces,
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
# FOUR只用于设备内部路径隔离；发送UART结果时仍映射为协议既有的UNKNOWN模式。
MODE_FOUR = "four"
# FOUR第三功能按钮循环的纸面预览；camera显示实景，其余三项直接显示分割阶段掩膜。
FOUR_DEBUG_VIEWS = ("camera", "strict", "support", "final")
FOUR_DEBUG_VIEW_LABELS = {
    "camera": "CAM",
    "strict": "CORE",
    "support": "SUPPORT",
    "final": "FINAL",
}


def format_auto_roi_diagnostic_fields(location):
    """把AUTO ROI结构化诊断转换为稳定的单行字段。

    主要流程：先根据各拒绝计数生成``gates``标签，再依次输出计数、最大轮廓面积、
    非四角顶点分布和最佳四角候选的评分分量。关键参数location可来自真实定位器或
    测试替身；缺少diagnostics、字段为空或单个值非法时安全省略对应字段。
    返回值：不含首尾空格的ASCII诊断字符串；没有诊断数据时返回空字符串。
    """
    diagnostics = getattr(location, "diagnostics", None)
    if not isinstance(diagnostics, dict) or not diagnostics:
        return ""

    def safe_int(key, default=0):
        """读取非负整数统计，非法值回退default，避免调试日志中断AUTO流程。"""
        try:
            return max(0, int(diagnostics.get(key, default)))
        except (TypeError, ValueError):
            return int(default)

    # gates保留所有实际出现的拒绝门，而不是猜测单一根因；同一帧多个外轮廓可能
    # 分别在面积、四角和矩形度阶段失败，同时显示才能避免现场误判。
    gate_specs = (
        ("area_small_count", "AREA_SMALL"),
        ("area_large_count", "AREA_LARGE"),
        ("not_quad_count", "NOT_QUAD"),
        ("rectangularity_reject_count", "RECT_LOW"),
    )
    gates = [label for key, label in gate_specs if safe_int(key) > 0]
    if str(getattr(location, "reason", "")) == "low_confidence":
        gates.append("CONF_LOW")
    if not gates:
        gates.append("NO_DARK_CONTOUR" if safe_int("contour_count") == 0 else "PASS")

    fields = [f"gates={','.join(gates)}"]
    count_fields = (
        ("contour_count", "contours"),
        ("area_small_count", "area_small"),
        ("area_large_count", "area_large"),
        ("not_quad_count", "not_quad"),
        ("rectangularity_reject_count", "rect_low"),
        ("eligible_count", "eligible"),
    )
    fields.extend(f"{label}={safe_int(key)}" for key, label in count_fields)

    try:
        largest_area_ratio = float(diagnostics.get("largest_area_ratio", 0.0))
        if np.isfinite(largest_area_ratio):
            fields.append(f"largest_area={largest_area_ratio * 100.0:.1f}%")
    except (TypeError, ValueError):
        pass

    vertex_counts = diagnostics.get("approx_vertex_counts")
    if isinstance(vertex_counts, dict) and vertex_counts:
        normalized_vertices = []
        for raw_vertices, raw_count in vertex_counts.items():
            try:
                vertices = max(0, int(raw_vertices))
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                continue
            if count > 0:
                normalized_vertices.append((vertices, count))
        if normalized_vertices:
            normalized_vertices.sort()
            vertex_text = ",".join(
                f"{vertices}x{count}" for vertices, count in normalized_vertices
            )
            fields.append(f"quad_vertices={vertex_text}")

    best_candidate = diagnostics.get("best_candidate")
    if isinstance(best_candidate, dict):
        metric_specs = (
            ("area_ratio", "best_area", 100.0, ".1f", "%"),
            ("observed_aspect", "aspect", 1.0, ".3f", ""),
            ("aspect_score", "aspect_score", 1.0, ".3f", ""),
            ("rectangularity", "rect", 1.0, ".3f", ""),
            ("convexity", "convex", 1.0, ".3f", ""),
            ("darkness_score", "dark", 1.0, ".3f", ""),
            ("confidence", "best_conf", 100.0, ".1f", "%"),
        )
        for key, label, multiplier, number_format, suffix in metric_specs:
            try:
                value = float(best_candidate[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                fields.append(
                    f"{label}={format(value * multiplier, number_format)}{suffix}"
                )
        try:
            strict_vertex_count = max(0, int(best_candidate["strict_vertex_count"]))
            fields.append(f"strict_vertices={strict_vertex_count}")
        except (KeyError, TypeError, ValueError):
            pass
        try:
            quad_epsilon_ratio = float(best_candidate["quad_epsilon_ratio"])
            if np.isfinite(quad_epsilon_ratio):
                fields.append(f"quad_eps={quad_epsilon_ratio:.3f}")
        except (KeyError, TypeError, ValueError):
            pass
    return " ".join(fields)


def log_auto_roi_diagnostics(location, debug_enabled=None):
    """按共用调试开关输出一次AUTO ROI结果和候选门诊断。

    主要流程：debug_enabled为None时读取assembly_planner.py文件顶部开关；关闭时立即
    返回，不转换四角也不构造字符串。成功结果计算四条循环边的像素长度，横纸显示H、
    竖纸显示V；成功与失败都会追加面积、四角、矩形度和最佳候选评分字段。
    返回值始终为None，每次AUTO最多打印一行。
    """
    enabled = UNKNOWN_SOLVER_DEBUG if debug_enabled is None else bool(debug_enabled)
    if not enabled:
        return
    diagnostic_fields = format_auto_roi_diagnostic_fields(location)
    diagnostic_suffix = "" if not diagnostic_fields else f" {diagnostic_fields}"
    if not getattr(location, "success", False) or location.paper_quad is None:
        print(
            "[ROI] AUTO result=FAIL "
            f"reason={getattr(location, 'reason', 'unknown')} "
            f"confidence={float(getattr(location, 'confidence', 0.0)) * 100.0:.0f}% "
            f"threshold={float(getattr(location, 'threshold', 0.0)):.1f}"
            f"{diagnostic_suffix}"
        )
        return

    quad = np.asarray(location.paper_quad, dtype=np.float64).reshape(4, 2)
    closed_quad = np.vstack((quad, quad[:1]))
    edge_lengths = np.linalg.norm(np.diff(closed_quad, axis=0), axis=1)
    orientation_label = (
        "H" if str(getattr(location, "paper_orientation", "portrait")) == "landscape" else "V"
    )
    edge_text = ",".join(f"{length:.1f}" for length in edge_lengths)
    print(
        "[ROI] AUTO result=OK "
        f"orientation={orientation_label} "
        f"confidence={float(getattr(location, 'confidence', 0.0)) * 100.0:.0f}% "
        f"threshold={float(getattr(location, 'threshold', 0.0)):.1f} "
        f"edges_px=[{edge_text}]"
        f"{diagnostic_suffix}"
    )


def toggle_unknown_profile(current_profile):
    """在WHITE与CARD两个UNKNOWN子模式之间切换。

    关键参数current_profile必须是white或card，大小写不敏感。返回规范的小写新模式；
    非法值直接抛出ValueError，避免界面文字与实际求解策略静默不一致。
    """
    normalized_profile = str(current_profile).lower()
    if normalized_profile == UNKNOWN_PROFILE_WHITE:
        return UNKNOWN_PROFILE_CARD
    if normalized_profile == UNKNOWN_PROFILE_CARD:
        return UNKNOWN_PROFILE_WHITE
    raise ValueError("UNKNOWN子模式必须是white或card")


def toggle_four_debug_view(current_view):
    """循环切换FOUR相机、严格核心、宽松支撑和最终掩膜预览。

    关键参数current_view大小写不敏感但必须属于FOUR_DEBUG_VIEWS；返回下一个规范小写
    视图。切换只影响显示，不重置锁定快照、求解任务或UART结果。
    """
    normalized = str(current_view).strip().lower()
    if normalized not in FOUR_DEBUG_VIEWS:
        raise ValueError("FOUR调试视图必须是camera、strict、support或final")
    index = FOUR_DEBUG_VIEWS.index(normalized)
    return FOUR_DEBUG_VIEWS[(index + 1) % len(FOUR_DEBUG_VIEWS)]


def _validate_capture_runtime(planner_runtime):
    """校验采集状态动作使用的运行器接口。

    关键参数planner_runtime必须提供可调用的reset方法；校验成功无返回值，接口不完整
    时抛出ValueError。集中校验可保证模式选择和START使用完全一致的资源清理契约。
    """
    if planner_runtime is None or not callable(getattr(planner_runtime, "reset", None)):
        raise ValueError("planner_runtime必须提供reset方法")


def _reset_capture_runtimes(planner_runtime, four_runtime=None):
    """同时复位旧拼图运行器和可选FOUR运行器，确保模式之间没有共享快照。

    关键参数planner_runtime始终必需；four_runtime为None时保持旧测试和离线调用兼容。
    两个对象都必须提供reset方法。返回值为None。
    """
    _validate_capture_runtime(planner_runtime)
    planner_runtime.reset()
    if four_runtime is None:
        return
    _validate_capture_runtime(four_runtime)
    four_runtime.reset()


def _normalize_capture_mode(requested_mode):
    """把外部模式值规范为known、unknown或four，并拒绝未定义模式。

    关键参数requested_mode允许大小写和首尾空白；返回规范小写字符串，非法值抛出
    ValueError，防止按钮高亮、模板路径和求解模式出现不一致。
    """
    mode = str(requested_mode).strip().lower()
    if mode not in (MODE_KNOWN, MODE_UNKNOWN, MODE_FOUR):
        raise ValueError("识别模式必须是known、unknown或four")
    return mode


def protocol_mode_for_capture(mode):
    """把内部采集模式转换为现有UART协议支持的模式字符串。

    主要流程：先复用统一模式校验；FOUR在视觉与求解层保持独立，但线上仍属于UNKNOWN，
    因此返回unknown。KNOWN和普通UNKNOWN原样返回，F4无需增加新的模式分支。
    关键参数mode允许大小写和首尾空白。返回值为known或unknown。
    """
    normalized_mode = _normalize_capture_mode(mode)
    if normalized_mode == MODE_FOUR:
        return MODE_UNKNOWN
    return normalized_mode


def _reset_serial_result_context(serial_runtime):
    """按需清除通信运行器中的旧拼图结果上下文。

    关键参数serial_runtime允许为None，以保持纯视觉PC调用兼容；非None时必须提供
    reset_result_context方法。该函数不清除手动A4帧和心跳，只取消上一轮机械目标。
    返回值：无；接口不完整时抛出ValueError，避免静默沿用旧目标。
    """
    if serial_runtime is None:
        return
    reset_context = getattr(serial_runtime, "reset_result_context", None)
    if not callable(reset_context):
        raise ValueError("serial_runtime必须提供reset_result_context方法")
    reset_context()


def select_capture_mode(
    requested_mode,
    planner_runtime,
    serial_runtime=None,
    four_runtime=None,
):
    """选择KNOWN或UNKNOWN，并把正常页恢复为完全待机。

    主要流程：校验模式与运行器，释放旧稳定计数、锁定快照、求解任务和规划缓存；只
    保存模式选择，不启动视觉分析；同时取消旧PUZZLE_RESULT，防止F4继续执行上一轮
    目标。serial_runtime为空时保持旧测试和纯视觉调用兼容。返回值为
    ``(规范模式, False, "PRESS START")``，其中False可直接写回主循环capture_armed。
    """
    mode = _normalize_capture_mode(requested_mode)
    _reset_capture_runtimes(planner_runtime, four_runtime=four_runtime)
    _reset_serial_result_context(serial_runtime)
    return mode, False, "PRESS START"


def start_capture(
    mode,
    planner_runtime,
    unknown_profile=UNKNOWN_PROFILE_WHITE,
    serial_runtime=None,
    four_runtime=None,
):
    """确认当前选择并开始一次全新的稳定快照采集。

    主要流程：校验模式、UNKNOWN材料和运行器，始终清除上一轮状态，使重复点击START
    成为明确的重拍入口；通信上下文同步复位，使本轮成功规划可以且只可以重新发送
    一次。返回值为``(True, 状态文字)``；True可直接写回capture_armed，状态文字
    用于提示本轮实际采用的模式和材料。
    """
    normalized_mode = _normalize_capture_mode(mode)
    _reset_capture_runtimes(planner_runtime, four_runtime=four_runtime)
    _reset_serial_result_context(serial_runtime)
    if normalized_mode == MODE_KNOWN:
        return True, "KNOWN CAPTURE"
    if normalized_mode == MODE_FOUR:
        return True, "FOUR CAPTURE"
    profile = str(unknown_profile).strip().lower()
    if profile not in (UNKNOWN_PROFILE_WHITE, UNKNOWN_PROFILE_CARD):
        raise ValueError("UNKNOWN子模式必须是white或card")
    return True, f"UNKNOWN {profile.upper()} CAPTURE"


def select_capture_runtime(mode, planner_runtime, four_runtime):
    """按内部模式返回本帧状态、显示和心跳应读取的唯一运行器。

    FOUR只选择专用运行器；KNOWN和普通UNKNOWN继续选择原AssemblyRuntime。两个运行器
    均不能为空，非法模式通过统一规范函数拒绝。返回值为传入对象本身。
    """
    normalized_mode = _normalize_capture_mode(mode)
    if planner_runtime is None or four_runtime is None:
        raise ValueError("旧运行器和FOUR运行器均不能为空")
    if normalized_mode == MODE_FOUR:
        return four_runtime
    return planner_runtime


def queue_successful_plan_result(
    serial_runtime,
    assembly_plan,
    mode,
    paper_orientation,
):
    """把一次完整成功规划交给通信运行器的单次结果队列。

    主要流程：先拒绝None、失败结果和空placements，再把同一列表整体交给协议层；
    协议层负责1～4片校验、定点编码和同一START上下文去重。关键参数中的mode和
    paper_orientation必须与本次求解使用的运行设置一致。返回True表示首次排队，
    False表示没有可发送规划或协议层判定本上下文已经发送过。
    """
    if assembly_plan is None or not bool(getattr(assembly_plan, "success", False)):
        return False
    placements = getattr(assembly_plan, "placements", None)
    if not placements:
        return False
    queue_result = getattr(serial_runtime, "queue_puzzle_result_once", None)
    if not callable(queue_result):
        raise ValueError("serial_runtime必须提供queue_puzzle_result_once方法")
    return bool(queue_result(mode, paper_orientation, placements))


def append_uart_status(status_text, link_text):
    """把最新UART链路状态作为唯一后缀加入屏幕状态文字。

    主要流程：按空格拆分旧状态，删除任何UART:开头的历史后缀，再追加协议运行器
    给出的UART:OK/OFFLINE/ERROR。返回新的ASCII字符串，不修改输入；这种替换方式
    避免每帧重复拼接导致状态栏越来越长。
    """
    normalized_link = str(link_text).strip()
    if normalized_link not in ("UART:OK", "UART:OFFLINE", "UART:ERROR"):
        raise ValueError("UART链路状态无效")
    status_tokens = [
        token
        for token in str(status_text).strip().split()
        if not token.startswith("UART:")
    ]
    status_tokens.append(normalized_link)
    return " ".join(status_tokens)


def select_calibration_serial_status(status_text, serial_runtime):
    """在SEND A4流程中选择需要显示的最新业务通信事件。

    主要流程：仅当当前状态仍属于A4/UART发送反馈时，才读取运行器最近事件；A4的
    ACK/NACK以及UART错误会替换旧的A4 QUEUED。普通调参页面状态原样返回，避免心跳
    或后台通信覆盖用户刚执行的ROI、MASK、RESULT等操作。返回值为新的状态字符串。
    """
    current_status = str(status_text).strip()
    if not current_status.startswith(("A4 ", "UART ")):
        return current_status
    event_text = str(getattr(serial_runtime, "last_event_text", "")).strip()
    if event_text.startswith("A4 ") or "ERROR" in event_text:
        return event_text
    return current_status


def select_result_serial_status(status_text, assembly_plan, serial_runtime):
    """把成功规划的本地协议编码错误转换为正常页可见状态。

    只有成功且包含碎片的规划才检查通信事件，防止上一轮残留的RESULT ERROR覆盖当前
    几何失败原因。运行器报告其他事件时保留原规划状态。返回值为新的状态字符串。
    """
    if assembly_plan is None or not bool(getattr(assembly_plan, "success", False)):
        return str(status_text)
    if not getattr(assembly_plan, "placements", None):
        return str(status_text)
    if str(getattr(serial_runtime, "last_event_text", "")) == "RESULT ERROR":
        return "RESULT ERROR"
    return str(status_text)


def select_serial_app_state(is_calibrating, capture_armed, planner_runtime, assembly_plan):
    """把视觉主循环状态映射为心跳载荷中的0至4状态码。

    优先级依次为CAL、结果就绪、正在求解、已开始采集和完全待机。CAL优先可让F4在
    用户调参时禁止机械动作；成功规划只有包含实际碎片位姿时才报告结果就绪。
    返回0待机、1调参、2采集、3求解或4结果就绪。
    """
    if bool(is_calibrating):
        return 1
    if (
        assembly_plan is not None
        and bool(getattr(assembly_plan, "success", False))
        and bool(getattr(assembly_plan, "placements", None))
    ):
        return 4
    if bool(getattr(planner_runtime, "is_solving", False)):
        return 3
    return 2 if bool(capture_armed) else 0


def select_display_pieces(live_pieces, planner_runtime):
    """为正常界面选择实时轮廓或已锁定轮廓。

    稳定门尚未满足时返回调用方的实时列表，以便观察识别质量；运行器一旦锁定就返回
    同一份只读快照，求解结束或超时也不切回抖动红点。运行器显式reset后自然恢复实时
    列表。该函数只选择引用，不复制或修改任何碎片数据。
    """
    if planner_runtime is None:
        raise ValueError("planner_runtime不能为空")
    if bool(getattr(planner_runtime, "snapshot_locked", False)):
        return planner_runtime.locked_pieces
    return live_pieces


def should_analyze_live_frame(
    is_calibrating,
    capture_armed,
    snapshot_locked,
    has_cached_detection,
):
    """判断当前相机帧是否还需要运行视觉分割。

    CAL始终需要实时MASK/RESULT，不受START限制；正常页未确认START时返回False，确保
    完全不执行碎片分割。START后若已经锁定且保留第3稳定帧结果则返回False，否则继续
    分析以获得首帧或丢失的缓存。返回严格布尔值，不修改任何运行状态。
    """
    if bool(is_calibrating):
        return True
    if not bool(capture_armed):
        return False
    return not (bool(snapshot_locked) and bool(has_cached_detection))


class QuadFrameAnalysis:
    """保存A版单帧分析的检测结果、有效四边形和实际检测ROI。"""

    def __init__(self, detection, active_quad, roi):
        """初始化纯数据结果，供设备主循环、PC回归和A/B对比工具共同使用。"""
        self.detection = detection
        self.active_quad = active_quad
        self.roi = tuple(int(value) for value in roi)


def build_runtime_active_quad(settings):
    """从已锁定纸张参数派生A版机械有效四边形。

    主要流程：没有 ``paper_quad`` 时返回 None 触发兼容矩形回退；存在四角时根据
    ``inset_mm`` 调用统一物理映射。返回值：None或4×2 float32四角。
    """
    paper_quad = settings.get("paper_quad")
    if paper_quad is None:
        return None
    work_region = (
        settings["work_x_mm"],
        settings["work_y_mm"],
        settings["work_width_mm"],
        settings["work_height_mm"],
    )
    return build_work_quad(
        paper_quad,
        work_region,
        paper_orientation=settings["paper_orientation"],
    )


def _quad_bounding_roi(active_quad, frame_size):
    """计算完整包含浮点四角且限制在相机画面内的整数外接ROI。"""
    frame_width, frame_height = (int(value) for value in frame_size)
    min_x = max(0, int(np.floor(np.min(active_quad[:, 0]))))
    min_y = max(0, int(np.floor(np.min(active_quad[:, 1]))))
    max_x = min(frame_width - 1, int(np.ceil(np.max(active_quad[:, 0]))))
    max_y = min(frame_height - 1, int(np.ceil(np.max(active_quad[:, 1]))))
    if max_x < min_x or max_y < min_y:
        raise ValueError("active_quad 无法生成有效外接ROI")
    return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1


def analyze_quad_frame(frame_bgr, runtime_settings, config=None):
    """执行A版四边形掩膜单帧识别并补充完整A4毫米中心。

    主要流程：从设置派生有效四边形和外接ROI，合并现场分割参数，调用视觉核心；
    锁定纸张时再把每片相机中心反算为A4毫米坐标。
    返回值：``QuadFrameAnalysis``；输入或单应性异常由调用方捕获并显示。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr 必须是有效的 numpy 图像")
    frame_height, frame_width = frame_bgr.shape[:2]
    active_quad = build_runtime_active_quad(runtime_settings)
    if active_quad is None:
        roi = tuple(int(value) for value in runtime_settings["roi"])
    else:
        roi = _quad_bounding_roi(active_quad, (frame_width, frame_height))

    base_config = DEFAULT_CONFIG if config is None else config
    detection_config = merge_runtime_config(base_config, runtime_settings)
    detection = detect_pieces(
        frame_bgr,
        roi,
        detection_config,
        active_quad=active_quad,
    )

    paper_quad = runtime_settings.get("paper_quad")
    if paper_quad is not None:
        paper_orientation = runtime_settings["paper_orientation"]
        for piece in detection.pieces:
            # A版保留相机坐标供画面叠加，同时批量增加毫米多边形供规划器使用。
            piece["center_mm"] = image_point_to_paper_mm(
                piece["center"],
                paper_quad,
                paper_orientation=paper_orientation,
            )
            vertices_mm = image_points_to_paper_mm(
                piece["vertices"],
                paper_quad,
                paper_orientation=paper_orientation,
            )
            piece["vertices_mm"] = vertices_mm.astype(float).tolist()
            piece["region"] = classify_piece_region(
                vertices_mm,
                runtime_settings["split_y_mm"],
            )
            piece["edge_features"] = sample_piece_edge_features(
                frame_bgr,
                piece["vertices"],
            )
    return QuadFrameAnalysis(detection, active_quad, roi)


def register_and_save_known_layout(
    pieces,
    template_path,
    work_region_mm,
    split_y_mm,
    max_nodes=12000,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """从下半区已正确拼好的四片同步登记KNOWN布局并原子保存模板。

    主要流程：调用固定规模登记器验收联合矩形，得到精确100×60mm模板和即时机械
    规划；只有成功时才写模板文件，失败保留旧模板。max_nodes仅保留旧接口兼容，
    当前路径不运行UNKNOWN搜索。返回值：``(plan, templates)``。
    """
    plan, templates = solve_and_register_known_layout(
        pieces,
        work_region_mm,
        split_y_mm,
        max_nodes=max_nodes,
        paper_orientation=paper_orientation,
    )
    if plan.success:
        save_templates(template_path, templates)
    return plan, templates


def perform_known_save_action(
    pieces,
    current_templates,
    template_path,
    work_region_mm,
    split_y_mm,
    planner_runtime,
    max_nodes=12000,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """执行一次KNOWN触摸保存，并保持内存、磁盘和规划缓存一致。

    主要流程：同步登记并保存下半区正确布局；失败时返回原模板和具体SAVE原因；成功时
    把新规划绑定到当前运行上下文，使本帧及后续帧立即绘制目标。文件系统异常会
    转换为现场可见状态，避免设备主循环退出。返回值为``(模板, plan, 状态文字)``。
    """
    try:
        plan, new_templates = register_and_save_known_layout(
            pieces,
            template_path,
            work_region_mm,
            split_y_mm,
            max_nodes=max_nodes,
            paper_orientation=paper_orientation,
        )
    except Exception as error:
        # 写文件失败时register_and_save_known_layout不会返回，内存继续使用旧模板。
        return current_templates, None, f"SAVE ERROR {type(error).__name__}"
    if not plan.success:
        return current_templates, plan, f"SAVE {plan.reason.upper()}"

    planner_runtime.cache_plan(
        MODE_KNOWN,
        plan,
        new_templates,
        work_region_mm,
        split_y_mm,
        pieces=pieces,
        paper_orientation=paper_orientation,
    )
    return new_templates, plan, "KNOWN SAVED PLAN OK"


def perform_known_save_request(
    capture_armed,
    pieces,
    current_templates,
    template_path,
    work_region_mm,
    split_y_mm,
    planner_runtime,
    max_nodes=12000,
    paper_orientation=PAPER_ORIENTATION_PORTRAIT,
):
    """处理带START确认门的KNOWN保存请求。

    主要流程：未启动时立即返回原模板、空规划和PRESS START，不重置运行器、不运行
    几何登记、更不会写模板文件；已启动时先释放旧求解状态，再调用现有同步SAVE入口。
    关键参数capture_armed表示本轮是否由START确认，其余参数原样传给
    perform_known_save_action。返回值仍为``(模板, plan, 状态文字)``。
    """
    if not bool(capture_armed):
        return current_templates, None, "PRESS START"

    _validate_capture_runtime(planner_runtime)
    planner_runtime.reset()
    return perform_known_save_action(
        pieces,
        current_templates,
        template_path,
        work_region_mm,
        split_y_mm,
        planner_runtime,
        max_nodes=max_nodes,
        paper_orientation=paper_orientation,
    )


class KnownSaveController:
    """保留旧跨帧KNOWN登记接口，供v1.3兼容测试和离线工具使用。

    v1.4设备run_app不再构造本控制器；实机SAVE统一调用perform_known_save_action。
    """

    def __init__(
        self,
        time_budget_ms=12.0,
        work_unit_limit=32,
        max_nodes=12000,
        texture_refinement_nodes=400,
    ):
        """初始化每帧预算和总搜索参数，尚未持有SAVE任务。

        关键参数：time_budget_ms与work_unit_limit限制单帧工作量，max_nodes限制无解
        总搜索量，texture_refinement_nodes限制首解后的牌面择优量。参数非法会抛出
        ValueError；结果通过start()、advance()和active属性访问。
        """
        self.time_budget_ms = float(time_budget_ms)
        self.work_unit_limit = int(work_unit_limit)
        self.max_nodes = int(max_nodes)
        self.texture_refinement_nodes = int(texture_refinement_nodes)
        if (
            self.time_budget_ms <= 0.0
            or self.work_unit_limit <= 0
            or self.max_nodes <= 0
            or self.texture_refinement_nodes < 0
        ):
            raise ValueError("KNOWN SAVE求解预算参数无效")
        self._job = None
        self._work_region_mm = None
        self._split_y_mm = None

    @property
    def active(self):
        """返回是否存在尚未结束的KNOWN登记任务。"""
        return self._job is not None and not self._job.done

    @property
    def search_nodes(self):
        """返回当前SAVE任务节点数；空闲时返回0。"""
        return 0 if self._job is None else int(self._job.search_nodes)

    def cancel(self):
        """取消未完成任务并清除绑定的机械上下文，不修改已有模板文件。"""
        if self._job is not None and not self._job.done:
            try:
                self._job.cancel()
            except Exception:
                # 取消用于释放资源，清理异常不能阻止模式切换或CAL界面响应。
                pass
        self._job = None
        self._work_region_mm = None
        self._split_y_mm = None

    def start(self, pieces, work_region_mm, split_y_mm):
        """从当前四片快照启动KNOWN登记，按钮回调中不执行完整搜索。

        已有任务执行时返回SAVE BUSY；输入数量或几何立即不合法时返回具体SAVE失败
        且不保持活动任务。成功启动返回含0节点的SOLVING状态，后续由advance推进。
        """
        if self.active:
            return "SAVE BUSY"
        try:
            self._job = KnownRegistrationJob(
                pieces,
                work_region_mm,
                split_y_mm,
                max_nodes=self.max_nodes,
                texture_refinement_nodes=self.texture_refinement_nodes,
            )
            self._work_region_mm = tuple(float(value) for value in work_region_mm)
            self._split_y_mm = float(split_y_mm)
        except Exception as error:
            self.cancel()
            return f"SAVE ERROR {type(error).__name__}"
        if self._job.done:
            status = f"SAVE {self._job.result.reason.upper()}"
            self.cancel()
            return status
        return "SAVE SOLVING N=0"

    def advance(self, current_templates, template_path, planner_runtime):
        """推进一个SAVE时间片，成功后才保存模板并写入即时规划缓存。

        返回值：``(templates, plan, status)``。未完成和写文件失败均保留调用方传入的
        current_templates；未完成plan为None，求解失败plan为结构化失败结果。
        """
        if not self.active:
            return current_templates, None, "SAVE IDLE"
        try:
            plan = self._job.advance(
                time_budget_ms=self.time_budget_ms,
                work_unit_limit=self.work_unit_limit,
            )
        except Exception as error:
            self.cancel()
            return current_templates, None, f"SAVE ERROR {type(error).__name__}"
        if plan is None:
            return current_templates, None, f"SAVE SOLVING N={self.search_nodes}"

        new_templates = self._job.templates
        work_region_mm = self._work_region_mm
        split_y_mm = self._split_y_mm
        if not plan.success:
            status = f"SAVE {plan.reason.upper()}"
            self.cancel()
            return current_templates, plan, status
        try:
            # 先校验并预装缓存，缓存失败时磁盘旧模板尚未被原子替换。
            planner_runtime.cache_plan(
                MODE_KNOWN,
                plan,
                new_templates,
                work_region_mm,
                split_y_mm,
            )
            # 写盘失败会在异常分支清掉预装缓存，使内存继续沿用调用方的旧模板。
            save_templates(template_path, new_templates)
        except Exception as error:
            try:
                planner_runtime.reset()
            except Exception:
                # 错误状态必须优先返回；清缓存异常不能再次穿透相机主循环。
                pass
            self.cancel()
            return current_templates, None, f"SAVE ERROR {type(error).__name__}"
        self.cancel()
        return new_templates, plan, "KNOWN SAVED PLAN OK"


def match_known_pieces_safely(pieces, templates, max_score):
    """执行KNOWN形状匹配并把损坏模板异常转换成可见状态。

    成功返回原模板和None；异常时把当前碎片全部恢复为UNKNOWN，返回空内存模板和
    `TEMPLATE ERROR 类型`。磁盘文件不在此处修改，用户可重新SAVE覆盖。
    """
    try:
        match_known_pieces(pieces, templates, float(max_score))
    except Exception as error:
        for piece in pieces:
            piece["id"] = "UNKNOWN"
            piece["match_score"] = float("inf")
        return [], f"TEMPLATE ERROR {type(error).__name__}"
    return templates, None


def select_known_template_status(templates, current_status, save_active=False):
    """在KNOWN无模板提示与SAVE状态之间选择优先级更高的文字。

    SAVE进行中、成功、失败或文件错误都必须保留，便于现场判断按钮结果；只有确实
    没有模板且当前没有SAVE相关状态时才返回NO TEMPLATE。已有模板时原样返回状态。
    """
    status = str(current_status)
    if templates or save_active or status.startswith("SAVE") or status.startswith("KNOWN SAVED"):
        return status
    if status.startswith("TEMPLATE ERROR"):
        return status
    return "NO TEMPLATE"


def select_planning_status(
    current_status,
    assembly_plan,
    stable_count,
    stable_frames,
    preserve_current=False,
    solving=False,
    search_nodes=0,
    edge_candidates=0,
    max_frontier_width=0,
    first_solution_node=None,
    snapshot_locked=False,
):
    """按动作优先级选择正常界面状态文字。

    SAVE或模式切换发生的当前帧可设置preserve_current，阻止后续规划更新覆盖关键
    操作结果；普通帧依次显示增量求解进度、稳定计数、成功目标数量或具体失败原因。
    solving为True时同时显示搜索节点N、边候选E、最大前沿F和首解标记S；S=1表示
    已有可在截止时返回的合法规划，S=0表示尚未找到。snapshot_locked为True时给求解
    和终态加`LOCKED`前缀，明确当前红点不会再随实时识别变化。返回字符串。
    """
    if preserve_current:
        return str(current_status)
    if str(current_status).startswith("SAVE ") and (
        assembly_plan is None or not assembly_plan.success
    ):
        # SAVE失败必须持续到用户切换模式、重新SAVE或产生成功规划，避免一闪而过。
        return str(current_status)
    locked_prefix = "LOCKED " if bool(snapshot_locked) else ""
    if solving:
        has_first_solution = 0 if first_solution_node is None else 1
        return (
            f"{locked_prefix}SOLVING N={int(search_nodes)} "
            f"E={int(edge_candidates)} "
            f"F={int(max_frontier_width)} "
            f"S={has_first_solution}"
        )
    if assembly_plan is None:
        if int(stable_count) > 0:
            return f"STABLE {int(stable_count)}/{int(stable_frames)}"
        return str(current_status)
    if assembly_plan.success:
        return f"{locked_prefix}PLAN OK N={len(assembly_plan.placements)}"
    return f"{locked_prefix}PLAN {assembly_plan.reason.upper()}"


def select_four_runtime_status(current_status, four_runtime, preserve_current=False):
    """按FOUR检测、稳定、求解和终态优先级生成正常页状态文字。

    失败计划优先且永久保留；求解显示锁定和候选数；检测阶段显示真实连通域数量、
    稳定帧以及是否执行受限拆分。关键参数four_runtime提供组合运行器公开属性。
    """
    if preserve_current:
        return str(current_status)
    plan = getattr(four_runtime, "plan", None)
    if plan is not None:
        if bool(getattr(plan, "success", False)):
            return f"LOCKED FOUR PLAN OK N={len(getattr(plan, 'placements', ())) }"
        return f"LOCKED FOUR {str(getattr(plan, 'reason', 'fail')).upper()}"
    if bool(getattr(four_runtime, "is_solving", False)):
        return f"LOCKED FOUR SOLVING N={int(getattr(four_runtime, 'search_nodes', 0))}"
    detection = getattr(four_runtime, "last_detection", None)
    if detection is None:
        return str(current_status)
    stable_count = int(getattr(four_runtime, "stable_count", 0))
    stable_frames = int(getattr(four_runtime, "stable_frames", 3))
    split_suffix = " SPLIT" if bool(getattr(detection, "split_applied", False)) else ""
    if stable_count > 0:
        return f"FOUR STABLE {stable_count}/{stable_frames}{split_suffix}"
    count = int(getattr(detection, "valid_contour_count", 0))
    return f"FOUR COUNT {count}/4{split_suffix}"


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
    serial_runtime=None,
):
    """处理五页调参动作、A4发送和两组互不覆盖的持久化。

    主要流程：顶部动作切换页面；底部固定槽先由会话映射为AUTO ROI、工作区、LOCK、
    SEND A4或ADV参数动作。LOCK只合并纸张字段，ADV SAVE只合并分割字段，SEND A4
    只排队协议帧且不写设置。frame_bgr仅在AUTO ROI单次点击时使用。
    返回值为 ``(运行参数, 状态文字)``。
    """
    if not interface_state.is_calibrating:
        raise ValueError("只有调参界面可以处理调参动作")

    session = interface_state.calibration_session
    action = str(action)
    if action in ("roi", "mask", "result", "adv"):
        if action == VIEW_RESULT and session.view == VIEW_RESULT:
            # 五个顶部槽已占满，重复点击RESULT进入量化页，不改变既有触摸布局。
            session.select_view(VIEW_MEASURE)
            return runtime_settings, "MEASURE"
        if action == VIEW_RESULT and session.view == VIEW_MEASURE:
            session.select_view(VIEW_RESULT)
            return runtime_settings, "RESULT"
        session.select_view(action)
        return runtime_settings, action.upper()
    logical_action = session.resolve_control_action(action)
    if logical_action == "auto_roi":
        if frame_bgr is None:
            raise ValueError("AUTO ROI需要当前相机帧")
        location = locate_black_paper(frame_bgr)
        log_auto_roi_diagnostics(location)
        session.apply_auto_roi(location)
        return runtime_settings, session.status_text
    if logical_action == "work_dec":
        session.adjust_work(-1)
        return runtime_settings, session.status_text
    if logical_action == "work_inc":
        session.adjust_work(1)
        return runtime_settings, session.status_text
    if logical_action == "work_value":
        return runtime_settings, f"WORK {session.cycle_work_item()}"
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
    if logical_action == "send_a4":
        if session.settings.get("paper_quad") is None:
            session.status_text = "ROI NOT SET"
            return runtime_settings, session.status_text
        if serial_runtime is None:
            session.status_text = "UART ERROR"
            return runtime_settings, session.status_text
        queue_paper_frame = getattr(serial_runtime, "queue_paper_frame", None)
        if not callable(queue_paper_frame):
            raise ValueError("serial_runtime必须提供queue_paper_frame方法")
        queue_paper_frame(session.settings["paper_orientation"])
        session.status_text = str(serial_runtime.last_event_text)
        return runtime_settings, session.status_text
    if logical_action == "disabled":
        # ADV第六槽是为保持六槽布局而保留的空白区，触摸后不改变任何状态。
        return runtime_settings, session.status_text
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


def evaluate_active_calibration_measurement(session, detection):
    """只在MEASURE页计算高成本量化指标并推进10帧稳定窗口。

    主要流程：其他调参页立即返回None，不执行连通域或距离变换；MEASURE页使用会话
    当前A4四角和专属跟踪器计算证据。关键参数session为CalibrationSession，
    detection为当前检测结果。返回CalibrationMeasurement或None。
    """
    if session.view != VIEW_MEASURE:
        return None
    return evaluate_calibration_measurement(
        detection,
        session.settings.get("paper_quad"),
        session.measurement_tracker,
    )


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


def build_paper_display_canvas(
    frame_bgr,
    paper_quad,
    paper_orientation="portrait",
    canvas_size=(640, 480),
):
    """把已标定A4四角展开到等比例屏幕内容区，并清除全部纸外相机画面。

    主要流程：校验原相机帧，根据蓝框在相机画面中的实际横竖外观计算最大等比例
    内容矩形，再把画面左上起顺时针四角映射到显示画布。`warpPerspective`会计算整张
    画布，因此最后只复制目标内容矩形，矩形外保持纯黑，防止纸外龙门架或地面泄露。

    关键参数：`canvas_size`为屏幕宽高；`paper_orientation`仍校验机械纸面设置，但不再
    决定显示旋转。侧装相机下正常页由蓝框外观保持与CAL一致，机械毫米坐标单独处理。
    返回值：`(canvas, display_transform, content_roi)`；矩阵用于把相机轮廓与纸面叠加
    统一映射到同一显示坐标，`content_roi`为`(x, y, width, height)`。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效图像")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr必须是三通道BGR图像")
    canvas_width, canvas_height = (int(value) for value in canvas_size)
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas_size宽高必须大于零")

    validate_paper_orientation(paper_orientation)
    source_quad = order_a4_quad(paper_quad)
    display_orientation = infer_paper_orientation(source_quad)
    paper_width_mm, paper_height_mm = paper_size_mm(display_orientation)
    # 显示方向与机械方向故意分离：侧装相机的PAPER V负责毫米轴和UART，正常页则
    # 保持CAL中看到的横纸外观。全部轮廓和规划叠加随后经过同一display_transform，
    # 因此只改变屏幕排版，不改变碎片的源/目标毫米坐标和旋转角。

    display_scale = min(
        canvas_width / float(paper_width_mm),
        canvas_height / float(paper_height_mm),
    )
    content_width = max(1, int(round(paper_width_mm * display_scale)))
    content_height = max(1, int(round(paper_height_mm * display_scale)))
    content_x = (canvas_width - content_width) // 2
    content_y = (canvas_height - content_height) // 2
    destination_quad = np.float32(
        (
            (content_x, content_y),
            (content_x + content_width - 1, content_y),
            (content_x + content_width - 1, content_y + content_height - 1),
            (content_x, content_y + content_height - 1),
        )
    )
    display_transform = cv2.getPerspectiveTransform(source_quad, destination_quad)
    warped = cv2.warpPerspective(
        frame_bgr,
        display_transform,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Homography在目标矩形外仍能反算到纸张延长平面；只复制内容矩形才能保证屏幕上的
    # 相机图像百分之百来自已标定四角内部，界面按钮和状态栏随后再独立叠加。
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    content_slice = (
        slice(content_y, content_y + content_height),
        slice(content_x, content_x + content_width),
    )
    canvas[content_slice] = warped[content_slice]
    return (
        canvas,
        display_transform,
        (content_x, content_y, content_width, content_height),
    )


def transform_points_for_display(points, display_transform):
    """使用纸面显示Homography转换任意数量二维相机点。

    `points`接受`N×2`或可展平为该形状的数组；返回独立`N×2 float32`数组。该函数
    只用于屏幕绘制，不得把返回值写回识别结果或机械毫米坐标。
    """
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.size == 0 or point_array.size % 2 != 0:
        raise ValueError("显示点必须包含至少一个有限二维坐标")
    point_array = point_array.reshape(-1, 2)
    if not np.all(np.isfinite(point_array)):
        raise ValueError("显示点必须包含有限二维坐标")
    matrix = np.asarray(display_transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("display_transform必须是有限3×3矩阵")
    return cv2.perspectiveTransform(point_array.reshape(1, -1, 2), matrix)[0]


def orient_paper_mask_for_display(mask, paper_quad, paper_orientation="portrait"):
    """把机械纸面掩膜旋转到正常页采用的相机蓝框外观。

    主要流程：分别取得相机左上起四角和机械毫米逻辑四角，查找二者相差的循环位移，
    再用相同的四分之一圈数旋转掩膜。这样top、正负90度侧装、倒置180度以及PAPER
    手动兜底都使用同一关系，不需要为某一种安装方式写死方向。

    关键参数：mask必须是二维uint8纸面掩膜；paper_quad为原相机蓝框；
    paper_orientation为生成该掩膜时的机械纸张方向。返回连续内存中的新数组，输入不变。
    无法确认四角循环关系时抛出ValueError，避免显示一个与相机画面错位的调试掩膜。
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("paper mask必须是二维uint8数组")
    camera_quad = order_a4_quad(paper_quad)
    coordinate_quad = orient_a4_quad_for_coordinates(
        paper_quad,
        paper_orientation,
    )
    quarter_turns = None
    for candidate_turns in range(4):
        if np.allclose(
            np.roll(camera_quad, candidate_turns, axis=0),
            coordinate_quad,
            atol=0.05,
        ):
            quarter_turns = candidate_turns
            break
    if quarter_turns is None:
        raise ValueError("机械纸面四角无法转换为相机显示方向")
    return np.ascontiguousarray(np.rot90(mask, k=quarter_turns))


def draw_overlay(
    frame_bgr,
    pieces,
    roi,
    buttons,
    mode,
    threshold,
    status_message,
    paper_quad=None,
    active_quad=None,
    work_region_mm=None,
    split_y_mm=None,
    assembly_plan=None,
    display_size=None,
    unknown_profile=UNKNOWN_PROFILE_WHITE,
    paper_orientation="portrait",
    capture_armed=False,
    four_debug_view="camera",
    four_debug_mask=None,
):
    """把相机几何映射后绘制到固定显示画布。

    主要流程：纸张未锁定时保留整帧缩放兼容视图；已有paper_quad时只显示等比例A4
    内容，并用同一显示Homography转换ROI、纸张、有效区和碎片几何。按钮始终位于显示
    坐标。关键参数pieces保持采集坐标，unknown_profile控制第三按钮文字，
    capture_armed控制START绿色活动态。返回新的BGR画布，原始帧、碎片字典和机械毫米
    数据均不修改。four_debug_mask为可选完整A4二维掩膜，只覆盖纸面内容区。
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr必须是有效图像")
    source_height, source_width = frame_bgr.shape[:2]
    target_width, target_height = (
        (source_width, source_height)
        if display_size is None
        else tuple(int(value) for value in display_size)
    )
    if min(target_width, target_height) <= 0:
        raise ValueError("显示宽高必须大于零")

    if paper_quad is None:
        output = cv2.resize(
            frame_bgr,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)

        def map_display_points(points):
            """未标定回退分支按横纵比例转换相机点。"""
            return np.asarray(points, dtype=np.float32).reshape(-1, 2) * np.float32(
                [scale_x, scale_y]
            )

        display_paper_quad = None
        display_active_quad = (
            None if active_quad is None else map_display_points(active_quad)
        )
    else:
        output, display_transform, content_roi = build_paper_display_canvas(
            frame_bgr,
            paper_quad,
            paper_orientation=paper_orientation,
            canvas_size=(target_width, target_height),
        )
        normalized_four_view = str(four_debug_view).strip().lower()
        if normalized_four_view not in FOUR_DEBUG_VIEWS:
            raise ValueError("FOUR调试视图无效")
        if four_debug_mask is not None:
            if (
                not isinstance(four_debug_mask, np.ndarray)
                or four_debug_mask.ndim != 2
                or four_debug_mask.dtype != np.uint8
            ):
                raise ValueError("four_debug_mask必须是二维uint8掩膜")
            content_x, content_y, content_width, content_height = content_roi
            display_mask = orient_paper_mask_for_display(
                four_debug_mask,
                paper_quad,
                paper_orientation,
            )
            resized_mask = cv2.resize(
                display_mask,
                (content_width, content_height),
                interpolation=cv2.INTER_NEAREST,
            )
            output[
                content_y : content_y + content_height,
                content_x : content_x + content_width,
            ] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

        def map_display_points(points):
            """纸面专用分支通过统一Homography转换相机点。"""
            return transform_points_for_display(points, display_transform)

        display_paper_quad = map_display_points(paper_quad)
        display_active_quad = (
            None if active_quad is None else map_display_points(active_quad)
        )

    frame_height, frame_width = output.shape[:2]
    scale = max(0.45, min(0.75, frame_width / 900.0))
    text_thickness = 1 if frame_width < 960 else 2

    roi_x, roi_y, roi_width, roi_height = roi
    if paper_quad is None:
        # 尚未锁定完整A4时保留稳定版矩形ROI，便于用户仍能进入CAL进行首次标定。
        display_roi_corners = map_display_points(
            (
                (roi_x, roi_y),
                (roi_x + roi_width - 1, roi_y + roi_height - 1),
            )
        )
        display_roi_x, display_roi_y = np.rint(display_roi_corners[0]).astype(int)
        display_roi_max_x, display_roi_max_y = np.rint(display_roi_corners[1]).astype(int)
        cv2.rectangle(
            output,
            (display_roi_x, display_roi_y),
            (display_roi_max_x, display_roi_max_y),
            (255, 200, 0),
            1,
        )
    else:
        cv2.polylines(
            output,
            [np.rint(display_paper_quad).astype(np.int32)],
            True,
            (255, 255, 0),
            2,
        )
    if active_quad is not None:
        cv2.polylines(
            output,
            [np.rint(display_active_quad).astype(np.int32)],
            True,
            (0, 255, 255),
            2,
        )
    if (
        paper_quad is not None
        and work_region_mm is not None
        and split_y_mm is not None
    ):
        output = draw_assembly_plan(
            output,
            assembly_plan,
            display_paper_quad,
            work_region_mm,
            split_y_mm,
            paper_orientation=paper_orientation,
        )

    for piece in pieces:
        # 完整轮廓使用绿色，不完整轮廓使用橙色，防止边界截断结果被误用于机械控制。
        contour_color = (0, 210, 0) if piece.get("complete", False) else (0, 140, 255)
        display_contour = np.rint(
            map_display_points(piece["contour"])
        ).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(output, [display_contour], -1, contour_color, 2)

        vertices = piece.get("vertices", [])
        display_vertices = (
            np.empty((0, 2), dtype=np.float32)
            if len(vertices) == 0
            else map_display_points(vertices)
        )
        for display_vertex in display_vertices:
            cv2.circle(
                output,
                tuple(int(round(value)) for value in display_vertex),
                4,
                (0, 0, 255),
                -1,
            )

        display_center = map_display_points([piece["center"]])[0]
        center_x, center_y = (int(round(value)) for value in display_center)
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
        # 模式按钮表示当前选择；START单独表示本轮是否已经由用户确认开始。
        active = name == mode or (name == "start" and bool(capture_armed))
        enabled = name != "save" or mode in (MODE_KNOWN, MODE_UNKNOWN, MODE_FOUR)
        button_label = name.upper()
        if name == "save":
            if mode == MODE_KNOWN:
                button_label = "SAVE"
            elif mode == MODE_FOUR:
                normalized_four_view = str(four_debug_view).strip().lower()
                if normalized_four_view not in FOUR_DEBUG_VIEW_LABELS:
                    raise ValueError("FOUR调试视图无效")
                button_label = FOUR_DEBUG_VIEW_LABELS[normalized_four_view]
            else:
                normalized_profile = str(unknown_profile).lower()
                if normalized_profile not in (
                    UNKNOWN_PROFILE_WHITE,
                    UNKNOWN_PROFILE_CARD,
                ):
                    raise ValueError("UNKNOWN子模式必须是white或card")
                button_label = normalized_profile.upper()
        if active:
            fill_color = (32, 150, 48)
        elif enabled:
            fill_color = (60, 60, 60)
        else:
            fill_color = (25, 25, 25)
        cv2.rectangle(output, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(output, (x1, y1), (x2, y2), (220, 220, 220), 1)
        # 六按钮布局中SUPPORT等长标签按按钮实际宽度缩小，避免覆盖相邻触摸控件。
        button_scale = scale
        text_width = cv2.getTextSize(
            button_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            button_scale,
            text_thickness,
        )[0][0]
        available_text_width = max(1, int(button.width) - 12)
        if text_width > available_text_width:
            button_scale = max(0.32, button_scale * available_text_width / text_width)
        cv2.putText(
            output,
            button_label,
            (x1 + 7, y1 + int(button.height * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            button_scale,
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


def create_camera_with_fallback(camera_module, image_format, config=None):
    """优先创建高分辨率相机，构造失败时回退到640×480显示分辨率。

    主要流程：读取capture_width/height尝试camera.Camera；仅高分辨率构造异常时
    再用display_width/height重试，第二次异常原样抛出。关键参数camera_module为
    Maix camera模块或测试替身。返回 ``(相机对象, 实际尺寸, 是否回退)``。
    """
    selected_config = DEFAULT_CONFIG if config is None else config
    capture_size = (
        int(selected_config.get("capture_width", selected_config["camera_width"])),
        int(selected_config.get("capture_height", selected_config["camera_height"])),
    )
    fallback_size = (
        int(selected_config.get("display_width", selected_config["camera_width"])),
        int(selected_config.get("display_height", selected_config["camera_height"])),
    )
    try:
        camera_object = camera_module.Camera(
            capture_size[0],
            capture_size[1],
            image_format,
        )
        return camera_object, capture_size, False
    except Exception:
        if fallback_size == capture_size:
            raise
        camera_object = camera_module.Camera(
            fallback_size[0],
            fallback_size[1],
            image_format,
        )
        return camera_object, fallback_size, True


def run_app():
    """初始化 MaixCAM2 并运行识别与调参双界面主循环。

    主要流程：加载持久现场参数、读取固定相机画面、按RUN或CAL选择参数，处理同一个
    CAL按钮的往返切换，再显示正常识别叠加或ROI/MASK/RESULT/MEASURE调参画面。
    关键参数：无；默认算法来自 config.DEFAULT_CONFIG，现场参数来自持久JSON。
    返回值：正常退出应用时返回 None。
    """
    from maix import app, camera, display, image, time, touchscreen

    display_width = int(DEFAULT_CONFIG["display_width"])
    display_height = int(DEFAULT_CONFIG["display_height"])
    display_size = (display_width, display_height)
    template_path = PERSISTENT_TEMPLATE_PATH

    disp = display.Display()
    cam, frame_size, resolution_fallback = create_camera_with_fallback(
        camera,
        image.Format.FMT_BGR888,
        DEFAULT_CONFIG,
    )
    camera_width, camera_height = frame_size
    cam.skip_frames(30)
    touch = touchscreen.TouchScreen()
    touch_tracker = TouchReleaseTracker()
    # 所有按钮与触摸始终使用显示坐标，不能随1280×960采集分辨率放大。
    run_buttons = build_button_layout(display_width, display_height)
    calibration_buttons = build_calibration_layout(display_width, display_height)
    interface_state = InterfaceState()
    # 固定机位仍要求连续3帧；中心容差放宽到3mm以容纳远距离轮廓量化抖动。
    planner_runtime = AssemblyRuntime(stable_frames=3, position_tolerance_mm=3.0)
    # FOUR拥有独立视觉稳定门和增量求解任务，不共享旧UNKNOWN的计数、快照或FALLBACK。
    four_runtime = FourPieceRuntime()
    # 构造函数不打开硬件；第一次poll才配置A21/A22并尝试打开UART4，因此串口故障
    # 不会阻止相机和显示初始化。
    serial_runtime = VisionSerialRuntime(create_maix_uart4)

    mode = MODE_UNKNOWN
    # 现场白色覆膜可能因反光产生伪纹理，因此启动默认明确选择几何首解即停的WHITE。
    unknown_profile = UNKNOWN_PROFILE_WHITE
    # FOUR默认显示相机实景；第三功能按钮只切换调试预览，不改变检测和求解数据。
    four_debug_view = "camera"
    # 正常页默认完全待机；只有START能把该状态改为True并开启视觉、稳定门与求解器。
    capture_armed = False
    status_message = "PRESS START"
    try:
        runtime_settings = load_runtime_settings(
            PERSISTENT_SETTINGS_PATH,
            DEFAULT_CONFIG,
            frame_size=frame_size,
        )
    except Exception as error:
        # 损坏设置不能阻止相机启动，回退默认整帧并把错误类型显示给现场人员。
        runtime_settings = build_default_runtime_settings(
            DEFAULT_CONFIG,
            frame_size=frame_size,
        )
        status_message = f"SETTINGS ERROR {type(error).__name__}"

    try:
        templates = load_templates(template_path)
    except Exception as error:
        # 模板损坏不能阻止未知模式工作，错误保留在状态栏供现场判断。
        templates = []
        status_message = f"TEMPLATE ERROR {type(error).__name__}"

    # detection同时充当第3稳定帧的轻量识别缓存。锁定后仍读取相机用于实时纸面画面和
    # 触摸，但不再重复运行视觉分割；模式重拍或CAL会把它清空并恢复下一帧分析。
    detection = None
    while not app.need_exit():
        camera_image = cam.read()
        frame_bgr = image.image2cv(camera_image, ensure_bgr=False, copy=False)

        # 每帧只推进一次非阻塞UART状态机；完整状态码让F4区分待机、调参、采集、
        # 求解和结果就绪，同时不改变相机主循环的非阻塞顺序。
        active_capture_runtime = select_capture_runtime(
            mode,
            planner_runtime,
            four_runtime,
        )
        serial_app_state = select_serial_app_state(
            interface_state.is_calibrating,
            capture_armed,
            active_capture_runtime,
            active_capture_runtime.plan,
        )
        serial_runtime.poll(app_state=serial_app_state)

        # CAL使用未保存工作副本，RUN只使用已经保存并生效的参数。
        if interface_state.is_calibrating:
            active_settings = interface_state.calibration_session.settings
        else:
            active_settings = runtime_settings
        roi = tuple(int(value) for value in active_settings["roi"])
        paper_quad = active_settings.get("paper_quad")
        active_quad = None
        work_region_mm = (
            runtime_settings["work_x_mm"],
            runtime_settings["work_y_mm"],
            runtime_settings["work_width_mm"],
            runtime_settings["work_height_mm"],
        )

        uses_four_pipeline = not interface_state.is_calibrating and mode == MODE_FOUR
        analyze_current_frame = False
        if not uses_four_pipeline:
            analyze_current_frame = should_analyze_live_frame(
                is_calibrating=interface_state.is_calibrating,
                capture_armed=capture_armed,
                snapshot_locked=planner_runtime.snapshot_locked,
                has_cached_detection=detection is not None,
            )
        if uses_four_pipeline:
            # FOUR每帧调用组合运行器：锁定前执行透视分割，锁定后内部只推进有限求解
            # 时间片。正常页不再调用旧detect_pieces或AssemblyRuntime.update。
            roi_x, roi_y, roi_width, roi_height = roi
            detection = DetectionResult(
                [],
                np.zeros((roi_height, roi_width), dtype=np.uint8),
                0.0,
                roi,
                valid_contour_count=0,
                white_ratio=0.0,
            )
            active_quad = build_runtime_active_quad(active_settings)
            if capture_armed:
                if paper_quad is None:
                    capture_armed = False
                    status_message = "FOUR NO PAPER"
                else:
                    try:
                        four_runtime.update(
                            frame_bgr,
                            paper_quad,
                            runtime_settings["paper_orientation"],
                            work_region_mm,
                            runtime_settings["split_y_mm"],
                        )
                    except Exception as error:
                        # FOUR异常终止本次START，保留错误文字等待用户手动重试，不能自动
                        # 重拍形成不可控循环。
                        capture_armed = False
                        status_message = f"FOUR ERROR {type(error).__name__}"
        elif analyze_current_frame:
            try:
                analysis = analyze_quad_frame(frame_bgr, active_settings)
                detection = analysis.detection
                roi = analysis.roi
                active_quad = analysis.active_quad
                if paper_quad is None and status_message in ("READY", "RUN MODE"):
                    status_message = "ROI NOT SET"
            except Exception as error:
                # 创建与ROI同尺寸的空诊断结果，使调参界面仍能显示并允许按CAL退出。
                roi_x, roi_y, roi_width, roi_height = roi
                empty_mask = cv2.cvtColor(
                    frame_bgr[
                        roi_y : roi_y + roi_height,
                        roi_x : roi_x + roi_width,
                    ],
                    cv2.COLOR_BGR2GRAY,
                )
                empty_mask[:] = 0
                detection = DetectionResult(
                    [],
                    empty_mask,
                    0.0,
                    roi,
                    valid_contour_count=0,
                    white_ratio=0.0,
                )
                status_message = f"VISION ERROR {type(error).__name__}"
        elif detection is not None:
            # detection保持第3稳定帧对象；显示用轮廓来自运行器的深复制快照。有效四边形
            # 只依赖持久设置，可以直接重建而无需再次进行阈值和轮廓计算。
            roi = tuple(int(value) for value in detection.roi)
            active_quad = build_runtime_active_quad(active_settings)
        else:
            # 完全待机不能调用analyze_quad_frame；这里只创建与ROI同尺寸的零掩膜数据
            # 供统一绘制代码读取。该分支不做灰度化、阈值、轮廓、模板或稳定计数。
            roi_x, roi_y, roi_width, roi_height = roi
            detection = DetectionResult(
                [],
                np.zeros((roi_height, roi_width), dtype=np.uint8),
                0.0,
                roi,
                valid_contour_count=0,
                white_ratio=0.0,
            )
            active_quad = build_runtime_active_quad(active_settings)

        four_detection = four_runtime.last_detection if uses_four_pipeline else None
        pieces = (
            four_detection.pieces
            if four_detection is not None
            else detection.pieces
        )
        threshold = 0.0 if uses_four_pipeline else detection.threshold
        # 触摸动作的结果在当前帧优先于规划状态，尤其不能把SAVE失败原因覆盖掉。
        preserve_planning_status = False
        capture_reset_requested = False
        active_buttons = (
            calibration_buttons if interface_state.is_calibrating else run_buttons
        )

        touch_x, touch_y, pressed = touch.read()
        clicked_display = touch_tracker.update(touch_x, touch_y, pressed)
        if clicked_display is not None:
            clicked_image = map_display_to_image(
                clicked_display,
                display_size,
                (disp.width(), disp.height()),
            )
            action = hit_test(clicked_image, active_buttons)
            if action == "cal":
                # CAL会改变纸张或分割参数，进入和退出都必须丢弃旧机械目标。
                _reset_capture_runtimes(planner_runtime, four_runtime=four_runtime)
                serial_runtime.reset_result_context()
                capture_armed = False
                entered_calibration = interface_state.toggle_calibration(
                    runtime_settings,
                    frame_size,
                )
                status_message = "CAL MODE" if entered_calibration else "PRESS START"
                capture_reset_requested = True
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
                        serial_runtime=serial_runtime,
                    )
                except Exception as error:
                    # 调参动作失败保持旧运行参数和当前会话，便于用户修正后重试。
                    status_message = f"CAL ERROR {type(error).__name__}"
            elif action in (MODE_KNOWN, MODE_UNKNOWN, MODE_FOUR):
                # 模式按钮只选择上下文并回到待机；真正采集由独立START确认。
                mode, capture_armed, status_message = select_capture_mode(
                    action,
                    planner_runtime,
                    serial_runtime=serial_runtime,
                    four_runtime=four_runtime,
                )
                preserve_planning_status = True
                capture_reset_requested = True
            elif action == "save" and mode == MODE_UNKNOWN:
                # UNKNOWN复用第三按钮切换材料类型；切换后旧稳定门和求解结果必须失效，
                # 并回到完全待机，防止WHITE/CARD切换后自动开始一轮用户未确认的采集。
                unknown_profile = toggle_unknown_profile(unknown_profile)
                mode, capture_armed, status_message = select_capture_mode(
                    MODE_UNKNOWN,
                    planner_runtime,
                    serial_runtime=serial_runtime,
                    four_runtime=four_runtime,
                )
                preserve_planning_status = True
                capture_reset_requested = True
            elif action == "save" and mode == MODE_FOUR:
                # FOUR第三功能按钮只切换纸面诊断视图，不能清快照或重复启动求解器。
                four_debug_view = toggle_four_debug_view(four_debug_view)
                status_message = f"FOUR VIEW {FOUR_DEBUG_VIEW_LABELS[four_debug_view]}"
                preserve_planning_status = True
            elif action == "start":
                # START是正常页唯一的分析入口；重复点击会清空旧快照并从下一帧重拍。
                capture_armed, status_message = start_capture(
                    mode,
                    planner_runtime,
                    unknown_profile=unknown_profile,
                    serial_runtime=serial_runtime,
                    four_runtime=four_runtime,
                )
                preserve_planning_status = True
                capture_reset_requested = True
            elif action == "save" and mode == MODE_KNOWN:
                # 未START时门禁只提示PRESS START；已启动才登记下半区100x60mm正确布局。
                templates, _save_plan, status_message = perform_known_save_request(
                    capture_armed,
                    pieces,
                    templates,
                    template_path,
                    work_region_mm,
                    runtime_settings["split_y_mm"],
                    planner_runtime,
                    paper_orientation=runtime_settings["paper_orientation"],
                )
                preserve_planning_status = True

        if interface_state.is_calibrating:
            quality = evaluate_calibration(detection)
            calibration_session = interface_state.calibration_session
            measurement = evaluate_active_calibration_measurement(
                calibration_session,
                detection,
            )
            # SEND A4后的ACK/NACK和写失败发生在后续poll帧，必须在绘制前动态同步。
            display_status = select_calibration_serial_status(
                status_message,
                serial_runtime,
            )
            if resolution_fallback and "RES LOW" not in display_status:
                display_status = f"{display_status} RES LOW".strip()
            display_status = append_uart_status(display_status, serial_runtime.link_text)
            display_frame = draw_calibration_frame(
                frame_bgr,
                detection,
                calibration_session,
                calibration_buttons,
                quality,
                display_status,
                measurement=measurement,
                display_size=display_size,
            )
        else:
            assembly_plan = None
            # 只有当前START会话才能执行模板匹配、ID编号、稳定计数和组合求解。
            if capture_armed and not capture_reset_requested:
                if mode == MODE_FOUR:
                    assembly_plan = four_runtime.plan
                    status_message = select_four_runtime_status(
                        status_message,
                        four_runtime,
                        preserve_current=preserve_planning_status,
                    )
                elif mode == MODE_KNOWN:
                    templates, template_error = match_known_pieces_safely(
                        pieces,
                        templates,
                        max_score=float(DEFAULT_CONFIG["known_match_threshold"]),
                    )
                    if template_error is not None:
                        planner_runtime.reset()
                        status_message = template_error
                    status_message = select_known_template_status(
                        templates,
                        status_message,
                    )
                elif mode == MODE_UNKNOWN:
                    assign_unknown_ids(
                        pieces,
                        row_tolerance_px=max(20, camera_height * 0.08),
                    )

                if mode in (MODE_KNOWN, MODE_UNKNOWN):
                    assembly_plan = planner_runtime.update(
                        mode,
                        pieces,
                        templates,
                        work_region_mm,
                        runtime_settings["split_y_mm"],
                        known_match_threshold=float(DEFAULT_CONFIG["known_match_threshold"]),
                        unknown_profile=unknown_profile,
                        paper_orientation=runtime_settings["paper_orientation"],
                    )
                    status_message = select_planning_status(
                        status_message,
                        assembly_plan,
                        planner_runtime.stable_count,
                        planner_runtime.stable_frames,
                        preserve_current=preserve_planning_status,
                        solving=planner_runtime.is_solving,
                        search_nodes=planner_runtime.search_nodes,
                        edge_candidates=planner_runtime.edge_candidates,
                        max_frontier_width=planner_runtime.max_frontier_width,
                        first_solution_node=planner_runtime.first_solution_node,
                        snapshot_locked=planner_runtime.snapshot_locked,
                    )
                queue_successful_plan_result(
                    serial_runtime,
                    assembly_plan,
                    protocol_mode_for_capture(mode),
                    runtime_settings["paper_orientation"],
                )
                status_message = select_result_serial_status(
                    status_message,
                    assembly_plan,
                    serial_runtime,
                )

            # 稳定门达到前画实时轮廓；达到后始终画运行器深复制的同一快照，使屏幕
            # 红点与求解输入完全一致。手动重拍点击帧不显示旧点，也不把旧检测结果计入
            # 新稳定门；下一相机帧才重新执行分析并显示新轮廓。
            display_pieces = (
                ()
                if (not capture_armed or capture_reset_requested)
                else select_display_pieces(pieces, active_capture_runtime)
            )

            # CAL退出后立即使用已保存ROI画框；当前帧结果最迟在下一帧同步该参数。
            display_roi = tuple(int(value) for value in runtime_settings["roi"])
            display_status = status_message
            if resolution_fallback and "RES LOW" not in display_status:
                display_status = f"{display_status} RES LOW".strip()
            display_status = append_uart_status(display_status, serial_runtime.link_text)
            four_debug_mask = None
            if (
                mode == MODE_FOUR
                and four_detection is not None
                and four_debug_view != "camera"
            ):
                four_debug_mask = getattr(
                    four_detection.masks,
                    four_debug_view,
                )
            display_frame = draw_overlay(
                frame_bgr,
                display_pieces,
                display_roi,
                run_buttons,
                mode,
                threshold,
                display_status,
                paper_quad=runtime_settings.get("paper_quad"),
                active_quad=build_runtime_active_quad(runtime_settings),
                work_region_mm=work_region_mm,
                split_y_mm=runtime_settings["split_y_mm"],
                assembly_plan=assembly_plan,
                display_size=display_size,
                unknown_profile=unknown_profile,
                paper_orientation=runtime_settings["paper_orientation"],
                capture_armed=capture_armed,
                four_debug_view=four_debug_view,
                four_debug_mask=four_debug_mask,
            )
        display_image = image.cv2image(display_frame, bgr=True, copy=False)
        disp.show(display_image, fit=image.Fit.FIT_CONTAIN)
        if capture_reset_requested:
            detection = None
        time.sleep_ms(1)
    # app.need_exit正常结束后释放UART文件描述符；运行器close可重复调用且内部吞掉
    # 驱动关闭异常，不能影响Maix应用退出。
    serial_runtime.close()


if __name__ == "__main__":
    run_app()
