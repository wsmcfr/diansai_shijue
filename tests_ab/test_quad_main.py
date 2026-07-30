"""验证A版入口构造有效四边形、调用识别并补充毫米中心。"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from maixcam2_app_A_quad.config import DEFAULT_CONFIG
from maixcam2_app_A_quad.settings_store import build_default_runtime_settings
from tests_ab.synthetic_paper import (
    DEFAULT_PAPER_QUAD,
    make_paper_scene,
    make_quad_scene_with_four_pieces,
)


def _make_locked_settings(inset_mm=2.0):
    """构造带合法完整A4四角的A版运行设置。"""
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["paper_quad"] = DEFAULT_PAPER_QUAD.astype(float).tolist()
    settings["inset_mm"] = float(inset_mm)
    return settings


def _unsupported_draw_calibration_keywords():
    """找出A版主循环传给调参绘制函数、但函数签名不支持的关键字参数。

    主要流程：解析实际部署入口的AST，定位唯一的 `draw_calibration_frame` 调用，
    再与实际导入函数的签名逐项比较。返回值是不受支持的关键字名称集合；若目标
    函数显式接受 `**kwargs`，则返回空集合。
    """
    from maixcam2_app_A_quad import calibration_ui, main

    source_path = Path(main.__file__).resolve()
    syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    draw_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "draw_calibration_frame"
    ]
    assert len(draw_calls) == 1, "A版入口必须且只能调用一次调参绘制函数"

    keyword_names = {
        keyword.arg for keyword in draw_calls[0].keywords if keyword.arg is not None
    }
    signature = inspect.signature(calibration_ui.draw_calibration_frame)
    accepts_extra_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_extra_keywords:
        return set()
    return keyword_names - set(signature.parameters)


def test_quad_calibration_draw_call_only_uses_supported_keywords():
    """验证A版进入CAL时不会向绘制函数传入其不支持的关键字参数。"""
    assert _unsupported_draw_calibration_keywords() == set()


def test_quad_runtime_builds_active_quad_from_locked_settings():
    """验证入口只从已锁定完整A4与毫米INSET派生机械有效区。"""
    from maixcam2_app_A_quad.main import build_runtime_active_quad

    active_quad = build_runtime_active_quad(_make_locked_settings(inset_mm=2.0))

    assert active_quad.shape == (4, 2)
    assert abs(cv2.contourArea(active_quad)) > 0


def test_quad_runtime_returns_none_without_locked_paper():
    """验证首次启动没有纸张四角时明确回退兼容矩形ROI。"""
    from maixcam2_app_A_quad.main import build_runtime_active_quad

    settings = build_default_runtime_settings(DEFAULT_CONFIG)

    assert build_runtime_active_quad(settings) is None


def test_quad_runtime_builds_landscape_active_quad_from_saved_orientation():
    """横放设置必须使用297×210mm纸面生成左右裁剪后的机械四边形。"""
    from maixcam2_app_A_quad import main, paper_locator

    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings.update(
        {
            "paper_orientation": "landscape",
            "paper_quad": [[20, 20], [614, 20], [614, 440], [20, 440]],
            "work_x_mm": 33.5,
            "work_y_mm": 0.0,
            "work_width_mm": 230.0,
            "work_height_mm": 210.0,
            "split_y_mm": 105.0,
        }
    )

    actual = main.build_runtime_active_quad(settings)
    expected = paper_locator.build_work_quad(
        settings["paper_quad"],
        (33.5, 0.0, 230.0, 210.0),
        paper_orientation="landscape",
    )

    np.testing.assert_allclose(actual, expected, atol=0.01)


def test_auto_roi_debug_log_reports_orientation_confidence_and_edges(capsys):
    """AUTO ROI调试日志必须显示H/V、置信度、阈值和四边像素长度。"""
    from maixcam2_app_A_quad import main
    from maixcam2_app_A_quad.paper_locator import PaperLocation

    location = PaperLocation(
        True,
        paper_quad=np.float32(((20, 30), (620, 45), (600, 440), (30, 430))),
        active_quad=np.float32(((40, 50), (600, 60), (580, 420), (50, 410))),
        paper_orientation="landscape",
        confidence=0.82,
        threshold=73.0,
    )

    main.log_auto_roi_diagnostics(location, debug_enabled=True)

    output = capsys.readouterr().out
    assert "[ROI] AUTO result=OK" in output
    assert "orientation=H" in output
    assert "confidence=82%" in output
    assert "threshold=73.0" in output
    assert "edges_px=[" in output
    assert output.count(",") >= 3


def test_auto_roi_debug_switch_disables_output(capsys):
    """共用调试开关关闭时AUTO ROI不得输出控制台日志。"""
    from maixcam2_app_A_quad import main
    from maixcam2_app_A_quad.paper_locator import PaperLocation

    main.log_auto_roi_diagnostics(
        PaperLocation.failed("paper_not_found", threshold=65.0),
        debug_enabled=False,
    )

    assert "[ROI]" not in capsys.readouterr().out


def test_quad_analysis_detects_four_pieces_and_adds_paper_mm_centers():
    """验证A版单帧数据流保留相机中心并额外输出完整A4毫米中心。"""
    from maixcam2_app_A_quad.main import analyze_quad_frame

    frame, paper_quad, _active_quad = make_quad_scene_with_four_pieces(inset_mm=0.0)
    settings = _make_locked_settings(inset_mm=0.0)
    settings["paper_quad"] = paper_quad.astype(float).tolist()

    analysis = analyze_quad_frame(frame, settings)

    assert len(analysis.detection.pieces) == 4
    assert analysis.active_quad.shape == (4, 2)
    assert all("center_mm" in piece for piece in analysis.detection.pieces)
    assert all(0.0 < piece["center_mm"][0] < 210.0 for piece in analysis.detection.pieces)
    assert all(33.5 < piece["center_mm"][1] < 263.5 for piece in analysis.detection.pieces)


def test_quad_analysis_maps_landscape_piece_to_297_by_210_mm_plane():
    """横放识别出的相机轮廓必须反算到297×210mm纸面，而不是交换长宽。"""
    from maixcam2_app_A_quad.main import analyze_quad_frame

    paper_quad = np.float32([[60, 90], [580, 80], [550, 390], [90, 400]])
    # 生产配置是侧装相机，合成轮廓必须复用同一纸面方向；否则测试图等价于顶置相机，
    # 与analyze_quad_frame读取的固定安装配置不一致。
    from maixcam2_app_A_quad.paper_locator import paper_points_to_image_px

    piece_mm = np.float32([[205, 45], [250, 45], [250, 90], [205, 90]])
    piece_px = paper_points_to_image_px(
        piece_mm,
        paper_quad,
        paper_orientation="landscape",
    )
    frame = make_paper_scene(paper_quad, white_pieces=(piece_px,))
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings.update(
        {
            "paper_orientation": "landscape",
            "paper_quad": paper_quad.astype(float).tolist(),
            "work_x_mm": 33.5,
            "work_y_mm": 0.0,
            "work_width_mm": 230.0,
            "work_height_mm": 210.0,
            "split_y_mm": 105.0,
        }
    )

    analysis = analyze_quad_frame(frame, settings)

    assert len(analysis.detection.pieces) == 1
    center_mm = analysis.detection.pieces[0]["center_mm"]
    assert center_mm == pytest.approx((227.5, 67.5), abs=1.5)
    assert analysis.detection.pieces[0]["region"] == "upper"


def test_quad_analysis_uses_compatibility_roi_when_paper_is_not_locked():
    """验证无四角回退仍可运行旧矩形识别且不会伪造毫米坐标。"""
    from maixcam2_app_A_quad.main import analyze_quad_frame

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["roi"] = [20, 30, 500, 400]

    analysis = analyze_quad_frame(frame, settings)

    assert analysis.active_quad is None
    assert analysis.roi == (20, 30, 500, 400)
    assert analysis.detection.roi == analysis.roi


def test_camera_open_prefers_high_resolution_and_reports_fallback():
    """相机辅助接口必须优先1280x960，失败时明确回退640x480。"""
    from maixcam2_app_A_quad.main import create_camera_with_fallback

    class FakeCameraModule:
        """记录构造参数并可选择让高分辨率构造失败。"""

        def __init__(self, fail_high=False):
            self.fail_high = bool(fail_high)
            self.calls = []

        def Camera(self, width, height, image_format):
            """模拟Maix camera.Camera并返回构造参数。"""
            self.calls.append((width, height, image_format))
            if self.fail_high and (width, height) == (1280, 960):
                raise RuntimeError("unsupported")
            return {"width": width, "height": height}

    normal_module = FakeCameraModule()
    normal_camera, normal_size, normal_fallback = create_camera_with_fallback(
        normal_module,
        "BGR",
        DEFAULT_CONFIG,
    )
    fallback_module = FakeCameraModule(fail_high=True)
    fallback_camera, fallback_size, fallback_used = create_camera_with_fallback(
        fallback_module,
        "BGR",
        DEFAULT_CONFIG,
    )

    assert normal_camera == {"width": 1280, "height": 960}
    assert normal_size == (1280, 960)
    assert normal_fallback is False
    assert fallback_camera == {"width": 640, "height": 480}
    assert fallback_size == (640, 480)
    assert fallback_used is True


def test_runtime_overlay_scales_high_resolution_frame_to_display_size():
    """运行叠加必须缩放几何并固定输出640x480，不能修改1280x960输入。"""
    from maixcam2_app_A_quad.main import draw_overlay
    from maixcam2_app_A_quad.touch_ui import build_button_layout

    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    piece = {
        "id": "U1",
        "complete": True,
        "contour": np.asarray([[[300, 300]], [[600, 300]], [[600, 600]], [[300, 600]]]),
        "vertices": [[300, 300], [600, 300], [600, 600], [300, 600]],
        "center": (450.0, 450.0),
        "angle_deg": 0.0,
    }

    output = draw_overlay(
        frame,
        [piece],
        (0, 0, 1280, 960),
        build_button_layout(640, 480),
        "unknown",
        108.0,
        "READY",
        display_size=(640, 480),
    )

    assert output.shape == (480, 640, 3)
    assert np.array_equal(frame, original)
    assert np.count_nonzero(output) > 0


@pytest.mark.parametrize(
    ("paper_quad", "paper_orientation", "expected_content_roi"),
    (
        (
            np.float32([[180, 30], [460, 50], [500, 450], [140, 430]]),
            "portrait",
            (150, 0, 339, 480),
        ),
        (
            np.float32([[120, 60], [520, 40], [550, 420], [90, 430]]),
            "landscape",
            (0, 13, 640, 453),
        ),
    ),
)
def test_paper_display_canvas_preserves_a4_aspect_and_hides_outside_scene(
    paper_quad,
    paper_orientation,
    expected_content_roi,
):
    """正常纸面视图只能显示四角内部，并按蓝框实际横竖外观居中填充黑边。"""
    from maixcam2_app_A_quad.main import build_paper_display_canvas

    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    paper_color = (25, 45, 65)
    cv2.fillConvexPoly(frame, np.rint(paper_quad).astype(np.int32), paper_color)

    canvas, display_transform, content_roi = build_paper_display_canvas(
        frame,
        paper_quad,
        paper_orientation=paper_orientation,
        canvas_size=(640, 480),
    )

    assert canvas.shape == (480, 640, 3)
    assert content_roi == expected_content_roi
    content_x, content_y, content_width, content_height = content_roi
    # 内容矩形以外必须完全为黑色，不能从纸张四角之外采到亮地面或龙门架。
    outside_mask = np.full(canvas.shape[:2], 255, dtype=np.uint8)
    outside_mask[
        content_y : content_y + content_height,
        content_x : content_x + content_width,
    ] = 0
    assert np.count_nonzero(canvas[outside_mask > 0]) == 0
    center_pixel = canvas[
        content_y + content_height // 2,
        content_x + content_width // 2,
    ]
    np.testing.assert_allclose(center_pixel, paper_color, atol=2)

    # 显示层应保持CAL中的相机四角外观；机械逻辑四角只用于毫米坐标和UART。
    from maixcam2_app_A_quad.paper_locator import order_a4_quad

    display_quad = order_a4_quad(paper_quad)
    mapped_quad = cv2.perspectiveTransform(
        display_quad.reshape(1, -1, 2),
        display_transform,
    )[0]
    expected_quad = np.float32(
        [
            [content_x, content_y],
            [content_x + content_width - 1, content_y],
            [content_x + content_width - 1, content_y + content_height - 1],
            [content_x, content_y + content_height - 1],
        ]
    )
    np.testing.assert_allclose(mapped_quad, expected_quad, atol=0.1)


def test_side_camera_normal_view_keeps_cal_landscape_appearance():
    """侧装相机的正常页必须保持CAL中的横纸外观，不能按机械PAPER V竖向展开。

    蓝框在相机画面中明显横向，但机械坐标仍为210×297mm的portrait。显示层应把
    画面左上、右上、右下、左下映到横向内容区；机械红线经同一显示矩阵后仍应保持
    画面中的竖向分隔，证明这里只旋转显示而没有修改毫米上下区。
    """
    from maixcam2_app_A_quad.main import (
        build_paper_display_canvas,
        transform_points_for_display,
    )
    from maixcam2_app_A_quad.paper_locator import (
        build_split_segment,
        order_a4_quad,
    )

    paper_quad = np.float32([[90, 76], [520, 60], [535, 350], [85, 365]])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.fillConvexPoly(frame, paper_quad.astype(np.int32), (25, 45, 65))

    _, display_transform, content_roi = build_paper_display_canvas(
        frame,
        paper_quad,
        paper_orientation="portrait",
        canvas_size=(640, 480),
    )

    assert content_roi == (0, 13, 640, 453)
    content_x, content_y, content_width, content_height = content_roi
    mapped_quad = transform_points_for_display(
        order_a4_quad(paper_quad),
        display_transform,
    )
    expected_quad = np.float32(
        (
            (content_x, content_y),
            (content_x + content_width - 1, content_y),
            (content_x + content_width - 1, content_y + content_height - 1),
            (content_x, content_y + content_height - 1),
        )
    )
    np.testing.assert_allclose(mapped_quad, expected_quad, atol=0.1)

    camera_split = build_split_segment(
        paper_quad,
        (0.0, 0.0, 210.0, 297.0),
        148.5,
        paper_orientation="portrait",
    )
    display_split = transform_points_for_display(camera_split, display_transform)
    split_delta = display_split[1] - display_split[0]
    assert abs(float(split_delta[1])) > abs(float(split_delta[0])) * 4.0


def test_paper_display_canvas_normalizes_cyclic_and_reversed_quad_order():
    """显示Homography必须规范相机四角，保存顺序变化不能旋转或镜像画面。"""
    from maixcam2_app_A_quad.main import build_paper_display_canvas

    paper_quad = np.float32([[120, 60], [520, 40], [550, 420], [90, 430]])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 四个不同角使用不同颜色，单纯黑纸无法发现循环移位或反向顺序造成的旋转镜像。
    for point, color in zip(
        paper_quad,
        ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)),
    ):
        cv2.circle(frame, tuple(np.rint(point).astype(int)), 24, color, -1)
    cv2.fillConvexPoly(
        frame,
        np.rint(paper_quad).astype(np.int32),
        (35, 45, 55),
    )
    # 在四角内部重新放置颜色标记，避免fillConvexPoly覆盖测试方向特征。
    center = np.mean(paper_quad, axis=0)
    for point, color in zip(
        paper_quad,
        ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)),
    ):
        inner = point * 0.86 + center * 0.14
        cv2.circle(frame, tuple(np.rint(inner).astype(int)), 14, color, -1)

    baseline, _, baseline_roi = build_paper_display_canvas(
        frame,
        paper_quad,
        paper_orientation="portrait",
    )
    variants = (
        np.roll(paper_quad, 1, axis=0),
        paper_quad[::-1].copy(),
        np.roll(paper_quad[::-1], 2, axis=0),
    )

    for variant in variants:
        actual, _, actual_roi = build_paper_display_canvas(
            frame,
            variant,
            paper_orientation="portrait",
        )
        assert actual_roi == baseline_roi
        assert np.array_equal(actual, baseline)


def test_runtime_overlay_uses_paper_only_canvas_after_roi_is_locked():
    """已有A4四角时正常叠加不能继续缩放整幅相机画面显示纸外内容。"""
    from maixcam2_app_A_quad.main import draw_overlay
    from maixcam2_app_A_quad.touch_ui import build_button_layout

    paper_quad = np.float32([[120, 60], [520, 40], [550, 420], [90, 430]])
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.fillConvexPoly(frame, np.rint(paper_quad).astype(np.int32), (20, 30, 40))

    output = draw_overlay(
        frame,
        [],
        (90, 40, 461, 391),
        build_button_layout(640, 480),
        "unknown",
        108.0,
        "READY",
        paper_quad=paper_quad,
        active_quad=paper_quad,
        work_region_mm=(0.0, 33.5, 210.0, 230.0),
        split_y_mm=148.5,
        display_size=(640, 480),
        paper_orientation="portrait",
    )

    # 蓝框实际为横向，正常页应铺满屏幕宽度；边缘仍必须来自黑纸而非亮色纸外背景。
    np.testing.assert_allclose(output[220, 20], (20, 30, 40), atol=2)
    np.testing.assert_allclose(output[220, 620], (20, 30, 40), atol=2)
    assert np.any(output[220, 320] != 0)


def test_unknown_profile_toggle_cycles_white_and_card():
    """UNKNOWN子模式必须默认支持WHITE，并在每次点击后与CARD互相切换。"""
    from maixcam2_app_A_quad.main import (
        UNKNOWN_PROFILE_CARD,
        UNKNOWN_PROFILE_WHITE,
        toggle_unknown_profile,
    )

    assert toggle_unknown_profile(UNKNOWN_PROFILE_WHITE) == UNKNOWN_PROFILE_CARD
    assert toggle_unknown_profile(UNKNOWN_PROFILE_CARD) == UNKNOWN_PROFILE_WHITE


def test_four_debug_view_cycles_camera_core_support_and_final():
    """FOUR第三功能按钮必须循环四种可现场判断阈值的预览。"""
    from maixcam2_app_A_quad.main import toggle_four_debug_view

    view = "camera"
    observed = []
    for _ in range(4):
        view = toggle_four_debug_view(view)
        observed.append(view)

    assert observed == ["strict", "support", "final", "camera"]


def test_select_capture_mode_resets_even_when_current_mode_is_clicked_again():
    """重复选择当前模式也必须释放旧快照，并保持完全待机等待START。"""
    from maixcam2_app_A_quad.main import (
        MODE_UNKNOWN,
        select_capture_mode,
    )

    class RecordingRuntime:
        """记录模式动作是否无条件调用运行器reset。"""

        def __init__(self):
            """初始化重置次数。"""
            self.reset_count = 0

        def reset(self):
            """模拟释放已有锁定快照。"""
            self.reset_count += 1

    runtime = RecordingRuntime()

    mode, capture_armed, status = select_capture_mode(
        MODE_UNKNOWN,
        runtime,
    )
    same_mode, same_capture_armed, same_status = select_capture_mode(
        MODE_UNKNOWN,
        runtime,
    )

    assert mode == MODE_UNKNOWN
    assert same_mode == MODE_UNKNOWN
    assert capture_armed is False
    assert same_capture_armed is False
    assert status == "PRESS START"
    assert same_status == status
    assert runtime.reset_count == 2


def test_select_display_pieces_prefers_locked_snapshot_over_live_jitter():
    """求解开始后正常叠加必须画锁定轮廓，不能继续显示不断跳动的实时顶点。"""
    from maixcam2_app_A_quad.main import select_display_pieces

    locked = ({"id": "U1", "center": (80.0, 80.0)},)

    class LockedRuntime:
        """提供正常界面选择绘制数据所需的最小运行器接口。"""

        snapshot_locked = True
        locked_pieces = locked

    live = [{"id": "U1", "center": (150.0, 100.0)}]

    assert select_display_pieces(live, LockedRuntime()) is locked


@pytest.mark.parametrize(
    (
        "is_calibrating",
        "capture_armed",
        "snapshot_locked",
        "has_cached_detection",
        "expected",
    ),
    (
        (False, False, False, False, False),
        (False, True, False, False, True),
        (False, True, True, True, False),
        (False, True, True, False, True),
        (True, False, True, True, True),
    ),
)
def test_should_analyze_live_frame_stops_detection_only_after_snapshot_lock(
    is_calibrating,
    capture_armed,
    snapshot_locked,
    has_cached_detection,
    expected,
):
    """正常页还要经过START门；CAL和START后的缺失缓存仍必须分析当前帧。"""
    from maixcam2_app_A_quad.main import should_analyze_live_frame

    actual = should_analyze_live_frame(
        is_calibrating=is_calibrating,
        capture_armed=capture_armed,
        snapshot_locked=snapshot_locked,
        has_cached_detection=has_cached_detection,
    )

    assert actual is expected


def test_run_app_wires_locked_display_pieces_and_start_action():
    """设备入口必须接入锁定轮廓、START动作和带确认状态的分析门。"""
    from maixcam2_app_A_quad import main

    syntax_tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    run_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_app"
    )
    calls = [node for node in ast.walk(run_function) if isinstance(node, ast.Call)]
    start_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "start_capture"
    ]
    display_selection_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "select_display_pieces"
    ]
    analysis_gate_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "should_analyze_live_frame"
    ]
    overlay_call = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "draw_overlay"
    )

    assert len(start_calls) == 1
    assert len(display_selection_calls) == 1
    assert len(analysis_gate_calls) == 1
    assert any(
        keyword.arg == "capture_armed"
        for keyword in analysis_gate_calls[0].keywords
    )
    assert isinstance(overlay_call.args[1], ast.Name)
    assert overlay_call.args[1].id == "display_pieces"


def test_runtime_overlay_uses_white_card_button_in_unknown_and_save_in_known(
    monkeypatch,
):
    """第三按钮在UNKNOWN显示当前子模式，切回KNOWN后必须恢复SAVE文字。"""
    from maixcam2_app_A_quad import main
    from maixcam2_app_A_quad.touch_ui import build_button_layout

    rendered_labels = []
    original_put_text = main.cv2.putText

    def record_put_text(image, text, *args, **kwargs):
        """记录OpenCV实际收到的文字，同时保留原绘制行为供输出图像正常生成。"""
        rendered_labels.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(main.cv2, "putText", record_put_text)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    buttons = build_button_layout(640, 480)

    main.draw_overlay(
        frame,
        [],
        (0, 0, 640, 480),
        buttons,
        main.MODE_UNKNOWN,
        108.0,
        "READY",
        unknown_profile=main.UNKNOWN_PROFILE_CARD,
    )
    assert "CARD" in rendered_labels
    assert "SAVE" not in rendered_labels

    rendered_labels.clear()
    main.draw_overlay(
        frame,
        [],
        (0, 0, 640, 480),
        buttons,
        main.MODE_KNOWN,
        108.0,
        "READY",
        unknown_profile=main.UNKNOWN_PROFILE_CARD,
    )
    assert "SAVE" in rendered_labels
    assert "CARD" not in rendered_labels


def test_four_overlay_labels_mask_button_and_renders_selected_paper_mask(monkeypatch):
    """FOUR调试视图必须显示当前掩膜名，并把掩膜限制在A4内容区域。"""
    from maixcam2_app_A_quad import main
    from maixcam2_app_A_quad.touch_ui import build_button_layout

    rendered_labels = []
    original_put_text = main.cv2.putText

    def record_put_text(image, text, *args, **kwargs):
        """记录按钮文字并保留真实OpenCV绘制。"""
        rendered_labels.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(main.cv2, "putText", record_put_text)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    paper_quad = np.float32(((110, 20), (530, 20), (530, 460), (110, 460)))
    debug_mask = np.full((891, 630), 255, dtype=np.uint8)

    output = main.draw_overlay(
        frame,
        [],
        (0, 0, 640, 480),
        build_button_layout(640, 480),
        main.MODE_FOUR,
        0.0,
        "FOUR COUNT 0/4",
        paper_quad=paper_quad,
        active_quad=paper_quad,
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
        display_size=(640, 480),
        paper_orientation="portrait",
        four_debug_view="strict",
        four_debug_mask=debug_mask,
    )

    assert "CORE" in rendered_labels
    assert np.all(output[120, 320] >= 240)
    assert np.all(output[220, 20] == 0)


def test_side_camera_four_debug_mask_rotates_to_landscape_display():
    """侧装时FOUR机械竖向掩膜必须旋转后覆盖横向正常页，不能直接横向拉伸。

    side_lower_right的逻辑左上角对应原始相机蓝框左下角，因此逻辑掩膜左上白块
    应显示在横向内容区左下；若未旋转，它会错误出现在左上。
    """
    from maixcam2_app_A_quad import main
    from maixcam2_app_A_quad.touch_ui import build_button_layout

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    paper_quad = np.float32(((90, 76), (520, 60), (535, 350), (85, 365)))
    debug_mask = np.zeros((891, 630), dtype=np.uint8)
    debug_mask[80:250, 80:250] = 255

    output = main.draw_overlay(
        frame,
        [],
        (0, 0, 640, 480),
        build_button_layout(640, 480),
        main.MODE_FOUR,
        0.0,
        "FOUR COUNT 0/4",
        paper_quad=paper_quad,
        active_quad=paper_quad,
        work_region_mm=(0.0, 0.0, 210.0, 297.0),
        split_y_mm=148.5,
        display_size=(640, 480),
        paper_orientation="portrait",
        four_debug_view="strict",
        four_debug_mask=debug_mask,
    )

    assert float(np.mean(output[330:410, 70:150])) > 220.0
    assert float(np.mean(output[90:170, 70:150])) < 40.0


def test_four_status_reports_count_stability_solving_and_cached_failure():
    """FOUR正常页必须显示当前真正阶段，失败结果不能被下一帧覆盖。"""
    from types import SimpleNamespace

    from maixcam2_app_A_quad.main import select_four_runtime_status

    count_runtime = SimpleNamespace(
        plan=None,
        is_solving=False,
        snapshot_locked=False,
        stable_count=0,
        stable_frames=3,
        search_nodes=0,
        last_detection=SimpleNamespace(valid_contour_count=3, split_applied=False),
    )
    stable_runtime = SimpleNamespace(
        **{
            **count_runtime.__dict__,
            "stable_count": 2,
            "last_detection": SimpleNamespace(valid_contour_count=4, split_applied=True),
        }
    )
    solving_runtime = SimpleNamespace(
        **{
            **stable_runtime.__dict__,
            "is_solving": True,
            "snapshot_locked": True,
            "search_nodes": 37,
        }
    )
    failed_plan = SimpleNamespace(success=False, placements=[], reason="no_rect")
    failed_runtime = SimpleNamespace(**{**solving_runtime.__dict__, "is_solving": False, "plan": failed_plan})

    assert select_four_runtime_status("FOUR CAPTURE", count_runtime) == "FOUR COUNT 3/4"
    assert select_four_runtime_status("FOUR CAPTURE", stable_runtime) == "FOUR STABLE 2/3 SPLIT"
    assert select_four_runtime_status("FOUR CAPTURE", solving_runtime) == "LOCKED FOUR SOLVING N=37"
    assert select_four_runtime_status("IGNORED", failed_runtime) == "LOCKED FOUR NO_RECT"


def test_run_app_wires_unknown_profile_to_runtime_overlay_and_toggle():
    """设备入口必须把子模式同时接入求解上下文、按钮绘制和SAVE触摸切换。"""
    from maixcam2_app_A_quad import main

    source = Path(main.__file__).read_text(encoding="utf-8")
    syntax_tree = ast.parse(source)
    run_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_app"
    )
    calls = [node for node in ast.walk(run_function) if isinstance(node, ast.Call)]
    toggle_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "toggle_unknown_profile"
    ]
    runtime_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "planner_runtime"
    ]
    overlay_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "draw_overlay"
    ]

    assert len(toggle_calls) == 1
    assert len(runtime_calls) == 1
    assert any(keyword.arg == "unknown_profile" for keyword in runtime_calls[0].keywords)
    assert len(overlay_calls) == 1
    assert any(keyword.arg == "unknown_profile" for keyword in overlay_calls[0].keywords)


def test_run_app_passes_saved_paper_orientation_to_runtime_overlay():
    """设备主循环必须把V6方向传给正常界面，确保红线和规划目标按同一纸面回绘。"""
    from maixcam2_app_A_quad import main

    syntax_tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    run_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_app"
    )
    overlay_call = next(
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "draw_overlay"
    )
    orientation_keyword = next(
        (item for item in overlay_call.keywords if item.arg == "paper_orientation"),
        None,
    )

    assert orientation_keyword is not None


def test_run_app_known_save_uses_direct_registration_without_job_start():
    """实机SAVE必须调用带START门的同步入口，不能启动旧登记任务控制器。"""
    from maixcam2_app_A_quad import main

    syntax_tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    run_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_app"
    )
    gated_calls = [
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "perform_known_save_request"
    ]
    old_job_starts = [
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "known_save_controller"
    ]

    assert len(gated_calls) == 1
    assert old_job_starts == []


def test_successful_plan_queues_all_placements_once_through_serial_runtime():
    """成功规划必须把1～4片全部位姿原样交给通信运行器的单次结果接口。"""
    from maixcam2_app_A_quad.main import queue_successful_plan_result

    placements = [
        SimpleNamespace(
            piece_id=f"U{index}",
            source_center_mm=(20.0 * index, 30.0),
            target_center_mm=(40.0 * index, 210.0),
            rotation_delta_deg=15.0 * index,
        )
        for index in range(1, 4)
    ]
    plan = SimpleNamespace(success=True, placements=placements)

    class RecordingSerialRuntime:
        """记录结果帧接口收到的模式、纸张方向和完整碎片列表。"""

        def __init__(self):
            """初始化为空调用记录。"""
            self.calls = []

        def queue_puzzle_result_once(self, mode, orientation, actual_placements):
            """保存调用参数并模拟本采集上下文首次排队成功。"""
            self.calls.append((mode, orientation, actual_placements))
            return True

    serial_runtime = RecordingSerialRuntime()

    queued = queue_successful_plan_result(
        serial_runtime,
        plan,
        "unknown",
        "portrait",
    )

    assert queued is True
    assert serial_runtime.calls == [("unknown", "portrait", placements)]


@pytest.mark.parametrize(
    "plan",
    (
        None,
        SimpleNamespace(success=False, placements=[]),
        SimpleNamespace(success=True, placements=[]),
    ),
)
def test_failed_or_incomplete_plan_is_never_sent(plan):
    """无规划、失败规划和空成功规划都不能形成F4机械动作。"""
    from maixcam2_app_A_quad.main import queue_successful_plan_result

    class RejectUnexpectedQueue:
        """任何结果排队调用都表示门禁失效。"""

        def queue_puzzle_result_once(self, *_args):
            """拒绝不应发生的通信调用。"""
            raise AssertionError("失败或空规划不得发送")

    assert queue_successful_plan_result(
        RejectUnexpectedQueue(),
        plan,
        "unknown",
        "portrait",
    ) is False


def test_result_encoding_error_becomes_visible_normal_page_status():
    """成功规划编码失败时状态栏必须显示RESULT ERROR，不能只剩UART链路状态。"""
    from maixcam2_app_A_quad.main import select_result_serial_status

    successful_plan = SimpleNamespace(success=True, placements=[object()])
    failed_plan = SimpleNamespace(success=False, placements=[])
    serial_runtime = SimpleNamespace(last_event_text="RESULT ERROR")

    assert (
        select_result_serial_status("PLAN OK", successful_plan, serial_runtime)
        == "RESULT ERROR"
    )
    assert select_result_serial_status("PLAN FAIL", failed_plan, serial_runtime) == "PLAN FAIL"

    serial_runtime.last_event_text = "RESULT QUEUED"
    assert (
        select_result_serial_status("PLAN OK", successful_plan, serial_runtime)
        == "PLAN OK"
    )


def test_uart_status_replaces_previous_suffix_without_repetition():
    """每帧状态只保留一个最新UART后缀，避免不断拼接导致文字溢出。"""
    from maixcam2_app_A_quad.main import append_uart_status

    assert append_uart_status("PLAN OK", "UART:OFFLINE") == "PLAN OK UART:OFFLINE"
    assert append_uart_status("PLAN OK UART:OFFLINE", "UART:OK") == "PLAN OK UART:OK"
    assert append_uart_status("UART:ERROR", "UART:OK") == "UART:OK"


def test_heartbeat_app_state_distinguishes_calibration_solving_and_ready_result():
    """心跳状态必须完整表达待机、CAL、采集、求解和结果就绪五种阶段。"""
    from maixcam2_app_A_quad.main import select_serial_app_state

    idle_runtime = SimpleNamespace(is_solving=False)
    solving_runtime = SimpleNamespace(is_solving=True)
    successful_plan = SimpleNamespace(success=True, placements=[object()])

    assert select_serial_app_state(False, False, idle_runtime, None) == 0
    assert select_serial_app_state(True, True, solving_runtime, successful_plan) == 1
    assert select_serial_app_state(False, True, idle_runtime, None) == 2
    assert select_serial_app_state(False, True, solving_runtime, None) == 3
    assert select_serial_app_state(False, True, idle_runtime, successful_plan) == 4


def test_main_imports_serial_protocol_in_package_and_flat_modes():
    """源码包和MaixVision平铺运行两条导入路径都必须包含UART4协议模块。"""
    from maixcam2_app_A_quad import main

    source = Path(main.__file__).read_text(encoding="utf-8")

    assert "from maixcam2_app_A_quad.serial_protocol import" in source
    assert "from serial_protocol import" in source


def test_run_app_wires_nonblocking_serial_poll_reset_submit_and_close():
    """设备入口必须逐帧推进串口，并接入CAL复位、成功规划发送和正常退出释放。"""
    from maixcam2_app_A_quad import main

    syntax_tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    run_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_app"
    )
    calls = [node for node in ast.walk(run_function) if isinstance(node, ast.Call)]

    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "VisionSerialRuntime"
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "serial_runtime"
        and node.func.attr == "poll"
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "serial_runtime"
        and node.func.attr == "reset_result_context"
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "queue_successful_plan_result"
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "serial_runtime"
        and node.func.attr == "close"
        for node in calls
    )


def test_run_app_wires_dedicated_four_runtime_and_protocol_mode_mapping():
    """设备入口必须创建FOUR运行器、逐帧推进并在发送前映射为UNKNOWN协议模式。"""
    from maixcam2_app_A_quad import main

    syntax_tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    run_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_app"
    )
    calls = [node for node in ast.walk(run_function) if isinstance(node, ast.Call)]

    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "FourPieceRuntime"
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "four_runtime"
        and node.func.attr == "update"
        for node in calls
    )
    assert any(
        isinstance(node.func, ast.Name)
        and node.func.id == "protocol_mode_for_capture"
        for node in calls
    )
