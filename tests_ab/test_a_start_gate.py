"""验证A版START确认门、完全待机和五按钮正常页布局。"""

import numpy as np
import pytest


class RecordingRuntime:
    """记录采集状态动作是否清除了旧快照、任务和规划结果。"""

    def __init__(self):
        """初始化重置次数，供模式选择和START重拍测试断言。"""
        self.reset_count = 0

    def reset(self):
        """模拟AssemblyRuntime.reset并累计调用次数。"""
        self.reset_count += 1


class RecordingSerialContext:
    """记录通信结果上下文是否随模式和START动作同步清除。"""

    def __init__(self):
        """初始化通信上下文重置次数。"""
        self.reset_count = 0

    def reset_result_context(self):
        """模拟取消旧结果帧并允许新START发送一次。"""
        self.reset_count += 1


def test_run_layout_contains_non_overlapping_start_button():
    """640x480正常页必须提供独立FOUR和START按钮且所有触摸区互不重叠。"""
    from maixcam2_app_A_quad.touch_ui import build_button_layout, hit_test

    buttons = build_button_layout(640, 480)

    assert tuple(buttons) == (
        "known",
        "unknown",
        "four",
        "save",
        "start",
        "cal",
    )
    assert hit_test(buttons["four"].center, buttons) == "four"
    assert hit_test(buttons["start"].center, buttons) == "start"
    ordered = [buttons[name] for name in buttons]
    for left, right in zip(ordered, ordered[1:]):
        assert left.x + left.width < right.x


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
        (False, False, True, True, False),
        (False, True, False, False, True),
        (False, True, True, True, False),
        (False, True, True, False, True),
        (True, False, True, True, True),
    ),
)
def test_live_analysis_requires_start_outside_calibration(
    is_calibrating,
    capture_armed,
    snapshot_locked,
    has_cached_detection,
    expected,
):
    """正常页只有START后才分析，CAL始终实时分析，锁定缓存后停止重复分割。"""
    from maixcam2_app_A_quad.main import should_analyze_live_frame

    actual = should_analyze_live_frame(
        is_calibrating=is_calibrating,
        capture_armed=capture_armed,
        snapshot_locked=snapshot_locked,
        has_cached_detection=has_cached_detection,
    )

    assert actual is expected


def test_selecting_mode_disarms_capture_and_resets_runtime():
    """点击模式只能完成选择并进入待机，不能沿用上一轮快照或立即开始分析。"""
    from maixcam2_app_A_quad.main import MODE_KNOWN, select_capture_mode

    runtime = RecordingRuntime()

    mode, capture_armed, status = select_capture_mode(MODE_KNOWN, runtime)

    assert mode == MODE_KNOWN
    assert capture_armed is False
    assert status == "PRESS START"
    assert runtime.reset_count == 1


def test_selecting_mode_also_resets_serial_result_context():
    """模式选择必须取消旧PUZZLE_RESULT，避免F4执行上一轮机械目标。"""
    from maixcam2_app_A_quad.main import MODE_KNOWN, select_capture_mode

    planner_runtime = RecordingRuntime()
    serial_runtime = RecordingSerialContext()

    select_capture_mode(MODE_KNOWN, planner_runtime, serial_runtime=serial_runtime)

    assert planner_runtime.reset_count == 1
    assert serial_runtime.reset_count == 1


def test_start_capture_arms_and_restarts_current_selection():
    """每次点击START都必须清空旧状态并按当前UNKNOWN材料重新采集。"""
    from maixcam2_app_A_quad.main import (
        MODE_UNKNOWN,
        UNKNOWN_PROFILE_CARD,
        start_capture,
    )

    runtime = RecordingRuntime()

    first_armed, first_status = start_capture(
        MODE_UNKNOWN,
        runtime,
        unknown_profile=UNKNOWN_PROFILE_CARD,
    )
    second_armed, second_status = start_capture(
        MODE_UNKNOWN,
        runtime,
        unknown_profile=UNKNOWN_PROFILE_CARD,
    )

    assert first_armed is True
    assert second_armed is True
    assert first_status == "UNKNOWN CARD CAPTURE"
    assert second_status == first_status
    assert runtime.reset_count == 2


def test_start_capture_also_opens_new_serial_result_context():
    """每次START重拍必须让本轮成功规划能够重新排队一次结果帧。"""
    from maixcam2_app_A_quad.main import MODE_UNKNOWN, start_capture

    planner_runtime = RecordingRuntime()
    serial_runtime = RecordingSerialContext()

    start_capture(MODE_UNKNOWN, planner_runtime, serial_runtime=serial_runtime)
    start_capture(MODE_UNKNOWN, planner_runtime, serial_runtime=serial_runtime)

    assert planner_runtime.reset_count == 2
    assert serial_runtime.reset_count == 2


def test_four_mode_starts_dedicated_capture_and_uses_unknown_wire_mode():
    """FOUR必须拥有独立采集状态，但串口协议继续使用UNKNOWN模式编码。"""
    from maixcam2_app_A_quad.main import (
        MODE_FOUR,
        MODE_UNKNOWN,
        protocol_mode_for_capture,
        select_capture_mode,
        start_capture,
    )

    runtime = RecordingRuntime()
    mode, capture_armed, status = select_capture_mode(MODE_FOUR, runtime)
    started, start_status = start_capture(MODE_FOUR, runtime)

    assert (mode, capture_armed, status) == (MODE_FOUR, False, "PRESS START")
    assert (started, start_status) == (True, "FOUR CAPTURE")
    assert protocol_mode_for_capture(MODE_FOUR) == MODE_UNKNOWN
    assert runtime.reset_count == 2


@pytest.mark.parametrize("mode", ("known", "unknown"))
def test_protocol_mode_mapping_preserves_existing_modes(mode):
    """新增FOUR内部模式不得改变KNOWN和UNKNOWN原有协议编码。"""
    from maixcam2_app_A_quad.main import protocol_mode_for_capture

    assert protocol_mode_for_capture(mode) == mode


def test_unarmed_known_save_returns_press_start_without_entering_save(monkeypatch):
    """未START时KNOWN SAVE必须在调用登记与写模板入口前直接拒绝。"""
    from maixcam2_app_A_quad import main

    def fail_if_called(*args, **kwargs):
        """若门禁错误地进入真实SAVE路径，立即让测试失败。"""
        raise AssertionError("未START不得调用perform_known_save_action")

    monkeypatch.setattr(main, "perform_known_save_action", fail_if_called)
    templates = [{"id": "K1"}]

    runtime = RecordingRuntime()
    actual_templates, plan, status = main.perform_known_save_request(
        False,
        (),
        templates,
        "unused.json",
        (0.0, 33.5, 210.0, 230.0),
        148.5,
        runtime,
    )

    assert actual_templates is templates
    assert plan is None
    assert status == "PRESS START"
    assert runtime.reset_count == 0


def test_armed_known_save_resets_snapshot_and_delegates(monkeypatch):
    """已START的KNOWN SAVE应先清旧求解状态，再把当前快照交给原登记入口。"""
    from maixcam2_app_A_quad import main

    expected = ([{"id": "K1"}], object(), "KNOWN SAVED PLAN OK")

    def record_save(*args, **kwargs):
        """模拟成功登记并返回可辨认的结果三元组。"""
        return expected

    monkeypatch.setattr(main, "perform_known_save_action", record_save)
    runtime = RecordingRuntime()

    actual = main.perform_known_save_request(
        True,
        (),
        [],
        "unused.json",
        (0.0, 33.5, 210.0, 230.0),
        148.5,
        runtime,
    )

    assert actual is expected
    assert runtime.reset_count == 1


def test_start_button_fill_reflects_armed_state():
    """START待机时必须为灰色，活动采集时必须为绿色，模式选择高亮不受影响。"""
    from maixcam2_app_A_quad import main
    from maixcam2_app_A_quad.touch_ui import build_button_layout

    buttons = build_button_layout(640, 480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    common_arguments = (
        frame,
        (),
        (0, 0, 640, 480),
        buttons,
        main.MODE_UNKNOWN,
        0.0,
        "PRESS START",
    )

    standby = main.draw_overlay(*common_arguments, capture_armed=False)
    armed = main.draw_overlay(*common_arguments, capture_armed=True)
    start = buttons["start"]
    sample_x = int(start.x + 2)
    sample_y = int(start.y + 2)

    assert tuple(standby[sample_y, sample_x]) == (60, 60, 60)
    assert tuple(armed[sample_y, sample_x]) == (32, 150, 48)
