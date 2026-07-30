"""触摸松开事件、按钮命中和屏幕坐标映射测试。"""

import pytest


def test_touch_tracker_emits_one_click_on_release():
    """一次按下和松开只能产生一个点击事件，长按期间不得重复触发。"""
    from maixcam2_app.touch_ui import TouchReleaseTracker

    tracker = TouchReleaseTracker()

    assert tracker.update(30, 20, True) is None
    assert tracker.update(31, 21, True) is None
    assert tracker.update(31, 21, False) == (31, 21)
    assert tracker.update(31, 21, False) is None


def test_mode_buttons_are_hit_by_their_centers():
    """正常界面四个按钮的中心必须命中各自的稳定名称。"""
    from maixcam2_app.touch_ui import build_button_layout, hit_test

    buttons = build_button_layout(640, 480)

    assert hit_test(buttons["known"].center, buttons) == "known"
    assert hit_test(buttons["unknown"].center, buttons) == "unknown"
    assert hit_test(buttons["save"].center, buttons) == "save"
    assert hit_test(buttons["cal"].center, buttons) == "cal"


def test_calibration_layout_reuses_same_cal_button_rectangle():
    """正常与调参界面的CAL按钮必须保持同一位置和尺寸。"""
    from maixcam2_app.touch_ui import (
        build_button_layout,
        build_calibration_layout,
    )

    run_cal = build_button_layout(640, 480)["cal"]
    calibration_cal = build_calibration_layout(640, 480)["cal"]

    assert (
        run_cal.x,
        run_cal.y,
        run_cal.width,
        run_cal.height,
    ) == (
        calibration_cal.x,
        calibration_cal.y,
        calibration_cal.width,
        calibration_cal.height,
    )


def test_calibration_buttons_are_hit_by_their_centers():
    """调参页签和底部控制按钮必须都能通过中心触摸命中。"""
    from maixcam2_app.touch_ui import build_calibration_layout, hit_test

    buttons = build_calibration_layout(640, 480)
    expected_names = {
        "roi",
        "mask",
        "result",
        "cal",
        "item",
        "minus",
        "value",
        "plus",
        "step",
        "save_settings",
    }

    assert set(buttons) == expected_names
    for name in expected_names:
        assert hit_test(buttons[name].center, buttons) == name


def test_calibration_buttons_do_not_overlap_within_each_row():
    """640×480布局的顶部和底部按钮之间不得相互覆盖。"""
    from maixcam2_app.touch_ui import build_calibration_layout

    buttons = build_calibration_layout(640, 480)
    rows = [
        [buttons[name] for name in ("roi", "mask", "result", "cal")],
        [
            buttons[name]
            for name in ("item", "minus", "value", "plus", "step", "save_settings")
        ],
    ]

    for row in rows:
        ordered = sorted(row, key=lambda button: button.x)
        for first, second in zip(ordered, ordered[1:]):
            assert first.x + first.width < second.x


def test_display_center_maps_to_image_center_with_fit_contain():
    """FIT_CONTAIN产生横向黑边时，屏幕中心仍必须映射到图像中心。"""
    from maixcam2_app.touch_ui import map_display_to_image

    mapped = map_display_to_image(
        point=(276, 184),
        image_size=(640, 480),
        display_size=(552, 368),
    )

    assert mapped[0] == pytest.approx(320.0, abs=1.0)
    assert mapped[1] == pytest.approx(240.0, abs=1.0)


def test_touch_in_fit_contain_black_bar_is_ignored():
    """触摸屏幕黑边不得映射为相机图像内的按钮点击。"""
    from maixcam2_app.touch_ui import map_display_to_image

    assert (
        map_display_to_image(
            point=(5, 184),
            image_size=(640, 480),
            display_size=(552, 368),
        )
        is None
    )
