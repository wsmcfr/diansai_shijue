"""验证A版高分辨率采集、设置迁移和不会粘连相邻碎片的前景分割。"""

import json

import cv2
import numpy as np


def _two_piece_scene_with_hole():
    """构造1280x960黑底场景，两片之间保留4像素黑缝且左片包含纹理孔洞。

    4像素对应设计下最低2px/mm采样密度中的2mm物理间隙。返回值为BGR图像，
    供掩膜测试直接验证孔洞填充不会向相邻连通域扩张。
    """
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (180, 220), (500, 700), (245, 245, 245), -1)
    cv2.rectangle(frame, (505, 220), (825, 700), (245, 245, 245), -1)
    cv2.circle(frame, (340, 460), 45, (0, 0, 0), -1)
    return frame


def test_a_defaults_use_high_resolution_capture_and_low_resolution_display():
    """A版必须提高识别采集分辨率，同时保持640x480屏幕与触摸坐标。"""
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG

    assert (
        DEFAULT_CONFIG["capture_width"],
        DEFAULT_CONFIG["capture_height"],
    ) == (1280, 960)
    assert (
        DEFAULT_CONFIG["display_width"],
        DEFAULT_CONFIG["display_height"],
    ) == (640, 480)
    assert DEFAULT_CONFIG["gaussian_kernel"] == 3
    assert DEFAULT_CONFIG["close_kernel"] == 1


def test_default_mask_fills_internal_hole_without_bridging_four_pixel_gap():
    """默认分割必须填牌面内部黑孔，但不能连接相距4像素的两片。"""
    from maixcam2_app_A_quad.puzzle_vision import build_foreground_mask

    frame = _two_piece_scene_with_hole()
    mask, _ = build_foreground_mask(frame, (0, 0, 1280, 960))
    component_count, _ = cv2.connectedComponents(mask)

    # connectedComponents把背景计为一个标签，所以两片应得到三个标签。
    assert component_count == 3
    assert mask[460, 340] == 255
    assert np.count_nonzero(mask[220:701, 501:505]) == 0


def test_fill_internal_holes_keeps_open_notch_connected_to_background():
    """与外部相通的真实凹口不是纹理孔洞，填孔步骤不得把凹口补平。"""
    from maixcam2_app_A_quad.puzzle_vision import build_foreground_mask

    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    polygon = np.asarray(
        [[180, 220], [700, 220], [700, 700], [180, 700], [180, 540], [430, 460], [180, 380]],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [polygon], (245, 245, 245))

    mask, _ = build_foreground_mask(frame, (0, 0, 1280, 960))

    assert mask[460, 240] == 0
    assert mask[460, 500] == 255


def test_a_v3_pixel_coordinates_scale_to_actual_high_resolution(tmp_path):
    """旧V3的640x480像素坐标必须按实际采集尺寸迁移且关闭危险闭运算。"""
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
    from maixcam2_app_A_quad.settings_store import load_runtime_settings

    path = tmp_path / "vision_settings_v3.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "roi": [80, 30, 480, 420],
                "paper_quad": [[120, 40], [500, 50], [510, 440], [110, 430]],
                "inset_mm": 1.5,
                "work_x_mm": 1.5,
                "work_y_mm": 35.0,
                "work_width_mm": 207.0,
                "work_height_mm": 227.0,
                "split_y_mm": 148.5,
                "fixed_threshold": 108.0,
                "min_area_ratio": 0.002,
                "open_kernel": 3,
                "close_kernel": 5,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_runtime_settings(
        path,
        DEFAULT_CONFIG,
        frame_size=(1280, 960),
    )

    assert loaded["roi"] == [160, 60, 960, 840]
    assert loaded["paper_quad"] == [
        [240.0, 80.0],
        [1000.0, 100.0],
        [1020.0, 880.0],
        [220.0, 860.0],
    ]
    assert loaded["close_kernel"] == 1
    assert loaded["work_width_mm"] == 207.0
    assert loaded["fixed_threshold"] == 108.0


def test_a_v6_saves_normalized_coordinates_and_loads_at_new_resolution(tmp_path):
    """V6磁盘坐标必须与分辨率无关，换成640x480加载后应等比还原。"""
    from maixcam2_app_A_quad.config import DEFAULT_CONFIG
    from maixcam2_app_A_quad.settings_store import (
        build_default_runtime_settings,
        load_runtime_settings,
        save_runtime_settings,
    )

    path = tmp_path / "vision_settings_v6.json"
    settings = build_default_runtime_settings(DEFAULT_CONFIG, frame_size=(1280, 960))
    settings["roi"] = [160, 60, 960, 840]
    settings["paper_quad"] = [
        [240.0, 80.0],
        [1000.0, 100.0],
        [1020.0, 880.0],
        [220.0, 860.0],
    ]

    save_runtime_settings(path, settings, frame_size=(1280, 960))
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_runtime_settings(path, DEFAULT_CONFIG, frame_size=(640, 480))

    assert payload["version"] == 6
    assert payload["coordinate_space"] == "normalized"
    assert all(0.0 <= value <= 1.0 for value in payload["roi"])
    assert all(
        0.0 <= value <= 1.0
        for point in payload["paper_quad"]
        for value in point
    )
    assert loaded["roi"] == [80, 30, 480, 420]
    np.testing.assert_allclose(
        loaded["paper_quad"],
        [[120, 40], [500, 50], [510, 440], [110, 430]],
        atol=1e-6,
    )
