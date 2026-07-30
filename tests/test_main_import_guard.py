"""MaixCAM2 应用入口的PC导入隔离和叠加绘制测试。"""

import importlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np

from maixcam2_app.touch_ui import build_button_layout


# 设备入口在 MaixVision 平铺目录中运行时依赖的全部同级 Python 模块。
RUNTIME_MODULE_FILENAMES = {
    "main.py",
    "config.py",
    "puzzle_vision.py",
    "template_store.py",
    "touch_ui.py",
    "settings_store.py",
    "calibration_ui.py",
}


def test_pc_can_import_main_without_maix_runtime():
    """PC环境导入入口模块时不得立即导入或初始化Maix硬件。"""
    module = importlib.import_module("maixcam2_app.main")

    assert callable(module.run_app)


def test_draw_overlay_returns_new_image_without_modifying_source():
    """结果叠加必须绘制到副本，不能污染视觉核心使用的原始帧。"""
    from maixcam2_app.main import draw_overlay

    source = np.zeros((240, 320, 3), dtype=np.uint8)
    contour = np.asarray(
        [[[80, 80]], [[180, 80]], [[140, 170]]],
        dtype=np.int32,
    )
    pieces = [
        {
            "id": "U1",
            "contour": contour,
            "vertices": [(80, 80), (180, 80), (140, 170)],
            "center": (133.0, 110.0),
            "angle_deg": 15.0,
            "complete": True,
        }
    ]
    buttons = build_button_layout(320, 240)

    output = draw_overlay(
        source,
        pieces,
        roi=(0, 0, 320, 240),
        buttons=buttons,
        mode="unknown",
        threshold=120.0,
        status_message="READY",
    )

    assert output is not source
    assert np.count_nonzero(source) == 0
    assert np.count_nonzero(output) > 0


def test_status_text_counts_only_complete_pieces_as_actionable():
    """状态栏N必须表示完整碎片数，并单独显示边界轮廓数量。"""
    from maixcam2_app.main import format_status_text

    pieces = [{"complete": True}, {"complete": False}, {"complete": False}]

    text = format_status_text(
        mode="unknown",
        pieces=pieces,
        threshold=134.0,
        status_message="REPLAY",
    )

    assert text == "UNKNOWN N=1 EDGE=2 TH=134 REPLAY"


def test_direct_main_entry_resolves_package_from_other_directory(tmp_path):
    """设备入口从任意工作目录加载时必须能解析项目包，且测试不启动硬件。"""
    main_path = Path(__file__).resolve().parents[1] / "maixcam2_app" / "main.py"
    command = (
        "import runpy; "
        f"runpy.run_path({str(main_path)!r}, run_name='maix_entry_import_test')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_interface_state_cal_toggle_discards_unsaved_edits_and_preserves_mode():
    """同一个CAL动作必须往返切换，退出时丢弃未保存参数且不改变识别模式。"""
    from maixcam2_app.main import InterfaceState

    saved_settings = {
        "roi": [0, 0, 640, 480],
        "fixed_threshold": None,
        "min_area_ratio": 0.002,
        "open_kernel": 3,
        "close_kernel": 5,
    }
    mode = "known"
    interface = InterfaceState()

    assert interface.toggle_calibration(saved_settings, (640, 480)) is True
    assert interface.is_calibrating is True
    interface.calibration_session.select_item("LEFT")
    interface.calibration_session.adjust(1)
    assert interface.calibration_session.settings["roi"] != saved_settings["roi"]

    assert interface.toggle_calibration(saved_settings, (640, 480)) is False
    assert interface.is_calibrating is False
    assert interface.calibration_session is None
    assert saved_settings["roi"] == [0, 0, 640, 480]
    assert mode == "known"


def _make_calibration_detection(complete_count):
    """构造入口动作分发测试使用的最小检测结果。"""
    return SimpleNamespace(
        pieces=[
            {"complete": True, "vertex_count": 4}
            for _ in range(complete_count)
        ],
        valid_contour_count=complete_count,
        edge_contours=[],
        small_contours=[],
        large_contours=[],
        white_ratio=0.1,
        threshold=134.0,
    )


def test_good_calibration_action_saves_working_parameters(tmp_path):
    """GOOD状态点击保存必须持久化工作副本并返回新的运行参数。"""
    from maixcam2_app.main import InterfaceState, handle_calibration_action

    saved_settings = {
        "roi": [0, 0, 640, 480],
        "fixed_threshold": None,
        "min_area_ratio": 0.002,
        "open_kernel": 3,
        "close_kernel": 5,
    }
    interface = InterfaceState()
    interface.toggle_calibration(saved_settings, (640, 480))
    interface.calibration_session.select_item("LEFT")
    interface.calibration_session.adjust(1)
    settings_path = tmp_path / "vision_settings.json"

    updated, message = handle_calibration_action(
        "save_settings",
        interface,
        saved_settings,
        _make_calibration_detection(4),
        settings_path,
        (640, 480),
    )

    assert message == "SETTINGS SAVED"
    assert settings_path.exists()
    assert updated["roi"] == [5, 0, 635, 480]
    assert saved_settings["roi"] == [0, 0, 640, 480]


def test_bad_calibration_action_does_not_write_settings(tmp_path):
    """校准质量不是GOOD时，保存动作必须保留旧参数且不得创建文件。"""
    from maixcam2_app.main import InterfaceState, handle_calibration_action

    saved_settings = {
        "roi": [0, 0, 640, 480],
        "fixed_threshold": None,
        "min_area_ratio": 0.002,
        "open_kernel": 3,
        "close_kernel": 5,
    }
    interface = InterfaceState()
    interface.toggle_calibration(saved_settings, (640, 480))
    settings_path = tmp_path / "vision_settings.json"

    updated, message = handle_calibration_action(
        "save_settings",
        interface,
        saved_settings,
        _make_calibration_detection(3),
        settings_path,
        (640, 480),
    )

    assert message == "SAVE NEEDS GOOD"
    assert updated is saved_settings
    assert not settings_path.exists()


def test_value_action_only_toggles_threshold_when_th_is_selected(tmp_path):
    """数值按钮只有选中TH时才能切换自动与固定阈值。"""
    from maixcam2_app.main import InterfaceState, handle_calibration_action

    saved_settings = {
        "roi": [0, 0, 640, 480],
        "fixed_threshold": None,
        "min_area_ratio": 0.002,
        "open_kernel": 3,
        "close_kernel": 5,
    }
    interface = InterfaceState()
    interface.toggle_calibration(saved_settings, (640, 480))
    detection = _make_calibration_detection(4)

    handle_calibration_action(
        "value",
        interface,
        saved_settings,
        detection,
        tmp_path / "unused.json",
        (640, 480),
    )
    assert interface.calibration_session.settings["fixed_threshold"] is None

    interface.calibration_session.select_item("TH")
    handle_calibration_action(
        "value",
        interface,
        saved_settings,
        detection,
        tmp_path / "unused.json",
        (640, 480),
    )
    assert interface.calibration_session.settings["fixed_threshold"] == 134.0


def test_maixvision_flat_deployment_imports_sibling_modules(tmp_path):
    """MaixVision把文件平铺到临时目录后，入口必须能导入同级视觉模块。"""
    project_root = Path(__file__).resolve().parents[1]
    source_directory = project_root / "maixcam2_app"
    for filename in RUNTIME_MODULE_FILENAMES:
        shutil.copy2(source_directory / filename, tmp_path / filename)

    flat_main_path = tmp_path / "main.py"
    command = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        f"runpy.run_path({str(flat_main_path)!r}, run_name='flat_maix_import_test')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_app_manifest_includes_all_runtime_modules():
    """Maix 应用清单必须打包入口在设备上导入的全部同级模块。"""
    project_root = Path(__file__).resolve().parents[1]
    manifest_text = (project_root / "maixcam2_app" / "app.yaml").read_text(
        encoding="utf-8"
    )

    for filename in RUNTIME_MODULE_FILENAMES:
        assert f"  - {filename}\n" in manifest_text


def test_distribution_zip_includes_all_runtime_modules():
    """现成发布 ZIP 必须与当前 Maix 应用清单保持同一运行模块集合。"""
    project_root = Path(__file__).resolve().parents[1]
    archive_path = (
        project_root / "maixcam2_app" / "dist" / "maix-diansai_1-v1.0.0.zip"
    )

    with zipfile.ZipFile(archive_path) as archive:
        archived_names = set(archive.namelist())

    assert RUNTIME_MODULE_FILENAMES <= archived_names
