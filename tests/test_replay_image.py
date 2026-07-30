"""PC实拍图回放工具测试。"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys

from tests.synthetic_images import make_black_scene


def test_analyze_frame_reuses_core_and_returns_numbered_pieces():
    """回放工具必须返回未知模式编号结果和独立叠加图。"""
    from tools.replay_image import analyze_frame

    source = make_black_scene(
        [
            [(80, 90), (180, 80), (140, 180)],
            [(260, 90), (370, 90), (360, 190), (250, 180)],
        ],
        size=(480, 320),
    )
    original = source.copy()

    overlay, pieces, threshold = analyze_frame(
        source,
        roi=(0, 0, 480, 260),
        fixed_threshold=180,
    )

    assert [piece["id"] for piece in pieces] == ["U1", "U2"]
    assert threshold == 180.0
    assert overlay is not source
    assert np.array_equal(source, original)
    assert np.count_nonzero(overlay != source) > 0


def test_parse_args_accepts_roi_and_fixed_threshold():
    """命令行必须能解析输入、输出、四元ROI和固定阈值。"""
    from tools.replay_image import parse_args

    args = parse_args(
        [
            "input.jpg",
            "--output",
            "output.jpg",
            "--roi",
            "10",
            "20",
            "300",
            "200",
            "--threshold",
            "175",
        ]
    )

    assert args.input == "input.jpg"
    assert args.output == "output.jpg"
    assert args.roi == [10, 20, 300, 200]
    assert args.threshold == 175.0


def test_cli_main_writes_overlay_and_prints_piece_json(tmp_path, capsys):
    """完整回放命令必须写出叠加图，并打印含U1编号的几何JSON。"""
    from tools.replay_image import main

    source = make_black_scene(
        [[(80, 70), (240, 80), (160, 220)]],
        size=(320, 240),
    )
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "output.jpg"
    assert cv2.imwrite(str(input_path), source)

    exit_code = main(
        [
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            "180",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert '"id": "U1"' in captured.out


def test_direct_script_entrypoint_resolves_project_package(tmp_path):
    """从任意工作目录直接执行脚本时，也必须能导入同级项目包。"""
    source = make_black_scene(
        [[(60, 60), (250, 70), (150, 210)]],
        size=(320, 240),
    )
    input_path = tmp_path / "direct-input.jpg"
    output_path = tmp_path / "direct-output.jpg"
    assert cv2.imwrite(str(input_path), source)
    script_path = Path(__file__).resolve().parents[1] / "tools" / "replay_image.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            "180",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert '"id": "U1"' in completed.stdout


def test_replay_does_not_number_bright_regions_touching_roi_border():
    """回放结果只能返回完整碎片，接触ROI边界的亮区仅用于调试绘制。"""
    from tools.replay_image import analyze_frame

    source = make_black_scene(
        [
            [(80, 80), (230, 90), (160, 220)],
            [(0, 0), (319, 0), (319, 20), (0, 20)],
        ],
        size=(320, 240),
    )

    _, pieces, _ = analyze_frame(source, fixed_threshold=180)

    assert len(pieces) == 1
    assert pieces[0]["id"] == "U1"
    assert pieces[0]["complete"] is True
