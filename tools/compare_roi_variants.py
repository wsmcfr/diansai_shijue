"""对同一原始帧离线生成A四边形掩膜与B透视展开对比图。"""

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np


# 直接执行tools内脚本时把项目根目录加入模块搜索路径，保证两个变体包可导入。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maixcam2_app_A_quad.config import DEFAULT_CONFIG as CONFIG_A
from maixcam2_app_A_quad.main import analyze_quad_frame
from maixcam2_app_A_quad.paper_locator import locate_black_paper
from maixcam2_app_A_quad.settings_store import build_default_runtime_settings as defaults_a
from maixcam2_app_B_warp.config import DEFAULT_CONFIG as CONFIG_B
from maixcam2_app_B_warp.main import analyze_warped_frame, build_warp_display_canvas
from maixcam2_app_B_warp.settings_store import build_default_runtime_settings as defaults_b


def _draw_detection(frame_bgr, detection, paper_quad=None, active_quad=None):
    """在图像副本上绘制纸张边界、有效区和碎片轮廓。

    绿色表示完整碎片，橙色表示接触边界；paper_quad和active_quad均使用当前图像
    坐标，可选参数为None时只绘制碎片。
    """
    output = frame_bgr.copy()
    if paper_quad is not None:
        cv2.polylines(
            output,
            [np.rint(np.asarray(paper_quad)).astype(np.int32)],
            True,
            (255, 255, 0),
            2,
        )
    if active_quad is not None:
        cv2.polylines(
            output,
            [np.rint(np.asarray(active_quad)).astype(np.int32)],
            True,
            (0, 255, 255),
            2,
        )
    for piece in detection.pieces:
        color = (0, 210, 0) if piece.get("complete") is True else (0, 140, 255)
        cv2.drawContours(output, [piece["contour"]], -1, color, 2)
        center = tuple(int(round(value)) for value in piece["center"])
        cv2.drawMarker(output, center, (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    return output


def _put_variant_label(image, label, piece_count):
    """在对比图左上角绘制版本名和识别数量，返回同一图像对象。"""
    text = f"{label}  N={int(piece_count)}"
    cv2.rectangle(image, (0, 0), (220, 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _write_image(path, image):
    """写入单张对比图，OpenCV返回失败时抛出OSError而不是静默继续。"""
    if not cv2.imwrite(os.fspath(path), image):
        raise OSError(f"对比图写入失败: {path}")


def compare_frame(image_path, output_dir, paper_quad=None, inset_mm=0.0):
    """对同一原始帧运行A/B算法并写出三张对比图片。

    主要流程：读取原图；未给四角时单次自动定位；分别构造独立运行设置并调用A/B
    分析函数；输出A原图叠加、B展开画布和左右并排图。
    返回值：包含 ``quad/warp/side_by_side`` 三个Path的字典。
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")
    frame_bgr = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"输入图片解码失败: {image_path}")

    if paper_quad is None:
        location = locate_black_paper(frame_bgr)
        if not location.success:
            raise ValueError(f"AUTO ROI FAIL: {location.reason}")
        paper_quad = location.paper_quad
    paper_quad = np.asarray(paper_quad, dtype=np.float32)

    settings_a = defaults_a(CONFIG_A)
    settings_a["paper_quad"] = paper_quad.astype(float).tolist()
    settings_a["inset_mm"] = float(inset_mm)
    analysis_a = analyze_quad_frame(frame_bgr, settings_a)
    image_a = _draw_detection(
        frame_bgr,
        analysis_a.detection,
        paper_quad=paper_quad,
        active_quad=analysis_a.active_quad,
    )
    image_a = cv2.resize(image_a, (640, 480), interpolation=cv2.INTER_LINEAR)
    _put_variant_label(image_a, "A QUAD", len(analysis_a.detection.pieces))

    settings_b = defaults_b(CONFIG_B)
    settings_b["paper_quad"] = paper_quad.astype(float).tolist()
    settings_b["inset_mm"] = float(inset_mm)
    analysis_b = analyze_warped_frame(
        frame_bgr,
        paper_quad,
        inset_mm=float(inset_mm),
        runtime_settings=settings_b,
    )
    work_b = _draw_detection(analysis_b.work_frame, analysis_b.detection)
    image_b, _content_roi = build_warp_display_canvas(work_b)
    _put_variant_label(image_b, "B WARP", len(analysis_b.detection.pieces))

    side_by_side = np.hstack((image_a, image_b))
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    outputs = {
        "quad": output_directory / f"{stem}_A_quad.jpg",
        "warp": output_directory / f"{stem}_B_warp.jpg",
        "side_by_side": output_directory / f"{stem}_AB.jpg",
    }
    _write_image(outputs["quad"], image_a)
    _write_image(outputs["warp"], image_b)
    _write_image(outputs["side_by_side"], side_by_side)
    return outputs


def _build_argument_parser():
    """构造命令行参数解析器并返回，参数名与操作文档保持一致。"""
    parser = argparse.ArgumentParser(description="比较MaixCAM2自动ROI方案A与方案B")
    parser.add_argument("--image", required=True, help="无叠加原始相机图片路径")
    parser.add_argument("--output-dir", required=True, help="三张结果图输出目录")
    parser.add_argument(
        "--quad",
        nargs=8,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        help="可选四角；省略时执行一次AUTO ROI",
    )
    parser.add_argument("--inset-mm", type=float, default=0.0, help="四边整体内缩毫米")
    return parser


def main(argv=None):
    """解析命令行并执行对比；成功返回0，业务错误打印到stderr并返回2。"""
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    quad = None
    if arguments.quad is not None:
        quad = np.asarray(arguments.quad, dtype=np.float32).reshape(4, 2)
    try:
        outputs = compare_frame(
            arguments.image,
            arguments.output_dir,
            paper_quad=quad,
            inset_mm=arguments.inset_mm,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
