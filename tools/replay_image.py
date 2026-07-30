"""在PC上使用与MaixCAM2相同的视觉核心回放单张实拍图。"""

import argparse
import json
import os
import sys

import cv2

# 直接执行脚本时，Python只把tools目录加入搜索路径；补入项目根目录以加载正式算法包。
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from maixcam2_app.main import MODE_UNKNOWN, draw_overlay
from maixcam2_app.puzzle_vision import (
    assign_unknown_ids,
    detect_pieces,
    select_actionable_pieces,
)
from maixcam2_app.touch_ui import build_button_layout


def analyze_frame(frame_bgr, roi=None, fixed_threshold=None):
    """识别一帧实拍图并生成未知模式叠加结果。

    主要流程：确定ROI、调用正式视觉核心、稳定编号未知碎片，再复用设备显示层绘制叠加。
    关键参数：frame_bgr 为BGR图像，roi可省略为整帧，fixed_threshold可锁定灰度阈值。
    返回值：``(叠加图, 碎片列表, 实际阈值)``，输入图像不会被修改。
    """
    if frame_bgr is None:
        raise ValueError("输入图像不能为空")
    frame_height, frame_width = frame_bgr.shape[:2]
    if roi is None:
        roi = (0, 0, frame_width, frame_height)

    config = {}
    if fixed_threshold is not None:
        config["fixed_threshold"] = float(fixed_threshold)
    detection = detect_pieces(frame_bgr, roi, config=config)
    detected_pieces = detection.pieces
    for piece in detected_pieces:
        if not piece["complete"]:
            piece["id"] = "EDGE"
    actionable_pieces = select_actionable_pieces(detected_pieces)
    assign_unknown_ids(
        actionable_pieces,
        row_tolerance_px=max(20, frame_height * 0.08),
    )
    buttons = build_button_layout(frame_width, frame_height)
    overlay = draw_overlay(
        frame_bgr,
        detected_pieces,
        roi,
        buttons,
        MODE_UNKNOWN,
        detection.threshold,
        "REPLAY",
    )
    return overlay, actionable_pieces, detection.threshold


def parse_args(argv=None):
    """解析输入图片、输出图片、可选ROI和固定阈值参数。"""
    parser = argparse.ArgumentParser(description="回放MaixCAM2拼图碎片实拍图")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--output", required=True, help="叠加结果图片路径")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="可选工作区矩形，省略时使用整张图片",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="0到255固定灰度阈值，省略时使用Otsu自动阈值",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """执行单张实拍图识别，保存叠加图并打印JSON几何结果。

    主要流程：解析参数、读取BGR图、调用analyze_frame、写入输出并打印可序列化结果。
    关键参数：argv可由测试传入；正常命令行运行时保持None读取系统参数。
    返回值：成功时返回0；输入读取或输出写入失败时抛出明确异常。
    """
    args = parse_args(argv)
    if args.threshold is not None and not 0.0 <= args.threshold <= 255.0:
        raise ValueError("threshold 必须位于 0 到 255 之间")

    frame_bgr = cv2.imread(args.input)
    if frame_bgr is None:
        raise FileNotFoundError(f"无法读取输入图片：{args.input}")

    roi = tuple(args.roi) if args.roi is not None else None
    overlay, pieces, threshold = analyze_frame(
        frame_bgr,
        roi=roi,
        fixed_threshold=args.threshold,
    )

    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)
    if not cv2.imwrite(args.output, overlay):
        raise OSError(f"无法写入输出图片：{args.output}")

    # 仅输出机械和拼图算法需要的JSON原生字段，排除不可序列化的OpenCV轮廓数组。
    records = []
    for piece in pieces:
        records.append(
            {
                "id": piece["id"],
                "vertices": piece["vertices"],
                "center": piece["center"],
                "angle_deg": piece["angle_deg"],
                "area": piece["area"],
                "complete": piece["complete"],
            }
        )
    print(
        json.dumps(
            {"threshold": threshold, "piece_count": len(records), "pieces": records},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
