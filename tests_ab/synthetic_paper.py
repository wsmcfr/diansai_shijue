"""为自动黑纸 ROI、四边形掩膜和透视展开提供可复用合成场景。"""

from pathlib import Path

import cv2
import numpy as np


DEFAULT_PAPER_QUAD = np.float32(
    [[220, 70], [390, 80], [410, 330], [205, 325]]
)


def _map_physical_polygon(polygon_mm, paper_quad):
    """把A4毫米坐标多边形映射到给定相机四边形。

    主要流程：建立标准210×297毫米平面到相机画面的单应性，再批量映射顶点。
    关键参数：polygon_mm 为二维毫米点列，paper_quad 按左上、右上、右下、左下排序。
    返回值：可直接传给 OpenCV 绘制函数的 int32 像素点列。
    """
    source_quad = np.float32([[0, 0], [210, 0], [210, 297], [0, 297]])
    matrix = cv2.getPerspectiveTransform(source_quad, np.float32(paper_quad))
    points = np.asarray(polygon_mm, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(points, matrix)[0].round().astype(np.int32)


def make_paper_scene(paper_quad, white_pieces=(), dark_objects=(), size=(640, 480)):
    """生成亮地面、黑色A4、白色碎片和外部暗色干扰物组成的测试图。

    关键参数：size 为 ``(宽, 高)``；所有多边形使用相机像素坐标。
    返回值：三通道 BGR uint8 图像，颜色留出阈值裕量以模拟实际亮度差。
    """
    width, height = size
    frame = np.full((height, width, 3), 210, dtype=np.uint8)
    cv2.fillConvexPoly(frame, np.asarray(paper_quad, dtype=np.int32), (20, 20, 20))
    for polygon in white_pieces:
        cv2.fillConvexPoly(frame, np.asarray(polygon, dtype=np.int32), (245, 245, 245))
    for polygon in dark_objects:
        cv2.fillConvexPoly(frame, np.asarray(polygon, dtype=np.int32), (15, 15, 15))
    return frame


def _piece_polygons_mm(piece_count):
    """返回分散在机械有效区内的1～4片白色测试多边形。

    多边形与A4外边缘保持明显距离，保证自动定位测试只考察内部孔洞而非纸边遮挡。
    """
    polygons = (
        [[22, 55], [65, 50], [70, 91], [28, 96]],
        [[126, 58], [174, 65], [168, 104], [122, 98]],
        [[30, 174], [72, 164], [83, 205], [39, 216]],
        [[126, 174], [171, 168], [181, 207], [137, 220]],
    )
    if not 0 <= int(piece_count) <= len(polygons):
        raise ValueError("piece_count 必须位于0到4之间")
    return polygons[: int(piece_count)]


def make_scene_with_piece_count(paper_quad, piece_count, add_dark_rod=False):
    """生成带指定碎片数和可选外部暗杆的完整A4场景。

    返回值：640×480 BGR 图像；暗杆不接触A4，用于验证候选评分不会误选细长物体。
    """
    white_pieces = [
        _map_physical_polygon(polygon, paper_quad)
        for polygon in _piece_polygons_mm(piece_count)
    ]
    dark_objects = []
    if add_dark_rod:
        dark_objects.append(np.int32([[42, 25], [72, 25], [72, 440], [42, 440]]))
    return make_paper_scene(paper_quad, white_pieces, dark_objects)


def make_uneven_brightness_paper_scene():
    """生成单次Otsu只能看见半张纸、提高阈值后才能恢复完整A4的场景。

    主要流程：先绘制灰度25的完整黑纸，再把纸面右半覆盖为灰度100，同时把背景
    设为灰度150。当前单次Otsu会优先分离最暗的左半纸面；右半纸面与背景仍保留
    50级灰度边缘，可供后续多阈值与边缘支持路径恢复。返回值为三通道BGR帧和
    期望的完整A4四角。
    """
    paper_quad = np.float32(
        [[115, 85], [535, 72], [550, 390], [100, 405]]
    )
    frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    cv2.fillConvexPoly(frame, paper_quad.astype(np.int32), (25, 25, 25))
    bright_half = np.int32(
        [[270, 80], [535, 72], [550, 390], [260, 400]]
    )
    cv2.fillConvexPoly(frame, bright_half, (100, 100, 100))
    return frame, paper_quad


def make_large_distractor_with_small_a4_like_block():
    """生成大型非纸暗区和约1.2%画面的A4比例小框。

    大三角形用于代表龙门架、阴影或其它主暗区，它不能被四角拟合为A4；右上小框
    的短长边比例接近210:297，旧算法会因缺少相对面积门而把它误锁为纸张。返回值
    包含测试帧和小框四角，便于失败时确认算法没有返回该干扰物。
    """
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    large_triangle = np.int32([[40, 420], [320, 55], [420, 430]])
    small_block = np.int32([[525, 45], [600, 45], [600, 98], [525, 98]])
    cv2.fillConvexPoly(frame, large_triangle, (22, 22, 22))
    cv2.fillConvexPoly(frame, small_block, (18, 18, 18))
    return frame, small_block.astype(np.float32)


def _active_quad_for_test(paper_quad, inset_mm=0.0):
    """直接由物理坐标生成测试期望有效四边形，避免依赖被测生产函数。"""
    active_mm = np.float32(
        [
            [inset_mm, 33.5 + inset_mm],
            [210.0 - inset_mm, 33.5 + inset_mm],
            [210.0 - inset_mm, 263.5 - inset_mm],
            [inset_mm, 263.5 - inset_mm],
        ]
    )
    return _map_physical_polygon(active_mm, paper_quad).astype(np.float32)


def make_quad_scene_with_four_pieces(inset_mm=0.0):
    """生成A版四边形掩膜测试帧，并返回完整纸张与有效区四角。"""
    paper_quad = DEFAULT_PAPER_QUAD.copy()
    frame = make_scene_with_piece_count(paper_quad, 4, add_dark_rod=False)
    return frame, paper_quad, _active_quad_for_test(paper_quad, inset_mm)


def make_axis_aligned_a4_scene():
    """生成轴对齐A4，便于精确验证B版展开尺寸和裁剪坐标。"""
    paper_quad = np.float32([[145, 35], [355, 35], [355, 332], [145, 332]])
    return make_scene_with_piece_count(paper_quad, 4), paper_quad


def make_perspective_scene_with_four_pieces():
    """生成带明显透视的四片场景，供B版完整数据流测试。"""
    paper_quad = DEFAULT_PAPER_QUAD.copy()
    return make_scene_with_piece_count(paper_quad, 4), paper_quad


def write_synthetic_a4_frame(directory):
    """把标准合成帧写到目录并返回路径和四角，供命令行工具测试复用。"""
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame, paper_quad = make_perspective_scene_with_four_pieces()
    frame_path = output_directory / "synthetic_a4.jpg"
    if not cv2.imwrite(str(frame_path), frame):
        raise OSError("合成A4测试图写入失败")
    return frame_path, paper_quad
