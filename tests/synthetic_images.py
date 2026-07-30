"""为视觉算法测试生成可重复的黑底白片合成图。"""

import cv2
import numpy as np


def make_black_scene(polygons, size=(640, 480)):
    """生成黑底白色多边形图像。

    主要流程：创建指定尺寸的三通道黑色图像，再逐个填充白色多边形。
    关键参数：polygons 为顶点坐标序列列表，size 为 ``(宽, 高)``。
    返回值：OpenCV BGR 格式的 ``numpy.ndarray`` 图像。
    """
    width, height = size
    scene = np.zeros((height, width, 3), dtype=np.uint8)

    for polygon in polygons:
        # OpenCV 要求每个多边形使用 int32 坐标；负坐标会由绘制函数自动裁剪。
        points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(scene, [points], (255, 255, 255))

    return scene
