"""验证同一原始帧的A/B离线对比输出和错误处理。"""

import cv2
import pytest

from tests_ab.synthetic_paper import write_synthetic_a4_frame


def test_compare_tool_writes_quad_warp_and_side_by_side_images(tmp_path):
    """验证工具一次生成A、B和同帧并排三张可解码图片。"""
    from tools.compare_roi_variants import compare_frame

    frame_path, paper_quad = write_synthetic_a4_frame(tmp_path)

    outputs = compare_frame(
        frame_path,
        tmp_path / "compare",
        paper_quad,
        inset_mm=2.0,
    )

    assert outputs["quad"].is_file()
    assert outputs["warp"].is_file()
    assert outputs["side_by_side"].is_file()
    side_by_side = cv2.imread(str(outputs["side_by_side"]))
    assert side_by_side is not None
    assert side_by_side.shape[:2] == (480, 1280)


def test_compare_tool_can_auto_locate_when_quad_is_omitted(tmp_path):
    """验证未提供四角时只自动定位一次并继续生成对比结果。"""
    from tools.compare_roi_variants import compare_frame

    frame_path, _paper_quad = write_synthetic_a4_frame(tmp_path)

    outputs = compare_frame(frame_path, tmp_path / "auto", inset_mm=0.0)

    assert outputs["side_by_side"].is_file()


def test_compare_tool_rejects_missing_or_undecodable_image(tmp_path):
    """验证输入路径错误和图片解码失败不会生成误导性空白对比图。"""
    from tools.compare_roi_variants import compare_frame

    with pytest.raises(FileNotFoundError):
        compare_frame(tmp_path / "missing.jpg", tmp_path / "output")

    invalid_path = tmp_path / "invalid.jpg"
    invalid_path.write_text("not an image", encoding="ascii")
    with pytest.raises(ValueError, match="解码"):
        compare_frame(invalid_path, tmp_path / "output")
