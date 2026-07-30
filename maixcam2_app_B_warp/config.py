"""集中保存视觉阈值和题目约束，便于现场统一调参。"""


# MaixVision会在/tmp运行脚本，现场参数必须放到设备持久目录，避免重启后丢失。
PERSISTENT_SETTINGS_PATH = "/root/maixcam2_puzzle_B/vision_settings.json"


# 默认视觉配置：所有会影响识别结果的参数集中在这里，避免算法中散落魔法数字。
DEFAULT_CONFIG = {
    "camera_width": 640,
    "camera_height": 480,
    "max_pieces": 4,
    "min_vertices": 3,
    "max_vertices": 5,
    "min_area_ratio": 0.002,
    "max_area_ratio": 0.60,
    "gaussian_kernel": 5,
    "open_kernel": 3,
    "close_kernel": 5,
    # None 表示使用 Otsu 自动阈值；现场光照固定后可改为 0 至 255 的灰度阈值。
    "fixed_threshold": None,
    "approx_epsilon_min": 0.006,
    "approx_epsilon_max": 0.040,
    "approx_epsilon_step": 0.002,
    "border_margin_px": 2,
    "known_match_threshold": 1.20,
    # 自动黑纸定位参数只在点击 AUTO ROI 时使用，不参与正常逐帧碎片识别。
    "paper_min_area_ratio": 0.01,
    "paper_max_area_ratio": 0.50,
    "paper_expected_aspect": 210.0 / 297.0,
    "paper_min_rectangularity": 0.70,
    "paper_min_confidence": 0.65,
    "paper_close_kernel": 9,
}
