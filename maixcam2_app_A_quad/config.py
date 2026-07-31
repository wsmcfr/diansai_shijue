"""集中保存视觉阈值和题目约束，便于现场统一调参。"""


# MaixVision会在/tmp运行脚本，现场参数必须放到设备持久目录，避免重启后丢失。
PERSISTENT_SETTINGS_PATH = "/root/maixcam2_puzzle_A/vision_settings.json"

# 已知四片的形状与正确布局必须跨MaixVision临时目录和设备重启保留。
PERSISTENT_TEMPLATE_PATH = "/root/maixcam2_puzzle_A/known_templates.json"


# AUTO ROI四角拟合比例：第一个2%是干净纸张严格路径，只有无法得到四角时才依次
# 使用后续容错比例。现场若需调整只能保持正数、严格递增，且最大值不得超过10%。
PAPER_QUAD_EPSILON_RATIOS = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050)


# 默认视觉配置：所有会影响识别结果的参数集中在这里，避免算法中散落魔法数字。
DEFAULT_CONFIG = {
    # 识别采集使用1280x960提高纸面像素密度；屏幕和触摸继续使用640x480。
    # main.py会在初始化失败时回退到显示分辨率，并把RES LOW状态留在屏幕上。
    "capture_width": 1280,
    "capture_height": 960,
    "display_width": 640,
    "display_height": 480,
    "camera_width": 640,
    "camera_height": 480,
    "max_pieces": 4,
    "min_vertices": 3,
    "max_vertices": 5,
    "min_area_ratio": 0.002,
    "max_area_ratio": 0.60,
    # 远距离下2mm缝隙只有少量像素，3x3模糊保留抗噪能力且不过度抹平黑缝。
    "gaussian_kernel": 3,
    "open_kernel": 3,
    # 1x1闭运算等价于不扩张外轮廓；牌面内部孔洞由连通域填孔单独处理。
    "close_kernel": 1,
    "fill_internal_holes": True,
    # None 表示使用 Otsu 自动阈值；现场光照固定后可改为 0 至 255 的灰度阈值。
    "fixed_threshold": None,
    "approx_epsilon_min": 0.006,
    "approx_epsilon_max": 0.040,
    "approx_epsilon_step": 0.002,
    "border_margin_px": 2,
    # 32点轮廓距离已消除4/5顶点跳变；1.60容纳约2mm边缘毛刺，镜像和异形仍拒识。
    "known_match_threshold": 1.60,
    # 自动黑纸定位参数只在点击 AUTO ROI 时使用，不参与正常逐帧碎片识别。
    "paper_min_area_ratio": 0.01,
    "paper_max_area_ratio": 0.50,
    "paper_expected_aspect": 210.0 / 297.0,
    "paper_min_rectangularity": 0.70,
    "paper_min_confidence": 0.65,
    "paper_close_kernel": 9,
    "paper_quad_epsilon_ratios": PAPER_QUAD_EPSILON_RATIOS,
    # 固定相机安装方向同时决定AUTO H/V转换、毫米原点和上下区。当前结构中源碎片区
    # 在原始CAL画面左侧、目标区在右侧；若实机相反改为side_lower_left。顶置相机用top。
    "camera_mount_direction": "side_lower_right",
}
