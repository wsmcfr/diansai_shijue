"""集中保存视觉阈值和题目约束，便于现场统一调参。"""


# MaixVision会在/tmp运行脚本，现场参数必须放到设备持久目录，避免重启后丢失。
PERSISTENT_SETTINGS_PATH = "/root/maixcam2_puzzle_A/vision_settings.json"

# 已知四片的形状与正确布局必须跨MaixVision临时目录和设备重启保留。
PERSISTENT_TEMPLATE_PATH = "/root/maixcam2_puzzle_A/known_templates.json"


# AUTO ROI四角拟合比例：第一个2%是干净纸张严格路径，只有无法得到四角时才依次
# 使用后续容错比例。现场若需调整只能保持正数、严格递增，且最大值不得超过10%。
PAPER_QUAD_EPSILON_RATIOS = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050)


# AUTO ROI严格路径失败后，以Otsu阈值为中心扫描的灰度偏移。负偏移用于断开黑纸与
# 阴影/龙门架粘连，正偏移用于补回受反光影响而变亮的黑纸区域；0保留同阈值宽松验收。
PAPER_AUTO_THRESHOLD_OFFSETS = (-24, -12, 0, 12, 24)

# 宽松路径只允许足够大的主轮廓进入评分，避免约1%的A4比例小暗块被误锁。
PAPER_AUTO_RELAXED_MIN_AREA_RATIO = 0.08
PAPER_AUTO_MIN_AREA_TO_LARGEST = 0.55

# 原始轮廓受反光分裂时矩形度会下降，因此宽松路径降低该项，但必须与面积、A4比例、
# 内部暗度和真实边缘支持联合使用，不能单独继续降低。
PAPER_AUTO_RELAXED_MIN_RECTANGULARITY = 0.35
PAPER_AUTO_RELAXED_MIN_ASPECT_SCORE = 0.70
PAPER_AUTO_RELAXED_MIN_DARKNESS = 0.42

# 四边边缘支持要求平均命中率达到55%，并且至少三条边分别达到单边最低标准。
PAPER_AUTO_MIN_EDGE_SUPPORT = 0.55
PAPER_AUTO_MIN_SUPPORTED_SIDES = 3

# 每张宽松掩膜只处理面积最大的有限轮廓，限制MaixCAM2单次AUTO的CPU耗时。
PAPER_AUTO_MAX_CONTOURS_PER_MASK = 4

# 相机固定时，旧ROI兜底只允许小范围边缘修正；IoU过低表示纸张或候选位置已改变，
# 必须拒绝而不能依赖旧蓝框返回虚假成功。
PAPER_AUTO_PRIOR_MIN_IOU = 0.55
PAPER_AUTO_PRIOR_MAX_SHIFT_RATIO = 0.04


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
    "paper_auto_threshold_offsets": PAPER_AUTO_THRESHOLD_OFFSETS,
    "paper_auto_relaxed_min_area_ratio": PAPER_AUTO_RELAXED_MIN_AREA_RATIO,
    "paper_auto_min_area_to_largest": PAPER_AUTO_MIN_AREA_TO_LARGEST,
    "paper_auto_relaxed_min_rectangularity": PAPER_AUTO_RELAXED_MIN_RECTANGULARITY,
    "paper_auto_relaxed_min_aspect_score": PAPER_AUTO_RELAXED_MIN_ASPECT_SCORE,
    "paper_auto_relaxed_min_darkness": PAPER_AUTO_RELAXED_MIN_DARKNESS,
    "paper_auto_min_edge_support": PAPER_AUTO_MIN_EDGE_SUPPORT,
    "paper_auto_min_supported_sides": PAPER_AUTO_MIN_SUPPORTED_SIDES,
    "paper_auto_max_contours_per_mask": PAPER_AUTO_MAX_CONTOURS_PER_MASK,
    "paper_auto_prior_min_iou": PAPER_AUTO_PRIOR_MIN_IOU,
    "paper_auto_prior_max_shift_ratio": PAPER_AUTO_PRIOR_MAX_SHIFT_RATIO,
    # 固定相机安装方向同时决定AUTO H/V转换、毫米原点和上下区。当前结构中源碎片区
    # 在原始CAL画面左侧、目标区在右侧；若实机相反改为side_lower_left。顶置相机用top。
    "camera_mount_direction": "side_lower_right",
}
