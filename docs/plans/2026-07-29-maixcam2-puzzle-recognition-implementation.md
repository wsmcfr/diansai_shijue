# MaixCAM2 Puzzle Recognition Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 MaixCAM2 上实现黑底白色拼图碎片识别，并通过触摸屏切换已知模板模式和未知碎片模式。

**Architecture:** 将算法拆成可在 PC 上测试的纯 OpenCV 核心和依赖 MaixPy 硬件接口的应用入口。视觉核心负责分割、轮廓拟合、几何描述和模板匹配；MaixCAM2 入口只负责取图、触摸事件、模式状态、结果绘制及模板持久化。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、MaixPy `camera/image/display/touchscreen/app`

---

## 实施约束

| 约束 | 执行方式 |
|---|---|
| 测试先行 | 每个算法行为先写 pytest，再编写最小实现 |
| 中文注释 | 每个函数、核心结构、关键变量和非直观分支均写中文注释或中文文档字符串 |
| 硬件隔离 | PC 单元测试不得导入 `maix` 模块 |
| 输入范围 | 第一版只保证识别互不接触、互不重叠的 1 至 4 片碎片 |
| 图像输出 | 核心算法不直接修改输入图，绘制在单独的显示帧上完成 |
| Git 状态 | 当前 `.git` 为空目录，不是有效仓库；所有提交步骤暂时跳过，不擅自初始化仓库 |

## 目标文件

```text
maixcam2_app/
  __init__.py
  config.py
  puzzle_vision.py
  template_store.py
  touch_ui.py
  main.py
tests/
  __init__.py
  synthetic_images.py
  test_puzzle_vision.py
  test_template_store.py
  test_touch_ui.py
tools/
  replay_image.py
requirements-dev.txt
README.md
```

### Task 1: 建立配置和合成图测试基础

**Files:**

- Create: `maixcam2_app/__init__.py`
- Create: `maixcam2_app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/synthetic_images.py`
- Create: `requirements-dev.txt`

**Step 1: 写配置导入失败测试**

在 `tests/test_puzzle_vision.py` 中先写：

```python
def test_default_config_limits_piece_count():
    """默认配置必须符合题目最多四片的限制。"""
    from maixcam2_app.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["max_pieces"] == 4
    assert DEFAULT_CONFIG["min_vertices"] == 3
    assert DEFAULT_CONFIG["max_vertices"] == 5
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_puzzle_vision.py::test_default_config_limits_piece_count -v`

Expected: FAIL，错误包含 `ModuleNotFoundError` 或缺少 `DEFAULT_CONFIG`。

**Step 3: 编写最小配置**

`maixcam2_app/config.py` 至少包含：

```python
"""集中保存视觉阈值和题目约束，便于现场统一调参。"""

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
    "approx_epsilon_min": 0.006,
    "approx_epsilon_max": 0.040,
    "approx_epsilon_step": 0.002,
    "border_margin_px": 2,
    "known_match_threshold": 1.20,
}
```

`tests/synthetic_images.py` 提供 `make_black_scene(polygons, size=(640, 480))`，用 `cv2.fillPoly` 生成无外部文件依赖的黑底白片测试图；函数必须说明输入顶点、图像尺寸和 BGR 返回值。

`requirements-dev.txt` 写入 `numpy`、`opencv-python` 和 `pytest`。

**Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_puzzle_vision.py::test_default_config_limits_piece_count -v`

Expected: PASS。

**Step 5: 检查语法**

Run: `python -m compileall maixcam2_app tests`

Expected: exit code 0，不出现 `SyntaxError`。

### Task 2: 实现黑底白片分割和轮廓筛选

**Files:**

- Create: `maixcam2_app/puzzle_vision.py`
- Modify: `tests/test_puzzle_vision.py`

**Step 1: 写失败测试**

覆盖以下行为：

```python
def test_detects_four_separated_white_pieces():
    """四个互不接触的白色多边形必须被识别为四片。"""
    scene = make_black_scene([
        [(40, 80), (140, 70), (120, 150)],
        [(190, 60), (290, 80), (270, 160), (180, 140)],
        [(340, 70), (430, 60), (460, 130), (390, 170), (330, 130)],
        [(480, 70), (580, 80), (570, 170), (500, 160)],
    ])

    result = detect_pieces(scene, roi=(0, 0, 640, 240))

    assert len(result.pieces) == 4
    assert result.threshold > 0


def test_ignores_small_white_noise():
    """面积过小的白点不得被当作拼图碎片。"""
    scene = make_black_scene([[(100, 80), (220, 80), (160, 180)]])
    cv2.circle(scene, (20, 20), 2, (255, 255, 255), -1)

    result = detect_pieces(scene, roi=(0, 0, 640, 240))

    assert len(result.pieces) == 1
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_puzzle_vision.py -k "detects_four or ignores_small" -v`

Expected: FAIL，错误指出 `detect_pieces` 尚未定义。

**Step 3: 编写最小分割实现**

在 `puzzle_vision.py` 中实现以下公开接口：

```python
class DetectionResult:
    """保存单帧检测结果、二值图和自动阈值，供显示与调试使用。"""

    def __init__(self, pieces, mask, threshold, roi):
        self.pieces = pieces
        self.mask = mask
        self.threshold = threshold
        self.roi = roi


def build_foreground_mask(frame_bgr, roi, config=None):
    """在指定工作区内分割黑色背景上的亮色碎片并返回掩膜与阈值。"""
    # 复制默认配置，避免现场调参时修改全局字典。
    # 检查 ROI 和奇数核尺寸，非法输入抛出 ValueError。
    # 灰度化、GaussianBlur、Otsu、开运算、闭运算。


def detect_pieces(frame_bgr, roi, config=None):
    """提取有效外轮廓，按面积降序最多保留四片并返回检测结果。"""
    # 使用 RETR_EXTERNAL，按 ROI 面积比例过滤轮廓。
    # 将 ROI 内轮廓坐标平移回原图坐标。
```

此任务只建立轮廓集合，不做多边形顶点和模板匹配。

**Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_puzzle_vision.py -k "detects_four or ignores_small" -v`

Expected: 2 passed。

**Step 5: 运行当前测试集**

Run: `python -m pytest -q`

Expected: 全部通过。

### Task 3: 实现多边形拟合和几何特征

**Files:**

- Modify: `maixcam2_app/puzzle_vision.py`
- Modify: `tests/test_puzzle_vision.py`

**Step 1: 写失败测试**

新增：

```python
def test_extracts_triangle_geometry():
    """三角碎片必须输出三个顶点、有效中心和边长。"""
    scene = make_black_scene([[(100, 80), (260, 100), (180, 220)]])

    piece = detect_pieces(scene, roi=(0, 0, 640, 300)).pieces[0]

    assert len(piece["vertices"]) == 3
    assert 150 <= piece["center"][0] <= 210
    assert 110 <= piece["center"][1] <= 170
    assert len(piece["edge_lengths"]) == 3
    assert all(length > 0 for length in piece["edge_lengths"])


def test_marks_contour_touching_roi_border_incomplete():
    """接触工作区边界的碎片必须标记为不完整。"""
    scene = make_black_scene([[(-20, 60), (100, 60), (80, 180), (0, 170)]])

    piece = detect_pieces(scene, roi=(0, 0, 640, 240)).pieces[0]

    assert piece["complete"] is False
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_puzzle_vision.py -k "triangle_geometry or roi_border" -v`

Expected: FAIL，缺少 `vertices` 或 `complete`。

**Step 3: 编写几何提取实现**

实现：

```python
def approximate_polygon(contour, config):
    """搜索轮廓周长比例误差，优先得到题目允许的三至五个主要顶点。"""
    # 从 epsilon_min 增长到 epsilon_max。
    # 首次得到 3~5 个顶点时返回；否则返回最接近范围的候选。


def compute_piece_geometry(contour, roi, config):
    """计算碎片顶点、中心、方向角、边长、内角和完整性标志。"""
    # 中心优先使用图像矩；矩为零时使用 minAreaRect 中心。
    # 角度规范到 [-90, 90) 范围，消除 OpenCV 宽高交换造成的跳变。
    # 使用顶点到 ROI 四边的距离判定轮廓是否完整。
```

每个 piece 字典必须至少含：`contour`、`vertices`、`center`、`angle_deg`、`area`、`perimeter`、`edge_lengths`、`interior_angles`、`vertex_count`、`complete`。

**Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_puzzle_vision.py -k "triangle_geometry or roi_border" -v`

Expected: 2 passed。

**Step 5: 运行全部视觉测试**

Run: `python -m pytest tests/test_puzzle_vision.py -v`

Expected: 全部通过。

### Task 4: 实现未知模式稳定编号

**Files:**

- Modify: `maixcam2_app/puzzle_vision.py`
- Modify: `tests/test_puzzle_vision.py`

**Step 1: 写失败测试**

```python
def test_assigns_unknown_ids_top_to_bottom_then_left_to_right():
    """未知碎片编号必须具有稳定的空间顺序。"""
    pieces = [
        {"center": (300.0, 160.0)},
        {"center": (220.0, 80.0)},
        {"center": (80.0, 80.0)},
    ]

    assign_unknown_ids(pieces, row_tolerance_px=30)

    assert [(piece["id"], piece["center"]) for piece in pieces] == [
        ("U1", (80.0, 80.0)),
        ("U2", (220.0, 80.0)),
        ("U3", (300.0, 160.0)),
    ]
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_puzzle_vision.py::test_assigns_unknown_ids_top_to_bottom_then_left_to_right -v`

Expected: FAIL，`assign_unknown_ids` 尚未定义。

**Step 3: 实现编号逻辑**

实现 `assign_unknown_ids(pieces, row_tolerance_px)`：先按中心 Y 分行，同一行内按 X 排序，再原地重排列表并赋予 `U1` 至 `U4`。行容差必须作为参数传入，不得隐藏为魔法数字。

**Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_puzzle_vision.py::test_assigns_unknown_ids_top_to_bottom_then_left_to_right -v`

Expected: PASS。

### Task 5: 实现模板保存、加载和一对一匹配

**Files:**

- Create: `maixcam2_app/template_store.py`
- Create: `tests/test_template_store.py`

**Step 1: 写失败测试**

测试至少包括：

1. 四个 piece 可以转换为不含 NumPy 对象的 JSON 模板；
2. 保存后重新加载保持模板编号和描述子数值；
3. 同一形状旋转后仍匹配相同 `K` 编号；
4. 一对一匹配不会把两个观测都分配给同一模板；
5. 分数超过阈值时编号为 `UNKNOWN`。

关键测试：

```python
def test_global_match_uses_each_template_once():
    """全局匹配必须保证每个已知模板最多分配一次。"""
    templates = [make_template("K1", triangle_piece), make_template("K2", quad_piece)]
    observations = [rotated_triangle_piece, rotated_quad_piece]

    matched = match_known_pieces(observations, templates, max_score=1.2)

    assert {piece["id"] for piece in matched} == {"K1", "K2"}
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_template_store.py -v`

Expected: FAIL，模块尚不存在。

**Step 3: 实现描述子与存储**

`template_store.py` 实现：

```python
def build_shape_descriptor(piece):
    """构造对平移、缩放和旋转不敏感的边长、内角及 Hu 矩描述子。"""


def register_templates(pieces):
    """按稳定几何顺序将四片已知碎片登记为 K1 至 K4。"""


def save_templates(path, templates):
    """通过临时文件和原子替换保存模板，防止断电留下半个 JSON。"""


def load_templates(path):
    """读取并校验模板版本；文件不存在时返回空列表。"""


def descriptor_distance(observation, template):
    """综合顶点数、循环边长、循环内角和 Hu 矩得到形状距离。"""


def match_known_pieces(pieces, templates, max_score):
    """穷举不超过四片的排列，返回总代价最低的一对一模板匹配。"""
```

写文件使用标准 `json` 和 `os.replace`。临时文件路径固定为目标路径加 `.tmp`，并在异常分支中清理临时文件。

**Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_template_store.py -v`

Expected: 全部通过。

**Step 5: 运行算法测试集**

Run: `python -m pytest tests/test_puzzle_vision.py tests/test_template_store.py -q`

Expected: 全部通过。

### Task 6: 实现触摸松开事件和模式切换

**Files:**

- Create: `maixcam2_app/touch_ui.py`
- Create: `tests/test_touch_ui.py`

**Step 1: 写失败测试**

```python
def test_touch_tracker_emits_one_click_on_release():
    """一次按下和松开只能产生一个点击事件。"""
    tracker = TouchReleaseTracker()

    assert tracker.update(30, 20, True) is None
    assert tracker.update(31, 21, True) is None
    assert tracker.update(31, 21, False) == (31, 21)
    assert tracker.update(31, 21, False) is None


def test_mode_button_switches_without_repeating():
    """模式按键命中后必须切换到对应模式。"""
    buttons = build_button_layout(640, 480)

    assert hit_test(buttons["known"].center, buttons) == "known"
    assert hit_test(buttons["unknown"].center, buttons) == "unknown"
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_touch_ui.py -v`

Expected: FAIL，模块尚不存在。

**Step 3: 实现纯逻辑 UI 状态**

`touch_ui.py` 不导入 `maix`，实现：

- `ButtonRect`：保存按钮名称和矩形坐标，并提供 `contains(x, y)`；
- `TouchReleaseTracker`：只在按下后首次松开时返回点击坐标；
- `build_button_layout(width, height)`：创建 `known`、`unknown`、`save` 三个稳定按钮区域；
- `map_display_to_image(...)`：按照 `FIT_CONTAIN` 的缩放和黑边将屏幕触点反算到图像坐标；
- `hit_test(point, buttons)`：返回命中的按钮名称或 `None`。

**Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_touch_ui.py -v`

Expected: 全部通过。

### Task 7: 集成 MaixCAM2 应用入口

**Files:**

- Create: `maixcam2_app/main.py`
- Modify: `maixcam2_app/config.py`
- Create: `tests/test_main_import_guard.py`

**Step 1: 写硬件导入隔离测试**

```python
def test_pc_can_import_main_without_maix_runtime():
    """PC 测试环境导入入口模块时不得立即初始化硬件。"""
    module = importlib.import_module("maixcam2_app.main")

    assert callable(module.run_app)
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_main_import_guard.py -v`

Expected: FAIL，入口模块不存在。

**Step 3: 编写入口和主循环**

`main.py` 必须遵循以下结构：

```python
def run_app():
    """初始化 MaixCAM2 硬件并运行识别、触摸和显示主循环。"""
    # 函数内部再导入 maix，保证 PC 可以导入模块做静态测试。
    from maix import app, camera, display, image, touchscreen

    # 1. 创建 640×480 BGR 摄像头并跳过 30 帧。
    # 2. 创建屏幕和触摸对象。
    # 3. 默认进入 unknown 模式，避免模板缺失时误判。
    # 4. 每帧调用 detect_pieces。
    # 5. known 模式加载模板并执行全局匹配；unknown 模式执行稳定编号。
    # 6. 处理松开点击，切换模式或在恰好四片完整碎片时保存模板。
    # 7. 在复制后的帧上绘制轮廓、顶点、中心、编号、角度、状态栏和按键。
    # 8. 转为 maix.image.Image 后显示，直到 app.need_exit()。


if __name__ == "__main__":
    run_app()
```

模板默认保存到 `os.path.join(os.path.dirname(__file__), "known_templates.json")`。保存失败必须捕获异常并在状态栏显示错误，不得终止摄像头主循环。

显示按钮使用 ASCII 标签 `KNOWN`、`UNKNOWN`、`SAVE`，避免设备缺少中文字库时出现方框；源代码注释和日志说明仍使用中文。

**Step 4: 运行 PC 导入测试**

Run: `python -m pytest tests/test_main_import_guard.py -v`

Expected: PASS。

**Step 5: 运行静态语法检查**

Run: `python -m compileall maixcam2_app`

Expected: exit code 0。

### Task 8: 添加实拍图回放工具和使用说明

**Files:**

- Create: `tools/replay_image.py`
- Create: `README.md`

**Step 1: 编写回放工具**

`tools/replay_image.py` 接收图片路径、可选 ROI 和可选阈值参数，调用与设备完全相同的 `detect_pieces`，输出每片的编号、顶点、中心和角度，并将叠加结果保存到用户指定的输出路径。

关键函数：

```python
def parse_args():
    """解析输入图片、输出图片和可选 ROI 参数。"""


def main():
    """运行单张实拍图识别并输出可视化结果和几何数据。"""
```

**Step 2: 编写 README**

README 必须包含：

1. 项目范围和目录结构；
2. PC 安装与测试命令；
3. 实拍图回放命令；
4. MaixVision 上传整个 `maixcam2_app` 目录并运行 `main.py` 的步骤；
5. 先将四片分开摆放，再在 KNOWN 模式点击 SAVE 录入模板；
6. 黑底、正常室内照明、相机固定和启动前停车的现场条件；
7. 当前不处理拼好后互相接触碎片的拆分。

**Step 3: 用当前实拍照片做回归检查**

Run:

```powershell
python tools/replay_image.py "C:\Users\caofengrui\xwechat_files\wxid_7rrsjrp61o4722_e346\temp\RWTemp\2026-07\9e20f478899dc29eb19741386f9343c8\64b25eed802dc2c401a6aa755ad4131f.jpg" --output "tmp\replay-black.jpg"
```

Expected: 命令成功结束并输出至少一个大轮廓；由于照片中四片已经接触，不要求识别为四片。

**Step 4: 检查输出图片**

使用图像查看工具确认轮廓、中心和文字没有越界，原图未被覆盖修改。

### Task 9: 完整验证与 MaixCAM2 实机检查

**Files:**

- Modify: `README.md`（只记录真实验证结果）

**Step 1: 运行全部自动化测试**

Run: `python -m pytest -v`

Expected: 全部通过，0 failed。

**Step 2: 运行语法编译检查**

Run: `python -m compileall maixcam2_app tests tools`

Expected: exit code 0，无语法错误。

**Step 3: 运行合成图压力测试**

生成不同位置、旋转角度、亮度和少量噪声的 1 至 4 片场景，至少循环 100 组；验证检测数量正确率并输出失败样本参数。验收要求为无接触合成场景数量识别 100% 正确。

**Step 4: MaixCAM2 实机验证**

在 MaixVision 中运行 `maixcam2_app/main.py`，逐项记录：

| 验证项 | 通过标准 |
|---|---|
| 摄像头 | 连续显示，无异常退出 |
| 未知模式 | 1 至 4 片分开摆放时数量和轮廓正确 |
| 已知录入 | 四片完整时 SAVE 成功，重启后模板仍存在 |
| 已知匹配 | 任意旋转四片后 K1 至 K4 编号保持一致 |
| 触摸切换 | 每次松开只切换一次，状态高亮正确 |
| 性能 | 记录平均处理耗时，不出现明显卡死 |

**Step 5: 只依据证据更新结论**

PC 测试通过但尚未上板时，README 只能写“PC 已验证，MaixCAM2 实机未验证”。获得实机画面、日志和操作结果后，才能标记对应硬件项已验证。

## 实施清单

| 序号 | 操作 | 验证标准 | 审查 |
|---|---|---|---|
| 1 | 创建配置与合成图测试基础 | 配置测试和 compileall 通过 | `review:false` |
| 2 | 实现阈值分割与轮廓筛选 | 四片检测及噪点过滤测试通过 | `review:true` |
| 3 | 实现多边形与几何特征 | 顶点、中心、边长及边界测试通过 | `review:true` |
| 4 | 实现未知碎片稳定编号 | 空间排序测试通过 | `review:false` |
| 5 | 实现已知模板登记与全局匹配 | JSON往返、旋转不变和一对一匹配测试通过 | `review:true` |
| 6 | 实现触摸事件与按钮映射 | 单次松开和坐标映射测试通过 | `review:true` |
| 7 | 集成 MaixCAM2 主循环 | PC可安全导入且 compileall 通过 | `review:true` |
| 8 | 添加实拍回放与README | 当前照片回放成功、叠加图可读 | `review:false` |
| 9 | 完整测试与实机验收 | pytest全绿，并区分PC已验证与硬件未验证 | `review:true` |

