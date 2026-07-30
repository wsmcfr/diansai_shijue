# MaixCAM2 Calibration UI Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为固定安装的 MaixCAM2 增加固定 ROI、视觉诊断、现场参数持久化和单个 `CAL` 按钮往返切换的屏幕调参界面。

**Architecture:** 保持现有 OpenCV 视觉核心与 MaixPy 硬件入口分离。视觉核心输出完整的轮廓分类诊断；新增纯 Python 调参状态机和 JSON 设置存储；`main.py` 只负责把相机帧、触摸动作、运行参数和两个界面连接起来。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、MaixPy `camera/image/display/touchscreen/app`

---

## 实施约束

| 约束 | 执行方式 |
|---|---|
| 测试先行 | 每个行为先运行失败测试，再编写最小实现 |
| 中文注释 | 新函数、状态类、关键变量和边界分支全部使用中文文档字符串或注释 |
| 硬件隔离 | PC 测试不得在模块导入阶段加载 `maix` |
| CAL 切换 | 同一个 `CAL` 按钮进入和退出调参界面 |
| 模式保留 | 切换 CAL 不改变 `KNOWN/UNKNOWN` |
| 未保存退出 | 丢弃本轮调参，恢复已保存参数 |
| 持久化 | 正式参数写入设备持久目录，不写 `/tmp/maixpy_run` |
| 改动控制 | 分四轮实施，每轮只处理一个可验证改动点 |
| Git 状态 | 当前 `.git` 为空目录，不是有效仓库；提交步骤记录为跳过，不初始化仓库 |

## 目标文件

```text
maixcam2_app/
  config.py
  puzzle_vision.py
  settings_store.py
  calibration_ui.py
  touch_ui.py
  main.py
tests/
  test_puzzle_vision.py
  test_settings_store.py
  test_calibration_ui.py
  test_touch_ui.py
  test_main_import_guard.py
docs/
  maixcam2-puzzle-recognition-guide.md
README.md
```

## Round 1：固定 ROI 根因回归与视觉诊断

`trace_id=maixcam2-cal-ui-20260729-r1`

**目标：** 让视觉核心解释轮廓为什么被接受或拒绝，并用自动测试固定“整帧失败、黑纸 ROI 成功”的现场根因。

**验证标准：** 新增诊断测试通过，原有 `tests/test_puzzle_vision.py` 全部通过。

**停止条件：** 如果修改需要改变现有碎片几何结构或模板匹配协议，停止并重新审查设计。

### Task 1: 写固定 ROI 回归测试

**Files:**

- Modify: `tests/test_puzzle_vision.py`

**Step 1: 写失败测试**

新增测试场景：640×480 亮灰背景，中部绘制黑色 A4 矩形，黑色矩形内绘制四个白色多边形。

```python
def test_fixed_black_paper_roi_excludes_bright_outer_background():
    """固定ROI必须排除亮地面，使黑纸内四片重新成为最外层轮廓。"""
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    cv2.rectangle(frame, (150, 20), (470, 459), (10, 10, 10), -1)
    polygons = [
        np.array([[190, 90], [300, 125], [195, 150]], np.int32),
        np.array([[330, 100], [430, 135], [410, 175], [345, 155]], np.int32),
        np.array([[210, 240], [300, 220], [320, 280], [235, 285]], np.int32),
        np.array([[350, 260], [410, 245], [430, 300], [370, 315]], np.int32),
    ]
    for polygon in polygons:
        cv2.fillPoly(frame, [polygon], (245, 245, 245))

    full_result = detect_pieces(frame, (0, 0, 640, 480))
    paper_result = detect_pieces(frame, (160, 30, 300, 420))

    assert len(full_result.pieces) == 0
    assert len(paper_result.pieces) == 4
    assert len(full_result.large_contours) >= 1
```

**Step 2: 验证测试因缺少诊断字段而失败**

Run: `python -m pytest tests/test_puzzle_vision.py::test_fixed_black_paper_roi_excludes_bright_outer_background -v`

Expected: FAIL，错误包含 `DetectionResult` 缺少 `large_contours`。

### Task 2: 扩展 DetectionResult 诊断数据

**Files:**

- Modify: `maixcam2_app/puzzle_vision.py`
- Modify: `tests/test_puzzle_vision.py`

**Step 1: 增加分类测试**

测试必须覆盖：

- 面积低于下限的轮廓进入 `small_contours`；
- 面积超过上限的轮廓进入 `large_contours`；
- 面积有效但接触 ROI 的轮廓进入 `edge_contours`；
- `white_ratio` 等于掩膜白色像素数除以 ROI 像素数；
- `valid_contour_count` 保留截断到四片之前的有效数量。

**Step 2: 运行分类测试并确认失败**

Run: `python -m pytest tests/test_puzzle_vision.py -k "diagnostic or fixed_black_paper" -v`

Expected: FAIL，缺少对应诊断属性。

**Step 3: 编写最小实现**

`DetectionResult.__init__` 接收并保存：

```python
def __init__(
    self,
    pieces,
    mask,
    threshold,
    roi,
    small_contours=None,
    large_contours=None,
    edge_contours=None,
    valid_contour_count=0,
    white_ratio=0.0,
):
    """保存单帧结果以及调参界面使用的轮廓分类诊断。"""
```

`detect_pieces` 在面积过滤时保留过小、过大和有效轮廓，统一转换到原图坐标。最终 `pieces` 仍只取面积最大的四片，保持现有业务行为不变。

**Step 4: 验证 Round 1**

Run: `python -m pytest tests/test_puzzle_vision.py -v`

Expected: 全部 PASS。

Run: `python -m compileall maixcam2_app tests`

Expected: exit code 0。

## Round 2：运行参数存储与调参状态机

`trace_id=maixcam2-cal-ui-20260729-r2`

**目标：** 实现可回退的工作参数、参数质量判断和原子 JSON 持久化，不依赖 MaixPy。

**验证标准：** 设置存储和调参状态机测试全部通过，损坏配置不会覆盖默认值。

**停止条件：** 如果设备持久目录不可写，仅保留错误状态，不回退到 `/tmp` 静默保存。

### Task 3: 实现运行参数存储

**Files:**

- Create: `maixcam2_app/settings_store.py`
- Create: `tests/test_settings_store.py`
- Modify: `maixcam2_app/config.py`

**Step 1: 写失败测试**

覆盖以下公开行为：

```python
def test_default_runtime_settings_use_full_frame():
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    assert settings["roi"] == [0, 0, 640, 480]
    assert settings["fixed_threshold"] is None


def test_runtime_settings_round_trip(tmp_path):
    path = tmp_path / "vision_settings.json"
    settings = build_default_runtime_settings(DEFAULT_CONFIG)
    settings["roi"] = [160, 30, 300, 420]
    save_runtime_settings(path, settings)
    assert load_runtime_settings(path, DEFAULT_CONFIG)["roi"] == [160, 30, 300, 420]


def test_invalid_runtime_settings_fall_back_without_partial_merge(tmp_path):
    path = tmp_path / "vision_settings.json"
    path.write_text('{"version": 1, "roi": [0, 0, -1, 20]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_settings(path, DEFAULT_CONFIG)
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_settings_store.py -v`

Expected: FAIL，错误包含 `ModuleNotFoundError`。

**Step 3: 编写最小实现**

公开接口：

```python
SETTINGS_VERSION = 1


def build_default_runtime_settings(config):
    """从默认视觉配置构造可持久化的现场参数。"""


def validate_runtime_settings(settings, frame_size):
    """校验ROI、阈值、面积比例和奇数形态学核并返回规范化副本。"""


def load_runtime_settings(path, config):
    """读取并校验现场参数；文件不存在时返回默认参数。"""


def save_runtime_settings(path, settings, frame_size):
    """使用同目录临时文件和原子替换保存已经校验的参数。"""


def merge_runtime_config(default_config, settings):
    """把现场参数合并到默认配置，返回供单帧检测使用的新字典。"""
```

在 `config.py` 增加设备持久路径常量：

```python
PERSISTENT_SETTINGS_PATH = "/root/maixcam2_puzzle/vision_settings.json"
```

**Step 4: 验证设置存储**

Run: `python -m pytest tests/test_settings_store.py -v`

Expected: 全部 PASS。

### Task 4: 实现调参状态机和质量判断

**Files:**

- Create: `maixcam2_app/calibration_ui.py`
- Create: `tests/test_calibration_ui.py`

**Step 1: 写失败测试**

测试覆盖：

- `CalibrationSession` 创建时复制原参数；
- ROI 四边按 1、5、10 像素步长移动且不能越界；
- `TH` 在 AUTO 和固定值之间切换；
- `MIN` 以 `0.0001 × step` 修改；
- OPEN/CLOSE 只允许正奇数；
- 未保存退出不修改原字典；
- GOOD、MISS、NOISE、EDGE、BACKGROUND 的优先级。

```python
def test_calibration_session_edits_copy_not_saved_settings():
    saved = {"roi": [0, 0, 640, 480], "fixed_threshold": None,
             "min_area_ratio": 0.002, "open_kernel": 3, "close_kernel": 5}
    session = CalibrationSession(saved, frame_size=(640, 480))
    session.select_item("LEFT")
    session.adjust(1)
    assert saved["roi"] == [0, 0, 640, 480]
    assert session.settings["roi"] != saved["roi"]
```

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_calibration_ui.py -v`

Expected: FAIL，错误包含 `ModuleNotFoundError`。

**Step 3: 编写最小状态机**

公开结构：

```python
class CalibrationSession:
    """维护一次未保存的调参会话、当前页面、参数项和步长。"""

    def select_view(self, view):
        """切换ROI、MASK或RESULT预览。"""

    def cycle_item(self):
        """按当前页面循环选择ROI边或分割参数。"""

    def adjust(self, direction):
        """按当前步长增减选中参数，并拒绝非法边界。"""

    def toggle_threshold_mode(self, current_otsu_threshold):
        """在Otsu自动阈值和当前固定阈值之间切换。"""

    def cycle_step(self):
        """在1、5、10三个步长之间循环。"""

    def snapshot(self):
        """返回可交给持久化层校验保存的参数副本。"""


def evaluate_calibration(result, expected_pieces=4):
    """根据完整数、有效候选、边缘、大轮廓和白色占比返回质量状态。"""
```

**Step 4: 验证 Round 2**

Run: `python -m pytest tests/test_settings_store.py tests/test_calibration_ui.py -v`

Expected: 全部 PASS。

## Round 3：触摸布局、单按钮切换和预览绘制

`trace_id=maixcam2-cal-ui-20260729-r3`

**目标：** 在纯 PC 环境完成固定尺寸按钮布局、同一个 CAL 矩形往返切换和三种调参预览图。

**验证标准：** 640×480 下所有按钮不重叠、文字区域固定，原图/MASK/RESULT 绘制测试通过。

**停止条件：** 如果按钮在 MaixCAM2 实际屏幕上小于现有按钮触摸尺寸，重新调整布局后再接入主循环。

### Task 5: 扩展触摸按钮布局

**Files:**

- Modify: `maixcam2_app/touch_ui.py`
- Modify: `tests/test_touch_ui.py`

**Step 1: 写失败测试**

覆盖：

- 正常布局包含 `known`、`unknown`、`save`、`cal`；
- 调参布局包含 `roi`、`mask`、`result`、`cal`、`item`、`minus`、`value`、`plus`、`step`、`save_settings`；
- 正常与调参布局的 `cal` 矩形完全一致；
- 640×480 下同一行按钮互不重叠；
- 各按钮中心均能命中对应动作。

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_touch_ui.py -v`

Expected: FAIL，缺少 CAL 或调参布局函数。

**Step 3: 编写最小布局实现**

新增：

```python
def build_cal_toggle_button(width, height):
    """创建正常与调参界面共用、位置完全一致的CAL按钮。"""


def build_calibration_layout(width, height):
    """创建调参页签和底部参数控制按钮，并保持预览区尺寸固定。"""
```

现有 `build_button_layout` 调用 `build_cal_toggle_button`，避免两个界面分别计算造成位置漂移。

**Step 4: 运行触摸测试**

Run: `python -m pytest tests/test_touch_ui.py -v`

Expected: 全部 PASS。

### Task 6: 绘制调参预览和状态

**Files:**

- Modify: `maixcam2_app/calibration_ui.py`
- Modify: `tests/test_calibration_ui.py`

**Step 1: 写失败测试**

覆盖：

- ROI 预览在副本上暗化外部区域并高亮选中边；
- MASK 预览只在 ROI 内显示黑白掩膜；
- RESULT 预览按 GREEN/ORANGE/RED/PURPLE 绘制分类轮廓；
- 状态栏文本包含质量、N、EDGE、SMALL、LARGE、TH 和 WHITE；
- 绘制不修改原始相机帧；
- 最长状态文本不会进入底部按钮区域。

**Step 2: 运行绘制测试并确认失败**

Run: `python -m pytest tests/test_calibration_ui.py -k "draw or status" -v`

Expected: FAIL，缺少绘制函数。

**Step 3: 编写最小绘制实现**

新增：

```python
def draw_calibration_frame(frame_bgr, result, session, buttons, quality):
    """根据当前ROI、MASK或RESULT页面返回完整640×480调参画面。"""


def format_calibration_status(result, quality):
    """生成固定字段顺序的ASCII诊断状态文本。"""
```

所有 OpenCV 颜色常量集中定义，按钮尺寸由 `touch_ui` 布局提供，不根据文字动态改变。

**Step 4: 验证 Round 3**

Run: `python -m pytest tests/test_touch_ui.py tests/test_calibration_ui.py -v`

Expected: 全部 PASS。

## Round 4：接入 MaixPy 主循环与文档

`trace_id=maixcam2-cal-ui-20260729-r4`

**目标：** 把固定 ROI、调参会话、CAL 往返切换、参数保存和三种预览接入现有设备入口。

**验证标准：** PC 可导入入口、平铺部署导入、全套测试和 compileall 全部通过；实机验证项明确标记为待用户运行。

**停止条件：** 如果 MaixPy 显示、触摸或持久目录 API 与 PC 假设不一致，保留完整设备堆栈并只修复一个 API 差异。

### Task 7: 写入口状态失败测试

**Files:**

- Modify: `tests/test_main_import_guard.py`
- Modify: `maixcam2_app/main.py`

**Step 1: 提取可测试界面状态并写测试**

新增纯状态结构：

```python
class InterfaceState:
    """维护运行/调参界面和未保存会话，不持有MaixPy硬件对象。"""

    def toggle_calibration(self, saved_settings, frame_size):
        """同一个CAL动作进入或退出调参；退出时丢弃未保存会话。"""
```

测试必须证明：

- 第一次调用进入 CAL；
- 第二次调用返回 RUN；
- 第二次调用丢弃未保存修改；
- 外部 `mode="known"` 在两次切换后保持不变；
- 主模块在 PC 和 MaixVision 平铺目录中仍可导入。

**Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_main_import_guard.py -k "calibration or flat" -v`

Expected: FAIL，缺少 `InterfaceState` 或 `cal` 布局。

### Task 8: 接入运行主循环

**Files:**

- Modify: `maixcam2_app/main.py`
- Modify: `maixcam2_app/config.py`

**Step 1: 加载持久参数**

`run_app` 初始化时：

1. 使用 `PERSISTENT_SETTINGS_PATH` 加载参数；
2. 不存在时使用默认整帧参数；
3. 损坏时显示 `SETTINGS ERROR` 并继续；
4. 每帧用 `settings["roi"]` 和合并配置调用 `detect_pieces`。

**Step 2: 接入 CAL 点击动作**

正常界面点击 `cal` 创建 `CalibrationSession`；调参界面点击同一矩形的 `cal` 丢弃会话并返回正常界面。`KNOWN/UNKNOWN` 变量不参与该切换。

**Step 3: 接入调参动作**

调参界面处理：

- `roi/mask/result`：切换页面；
- `item`：循环参数；
- `minus/plus`：调整参数；
- `value`：AUTO/固定阈值切换；
- `step`：循环步长；
- `save_settings`：仅在 GOOD 时原子保存，并用快照替换运行参数。

**Step 4: 接入调参显示**

RUN 使用现有 `draw_overlay`，CAL 使用 `draw_calibration_frame`。两者都通过同一个 `disp.show(..., FIT_CONTAIN)` 输出。

**Step 5: 运行入口测试**

Run: `python -m pytest tests/test_main_import_guard.py -v`

Expected: 全部 PASS。

### Task 9: 更新使用文档

**Files:**

- Modify: `docs/maixcam2-puzzle-recognition-guide.md`
- Modify: `README.md`

文档必须说明：

- CAL 第一次进入、第二次退出；
- ROI 优先于阈值调整；
- ROI/MASK/RESULT 三个页面；
- GOOD/MISS/NOISE/EDGE/BACKGROUND 含义；
- 四片校准摆放要求；
- 未保存退出恢复旧参数；
- 持久参数路径和实机持久化验证状态。

### Task 10: 最终验证

**Step 1: 运行全套测试**

Run: `python -m pytest -v`

Expected: 全部 PASS，失败数为 0。

**Step 2: 编译全部 Python 文件**

Run: `python -m compileall maixcam2_app tests tools`

Expected: exit code 0，无 `SyntaxError`。

**Step 3: 检查部署文件**

Run: `Get-ChildItem maixcam2_app -File | Select-Object Name,Length`

Expected: 至少包含 `main.py`、`config.py`、`puzzle_vision.py`、`template_store.py`、`touch_ui.py`、`settings_store.py` 和 `calibration_ui.py`。

**Step 4: 实机审查门**

在 MaixVision 打开整个 `maixcam2_app` 目录并运行 `main.py`，验证：

1. 正常界面点击 CAL 进入调参；
2. 再次点击同一个 CAL 返回正常界面；
3. 固定 ROI 排除亮色地面后四片可见；
4. MASK 和 RESULT 能解释漏检原因；
5. GOOD 4/4 时能够保存；
6. 重启后恢复已保存参数。

实机日志或画面未提供前，不声明 MaixCAM2 端验证通过。

