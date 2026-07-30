# MaixCAM2 Auto Paper ROI A/B Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不修改当前稳定版的前提下，交付可独立部署的四边形掩膜 A 版和透视展开 B 版，并提供相同实拍帧的 PC 对比工具。

**Architecture:** 从 `maixcam2_app` 机械复制两个独立包，两个包各自携带相同的黑纸定位与物理坐标模块。A 版在原图中使用四边形掩膜，B 版将完整 A4 展开为 420×594 并截取 420×460 工作区；两版共用相同触摸交互，但使用独立应用 ID、参数路径和发布包。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、MaixPy `camera/image/display/touchscreen/app`

---

## 执行规则

| 规则 | 要求 |
|---|---|
| 测试方法 | 每个行为先按 `@test-driven-development` 写失败测试并确认失败原因，再写最小实现 |
| 调试方法 | 非预期失败按 `@systematic-debugging` 收集证据，不连续叠加猜测修复 |
| 完成门 | 最终按 `@verification-before-completion` 重新运行全套命令 |
| 嵌入式约束 | 按 `@embedded-dev` 更新四文件记录、发布清单和实机待验证项 |
| 注释 | 新函数、类、关键变量、状态切换和异常回退必须有中文注释或中文文档字符串 |
| 稳定版 | `maixcam2_app/` 不得修改；若测试发现稳定版问题，单独报告，不顺带修复 |
| Git | 当前目录不是有效 Git 仓库，不能创建 worktree 或提交；每个提交检查点改为更新 `编辑清单.md` |

## Round 1：工程隔离与黑纸定位核心

### Task 1：建立 A/B 独立工程骨架

**Files:**
- Create: `maixcam2_app_A_quad/`
- Create: `maixcam2_app_B_warp/`
- Create: `tests_ab/test_variant_isolation.py`
- Create: `docs/plans/2026-07-29-maixcam2-stable-baseline-sha256.md`
- Modify: `maixcam2_app_A_quad/config.py`
- Modify: `maixcam2_app_B_warp/config.py`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `maixcam2_app_B_warp/app.yaml`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_B_warp/main.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_B_warp/calibration_ui.py`

**Step 1: 记录稳定版SHA256基线**

Run: `Get-ChildItem maixcam2_app -File | Where-Object { $_.Extension -in '.py','.yaml' } | Get-FileHash -Algorithm SHA256`

Expected: 输出稳定版全部 Python/YAML 文件的 SHA256。使用 `apply_patch` 把路径和哈希写入 `docs/plans/2026-07-29-maixcam2-stable-baseline-sha256.md`，供最终逐项比较。

**Step 2: 写工程隔离失败测试**

```python
def test_variant_packages_use_independent_setting_paths():
    from maixcam2_app_A_quad.config import PERSISTENT_SETTINGS_PATH as path_a
    from maixcam2_app_B_warp.config import PERSISTENT_SETTINGS_PATH as path_b

    assert path_a == "/root/maixcam2_puzzle_A/vision_settings.json"
    assert path_b == "/root/maixcam2_puzzle_B/vision_settings.json"
    assert path_a != path_b


def test_variant_sources_do_not_import_stable_package():
    project_root = Path(__file__).resolve().parents[1]
    for package_name in ("maixcam2_app_A_quad", "maixcam2_app_B_warp"):
        for source_path in (project_root / package_name).glob("*.py"):
            assert "from maixcam2_app." not in source_path.read_text(encoding="utf-8")
```

**Step 3: 运行测试并确认失败**

Run: `python -m pytest tests_ab/test_variant_isolation.py -v`

Expected: FAIL，因为两个变体目录尚不存在。

**Step 4: 机械复制稳定源码**

复制 `__init__.py`、`app.yaml`、`config.py`、`main.py`、`puzzle_vision.py`、`settings_store.py`、`template_store.py`、`touch_ui.py`、`calibration_ui.py` 到两个新目录。不得复制 `__pycache__`、`dist` 和设备生成的 JSON。

**Step 5: 修改独立标识和导入**

```python
# A 版配置
PERSISTENT_SETTINGS_PATH = "/root/maixcam2_puzzle_A/vision_settings.json"

# B 版配置
PERSISTENT_SETTINGS_PATH = "/root/maixcam2_puzzle_B/vision_settings.json"
```

`app.yaml` 分别使用 `id/name: diansai_quad` 和 `id/name: diansai_warp`。包内导入分别指向自己的包名，并保留 MaixVision 平铺部署的同级模块回退。

**Step 6: 运行隔离测试并记录检查点**

Run: `python -m pytest tests_ab/test_variant_isolation.py -v`

Expected: PASS。在 `编辑清单.md` 记录工程复制、验证命令和“无有效 Git 仓库，未提交”。

### Task 2：实现共用黑纸候选定位

**Files:**
- Create: `maixcam2_app_A_quad/paper_locator.py`
- Create: `maixcam2_app_B_warp/paper_locator.py`
- Create: `tests_ab/__init__.py`
- Create: `tests_ab/synthetic_paper.py`
- Create: `tests_ab/test_paper_locator.py`
- Modify: `maixcam2_app_A_quad/config.py`
- Modify: `maixcam2_app_B_warp/config.py`

**Step 1: 创建合成场景工具**

`tests_ab/synthetic_paper.py` 必须提供 `make_paper_scene`、`make_scene_with_piece_count`、`make_quad_scene_with_four_pieces`、`make_axis_aligned_a4_scene`、`make_perspective_scene_with_four_pieces` 和 `write_synthetic_a4_frame`，后续测试只从此文件复用场景。

```python
def make_paper_scene(paper_quad, white_pieces=(), dark_objects=(), size=(640, 480)):
    """生成亮地面、黑色A4、白色碎片和外部暗色干扰物组成的测试图。"""
    width, height = size
    image = np.full((height, width, 3), 210, dtype=np.uint8)
    cv2.fillConvexPoly(image, np.asarray(paper_quad, dtype=np.int32), (20, 20, 20))
    for polygon in white_pieces:
        cv2.fillConvexPoly(image, np.asarray(polygon, dtype=np.int32), (245, 245, 245))
    for polygon in dark_objects:
        cv2.fillConvexPoly(image, np.asarray(polygon, dtype=np.int32), (15, 15, 15))
    return image
```

**Step 2: 写定位成功与干扰拒绝失败测试**

```python
@pytest.mark.parametrize("piece_count", [1, 2, 3, 4])
def test_locator_finds_a4_with_white_pieces_and_dark_rod(piece_count):
    expected_quad = np.float32([[220, 70], [390, 80], [410, 330], [205, 325]])
    scene = make_scene_with_piece_count(expected_quad, piece_count, add_dark_rod=True)

    result = locate_black_paper(scene)

    assert result.success is True
    assert result.confidence >= 0.65
    np.testing.assert_allclose(result.paper_quad, expected_quad, atol=12)


def test_locator_rejects_scene_without_a4():
    result = locate_black_paper(np.full((480, 640, 3), 210, dtype=np.uint8))
    assert result.success is False
    assert result.paper_quad is None
```

**Step 3: 运行失败测试**

Run: `python -m pytest tests_ab/test_paper_locator.py -k "locator" -v`

Expected: FAIL，提示 `paper_locator` 或 `locate_black_paper` 不存在。

**Step 4: 实现最小定位模块**

模块提供 `PaperLocation`、`order_a4_quad(points)` 和 `locate_black_paper(frame_bgr, config=None)`。默认配置为：

```python
"paper_min_area_ratio": 0.01,
"paper_max_area_ratio": 0.50,
"paper_expected_aspect": 210.0 / 297.0,
"paper_min_rectangularity": 0.70,
"paper_min_confidence": 0.65,
"paper_close_kernel": 9,
```

流程固定为灰度、反向 Otsu、9×9 闭运算、外轮廓、凸包、四边形拟合与评分。总分由长宽比、矩形度、凸性、面积合理性和内部暗度组成，所有权重在实现旁用中文注释说明。

**Step 5: 复制定位实现并验证一致性**

A 版通过后把相同 `paper_locator.py` 复制到 B 版。测试参数化导入两个模块，确保同一场景返回等价四角和置信度。

Run: `python -m pytest tests_ab/test_paper_locator.py -v`

Expected: PASS。

### Task 3：实现物理有效区与毫米映射

**Files:**
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Modify: `maixcam2_app_B_warp/paper_locator.py`
- Modify: `tests_ab/test_paper_locator.py`

**Step 1: 写33.5mm裁剪与INSET失败测试**

```python
def test_active_quad_crops_long_edges_and_applies_inset_mm():
    paper_quad = np.float32([[100, 20], [310, 20], [310, 317], [100, 317]])
    active_quad = build_active_quad(paper_quad, inset_mm=2.0)
    expected = np.float32([[102, 55.5], [308, 55.5], [308, 281.5], [102, 281.5]])
    np.testing.assert_allclose(active_quad, expected, atol=1.0)
```

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_paper_locator.py::test_active_quad_crops_long_edges_and_applies_inset_mm -v`

Expected: FAIL，因为 `build_active_quad` 不存在。

**Step 3: 实现物理映射函数**

```python
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
WORK_HEIGHT_MM = 230.0
WORK_TRIM_MM = (A4_HEIGHT_MM - WORK_HEIGHT_MM) / 2.0
```

实现 `build_active_quad(paper_quad, inset_mm)` 和 `image_point_to_paper_mm(point, paper_quad)`。`inset_mm` 限制为0～20mm；超范围、非凸四边形和奇异矩阵抛出 `ValueError`。

**Step 4: 运行测试**

Run: `python -m pytest tests_ab/test_paper_locator.py -v`

Expected: PASS。

### Task 4：扩展独立设置格式

**Files:**
- Modify: `maixcam2_app_A_quad/settings_store.py`
- Modify: `maixcam2_app_B_warp/settings_store.py`
- Create: `tests_ab/test_variant_settings.py`

**Step 1: 写四角与INSET持久化失败测试**

```python
@pytest.mark.parametrize("module_name", [
    "maixcam2_app_A_quad.settings_store",
    "maixcam2_app_B_warp.settings_store",
])
def test_settings_round_trip_paper_quad_and_inset(module_name, tmp_path):
    module = importlib.import_module(module_name)
    settings = module.build_default_runtime_settings(DEFAULT_CONFIG)
    settings["paper_quad"] = [[220, 70], [390, 80], [410, 330], [205, 325]]
    settings["inset_mm"] = 2.0
    module.save_runtime_settings(tmp_path / "settings.json", settings, (640, 480))
    loaded = module.load_runtime_settings(tmp_path / "settings.json", DEFAULT_CONFIG)
    assert loaded["paper_quad"] == settings["paper_quad"]
    assert loaded["inset_mm"] == 2.0
```

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_variant_settings.py -v`

Expected: FAIL，因为旧设置格式没有 `paper_quad` 和 `inset_mm`。

**Step 3: 实现设置版本2和分组合并**

- `SETTINGS_VERSION = 2`；
- `paper_quad` 允许 `None` 或4×2有界凸四边形；
- `inset_mm` 允许0～20；
- 保留 `roi` 作为兼容矩形；
- `active_quad` 不落盘；
- 提供 `merge_paper_settings`，只合并 `paper_quad/inset_mm`；
- 提供 `merge_segmentation_settings`，只合并 `fixed_threshold/min_area_ratio/open_kernel/close_kernel`。

**Step 4: 运行设置测试并记录Round 1**

Run: `python -m pytest tests_ab/test_variant_settings.py tests_ab/test_paper_locator.py -v`

Expected: PASS。更新 `编辑清单.md` 记录 Round 1 结果。

## Round 2：方案 A 四边形掩膜

### Task 5：让 A 版视觉核心支持四边形工作区

**Files:**
- Modify: `maixcam2_app_A_quad/puzzle_vision.py`
- Create: `tests_ab/test_quad_vision.py`

**Step 1: 写纸外亮背景隔离失败测试**

```python
def test_quad_mask_excludes_bright_floor_and_keeps_four_pieces():
    frame, paper_quad, active_quad = make_quad_scene_with_four_pieces()
    result = detect_pieces(
        frame,
        roi=cv2.boundingRect(active_quad.astype(np.int32)),
        config=DEFAULT_CONFIG,
        active_quad=active_quad,
    )
    assert len(result.pieces) == 4
    assert result.large_contours == []
    assert result.white_ratio < 0.30
```

同一测试文件增加：轮廓接触斜边时 `complete=False`；面积上下限使用 `cv2.contourArea(active_quad)`。

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_quad_vision.py -v`

Expected: FAIL，因为 `detect_pieces` 不接受 `active_quad`。

**Step 3: 扩展前景掩膜、面积和边缘判断**

- `build_foreground_mask` 和 `detect_pieces` 增加可选 `active_quad=None`；
- `None` 保持旧矩形行为；
- 有四边形时按外接矩形裁图，把全局四角转换为局部坐标；
- 开闭运算完成后用 `cv2.fillConvexPoly` 与 `cv2.bitwise_and` 清零四边形外部；
- `white_ratio` 分母使用有效掩膜像素数；
- 面积上下限分母使用 `cv2.contourArea(active_quad)`；
- 使用 `cv2.pointPolygonTest` 判断轮廓点到四边形边缘的距离；
- 输出轮廓继续换算回原相机坐标。

**Step 4: 运行 A 版视觉测试**

Run: `python -m pytest tests_ab/test_quad_vision.py -v`

Expected: PASS。

### Task 6：接入 A 版正常识别与毫米中心

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Create: `tests_ab/test_quad_main.py`

**Step 1: 写运行数据流失败测试**

```python
def test_quad_runtime_builds_active_quad_from_locked_settings():
    settings = make_locked_settings(inset_mm=2.0)
    active_quad = build_runtime_active_quad(settings)
    assert active_quad.shape == (4, 2)
    assert cv2.contourArea(active_quad) > 0
```

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_quad_main.py -v`

Expected: FAIL，因为入口尚未构造四边形运行区域。

**Step 3: 接入主循环和叠加**

- 有锁定 `paper_quad` 时计算 `active_quad` 并传给 `detect_pieces`；
- 无 `paper_quad` 时使用兼容矩形 `roi` 并显示 `ROI NOT SET`；
- 正常叠加绘制青色完整 A4 和黄色有效四边形；
- 使用 `image_point_to_paper_mm` 为完整碎片增加 `center_mm`；
- 检测和绘制继续使用原相机像素坐标。

**Step 4: 运行入口测试并记录Round 2**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_quad_vision.py -v`

Expected: PASS。更新 `编辑清单.md` 记录 A 版结果与实机待验证项。

## Round 3：方案 B 透视展开

### Task 7：实现标准 A4 展开与毫米坐标

**Files:**
- Create: `maixcam2_app_B_warp/paper_warp.py`
- Create: `tests_ab/test_paper_warp.py`

**Step 1: 写尺寸与坐标失败测试**

```python
def test_warp_returns_420_by_460_work_area_at_two_pixels_per_mm():
    frame, paper_quad = make_axis_aligned_a4_scene()
    result = warp_to_work_area(frame, paper_quad, inset_mm=2.0)
    assert result.full_a4.shape[:2] == (594, 420)
    assert result.work_area.shape[:2] == (460, 420)
    assert result.valid_mask.shape == (460, 420)
    assert pixels_to_work_mm((200.0, 100.0)) == pytest.approx((100.0, 50.0))
```

同一测试文件增加透视四边形拉正和奇异四边形抛出 `ValueError` 测试。

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_paper_warp.py -v`

Expected: FAIL，因为 `paper_warp` 不存在。

**Step 3: 实现展开模块**

```python
PIXELS_PER_MM = 2.0
A4_SIZE_PX = (420, 594)
WORK_SIZE_PX = (420, 460)
WORK_TOP_PX = 67
WORK_BOTTOM_PX = 527
```

实现 `build_a4_homography(paper_quad)`、`warp_to_work_area(frame_bgr, paper_quad, inset_mm)` 和 `pixels_to_work_mm(point)`，全部写中文函数文档。展开使用 `cv2.getPerspectiveTransform` 和 `cv2.warpPerspective`，INSET 在420×460固定平面中生成边界掩膜。

**Step 4: 运行展开测试**

Run: `python -m pytest tests_ab/test_paper_warp.py -v`

Expected: PASS。

### Task 8：接入 B 版识别与显示

**Files:**
- Modify: `maixcam2_app_B_warp/puzzle_vision.py`
- Modify: `maixcam2_app_B_warp/main.py`
- Modify: `maixcam2_app_B_warp/calibration_ui.py`
- Create: `tests_ab/test_warp_main.py`

**Step 1: 写展开识别失败测试**

```python
def test_warp_variant_detects_four_pieces_and_reports_mm_centers():
    frame, paper_quad = make_perspective_scene_with_four_pieces()
    result = analyze_warped_frame(frame, paper_quad, inset_mm=2.0)
    assert result.work_frame.shape[:2] == (460, 420)
    assert len(result.detection.pieces) == 4
    assert all("center_mm" in piece for piece in result.detection.pieces)
```

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_warp_main.py -v`

Expected: FAIL，因为 B 版入口尚未调用透视展开。

**Step 3: 接入B版数据流**

- 有 `paper_quad` 时先调用 `warp_to_work_area`；
- 对420×460工作图使用固定矩形 ROI `(0, 0, 420, 460)`；
- 用 `valid_mask` 清除 INSET 边缘；
- 为每片写入 `center_mm=(center_x/2, center_y/2)`；
- 无锁定四角时显示原相机画面和 `ROI NOT SET`；
- 单应性异常时显示 `WARP ERROR <类型>`，主循环继续。

**Step 4: 构造640×480显示画布**

420×460工作图按210:230比例缩放到高度480、宽度约438，并居中放入640×480黑色画布。按钮按640×480画布布局，避免五个顶部按钮被压缩到420像素内。

**Step 5: 运行 B 版入口测试并记录Round 3**

Run: `python -m pytest tests_ab/test_warp_main.py tests_ab/test_paper_warp.py -v`

Expected: PASS。更新 `编辑清单.md` 记录 B 版结果与单应性实机待验证项。

## Round 4：自动锁定交互、现场调参、对比与发布

### Task 9：实现简化页与 ADV 参数页布局

**Files:**
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_B_warp/calibration_ui.py`
- Create: `tests_ab/test_variant_calibration_ui.py`

**Step 1: 写五页导航和两级参数布局失败测试**

```python
@pytest.mark.parametrize("module_name", [
    "maixcam2_app_A_quad.calibration_ui",
    "maixcam2_app_B_warp.calibration_ui",
])
def test_calibration_navigation_has_five_pages_and_simple_roi_controls(module_name):
    module = importlib.import_module(module_name)
    controller = module.CalibrationController(DEFAULT_CONFIG)

    assert controller.page_names == ("ROI", "MASK", "RESULT", "ADV")
    assert [button.action for button in controller.bottom_buttons()] == [
        "auto_roi", "inset_dec", "select_inset", "inset_inc", "lock_roi"
    ]
    controller.handle_action("select_page", "ADV")
    assert [button.action for button in controller.bottom_buttons()] == [
        "prev_param", "value_dec", "select_param", "value_inc", "save_segmentation"
    ]
```

同一测试文件还要验证：顶部固定为 `ROI/MASK/RESULT/ADV/CAL` 五个触摸区；`CAL` 负责退出调参；默认 ROI 页只展示 `AUTO ROI/-/INSET/+/LOCK ROI`；ADV 页逐项选择 `THRESH/MIN AREA/OPEN/CLOSE`，不能让四组参数同时挤在一行。

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_variant_calibration_ui.py -k "navigation or layout" -v`

Expected: FAIL，因为变体仍是稳定版的三页调参布局。

**Step 3: 实现固定尺寸触摸布局**

- 顶部导航使用五个等宽区域，显示 `ROI/MASK/RESULT/ADV/CAL`；
- 中间预览区保持固定尺寸，不因状态文字或按钮标签改变而跳动；
- ROI 页底部使用五个固定槽位，中央显示当前 `INSET=<数值>mm`；
- ADV 页底部使用五个固定槽位，中央显示当前参数和值；
- `MASK` 页显示分割二值图，`RESULT` 页显示轮廓分类，`ROI` 页显示完整 A4 和有效区，`ADV` 页在结果图上叠加高级参数；
- 所有按钮命中区域、状态文字和页面切换写中文注释，说明触摸坐标与640×480显示画布的关系。

**Step 4: 运行布局测试**

Run: `python -m pytest tests_ab/test_variant_calibration_ui.py -k "navigation or layout" -v`

Expected: PASS。

### Task 10：实现 AUTO ROI、LOCK ROI 与 ADV SAVE 状态机

**Files:**
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_B_warp/calibration_ui.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_B_warp/main.py`
- Modify: `tests_ab/test_variant_calibration_ui.py`

**Step 1: 写成功、失败保留和保存门槛失败测试**

```python
@pytest.mark.parametrize("module_name", [
    "maixcam2_app_A_quad.calibration_ui",
    "maixcam2_app_B_warp.calibration_ui",
])
def test_auto_roi_failure_preserves_last_saved_quad(module_name):
    module = importlib.import_module(module_name)
    old_quad = [[220, 70], [390, 80], [410, 330], [205, 325]]
    controller = module.CalibrationController(DEFAULT_CONFIG, {"paper_quad": old_quad})

    controller.apply_auto_roi(PaperLocation.failed("no_candidate"))

    assert controller.pending_settings["paper_quad"] == old_quad
    assert controller.status_text == "AUTO ROI FAIL"


@pytest.mark.parametrize("piece_count", [1, 2, 3, 4])
def test_lock_roi_accepts_one_to_four_complete_pieces(piece_count):
    controller = make_controller_with_candidate_and_piece_count(piece_count)
    assert controller.can_lock_roi() is True


def test_advanced_save_still_requires_good_four_of_four():
    controller = make_controller_with_candidate_and_piece_count(3)
    controller.handle_action("select_page", "ADV")
    assert controller.can_save_segmentation() is False
    assert controller.status_text == "NEED GOOD 4/4"
```

同一测试文件增加：AUTO ROI 成功只更新待确认四角、不立即写盘；再次按 AUTO ROI 可以重试；`LOCK ROI` 后只保存 `paper_quad/inset_mm`；ADV 保存只保存阈值、最小面积比例和开闭运算核；候选碎片接触纸张外边缘时禁止锁定。

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_variant_calibration_ui.py -k "auto_roi or lock_roi or advanced_save" -v`

Expected: FAIL，因为自动定位和分组保存状态机尚未接入。

**Step 3: 实现单次定位与人工锁定**

- `AUTO ROI` 只在点击时对当前帧调用一次 `locate_black_paper`，不逐帧跟踪；
- 成功后保存为内存候选四角，状态显示 `AUTO ROI OK <置信度>`；
- 失败显示 `AUTO ROI FAIL`，不得把 `None` 或失败候选覆盖到旧参数；
- `INSET` 以0.5mm步进，限制0～20mm，并实时重算有效四边形；
- `LOCK ROI` 验证候选四边形有效、碎片数为1～4且完整，然后只合并并持久化纸张参数；
- ADV `SAVE` 仅在诊断为 `GOOD 4/4` 时合并并持久化分割参数；
- 所有成功、失败、回退和持久化分支必须写中文注释，说明为何保留旧设置。

**Step 4: 把状态机接入两个主循环**

主循环把当前相机帧交给 `AUTO ROI` 点击处理，把当前诊断结果交给 `LOCK ROI` 和 ADV 保存门槛。保存成功后立即更新运行时设置；保存失败或写盘异常时继续使用旧设置并显示明确状态，不退出相机循环。

**Step 5: 运行状态机与入口测试**

Run: `python -m pytest tests_ab/test_variant_calibration_ui.py tests_ab/test_quad_main.py tests_ab/test_warp_main.py -v`

Expected: PASS。更新 `编辑清单.md` 记录现场状态机验证结果。

### Task 11：提供同一实拍帧的 A/B 对比工具

**Files:**
- Create: `tools/compare_roi_variants.py`
- Create: `tests_ab/test_compare_roi_variants.py`

**Step 1: 写同帧输出失败测试**

```python
def test_compare_tool_writes_quad_warp_and_side_by_side_images(tmp_path):
    frame_path, paper_quad = write_synthetic_a4_frame(tmp_path)
    outputs = compare_frame(frame_path, tmp_path / "compare", paper_quad, inset_mm=2.0)

    assert outputs["quad"].is_file()
    assert outputs["warp"].is_file()
    assert outputs["side_by_side"].is_file()
    image = cv2.imread(str(outputs["side_by_side"]))
    assert image is not None and image.shape[0] == 480
```

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_compare_roi_variants.py -v`

Expected: FAIL，因为对比工具尚不存在。

**Step 3: 实现对比函数和命令行入口**

`compare_frame(image_path, output_dir, paper_quad=None, inset_mm=0.0)` 在未提供四角时自动定位一次，分别调用 A 版四边形掩膜与 B 版透视分析，输出：

- `<stem>_A_quad.jpg`：原图、完整A4、有效四边形和A版轮廓；
- `<stem>_B_warp.jpg`：420×460展开工作图和B版轮廓；
- `<stem>_AB.jpg`：统一为640×480后的左右对照图，并写明两版碎片数和状态。

命令行参数固定为 `--image`、`--output-dir`、可选 `--quad x1 y1 x2 y2 x3 y3 x4 y4` 和 `--inset-mm`。输入不存在、自动定位失败和图片解码失败时返回非零退出码与明确错误文字。

**Step 4: 运行工具测试**

Run: `python -m pytest tests_ab/test_compare_roi_variants.py -v`

Expected: PASS。

### Task 12：生成两个独立应用包并补充操作文档

**Files:**
- Modify: `README.md`
- Modify: `docs/maixcam2-puzzle-recognition-guide.md`
- Create: `docs/maixcam2-auto-roi-ab-guide.md`
- Create: `tools/package_variants.py`
- Create: `tests_ab/test_variant_packages.py`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.1.0.zip`
- Create: `maixcam2_app_B_warp/dist/diansai_warp-v1.1.0.zip`

**Step 1: 写发布物失败测试**

测试必须检查两个 `app.yaml` 的ID不同、ZIP文件名不同、ZIP内不含顶层包目录、运行模块齐全、A包不含 `paper_warp.py`、B包包含 `paper_warp.py`，并确认平铺解压后的 `main.py` 不依赖稳定包路径。

**Step 2: 运行失败测试**

Run: `python -m pytest tests_ab/test_variant_packages.py -v`

Expected: FAIL，因为独立发布 ZIP 和打包脚本尚未生成。

**Step 3: 实现可重复打包脚本**

`tools/package_variants.py` 使用显式模块白名单生成两个ZIP，拒绝加入 `__pycache__`、测试文件、设置JSON和旧 `dist` 内容。脚本先校验各自 `app.yaml` 文件清单，再把文件平铺写入ZIP根目录，保证 MaixVision 直接运行时同级导入成立。

**Step 4: 编写现场操作文档**

文档按实际屏幕流程说明：进入 `CAL`、`AUTO ROI`、看青色完整A4与黄色210×230mm有效区、调 `INSET`、在1～4片完整碎片时 `LOCK ROI`、进入 `ADV` 用 `MASK/RESULT` 判断阈值、仅 `GOOD 4/4` 保存、再次按 `CAL` 返回识别。增加 `AUTO ROI FAIL` 保留旧ROI、A/B适用差异、现场光照变化调整顺序和同帧对比命令。

**Step 5: 生成发布物并运行测试**

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -v`

Expected: 两个命令退出码均为0，发布物测试PASS。

### Task 13：全量回归、稳定版哈希核对与四文件收尾

**Files:**
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `硬件资源表.md`
- Modify: `研究发现.md`
- Modify: `docs/plans/2026-07-29-maixcam2-stable-baseline-sha256.md`

**Step 1: 运行稳定版回归**

Run: `python -m pytest tests -v`

Expected: 原有65项测试全部PASS。

**Step 2: 运行A/B全量测试**

Run: `python -m pytest tests_ab -v`

Expected: 全部PASS且无warning。

**Step 3: 检查Python语法和两种导入方式**

Run: `python -m compileall -q maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python -c "import maixcam2_app_A_quad.main, maixcam2_app_B_warp.main"`

Expected: 两个命令退出码均为0。另将两个ZIP分别解压到独立临时目录，以平铺模块方式导入 `main.py`，确认没有 `ModuleNotFoundError`。

**Step 4: 逐项核对稳定版SHA256**

Run: `Get-ChildItem maixcam2_app -File | Where-Object { $_.Extension -in '.py','.yaml' } | Get-FileHash -Algorithm SHA256`

Expected: 每项与Task 1基线完全一致；把最终比对结果和时间写回基线文档。任何一项不一致都视为失败，先定位差异，不覆盖基线。

**Step 5: 对照设计文档逐项审查**

确认：单次自动定位、失败保留旧ROI、1～4片可锁定、33.5mm上下裁剪、毫米INSET、A版四边形掩膜、B版420×594展开及420×460工作区、五页界面、ADV `GOOD 4/4`保存门槛、独立设置路径、独立应用ID与ZIP均有对应测试或文件证据。

**Step 6: 更新四文件记录**

四文件必须记录 `trace_id=maixcam2-auto-roi-ab-20260729`、四轮目标、实际验证命令、测试数量、稳定版哈希结论、两个ZIP路径和以下实机待验证项：相机曝光下的自动定位置信度、龙门架遮挡但不接触纸边时的候选稳定性、触摸命中、帧率、保存后重启恢复。

## RIPER-5 实施清单

| 序号 | 操作 | 验证标准 | review |
|---:|---|---|:---:|
| 1 | 在 `maixcam2_app_A_quad/`、`maixcam2_app_B_warp/` 和 `tests_ab/` 建立隔离骨架、自动黑纸定位、物理有效区与独立设置格式 | Round 1 定向测试全部通过，稳定版SHA256基线已记录 | true |
| 2 | 在 `maixcam2_app_A_quad/` 接入四边形掩膜、有效面积、边缘诊断与毫米中心 | Round 2 定向测试全部通过，纸外亮背景不进入碎片候选 | true |
| 3 | 在 `maixcam2_app_B_warp/` 接入420×594透视展开、420×460工作区、毫米中心与640×480显示画布 | Round 3 定向测试全部通过，透视场景检出4片 | true |
| 4 | 在两个变体接入五页调参、单次AUTO ROI、INSET、LOCK ROI、ADV保存，并交付同帧对比工具、独立ZIP和文档 | 全量测试、compileall、包/平铺导入和稳定版SHA256比对全部通过 | true |

四轮均采用单代理串行执行。因为当前目录不是有效Git仓库，`review:true` 的存档动作替换为更新 `编辑清单.md` 和稳定版SHA256证据；用户已明确要求当前会话连续实施四轮，因此轮间只汇报证据，不等待额外确认，除非触发新增依赖、稳定版哈希变化或无法自行消除的实机阻塞。
