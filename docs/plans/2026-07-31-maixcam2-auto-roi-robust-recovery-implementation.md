# MaixCAM2 AUTO ROI 鲁棒恢复 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为A版AUTO ROI增加多阈值主轮廓恢复和旧ROI边缘兜底，同时阻止宽松路径锁定小暗块。

**Architecture:** 保留现有单次Otsu严格路径并优先返回；严格失败后扫描有限阈值，使用面积关系、A4比例、暗度和四边边缘支持联合验收。若仍失败且存在已保存ROI，只在旧四边附近修正边缘；任何宽松候选不满足安全门时继续返回失败并保留旧ROI。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、MaixPy运行时兼容API、可重复ZIP打包脚本。

---

### Task 1: 固化实机失败模式和配置契约

**Files:**
- Modify: `tests_ab/synthetic_paper.py`
- Modify: `tests_ab/test_paper_locator.py`
- Modify: `maixcam2_app_A_quad/config.py`

**Step 1: Write the failing tests**

在`tests_ab/synthetic_paper.py`增加带中文函数注释的两个场景生成器：

```python
def make_uneven_brightness_paper_scene():
    """生成左右亮度不均、严格Otsu只留下残缺主轮廓的A4场景。

    返回值为``(frame, expected_quad)``；纸张右半区使用接近背景但仍有可见边缘的
    灰度，模拟实机反光导致的``rect≈0.39``和``dark≈0.49``。
    """
    paper_quad = np.float32([[115, 85], [535, 72], [550, 390], [100, 405]])
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    paper_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(paper_mask, paper_quad.astype(np.int32), 255)
    x_gradient = np.linspace(28, 158, frame.shape[1], dtype=np.float32)
    paper_gray = np.tile(x_gradient, (frame.shape[0], 1))
    for channel in range(3):
        frame[:, :, channel][paper_mask > 0] = paper_gray[paper_mask > 0].astype(np.uint8)
    return frame, paper_quad


def make_large_distractor_with_small_a4_like_block():
    """生成大型非四边暗区和1.2%左右A4比例小框，复现宽松路径误锁。"""
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    large_triangle = np.int32([[40, 420], [320, 55], [420, 430]])
    small_block = np.int32([[525, 45], [600, 45], [600, 98], [525, 98]])
    cv2.fillConvexPoly(frame, large_triangle, (22, 22, 22))
    cv2.fillConvexPoly(frame, small_block, (18, 18, 18))
    return frame, small_block.astype(np.float32)
```

在`tests_ab/test_paper_locator.py`增加：

```python
def test_a_locator_recovers_uneven_brightness_paper_with_threshold_scan():
    """严格轮廓残缺时，多阈值路径必须恢复完整A4而不是降低全局硬门。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    frame, expected_quad = make_uneven_brightness_paper_scene()

    result = module.locate_black_paper(frame)

    assert result.success is True
    assert result.diagnostics["source"] == "TH_SCAN"
    np.testing.assert_allclose(result.paper_quad, expected_quad, atol=12.0)


def test_a_locator_rejects_small_quad_beside_larger_nonpaper_contour():
    """宽松候选远小于本帧主轮廓时必须失败，不能重现1.2%小框误锁。"""
    module = importlib.import_module("maixcam2_app_A_quad.paper_locator")
    frame, small_block = make_large_distractor_with_small_a4_like_block()

    result = module.locate_black_paper(frame)

    assert result.success is False
    assert result.paper_quad is None
    best = result.diagnostics["best_candidate"]
    assert best["area_to_largest"] < 0.10
```

增加配置宏导出测试，明确所有建议初值，并断言`DEFAULT_CONFIG`引用这些宏。

**Step 2: Run tests to verify they fail**

Run:

```powershell
pytest -q tests_ab/test_paper_locator.py -k "uneven_brightness or small_quad_beside"
```

Expected: FAIL；现有实现没有`source`、`area_to_largest`且会拒绝亮度不均纸面或接受小框。

**Step 3: Add the configuration contract**

在`config.py`顶部增加带中文用途说明的宏，并映射到`DEFAULT_CONFIG`：

```python
PAPER_AUTO_THRESHOLD_OFFSETS = (-24, -12, 0, 12, 24)
PAPER_AUTO_RELAXED_MIN_AREA_RATIO = 0.08
PAPER_AUTO_MIN_AREA_TO_LARGEST = 0.55
PAPER_AUTO_RELAXED_MIN_RECTANGULARITY = 0.35
PAPER_AUTO_RELAXED_MIN_ASPECT_SCORE = 0.70
PAPER_AUTO_RELAXED_MIN_DARKNESS = 0.42
PAPER_AUTO_MIN_EDGE_SUPPORT = 0.55
PAPER_AUTO_MIN_SUPPORTED_SIDES = 3
PAPER_AUTO_MAX_CONTOURS_PER_MASK = 4
PAPER_AUTO_PRIOR_MIN_IOU = 0.55
PAPER_AUTO_PRIOR_MAX_SHIFT_RATIO = 0.04
```

此步骤只建立配置和测试契约，不让生产测试提前变绿。

**Step 4: Run the focused configuration test**

Run: `pytest -q tests_ab/test_paper_locator.py -k "exposes_editable"`

Expected: PASS。

**Step 5: Commit**

```powershell
git add -- maixcam2_app_A_quad/config.py tests_ab/synthetic_paper.py tests_ab/test_paper_locator.py
git commit -m "test: cover robust auto roi recovery"
```

### Task 2: 提取严格路径并增加有限多阈值扫描

**Files:**
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Test: `tests_ab/test_paper_locator.py`

**Step 1: Add validation tests**

参数化测试空阈值序列、非有限偏移、超过`[-127, 127]`的偏移、无效面积关系和轮廓数量，
均须由`locate_black_paper()`抛出包含配置键名的`ValueError`。

**Step 2: Run validation tests to verify they fail**

Run: `pytest -q tests_ab/test_paper_locator.py -k "threshold_offsets or relaxed_gate_config"`

Expected: FAIL；当前尚未校验新配置。

**Step 3: Implement mask construction and candidate enumeration**

在`paper_locator.py`增加以下职责单一的函数，每个函数写完整中文注释：

```python
def _validate_threshold_offsets(raw_offsets):
    """校验有限整数偏移序列，去重后保留用户顺序并确保包含0。"""


def _build_dark_mask(blurred_gray, threshold, close_kernel):
    """按显式阈值生成反向暗色掩膜并执行一次闭运算。"""


def _find_mask_contours(dark_mask):
    """返回外轮廓、各轮廓面积和本掩膜最大轮廓面积，供严格与宽松路径共用。"""


def _strict_candidate_search(gray, dark_mask, threshold, config, epsilon_ratios):
    """完整保留旧面积、矩形度、置信度门和旧诊断字段，返回最佳严格候选。"""
```

`locate_black_paper()`先调用`cv2.threshold(...OTSU)`取得基准阈值，再用显式阈值重建
同一张严格掩膜。严格候选成功时立即返回，并写入：

```python
diagnostics["source"] = "STRICT"
metrics["used_threshold"] = float(otsu_threshold)
```

严格失败时，依次使用`otsu_threshold + offset`，通过`np.clip(..., 1, 254)`限制范围；
重复阈值只运行一次。每张掩膜最多向后续宽松评分传递面积最大的4条基础合格轮廓。

**Step 4: Run strict regression tests**

Run:

```powershell
pytest -q tests_ab/test_paper_locator.py -k "clean_paper or adaptive_epsilon or white_pieces or invalid"
```

Expected: PASS；干净纸仍为`STRICT`，5/6角和所有旧配置校验不回退。

**Step 5: Commit**

```powershell
git add -- maixcam2_app_A_quad/paper_locator.py tests_ab/test_paper_locator.py
git commit -m "refactor: isolate strict auto roi search"
```

### Task 3: 实现宽松安全门与四边边缘支持

**Files:**
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Modify: `tests_ab/test_paper_locator.py`

**Step 1: Add edge-support rejection test**

生成一个面积、比例和暗度都合格但其中两条候选边没有灰度边缘的大型暗色结构，断言：

```python
assert result.success is False
assert result.diagnostics["relaxed_edge_reject_count"] >= 1
```

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests_ab/test_paper_locator.py -k "edge_support_reject"`

Expected: FAIL；当前没有边缘支持门。

**Step 3: Implement edge support**

增加以下函数：

```python
def _measure_quad_edge_support(gray, quad):
    """计算四条候选边附近的Canny支持比例。

    主要流程：按图像中位灰度生成稳定Canny上下阈值，把边缘图膨胀到约画面短边
    1%的容差带，再逐边用1像素线采样命中率。返回``(平均支持率, 各边支持率)``，
    避免一条强龙门架直线掩盖缺失纸边。
    """
```

实现时仅在候选四边附近采样，不运行全帧Hough。宽松候选依次检查：

```python
area_ratio >= relaxed_min_area_ratio
area_to_largest >= min_area_to_largest
rectangularity >= relaxed_min_rectangularity
aspect_score >= relaxed_min_aspect_score
darkness_score >= relaxed_min_darkness
edge_support >= min_edge_support
supported_side_count >= min_supported_sides
```

宽松综合分数使用固定、可解释权重：

```python
recovery_confidence = (
    0.30 * aspect_score
    + 0.30 * edge_support
    + 0.15 * area_to_largest
    + 0.15 * darkness_score
    + 0.10 * rectangularity
)
```

通过门且`recovery_confidence >= paper_min_confidence`的最高分候选返回`TH_SCAN`。
`best_candidate`增加`area_to_largest`、`quad_fill`、`edge_support`、
`supported_side_count`和`used_threshold`。

**Step 4: Run robust recovery tests**

Run:

```powershell
pytest -q tests_ab/test_paper_locator.py -k "uneven_brightness or small_quad_beside or edge_support or clean_paper or adaptive_epsilon"
```

Expected: PASS；偏亮纸恢复，小框和缺边大暗块拒绝，严格路径无回退。

**Step 5: Commit**

```powershell
git add -- maixcam2_app_A_quad/paper_locator.py tests_ab/test_paper_locator.py
git commit -m "fix: recover primary paper across thresholds"
```

### Task 4: 增加旧ROI一致性兜底和完整日志

**Files:**
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_paper_locator.py`
- Modify: `tests_ab/test_quad_main.py`

**Step 1: Write failing prior and log tests**

增加三项测试：旧ROI附近存在四边边缘时返回`PRIOR_EDGE`；旧ROI明显偏离当前纸面时拒绝；
日志必须输出`source/used_threshold/area_to_largest/quad_fill/edge_support/edge_sides/prior_iou`。

**Step 2: Run tests to verify they fail**

Run:

```powershell
pytest -q tests_ab/test_paper_locator.py tests_ab/test_quad_main.py -k "prior_edge or robust_fields"
```

Expected: FAIL。

**Step 3: Implement prior geometry helpers**

在`paper_locator.py`增加：

```python
def _convex_quad_iou(left_quad, right_quad):
    """计算两个有效凸四边形的交并比；退化或无交集时安全返回0。"""


def _refine_prior_quad(gray, prior_quad, config):
    """只在旧ROI四边法向的小范围内搜索最强边缘并重建四角。

    最大搜索距离由画面短边乘``paper_auto_prior_max_shift_ratio``决定；每条修正线
    由相邻线求交得到角点。返回候选及边缘指标，无法形成有效凸四边形时返回None。
    """
```

`locate_black_paper()`签名扩展为：

```python
def locate_black_paper(frame_bgr, config=None, prior_quad=None):
```

PRIOR候选必须通过`prior_iou >= 0.55`、A4比例、暗度、四边支持和画面边界检查；只允许
在严格及多阈值路径都失败后运行。

在`main.handle_calibration_action()`中把会话工作副本传入：

```python
location = locate_black_paper(
    frame_bgr,
    prior_quad=session.settings.get("paper_quad"),
)
```

**Step 4: Extend stable diagnostics**

`format_auto_roi_diagnostic_fields()`只在字段存在且有限时追加：

```text
source=TH_SCAN used_threshold=153.0 area_to_largest=0.960
quad_fill=0.880 edge_support=0.720 edge_sides=4 prior_iou=0.910
```

增加`RELAX_AREA/RELAX_RECT/RELAX_ASPECT/RELAX_DARK/EDGE_LOW/PRIOR_MISMATCH`拒绝门，
缺字段的旧`PaperLocation`测试替身仍应正常打印。

**Step 5: Run focused tests**

Run:

```powershell
pytest -q tests_ab/test_paper_locator.py tests_ab/test_quad_main.py tests_ab/test_variant_calibration_ui.py
```

Expected: PASS。

**Step 6: Commit**

```powershell
git add -- maixcam2_app_A_quad/paper_locator.py maixcam2_app_A_quad/main.py tests_ab/test_paper_locator.py tests_ab/test_quad_main.py
git commit -m "feat: add prior guided auto roi fallback"
```

### Task 5: 文档、全量验证和正式发布包

**Files:**
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `研究发现.md`
- Modify: `编辑清单.md`
- Modify: `maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip`

**Step 1: Update field documentation**

在调试手册AUTO ROI章节补充三条路径、新日志字段和现场判断表。明确：

- `STRICT`表示旧严格路径，无须调参。
- `TH_SCAN`表示亮度不均恢复，重点检查蓝框和`edge_sides`。
- `PRIOR_EDGE`表示旧ROI附近修正，必须检查`prior_iou`。
- `AUTO ROI FAIL`继续保留旧ROI。
- 不能单独降低`PAPER_AUTO_RELAXED_MIN_RECTANGULARITY`。

**Step 2: Run code verification**

Run:

```powershell
python -m compileall -q maixcam2_app_A_quad
pytest -q tests_ab/test_paper_locator.py tests_ab/test_quad_main.py tests_ab/test_variant_calibration_ui.py
pytest -q
```

Expected: compileall退出码0；聚焦测试及全量测试全部PASS。

**Step 3: Build the release ZIP**

Run:

```powershell
python tools/package_variants.py
pytest -q tests_ab/test_variant_packages.py
```

Expected: A/B发布包生成成功；发布测试全部PASS，ZIP为平铺结构且A版源码逐字节一致。

**Step 4: Record release evidence**

Run:

```powershell
Get-FileHash -Algorithm SHA256 maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip
git status --short
```

把测试数量、ZIP字节数和SHA256写入`编辑清单.md`；在`研究发现.md`记录本次实机根因、
严格路径保护以及小候选安全门。不得纳入用户的`maixcam2_app_A_quad.7z`和
`maixcam2_app_A_quad/dist/maix-diansai_quad-v2.1.0.zip`。

**Step 5: Commit**

```powershell
git add -- maixcam2_app_A_quad/A版实机调试手册.md 研究发现.md 编辑清单.md maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip
git commit -m "release: publish robust auto roi recovery"
```

**Step 6: Verify final commit and protected files**

Run:

```powershell
git status --short --branch
git log -6 --oneline --decorate
```

Expected: 分支只剩两份受保护用户文件未跟踪；最新提交为发布提交，设计和各阶段提交均可单独回退。
