# UNKNOWN 四片专用识别与求解链路 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 A 版新增独立 FOUR 模式，在完整 A4 透视平面上稳定识别四片白色碎片，并在 1～3 秒目标时间内生成可直接通过现有 UART4 协议发送的拼装位姿。

**Architecture:** KNOWN 和 UNKNOWN 一至三片继续使用现有 `puzzle_vision.py` 与 `assembly_planner.py`。FOUR 模式通过新的 `four_piece_vision.py` 完成 A4 展开、双阈值分割、四轮廓锁定和几何提取，再通过新的 `four_piece_solver.py` 执行候选接缝图搜索；结果复用 `AssemblyPlan`、`AssemblyPlacement` 和现有串口帧。

**Tech Stack:** Python 3、OpenCV、NumPy、pytest、MaixPy/MaixCAM2、UART4 二进制定点协议。

---

### Task 1: 固定现有基线并增加 FOUR 模式契约

**Files:**
- Modify: `maixcam2_app_A_quad/touch_ui.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_a_start_gate.py`
- Modify: `tests_ab/test_quad_main.py`

**Step 1: 写失败测试**

在正常页布局测试中要求按钮顺序为：

```python
assert tuple(buttons) == (
    "known",
    "unknown",
    "four",
    "save",
    "start",
    "cal",
)
```

增加 `MODE_FOUR = "four"` 的选择、START 状态和串口模式映射测试：

```python
def test_four_mode_starts_dedicated_capture_and_uses_unknown_wire_mode():
    """FOUR必须独立启动，但协议仍编码为UNKNOWN。"""
    mode, armed, status = select_capture_mode(MODE_FOUR, runtime)
    assert (mode, armed, status) == (MODE_FOUR, False, "PRESS START")
    assert start_capture(mode, runtime)[1] == "FOUR CAPTURE"
    assert protocol_mode_for_capture(mode) == MODE_UNKNOWN
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_start_gate.py tests_ab/test_quad_main.py -q`

Expected: FAIL，提示缺少 `four` 按钮或 `MODE_FOUR`。

**Step 3: 实现最小模式契约**

- 将六个按钮按640×480安全宽度重新排布，保证触摸区不重叠。
- `_normalize_capture_mode` 接受 `known/unknown/four`。
- `start_capture` 对 FOUR 返回 `FOUR CAPTURE`。
- 新增纯函数 `protocol_mode_for_capture`，把 FOUR映射为UNKNOWN，其余模式原样返回。
- KNOWN 的 SAVE 与 UNKNOWN 的 WHITE/CARD切换逻辑不变；FOUR下禁用SAVE/profile按钮。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_start_gate.py tests_ab/test_quad_main.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/touch_ui.py maixcam2_app_A_quad/main.py tests_ab/test_a_start_gate.py tests_ab/test_quad_main.py
git commit -m "feat: add isolated four-piece capture mode"
```

### Task 2: 实现完整 A4 透视展开与毫米映射

**Files:**
- Create: `maixcam2_app_A_quad/four_piece_vision.py`
- Create: `tests_ab/test_a_four_piece_vision.py`

**Step 1: 写失败测试**

覆盖竖放、横放、倾斜四边形和非法标定：

```python
def test_warp_full_paper_preserves_landscape_mm_axes():
    """横放A4必须展开为297×210mm且像素重心可直接换算毫米。"""
    result = warp_full_paper(frame, paper_quad, "landscape", pixels_per_mm=3.0)
    assert result.image.shape[:2] == (630, 891)
    assert result.pixel_to_mm((300.0, 150.0)) == pytest.approx((100.0, 50.0))
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_four_piece_vision.py -q`

Expected: FAIL，模块尚不存在。

**Step 3: 实现最小展开接口**

在文件顶部定义有中文解释的现场宏：

```python
FOUR_WARP_PIXELS_PER_MM = 3.0
FOUR_SOLVER_DEBUG = True
```

实现：

- `PaperWarpResult`：保存展开图、正逆单应矩阵、毫米尺寸和比例。
- `build_paper_warp`：使用 `order_a4_quad` 与 `paper_size_mm` 生成相机到纸面的矩阵。
- `warp_full_paper`：调用 `cv2.warpPerspective` 展开完整蓝框。
- `warp_pixel_to_paper_mm`：按固定比例转换，不重复计算 Homography。
- `paper_mm_to_image_px`：复用逆矩阵回绘源轮廓。

所有函数写中文函数注释，说明输入坐标系、返回值和非法输入。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_four_piece_vision.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/four_piece_vision.py tests_ab/test_a_four_piece_vision.py
git commit -m "feat: add full-paper four-piece warp"
```

### Task 3: 实现 HSV/LAB 双阈值分割和区域恢复

**Files:**
- Modify: `maixcam2_app_A_quad/four_piece_vision.py`
- Modify: `tests_ab/test_a_four_piece_vision.py`

**Step 1: 写失败测试**

用合成图覆盖白纸、灰色裸露金属、局部阴影、亮点噪声和2mm黑缝：

```python
def test_dual_mask_recovers_metal_without_bridging_two_mm_gap():
    """宽松支撑可补回单片灰区，但不能跨越严格核心之间的黑缝。"""
    masks = build_four_piece_masks(warped_bgr, pixels_per_mm=3.0)
    count, _ = cv2.connectedComponents(masks.final)
    assert count - 1 == 4
    assert np.all(masks.final[:, gap_slice] == 0)
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_four_piece_vision.py -q`

Expected: FAIL，缺少 `build_four_piece_masks`。

**Step 3: 实现双阈值与受限恢复**

新增独立宏：

```python
FOUR_HSV_S_MAX = 80
FOUR_HSV_V_MIN = 150
FOUR_LAB_L_MIN = 150
FOUR_SUPPORT_V_MIN = 95
FOUR_MIN_NOISE_AREA_MM2 = 4.0
```

实现：

- `FourPieceMasks` 数据结构保存 strict/support/final。
- 严格核心使用低饱和高亮度与LAB亮度联合门。
- 宽松支撑只作为核心连通恢复的允许区域。
- 用 `connectedComponentsWithStats` 对每个严格核心独立恢复。
- 删除小面积噪点并仅填闭合内部小孔。
- 不使用全局膨胀和大闭运算。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_four_piece_vision.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/four_piece_vision.py tests_ab/test_a_four_piece_vision.py
git commit -m "feat: segment four pieces with dual color masks"
```

### Task 4: 实现四轮廓提取、受限拆分和稳定锁定

**Files:**
- Modify: `maixcam2_app_A_quad/four_piece_vision.py`
- Create: `tests_ab/test_a_four_piece_runtime.py`

**Step 1: 写失败测试**

测试四片提取、三连通区域受限拆分、噪声拒绝和三帧冻结：

```python
def test_runtime_locks_exact_third_stable_observation_once():
    """第三个稳定四片结果必须被冻结，后续抖动不得覆盖。"""
    runtime = FourPieceVisionRuntime(stable_frames=3)
    assert runtime.update(frame_1, calibration).locked is False
    assert runtime.update(frame_2, calibration).locked is False
    locked = runtime.update(frame_3, calibration)
    assert locked.locked is True
    assert runtime.update(jitter_frame, calibration).pieces is locked.pieces
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_four_piece_runtime.py -q`

Expected: FAIL，运行器尚不存在。

**Step 3: 实现轮廓和运行器**

- `extract_four_piece_candidates`：面积、完整性、核心支持率和轮廓质量过滤。
- `split_one_connected_candidate`：仅在3/4时对异常大区域尝试严格核心/黑缝/凹陷拆分。
- `build_piece_geometry`：输出 `U1～U4`、`center_mm`、`vertices_mm`、面积、原图回绘点。
- `FourPieceDetection`：保存三张掩膜、计数、拆分标志和失败原因。
- `FourPieceVisionRuntime`：按重心最小代价匹配、重心毫米抖动和面积变化率累计稳定帧。
- `reset`：释放快照并回到完全待机。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_four_piece_vision.py tests_ab/test_a_four_piece_runtime.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/four_piece_vision.py tests_ab/test_a_four_piece_runtime.py
git commit -m "feat: lock stable four-piece geometry"
```

### Task 5: 实现四片边线假设和候选接缝关系

**Files:**
- Create: `maixcam2_app_A_quad/four_piece_solver.py`
- Create: `tests_ab/test_a_four_piece_solver.py`

**Step 1: 写失败测试**

覆盖伪短边合并、整边、分段T形和禁止镜像：

```python
def test_build_relations_finds_segmented_t_junction_without_reflection():
    """一条长边允许与两条连续短边配合，但关系只能是旋转和平移。"""
    relations = build_pair_relations(piece_a, piece_b)
    assert any(item.segmented for item in relations)
    assert all(np.linalg.det(item.rotation) > 0.0 for item in relations)
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_four_piece_solver.py -q`

Expected: FAIL，模块尚不存在。

**Step 3: 实现边线和关系结构**

定义四片独立宏：

```python
FOUR_MIN_EDGE_MM = 10.0
FOUR_EDGE_LENGTH_TOLERANCE_MM = 3.0
FOUR_EDGE_DIRECTION_TOLERANCE_DEG = 12.0
FOUR_PAIR_RELATION_LIMIT = 6
```

实现：

- `ShapeHypothesis`、`EdgeSegment`、`PairRelation` 数据结构。
- 多尺度 `approxPolyDP` 形状假设生成与轮廓误差评分。
- 伪短边和近共线边清理。
- 反向接缝刚体变换计算。
- 整边和受限分段边关系生成。
- 每个有向片对按误差排序并截断到固定数量。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_four_piece_solver.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/four_piece_solver.py tests_ab/test_a_four_piece_solver.py
git commit -m "feat: build four-piece seam relation graph"
```

### Task 6: 实现增量四片图搜索和矩形硬验收

**Files:**
- Modify: `maixcam2_app_A_quad/four_piece_solver.py`
- Modify: `tests_ab/test_a_four_piece_solver.py`

**Step 1: 写失败测试**

覆盖任意源旋转、四宫格、斜接缝、T形、尺寸越界、重叠、超时和失败不携带目标：

```python
def test_four_solver_returns_all_source_and_target_poses():
    """专用求解器必须一次返回四个可发送位姿。"""
    plan = solve_four_piece_layout(pieces, work_region, split_y)
    assert plan.success is True
    assert len(plan.placements) == 4
    assert {item.piece_id for item in plan.placements} == {"U1", "U2", "U3", "U4"}
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_four_piece_solver.py -q`

Expected: FAIL，求解入口尚不存在。

**Step 3: 实现分层搜索**

新增：

```python
FOUR_STRICT_MIN_FILL_RATIO = 0.92
FOUR_RELAXED_MIN_FILL_RATIO = 0.85
FOUR_MAX_OVERLAP_RATIO = 0.03
FOUR_BEAM_WIDTH = 32
FOUR_ACTIVE_BUDGET_SECONDS = 3.0
```

实现：

- `FourPieceSolveJob`：每次 `advance(time_budget_ms, work_unit_limit)` 只推进有限候选。
- 两片、三片、四片逐层扩展并按联合外框、面积缺口和接缝误差排序。
- 中间层用解析多边形/三角剖分检查重叠；完整层执行栅格精确验收。
- 矩形尺寸限制为 `90～120mm × 50～90mm`。
- 严格轮无解后才进入0.85宽松轮，重叠和尺寸不随之放宽。
- 完整布局整体放入分界线下方的工作区域。
- 使用 `AssemblyPlan` 和 `AssemblyPlacement` 返回兼容结果。
- 超时、无边、无矩形、目标越界均返回无placements的失败结果。

**Step 4: 运行测试确认通过并检查性能**

Run: `python -m pytest tests_ab/test_a_four_piece_solver.py -q`

Expected: PASS；合成四片单例在PC上明显低于1秒。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/four_piece_solver.py tests_ab/test_a_four_piece_solver.py
git commit -m "feat: solve four-piece layouts incrementally"
```

### Task 7: 将 FOUR 视觉和求解运行器接入设备主循环

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/touch_ui.py`
- Modify: `tests_ab/test_quad_main.py`
- Modify: `tests_ab/test_a_four_piece_runtime.py`

**Step 1: 写失败测试**

- KNOWN/UNKNOWN只调用旧检测器和旧运行器。
- FOUR只调用新视觉运行器和新求解任务。
- 锁定后停止重新分割。
- 失败后不自动重启。
- START同时重置两个运行器和串口结果上下文。

```python
def test_non_four_modes_never_call_four_pipeline(monkeypatch):
    """四片模块不得进入原有一至三片模式。"""
    monkeypatch.setattr(main, "analyze_four_piece_frame", fail_if_called)
    run_one_frame(mode=main.MODE_UNKNOWN)
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_a_four_piece_runtime.py -q`

Expected: FAIL，主循环尚未分流。

**Step 3: 实现非阻塞主循环分流**

- 实例化独立 `FourPieceRuntime`。
- 模式/CAL/START动作同时安全重置旧运行器和四片运行器。
- FOUR检测阶段使用保存的 `paper_quad` 和 `paper_orientation`。
- FOUR锁定后每帧只给求解器固定时间片，保持触摸和UART心跳刷新。
- 成功结果交给现有 `queue_successful_plan_result`，模式先经 `protocol_mode_for_capture` 映射。
- FOUR失败保持快照，不自动创建新任务。
- 回绘使用四片结果中的原图轮廓和兼容 `AssemblyPlan`。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_start_gate.py tests_ab/test_quad_main.py tests_ab/test_a_four_piece_runtime.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/main.py maixcam2_app_A_quad/touch_ui.py tests_ab/test_quad_main.py tests_ab/test_a_four_piece_runtime.py
git commit -m "feat: wire four-piece pipeline into runtime"
```

### Task 8: 增加 FOUR 调试画面、失败原因和单次串口发送验证

**Files:**
- Modify: `maixcam2_app_A_quad/four_piece_vision.py`
- Modify: `maixcam2_app_A_quad/four_piece_solver.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_a_four_piece_runtime.py`
- Modify: `tests_ab/test_a_uart_protocol.py`

**Step 1: 写失败测试**

验证 strict/support/final 掩膜切换、失败文本、最佳指标、四片同帧发送和任务级去重。

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_a_four_piece_runtime.py tests_ab/test_a_uart_protocol.py -q`

Expected: FAIL，缺少诊断或FOUR结果集成。

**Step 3: 实现诊断与发送保护**

- 状态文字覆盖 `NO PAPER/COUNT/UNSTABLE/NO EDGE/NO RECT/TIMEOUT/TARGET RANGE`。
- 调试叠加显示4/4、稳定计数、拆分标志、边线和最佳矩形指标。
- `FOUR_SOLVER_DEBUG=False` 时不构造详细日志字符串。
- 四片成功计划仍通过现有1～4片协议编码器，验证只排队一次。

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests_ab/test_a_four_piece_runtime.py tests_ab/test_a_uart_protocol.py -q`

Expected: PASS。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/four_piece_vision.py maixcam2_app_A_quad/four_piece_solver.py maixcam2_app_A_quad/main.py tests_ab/test_a_four_piece_runtime.py tests_ab/test_a_uart_protocol.py
git commit -m "feat: add four-piece diagnostics and result guard"
```

### Task 9: 更新设备清单、调试文档和发布包

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `tools/package_variants.py`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip`

**Step 1: 写失败测试**

把A版目标版本改为 `2.1.0`，运行文件白名单增加：

```python
{"four_piece_vision.py", "four_piece_solver.py"}
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: FAIL，清单和ZIP仍为2.0.1。

**Step 3: 更新发布内容**

- `app.yaml` 增加两个运行模块并更新版本。
- 打包脚本和发布测试同步为2.1.0。
- 调试手册增加FOUR按键、阈值、掩膜和失败原因说明。
- 生成平铺ZIP，不删除或覆盖v2.0.1发布包。

Run: `python tools/package_variants.py maixcam2_app_A_quad`

**Step 4: 运行发布测试确认通过**

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: PASS，ZIP内容与当前源码逐字节一致，平铺导入成功。

**Step 5: 提交**

```bash
git add maixcam2_app_A_quad/app.yaml maixcam2_app_A_quad/A版实机调试手册.md maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip tests_ab/test_variant_packages.py tools/package_variants.py
git commit -m "release: package dedicated four-piece pipeline v2.1.0"
```

### Task 10: 全量回归、静态检查和版本标记

**Files:**
- Verify only unless a测试暴露本功能缺陷。

**Step 1: 运行四片专项测试**

Run: `python -m pytest tests_ab/test_a_four_piece_vision.py tests_ab/test_a_four_piece_runtime.py tests_ab/test_a_four_piece_solver.py -q`

Expected: PASS。

**Step 2: 运行A版及A/B共享回归**

Run: `python -m pytest tests_ab -q`

Expected: 当前420项及新增测试全部PASS。

**Step 3: 检查源码和提交范围**

Run: `git diff --check HEAD~1..HEAD`

Run: `git status --short`

Expected: 无空白错误；用户 `.7z` 不在提交中；无意外修改B版。

**Step 4: 记录发布校验值**

Run: `Get-FileHash maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip -Algorithm SHA256`

Expected: 输出非空SHA256，写入最终交付说明。

**Step 5: 标记并推送**

```bash
git tag -a v2.1.0 -m "Dedicated UNKNOWN four-piece pipeline"
git push origin main
git push origin v2.1.0
```

若远端权限或网络失败，保留本地提交与标签，并向用户给出完整错误信息和可重试命令。
