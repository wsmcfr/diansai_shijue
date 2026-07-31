# UNKNOWN Best-Effort Result Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让普通UNKNOWN忽略矩形毫米尺寸，并在没有可靠解或超时时显示、发送带明确警告标志的最佳完整拼法。

**Architecture:** 在现有完整布局规范化函数中把尺寸检查改成调用方可选，KNOWN继续使用默认硬门，UNKNOWN显式关闭。每个UNKNOWN阶段分别保存可靠`best_plan`和不可靠`best_effort_plan`，自然结束与超时均按可靠优先返回。`AssemblyPlan.reliable`贯穿屏幕状态与UART v3结果flags。

**Tech Stack:** MaixPy兼容Python、NumPy、OpenCV、pytest、UART4小端二进制协议、PowerShell发布脚本。

---

### Task 1: UNKNOWN取消尺寸硬门并保存最佳完整候选

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py:99-138,1262-1335,1436-1608,1875-2068,2110-2550,2628-3367`
- Test: `tests_ab/test_a_unknown_planner.py`
- Test: `tests_ab/test_a_runtime_planner.py`

**Step 1: 写本次三片快照的失败回归**

在`tests_ab/test_a_unknown_planner.py`增加设备第三次快照构造器，并断言：

```python
plan = solve_unknown_layout(
    pieces,
    (0.0, 0.0, 297.0, 210.0),
    105.0,
    stop_at_first_solution=True,
    paper_orientation="landscape",
)
assert plan.success is True
assert plan.reliable is True
assert plan.target_rect_mm[2] < 88.0
assert plan.diagnostics["fill_permille"] >= 920
```

**Step 2: 运行测试确认红灯**

Run: `pytest -q tests_ab/test_a_unknown_planner.py -k "ignores_size_gate"`

Expected: FAIL，当前结果为`fill_reject`或`AssemblyPlan`没有`reliable`字段。

**Step 3: 写自然失败与超时最佳候选测试**

构造低填充但可完整连接的三片输入，断言自然搜索结束返回：

```python
assert plan.success is True
assert plan.reliable is False
assert plan.reason == "best_effort"
assert plan.diagnostics["best_effort"] == 1
```

在运行器测试中注入可控时钟和`progress["best_effort_plan"]`，断言`_finish_timeout()`优先级为`best_plan > best_effort_plan > solver_timeout`。

**Step 4: 运行新增测试确认红灯原因准确**

Run: `pytest -q tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -k "best_effort or ignores_size_gate"`

Expected: FAIL，失败集中在尺寸门、最佳候选没有保存或超时没有回退。

**Step 5: 最小实现结果模型和可选尺寸门**

在`AssemblyPlan`增加：

```python
self.reliable = bool(reliable)
```

失败计划固定`reliable=False`。`_canonicalize_complete_layout()`增加`enforce_size_bounds=True`，KNOWN保持默认值；UNKNOWN三条路径显式传`False`。`_partial_layout_priority()`增加相同开关，关闭时只计算排序信息，不按题目尺寸范围剪枝。

**Step 6: 最小实现最佳候选生命周期**

完整候选未通过可靠门时，用`min_fill_ratio=0.0`、放宽验收重算几何指标，按低重叠、低空洞、高外边数、低接缝误差更新`progress["best_effort_plan"]`。最佳回退计划使用：

```python
AssemblyPlan(
    True,
    placements=placements,
    target_rect_mm=target_rect,
    score=score,
    reason="best_effort",
    reliable=False,
    diagnostics={"best_effort": 1, ...},
)
```

FALLBACK自然结束没有可靠解时返回该计划；`UnknownSolveJob._finish_timeout()`优先克隆可靠计划，否则克隆最佳回退计划。

**Step 7: 运行聚焦测试并提交**

Run: `pytest -q tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py`

Expected: PASS。

Commit:

```bash
git add maixcam2_app_A_quad/assembly_planner.py tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py
git commit -m "feat: return unknown best effort layouts"
```

### Task 2: 屏幕警告与UART v3可靠性标志

**Files:**
- Modify: `maixcam2_app_A_quad/main.py:495-516,984-1027`
- Modify: `maixcam2_app_A_quad/serial_protocol.py:9-30,267-351,695-712`
- Test: `tests_ab/test_quad_main.py`
- Test: `tests_ab/test_a_uart_protocol.py`

**Step 1: 写屏幕状态红灯测试**

增加可靠和最佳回退计划断言：

```python
assert select_planning_status(..., reliable_plan, ...) == "LOCKED PLAN OK N=3"
assert select_planning_status(..., best_effort_plan, ...) == "LOCKED PLAN BEST ! N=3"
```

**Step 2: 写UART flags红灯测试**

断言协议版本为3，可靠结果flags为0，回退结果flags bit0为1，未知高位被拒绝：

```python
payload = encode_puzzle_result_payload(..., best_effort=True)
decoded = decode_puzzle_result_payload(payload)
assert decoded["best_effort"] is True
assert decoded["reliable"] is False
```

运行：`pytest -q tests_ab/test_quad_main.py tests_ab/test_a_uart_protocol.py -k "best_effort or puzzle_payload"`

Expected: FAIL，当前保留字节固定为0且队列接口没有标志参数。

**Step 3: 实现屏幕和协议字段**

`select_planning_status()`根据`AssemblyPlan.reliable`选择`PLAN OK`或`PLAN BEST !`。协议增加：

```python
PROTOCOL_VERSION = 3
RESULT_FLAG_BEST_EFFORT = 0x01
RESULT_FLAG_KNOWN_MASK = RESULT_FLAG_BEST_EFFORT
```

编码器、解码器和`VisionSerialRuntime.queue_puzzle_result_once()`增加`best_effort=False`参数；解码器拒绝未知标志位。`queue_successful_plan_result()`从计划可靠性生成bit0。

**Step 4: 运行协议和入口测试并提交**

Run: `pytest -q tests_ab/test_a_uart_protocol.py tests_ab/test_quad_main.py tests_ab/test_a_main_start_gate.py`

Expected: PASS。

Commit:

```bash
git add maixcam2_app_A_quad/main.py maixcam2_app_A_quad/serial_protocol.py tests_ab/test_quad_main.py tests_ab/test_a_uart_protocol.py
git commit -m "feat: flag best effort puzzle results"
```

### Task 3: 协议、调试和现场文档

**Files:**
- Modify: `maixcam2_app_A_quad/MaixCAM2与STM32F4串口协议说明.md`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`

**Step 1: 更新F4解析说明**

把协议版本改为3，记录`PUZZLE_RESULT`第4字节flags及以下F4逻辑：

```c
#define RESULT_FLAG_BEST_EFFORT 0x01u
bool result_may_be_inaccurate = (result_flags & RESULT_FLAG_BEST_EFFORT) != 0u;
```

明确bit0为1仍按原1至4片记录执行，但上层必须提示可能不准确；未知flags必须NACK。

**Step 2: 更新现场状态和日志说明**

补充`PLAN BEST !`、`reason=best_effort`、`reliable=0`以及“无完整候选仍可能失败”的边界。删除UNKNOWN尺寸拒绝作为现场首要调参项，保留KNOWN和FOUR的原尺寸说明。

**Step 3: 更新四文件并检查文档**

Run: `git diff --check`

Expected: exit 0且无输出。

Commit:

```bash
git add maixcam2_app_A_quad/MaixCAM2与STM32F4串口协议说明.md maixcam2_app_A_quad/A版实机调试手册.md 项目规划清单.md 编辑清单.md 研究发现.md
git commit -m "docs: describe best effort result flag"
```

### Task 4: 全量验证与v2.2.0发布

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `maixcam2_app_A_quad/dist/diansai_quad-v2.2.0.zip`
- Test: `tests_ab/test_variant_packages.py`

**Step 1: 写版本和发布物红灯测试**

将A版期望版本与文件名更新为v2.2.0，先运行发布测试确认旧元数据失败。

Run: `pytest -q tests_ab/test_variant_packages.py`

Expected: FAIL，当前app和正式包仍为v2.1.0。

**Step 2: 更新版本并构建发布包**

更新`app.yaml`到v2.2.0，使用仓库既有`tools/package_variants.py`生成A版白名单ZIP，不覆盖用户的`maix-diansai_quad-v2.1.0.zip`和根目录`maixcam2_app_A_quad.7z`。

**Step 3: 执行完整验证门**

Run:

```text
pytest -q tests tests_ab
python -m compileall -q maixcam2_app_A_quad
pytest -q tests_ab/test_variant_packages.py tests_ab/test_a_uart_protocol.py
git diff --check
```

Expected: 全部exit 0、零失败、无diff空白错误。

**Step 4: 核对ZIP并提交发布**

记录新ZIP字节数和SHA256，确认B版ZIP哈希未改变，确认两个用户压缩包仍为未跟踪文件。

Commit:

```bash
git add maixcam2_app_A_quad/app.yaml maixcam2_app_A_quad/dist/diansai_quad-v2.2.0.zip tests_ab/test_variant_packages.py 项目规划清单.md 编辑清单.md 研究发现.md
git commit -m "release: package unknown best effort v2.2.0"
git push
```

