# MaixCAM2 A版求解轮廓清理与START确认门 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** WHITE自动合并求解副本中的毫米伪短边，并让正常页只有点击START后才执行碎片检测和拼装求解。

**Architecture:** `assembly_planner.py`在WHITE构造GRAPH内部求解碎片时执行不修改原输入的短边合并，GRAPH失败后的兜底恢复原始锁定顶点；`main.py`维护独立`capture_armed`状态，`touch_ui.py`增加START按钮，正常页分析门同时受CAL、START和快照锁定控制。

**Tech Stack:** Python 3、NumPy、OpenCV、MaixPy触摸/显示主循环、pytest。

---

### Task 1: WHITE求解副本短边清理

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write failing cleanup tests**

增加以下行为测试：

- 文件顶部公开`UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM == 12.0`。
- 5点轮廓包含2.3/6.7/10.4mm伪短边时，清理后至少3点且所有剩余边不低于门槛。
- 所有边大于等于20mm的真实五边形不变化。
- 门槛0完全关闭清理。
- 输入数组和视觉字典逐字节不变。
- WHITE调用`_solver_piece`使用清理顶点，CARD继续使用原顶点和边缘特征。

**Step 2: Run cleanup RED tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "solver_cleanup or short_edge" -q`

Expected: FAIL，当前没有清理宏和清理函数，U2/U3仍包含短边。

**Step 3: Implement minimal geometry cleanup**

在现场常量区增加`UNKNOWN_WHITE_SOLVER_MIN_EDGE_MM = 12.0`。实现独立纯函数：校验顶点后循环选择最短短边，对两个端点分别计算删除后的点到弦线距离，删除距离较小端点；面积退化时拒绝该删除；至少保留3点。`_solver_piece`增加显式`clean_short_edges`参数，默认False保持现有调用兼容。

GRAPH构造内部碎片时启用清理；GRAPH失败后的WHITE兜底恢复原始锁定顶点，避免短边删除造成填充面积损失。CARD始终保留原始顶点和`edge_features`。

**Step 4: Add lazy CLEAN diagnostics**

扩展运行器快照日志，在WHITE且调试开启时输出原始/求解顶点数、删除数和原始最短边；关闭调试时不执行清理预览或字符串格式化。补开启/关闭测试。

**Step 5: Run planner GREEN tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -q`

Expected: 清理、GRAPH、CARD、容错和运行器测试全部通过。

---

### Task 2: START按钮与完全待机状态机

**Files:**
- Modify: `tests_ab/test_quad_main.py`
- Create: `tests_ab/test_a_start_gate.py`
- Modify: `maixcam2_app_A_quad/touch_ui.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write button and state RED tests**

增加以下行为测试：

- 640×480布局包含`start`，位于第三按钮与CAL之间，所有按钮互不重叠。
- 初始`capture_armed=False`时正常页不分析视觉帧。
- CAL无条件实时分析；START后分析；快照锁定后停止重复分析。
- 模式/材料切换将状态改为未启动并重置规划器。
- 点击START后状态为启动，重复START会重置并重新采集。
- 未START时KNOWN SAVE返回`PRESS START`且不写模板。

**Step 2: Run START RED tests**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py -k "start or armed or analyze_live" -q`

Expected: FAIL，当前布局没有START且主循环启动即分析。

**Step 3: Implement pure state helpers**

在`main.py`增加可独立测试的状态转换函数：模式选择、材料选择和CAL往返均返回未启动；START校验模式/材料、重置运行器并返回启动状态和`... CAPTURE`文字。扩展`should_analyze_live_frame()`接收`capture_armed`，正常页只有已启动且未锁定时返回True。

**Step 4: Implement button and overlay**

在`touch_ui.py`利用第三按钮与CAL之间剩余宽度创建`start`按钮；不足最小宽度时抛出明确布局错误。`draw_overlay()`接收`capture_armed`，START启动时显示绿色，待机时灰色。模式按钮仍显示当前选择。

**Step 5: Wire run_app standby data flow**

主循环初始化`capture_armed=False`。待机且非CAL时不调用`analyze_quad_frame()`，使用空检测结果、已保存纸面四角和空碎片绘制；模式/材料/CAL动作取消启动；START动作清缓存并从下一帧开始分析。规划器更新、模板匹配和未知编号只在已启动时执行；KNOWN SAVE未启动时只显示`PRESS START`。

**Step 6: Run UI GREEN tests**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py tests_ab/test_a_runtime_planner.py -q`

Expected: START、待机、CAL、SAVE和现有触摸回归全部通过。

---

### Task 3: v1.8.0文档、发布和全量验收

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `README.md`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `项目规划清单.md`
- Modify: `研究发现.md`
- Modify: `编辑清单.md`
- Modify: `硬件资源表.md`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.8.0.zip`

**Step 1: Update operator documentation and version**

记录清理宏准确行号、推荐范围、0关闭方法、CLEAN日志、完全待机、START重拍和KNOWN START→SAVE流程。A版版本和打包规格提升到1.8.0，B版保持1.1.0。

**Step 2: Run targeted verification**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py -q`

Expected: 全部通过且无意外标准输出错误。

**Step 3: Run full and compile verification**

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: 退出码均为0。

**Step 4: Build and verify release**

先记录B版ZIP SHA256，再运行`python tools/package_variants.py`，最后运行`python -m pytest tests_ab/test_variant_packages.py -q`。核对A版ZIP逐文件字节一致、平铺`main.py`导入成功、B版打包前后SHA256不变。

**Step 5: Record evidence**

把定向/全量数量、编译结果、A/B包大小与SHA256写入四文件。根目录`.git`为空且不是有效仓库，因此不执行commit/push。
