# MaixCAM2 A版 WHITE/CARD 随时求解 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为A版UNKNOWN增加显式WHITE/CARD子模式，并让跨帧求解按累计计算时间截止、在截止时保留已有合法解。

**Architecture:** UI复用UNKNOWN下的SAVE触摸区切换子模式，`AssemblyRuntime`把子模式作为规划上下文并向`UnknownSolveJob`传递停止策略。求解生成器持续公开当前最优规划，任务层分别统计累计计算时间和总墙钟，运行器只对无首解超时执行自动重试。

**Tech Stack:** MaixPy、Python、OpenCV、NumPy、pytest、MaixVision平铺ZIP。

---

### Task 1: 三片真实形状与WHITE首解停止

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

新增“两块四边形加一块三角形”随机位姿测试，并新增高纹理特征下
`stop_at_first_solution=True`时`search_nodes == first_solution_node`的测试。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "three_irregular or white_stops" -q`

Expected: FAIL，原因分别为缺少新场景覆盖或求解接口不接受`stop_at_first_solution`。

**Step 3: Implement the minimal solver behavior**

给`_solve_unknown_layout_steps()`、`UnknownSolveJob`和`solve_unknown_layout()`增加
`stop_at_first_solution=False`参数；`evaluate_complete()`保存首个合法候选后，在该参数
为真时立即停止。所有新函数参数和分支补充中文注释。

**Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "three_irregular or white_stops" -q`

Expected: PASS。

### Task 2: 累计计算预算与截止保留最优解

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

新增三个失败测试：帧间等待超过5秒但小于20秒不触发累计计算超时；单工作单元累计
计算超过5秒触发超时；进度中已有成功规划时到达截止线返回成功并标记
`returned_best_at_timeout=1`。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "active_budget or best_at_timeout" -q`

Expected: FAIL，旧实现只有从创建时刻开始的单一墙钟。

**Step 3: Implement dual budgets and anytime result**

求解核心每次得到更优合法布局时构造并写入`progress["best_plan"]`。任务新增
`active_timeout_seconds=5.0`和`wall_timeout_seconds=20.0`，只把每次`next()`调用耗时
累计到活动预算；兼容参数`timeout_seconds`仅覆盖硬墙钟。截止处理优先返回best_plan，
否则返回结构化失败，并补齐整数诊断字段。

**Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -q`

Expected: 全部PASS。

### Task 3: 运行器自动重试与现场进度

**Files:**
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing tests**

新增无首解timeout只返回一帧失败、`runtime.plan is None`、随后重新累计稳定帧的测试；
扩展状态文字测试，要求显示`N/E/F/S`。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py -k "timeout or progress" -q`

Expected: FAIL，旧运行器缓存timeout且状态只有节点数。

**Step 3: Implement retry and diagnostics**

`AssemblyRuntime`增加边候选、最大前沿和首解属性；无首解timeout结束时不赋给
`self.plan`，清除参考观测并返回本帧失败。`select_planning_status()`接收并显示诊断值。

**Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py -q`

Expected: 全部PASS。

### Task 4: WHITE/CARD触摸与动态按钮

**Files:**
- Modify: `tests_ab/test_quad_main.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing tests**

新增子模式切换纯函数测试、UNKNOWN动态按钮文字像素/接口测试和AST接线测试，确认
KNOWN下动作仍进入`perform_known_save_action()`。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_quad_main.py -k "unknown_profile or white_card" -q`

Expected: FAIL，旧入口没有子模式状态和动态标签。

**Step 3: Implement UI and runtime wiring**

增加`UNKNOWN_PROFILE_WHITE/CARD`常量和校验/切换辅助函数。`run_app()`默认WHITE，
UNKNOWN下点击`save`切换子模式并重置规划；调用`planner_runtime.update()`时传入子模式。
`draw_overlay()`在UNKNOWN下把第三按钮标签绘制为WHITE或CARD，KNOWN仍绘制SAVE。

**Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_a_runtime_planner.py -q`

Expected: 全部PASS。

### Task 5: v1.4.1发布与项目记录

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `README.md`
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`
- Modify: `硬件资源表.md`

**Step 1: Write the failing release tests**

把A版期望版本和ZIP名提升到1.4.1并运行发布测试，确认旧清单和打包脚本产生RED。

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: FAIL，报告A版版本或ZIP名仍为1.4.0。

**Step 2: Update release metadata and documentation**

同步A版清单、打包脚本、README、实机调试手册和四文件；明确WHITE/CARD按钮、双预算、
超时最优解、无首解重试和实机验证步骤。稳定版与B版业务源码不修改。

**Step 3: Run full verification**

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: 全部退出码0。

**Step 4: Verify release boundaries**

逐项核对稳定版9个Python/YAML文件SHA256与既有基线一致，确认B版ZIP仍为
`F08BFE4183241399073DA4DB6C16778678498E9214D62CC755EC22C37481ED92`，记录A版
v1.4.1 ZIP大小和SHA256。

当前根目录不是有效Git仓库，因此不执行计划模板中的commit/push步骤；以测试、
compileall、平铺ZIP导入和SHA256作为可回放证据。
