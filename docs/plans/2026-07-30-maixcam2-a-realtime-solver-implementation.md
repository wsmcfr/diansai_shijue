# MaixCAM2 A版跨帧实时拼装求解 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把A版UNKNOWN和KNOWN SAVE共用的同步未知拼装搜索改成跨帧增量任务，消除画面与触摸冻结，同时保留有限纹理择优并发布新版本。

**Architecture:** `assembly_planner.py`用生成器提供单一可暂停搜索核心，同步API和增量任务共同消费该核心；`AssemblyRuntime`管理UNKNOWN任务，`main.py`管理KNOWN SAVE任务及成功后的模板持久化。每帧按墙钟时间和工作单元双预算推进，找到首个合法矩形后只继续有限节点。

**Tech Stack:** MaixPy/Python 3、OpenCV、NumPy、pytest、MaixVision平铺ZIP。

---

## 实施清单

| 轮次 | 目标 | 验证标准 | 审查 |
|---|---|---|---|
| Round 1 | 增量搜索核心红绿TDD | 极小时间片不会同步完成；连续推进与原同步结果均成功 | `review:true` |
| Round 2 | UNKNOWN运行器跨帧接入 | 第3稳定帧进入SOLVING且立即返回，后续帧完成并缓存 | `review:true` |
| Round 3 | KNOWN SAVE异步登记接入 | 点击仅启动任务，成功后才保存并立即显示规划 | `review:true` |
| Round 4 | 文档、全量验证与发布 | 全量测试、编译、ZIP导入和版本边界全部通过 | `review:true` |

### Task 1: 增量未知搜索核心

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

新增复杂四片带纹理输入，要求 `UnknownSolveJob.advance(..., work_unit_limit=1)` 首次返回 `None` 且任务未完成；连续推进最终返回成功规划。新增首个合法矩形后的节点数必须受有限优化窗口约束，并保留同步 `solve_unknown_layout()` 的结果兼容测试。

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -v`

Expected: FAIL，缺少 `UnknownSolveJob` 和增量推进接口。

**Step 3: Write minimal implementation**

把未知搜索主体提取为返回 `AssemblyPlan` 的生成器，在每个经过重叠与优先级计算的候选后让出；新增带 `done/result/search_nodes` 的任务封装。同步函数循环消费生成器。记录首解节点，并在有限纹理优化节点后设置停止标志。

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_edge_features.py -v`

Expected: PASS，增量和同步路径共享相同几何规则。

### Task 2: UNKNOWN稳定运行器非阻塞化

**Files:**
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing tests**

新增带纹理四片运行器测试：第3稳定帧只进入 `is_solving`，不得同步得到最终计划；多帧推进后成功缓存；上下文变化会取消任务。状态选择测试要求求解期间显示 `SOLVING N=`，不能停在 `STABLE 3/3`。

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py -v`

Expected: FAIL，运行器仍在第3帧同步调用完整求解器。

**Step 3: Write minimal implementation**

为 `AssemblyRuntime` 增加任务字段、每帧时间/工作单元预算、求解中状态和取消逻辑。UNKNOWN达到稳定门后创建任务并逐帧推进；KNOWN模板快速匹配保持原路径。`select_planning_status()`优先显示求解进度。

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py -v`

Expected: PASS，第3帧主循环可返回且旧缓存规则保持有效。

### Task 3: KNOWN SAVE跨帧登记

**Files:**
- Modify: `tests_ab/test_a_known_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing tests**

新增 `KnownRegistrationJob` 极小时间片测试和主入口动作测试：点击SAVE后状态为 `SAVE SOLVING` 且文件不存在；连续推进成功后模板文件出现、规划缓存成功；失败或取消时旧模板不变。

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_ab/test_a_known_planner.py -v`

Expected: FAIL，SAVE仍直接调用同步登记函数。

**Step 3: Write minimal implementation**

拆分已知登记的输入准备、未知计划消费和模板完成阶段，供同步函数与 `KnownRegistrationJob` 共用。主循环保存待执行任务，每帧推进一次；成功后调用现有原子保存、更新内存模板并缓存即时KNOWN计划，切换模式和CAL时取消。

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_ab/test_a_known_planner.py tests_ab/test_a_runtime_planner.py -v`

Expected: PASS，SAVE回调不再包含完整搜索。

### Task 4: 文档、版本与发布验证

**Files:**
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`
- Test: `tests_ab/test_variant_packages.py`

**Step 1: Update documentation and version**

写清 `SOLVING`、`SAVE SOLVING`、`SEARCH LIMIT`的含义和实机判断方法；提升A版补丁版本并更新发布测试期望，不修改B版或稳定版文件。

**Step 2: Run focused and full verification**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_a_known_planner.py tests_ab/test_variant_packages.py -v`

Run: `python -m pytest -v`

Run: `python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -v`

Expected: 所有命令退出码0，A版ZIP平铺导入成功。

**Step 3: Verify immutable boundaries and record evidence**

核对稳定版基线9项SHA256与 `docs/plans/2026-07-29-maixcam2-stable-baseline-sha256.md` 一致；核对B版源码和发布包未变化。把测试数量、A版ZIP SHA256和实机待验项写入四文件。当前目录不是有效Git仓库，因此不执行commit/push。
