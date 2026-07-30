# MaixCAM2 A版FOURFAST独立活动预算 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为四片FOURFAST增加1.5秒独立活动计算预算，使无解或慢输入及时进入FALLBACK，同时保持1～3片路径与UNKNOWN统一预算不变。

**Architecture:** `UnknownSolveJob.advance()`继续作为唯一实际CPU计时边界，只在当前阶段为`four_fast`时累计子预算；达到上限后通过共享`progress`请求FOURFAST生成器在下一个工作单元边界正常结束。流水线把该结束当作快路径无解，输出阶段事件后进入原FALLBACK。

**Tech Stack:** Python 3、NumPy、OpenCV、MaixPy跨帧生成器、pytest。

---

### Task 1: 子预算与三片隔离RED测试

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`

**Step 1:** 使用假时钟和真实FOURFAST核心，使两个工作单元累计超过1秒；断言阶段诊断为时间上限、任务未超时且后续启动FALLBACK。

**Step 2:** 构造三片任务并配置极小FOURFAST子预算；断言FOURFAST未调用，原FALLBACK仍返回成功。

**Step 3:** 增加子预算内成功与非法预算参数测试。

**Step 4:** 运行新增节点，确认因缺少构造参数、常量和中止行为按预期失败。

### Task 2: 最小生产实现

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1:** 新增`UNKNOWN_FOUR_FAST_ACTIVE_BUDGET_SECONDS = 1.5`和构造参数校验。

**Step 2:** 在`advance()`每个工作单元结束后，仅按`current_stage == four_fast`累计活动秒数；达到上限时设置共享中止标志。

**Step 3:** FOURFAST核心在候选生成和完整验收边界读取中止标志，写入`four_active_limit_reached`、`four_active_elapsed_ms`并正常返回无解。

**Step 4:** 运行Task 1测试转绿，再运行UNKNOWN/运行器专项。

### Task 3: 日志、版本和文档

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `README.md`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`

**Step 1:** FOURFAST日志增加`time_limit`和`active_ms`字段及测试。

**Step 2:** 手册说明四片子预算、三片零影响和现场调整范围。

**Step 3:** 版本提升为v1.9.2并同步发布清单。

### Task 4: 回归与发布

**Files:**
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.9.2.zip`

**Step 1:** 运行A版UNKNOWN、运行器和主循环专项。

**Step 2:** 运行`python -m pytest -q tests tests_ab`与A版compileall。

**Step 3:** 记录B版哈希，重新打包并运行发布测试，确认B版哈希不变。

**Step 4:** 写入测试数量、A包大小/SHA256和实机下一步。

