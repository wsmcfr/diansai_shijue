# MaixCAM2 A版自动拼装修复 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复UNKNOWN识别四片后不显示目标，以及KNOWN无法从上半区随机碎片保存并规划的问题。

**Architecture:** 在 `assembly_planner.py` 中增强统一几何求解和结构化拒绝诊断，在 `main.py` 中把KNOWN保存改为“上半区自动求解后登记”，并修复触摸状态被规划状态覆盖的问题。所有行为先由纯PC单元测试固定，再接入MaixCAM2主循环。

**Tech Stack:** Python、NumPy、OpenCV、pytest、MaixPy运行时接口

---

### Task 1: 建立规划器失败复现测试

**Files:**
- Create: `tests/test_assembly_planner.py`
- Test: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

构造四片可组成100×60mm矩形的毫米多边形，随机旋转和平移后送入UNKNOWN求解器。再构造一条长接缝对应两条共线短接缝，以及顶点带小量毫米误差的输入，断言均返回成功并包含四个目标位姿。

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_assembly_planner.py -v`

Expected: 当前整边严格匹配实现至少有一组返回 `no_assembly`。

**Step 3: Add rejection diagnostics tests**

分别构造尺寸不合法、填充率不足和必然重叠的组合，断言失败原因不是笼统的 `no_assembly`，而是稳定的结构化原因。

**Step 4: Run focused tests**

Run: `python -m pytest tests/test_assembly_planner.py -v`

Expected: 新断言先失败，证明测试覆盖现场症状。

### Task 2: 增强UNKNOWN几何求解

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Test: `tests/test_assembly_planner.py`

**Step 1: Implement diagnostic result tracking**

让完整布局验收返回明确拒绝类别，并在搜索过程中累计边候选、尺寸拒绝、填充拒绝、重叠拒绝数量。搜索结束时按证据选择最具体的 `AssemblyPlan.reason`。

**Step 2: Implement tolerant seam candidates**

保留整边快速路径，增加端点到线段的分段共线拼接候选。候选必须保持刚体变换、相反侧放置、不镜像，并通过重叠剪枝。

**Step 3: Run focused tests**

Run: `python -m pytest tests/test_assembly_planner.py -v`

Expected: 合法整边、分段边和测量误差输入全部PASS；无解输入返回具体原因。

### Task 3: 改造KNOWN上半区自动保存

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Test: `tests/test_assembly_planner.py`
- Test: `tests/test_main_import_guard.py`

**Step 1: Write failing KNOWN registration tests**

测试四片 `region=upper` 时能够从自动求解结果生成四个带 `target_vertices_mm` 的模板；少片、跨线、无解时拒绝且不写入模板。

**Step 2: Implement solved-layout registration**

增加从成功 `AssemblyPlan` 生成KNOWN模板的纯函数。目标轮廓统一规范到100×60mm，模板ID与形状描述子稳定绑定。

**Step 3: Update SAVE flow**

主循环的 `SAVE` 只取上半区完整碎片，先求解、再保存、重置并立即生成KNOWN规划。保存异常保留旧模板与具体屏幕原因。

**Step 4: Run focused tests**

Run: `python -m pytest tests/test_assembly_planner.py tests/test_main_import_guard.py -v`

Expected: PASS。

### Task 4: 修复状态优先级与完成回归

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Test: `tests/test_main_import_guard.py`

**Step 1: Write failing status-priority test**

把单帧状态选择提取为纯函数，断言KNOWN保存成功或失败文字不会被同一帧规划状态覆盖。

**Step 2: Implement status selection and documentation**

增加触摸动作状态优先级，文档改为上半区自动SAVE流程，并补充UNKNOWN具体拒绝原因对照表。

**Step 3: Run full verification**

Run: `python -m pytest -q`

Expected: 全部测试PASS。

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: exit code 0。

Run: `python -c "import runpy; runpy.run_path('maixcam2_app_A_quad/main.py', run_name='flat_import_check')"`

Expected: exit code 0，且不启动设备主循环。

### Task 5: 生成A版设备发布包

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.3.0.zip`

**Step 1: Update version metadata**

把A版版本提升到 `v1.3.0`，不修改稳定版和B版目录。

**Step 2: Build flat MaixVision archive**

ZIP根目录直接包含 `main.py`、规划器、视觉模块、配置、UI、存储模块、`app.yaml` 和调试手册，不能额外嵌套工程目录。

**Step 3: Verify archive**

检查ZIP文件列表和解压后的平铺导入，再运行全量pytest与compileall。

> 当前工作区不是Git仓库，因此本计划不执行commit步骤；所有改动通过文件差异和测试结果验收。
