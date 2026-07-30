# MaixCAM2 A版故障调试手册增补 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 扩充A版实机调试手册，使现场人员能根据状态文字定位故障，并明确KNOWN保存后显示下半区目标的完整条件。

**Architecture:** 保留现有手册按标定、分割、模式和坐标组织的主结构，在KNOWN章节就近补充两阶段工作流和截图专项诊断，再在状态表之前增加统一故障决策流程。所有结论直接对应A版v1.2.0的区域分类、规划过滤和绘制条件。

**Tech Stack:** Markdown、MaixCAM2 A版v1.2.0界面状态、OpenCV轮廓与规划逻辑

---

### Task 1: 补充KNOWN保存与规划两阶段流程

**Files:**
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md:279`

**Step 1: 增加保存行为说明**

明确`SAVE`只登记红线下方的正确布局并清除旧规划，不会立即在下半区绘制目标。

**Step 2: 增加规划行为说明**

明确保存成功后必须把四片全部移到红线上方，且每片全部顶点位于红线以上；连续稳定3帧并显示`PLAN OK N=4`后才绘制目标。

**Step 3: 增加当前截图专项诊断**

增加`KNOWN N=4 EDGE=0 ... PLAN KNOWN_NEEDS_FOUR`说明：`N=4`统计所有完整轮廓，规划只使用`complete=True && region=upper`，因此二者可以同时出现。

**Step 4: 检查关键文字**

Run: `rg -n "保存与规划是两个阶段|PLAN KNOWN_NEEDS_FOUR|region=upper|PLAN OK N=4" "maixcam2_app_A_quad\\A版实机调试手册.md"`

Expected: 四类关键文字均有匹配。

### Task 2: 增加现场故障决策流程

**Files:**
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md:398`

**Step 1: 增加分层定位顺序**

按原始画面、ROI、MASK、RESULT、区域分类、模板、规划、绘制八层组织，说明每层通过标准和下一步。

**Step 2: 增加常见现象调试表**

覆盖AUTO ROI失败、碎片粘连、漏检、SMALL、EDGE、LARGE/BACKGROUND、N正确但规划失败、KNOWN模板失败、UNKNOWN无解和下半区不显示目标。

**Step 3: 增加一次只改一项的现场复测规则**

规定每次记录页面、状态、TH/MIN/OPEN/CLOSE、碎片位置和修改结果，避免同时调多个参数。

**Step 4: 检查故障覆盖**

Run: `rg -n "AUTO ROI FAIL|粘连|SMALL|EDGE|BACKGROUND|KNOWN_NEEDS_FOUR|KNOWN_MATCH_FAILED|NO_ASSEMBLY|下半区没有" "maixcam2_app_A_quad\\A版实机调试手册.md"`

Expected: 每类现场故障均有对应操作。

### Task 3: 校验文档与程序一致性

**Files:**
- Verify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Reference: `maixcam2_app_A_quad/main.py`
- Reference: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: 核对状态统计**

确认手册中的`N`对应`complete=True`总数，`EDGE`对应不完整有效轮廓数量。

**Step 2: 核对区域过滤**

确认手册写明KNOWN规划只接受四片`region=upper`，跨线或下半区碎片不参与。

**Step 3: 核对绘制条件**

确认手册写明失败或未求解时只保留红线，成功规划才绘制目标。

**Step 4: 执行Markdown交付检查**

Run: `rg -n "N=4.*不等于|crossing|KNOWN LAYOUT SAVED|STABLE 1/3|紫色目标外框|独立黑底" "maixcam2_app_A_quad\\A版实机调试手册.md"`

Expected: 全部关键行为均在手册中出现；本任务为文档修改，不运行Python测试。

**Step 5: 版本控制说明**

当前`F:\diansia`不是Git仓库，因此不执行提交；保留设计和实施计划文件作为变更依据。
