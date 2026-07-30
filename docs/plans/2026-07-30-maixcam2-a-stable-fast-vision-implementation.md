# MaixCAM2 A版高稳定快速拼图 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让远距离固定MaixCAM2稳定分开2mm黑缝，提供可量化的毫米标定质量，KNOWN从下半区正确布局瞬时登记，并让UNKNOWN四片在受控时间内完成。

**Architecture:** 高分辨率采集坐标与640x480显示坐标分离，设置使用归一化ROI；分割改为不扩张外轮廓的连通域填孔；KNOWN直接登记下半区目标；UNKNOWN使用预计算边候选和限宽图搜索。所有运行路径继续输出现有 `AssemblyPlan`，保持下半区绘制和机械接口兼容。

**Tech Stack:** MaixPy、OpenCV、NumPy、pytest、MaixVision平铺ZIP。

---

## 实施清单

| 轮次 | 目标 | 验证标准 | 审查 |
|---|---|---|---|
| Round 1 | 高分辨率坐标与防粘连分割 | 2mm等效黑缝保持两个连通域，旧设置可迁移 | `review:true` |
| Round 2 | MEASURE标定质量与稳定诊断 | SCALE/GAP/JITTER/RECT均有确定PASS/FAIL测试 | `review:true` |
| Round 3 | 下半区KNOWN直接登记 | SAVE不创建搜索任务并立即生成四模板和计划 | `review:true` |
| Round 4 | UNKNOWN快速边图、全量验证与发布 | 复杂四片性能基准、全量测试、编译和ZIP导入通过 | `review:true` |

### Task 1: 高分辨率坐标与防粘连分割

**Files:**
- Modify: `tests/test_settings_store.py`
- Modify: `tests/test_puzzle_vision.py`
- Modify: `maixcam2_app_A_quad/config.py`
- Modify: `maixcam2_app_A_quad/settings_store.py`
- Modify: `maixcam2_app_A_quad/puzzle_vision.py`

**Step 1:** 新增失败测试，覆盖1280x960默认采集、归一化V4纸张坐标、V3的640x480像素坐标迁移，以及模拟2mm缝隙经过默认分割后仍返回两个轮廓。

**Step 2:** 运行 `python -m pytest tests/test_settings_store.py tests/test_puzzle_vision.py tests_ab/test_a_quad_mask.py -v`，确认失败原因来自缺少新行为。

**Step 3:** 实现设置V4迁移和采集/显示尺寸配置；把默认模糊改为3、闭运算改为1；增加只填内部孔洞而不扩张外边界的掩膜处理。

**Step 4:** 重跑定向测试并执行A版 `compileall`，要求全部通过且无导入错误。

### Task 2: MEASURE标定质量与稳定诊断

**Files:**
- Modify: `tests_ab/test_calibration_quality.py`
- Modify: `tests_ab/test_calibration_ui.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1:** 新增失败测试，构造已知A4四角、两个相邻碎片和10帧观测，断言SCALE、GAP、JITTER和100x60标准片尺寸的数值与质量状态。

**Step 2:** 运行 `python -m pytest tests_ab/test_calibration_quality.py tests_ab/test_calibration_ui.py -v`，确认MEASURE页面和质量指标尚不存在。

**Step 3:** 实现纯计算质量对象、10帧稳定窗口和MEASURE绘制；主循环把高分辨率分析结果映射到640x480预览，触摸仍使用显示坐标。

**Step 4:** 重跑定向测试，并用合成1280x960帧验证预览尺寸固定为640x480、所有文字位于状态栏内。

### Task 3: 下半区KNOWN直接登记

**Files:**
- Modify: `tests_ab/test_a_known_planner.py`
- Modify: `tests/test_assembly_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1:** 新增失败测试：四片正确布局位于下半区时同步登记成功；上半区随机片返回 `known_layout_must_be_lower`；KNOWN SAVE不创建 `KnownRegistrationJob`，成功后立即写模板并缓存计划。

**Step 2:** 运行 `python -m pytest tests_ab/test_a_known_planner.py tests/test_assembly_planner.py -v`，确认旧代码仍筛选上半区并启动UNKNOWN搜索。

**Step 3:** 重写登记入口为下半区联合布局验收与目标归一化；SAVE按钮直接调用登记、原子保存和缓存，删除主循环对KNOWN SAVE增量任务的依赖，但保留旧模板失败保护。

**Step 4:** 重跑定向测试，记录登记墙钟时间，并验证随机上半区KNOWN模板匹配仍输出100x60下半区目标。

### Task 4: UNKNOWN快速边图、发布与实机文档

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`

**Step 1:** 新增失败测试和性能基准，覆盖2到4片、中心共点、条带、相同矩形、T形分段边、凹片、纹理消歧和无解墙钟超时；复杂四片PC目标小于100ms。

**Step 2:** 运行UNKNOWN定向测试并记录旧算法耗时和失败用例，确认新测试能捕获当前实机慢路径。

**Step 3:** 实现预计算边候选、每碎片对限额、限宽状态搜索、非栅格部分剪枝、完整布局一次栅格验收和5秒硬超时；保留 `UnknownSolveJob` 外部接口，进度改为候选/状态/耗时。

**Step 4:** 更新手册、A版版本和发布白名单，运行：

```text
python -m pytest tests tests_ab -q
python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools
python tools/package_variants.py
python -m pytest tests_ab/test_variant_packages.py -v
```

**Step 5:** 核对稳定版9项SHA256和B版ZIP哈希，记录新A版ZIP大小、SHA256、PC性能与必须实机验证的高分辨率支持和实际帧率。当前 `.git` 目录不是有效仓库，不执行提交或推送。
