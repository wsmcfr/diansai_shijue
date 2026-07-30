# MaixCAM2 A版单次快照与纸面专用视图 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 连续三帧后锁定一次识别结果完成求解，并让正常界面只显示已标定A4纸面。

**Architecture:** `AssemblyRuntime`持有不可被后续帧替换的碎片快照和终态规划；`main.py`把锁定快照用于正常叠加，并通过独立显示Homography把原相机纸面映射到等比例屏幕内容区。检测链路继续使用现有原图四边形掩膜和毫米Homography。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、MaixPy相机/屏幕/触摸接口。

---

### Task 1: 锁定运行器快照

**Files:**
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing test**

增加测试，输入三帧稳定碎片后再传入明显移动的轮廓，断言活动任务和最终规划仍使用第3帧深复制快照；模拟`solver_timeout`后断言失败被缓存、稳定计数不清零且下一帧不创建新任务。

**Step 2: Run test to verify it fails**

Run: `pytest -q tests_ab/test_a_runtime_planner.py -k "locked_snapshot or timeout_is_locked"`

Expected: FAIL，现版没有公开锁定快照且超时会清空稳定门。

**Step 3: Write minimal implementation**

在`AssemblyRuntime`中增加锁定碎片深复制、只读访问属性和统一清理；稳定门达到后立即保存快照，KNOWN/UNKNOWN都只消费该快照；超时失败写入`plan`，不再自动重新采集。

**Step 4: Run test to verify it passes**

Run: `pytest -q tests_ab/test_a_runtime_planner.py -k "locked_snapshot or timeout_is_locked"`

Expected: PASS。

### Task 2: A4纸面专用显示

**Files:**
- Modify: `tests_ab/test_quad_main.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing test**

增加竖放、横放和透视场景测试，断言640×480输出只在等比例A4内容矩形内保留相机像素，内容区外全黑；叠加轮廓和纸面边界使用同一显示变换。

**Step 2: Run test to verify it fails**

Run: `pytest -q tests_ab/test_quad_main.py -k "paper_display or paper_only"`

Expected: FAIL，现版直接缩放整幅相机图。

**Step 3: Write minimal implementation**

增加纸面显示画布及点变换辅助函数；`draw_overlay()`在纸张已锁定时使用显示Homography，在未锁定时保留整帧回退；所有轮廓、顶点、中心、黄色区、红线和目标规划统一映射。

**Step 4: Run test to verify it passes**

Run: `pytest -q tests_ab/test_quad_main.py -k "paper_display or paper_only"`

Expected: PASS。

### Task 3: 主循环重新采集与锁定状态

**Files:**
- Modify: `tests_ab/test_quad_main.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing test**

增加状态文字和入口AST契约测试，断言求解时显示`LOCKED`、正常叠加优先使用`planner_runtime.locked_pieces`、重复点击当前模式也调用`reset()`。

**Step 2: Run test to verify it fails**

Run: `pytest -q tests_ab/test_quad_main.py tests_ab/test_a_runtime_planner.py -k "locked or recapture"`

Expected: FAIL，现版只在模式改变时reset且叠加始终使用实时pieces。

**Step 3: Write minimal implementation**

扩展状态选择函数和主循环：同模式点击触发重新采集，规划更新后选择锁定或实时碎片用于显示，CAL与WHITE/CARD切换继续清除快照。

**Step 4: Run test to verify it passes**

Run: `pytest -q tests_ab/test_quad_main.py tests_ab/test_a_runtime_planner.py -k "locked or recapture"`

Expected: PASS。

### Task 4: 文档、版本与发布验证

**Files:**
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `README.md`
- Modify: `tests_ab/test_variant_packages.py`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.6.0.zip`

**Step 1: Update operator documentation and package expectations**

记录`SEARCHING/LOCKED`、同模式重新采集、纸面专用视图和失败不自动重试；版本提升到v1.6.0。

**Step 2: Run targeted and full verification**

Run: `pytest -q tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py tests_ab/test_quad_vision.py`

Run: `pytest -q`

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: 全部退出码0且零失败。

**Step 3: Build and verify release archive**

Run: `python tools/package_variants.py`

Run: `pytest -q tests_ab/test_variant_packages.py`

Expected: A版v1.6.0平铺ZIP存在、文件清单完整、解压导入通过；稳定版与B版边界测试不变。

**Step 4: Record evidence**

把测试数量、编译结果、发布包大小与SHA256写入四文件工作记忆。当前根目录不是有效Git仓库，因此跳过commit并明确记录原因。
