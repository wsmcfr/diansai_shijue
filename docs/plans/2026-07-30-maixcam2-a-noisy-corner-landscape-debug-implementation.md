# MaixCAM2 A版噪声角点容错、求解日志与横纸诊断 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让5+4+3随机白片在角点误差下快速得到受约束的矩形规划，并提供可关闭的控制台诊断和明确的横纸AUTO反馈。

**Architecture:** `assembly_planner.py`顶部集中三个现场常量，完整布局先走92%严格门，WHITE失败后才走86%容错门与逐片外边硬约束。`AssemblyRuntime`把锁定输入、图路径、兜底拒绝计数和终态写到可关闭日志；AUTO ROI继续使用现有横竖Homography，只增强方向反馈和回归覆盖。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、MaixPy控制台与相机主循环。

---

### Task 1: 真实近似轮廓与负例红灯

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`

**Step 1: Write the failing tests**

增加用户截图近似5边形、4边形、3边形毫米轮廓，断言WHITE成功、目标尺寸在题目范围、诊断含`relaxed_accept=1`且每片都有目标外边。增加相同几何在CARD中仍被严格门拒绝，以及缺少外边的错误紧凑组合不能通过容错层。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "noisy_corner or relaxed_fill" -q`

Expected: FAIL，当前92%门返回`fill_reject`或缺少容错诊断。

### Task 2: 双层验收与有界最佳候选

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `tests_ab/test_a_unknown_planner.py`

**Step 1: Add top-level controls**

在导入区后新增并用中文注释说明：

```python
UNKNOWN_STRICT_MIN_FILL_RATIO = 0.92
UNKNOWN_RELAXED_MIN_FILL_RATIO = 0.86
UNKNOWN_SOLVER_DEBUG = True
```

**Step 2: Implement strict and relaxed validation**

扩展完整布局分析，输出尺寸、填充率、重叠率和目标外边片数。严格验收使用92%；WHITE严格失败后使用86%且要求所有碎片至少一条边落在外框。GRAPH比较最多90组中的最佳容错结果；分段边兜底在首个容错解后继续固定64节点择优，不跑满12000节点。

**Step 3: Run focused GREEN tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "noisy_corner or relaxed_fill or graph_fast_path_rejects" -q`

Expected: 新实机回归通过，既有错误三角形与尺寸负例仍失败。

**Step 4: Run complete planner tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_a_known_planner.py -q`

Expected: 全部通过，KNOWN和CARD没有使用容错门。

### Task 3: 默认开启且可关闭的控制台诊断

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `tests_ab/test_quad_main.py`

**Step 1: Write logging RED tests**

构造开启/关闭两种运行器，开启时断言输出`SNAPSHOT/PIECE/GRAPH/RESULT`以及顶点、边长、拒绝计数；关闭时断言没有`[SOLVER]`。AUTO定位开启时断言输出`[ROI]`方向和四边长度。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py -k "debug_log" -q`

Expected: FAIL，现版没有调试输出接口。

**Step 3: Implement bounded logging**

`AssemblyRuntime`从顶部常量接收调试状态，只在锁定、GRAPH结束、兜底结束或超时打印一次；`UnknownSolveJob`把实时拒绝计数暴露到timeout诊断。`main.py`在AUTO ROI完成后打印方向、四边长度和置信度。关闭开关时所有日志分支立即返回。

**Step 4: Run logging GREEN tests**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py -k "debug_log" -q`

Expected: 开启日志字段完整，关闭时零输出。

### Task 4: 横纸AUTO可见反馈

**Files:**
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `tests_ab/test_variant_calibration_ui.py`
- Modify: `tests_ab/test_paper_locator.py`

**Step 1: Write orientation RED tests**

增加接近用户截图的旋转透视横纸四边形，断言AUTO为landscape；会话应用后状态必须以`AUTO ROI OK H`开头，机械区为`(33.5,0,230,210)`且红线为105mm。

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests_ab/test_variant_calibration_ui.py tests_ab/test_paper_locator.py -k "landscape and auto" -q`

Expected: 判向用例通过，但状态H断言失败。

**Step 3: Implement explicit H/V status**

`CalibrationSession.apply_auto_roi()`在成功状态中加入`H`或`V`，保持失败保留旧ROI、手动PAPER兜底和同方向微调参数不变。

**Step 4: Run orientation GREEN tests**

Run: `python -m pytest tests_ab/test_variant_calibration_ui.py tests_ab/test_paper_locator.py tests_ab/test_a_work_region.py -q`

Expected: 横竖判向、机械区、红线与状态全部通过。

### Task 5: v1.7.0文档、发布与验收

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
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.7.0.zip`

**Step 1: Update version and operator documentation**

记录三个顶部常量的准确文件位置、推荐范围、调整方向、控制台日志字段、横纸H状态和容错层的安全边界；版本提升到1.7.0。

**Step 2: Run targeted and full verification**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py tests_ab/test_variant_calibration_ui.py tests_ab/test_paper_locator.py -q`

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: 全部退出码0且零失败。

**Step 3: Build and verify release**

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: A版v1.7.0 ZIP文件与当前源码逐字节一致，平铺导入通过；B版SHA256不变。

**Step 4: Record evidence**

把定向/全量数量、编译结果、A/B包大小和SHA256写入四文件。根目录不是有效Git仓库，因此不执行commit/push，以测试和哈希作为快照证据。
