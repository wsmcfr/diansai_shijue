# MaixCAM2 A版UNKNOWN WHITE四片快速求解 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为UNKNOWN WHITE四片输入增加支持T形分段接缝的分层Beam快路径，并减少中间候选的栅格重叠成本，使现场四片日志在超时前稳定得到目标位姿。

**Architecture:** WHITE在单个`UnknownSolveJob`内依次执行GRAPH、仅四片启用的`FOUR_FAST`和FALLBACK，三阶段共享跨帧预算与超时；FOUR_FAST逐层扩展两片、三片、四片状态并在每层全局去重、排序和限宽；中间候选优先使用外框、凸多边形或耳切三角形几何交集判断，只有异常多边形回退旧栅格。最终矩形仍使用原严格/容错硬验收。

**Tech Stack:** Python 3、NumPy、OpenCV、MaixPy增量主循环、pytest。

---

### Task 1: 现场四片回归与快速重叠RED测试

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`

**Step 1: Write the failing field-log regression test**

添加固定构造器保存本次U1～U4毫米顶点。测试新入口`_solve_unknown_four_fast_path()`：必须成功返回四个placement，诊断包含分层状态数，`fill_permille >= 920`、`overlap_permille <= 30`，且工作单元低于公开上限。

**Step 2: Write the failing overlap tests**

添加共享边凸矩形、实体重叠凸矩形、2%浅交叠和简单凹多边形用例，要求新快速判断与最终3%总重叠安全门一致，并能通过诊断确认有效多边形没有进入栅格回退。

**Step 3: Write runtime routing tests**

验证UNKNOWN WHITE四片在同一任务内按`GRAPH -> FOUR_FAST -> FALLBACK`路由；FOUR_FAST成功不进入FALLBACK，失败继续FALLBACK；CARD和1～3片不调用FOUR_FAST；关闭宏恢复旧行为。

**Step 4: Run RED tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -k "four_fast or fast_overlap or field_four" -q`

Expected: FAIL，新常量、快路径和路由尚不存在。

---

### Task 2: 快速重叠判断

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Test: `tests_ab/test_a_unknown_planner.py`

**Step 1: Add configurable constants**

在现场调试常量区增加`UNKNOWN_FOUR_FAST_ENABLED`、`UNKNOWN_FOUR_FAST_BEAM_WIDTH`和`UNKNOWN_FOUR_FAST_MAX_WORK_UNITS`，对布尔、正整数和有限值进行入口校验。

**Step 2: Implement geometric helpers**

实现多边形AABB、凸性校验、耳切三角化与快速重叠函数。AABB分离直接返回；双凸或三角形分解使用`cv2.intersectConvexConvex`累计面积；自交、严重共线或无效几何结果调用现有`_candidate_overlaps()`。

**Step 3: Run overlap GREEN tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "fast_overlap" -q`

Expected: PASS，共享边和2%浅交叠留给最终硬门，明显实体重叠拒绝，简单凹多边形不创建MASK。

---

### Task 3: FOUR_FAST分层Beam核心

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Test: `tests_ab/test_a_unknown_planner.py`

**Step 1: Implement state representation and deduplication**

状态保存`placed_by_index`、剩余索引、累计接缝优先级和量化位姿键。位姿键包含已放置碎片索引和0.01mm量化后的顶点，保证不同生成关系得到相同布局时只保留一次。

**Step 2: Implement layer expansion**

固定第0片，针对剩余碎片查询现有整边/分段兼容图，生成一至两个端点锚定候选；执行快速重叠、部分外框安全剪枝、去重和统一排序。每层全局保留并在预算内实际展开最多`UNKNOWN_FOUR_FAST_BEAM_WIDTH`个状态；中间层达到工作单元上限时安全进入旧FALLBACK，完整层触顶时仍验收已经生成的候选。

**Step 3: Implement complete validation**

四片完整状态调用现有严格验收；只有严格填充不足时调用WHITE容错验收。严格首解立即构造规划；容错候选在当前完整层中选择最低几何分。所有机械目标继续由`_build_unknown_success_plan()`生成。

**Step 4: Expose structured diagnostics**

返回`pairs`、`triples`、`complete`、`dedupe`、`overlap_reject`、`work_units`和验收比例等整数诊断，便于运行器日志和测试读取。

**Step 5: Run field-log GREEN tests**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "four_fast or field_four" -q`

Expected: PASS，现场四片在工作单元上限内得到严格成功结果。

---

### Task 4: 运行器路由与调度

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `tests_ab/test_quad_main.py`

**Step 1: Route WHITE four-piece snapshots**

达到稳定门后只创建一个`UnknownSolveJob`。WHITE任务先增量运行GRAPH；GRAPH失败且恰好四片、开关启用时增量运行FOUR_FAST；两个快路径都无解时在同一任务和同一超时预算内继续FALLBACK。同步`solve_unknown_layout()`完整消费同一生成器。

**Step 2: Add lazy FOUR_FAST debug log**

仅`UNKNOWN_SOLVER_DEBUG=True`时输出分层计数、拒绝数、工作单元和耗时；关闭时不格式化字符串。

**Step 3: Adjust solver scheduling guards**

把活动计算保护提高到8秒、墙钟提高到30秒；`AssemblyRuntime`每帧最多推进24ms或64个工作单元，GRAPH布局、FOUR_FAST候选和FALLBACK候选都必须处于该调度边界内，不改变触摸与显示主循环。

**Step 4: Run routing GREEN tests**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py -k "four_fast or unknown or solving" -q`

Expected: PASS，成功短路、失败兜底、CARD隔离和状态日志均正确。

---

### Task 5: 回归、文档和v1.9.1发布

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.9.1.zip`

**Step 1: Update operator documentation**

说明四片日志字段、三个新宏、严格与容错硬门保持不变、FOUR_FAST失败后的FALLBACK流程，以及现场如何判断速度瓶颈。

**Step 2: Run targeted verification**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py -q`

Expected: 全部通过。

**Step 3: Run full and compile verification**

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: 退出码均为0。

**Step 4: Build and verify release**

记录B版ZIP SHA256，运行`python tools/package_variants.py`，再运行`python -m pytest tests_ab/test_variant_packages.py -q`。核对A版ZIP平铺文件与源码逐字节一致，B版打包前后SHA256不变。

**Step 5: Record delivery evidence**

记录专项/全量测试数量、现场四片诊断、A版ZIP大小与SHA256。根目录不是有效Git仓库，因此不执行commit/push。
