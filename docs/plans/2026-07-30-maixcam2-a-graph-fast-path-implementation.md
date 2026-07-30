# MaixCAM2 A版 GRAPH_AUTO 快路径 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在A版WHITE未知拼装中增加有界连接图快路径，使倾斜相机经Homography校正后的1～4片毫米轮廓能快速求解，并保留现有硬验收与搜索兜底。

**Architecture:** 视觉层继续把相机顶点批量反算为A4毫米坐标；规划层先构造最多32条无向边候选和90组连通匹配，BFS传播刚体位姿后复用现有完整布局硬验收。WHITE首个图解立即返回，图路径无解才创建现有生成器任务；CARD与KNOWN不改。

**Tech Stack:** MaixPy、Python、OpenCV、NumPy、pytest、MaixVision平铺ZIP。

---

### Task 1: 倾斜A4 Homography求解不变性

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_known_planner.py`

**Step 1: Write the failing tests**

新增同一组三片物理轮廓分别映射到正视和明显倾斜`paper_quad`，再用
`image_points_to_paper_mm()`反算的测试；要求毫米顶点误差在既有容差内，两个输入的
WHITE规划目标尺寸、片数和成功状态一致。测试同时确认相机坐标没有直接进入求解器。

**Step 2: Run tests to verify RED or characterize existing support**

Run: `python -m pytest tests_ab/test_a_known_planner.py tests_ab/test_a_unknown_planner.py -k "homography and graph" -q`

Expected: Homography映射基础测试可能直接PASS；GRAPH路径断言必须因缺少诊断字段而FAIL。

**Step 3: Preserve the existing mapping contract**

若Homography测试通过，不改`paper_locator.py`生产代码，只把它作为快路径输入契约；若
发现批量反算误差，则先针对矩阵方向或角点排序增加最小修复，不能在规划器内重复透视。

**Step 4: Run mapping tests**

Run: `python -m pytest tests_ab/test_paper_locator.py tests_ab/test_a_known_planner.py -q`

Expected: PASS。

### Task 2: 32边/90组合的连接图核心

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

增加确定性1.0mm和1.5mm顶点误差三片场景，要求内部图核心返回候选布局；增加候选边
上限32、组合上限90、每条边只用一次、图必须连通的测试。增加断开图返回空列表测试。

**Step 2: Verify RED**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "graph_candidate or graph_matching or graph_propagation" -q`

Expected: FAIL，图核心接口尚不存在。

**Step 3: Implement the bounded graph core**

在`assembly_planner.py`增加中文注释完整的内部函数：

- `_build_graph_edge_candidates()`：只枚举不同碎片，按相对边长误差排序并截断32条。
- `_collect_graph_matching_sets()`：枚举`N-1`及可选`N`条边组合，执行边唯一、度数和连通约束，截断90组。
- `_align_edge_midpoint_pose()`：反向边中点对齐，不缩放、不镜像。
- `_propagate_graph_layout()`：BFS传播位姿并计算闭合误差。
- `_solve_unknown_graph_fast_path()`：排序布局并调用现有毫米硬验收，返回规划或None及整数诊断。

**Step 4: Verify GREEN**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "graph_candidate or graph_matching or graph_propagation" -q`

Expected: PASS。

### Task 3: WHITE优先接入与原搜索兜底

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

新增WHITE在第一稳定求解帧优先得到`graph_fast_path=1`的测试；新增图路径故意无解时仍
创建原`UnknownSolveJob`的测试；新增CARD不调用WHITE图快路径的测试。

**Step 2: Verify RED**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -k "graph_fast_path or graph_fallback or card_skips_graph" -q`

Expected: FAIL，运行器尚未调用图快路径。

**Step 3: Implement minimal runtime wiring**

`AssemblyRuntime.update()`在UNKNOWN、WHITE且稳定门满足时先同步执行有界图快路径；成功
直接缓存规划并返回，失败才按现有方式构造`UnknownSolveJob`。图路径只执行固定32/90
工作量，不共享或污染后续生成器状态。诊断合并到成功规划。

**Step 4: Verify GREEN and regressions**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -q`

Expected: 全部PASS。

### Task 4: 鲁棒性、性能和发布收尾

**Files:**
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `README.md`
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`
- Modify: `硬件资源表.md`

**Step 1: Add failure-prevention tests**

覆盖1～4片、明显非矩形、错误尺寸、重叠、1～2mm确定性噪声和候选上限；增加PC对照探针，
要求图路径检查布局数不超过90，并比同输入旧搜索节点数更低。不得用宽松墙钟作为唯一
正确性断言。

**Step 2: Run focused verification**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_a_known_planner.py -q`

Expected: PASS。

**Step 3: Update release documentation**

记录GRAPH_AUTO诊断、Homography倾斜适用范围、镜头畸变/高度视差边界、WHITE快路径和
实机五点100×60mm校验步骤。完成此前暂缓的A版v1.4.1清单与ZIP更新；稳定版和B版业务
源码不修改。

**Step 4: Run full verification**

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: 全部退出码0。

**Step 5: Verify release boundaries**

核对稳定版9个Python/YAML SHA256与既有基线一致，确认B版ZIP仍为
`F08BFE4183241399073DA4DB6C16778678498E9214D62CC755EC22C37481ED92`，记录新的A版
v1.4.1 ZIP大小和SHA256。根目录不是有效Git仓库，因此不执行commit/push。

