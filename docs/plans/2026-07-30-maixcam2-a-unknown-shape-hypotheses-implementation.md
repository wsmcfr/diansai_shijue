# MaixCAM2 A版UNKNOWN多轮廓候选求解 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在完全不使用KNOWN模板的前提下，让A版UNKNOWN根据每片完整轮廓和最多三个接缝候选稳定求解1～4片，并修复伪短边导致的四片超时。

**Architecture:** 视觉层保留完整像素轮廓并生成全部3～5边拟合候选，`main.py`统一转换为毫米数据；求解器以完整轮廓质心为公共原点，候选边只提出刚体位姿，完整轮廓负责中间重叠和最终硬验收；GRAPH、FOURFAST和FALLBACK共享带候选等级的同一几何结构，并在单个`UnknownSolveJob`预算内逐级开放候选。

**Tech Stack:** Python 3、NumPy、OpenCV、MaixPy相机主循环、pytest、ZIP发布脚本。

---

### Task 1: 建立伪角点和模板隔离RED基线

**Files:**
- Modify: `tests_ab/test_quad_vision.py`
- Modify: `tests_ab/test_quad_main.py`
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`

**Step 1: 添加视觉候选失败测试**

构造一个真实100×60mm四边形的像素轮廓，在两个角附近加入2.9～7.7mm等效短边和轻微反光凹口。断言新的候选接口返回去重后的3～5边集合，且至少存在一个四边形候选；同时增加真实三角形和真实五边形，断言候选不会统一退化成三角形。

**Step 2: 添加毫米数据流失败测试**

调用`analyze_quad_frame()`，断言每片同时具有`outline_mm`、`shape_hypotheses_mm`和与候选数量一致的纹理特征组；逐点验证它们都使用当前横/竖纸Homography，而不是ROI比例换算。

**Step 3: 添加四片UNKNOWN失败回归**

以默认四片真实外形生成高保真轮廓，再向接缝候选注入本次日志中的伪短边。断言UNKNOWN WHITE成功、四个placement齐全、严格或容错硬验收合格，并记录候选等级诊断。生产函数只接收碎片几何，测试不得把KNOWN模板传给求解器。

**Step 4: 添加模板禁止调用测试**

使用`monkeypatch`把`solve_known_layout()`、`match_known_pieces()`以及模板加载入口替换为调用即抛出`AssertionError`，再推进UNKNOWN任务到成功或确定失败，证明UNKNOWN路径没有访问模板。

**Step 5: 把错误的旧预期改为RED断言**

替换`test_graph_cleans_white_but_both_fallback_profiles_preserve_original`：新断言要求GRAPH、FOURFAST和FALLBACK均收到同一个包含`outline_local`和`hypotheses`的求解结构，不再检查三个不同的`clean_short_edges`布尔值。

**Step 6: 运行RED测试**

Run: `python -m pytest tests_ab/test_quad_vision.py tests_ab/test_quad_main.py tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -k "shape_hypothesis or noisy_vertex or unknown_never_uses_known or unified_solver_geometry" -q`

Expected: FAIL，候选接口、毫米字段和统一求解结构尚不存在；失败不得来自测试夹具语法或导入错误。

---

### Task 2: 视觉多边形候选与毫米映射

**Files:**
- Modify: `maixcam2_app_A_quad/puzzle_vision.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_quad_vision.py`
- Modify: `tests_ab/test_quad_main.py`

**Step 1: 实现全部epsilon候选收集**

新增`approximate_polygon_candidates(contour, config)`：完整遍历`approx_epsilon_min..max`，只收集3～5顶点简单多边形；按规范化顶点和边界距离去重并保留epsilon元数据。函数必须有中文函数注释，说明输入、遍历流程、去重规则和返回顺序。

现有`approximate_polygon()`改为调用候选接口并返回兼容主候选；没有合法候选时继续返回顶点数最接近的异常显示轮廓，保证KNOWN显示和旧调用不崩溃。

**Step 2: 扩展单片检测结果**

`compute_piece_geometry()`保留原`contour`和`vertices`字段，并新增`shape_hypotheses_px`。候选使用独立列表，不能让后续排序原地修改显示主轮廓。

**Step 3: 映射完整轮廓和全部候选**

`analyze_quad_frame()`使用`image_points_to_paper_mm()`把`piece["contour"]`转换为`outline_mm`，把每个`shape_hypotheses_px`转换为`shape_hypotheses_mm`。所有字段在锁定快照前完成，求解期间不再读取新帧。

**Step 4: 为CARD候选采样边特征**

复用`sample_piece_edge_features()`为每个像素候选生成`shape_edge_features`。现有`edge_features`保留为主候选兼容字段；WHITE虽然不比较纹理，也不能让字段长度与候选边数错配。

**Step 5: 运行视觉GREEN测试**

Run: `python -m pytest tests_ab/test_quad_vision.py tests_ab/test_quad_main.py tests_ab/test_a_edge_features.py -k "shape_hypothesis or noisy_vertex or edge_feature" -q`

Expected: PASS；同一伪角点轮廓包含正确四边形候选，横竖纸毫米映射均正确。

---

### Task 3: 高保真轮廓与候选评分结构

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`

**Step 1: 增加集中调试宏**

在现有UNKNOWN现场常量区新增并用中文说明：

```python
UNKNOWN_SHAPE_MAX_HYPOTHESES = 3
UNKNOWN_REAL_EDGE_MIN_MM = 20.0
UNKNOWN_EDGE_HARD_FLOOR_MM = 14.0
UNKNOWN_SHAPE_AREA_RETENTION_MIN = 0.96
UNKNOWN_SHAPE_MAX_DEVIATION_MM = 3.0
```

所有入口校验有限值、正数关系和`HARD_FLOOR <= REAL_EDGE_MIN`；非法值返回结构化几何失败，不能把异常泄漏到相机主循环。

**Step 2: 实现轮廓质量计算**

新增完整轮廓高保真压缩、点到候选闭合边界最大距离、面积保持率、短边惩罚、尖刺/共线惩罚和候选归一化去重辅助函数。高保真压缩只有达到99%面积保持时才能采用，否则保留完整输入。

**Step 3: 实现候选构造器**

新增`_build_solver_shape_hypotheses(piece)`：优先读取`shape_hypotheses_mm`，旧测试夹具缺少该字段时把`vertices_mm`作为单候选兼容输入；候选经过硬门、评分、去重后最多保留三个。若全部候选被拒绝，抛出可由求解入口转换为`shape_hypothesis_empty`的内部几何错误。

**Step 4: 重构`_solver_piece()`**

删除`clean_short_edges`参数和三条路径的布尔分支。新结构统一返回`outline_local`、`source_center`、`hypotheses`和候选诊断。所有候选与完整轮廓必须减去同一个完整轮廓质心，不能分别使用候选质心。

旧`_clean_solver_short_edges()`暂时保留为不再被生产路径调用的兼容辅助函数，待全量测试证明没有外部依赖后再决定删除，避免本轮混入无关清理。

**Step 5: 运行几何GREEN测试**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -k "shape_hypothesis or unified_solver_geometry or cleanup or noisy_vertex" -q`

Expected: PASS；真实三至五边保留，伪短边候选降级或拒绝，三条求解入口不再传`clean_short_edges`。

---

### Task 4: 候选刚体变换与统一接缝图

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `tests_ab/test_a_unknown_planner.py`

**Step 1: 把接缝对齐改为返回位姿**

新增`_edge_alignment_pose()`，返回二维旋转矩阵和平移向量；新增`_transform_solver_outline()`把同一位姿应用到简化候选和`outline_local`。共享边方向必须保持反向，无缩放、无镜像。

**Step 2: 扩展接缝关系标识**

`_build_graph_edge_candidates()`和`_build_edge_compatibility_graph()`为每条关系记录`source_hypothesis`、`target_hypothesis`、各自边号、候选最高等级和候选总分。关系排序首先比较候选等级，再比较现有长度误差、纹理和整边/分段优先级。

**Step 3: 约束同片候选一致性**

搜索状态新增`hypothesis_by_index`。放置新片时可以选择尚未确定的候选；已经放置的碎片只能继续使用状态中记录的候选编号。同一状态中发现候选冲突必须立即拒绝。

**Step 4: 更新位姿去重键**

去重键加入每片候选编号和量化刚体位姿。两种候选即使外框接近，也不能在最终硬验收前被错误合并。

**Step 5: 运行接缝图测试**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py -k "hypothesis and (edge_graph or alignment or pose or dedupe)" -q`

Expected: PASS；候选边产生正确无镜像位姿，同片候选不混用，重复状态受控。

---

### Task 5: GRAPH、FOURFAST和FALLBACK分级搜索

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `tests_ab/test_a_unknown_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`

**Step 1: 统一状态中的两套多边形**

每个已放置片同时保存用于提出接缝的`seam_polygon`和用于重叠/验收的`outline_polygon`。中间AABB、凸交集或栅格回退全部读取`outline_polygon`；边对齐只读取`seam_polygon`。

**Step 2: 改造GRAPH**

GRAPH匹配集合必须满足每片最多选择一个候选。完整布局闭环由候选边计算，随后用变换后的完整轮廓执行严格/容错验收。诊断增加各候选等级的边数、状态数和拒绝数。

**Step 3: 改造FOURFAST**

根状态和每层Beam按“最高候选等级、候选总分、原几何分”排序；继续使用现有32宽度、关系数和1.5秒活动预算。活动预算达到时只结束FOURFAST并把同一几何结构交给FALLBACK，不能重建候选或重置计时。

**Step 4: 改造FALLBACK**

前沿优先扩展排名0候选；排名0关系耗尽后自然进入排名1/2关系。`search_width`和`max_nodes`仍是全任务上限，不按候选层翻倍。找到首个WHITE硬验收结果立即保存`best_plan`。

**Step 5: 增加结构化失败分类**

候选为空返回`shape_hypothesis_empty`；边图为空返回`edge_graph_empty`；有完整候选时根据`size_reject`、`fill_reject`、`overlap_reject`主导计数返回`layout_size`、`layout_fill`或`layout_overlap`；真正越过总预算才返回`solver_timeout`。

**Step 6: 运行1～4片GREEN测试**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py -k "unknown or graph or four_fast or fallback or hypothesis" -q`

Expected: PASS；伪角点四片得到规划，现有三片首选候选用例不进入第二候选等级，1～4片参数化测试全部通过。

---

### Task 6: 机械结果、日志和模板隔离验证

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `tests_ab/test_quad_main.py`

**Step 1: 使用完整轮廓质心构造placement**

`_build_unknown_success_plan()`从求解位姿直接计算`source_center_mm`、`target_center_mm`和`rotation_deg`；目标显示轮廓使用变换后的`outline_local`。新增逆变换测试，允许数值误差内恢复锁定中心和目标中心。

**Step 2: 增加惰性候选日志**

`AssemblyRuntime`在锁定时输出每片候选数量、最佳顶点数、最短边和分数；候选层变化时输出边和状态计数；成功时输出每片最终候选编号。`UNKNOWN_SOLVER_DEBUG=False`时不得调用候选预览和字符串格式化辅助函数。

**Step 3: 验证完全不访问KNOWN**

运行Task 1的模板禁止调用测试，并增加源码级路由断言：UNKNOWN分支只创建`UnknownSolveJob`，只有`mode == "known"`分支能够调用`solve_known_layout()`。

**Step 4: 运行运行器GREEN测试**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_quad_main.py tests_ab/test_a_known_planner.py -q`

Expected: PASS；UNKNOWN模板隔离、机械位姿、调试开关、KNOWN原行为均正确。

---

### Task 7: 全量回归、文档与v1.10.0发布

**Files:**
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `项目规划清单.md`
- Modify: `研究发现.md`
- Modify: `编辑清单.md`
- Modify: `硬件资源表.md`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.10.0.zip`

**Step 1: 更新现场调试文档**

记录五个候选宏的准确位置、推荐调整方向、`SHAPE/TIER/ACCEPT`日志解释、六种失败原因，以及为什么不能通过继续增大超时掩盖错误轮廓。

**Step 2: 运行定向测试**

Run: `python -m pytest tests_ab/test_quad_vision.py tests_ab/test_quad_main.py tests_ab/test_a_edge_features.py tests_ab/test_a_unknown_planner.py tests_ab/test_a_runtime_planner.py tests_ab/test_a_known_planner.py -q`

Expected: 退出码0且零失败。

**Step 3: 运行全量和编译验证**

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: 两条命令均退出0；不得删除或跳过既有测试来取得通过。

**Step 4: 验证性能与硬门**

运行三片和四片固定基准各五次，记录搜索节点、候选等级、FOURFAST活动时间和总耗时。三片首选候选用例不得进入第二候选等级；四片不得超过现有FOURFAST活动预算。重新运行尺寸、填充、重叠和外边负例，确认全部继续拒绝。

**Step 5: 构建和验证发布包**

构建前记录稳定版源码哈希和B版ZIP SHA256；运行`python tools/package_variants.py`后执行`python -m pytest tests_ab/test_variant_packages.py -q`。A版ZIP必须与当前源码逐字节一致并通过MaixVision平铺导入；稳定版源码和B版ZIP哈希不得变化。

**Step 6: 记录交付证据**

把RED/GREEN用例、定向/全量数量、编译结果、三片/四片诊断、A版ZIP大小和SHA256写入四文件。根目录不是有效Git仓库，因此不执行commit/push，以测试结果和哈希作为发布快照。

