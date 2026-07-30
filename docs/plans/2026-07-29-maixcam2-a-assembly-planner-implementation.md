# MaixCAM2 A版机械区域与拼装规划器 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为A版增加可调毫米机械区域、上下区判断、已知四片快速规划、未知碎片回溯求解、稳定帧缓存和下半区目标显示，并发布v1.2.0。

**Architecture:** 完整A4四角是唯一像素/毫米基准；`settings_store.py`管理V3机械参数，`paper_locator.py`提供双向单应映射，新增`assembly_planner.py`集中保存纯几何求解、稳定门和绘制逻辑。`main.py`只组织相机帧、模式、缓存与UI，B版不接入任何新模块。

**Tech Stack:** MaixPy/Python 3、OpenCV、NumPy、pytest、MaixVision平铺ZIP。

---

## 实施清单

| 轮次 | 目标 | 验证标准 | 审查 |
|---|---|---|---|
| Round 1 | 设置V3与机械区域调参 | V2迁移、边界校验、黄色区域和红线绘制测试通过 | `review:true` |
| Round 2 | 毫米碎片与已知布局 | 上下区分类、布局登记、24种匹配和刚体目标位姿测试通过 | `review:true` |
| Round 3 | 未知几何与牌面评分 | 1～4片回溯、矩形栅格验证、无解上限和接缝评分测试通过 | `review:true` |
| Round 4 | 主循环缓存、绘制与发布 | 全量测试、编译、平铺导入、哈希边界全部通过 | `review:true` |

### Task 1: 设置V3与毫米机械区域

**Files:**
- Modify: `maixcam2_app_A_quad/settings_store.py`
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Test: `tests_ab/test_a_work_region.py`

**Step 1: Write the failing tests**

新增测试覆盖默认 `[0,33.5,210,230]`、`SPLIT_Y=148.5`、V2 INSET迁移、V3往返、越界拒绝、透视工作区映射、五参数循环及0.5mm调整、ROI页红线绘制。

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests_ab/test_a_work_region.py -v`

Expected: FAIL，缺少V3字段、`build_work_quad()`和机械参数UI动作。

**Step 3: Write minimal implementation**

在A版设置中增加 `work_x_mm/work_y_mm/work_width_mm/work_height_mm/split_y_mm`，保存版本改为3并兼容读取版本2；增加 `validate_work_region_mm()`、`build_work_quad()`、`build_split_segment()`和毫米到像素反向映射；ROI会话用 `WORK_ITEMS=(X,Y,W,H,SPLIT)` 取代INSET操作，主入口改用新区域。

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_ab/test_a_work_region.py tests_ab/test_paper_locator.py tests_ab/test_quad_main.py tests_ab/test_variant_calibration_ui.py -v`

Expected: PASS，且B版参数化旧行为仍通过。

**Step 5: Record checkpoint**

更新 `项目规划清单.md` 与 `编辑清单.md`；当前目录无有效Git仓库，因此记录测试输出和文件SHA256，不执行伪提交。

### Task 2: 毫米多边形与已知布局规划

**Files:**
- Create: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/puzzle_vision.py`
- Modify: `maixcam2_app_A_quad/template_store.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Test: `tests_ab/test_a_known_planner.py`

**Step 1: Write the failing tests**

测试逐顶点单应映射、`upper/lower/crossing`分类、正确布局登记、旧形状模板可加载但报告无布局、四片随机平移旋转后的唯一匹配、目标位姿无镜像且位于下半区。

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests_ab/test_a_known_planner.py -v`

Expected: FAIL，缺少 `assembly_planner`、`vertices_mm` 和目标布局字段。

**Step 3: Write minimal implementation**

`analyze_quad_frame()`为中心和每个拟合顶点附加毫米坐标并分类；`register_known_layout()`把形状描述子与目标局部多边形绑定；`solve_known_layout()`复用模板全局分配并通过二维刚体配准输出每片 `target_center_mm/rotation_delta_deg/target_polygon_mm`。

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_ab/test_a_known_planner.py tests_ab/test_quad_main.py -v`

Expected: PASS，已知四片最多24种分配且目标框固定为100×60mm。

**Step 5: Record checkpoint**

更新四文件，记录布局保存必须保留1～2mm可分割黑缝的实机注意事项。

### Task 3: 未知几何回溯与扑克牌接缝评分

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/puzzle_vision.py`
- Test: `tests_ab/test_a_unknown_planner.py`
- Test: `tests_ab/test_a_edge_features.py`

**Step 1: Write the failing tests**

构造1～4片毫米多边形矩形分割，测试任意初始旋转后可恢复矩形；测试明显不匹配边和无解集合在节点上限内退出；构造两组几何等价候选，验证有纹理时颜色/梯度连续的接缝分数更低，纯白边退化为几何评分。

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_edge_features.py -v`

Expected: FAIL，缺少未知求解和边缘纹理接口。

**Step 3: Write minimal implementation**

实现无缩放无镜像的边对齐变换、长度容差、栅格重叠剪枝、节点/候选硬上限、完整组合的最小外接矩形旋正和填充率校验；从相机帧沿每条边内侧采样BGR与梯度，只有纹理能量超过门槛才计入接缝排序。

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_ab/test_a_unknown_planner.py tests_ab/test_a_edge_features.py -v`

Expected: PASS，合法组合得到下半区目标，无解返回结构化失败且不超节点上限。

**Step 5: Record checkpoint**

更新四文件，记录求解节点数、评分和仍需实机验证的牌面采样质量。

### Task 4: 稳定帧缓存、目标绘制与A版v1.2.0发布

**Files:**
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `tools/package_variants.py`
- Test: `tests_ab/test_a_runtime_planner.py`
- Test: `tests_ab/test_variant_packages.py`
- Modify: `docs/maixcam2-auto-roi-ab-guide.md`

**Step 1: Write the failing tests**

测试连续三帧才触发、缓存后不重复调用、模式/CAL/模板变化清缓存、红线和目标轮廓/箭头绘制、A清单新增规划器且版本1.2.0、B发布规格和哈希不变。

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_variant_packages.py -v`

Expected: FAIL，入口尚未接入稳定门且A版仍为1.1.1。

**Step 3: Write minimal implementation**

在 `assembly_planner.py` 增加稳定签名和缓存状态机及目标叠加绘制；主循环在模式编号后更新状态机，SAVE登记布局并清缓存，CAL往返清缓存；清单和打包白名单增加规划器，A发布名改为 `diansai_quad-v1.2.0.zip`，同步现场操作文档。

**Step 4: Run focused and full verification**

Run: `python -m pytest tests_ab/test_a_runtime_planner.py tests_ab/test_variant_packages.py -v`

Run: `python -m pytest -v`

Run: `python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -v`

Expected: 所有命令退出码0，A版ZIP可平铺导入。

**Step 5: Verify immutable boundaries**

核对 `docs/plans/2026-07-29-maixcam2-stable-baseline-sha256.md` 的稳定版9项哈希全部一致；核对B版 `diansai_warp-v1.1.0.zip` SHA256仍为 `F08BFE4183241399073DA4DB6C16778678498E9214D62CC755EC22C37481ED92`。

**Step 6: Final review record**

更新四文件到REVIEW，记录PC验证项和MaixCAM2实机待验项；因Git仓库无效，不执行commit/push。
