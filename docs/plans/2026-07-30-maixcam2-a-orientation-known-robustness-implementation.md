# MaixCAM2 A版横竖纸张与KNOWN稳健识别 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让A版自动识别横放或竖放A4并允许现场手动保存方向，同时消除KNOWN因4/5顶点抖动造成的反复UNKNOWN和稳定门清零。

**Architecture:** 纸张方向作为V5运行设置贯穿单应映射、机械区、调参页和规划回绘；自动定位通过两组对边平均像素长度判向。KNOWN将闭合轮廓等弧长重采样为32点，用无镜像循环Kabsch统一完成形状距离、跨帧稳定和旋转配准，旧模板保留描述子回退。

**Tech Stack:** MaixPy、Python、OpenCV、NumPy、pytest、MaixVision平铺ZIP。

---

### Task 1: 横竖纸面坐标与AUTO方向

**Files:**
- Modify: `tests_ab/test_paper_locator.py`
- Modify: `maixcam2_app_A_quad/paper_locator.py`

**Step 1: Write the failing tests**

增加横放A4自动方向、横放297×210mm单应往返、横放默认230×210机械区、竖放旧API兼容
测试。构造明显透视四边形，断言使用两组对边平均值而不是单边长度。

**Step 2: Run RED**

Run: `python -m pytest tests_ab/test_paper_locator.py -k "orientation or landscape" -q`

Expected: FAIL，A版尚无方向字段和横放映射参数。

**Step 3: Implement minimal mapping support**

在`paper_locator.py`增加方向常量、校验、纸面尺寸、默认机械区和默认分界线辅助函数；
`PaperLocation`保存方向；所有映射函数增加默认`portrait`的可选参数；定位成功按对边均值返回
自动方向。

**Step 4: Run GREEN**

Run: `python -m pytest tests_ab/test_paper_locator.py -q`

Expected: PASS。

### Task 2: 设置V5与ROI页手动兜底

**Files:**
- Modify: `tests_ab/test_variant_settings.py`
- Modify: `tests_ab/test_a_high_resolution_vision.py`
- Modify: `tests_ab/test_variant_calibration_ui.py`
- Modify: `maixcam2_app_A_quad/settings_store.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`

**Step 1: Write the failing tests**

增加V2/V3/V4迁移到`portrait`、V5横放往返、`PAPER`参数循环、`+/-`切换V/H、自动定位
同步方向和切换后重置默认机械区/分界线测试。

**Step 2: Run RED**

Run: `python -m pytest tests_ab/test_variant_settings.py tests_ab/test_a_high_resolution_vision.py tests_ab/test_variant_calibration_ui.py -k "orientation or paper_mode" -q`

Expected: FAIL，设置V4和ROI参数列表尚无方向。

**Step 3: Implement V5 and UI**

设置格式升级到V5并显式保存方向；旧版本加载后补`portrait`。ROI页显示`PAPER V/H`，切换
或AUTO方向改变时调用统一默认区域辅助函数，`LOCK ROI`沿用现有整组保存。

**Step 4: Run GREEN**

Run: `python -m pytest tests_ab/test_variant_settings.py tests_ab/test_a_high_resolution_vision.py tests_ab/test_variant_calibration_ui.py -q`

Expected: PASS。

### Task 3: 主循环与绘制全链路传播方向

**Files:**
- Modify: `tests_ab/test_quad_main.py`
- Modify: `tests_ab/test_a_work_region.py`
- Modify: `tests_ab/test_calibration_quality.py`
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`

**Step 1: Write the failing tests**

增加横放活动四边形、横放碎片毫米反算、横放红线和目标回绘测试；断言所有入口调用都传递
同一`paper_orientation`。

**Step 2: Run RED**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_a_work_region.py tests_ab/test_calibration_quality.py -k "orientation or landscape" -q`

Expected: FAIL，运行路径仍固定210×297mm。

**Step 3: Wire orientation through runtime**

修改分析、有效区构造、标定尺度、分界线和规划绘制调用，统一读取设置方向。公共API方向参数
保留默认值，减少旧测试和离线工具回归。

**Step 4: Run GREEN and orientation regressions**

Run: `python -m pytest tests_ab/test_paper_locator.py tests_ab/test_variant_settings.py tests_ab/test_variant_calibration_ui.py tests_ab/test_quad_main.py tests_ab/test_a_work_region.py tests_ab/test_calibration_quality.py -q`

Expected: PASS。

### Task 4: KNOWN 32点轮廓测试与实现

**Files:**
- Modify: `tests_ab/test_a_known_planner.py`
- Modify: `tests_ab/test_a_runtime_planner.py`
- Modify: `maixcam2_app_A_quad/template_store.py`
- Modify: `maixcam2_app_A_quad/assembly_planner.py`
- Modify: `maixcam2_app_A_quad/config.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write the failing tests**

增加同一形状4/5顶点和2mm毛刺的匹配测试、4/5顶点跨帧稳定测试、重采样旋转姿态测试，
再增加不同凹凸形状和镜像必须拒绝的反例。断言默认阈值为1.60、主循环容差为3mm。

**Step 2: Run RED**

Run: `python -m pytest tests_ab/test_a_known_planner.py tests_ab/test_a_runtime_planner.py -k "resample or vertex_jitter or relaxed" -q`

Expected: FAIL，现有逻辑仍因顶点数不同拒绝。

**Step 3: Implement resampling and no-reflection alignment**

在`template_store.py`实现32点闭合轮廓等弧长重采样、中心化尺度归一化和无镜像循环
Kabsch距离。KNOWN有目标毫米轮廓时优先使用新距离，缺少时使用旧描述子。规划器稳定门和
旋转配准复用同一重采样逻辑；中心容差改3mm，配置阈值改1.60。

**Step 4: Run GREEN and negative cases**

Run: `python -m pytest tests_ab/test_a_known_planner.py tests_ab/test_a_runtime_planner.py -q`

Expected: PASS，错误形状与镜像反例仍拒绝。

### Task 5: 文档、版本和发布验证

**Files:**
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `README.md`
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`
- Modify: `硬件资源表.md`

**Step 1: Write release RED tests**

将A版期望版本和ZIP更新为v1.5.0，增加V5设置、PAPER V/H和KNOWN稳健参数的发布断言。

**Step 2: Run RED**

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: FAIL，清单和发布脚本仍是v1.4.1。

**Step 3: Update docs and packaging**

记录横竖AUTO规则、手动兜底、方向改变后重新锁定、KNOWN放宽边界和实机验证步骤；更新
版本与显式打包白名单，不修改稳定版和B版业务源码。

**Step 4: Full verification**

Run: `python -m pytest tests tests_ab -q`

Run: `python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -q`

Expected: 全部退出码0。

**Step 5: Verify immutable boundaries**

按`docs/plans/2026-07-29-maixcam2-stable-baseline-sha256.md`核对稳定版9项哈希；确认B版
业务源码和固定ZIP哈希未变，记录A版v1.5.0 ZIP大小与SHA256。根目录不是有效Git仓库，
因此不执行commit/push。

