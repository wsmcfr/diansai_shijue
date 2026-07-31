# MaixCAM2 AUTO ROI Diagnostics Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** 为A版AUTO ROI增加单次、结构化、可定位具体拒绝门的电脑端诊断日志。

**Architecture:** `paper_locator.py`负责采集与返回诊断数据，`main.py`只负责稳定格式化；两者通过`PaperLocation.diagnostics`传递，不在轮廓循环中直接输出。

**Tech Stack:** Python 3、OpenCV、NumPy、pytest、MaixPy控制台

---

### Task 1: 用测试定义诊断数据与日志格式

**Files:**
- Modify: `tests_ab/test_paper_locator.py`
- Modify: `tests_ab/test_quad_main.py`

1. 新增面积超过50%的黑色矩形场景，断言失败结果包含`area_large=1`和最大面积。
2. 新增失败日志测试，构造带完整诊断字典的`PaperLocation.failed()`，断言一行输出包含各门计数、顶点分布和最佳候选指标。
3. 运行两个新增用例，确认因`diagnostics`接口尚不存在而按预期失败。

### Task 2: 返回结构化AUTO诊断数据

**Files:**
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Test: `tests_ab/test_paper_locator.py`

1. 为`PaperLocation`和`failed()`增加可选`diagnostics`参数，并复制保存字典。
2. 让候选四角近似同时返回顶点数，统计面积过小、面积过大、非四角、矩形度不足和有效候选数量。
3. 扩展候选评分内部结果，记录面积比例、观测短长边比、比例分、矩形度、凸性、暗度和总置信度。
4. 在成功、`no_candidate`和`low_confidence`三个出口附加同一结构的诊断字典。
5. 运行定位器新增用例和原有定位器回归。

### Task 3: 输出单行详细日志并完成回归

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Test: `tests_ab/test_quad_main.py`

1. 增加稳定字段格式化辅助函数，字典缺字段时安全省略。
2. 扩展成功和失败日志，使每次AUTO仍只输出一行。
3. 运行新增日志测试，确认由RED转GREEN。
4. 运行`python -m pytest tests_ab -q`、`python -m compileall -q maixcam2_app_A_quad`和发布物相关回归。
5. 检查`git diff`，确认未调整任何AUTO阈值或识别判断。
